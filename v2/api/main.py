"""Final FastAPI backend — HF Spaces ready (port 7860).

Query + ingest + review + eval + redteam. Stdlib-first, optional deps degrade gracefully.
"""
from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .agents import code_agent, multi_agent, table_agent, text_agent, vision_agent
from .eval import run_golden
from .models import Citation, QueryRequest, QueryResponse, ReviewRequest
from .reasoning import plan, route
from .redteam import run_all as redteam_all
from .redteam import run_suite as redteam_suite
from .retrieval import compress, get_chunk, hybrid_search, load_chunks, reload_index, score
from .store import log_query, recent_queries, save_review, stats
from .validation import auditor, gatekeeper, output_safe, redact_pii

app = FastAPI(title="multimodal-rag (final)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
UPLOADS = DATA_DIR / "uploads"

_AGENT = {"text": text_agent, "table": table_agent, "image": vision_agent,
          "code": code_agent, "multi": multi_agent}


@app.get("/health")
def health() -> dict:
    return {"ok": True, "chunks": len(load_chunks())}


@app.get("/api/docs")
def docs() -> dict:
    chunks = load_chunks()
    by_doc: dict[str, dict] = {}
    for c in chunks:
        d = by_doc.setdefault(c["doc_id"], {"doc_id": c["doc_id"], "chunks": 0, "modalities": set()})
        d["chunks"] += 1
        d["modalities"].add(c.get("modality"))
    return {"docs": [{"doc_id": k, "chunks": v["chunks"], "modalities": sorted(v["modalities"])}
                     for k, v in sorted(by_doc.items())]}


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    t0 = time.perf_counter()
    qa_id = str(uuid.uuid4())[:8]
    blocked, reason = gatekeeper(req.query)
    if blocked:
        log_query(qa_id, req.session_id, req.query, "blocked", "text", 0.0, 0, True)
        return QueryResponse(answer_markdown=reason, citations=[], route="blocked",
                             confidence=0.0,
                             latency_ms=int((time.perf_counter() - t0) * 1000),
                             needs_review=True, blocked=True)

    p = plan(req.query, req.modality)
    hits = compress(hybrid_search(req.query, top_k=max(req.top_k, 8), modality=req.modality))[:req.top_k]
    agent = _AGENT.get(p["intent"], text_agent)
    answer, conf = agent(req.query, hits)
    safe, answer = output_safe(redact_pii(answer))
    if not safe:
        log_query(qa_id, req.session_id, req.query, "withheld", p["intent"], 0.0,
                  int((time.perf_counter() - t0) * 1000), False)
        return QueryResponse(answer_markdown=answer, citations=[], route="withheld",
                             confidence=0.0,
                             latency_ms=int((time.perf_counter() - t0) * 1000),
                             needs_review=True)

    best = max([score(req.query, h) for h in hits], default=0.0)
    conf = round(min(0.95, max(conf, min(0.9, best / 8.0))), 2)
    route_name, needs_review = route(p["intent"], hits, conf)
    if auditor(answer, hits):
        needs_review = True
    citations = [Citation(chunk_id=h["chunk_id"], doc_id=h["doc_id"], page=h.get("page"),
                          modality=h.get("modality", "text"),
                          heading_path=h.get("heading_path", []), ref=h.get("ref"),
                          snippet=(h.get("content_text", "")[:240]
                                   + ("…" if len(h.get("content_text", "")) > 240 else "")))
                 for h in hits]
    latency_ms = int((time.perf_counter() - t0) * 1000)
    log_query(qa_id, req.session_id, req.query, route_name, p["intent"], conf, latency_ms, False)
    return QueryResponse(answer_markdown=answer, citations=citations, route=route_name,
                         confidence=conf, latency_ms=latency_ms, needs_review=needs_review)


@app.get("/api/citations/{chunk_id}")
def citation(chunk_id: str) -> dict:
    c = get_chunk(chunk_id)
    if not c:
        raise HTTPException(status_code=404, detail="chunk not found")
    return c


@app.post("/api/ingest")
def ingest(file: UploadFile = File(...)) -> dict:
    UPLOADS.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS / Path(file.filename or "upload.bin").name
    try:
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()
    from ingest.index_local import main as reindex  # lazy: avoids import cycles
    reindex()
    reload_index()
    c = get_chunk(dest.stem + "-c0")
    return {"ok": True, "filename": dest.name, "chunks": len(load_chunks()),
            "sample_chunk": (c or {}).get("chunk_id")}


@app.get("/api/audit")
def audit(limit: int = 20) -> dict:
    return {"queries": recent_queries(max(1, min(limit, 100)))}


@app.post("/api/review/{qa_id}")
def review(qa_id: str, req: ReviewRequest) -> dict:
    save_review(qa_id, req.decision, bool(req.edited_answer))
    return {"ok": True, "qa_id": qa_id}


@app.get("/api/eval/summary")
def eval_summary() -> dict:
    return stats()


@app.post("/api/eval/run")
def eval_run(top_k: int = 5) -> dict:
    return run_golden(top_k=top_k)


@app.post("/api/redteam/run")
def redteam_run(suite: str = "all") -> dict:
    if suite == "all":
        return redteam_all()
    from .redteam import SUITES
    if suite not in SUITES:
        raise HTTPException(status_code=404, detail=f"unknown suite: {suite}")
    return redteam_suite(suite)
