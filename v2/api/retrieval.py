"""Hybrid retrieval: keyword BM25-ish + dense-overlap, RRF fusion, rerank, cache.

Stdlib only. `hybrid_search()` keeps its v0 signature so callers don't break.
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

TOKEN = re.compile(r"[a-z0-9]+")

INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "index.json"

_cache_chunks: list[dict] | None = None
_cache_docs: dict[str, float] = {}
_query_cache: dict[str, tuple[float, list[dict]]] = {}
CACHE_TTL = 120.0


def load_chunks() -> list[dict]:
    global _cache_chunks
    if _cache_chunks is not None:
        return _cache_chunks
    if not INDEX_PATH.exists():
        _cache_chunks = []
        return _cache_chunks
    _cache_chunks = json.loads(INDEX_PATH.read_text())
    return _cache_chunks


def reload_index() -> int:
    global _cache_chunks, _query_cache
    _cache_chunks = None
    _query_cache = {}
    return len(load_chunks())


def tokenize(s: str) -> set[str]:
    return set(TOKEN.findall(s.lower()))


def _idf(term: str, docs: list[set[str]]) -> float:
    df = sum(1 for d in docs if term in d)
    return math.log((len(docs) + 1) / (df + 1)) + 1.0


def score(query: str, chunk: dict) -> float:
    """Backward-compatible weighted overlap (kept for confidence + tests)."""
    q = tokenize(query)
    if not q:
        return 0.0
    hay_q = tokenize(" ".join(chunk.get("hypothetical_questions", [])))
    hay_kw = {k.lower() for k in chunk.get("keywords", [])}
    hay_c = tokenize(chunk.get("content_text", "") + " " + chunk.get("summary", ""))
    return 3.0 * len(q & hay_q) + 2.0 * len(q & hay_kw) + len(q & hay_c) / max(1, len(q))


def _bm25_terms(query: set[str], chunk: dict, idfs: dict[str, float]) -> float:
    hay = tokenize(chunk.get("content_text", "") + " " + chunk.get("summary", "")
                   + " " + " ".join(chunk.get("hypothetical_questions", [])))
    return sum(idfs.get(t, 1.0) for t in query if t in hay)


def _dense_terms(query: set[str], chunk: dict) -> float:
    hay_q = tokenize(" ".join(chunk.get("hypothetical_questions", [])))
    hay_kw = {k.lower() for k in chunk.get("keywords", [])}
    return 2.0 * len(query & hay_q) + 1.5 * len(query & hay_kw)


def _rrf(ranks: list[list[str]], k: int = 60) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranking in ranks:
        for rank, cid in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return fused


def _rerank(query: str, chunks: list[dict]) -> list[dict]:
    """Cheap cross-encoder proxy: phrase/proximity bonus on top of fusion score."""
    q = query.lower()
    q_terms = tokenize(query)
    out = []
    for c in chunks:
        body = (c.get("content_text", "") + " " + c.get("summary", "")).lower()
        phrase = 2.0 if q in body else 0.0
        proximity = sum(1 for t in q_terms if t in body) / max(1, len(q_terms))
        c = dict(c)
        c["_fused"] = c.get("_fused", 0.0) + phrase + proximity
        out.append(c)
    return sorted(out, key=lambda c: c["_fused"], reverse=True)


def compress(chunks: list[dict], budget_chars: int = 5000) -> list[dict]:
    """Lost-in-the-middle mitigation: keep head+tail, trim middle chunks."""
    kept, used = [], 0
    order = list(chunks)
    # reorder: best first AND last (models attend to edges)
    if len(order) > 3:
        order = [order[0], order[-1]] + order[1:-1]
    for c in order:
        size = len(c.get("content_text", ""))
        if used + size > budget_chars and kept:
            break
        kept.append(c)
        used += size
    return kept


def hybrid_search(query: str, top_k: int = 5, modality: str | None = None) -> list[dict]:
    key = f"{query}\x00{top_k}\x00{modality}"
    now = time.time()
    if key in _query_cache and now - _query_cache[key][0] < CACHE_TTL:
        return _query_cache[key][1]

    chunks = load_chunks()
    if modality:
        chunks = [c for c in chunks if c.get("modality") == modality]
    if not chunks:
        return []

    docs = [tokenize(c.get("content_text", "") + " " + c.get("summary", "")) for c in chunks]
    idfs = {t: _idf(t, docs) for t in tokenize(query)}
    by_id = {c["chunk_id"]: c for c in chunks}

    bm25_rank = sorted(chunks, key=lambda c: _bm25_terms(tokenize(query), c, idfs), reverse=True)
    dense_rank = sorted(chunks, key=lambda c: _dense_terms(tokenize(query), c), reverse=True)
    fused = _rrf([[c["chunk_id"] for c in bm25_rank], [c["chunk_id"] for c in dense_rank]])
    scored = []
    for cid, f in fused.items():
        c = dict(by_id[cid])
        c["_fused"] = f + score(query, c) * 0.05
        scored.append(c)
    ranked = _rerank(query, sorted(scored, key=lambda c: c["_fused"], reverse=True))
    hits = [c for c in ranked if score(query, c) > 0][: max(1, top_k)]
    final = hits if hits else ranked[:top_k]
    final = [{k: v for k, v in c.items() if not k.startswith("_")} for c in final]
    _query_cache[key] = (now, final)
    return final


def get_chunk(chunk_id: str) -> dict | None:
    for c in load_chunks():
        if c["chunk_id"] == chunk_id:
            return c
    return None
