"""Offline eval: golden-set runner with latency + citation coverage.

No external deps. Pass = expected substring present (case-insensitive).
Set NIM_EVAL_JUDGE=1 to also score each answer with the NIM judge
(diagram's "LLM Judges" box); off by default to keep eval fast/offline.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .agents import code_agent, multi_agent, table_agent, text_agent, vision_agent
from .reasoning import plan
from .retrieval import hybrid_search
from .validation import gatekeeper

GOLDEN = Path(__file__).resolve().parent.parent / "eval_data" / "golden_qa.jsonl"

_AGENT = {"text": text_agent, "table": table_agent, "image": vision_agent,
          "code": code_agent, "multi": multi_agent}


def _norm(s: str) -> str:
    """Lowercase + unify dashes/quotes so LLM typography never fails a match."""
    s = s.lower()
    for a, b in (("‑", "-"), ("–", "-"), ("—", "-"), ("‘", "'"), ("’", "'"),
                 ("“", '"'), ("”", '"')):
        s = s.replace(a, b)
    return s


def run_case(query: str, top_k: int = 5) -> dict:
    t0 = time.perf_counter()
    blocked, _ = gatekeeper(query)
    if blocked:
        return {"answer": "Blocked", "route": "blocked", "citations": 0,
                "latency_ms": 0, "confidence": 0.0}
    p = plan(query)
    hits = hybrid_search(query, top_k=top_k)
    agent = _AGENT.get(p["intent"], text_agent)
    answer, conf = agent(query, hits)
    return {"answer": answer, "route": p["intent"], "citations": len(hits),
            "latency_ms": int((time.perf_counter() - t0) * 1000), "confidence": conf}


def run_golden(top_k: int = 5) -> dict:
    if not GOLDEN.exists():
        return {"cases": 0, "pass": 0, "pass_rate": 0.0, "results": []}
    results = []
    lat: list[int] = []
    for line in GOLDEN.read_text().splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        out = run_case(case["query"], top_k=top_k)
        expected = case.get("expected_contains", "")
        passed = _norm(expected) in _norm(out["answer"])
        lat.append(out["latency_ms"])
        entry: dict = {"query": case["query"], "modality": case.get("modality"),
                       "expected": expected, "passed": passed, **out}
        if passed and os.environ.get("NIM_EVAL_JUDGE") == "1":
            entry["judge"] = judge_answer(case["query"], out["answer"], top_k=top_k)
        results.append(entry)
    n = len(results)
    ok = sum(1 for r in results if r["passed"])
    judged = [r["judge"]["score"] for r in results
              if isinstance(r.get("judge"), dict) and r["judge"].get("score") is not None]
    summary: dict = {"cases": n, "pass": ok, "pass_rate": round(ok / n, 3) if n else 0.0,
                     "p50_latency_ms": sorted(lat)[len(lat) // 2] if lat else 0,
                     "citation_coverage": round(sum(1 for r in results if r["citations"] > 0) / n, 3) if n else 0.0,
                     "results": results}
    if judged:
        summary["judge_mean"] = round(sum(judged) / len(judged), 3)
    return summary


def judge_answer(query: str, answer: str, top_k: int = 5) -> dict | None:
    """LLM-judge grounding score for one answer. None when unavailable."""
    try:
        from .llm import judge
        from .retrieval import hybrid_search
    except Exception:
        return None
    return judge(query, answer, hybrid_search(query, top_k=top_k))
