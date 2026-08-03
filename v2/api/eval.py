"""Offline eval: golden-set runner with latency + citation coverage.

No external deps. Pass = expected substring present (case-insensitive).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .agents import code_agent, multi_agent, table_agent, text_agent, vision_agent
from .reasoning import plan
from .retrieval import hybrid_search
from .validation import gatekeeper

GOLDEN = Path(__file__).resolve().parent.parent / "eval_data" / "golden_qa.jsonl"

_AGENT = {"text": text_agent, "table": table_agent, "image": vision_agent,
          "code": code_agent, "multi": multi_agent}


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
        passed = expected.lower() in out["answer"].lower()
        lat.append(out["latency_ms"])
        results.append({"query": case["query"], "modality": case.get("modality"),
                        "expected": expected, "passed": passed, **out})
    n = len(results)
    ok = sum(1 for r in results if r["passed"])
    return {"cases": n, "pass": ok, "pass_rate": round(ok / n, 3) if n else 0.0,
            "p50_latency_ms": sorted(lat)[len(lat) // 2] if lat else 0,
            "citation_coverage": round(sum(1 for r in results if r["citations"] > 0) / n, 3) if n else 0.0,
            "results": results}
