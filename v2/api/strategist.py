"""Strategist (diagram box, advisory v0): reads the audit log and suggests
tuning actions. Read-only, rule-based, human approves — never auto-applies.
"""
from __future__ import annotations

from .store import recent_queries, stats


def analyze(limit: int = 100) -> dict:
    qs = recent_queries(limit)
    suggestions: list[dict] = []
    if not qs:
        return {"queries_seen": 0, "suggestions": [
            {"signal": "no traffic yet",
             "action": "Ask a few questions, then re-check this endpoint."}]}

    by_route: dict[str, list[float]] = {}
    blocked = sum(1 for q in qs if q.get("blocked"))
    slow = [q for q in qs if (q.get("latency_ms") or 0) > 15000]
    low = [q for q in qs if (q.get("confidence") or 0) < 0.3 and not q.get("blocked")]
    for q in qs:
        by_route.setdefault(q.get("route", "?"), []).append(q.get("confidence") or 0.0)

    if blocked / len(qs) > 0.3:
        suggestions.append({"signal": f"block rate {blocked}/{len(qs)} is high",
                            "action": "Review gatekeeper patterns in validation.py; users may be tripping injection regex."})
    if low:
        worst = sorted(low, key=lambda q: q.get("confidence") or 0.0)[:3]
        suggestions.append({"signal": f"{len(low)} low-confidence answers",
                            "action": "Check retrieval for: " + "; ".join(q["query"][:60] for q in worst)})
    table_confs = by_route.get("table_agent", [])
    if table_confs and sum(table_confs) / len(table_confs) < 0.6:
        suggestions.append({"signal": "table_agent confidence is low on average",
                            "action": "Verify native PDF tables parse as pipe tables; see ingest chunk_text()."})
    if slow:
        suggestions.append({"signal": f"{len(slow)} queries slower than 15s",
                            "action": "LLM latency dominates: lower top_k, shorten _pieces_block, or pick a faster NIM model."})
    if not suggestions:
        suggestions.append({"signal": "all signals nominal",
                            "action": "No tuning needed. Re-check after more traffic or a red-team failure."})
    routes = {r: {"n": len(v), "avg_conf": round(sum(v) / len(v), 2)} for r, v in by_route.items()}
    return {"queries_seen": len(qs), "stats": stats(), "routes": routes,
            "suggestions": suggestions}
