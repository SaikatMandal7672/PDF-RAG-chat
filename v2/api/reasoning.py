"""Planner + conditional router. Rules first, optional Gemini upgrade.

If GEMINI_API_KEY is set, the planner asks gemini-2.0-flash for a one-word
intent and falls back to rules on any failure/timeout. Offline-safe.
"""
from __future__ import annotations

import json
import os
import urllib.request

TABLE_HINTS = {"table", "row", "column", "sum", "total", "average", "avg", "revenue",
               "quarter", "csv", "sheet", "xlsx", "mean", "max", "min", "count"}
IMAGE_HINTS = {"image", "photo", "chart", "figure", "diagram", "screenshot",
               "picture", "bbox", "graph", "plot"}
CODE_HINTS = {"code", "function", "class", "traceback", "import", "def ",
              "const ", "stack", "error", "exception"}


def _rule_intent(query: str, modality_filter: str | None) -> str:
    if modality_filter in ("table", "image", "code", "text"):
        return modality_filter
    q = query.lower()
    t = any(h in q for h in TABLE_HINTS)
    v = any(h in q for h in IMAGE_HINTS)
    c = any(h in q for h in CODE_HINTS)
    if t and v:
        return "multi"
    if t:
        return "table"
    if v:
        return "image"
    if c:
        return "code"
    return "text"


def _llm_intent(query: str) -> str | None:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key or len(query) < 12:
        return None
    try:
        url = ("https://generativelanguage.googleapis.com/v1beta/models/"
               f"gemini-2.0-flash:generateContent?key={key}")
        body = json.dumps({
            "contents": [{"parts": [{
                "text": "Classify the query into exactly one word: text, table, image, code, or multi.\nQuery: " + query[:500]
            }]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 5},
        }).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=6) as r:
            data = json.loads(r.read().decode())
        word = data["candidates"][0]["content"]["parts"][0]["text"].strip().lower()
        for cand in ("multi", "table", "image", "code", "text"):
            if cand in word:
                return cand
    except Exception:
        return None
    return None


def plan(query: str, modality_filter: str | None = None) -> dict:
    intent = _rule_intent(query, modality_filter)
    if modality_filter is None:
        llm = _llm_intent(query)
        if llm:
            intent = llm
    return {
        "intent": intent,
        "needs_table": intent in ("table", "multi"),
        "needs_vision": intent in ("image", "multi"),
        "rewritten_queries": [query.strip()],
        "planner": "gemini" if os.environ.get("GEMINI_API_KEY") and modality_filter is None and intent != _rule_intent(query, None) else "rules",
    }


def route(intent: str, hits: list[dict], confidence: float) -> tuple[str, bool]:
    if not hits:
        return ("ask_clarifying", True)
    if confidence < 0.15:
        return ("escalate_to_human", True)
    if intent == "table":
        return ("table_agent", False)
    if intent == "image":
        return ("vision_agent", False)
    if intent == "code":
        return ("code_agent", False)
    if intent == "multi":
        return ("multi_agent", False)
    return ("answer_directly", False)
