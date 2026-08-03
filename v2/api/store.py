"""SQLite audit store with JSONL fallback. Stdlib only, free-tier safe."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "audit.db"
LOG_PATH = DATA_DIR / "queries.jsonl"


def _db() -> sqlite3.Connection | None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(DB_PATH), timeout=5)
        con.execute("""CREATE TABLE IF NOT EXISTS qa_log(
          qa_id TEXT PRIMARY KEY, ts REAL, session_id TEXT, query TEXT,
          route TEXT, intent TEXT, confidence REAL, latency_ms INTEGER, blocked INTEGER)""")
        con.execute("""CREATE TABLE IF NOT EXISTS reviews(
          qa_id TEXT PRIMARY KEY, decision TEXT, edited INTEGER, ts REAL)""")
        con.commit()
        return con
    except Exception:
        return None


def log_query(qa_id: str, session_id: str, query: str, route: str,
              intent: str, confidence: float, latency_ms: int, blocked: bool = False) -> None:
    try:
        with LOG_PATH.open("a") as f:
            f.write(json.dumps({"qa_id": qa_id, "session_id": session_id, "query": query,
                                "route": route, "intent": intent, "confidence": confidence,
                                "latency_ms": latency_ms, "blocked": blocked}) + "\n")
    except OSError:
        pass
    con = _db()
    if con is None:
        return
    try:
        con.execute("INSERT OR REPLACE INTO qa_log VALUES(?,?,?,?,?,?,?,?,?)",
                    (qa_id, time.time(), session_id, query[:1000], route, intent,
                     confidence, latency_ms, int(blocked)))
        con.commit()
    finally:
        con.close()


def save_review(qa_id: str, decision: str, edited: bool) -> None:
    con = _db()
    if con is None:
        return
    try:
        con.execute("INSERT OR REPLACE INTO reviews VALUES(?,?,?,?)",
                    (qa_id, decision, int(edited), time.time()))
        con.commit()
    finally:
        con.close()


def recent_queries(limit: int = 20) -> list[dict]:
    con = _db()
    if con is None:
        return []
    try:
        rows = con.execute(
            "SELECT qa_id, session_id, query, route, intent, confidence, latency_ms, blocked"
            " FROM qa_log ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [{"qa_id": r[0], "session_id": r[1], "query": r[2], "route": r[3],
                 "intent": r[4], "confidence": r[5], "latency_ms": r[6],
                 "blocked": bool(r[7])} for r in rows]
    finally:
        con.close()


def stats() -> dict:
    con = _db()
    if con is None:
        n = sum(1 for _ in LOG_PATH.open()) if LOG_PATH.exists() else 0
        return {"queries_logged": n, "reviews": 0, "avg_latency_ms": None, "block_rate": None}
    try:
        n = con.execute("SELECT COUNT(*) FROM qa_log").fetchone()[0]
        avg = con.execute("SELECT AVG(latency_ms) FROM qa_log").fetchone()[0]
        blocked = con.execute("SELECT COUNT(*) FROM qa_log WHERE blocked=1").fetchone()[0]
        rev = con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        return {"queries_logged": n, "reviews": rev,
                "avg_latency_ms": round(avg, 1) if avg is not None else None,
                "block_rate": round(blocked / n, 3) if n else 0.0}
    finally:
        con.close()
