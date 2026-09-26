"""Qdrant vector store (Cloud or local docker). Fail-soft throughout.

Without QDRANT_URL (or without the client lib / NIM key) every function
returns an empty/False result and retrieval falls back to keyword search.
Collection holds one point per chunk: 2048-dim NIM vector + chunk payload.
"""
from __future__ import annotations

import os
import uuid

COLLECTION = os.environ.get("QDRANT_COLLECTION", "mmrag")
VECTOR_DIM = 2048


def _cfg() -> dict:
    return {
        "url": os.environ.get("QDRANT_URL", "").strip(),
        "api_key": os.environ.get("QDRANT_API_KEY", "").strip(),
    }


def configured() -> bool:
    return bool(_cfg()["url"])


def _client():
    cfg = _cfg()
    if not cfg["url"]:
        return None
    try:
        from qdrant_client import QdrantClient  # type: ignore
    except ImportError:
        return None
    try:
        return QdrantClient(url=cfg["url"], api_key=cfg["api_key"] or None,
                            timeout=20.0)
    except Exception:
        return None


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def prune(live_ids: list[str]) -> int:
    """Delete points whose chunk_id is gone from the index. Returns removed count."""
    client = _client()
    if client is None:
        return 0
    try:
        from qdrant_client.models import PointIdsList  # type: ignore
        live = {_point_id(c) for c in live_ids}
        stale: list[str] = []
        offset = None
        while True:
            pts, offset = client.scroll(COLLECTION, limit=256, offset=offset,
                                        with_payload=False, with_vectors=False)
            for p in pts:
                if str(p.id) not in live:
                    stale.append(p.id)
            if offset is None:
                break
        if stale:
            client.delete(collection_name=COLLECTION, points_selector=PointIdsList(points=stale))
        return len(stale)
    except Exception:
        return 0


def ensure_collection() -> bool:
    client = _client()
    if client is None:
        return False
    try:
        from qdrant_client.models import Distance, VectorParams  # type: ignore
        if not client.collection_exists(COLLECTION):
            client.create_collection(
                COLLECTION,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
        return True
    except Exception:
        return False


def upsert(chunks: list[dict], vectors: list[list[float]]) -> int:
    """Store chunks with their vectors. Returns points written (0 on failure)."""
    client = _client()
    if client is None or not chunks or len(chunks) != len(vectors):
        return 0
    try:
        from qdrant_client.models import PointStruct  # type: ignore
        points = [PointStruct(id=_point_id(c["chunk_id"]), vector=v,
                              payload={k: v2 for k, v2 in c.items()
                                       if isinstance(v2, (str, int, float, bool, list, dict)) or v2 is None})
                  for c, v in zip(chunks, vectors)]
        client.upsert(collection_name=COLLECTION, points=points)
        return len(points)
    except Exception:
        return 0


def search(vector: list[float], top_k: int = 8,
           modality: str | None = None) -> list[dict]:
    """Dense search. Returns [{...chunk, '_dense': score}] (no _dense kept)."""
    client = _client()
    if client is None or not vector:
        return []
    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue  # type: ignore
        qfilter = (Filter(must=[FieldCondition(key="modality",
                                               match=MatchValue(value=modality))])
                   if modality else None)
        hits = client.query_points(collection_name=COLLECTION, query=vector,
                                   query_filter=qfilter, limit=top_k).points
        out = []
        for h in hits:
            c = dict(h.payload or {})
            c["_dense"] = h.score
            out.append(c)
        return out
    except Exception:
        return []
