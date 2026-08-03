"""Shared pydantic contracts (mirrors plan.md §9)."""
from typing import Literal, Optional
from pydantic import BaseModel, Field

Modality = Literal["text", "table", "image", "code"]


class QueryRequest(BaseModel):
    session_id: str = Field(default="local")
    query: str
    top_k: int = Field(default=5, ge=1, le=10)
    modality: Optional[Modality] = None


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    page: Optional[int] = None
    modality: str
    heading_path: list[str] = []
    ref: Optional[str] = None
    snippet: str = ""


class QueryResponse(BaseModel):
    answer_markdown: str
    citations: list[Citation]
    route: str
    confidence: float
    latency_ms: int
    needs_review: bool
    blocked: bool = False


class ReviewRequest(BaseModel):
    decision: Literal["approve", "edit"]
    edited_answer: Optional[str] = None
