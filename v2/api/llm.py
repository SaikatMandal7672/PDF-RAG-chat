"""NIM client (OpenAI-compatible) for all LLM calls. Fail-soft by design.

Every function returns None when the key is missing, the library is missing,
or the call fails/times out — callers must keep their non-LLM fallback.
Keys and model names come from `.env` (see `.env.example`); values are read
lazily so tests can set env vars after import. Never log or return the key.
"""
from __future__ import annotations

import os

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
# NOTE: meta/llama-3.1-8b-instruct EOL'd on NIM (Aug 2026); deepseek-v4.1-flash
# hangs. meta/llama-3.3-70b-versatile answers trivial prompts (~5s) but times
# out on full grounded prompts. openai/gpt-oss-20b verified: full grounded
# prompt in ~4.5s with correct [chunk_id] citations (Sep 2026).
DEFAULT_CHAT_MODEL = "openai/gpt-oss-20b"
DEFAULT_EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"  # text+image joint space, verified

# Grounded answering: the model may ONLY use the given pieces, and every
# factual claim must carry its [chunk_id]. This is what keeps RAG honest.
SYSTEM_ANSWER = """You are a careful assistant that answers ONLY from the retrieved context below.
Rules:
1. Use only facts stated in the context. Never invent numbers, names, or dates.
2. Cite every factual claim with its piece id like [demo-revenue-table].
   Use plain square brackets [id] only, never special bracket styles.
3. If the context does not contain the answer, say exactly: I couldn't find this in the indexed docs.
4. Keep tables exact: copy numbers as-is, never round or estimate.
5. Answer in plain sentences a beginner can follow. No markdown tables unless the question asks for one."""

INTENT_HINTS = {
    "table": "The question is about a table. Quote the relevant rows exactly, then state any sum/average given in the context.",
    "image": "The question is about a chart or image. Describe what the caption shows and cite the image piece.",
    "code": "The question is about code. Quote the relevant symbols and file path from the context.",
    "text": "Answer from the text pieces, citing each claim.",
    "multi": "Combine evidence from text, table and image pieces. Cite each part separately.",
}

# Judge: scores a draft answer for grounding, used by eval (not the hot path).
SYSTEM_JUDGE = """You judge whether an answer is grounded in the given context pieces.
Reply with exactly two lines:
score: <0 or 1> (1 only if every factual claim appears in the context with correct numbers)
note: <one short reason>"""


def _cfg() -> dict:
    return {
        "key": os.environ.get("NVIDIA_NIM_API_KEY", "").strip(),
        "base_url": os.environ.get("NVIDIA_NIM_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL,
        "chat_model": os.environ.get("NIM_CHAT_MODEL", DEFAULT_CHAT_MODEL).strip() or DEFAULT_CHAT_MODEL,
        "embed_model": os.environ.get("NIM_EMBED_MODEL", DEFAULT_EMBED_MODEL).strip() or DEFAULT_EMBED_MODEL,
    }


def llm_configured() -> bool:
    """True if a key is present. Never exposes the key itself."""
    return bool(_cfg()["key"])


INTENT_WORDS = ("multi", "table", "image", "code", "text")


def classify(query: str) -> str | None:
    """One-word intent via NIM. None when unconfigured, short, or failing."""
    client = _client()
    if client is None or len(query.strip()) < 12:
        return None
    try:
        resp = client.chat.completions.create(
            model=_cfg()["chat_model"],
            temperature=0.0,
            max_tokens=256,  # reasoning models think first; tiny budgets starve `content`
            messages=[
                {"role": "system", "content": "Classify the query into exactly one word: text, table, image, code, or multi. Reply with only that word. Hint words: table, row, sum, total, average, revenue, quarter, csv, sheet mean table; image, chart, figure, diagram, photo mean image; code, function, class, error mean code; both table and image words mean multi."},
                {"role": "user", "content": query[:500]},
            ],
        )
        word = (resp.choices[0].message.content or "").strip().lower()
        for cand in INTENT_WORDS:
            if cand in word:
                return cand
        return None
    except Exception:
        return None


def _client():
    cfg = _cfg()
    if not cfg["key"]:
        return None
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        return None
    return OpenAI(api_key=cfg["key"], base_url=cfg["base_url"],
                  timeout=30.0, max_retries=0)  # fail fast so fallback answers stay snappy


def _pieces_block(hits: list[dict], limit: int = 6, chars: int = 1200) -> str:
    lines = []
    for h in hits[:limit]:
        body = (h.get("content_text", "") or "")[:chars]
        lines.append(f"[{h.get('chunk_id')}] ({h.get('modality', 'text')}"
                     f"{', p.' + str(h.get('page')) if h.get('page') else ''}):\n{body}")
    return "\n\n".join(lines)


def generate(query: str, hits: list[dict], intent: str = "text") -> str | None:
    """Ask NIM to write the answer from `hits`. None = fall back to paste."""
    client = _client()
    if client is None or not hits:
        return None
    hint = INTENT_HINTS.get(intent, INTENT_HINTS["text"])
    try:
        resp = client.chat.completions.create(
            model=_cfg()["chat_model"],
            temperature=0.2,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": SYSTEM_ANSWER + "\n" + hint},
                {"role": "user", "content": f"Context pieces:\n{_pieces_block(hits)}\n\nQuestion: {query}"},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or None
    except Exception:
        return None


def judge(query: str, answer: str, hits: list[dict]) -> dict | None:
    """Ask NIM to score grounding: {'score': 0|1, 'note': str}. None on failure."""
    client = _client()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=_cfg()["chat_model"],
            temperature=0.0,
            max_tokens=128,
            messages=[
                {"role": "system", "content": SYSTEM_JUDGE},
                {"role": "user", "content": f"Context pieces:\n{_pieces_block(hits)}\n\nQuestion: {query}\nAnswer: {answer}"},
            ],
        )
        out = (resp.choices[0].message.content or "").strip()
        score = 1 if out.lower().startswith("score: 1") else 0
        note = out.split("note:", 1)[1].strip() if "note:" in out.lower() else out[:140]
        return {"score": score, "note": note}
    except Exception:
        return None


def embed(texts: list[str], kind: str = "passage") -> list[list[float]] | None:
    """NIM embeddings (NV vision-language model, 2048-dim).

    kind: "passage" for chunks at ingest, "query" for user queries.
    Images go in as base64 data-URL strings in the same list (same space).
    None on any failure — callers keep keyword fallback.
    """
    client = _client()
    if client is None or not texts:
        return None
    try:
        resp = client.embeddings.create(model=_cfg()["embed_model"],
                                        input=texts[:32],
                                        extra_body={"input_type": kind})
        return [d.embedding for d in resp.data] or None
    except Exception:
        return None


def embed_image_b64(data_url: str) -> list[float] | None:
    """Embed one image given as 'data:image/png;base64,...'. None on failure."""
    vecs = embed([data_url], kind="passage")
    return vecs[0] if vecs else None


# Vision: pixels -> words for figure captions (ingest-time only).
SYSTEM_CAPTION = """You read charts, diagrams and figures inside PDF documents.
Describe exactly what you see in 2-4 sentences: chart type, labels, colors,
and every visible number with its label. Copy numbers exactly as shown.
If axes or values are unreadable, say what is unclear instead of guessing."""


def caption_image(data_url: str, page_text: str = "") -> str | None:
    """Caption one figure ('data:image/...;base64,...'). None on failure."""
    client = _client()
    if client is None:
        return None
    model = os.environ.get("NIM_VISION_MODEL",
                            "meta/llama-3.2-11b-vision-instruct").strip()
    if not model:
        return None
    prompt = "Caption this figure for a searchable document index."
    if page_text.strip():
        prompt += f" Nearby page text for context: {page_text[:400]}"
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=256,
            messages=[
                {"role": "system", "content": SYSTEM_CAPTION},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or None
    except Exception:
        return None
