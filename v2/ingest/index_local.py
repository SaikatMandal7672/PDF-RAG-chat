"""PDF-first indexer: pdf (+ md/txt notes, image captions) -> data/index.json.

Target: multimodal RAG around PDFs that contain text, tables and images.
- PDF: text extracted per page; table rows kept whole; chart/image pages
  kept as caption blocks (add a .caption.txt sidecar for real captions).
- MD/TXT: tiny reader for local notes and demo files (same chunker).
- PNG/JPG: standalone charts via sidecar `<name>.caption.txt`.
- Anything else: stored as a short reference stub so uploads never vanish.

Optional deps (pypdf, pymupdf) degrade gracefully: if neither is installed,
the PDF is still indexed by filename instead of crashing.

Usage:
  python3 ingest/index_local.py
  python3 ingest/index_local.py --src data/uploads   # extra dir
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:  # optional: load .env so NIM/Qdrant work when run directly
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
SAMPLES = ROOT / "data" / "samples"
UPLOADS = ROOT / "data" / "uploads"
OUT = ROOT / "data" / "index.json"
TABLES_OUT = ROOT / "data" / "extracted_tables.json"  # kept for compat, always []

STOP = {
    "the", "and", "for", "with", "this", "that", "from", "have", "will",
    "your", "about", "into", "such", "they", "them", "then", "than",
    "also", "were", "what", "when", "which", "their", "there", "been",
}

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def _keywords(text: str, limit: int = 12) -> list[str]:
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    freq: dict[str, int] = {}
    for w in words:
        if w in STOP:
            continue
        freq[w] = freq.get(w, 0) + 1
    return sorted(freq, key=lambda w: (-freq[w], w))[:limit]


def _questions(heading: str, keywords: list[str]) -> list[str]:
    qs = [f"What does {heading} describe?"]
    if keywords:
        qs.append(f"What is covered about {keywords[0]}?")
    return qs[:2]


def _mk(doc_id: str, cid: str, modality: str, content: str,
        heading: str, page=None, ref=None, summary=None) -> dict:
    content = content[:4000]
    kw = _keywords(content)
    return {
        "chunk_id": cid,
        "doc_id": doc_id,
        "modality": modality,
        "content_text": content,
        "ref": ref,
        "page": page,
        "heading_path": [doc_id, heading[:80]],
        "summary": (summary or content[:140]),
        "keywords": kw,
        "hypothetical_questions": _questions(heading[:60], kw),
    }


def chunk_text(doc_id: str, text: str) -> list[dict]:
    """Split on headings, never split a markdown table mid-row."""
    lines = text.splitlines()
    blocks: list[str] = []
    buf: list[str] = []
    in_table = False

    def flush():
        if buf:
            blocks.append("\n".join(buf).strip())
            buf.clear()

    for ln in lines:
        is_table_row = ln.strip().startswith("|")
        if is_table_row and not in_table:
            flush()
            in_table = True
        if not is_table_row and in_table:
            flush()
            in_table = False
        if re.match(r"^#{1,3} ", ln) and not in_table:
            flush()
        buf.append(ln)
        if len("\n".join(buf)) > 3000 and not in_table:
            flush()
    flush()
    chunks = []
    for i, b in enumerate([x for x in blocks if x.strip()]):
        first = next((l for l in b.splitlines() if l.strip()), f"part {i}")
        mod = "table" if first.strip().startswith("|") else "text"
        chunks.append(_mk(doc_id, f"{doc_id}-c{i}", mod, b, first.lstrip("# ").strip()[:60]))
    return chunks or [_mk(doc_id, f"{doc_id}-c0", "text", text[:4000], doc_id)]


def read_markdown(path: Path) -> list[dict]:
    return chunk_text(path.stem, path.read_text(encoding="utf-8", errors="replace"))


def read_pdf(path: Path) -> list[dict]:
    """Text per page + embedded figures as image chunks.

    Text: pypdf -> pymupdf -> filename stub (never crashes).
    Figures: pymupdf image pull (skips icons <80px), saved to data/blobs/,
    one `image` chunk each with page number + surrounding page text so
    keyword search still finds them without any ML model.
    """
    pages: list[tuple[int, str]] = []
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(str(path))
        for i, pg in enumerate(reader.pages[:30]):
            t = pg.extract_text() or ""
            if t.strip():
                pages.append((i + 1, t))
    except ImportError:
        pass
    except Exception:
        pass
    if not pages:
        try:
            import fitz  # pymupdf, optional  # type: ignore
            doc = fitz.open(str(path))
            for i, pg in enumerate(doc[:30]):
                t = pg.get_text()
                if t.strip():
                    pages.append((i + 1, t))
        except ImportError:
            pass
        except Exception:
            pass
    chunks: list[dict] = []
    if pages:
        text = "".join(f"\n## Page {n}\n{t}\n" for n, t in pages)
        chunks.extend(chunk_text(path.stem, text))
    else:
        # last-resort fallback: index the filename so uploads never silently vanish
        chunks.extend(chunk_text(
            path.stem,
            f"## {path.name}\nPDF uploaded ({path.stat().st_size} bytes, no extractable text layer)."))
    chunks.extend(read_pdf_images(path, dict(pages)))
    return chunks


def _llm():
    """Import api.llm when run from anywhere. None when unavailable."""
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from api import llm  # type: ignore
        return llm
    except Exception:
        return None


def read_pdf_images(path: Path, page_texts: dict[int, str]) -> list[dict]:
    """Pull embedded figures via pymupdf. [] without pymupdf or without figures."""
    try:
        import fitz  # type: ignore
    except ImportError:
        return []
    try:
        doc = fitz.open(str(path))
    except Exception:
        return []
    blobs = ROOT / "data" / "blobs"
    try:
        blobs.mkdir(parents=True, exist_ok=True)
    except OSError:
        return []
    chunks: list[dict] = []
    for i, pg in enumerate(doc[:30]):
        try:
            imgs = pg.get_images(full=True)
        except Exception:
            continue
        for j, im in enumerate(imgs):
            try:
                pix = fitz.Pixmap(doc, im[0])
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                if max(pix.width, pix.height) < 80:
                    continue  # icon/logo, not a figure
                fname = f"{path.stem}_p{i + 1}_img{j}.png"
                pix.save(str(blobs / fname))
            except Exception:
                continue
            ctx = (page_texts.get(i + 1, "") or "")[:300]
            body = f"Figure {j + 1} on page {i + 1} of {path.name}."
            body += _caption_blob(blobs / fname, ctx)
            body += f" Page text: {ctx}" if ctx.strip() else " (No caption yet.)"
            chunks.append(_mk(path.stem, f"{path.stem}-p{i + 1}-img{j}", "image",
                              body, f"Page {i + 1} figure {j + 1}",
                              page=i + 1, ref=f"blobs/{fname}", summary=body[:140]))
    return chunks


def _caption_blob(blob: Path, page_text: str) -> str:
    """Vision-caption one saved figure. '' when unconfigured/failing (fail-soft)."""
    llm = _llm()
    if llm is None or not llm.llm_configured():
        return ""
    try:
        import base64
        raw = blob.read_bytes()
        if len(raw) > 1_500_000:
            return ""
        cap = llm.caption_image("data:image/png;base64," + base64.b64encode(raw).decode(),
                                page_text)
        return f" Caption: {cap}" if cap else ""
    except Exception:
        return ""


def read_image(path: Path) -> list[dict]:
    """Standalone chart/photo: real caption comes from `<stem>.caption.txt`."""
    sidecar = path.with_suffix(".caption.txt")
    if sidecar.exists():
        caption = sidecar.read_text(encoding="utf-8", errors="replace").strip()
    else:
        caption = f"Image {path.name} (add {path.stem}.caption.txt next to it for a real caption)."
    return [_mk(path.stem, f"{path.stem}-img0", "image", caption,
                 path.name, ref=f"uploads/{path.name}",
                 summary=caption[:140])]


def index_file(path: Path) -> list[dict]:
    suf = path.suffix.lower()
    if suf in (".md", ".txt"):
        return read_markdown(path)
    if suf == ".pdf":
        return read_pdf(path)
    if suf in IMG_EXT:
        return read_image(path)
    return [_mk(path.stem, f"{path.stem}-c0", "text",
                 f"File {path.name} uploaded (only pdf/md/txt/png/jpg are parsed; stored as reference).",
                 path.name)]


def seed_demo() -> list[dict]:
    return [
        {
            "chunk_id": "demo-revenue-table", "doc_id": "demo-q3-report",
            "modality": "table",
            "content_text": "| Quarter | Revenue (₹ Cr) |\n|---|---|\n| Q1 | 12.4 |\n| Q2 | 15.1 |\n| Q3 | 18.9 |",
            "ref": None, "page": 4,
            "heading_path": ["Q3 Report", "Revenue table"],
            "summary": "Quarterly revenue table, Q3 at 18.9 Cr.",
            "keywords": ["revenue", "quarter", "table", "report"],
            "hypothetical_questions": ["What was Q3 revenue?", "Show the revenue table"],
        },
        {
            "chunk_id": "demo-chart-image", "doc_id": "demo-q3-report",
            "modality": "image",
            "content_text": "Bar chart comparing Q1–Q3 revenue growth; Q3 bar highest at 18.9 Cr.",
            "ref": "s3://demo/q3-chart.png", "page": 5,
            "heading_path": ["Q3 Report", "Revenue chart"],
            "summary": "Revenue growth chart image.",
            "keywords": ["chart", "image", "revenue", "growth"],
            "hypothetical_questions": ["What does the revenue chart show?"],
        },
    ]


def main() -> None:
    srcs = [SAMPLES]
    for a in sys.argv[1:]:
        if not a.startswith("--"):
            srcs.append(Path(a))
    if UPLOADS.exists():
        srcs.append(UPLOADS)
    chunks: list[dict] = seed_demo()
    seen: set[str] = set()
    for src in srcs:
        if not src.exists():
            continue
        for p in sorted(src.rglob("*")):
            if not p.is_file() or p.name.startswith(".") or p.suffix == ".caption.txt":
                continue
            if p.name in seen:
                continue
            seen.add(p.name)
            try:
                chunks.extend(index_file(p))
            except Exception as e:  # never fail the whole build on one file
                chunks.append(_mk(p.stem, f"{p.stem}-err", "text",
                                  f"Could not parse {p.name}: {e}", p.name))
    OUT.write_text(json.dumps(chunks, indent=2))
    TABLES_OUT.write_text("[]")
    print(f"wrote {len(chunks)} chunks -> {OUT}")
    n_vec = push_vectors(chunks)
    if n_vec:
        print(f"pushed {n_vec} vectors -> Qdrant")


def push_vectors(chunks: list[dict]) -> int:
    """Embed chunks with NIM + upsert to Qdrant. 0 when unconfigured/failing.

    Text chunks embed as text; image chunks with a local blob ref embed as
    base64 data-URLs (same joint space). index.json is always the fallback.
    """
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from api import llm, vectordb  # type: ignore
    except Exception:
        return 0
    if not chunks or not llm.llm_configured() or not vectordb.configured():
        return 0
    if not vectordb.ensure_collection():
        return 0
    try:
        import base64  # noqa: F401 (imported here to keep top light)
    except ImportError:
        return 0
    payloads: list[str] = []
    owners: list[int] = []
    for idx, c in enumerate(chunks):
        ref = c.get("ref") or ""
        if c.get("modality") == "image" and ref.startswith("blobs/"):
            blob = ROOT / "data" / ref
            try:
                raw = blob.read_bytes()
                if len(raw) > 1_500_000:  # skip giant figures for hosted embed
                    continue
                payloads.append("data:image/png;base64," + base64.b64encode(raw).decode())
                owners.append(idx)
                continue
            except OSError:
                pass
        text = f"{' '.join(c.get('heading_path', []))}\n{c.get('content_text', '')}"[:1500]
        if text.strip():
            payloads.append(text)
            owners.append(idx)
    vectors: list[list[float]] = []
    keep: list[int] = []
    for start in range(0, len(payloads), 32):
        batch = payloads[start:start + 32]
        vecs = llm.embed(batch, kind="passage")
        if not vecs:
            return 0
        for idx, v in zip(owners[start:start + 32], vecs):
            keep.append(idx)
            vectors.append(v)
    n = vectordb.upsert([chunks[i] for i in keep], vectors)
    vectordb.prune([chunks[i]["chunk_id"] for i in keep])  # drop stale test/doc points
    return n


if __name__ == "__main__":
    main()
