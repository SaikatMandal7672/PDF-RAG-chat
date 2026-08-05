"""Full local indexer: md/txt/csv/xlsx/pdf/docx/code/images -> data/index.json.

Dependency-light by design: every heavy parser is optional.
If a library is missing, the file is still indexed via a safe fallback
so `python3 ingest/index_local.py` never crashes (free-tier friendly).

Usage:
  python3 ingest/index_local.py
  python3 ingest/index_local.py --src data/uploads   # extra dir
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data" / "samples"
UPLOADS = ROOT / "data" / "uploads"
OUT = ROOT / "data" / "index.json"
TABLES_OUT = ROOT / "data" / "extracted_tables.json"

STOP = {
    "the", "and", "for", "with", "this", "that", "from", "have", "will",
    "your", "about", "into", "such", "they", "them", "then", "than",
    "also", "were", "what", "when", "which", "their", "there", "been",
}

CODE_EXT = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".md"}
TABLE_EXT = {".csv", ".tsv"}
SHEET_EXT = {".xlsx", ".xls"}
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
    """Structure-aware: headings split, markdown tables kept whole."""
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


def read_csv_file(path: Path) -> tuple[list[dict], dict | None]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
    except OSError:
        return ([], None)
    if not rows:
        return ([], None)
    header, body = rows[0], rows[1:6]
    md = "| " + " | ".join(header) + " |\n| " + " | ".join(["---"] * len(header)) + " |\n"
    for r in body:
        md += "| " + " | ".join(r) + " |\n"
    md += f"\n({len(body)} of {max(0, len(rows) - 1)} rows shown)"
    doc_id = path.stem
    chunk = _mk(doc_id, f"{doc_id}-table0", "table", md, f"{path.name} sheet",
                summary=f"Table from {path.name}: {', '.join(header[:5])}")
    table = {"table_id": f"{doc_id}-t0", "doc_id": doc_id, "columns": header,
             "rows": [dict(zip(header, r)) for r in rows[1:50]]}
    return ([chunk], table)


def read_xlsx(path: Path) -> tuple[list[dict], dict | None]:
    try:
        import openpyxl  # type: ignore
    except ImportError:
        return ([_mk(path.stem, f"{path.stem}-c0", "text",
                      f"Spreadsheet {path.name} uploaded (install openpyxl for cell-level parse).",
                      path.name)], None)
    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True, max_row=51))
    except Exception:
        return ([], None)
    rows = [[("" if v is None else str(v)) for v in r] for r in rows if any(v is not None for v in r)]
    if not rows:
        return ([], None)
    header, body = rows[0], rows[1:]
    md = "| " + " | ".join(header) + " |\n| " + " | ".join(["---"] * len(header)) + " |\n"
    for r in body[:8]:
        md += "| " + " | ".join(r) + " |\n"
    doc_id = path.stem
    chunk = _mk(doc_id, f"{doc_id}-table0", "table", md, f"{path.name}",
                summary=f"Sheet {ws.title}: {', '.join(header[:5])}")
    table = {"table_id": f"{doc_id}-t0", "doc_id": doc_id, "columns": header,
             "rows": [dict(zip(header, r)) for r in body[:50]]}
    return ([chunk], table)


def read_pdf(path: Path) -> list[dict]:
    text = ""
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(str(path))
        for i, pg in enumerate(reader.pages[:30]):
            t = pg.extract_text() or ""
            if t.strip():
                text += f"\n## Page {i + 1}\n{t}\n"
    except ImportError:
        pass
    except Exception:
        pass
    if not text.strip():
        try:
            import fitz  # pymupdf, optional  # type: ignore
            doc = fitz.open(str(path))
            for i, pg in enumerate(doc[:30]):
                text += f"\n## Page {i + 1}\n{pg.get_text()}\n"
        except ImportError:
            pass
        except Exception:
            pass
    if not text.strip():
        # last-resort fallback: index the filename so uploads never silently vanish
        text = f"## {path.name}\nPDF uploaded ({path.stat().st_size} bytes). Install pypdf for text extraction."
    return chunk_text(path.stem, text)


def read_docx(path: Path) -> list[dict]:
    try:
        import docx  # python-docx, optional  # type: ignore
        d = docx.Document(str(path))
        text = "\n".join(p.text for p in d.paragraphs if p.text.strip())
        if text.strip():
            return chunk_text(path.stem, text)
    except ImportError:
        pass
    except Exception:
        pass
    return chunk_text(path.stem, f"## {path.name}\nWord doc uploaded. Install python-docx for text extraction.")


def read_code(path: Path) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    # split large files on top-level defs/classes to keep symbols intact
    parts = re.split(r"(?m)^(?=(def |class |function |const |export ))", text)
    joined = [p for p in (parts or [text]) if p.strip()]
    chunks = []
    for i, part in enumerate(joined[:12]):
        first = next((l for l in part.splitlines() if l.strip()), path.name)[:60]
        chunks.append(_mk(path.stem, f"{path.stem}-code{i}", "code",
                           part[:4000], f"{path.name} :: {first}"))
    return chunks


def read_image(path: Path) -> list[dict]:
    sidecar = path.with_suffix(".caption.txt")
    if sidecar.exists():
        caption = sidecar.read_text(encoding="utf-8", errors="replace").strip()
    else:
        caption = f"Image {path.name} (add {path.stem}.caption.txt next to it for a real caption)."
    return [_mk(path.stem, f"{path.stem}-img0", "image", caption,
                 path.name, ref=f"uploads/{path.name}",
                 summary=caption[:140])]


def index_file(path: Path) -> tuple[list[dict], list[dict]]:
    suf = path.suffix.lower()
    if suf in (".md", ".txt"):
        return (read_markdown(path), [])
    if suf in TABLE_EXT:
        c, t = read_csv_file(path)
        return (c, [t] if t else [])
    if suf in SHEET_EXT:
        c, t = read_xlsx(path)
        return (c, [t] if t else [])
    if suf == ".pdf":
        return (read_pdf(path), [])
    if suf == ".docx":
        return (read_docx(path), [])
    if suf in IMG_EXT:
        return (read_image(path), [])
    if suf in CODE_EXT or suf in {".json", ".yml", ".yaml", ".toml", ".sh"}:
        return (read_code(path), [])
    return ([_mk(path.stem, f"{path.stem}-c0", "text",
                  f"File {path.name} uploaded (no parser for {suf}; stored as reference).",
                  path.name)], [])


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
    tables: list[dict] = []
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
                c, t = index_file(p)
            except Exception as e:  # never fail the whole build on one file
                c = [_mk(p.stem, f"{p.stem}-err", "text", f"Could not parse {p.name}: {e}", p.name)]
                t = []
            chunks.extend(c)
            tables.extend(t)
    OUT.write_text(json.dumps(chunks, indent=2))
    TABLES_OUT.write_text(json.dumps(tables, indent=2))
    print(f"wrote {len(chunks)} chunks -> {OUT} (+{len(tables)} tables)")


if __name__ == "__main__":
    main()
