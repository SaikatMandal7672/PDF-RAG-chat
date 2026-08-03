"""Specialist agents: text / table(+SQL-lite) / vision / code / multi."""
from __future__ import annotations

import re


def _snippet(c: dict, limit: int = 320) -> str:
    t = c.get("content_text", "")
    return t[:limit] + ("…" if len(t) > limit else "")


def text_agent(query: str, hits: list[dict]) -> tuple[str, float]:
    if not hits:
        return ("I couldn't find anything relevant in the indexed docs.", 0.0)
    lines = [f"- {_snippet(h)}" for h in hits[:3]]
    return ("Based on the indexed chunks:\n\n" + "\n".join(lines),
            min(0.9, 0.25 + 0.2 * len(hits)))


def _parse_markdown_table(md: str) -> tuple[list[str], list[list[float | str]]]:
    rows = [l.strip() for l in md.splitlines() if l.strip().startswith("|")]
    if len(rows) < 2:
        return ([], [])
    header = [h.strip() for h in rows[0].strip("|").split("|")]
    data: list[list[float | str]] = []
    for r in rows[2:]:
        cells = [c.strip().replace(",", "") for c in r.strip("|").split("|")]
        parsed: list[float | str] = []
        for cell in cells:
            try:
                parsed.append(float(cell))
            except ValueError:
                parsed.append(cell)
        data.append(parsed)
    return (header, data)


def _sql_lite(query: str, md: str) -> str | None:
    """Answer sum/avg/max/min/count over the first numeric column."""
    q = query.lower()
    op = next((o for o in ("sum", "total", "average", "avg", "max", "min", "count") if o in q), None)
    if not op:
        return None
    header, data = _parse_markdown_table(md)
    if not data:
        return None
    num_col = next((i for i, h in enumerate(header)
                    for row in data if isinstance(row[i] if i < len(row) else None, float)), None)
    if num_col is None:
        for i in range(len(header)):
            vals = [r[i] for r in data if i < len(r) and isinstance(r[i], float)]
            if vals:
                num_col = i
                break
    if num_col is None:
        return None
    vals = [r[num_col] for r in data if num_col < len(r) and isinstance(r[num_col], float)]
    if not vals:
        return None
    col = header[num_col] if num_col < len(header) else f"col{num_col}"
    if op in ("sum", "total"):
        return f"{op}({col}) = {sum(vals):g} over {len(vals)} rows"
    if op in ("average", "avg"):
        return f"avg({col}) = {sum(vals) / len(vals):g} over {len(vals)} rows"
    if op == "max":
        return f"max({col}) = {max(vals):g}"
    if op == "min":
        return f"min({col}) = {min(vals):g}"
    return f"count({col}) = {len(vals)} rows"


def table_agent(query: str, hits: list[dict]) -> tuple[str, float]:
    tables = [h for h in hits if h.get("modality") == "table"] or hits
    if not tables:
        return ("No tables matched your query.", 0.0)
    t = tables[0]
    md = t.get("content_text", "")
    calc = _sql_lite(query, md)
    head = (f"Relevant table `{t['chunk_id']}`"
            f"{' (p.' + str(t.get('page')) + ')' if t.get('page') else ''}:\n\n"
            f"```\n{_snippet(t, 800)}\n```")
    if calc:
        return (f"{head}\n\nSQL-lite: `{calc}`\n\nSummary: {t.get('summary', '')}", 0.8)
    return (f"{head}\n\nSummary: {t.get('summary', '')}", 0.7)


def vision_agent(query: str, hits: list[dict]) -> tuple[str, float]:
    imgs = [h for h in hits if h.get("modality") == "image"] or hits
    if not imgs:
        return ("No images matched your query.", 0.0)
    im = imgs[0]
    return ((f"Relevant image `{im['chunk_id']}`"
             f"{' (p.' + str(im.get('page')) + ')' if im.get('page') else ''}: "
             f"{im.get('ref', '')}\n\nCaption: {_snippet(im, 500)}"), 0.65)


def code_agent(query: str, hits: list[dict]) -> tuple[str, float]:
    code = [h for h in hits if h.get("modality") == "code"] or hits
    if not code:
        return ("No code matched your query.", 0.0)
    c = code[0]
    path = (c.get("heading_path") or [""])[-1]
    return (f"Relevant code `{c['chunk_id']}` ({path}):\n\n```\n{_snippet(c, 900)}\n```", 0.7)


def multi_agent(query: str, hits: list[dict]) -> tuple[str, float]:
    parts: list[str] = []
    confs: list[float] = []
    for agent in (text_agent, table_agent, vision_agent):
        a, c = agent(query, hits)
        if c > 0:
            parts.append(a)
            confs.append(c)
    if not parts:
        return ("I couldn't find anything relevant in the indexed docs.", 0.0)
    return ("\n\n---\n\n".join(parts[:3]), round(sum(confs) / len(confs), 2))
