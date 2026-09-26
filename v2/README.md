# multimodal-rag (final — all phases)

Multimodal RAG: single-page zinc UI (Ask/Data/Audit/Eval) + FastAPI backend.
PDF-first: text, tables and images inside your PDFs (+ md/txt notes, png/jpg charts). No login. Zero keys required;
Qdrant/Gemini light up automatically if env vars are set.

## Run locally

```bash
python3 ingest/index_local.py          # rebuild data/index.json (+ extracted_tables.json)
python3 -m pip install -r api/requirements.txt
python3 -m uvicorn api.main:app --port 7860
open ui/index.html                    # set API URL to http://localhost:7860
```

Try: `What was Q3 revenue?` · `total revenue across quarters` ·
`What does the revenue chart show?` · `What is the returns policy?`

## Verify (must stay green)

```bash
python3 -m compileall -q api ingest
curl -s localhost:7860/health
curl -s -X POST localhost:7860/api/eval/run
curl -s -X POST localhost:7860/api/redteam/run
```

Current gates: golden eval 4/4, red-team 6/6, injection blocked, 404s correct.

## Deploy ($0)
See `deploy-free.md`: HF Spaces Docker (API, port 7860) + Vercel static (`ui/`).
Commit `data/index.json`; live tier is query-first, ingest via `/api/ingest` or local reindex.

## Layout
- `api/` — `main.py` (query/ingest/audit/review/eval/redteam), `reasoning.py` (rules + optional Gemini planner),
  `retrieval.py` (hybrid + RRF + rerank + cache + compress), `agents.py` (text/table-SQL/vision/code/multi),
  `validation.py` (gatekeeper/redaction), `store.py` (SQLite audit + JSONL fallback),
  `eval.py` (golden runner), `redteam.py` (3 suites)
- `ingest/index_local.py` — pdf (+ md/txt notes, png/jpg charts) → `data/index.json`
- `ui/index.html` — single-file app (Ask/Data/Audit/Eval tabs, zinc)
- `plan.md` — architecture; `eval_data/golden_qa.jsonl` — golden set
