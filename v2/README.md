# PDF-RAG-chat — v2 (multimodal RAG)

Text, tables (+SQL-lite aggregation), images, code. FastAPI backend + single-file UI. No login. Zero keys required.

## Run

```bash
python3 v2/ingest/index_local.py
python3 -m pip install -r v2/api/requirements.txt
python3 -m uvicorn api.main:app --port 7860  # run from v2/ dir
# open v2/ui/index.html -> API http://localhost:7860
```

Try: `What was Q3 revenue?` · `What is the returns policy?`

## Verify

```bash
python3 -m compileall -q v2/api v2/ingest
curl -s localhost:7860/health
curl -s -X POST localhost:7860/api/eval/run
```

## Layout (v2 only)

- `api/` — query/ingest/audit/review/eval/redteam + agents + retrieval + validation
- `ingest/index_local.py` — md/txt/csv/xlsx/pdf/docx/code/images → `data/index.json` (rebuild locally, not committed)
- `ui/index.html` — Ask/Data/Audit/Eval single page
- `eval_data/golden_qa.jsonl` — golden set for `/api/eval/run`
- `data/` — local only (gitignored except `.gitkeep`); run indexer to generate `index.json`
