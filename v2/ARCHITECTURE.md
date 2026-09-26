# Multimodal AI RAG — Architecture (as built) + Interview Guide

> One-line pitch: "PDFs in, cited AI answers out — text, tables and figures,
> with safety gates, audit logs, golden evals and red-team suites around it."

## 1. Say it in 30 seconds

"PDFs go through an indexer that pulls page text, keeps tables whole, and
extracts embedded figures with page context into a JSON index — plus
2048-dim vision-language vectors in Qdrant. A query passes a safety gate,
gets classified to text, table, image, or multi, hybrid-retrieves
candidates, and a specialist agent has NIM write the answer from those
chunks only, with citations. Safety, audit logging, golden evals, and
red-team suites wrap the whole path; anything ungrounded gets flagged
for human review."

## 2. Say it in 2 minutes (follow the two flows)

OFFLINE (ingest/index_local.py, run once per doc set):
samples + uploads PDFs → pypdf/pymupdf page text ("## Page N" markers) →
tables kept whole (| rows never split) → embedded figures pulled to
data/blobs/*.png with page + surrounding text → keywords + 2 hypothetical
questions per chunk → data/index.json (6 chunks live: 4 text, 1 table,
1 image) → NIM-embedded (2048-d, vision-language joint space) + pushed to Qdrant
(Qdrant Cloud live: 6 points, 2048 cosine — typo/paraphrase queries retrieve
semantically, e.g. dense leg ranks chart-image 0.411 for "Q3 revnue?").

ONLINE (every question, api/main.py orchestrates):
UI Ask → POST /api/query → 1) gatekeeper blocks injection → 2) plan()
classifies intent (rules → NIM → legacy Gemini) → 3) hybrid_search fuses
keyword rankings + Qdrant dense ranking via RRF, reranks, compresses to
5k chars → 4) specialist agent (text/table/vision/code/multi) has NIM
write fresh sentences from hits, fallback = paste → 5) PII redact +
output-safe + auditor (empty hits ⇒ needs_review) → 6) logged to
audit.db + queries.jsonl → answer + citations + route + confidence.

## 3. Flow diagrams

### 3a. Ingest

```
PDFs ─┬─ TEXT: pypdf → pymupdf → stub ──▶ "## Page N" blocks
      ├─ TABLES: pipe-rows kept whole, headings split, 3000-char cap
      └─ IMAGES: pymupdf figures → data/blobs/ + page + page-text context
            (+ md/txt notes, + png/jpg via .caption.txt sidecar)
                        │
         keywords + HyDE questions (_mk)
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
   data/index.json (always)     NIM VL-embed + Qdrant (dark until creds)
```

### 3b. Query

```
POST /api/query {query, top_k, modality?, session_id}
  1. gatekeeper ── injection? ──▶ BLOCKED + logged + needs_review
  2. plan() ── intent ∈ {text,table,image,code,multi} (planner: rules|nim)
  3. hybrid_search: keyword BM25-ish + keyword-dense + Qdrant dense
     ──RRF fuse──▶ rerank (phrase/proximity) ──▶ compress (5k, edges kept)
  4. agent → llm.generate() (grounded, [chunk_id] cites) ──▶ fallback: paste
  5. redact_pii → output_safe → auditor ──▶ log_query()
◀── {answer_markdown, citations[], route, confidence, latency_ms}
```

### 3c. Per-modality handling

| Stage    | Text | Table | Image |
|----------|------|-------|-------|
| Extract  | page text w/ page markers | pipe-rows whole; native PDF tables land as text (known gap, no Camelot) | embedded figures → blobs; charts also via caption sidecars |
| Chunk    | heading-split blocks | `table` iff first line starts with `\|` | `image` chunk w/ `ref` + page |
| Retrieve | keyword (+ dense when lit) | same; SQL-lite math in fallback | caption/page-text keyword (+ pixel vectors when lit) |
| Agent    | `answer_directly` | `table_agent` (exact numbers; total=46.4 verified live) | `vision_agent` / `multi_agent` |
| Proof    | `[demo-notes-c2]` live | Q3=18.9 cited live | `[demo-chart-image]` live |

### 3d. Safety / ops ring (both reference diagrams)

```
Gatekeeper (regex block) → answer → Auditor (citation check) → Audit UI + review queue
Golden eval 4/4 (substring + optional NIM judge, NIM_EVAL_JUDGE=1)
Red-team 6/6 (injection / evasion / bias — CI-only, off prod path)
Strategist = human reading /api/eval/summary (only unautomated box)
```

## 4. File map (what each file does)

| File | Role | Notes |
|------|------|-------|
| `api/main.py` | FastAPI wiring: query/ingest/citations/audit/review/eval/redteam; loads `.env` | logic-free orchestrator |
| `api/models.py` | Pydantic contracts (request/response/citation/review) | 37 lines |
| `api/reasoning.py` | `plan()` intent + `route()` decision; rules → NIM classify → legacy Gemini | planner label exposed |
| `api/llm.py` | NIM phone: `generate`/`judge`/`embed`/`classify`; SYSTEM_ANSWER + hints; fail-soft `None` | key never logged |
| `api/retrieval.py` | keyword hybrid + RRF + rerank + compress + cache; dense Qdrant leg (dark) | stdlib core |
| `api/agents.py` | text/table/vision/code/multi; LLM-first, extractive fallback; stable confidences | SQL-lite math inside |
| `api/validation.py` | gatekeeper regex, PII redact, output-safe, auditor | bounds as code |
| `api/store.py` | SQLite audit.db + queries.jsonl fallback; stats for dashboard | free-tier safe |
| `api/vectordb.py` | Qdrant Cloud wrapper (2048 cosine, UUID-per-chunk); fail-soft | LIVE: 6 points; dense leg fused in RRF |
| `api/eval.py` | golden runner (normalized match) + env-gated NIM judge | 4/4 live |
| `api/redteam.py` | 3 suites × real pipeline functions | 6/6 live |
| `api/strategist.py` | audit-based tuning advisor (routes/conf/blocks → suggestions) | live at GET /api/strategy, read-only |
| `ingest/index_local.py` | PDF-first indexer + figure pull + vision captions + vector push + Qdrant prune | lines change as built |
| `ui/index.html` | single-file Ask/Data/Audit/Eval app | no build step |
| `eval_data/golden_qa.jsonl` | 4 cases: revenue table, chart, returns, injection | eval contract |
| `data/index.json` | built index (portable snapshot — clone + run, no rebuild) | 6 chunks |
| `.env` | NVIDIA_NIM_*, NIM_*_MODEL, QDRANT_* (gitignored, never committed) | user fills |

## 5. Numbers to quote

- 6 chunks (4 text / 1 table / 1 image); 2048-dim vectors (one model, words+pixels same space)
- Query path: classify ~1–5s + generate ~2.5–5s + retrieval ~0s → p50 ≈ 5–7s end-to-end
- Golden eval 4/4 (p50 ~5s, citation coverage 0.75); red-team 6/6
- Generation model: `openai/gpt-oss-20b` (only fast+clean one on this NIM account)
- Embed model: `nvidia/llama-nemotron-embed-vl-1b-v2` (text+image, `input_type` query/passage)

## 6. Model scoreboard (verified on this NIM account)

| Model | Result |
|-------|--------|
| `openai/gpt-oss-20b` | ✅ ~1–5s, clean, cites — THE pick |
| `meta/muse-glimmer-30b` | ⚠️ trivial ok, grounded prompts time out |
| `z-ai/glm-5.3-flash` | ⚠️ works, 68–96s |
| `nvidia/nemotron-3.5-lightning-30b-a3b` | ⚠️ 111s + leaks reasoning into answers |
| `deepseek-ai/deepseek-v4.1-flash` | ❌ hangs forever |
| `meta/llama-3.1-8b-instruct` | ❌ EOL (410) |
| `meta/muse-glimmer-30b`, most others | ❌ listed but 404 (not provisioned) |
| `nvidia/nvclip` | ❌ 404 — stays on VL embed model |

## 7. Likely interview Q&A

1. **RAG in one line?** Retrieve matching pieces, generate the answer from them only, attach proof.
2. **Why NIM?** OpenAI-compatible hosted models; no GPUs to manage; key-only auth.
3. **Why 2048-dim?** Fixed output width of the embed model; text+image share it so cross-modal cosine works; Qdrant collection is pinned to it.
4. **Why RRF fusion?** Three rankings (2 keyword + 1 dense) vote; robust when any leg is weak, and the dense leg can be dark without code changes.
5. **Why fallback everywhere?** Free-tier reality: no key / timeout / no Qdrant must still serve keyword+paste answers. Every LLM/vector call returns None on failure.
6. **How are tables handled?** Pipe tables preserved whole; SQL-lite does sum/avg/max in Python; native PDF tables currently land as text (stated gap, Camelot-class fix).
7. **How are images handled?** Embedded figures extracted with page+context (keyword-findable); pixel vectors live in Qdrant; machine captions via `llama-3.2-11b-vision` at ingest (proven: colors/order right, values need human check — blob stays attached).
8. **How do you stop hallucinations?** Grounded system prompt + cite-every-claim + auditor + needs_review + eval substring + optional LLM judge.
9. **The unicode eval bug?** Model wrote U+2011 hyphen; harness now normalizes dashes/quotes before matching — compare meaning, not typography.
10. **What's not automated?** Nothing structural — Strategist advises via `/api/strategy` (human approves by design); dense retrieval live; vision live.
11. **Latency story?** 0ms extractive → ~5s AI answers (~23s/figure one-time at ingest); p50 tracked in eval; semantic cache + top-k caps contain cost.
12. **What would you do with more time?** Real-PDF corpus proof round (only remaining proof — needs external data), bigger-corpus paraphrase benchmark, prompt A/B via judge_mean.

## 8. Go-live checklist — all buildable items done

- [x] `.env` chat model → `openai/gpt-oss-20b`
- [x] Qdrant Cloud → 6 vectors live, typo query answered with citation
- [x] Vision captioning wired (fail-soft, ingest-time)
- [x] Strategist advisory endpoint live
- [ ] Feed real PDFs (external data, not code)
