# Mutual Fund FAQ Assistant

A **facts-only RAG chatbot** that answers questions about five HDFC Asset
Management mutual fund schemes using official public pages as its only source.

> **Facts-only. No investment advice.**

Every answer is grounded in retrieved source chunks, is at most 3 sentences,
carries exactly one official citation link, and ends with a
`Last updated from sources:` date. Opinionated and portfolio questions are
refused. PII is never accepted or stored.

---

## Status

Phases 1-6 are planned in [`docs/implementation.md`](docs/implementation.md).
**Phase 1 (project setup) is complete.** Later phases are not built yet.

| Phase | What it does | Status |
|---|---|---|
| 1 | Project setup, config, corpus registry | done |
| 2 | Loading & chunking | not started |
| 3 | Embedding & vector store | not started |
| 4 | Guardrails | not started |
| 5 | Retrieval + LLM answer | not started |
| 6 | UI + deployment | not started |

## Scope

**AMC:** HDFC Asset Management. All plans are **Direct - Growth**.

| Category | Scheme |
|---|---|
| Large Cap | HDFC Large Cap Fund - Direct Growth |
| Flexi Cap | HDFC Equity (Flexi Cap) Fund - Direct Growth |
| ELSS | HDFC ELSS Tax Saver Fund - Direct Growth |
| Small Cap | HDFC Small Cap Fund - Direct Growth |
| Hybrid | HDFC Balanced Advantage Fund - Direct Growth |

Sources must be **official** AMC / SEBI / AMFI pages only. See
[`docs/sources.md`](docs/sources.md) once populated.

## Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.10+ |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, 384-dim, no API key) |
| Vector DB | ChromaDB, persisted to disk |
| LLM | Groq (temperature 0), key in `.env` |
| UI | FastAPI + hand-built HTML/CSS/JS in `static/` (no Streamlit) |

Everything is free tier. The only metered call is Groq.

## Setup

```bash
# 1. Virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 2. Dependencies
pip install -r requirements.txt

# 3. Secrets - copy the template and add your free-tier Groq key
cp .env.example .env        # Windows: copy .env.example .env
# Get a key at https://console.groq.com/keys

# 4. Confirm the config loads (prints "loaded", never the key)
python -c "import config; print(config.summary())"

# 5. Inspect the corpus registry
python sources.py
```

> **Note on this machine:** the `python` on `PATH` may be a broken install.
> If `import encodings` fails, use a working interpreter, e.g.
> `& "$env:LOCALAPPDATA\Python\pythoncore-3.14-64\python.exe"`.

## Run

```bash
python ingest.py            # once: load -> chunk -> embed -> store
python -m uvicorn server:app --port 8000    # serve the UI
```

Ingestion is idempotent: re-running it rebuilds the store rather than
duplicating records.

The deployed entry point is `server.py`, a FastAPI app that serves `static/` and
exposes `POST /api/chat`. It is a thin adapter over `rag.pipeline.answer_question()`
and changes no pipeline behaviour — same guards, same grounding gate, same
post-processing, same conversation memory. `app.py` is the superseded Streamlit
front end, kept only because `tests/test_app.py` exercises it.

## Project layout

```
.
├── server.py               # FastAPI: static/ + /api/chat            (Phase 6)
├── static/                 # the UI: index.html, styles.css, app.js
├── app.py                  # superseded Streamlit UI          (Phase 6, legacy)
├── config.py               # .env loader, paths, tunables     (Phase 1)
├── sources.py              # corpus registry: 5 schemes       (Phase 1)
├── ingest.py               # ingestion CLI - run once          (Phase 3)
├── rag/                    # pure pipeline modules            (Phases 2-5)
├── data/
│   ├── raw/                # source snapshots
│   ├── chunks/             # inspectable chunk dump           (Phase 2)
│   └── chroma/             # persisted vector store           (Phase 3)
├── tests/
├── evals/
└── docs/
```

## Known limits

To be completed at the end of Phase 6. Anticipated ones:

- Answers only cover the five schemes above, in the Direct - Growth plans.
- No returns, NAV history, or performance comparison is computed; the bot
  redirects those to the official factsheet.
- Corpus freshness is bounded by when `ingest.py` was last run.
- Render's free tier has no persistent disk, so a cold start re-runs ingestion.

## Documentation

- [`docs/PRD.md`](docs/PRD.md) - goals, scope, success criteria, constraints
- [`docs/architecture.md`](docs/architecture.md) - components, data flow, stack
- [`docs/implementation.md`](docs/implementation.md) - 6-phase build plan
- [`docs/chunking_strategy.md`](docs/chunking_strategy.md) - Phase 2
- [`docs/sources.md`](docs/sources.md) - source list deliverable
