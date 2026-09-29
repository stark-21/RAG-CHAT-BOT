# Implementation Plan — Mutual Fund FAQ Assistant

Six build phases for the RAG chatbot described in
[`PRD.md`](./PRD.md) and [`architecture.md`](./architecture.md).

Each phase below lists **files to create**, **what the phase does**, and
**how to verify it works** before moving on. No code is written here — this is
the plan you execute.

Two rules that make the phases work:

1. **Verify before advancing.** Each phase ends with a concrete check you run
   yourself. If the check fails, fix it in that phase. Do not carry a broken
   phase forward — every later phase builds on it.
2. **The verification column is the deliverable.** The demo grade (PRD
   success criteria) is judged on evidence: `chunks.txt`, the guard test output,
   the sample Q&A file, the working link. Each phase produces one of these.

---

## Phase Dependencies at a Glance

```
Phase 1  Setup            ──►  nothing depends on it
Phase 2  Load + Chunk     ──►  needs 1
Phase 3  Embed + Store    ──►  needs 2   (produces data/chroma, chunks.txt)
Phase 4  Guardrails       ──►  needs 1   (can run in parallel with 2–3)
Phase 5  Retrieve + LLM   ──►  needs 3 and 4
Phase 6  UI               ──►  needs 5
```

Phases 2–3 and 4 are the only parallelisable pair. Everything else is linear.

---

## Phase 1 — Project Setup

### Files to create

- `requirements.txt`
- `.env.example`
- `.gitignore`
- `config.py`
- `sources.py`
- `rag/__init__.py`
- `tests/` directory (empty placeholder)
- `data/raw/` directory (empty placeholder)
- `README.md` (skeleton: setup steps + scope; fill in known limits at the end)

### What this phase does

- Stand up the Python 3.10+ virtual environment and install the pinned stack:
  `sentence-transformers`, `chromadb`, `groq`, `streamlit`, `python-dotenv`,
  `requests`, `beautifulsoup4`, `pypdf`, `pytest`.
- Create `config.py`: loads `.env`, exposes the Groq key, the embedding model
  name, `top_k`, chunk size/overlap, and all paths (`data/raw`, `data/chunks.txt`,
  `data/chroma`). No secrets logged.
- Create `sources.py`: the scoped corpus registry — HDFC AMC, 5 Direct-Growth
  schemes, each with its official source URL(s) and doc type. This is the single
  source of truth for scope and matches PRD §3.
- Create `.env.example` with `GROQ_API_KEY=` (name only, empty value) and
  `.gitignore` that excludes `.env`, `data/chroma/`, `__pycache__/`.
- Skeleton `README.md` so the setup steps are written down as you go.

### How to verify it works

- `pip install -r requirements.txt` completes with no errors.
- `python -c "import chromadb, sentence_transformers, groq, streamlit, dotenv"` succeeds.
- A fresh `.env` copied from `.env.example` is read by `config.py`; printing a
  config summary shows the key as "loaded" without echoing the value.
- `python -c "import sources; print(len(sources.SOURCES))"` prints **5** (one
  per scheme) and each entry has a non-empty official `url` and `doc_type`.
- `.gitignore` does list `.env`; running `git status` shows `.env` is untracked.

---

## Phase 2 — Loading & Chunking

### Files to create

- `rag/loader.py`
- `rag/chunking.py`
- `docs/chunking_strategy.md`
- `docs/sources.md`
- `tests/test_chunking.py`
- (populated) `data/raw/` snapshots

### What this phase does

- **Inspect first.** Open the actual official pages/PDFs and look at their real
  structure before writing the splitter. Record what you see in
  `docs/chunking_strategy.md`.
- **`rag/loader.py`**: fetch/read each source (HTML via `requests`+`BeautifulSoup4`,
  factsheet PDFs via `pypdf`), normalise to clean text, stamp `fetched_at`, and
  optionally save a snapshot into `data/raw/` for repeatability.
- **`rag/chunking.py`**: split each document into retrievable chunks on heading
  and list boundaries so a chunk stays within one section and one scheme, and
  attach the metadata the architecture requires:
  `scheme`, `doc_type`, `section`, `source_url`, `fetched_at`, `chunk_idx`.
- Write the full chunk dump to `data/chunks.txt` (human-readable).
- **`docs/sources.md`**: the source list deliverable (the URLs actually ingested).

### How to verify it works

- `docs/chunking_strategy.md` states a concrete chunk size and overlap, and
  **justifies** why it suits this data.
- Run the loader+chunker and open `data/chunks.txt`. Confirm by reading it:
  - every chunk has all six metadata fields populated,
  - no chunk spans two schemes or two sections,
  - the text is clean (no stray HTML, nav bars, or broken tables).
- A sample fact is findable: search `chunks.txt` for "exit load" and confirm the
  HDFC Small Cap exit-load chunk is present and readable in one piece.
- `pytest tests/test_chunking.py` passes — at minimum it asserts metadata is
  present on every chunk and that chunks respect the max size.
- Sanity on coverage: at least one chunk exists for each of the five schemes
  (grep `chunks.txt` for each scheme name).

---

## Phase 3 — Embedding & Vector Store

### Files to create

- `rag/embeddings.py`
- `rag/vectorstore.py`
- `ingest.py` (the CLI that runs load → chunk → embed → store)
- `data/chroma/` (produced by ingest)

### What this phase does

- **`rag/embeddings.py`**: a single `all-MiniLM-L6-v2` model instance exposing
  both `embed_documents` (for chunks) and `embed_query` (for questions). One
  instance, so corpus and queries share the same 384-dim vector space.
- **`rag/vectorstore.py`**: wrap ChromaDB `PersistentClient` on disk. The
  collection holds each chunk's embedding, text, and the six metadata fields.
- **`ingest.py`**: orchestrate load → chunk → embed → store, write `chunks.txt`,
  and print a short summary (chunks per scheme, embedding dim, collection count).

### How to verify it works

- Run `python ingest.py` once. It completes, prints a summary, and creates
  `data/chroma/` on disk.
- The printed embedding dimension is **384** (confirms the right model loaded).
- Restart the store in a fresh Python process and query the collection: the
  chunk count is still there **without re-running ingest** (proves persistence
  / PRD N2).
- Spot-check a stored chunk: it round-trips with its `source_url`, `scheme`, and
  `section` metadata intact.
- Re-running `python ingest.py` is safe and does not duplicate records (idempotent).

---

## Phase 4 — Guardrails

### Files to create

- `rag/guard.py`
- `evals/guard_cases.json` (the opinion + PII probes)
- `tests/test_guard.py`

### What this phase does

- **`rag/guard.py`**: a fast, pre-LLM check run **before** retrieval. It returns
  a verdict of `ANSWER`, `REFUSE_ADVICE`, or `REFUSE_PII`:
  - **PII** — detect and reject PAN, Aadhaar, account numbers, OTPs, emails, and
    phone numbers. Never store or echo the value.
  - **Advice/portfolio intent** — detect "should I buy…", "which is better…",
    "how should I allocate…", timing calls. Route to a polite, facts-only
    refusal with a relevant educational link.
- This phase needs only Phase 1, so it can be built in parallel with 2–3.
- `evals/guard_cases.json` holds the 5 opinion probes and 3 PII probes from PRD §6.

### How to verify it works

- `pytest tests/test_guard.py` passes.
- Running `guard_cases.json` through the guard: **5/5 opinion questions →
  REFUSE_ADVICE**, **3/3 PII probes → REFUSE_PII** (satisfies PRD F4 and F5).
- A clean factual question ("What is the exit load on HDFC Small Cap Fund?") is
  **not** blocked (no false positive on the core use case).
- The refusal message contains a facts-only statement and an educational link,
  and does **not** repeat the PII back.
- Confirm nothing is written to disk or logs when a PII probe hits.

---

## Phase 5 — Retrieval + LLM Answer

### Files to create

- `rag/retrieval.py`
- `rag/prompt.py`
- `rag/generator.py`
- `rag/postprocess.py`
- `evals/sample_qa.md` (5–10 queries + expected shape + links)

### What this phase does

- **`rag/retrieval.py`**: embed the question with the same MiniLM model, query
  Chroma for top-k chunks (plus scores and metadata, optional `scheme` filter).
  Return a ground-truth "not found" when nothing clears the threshold.
- **`rag/prompt.py`**: build the system prompt (facts-only, ≤3 sentences, one
  citation, no performance/return claims, no advice, include
  "Last updated from sources:") plus numbered context blocks and the question.
- **`rag/generator.py`**: call Groq chat completions at temperature 0 with a
  bounded `max_tokens` to nudge the ≤3-sentence limit. The only module that
  knows the LLM vendor.
- **`rag/postprocess.py`**: the last line of defence — enforce exactly one
  citation link, ≤3 sentences, the freshness line, and refusal wording when the
  guard tripped. Return a structured `Answer`.
- Assemble the end-to-end query path: guard → retrieve → prompt → generate →
  postprocess.

### How to verify it works

- With a real `GROQ_API_KEY` in `.env`, the 5–10 questions in `sample_qa.md`
  return answers. For each, confirm:
  - the answer is **≤3 sentences** (F3),
  - it carries **exactly one working official citation link** (F2),
  - it ends with **"Last updated from sources:"** + a date (F6),
  - the fact **matches the cited source** — open the link and eyeball it (F1),
  - there is **no computed or compared return figure** (F7),
  - the answer is grounded in the retrieved chunks, not free recall (F8).
- Re-run the same questions twice: with temperature 0 the answers are stable
  (not required, but a good sanity signal).
- A deliberately off-topic question ("What is the weather?") returns the
  "couldn't find that in the official sources" path, not a guess.
- A return-comparison question ("Which of these has better returns?") is refused
  or redirected to the factsheet link (PRD §5).
- Latency check: each answer completes in a reasonable time on a laptop
  (PRD N1 target: full answer <10 s).

---

## Phase 6 — UI

### Files to create

- `app.py` (Streamlit)
- `render.yaml`
- `README.md` (finalised: setup, scope, known limits)
- Finalise `evals/sample_qa.md` and the disclaimer snippet

### What this phase does

- **`app.py`**: the tiny UI — welcome line, **3 example questions** (clickable
  to prefill), and the visible note **"Facts-only. No investment advice."**
  Wires the query path: guard → retrieve → generate → postprocess. Renders the
  answer, the clickable citation link, and the "Last updated from sources:" date.
  Renders refusals (advice + PII) cleanly, without echoing PII.
- **`render.yaml`**: Render free web service config (start command runs the app;
  cold start re-runs ingestion).
- **README.md**: setup steps, scope (AMC + 5 schemes), known limits, disclaimer
  snippet, and how to regenerate the corpus.
- Populate `evals/sample_qa.md` with the final 5–10 real Q&A (answers + links) as
  the deliverable.

### How to verify it works

- `streamlit run app.py` starts locally. The welcome line, 3 example questions,
  and the facts-only note are all visible on first load (PRD N7).
- Clicking an example question fills the box and returns a cited answer.
- Typing a real question returns answer + one working link + freshness date.
- An opinion question ("Should I buy HDFC ELSS Tax Saver Fund?") shows the polite
  facts-only refusal + an educational link, with no advice.
- A PII probe ("My PAN is ABCDE1234F …") is rejected and the PAN is not echoed
  back or stored.
- Restart Streamlit: it does **not** re-ingest (persistence holds locally).
- Deploy to Render free tier and confirm the public URL loads and answers a
  question; if hosting fails, record that and prepare the ≤3-min demo video
  fallback (PRD N6).
- Walk the PRD §9 deliverables checklist end to end; every box is tickable.

---

## Phase-to-Success-Criteria Traceability

| Phase | PRD criteria it proves | Evidence produced |
|---|---|---|
| 1 | N3 local run, N5 secrets safety | working env, `.gitignore` |
| 2 | N4 inspectability | `data/chunks.txt`, `chunking_strategy.md`, `sources.md` |
| 3 | N2 ingestion-once/persistence | `data/chroma/`, idempotent `ingest.py` |
| 4 | F4 refusal, F5 PII | `guard_cases.json` passing 5/5 + 3/3 |
| 5 | F1 accuracy, F2 citation, F3 length, F6 freshness, F7 no-perf, F8 grounding | `sample_qa.md` |
| 6 | N1 latency, N6 deploy, N7 UX | live link (or video), final README |

---

## Order of Execution (Recommended)

1. **Phase 1** — get the environment green.
2. **Phase 2** — inspect data, write `chunking_strategy.md`, produce `chunks.txt`.
   This is the highest-judgement phase; do not rush it.
3. **Phase 3** — embed and store; confirm persistence.
4. **Phase 4** — guardrails (build alongside 2–3 if you like).
5. **Phase 5** — retrieval + Groq + postprocess; fill `sample_qa.md`.
6. **Phase 6** — UI, then deploy (or video fallback), then finalise the README.
