# Architecture — Mutual Fund FAQ Assistant (Facts-Only RAG Chatbot)

Companion to [`PRD.md`](./PRD.md). This document describes **how** the system
is built: components, the data flow, the tech stack, the folder layout, and the
query path.

Everything here is constrained by the PRD §8 constraints: free-tier tools only,
runs locally, deployable to Render, public official sources, no PII, no
performance claims, answers ≤3 sentences with one citation.

---

## 1. Architecture at a Glance

The system is a **two-phase pipeline over a static corpus**:

- **Phase A — Ingestion (offline, run once).** Fetch/read the scoped source
  pages → normalise to text → chunk → embed → store in a persisted ChromaDB
  collection. Writes a human-readable `chunks.txt` for inspection.
- **Phase B — Query (online, per question).** Guard the input → embed the
  question with the same model → retrieve top-k chunks → build a prompt → call
  Groq → post-process (enforce citation, length, refusal) → render.

Ingestion and query are **separate processes** and share only the vector store
and the config. This is what makes "ingestion runs once, not on every restart"
true, and it is what lets Render start the app without re-scraping.

```
        OFFLINE / ONCE                          ONLINE / PER QUESTION
┌────────────────────────────┐            ┌──────────────────────────────────┐
│  Sources (official pages)  │            │  User question                   │
│            │               │            │        │                         │
│            ▼               │            │        ▼                         │
│  Loader  → normalise       │            │  Guard (PII / advice)            │
│            ▼               │            │        │                         │
│  Chunker (metadata-rich)   │            │        ▼                         │
│            ▼               │            │  MiniLM embed (384-d)            │
│  MiniLM embed (384-d)      │            │        │                         │
│            ▼               │            │        ▼                         │
│  ChromaDB (on disk) ◄──────┼────────────┼──  Top-k retrieve                │
│            │               │            │        │                         │
│            ▼               │            │        ▼                         │
│  chunks.txt (inspectable)  │            │  Prompt builder                  │
└────────────────────────────┘            │        │                         │
                                          │        ▼                         │
                                          │  Groq LLM (temp 0)               │
                                          │        │                         │
                                          │        ▼                         │
                                          │  Post-process + answer           │
                                          └──────────────────────────────────┘
```

---

## 2. Components

| # | Component | Responsibility | Key interface |
|---|---|---|---|
| 1 | **Source registry** | The scoped corpus definition: HDFC AMC, 5 Direct-Growth schemes, and the official URLs per scheme. Single source of truth for scope. | `sources.py` → list of `SourceDoc(url, scheme, doc_type)` |
| 2 | **Loader** | Fetch/read each source page and normalise HTML/PDF/text to clean plain text. Records `fetched_at`. | `load(docs) -> list[RawDoc]` |
| 3 | **Chunker** | Inspect-driven split into retrievable units, attaches metadata. Emits `chunks.txt`. | `chunk(raw_docs) -> list[Chunk]` |
| 4 | **Embedder** | Wraps `all-MiniLM-L6-v2`. **One** embedder instance used for both corpus and queries, so vectors live in the same space. | `embed_documents(texts)`, `embed_query(text)` |
| 5 | **Vector store** | ChromaDB `PersistentClient` on disk. Collection holds embeddings + metadata + source text. | `get_collection() -> Collection` |
| 6 | **Ingest pipeline** | Orchestrates 1→5, runs once, is idempotent (re-running replaces the collection). | `python ingest.py` |
| 7 | **Guard** | Fast pre-LLM checks: reject PII (PAN/Aadhaar/account no./OTP/email/phone) and detect advice/portfolio intent → route to refusal. | `inspect(question) -> Verdict` |
| 8 | **Retriever** | Embeds the question, queries Chroma with metadata filter, returns top-k chunks + scores. | `retrieve(question, k) -> list[ScoredChunk]` |
| 9 | **Prompt builder** | Assembles the system prompt (rules: facts-only, ≤3 sentences, one citation, no performance claims, "Last updated from sources:") plus the retrieved context and the question. | `build(context, question) -> messages` |
| 10 | **Generator** | Calls Groq chat completions, temperature 0, max tokens bounded to enforce brevity. | `complete(messages) -> str` |
| 11 | **Post-processor** | Enforces the non-negotiables: citation link present, ≤3 sentences, freshness line, refusal wording if guard tripped. Last line of defence. | `finalise(raw, context) -> Answer` |
| 12 | **UI** | Tiny web UI: welcome line, 3 example questions, "Facts-only. No investment advice." note, answer + citation. | Streamlit app |
| 13 | **Config** | Reads `.env` (Groq key, model names, k, paths). Never logs secrets. | `config.py` |

### Component rules

- **12 depends on 7 → 8 → 9 → 10 → 11**, and only on those. The UI never touches
  Chroma or Groq directly.
- **Only 4, 5, 6, 8** know about embeddings/vectors. 9 and 10 receive plain text.
- **Only 10** knows the LLM vendor. Swapping Groq means changing one module.
- **7 runs before 8.** PII and advice questions never reach retrieval or the LLM.

---

## 3. Data Flow

### 3.1 Ingestion (once)

```
sources.py (registry)
      │
      ▼
[1] Loader  ── fetch/read each official page ──► clean text + fetched_at
      │
      ▼
[2] Chunker ── structure-aware split ──► Chunk{text, scheme, doc_type, section, url, fetched_at}
      │
      ├──► write chunks.txt  (human-readable, for inspection)
      ▼
[3] Embedder ── all-MiniLM-L6-v2 ──► 384-dim vector per chunk
      │
      ▼
[4] Vector store ── ChromaDB PersistentClient ──► data/chroma/  (persisted)
```

Metadata carried on every chunk (this is what makes answers citable and
filterable):

| Field | Why it exists |
|---|---|
| `scheme` | HDFC Large Cap / Flexi Cap / ELSS / Small Cap / Balanced Advantage |
| `doc_type` | factsheet, KIM, SID, scheme FAQ, fee page, riskometer, statement guide |
| `section` | e.g. "Expense ratio", "Exit load", "Minimum SIP", "Lock-in" |
| `source_url` | **the citation link** shown to the user |
| `fetched_at` | powers the "Last updated from sources:" line |
| `chunk_idx` | stable ordering within a document |

Chunking strategy is chosen **after inspecting the real data**; the chosen
values and the reasoning behind them are recorded in
[`docs/chunking_strategy.md`](./chunking_strategy.md) (Phase 2 of
[`implementation.md`](./implementation.md)). Starting hypothesis to test
against the data: ~350–500 tokens with ~50–75 token overlap, split on heading
and list boundaries so a chunk never spans two sections or two schemes.

### 3.2 Query (per question)

```
question
   │
   ▼
[7] Guard ── PII? ──────────────► refuse, do not retrieve, do not store
   │        advice/portfolio? ───► refusal + educational link
   ▼
[8] Retriever ── embed(question) ─► Chroma top-k (+ optional scheme filter)
   │                                 + source_url + fetched_at
   ▼
[9] Prompt builder ── system rules + numbered context blocks + question
   │
   ▼
[10] Generator ── Groq, temperature 0, bounded max_tokens
   │
   ▼
[11] Post-processor ── ≤3 sentences, exactly one citation, freshness line,
   │                    no performance claims, no advice
   ▼
[12] UI ── answer + clickable source link + "Last updated from sources: <date>"
```

If retrieval returns nothing above threshold, the app answers
"I couldn't find that in the official sources" and links to the scheme page,
rather than letting the LLM guess. **Grounding beats fluency.**

---

## 4. Text Diagram — Query Flow

```
┌──────────────┐
│  User types  │   "What's the exit load on HDFC Small Cap Fund?"
└──────┬───────┘
       v
┌──────────────┐   PAN / Aadhaar / account no. / OTP / email / phone
│    GUARD     │──► present?  ──────────────► [REFUSE: facts-only] ──► UI
└──────┬───────┘   "should I buy…", "which is better…", "allocate my…"
       │            intent?  ───────────────► [REFUSE + educational link] ──► UI
       │ clean, factual
       v
┌──────────────┐
│   EMBEDDER   │   all-MiniLM-L6-v2  (same model as the corpus)  -> 384-d
└──────┬───────┘
       v
┌──────────────┐
│  CHROMADB    │   top-k = 4..6 chunks   (persisted, on disk)
│  (persist.)  │   + source_url, scheme, section, fetched_at
└──────┬───────┘
       v
┌──────────────┐
│   PROMPT     │   system: facts-only | <=3 sentences | cite 1 source
│   BUILDER    │           no returns/performance | no advice
└──────┬───────┘   context: [1] url | scheme | section | text …
       v
┌──────────────┐
│   GROQ LLM   │   temperature = 0, max_tokens bounded
└──────┬───────┘
       v
┌──────────────┐   enforce: <=3 sentences | 1 citation | "Last updated
│ POST-PROCESS │            from sources: <date>" | no performance claim
└──────┬───────┘
       v
┌──────────────────────────────────────────────────────────┐
│  UI                                                                 │
│  Answer text (2-3 sentences)                                 │
│  Source: <official link>                                      │
│  Last updated from sources: <date>                            │
│  Facts-only. No investment advice.                             │
└──────────────────────────────────────────────────────────┘
```

**Text diagram — Ingestion (once)**

```
sources.py  ──►  [Loader]  ──►  [Chunker]  ──┬──►  chunks.txt (inspect)
 5 schemes      fetch/read      split by        │
 official URLs   + clean         section,
                                    metadata
                                       │
                                       v
                                  [Embedder]  all-MiniLM-L6-v2
                                       │
                                       v
                                  [ChromaDB PersistentClient]
                                       │
                                       v
                                 data/chroma/  (runs once)
```

---

## 5. Tech Stack

| Layer | Choice | Version / notes | Why |
|---|---|---|---|
| Language | **Python** | 3.10+ | Brief-specified; best fit for the ML libs used |
| Embeddings | **`sentence-transformers/all-MiniLM-L6-v2`** | 384-dim, local, **no API key** | Brief-specified. Same model embeds chunks *and* questions. Runs on CPU, ~90 MB. |
| Vector DB | **ChromaDB** | `chromadb`, `PersistentClient` on disk | Brief-specified. Zero-config, persists locally, ingestion runs once |
| LLM | **Groq** | `groq` Python SDK, key in `.env` | Brief-specified. Free tier, fast; temperature 0 for factual answers |
| UI | **Streamlit** | `streamlit` | Tiny UI in a few lines; well suited to the "welcome + 3 examples + note" requirement |
| Config / secrets | **`python-dotenv`** | `.env` + `.gitignore` | Key never committed |
| Fetching | `requests` + `BeautifulSoup4` (static pages) | ingestion only | Official pages; optional if snapshots are committed instead |
| Parsing | `pypdf` for factsheet PDFs | ingestion only | Factsheets are commonly PDF |
| Deployment | **Render** | free web service, `render.yaml` | Brief-specified deploy target |
| Test/smoke | `pytest` | small Q&A regression set | Guards the "one citation, ≤3 sentences" rules |

**Free-tier check:** MiniLM runs locally (no cost), ChromaDB is local (no cost),
Groq free tier is the only metered call, Streamlit and Render free tier are
free. No paid dependency anywhere.

**Secrets:** `GROQ_API_KEY` lives only in `.env` (git-ignored). `.env.example`
ships the key *name* with an empty value. Nothing else in the stack needs a key.

---

## 6. Folder Structure

```
RAG chat Bot/
├── app.py                     # Streamlit UI (component 12)
├── config.py                  # .env loader, paths, tunables (component 13)
├── sources.py                 # Scoped corpus registry: 5 schemes + official URLs
├── ingest.py                  # Ingestion CLI — run once  (component 6)
│
├── rag/
│   ├── __init__.py
│   ├── loader.py              # fetch + clean + normalise      (component 2)
│   ├── chunking.py            # structure-aware splitter + metadata (component 3)
│   ├── embeddings.py          # all-MiniLM-L6-v2 singleton     (component 4)
│   ├── vectorstore.py         # ChromaDB PersistentClient wrapper (component 5)
│   ├── guard.py               # PII + advice-intent checks      (component 7)
│   ├── retrieval.py           # embed query + top-k             (component 8)
│   ├── prompt.py              # system rules + context assembly (component 9)
│   ├── generator.py           # Groq call, temp 0               (component 10)
│   └── postprocess.py         # enforce citation/length/freshness (component 11)
│
├── data/
│   ├── raw/                   # saved source snapshots (optional, for repeatability)
│   ├── chunks.txt             # human-readable chunk dump  ← inspect this
│   └── chroma/                # persisted ChromaDB directory  (created by ingest)
│
├── tests/
│   ├── test_chunking.py
│   ├── test_guard.py
│   └── test_postprocess.py
│
├── docs/
│   ├── PRD.md                 # product requirements (source of scope)
│   ├── architecture.md        # this document
│   ├── implementation.md      # 6-phase build plan + per-phase verification
│   ├── chunking_strategy.md   # the inspected-data strategy + rationale
│   └── sources.md             # the source list deliverable
│
├── evals/
│   ├── sample_qa.md           # 5–10 queries + expected answer shape + links
│   └── guard_cases.json       # opinion + PII probes used to verify F4/F5
││
├── .env.example               # GROQ_API_KEY=  (name only, no value)
├── .gitignore                 # .env, data/chroma/, __pycache__/
├── requirements.txt
├── render.yaml                # Render free web service
└── README.md                  # setup, scope (AMC + schemes), known limits
```

### Why this layout

- **`rag/` is pure and framework-free.** No Streamlit, no Groq SDK calls outside
  `generator.py`, no network I/O outside `loader.py`. That keeps components
  testable and swappable.
- **Top-level scripts map to entry points**, not to layers: `app.py` serves,
  `ingest.py` builds. A newcomer sees two commands, not twenty modules.
- **`data/` separates build artifacts from code.** `data/chroma/` is git-ignored
  and regenerated by `python ingest.py`; `chunks.txt` is committed-adjacent
  (human-readable, reviewable) evidence of the chunking step.
- **Configuration is one file.** Change the model, `k`, or chunk size in
  `config.py`, not across modules.

### Run order

```bash
pip install -r requirements.txt
cp .env.example .env        # add GROQ_API_KEY
python ingest.py            # once: load -> chunk -> embed -> store
streamlit run app.py        # query: serve the UI
```

On Render free tier the filesystem is **ephemeral — there is no persistent
disk**. The deployed app therefore re-runs `python ingest.py` on every cold
start. Two consequences that shape the design:

- Ingestion must be **fast and idempotent** (fixed corpus, no unbounded
  re-scraping, deterministic chunking), so a cold start is tolerable.
- Model download (~90 MB MiniLM) and embedding cost must stay within Render's
  free memory/time budget. If cold starts prove too slow, fall back to the
  ≤3-min demo video (PRD N6) and say so in the README.

---

## 7. Design Decisions and Their Reasons

| Decision | Reason | Trade-off accepted |
|---|---|---|
| Ingestion separate from the app process | Guarantees "ingestion runs once, not on every restart"; app boot stays fast | Two commands instead of one |
| One embedder for docs **and** queries | Vector space must match; the brief requires the same model | None — this is the correct call |
| ChromaDB `PersistentClient` on disk | No server, no re-embedding on restart, survives restarts | Needs a rebuild on ephemeral deploy disks |
| Guard before retrieval | PII and advice prompts never reach the vector store or the LLM | Fast heuristic may miss paraphrases; LLM prompt backs it up |
| Guard is a fast heuristic, not an LLM call | Keeps advice-refusal instant and zero-cost | Heuristic tuning needed for coverage |
| Temperature 0, bounded `max_tokens` | Factual, reproducible answers; nudges toward the ≤3-sentence limit | Slightly less fluent phrasing |
| Post-processing enforces the rules | The UI contract (one citation, ≤3 sentences, freshness line) must hold even if the LLM drifts | Some answer rewriting in code |
| Chunk metadata includes `source_url` | The citation is data, not something the model has to remember or invent | Richer metadata per chunk |
| Ground-truth refusal when retrieval is thin | Prevents confident hallucination; "not in sources" beats a guess | Some valid questions may be deflected if chunking is poor |
| No performance/return computation | Brief forbids it; reduces to "see the official factsheet" | Cannot answer comparative return questions |
| Snapshots optional in `data/raw/` | Keeps the demo reproducible and re-runnable without re-hitting AMC sites | Source drift over time |
| Free-tier only, local embeddings | Demo must be zero-cost and run offline except Groq | No proprietary/hosted embedding quality |

**Source-selection note.** The milestone brief lists `groww.in` scheme URLs but
also requires sources be **official** AMC/SEBI/AMFI pages with no third-party
blogs. Treat `groww.in` as a discovery/linking aid only; the ingested corpus
must be the official HDFC AMC scheme pages, factsheets, KIM/SID, fee pages, and
SEBI/AMFI documents. `sources.md` records exactly which URLs were ingested.

---

## 8. Mapping PRD → Architecture

| PRD requirement | Component |
|---|---|
| Ingestion runs once, persisted (N2) | `ingest.py` + `vectorstore.py` (Chroma on disk) |
| Same model for chunks and questions | `embeddings.py` single MiniLM instance |
| Inspectable `chunks.txt` + justified chunking | `chunking.py` → `data/chunks.txt`; `docs/chunking_strategy.md` |
| One citation in every answer | `source_url` metadata → `postprocess.py` |
| ≤3 sentences | Bounded `max_tokens` in `generator.py` + `postprocess.py` |
| "Last updated from sources:" | `fetched_at` metadata → `postprocess.py` |
| Refuse advice/portfolio questions | `guard.py` + system prompt in `prompt.py` |
| No PII accepted or stored | `guard.py` runs before retrieval; nothing persisted from user input |
| No performance claims | System prompt rule + `postprocess.py` filter |
| Facts-only disclaimer in UI | `app.py` banner |
| Free tier / local / Render | §5 stack; `render.yaml` |
