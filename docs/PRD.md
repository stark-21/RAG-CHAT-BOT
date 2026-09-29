# PRD — Mutual Fund FAQ Assistant (Facts-Only RAG Chatbot)

**Status:** Draft for class demo milestone
**Owner:** RAG Chat Bot team
**Last updated:** 2026-09-29

---

## 1. Goal

Build a small, working RAG chatbot that answers **factual** questions about a
scoped set of HDFC Asset Management mutual fund schemes using **only official
public pages** as its knowledge base.

Every answer must:

- be derived from retrieved source chunks (no free-form recall),
- be **≤ 3 sentences**,
- include **exactly one clear citation link**,
- end with a **"Last updated from sources:"** timestamp,
- stay **facts-only** — no recommendations, no return/performance claims.

The goal is to demonstrate the **full RAG pipeline** end to end:
`Ingestion (Load → Chunk → Embed → Store)` and
`Query (Question → Embed → Retrieve top-k → LLM → Answer)`,
with the chunking strategy deliberately chosen after inspecting the data.

**Success in one line:** a retail user asks "What's the exit load on HDFC Large Cap?"
and gets a short, cited, verifiable answer in under ~8 seconds.

---

## 2. Target Users

| User | Need | Pain today |
|---|---|---|
| **Retail investor (primary)** | Quick, trustworthy facts while comparing schemes — expense ratio, exit load, min SIP, lock-in, benchmark, riskometer | Facts are scattered across PDFs, AMC pages, and aggregator sites; often answered by opinion |
| **Support / content team (secondary)** | Deflect repetitive MF questions with consistent, source-backed answers | Agents repeat the same FAQ answers manually; risk of unsourced or outdated claims |
| **Evaluator / instructor** | See a working, reproducible RAG prototype that follows the required stack | Needs a live link or ≤3-min video plus inspectable artifacts |

**Non-users (explicit):** anyone seeking investment advice, tips, or
portfolio/allocation guidance.

---

## 3. Scope of the Knowledge Base

One AMC: **HDFC Asset Management**. Five schemes, all **Direct – Growth** plan:

| # | Category | Scheme |
|---|---|---|
| 1 | Large Cap | HDFC Large Cap Fund – Direct Growth |
| 2 | Flexi Cap | HDFC Equity (Flexi Cap) Fund – Direct Growth |
| 3 | ELSS | HDFC ELSS Tax Saver Fund – Direct Growth |
| 4 | Small Cap | HDFC Small Cap Fund – Direct Growth |
| 5 | Hybrid | HDFC Balanced Advantage Fund – Direct Growth |

Source page types (official only): scheme pages, factsheets, KIM/SID, scheme
FAQ pages, fee & charge pages, riskometer / benchmark notes, and
statement/tax-document guides. No third-party blogs. A source list of the URLs
used is a required deliverable.

---

## 4. In-Scope Features

1. **Document ingestion pipeline**
   - Load the scoped public pages / saved documents.
   - Inspect the data, then **propose and justify a chunking strategy**
     (chunk size, overlap, metadata fields) before coding.
   - Emit all chunks to a **human-readable `.txt` file** for inspection.
   - Embed with `sentence-transformers/all-MiniLM-L6-v2` (384-dim).
   - Store in **ChromaDB persisted to disk** so ingestion runs once, not per restart.

2. **Retrieval + answer generation**
   - Embed the user question with the *same* MiniLM model.
   - Retrieve top-k chunks; pass them to a Groq-hosted LLM.
   - Produce a ≤3 sentence, facts-only answer with one citation link and a
     "Last updated from sources:" line.

3. **Refusal behaviour**
   - Detect opinionated or portfolio questions ("Should I buy/sell this?",
     "Which is better for me?", "How should I allocate?").
   - Reply politely with a facts-only message plus a relevant **educational** link.

4. **Tiny UI**
   - Welcome line.
   - 3 example questions.
   - Visible note: **"Facts-only. No investment advice."**
   - Renders the answer, the citation link, and the source-updated date.

5. **PII safety rail**
   - Detect and reject PAN, Aadhaar, account numbers, OTPs, email addresses, and
     phone numbers in user input. Do not store them.

6. **Ops & deliverables**
   - `.env` for the Groq key; `.env` git-ignored, never committed.
   - README: setup steps, scope (AMC + schemes), known limits.
   - Sample Q&A file: 5–10 queries with answers and links.
   - Disclaimer snippet (facts-only, no advice).
   - Live prototype link on Render, or a ≤3-min demo video if hosting fails.

---

## 5. Out-of-Scope Features

- Investment advice, buy/sell/hold calls, timing calls, asset allocation,
  goal planning, or portfolio analysis.
- Any computation, ranking, or comparison of **returns, NAV history, CAGR, or
  performance**. If asked, link to the official factsheet instead.
- SIP calculators, lumpsum projections, or goal-scorer tools.
- Holdings tracking, statements parsing, or any login-gated / personalised data.
- Multi-AMC or multi-plan (Direct vs Regular) coverage beyond the scoped set.
- Voice, image, or multilingual input; document upload by end users.
- Account creation, user profiles, chat history persistence, or analytics.
- Realtime NAV, live market data, or any scheduled refresh beyond a manual
  re-ingestion command.
- Mobile apps; anything beyond a single-page local/Render web UI.

---

## 6. Example User Questions

**Fact questions the bot must answer (in scope):**

1. What is the expense ratio of HDFC Large Cap Fund – Direct Growth?
2. Is there an exit load on HDFC Small Cap Fund – Direct Growth, and how is it calculated?
3. What is the minimum SIP amount for HDFC Equity (Flexi Cap) Fund?
4. What is the lock-in period for HDFC ELSS Tax Saver Fund, and when can I redeem?
5. What is the riskometer level and benchmark of HDFC Balanced Advantage Fund?
6. What is the expense ratio and minimum investment for HDFC ELSS Tax Saver Fund?
7. How do I download a capital-gains statement from the AMC?
8. What documents do I need for a tax statement / how is TDS deducted on ELSS redemptions?
9. What is the AUM and category of HDFC Large Cap Fund? *(answer with the dated AUM fact from the factsheet + link; no performance/return claim, and no realtime NAV)*
10. What is the exit load on HDFC Balanced Advantage Fund after the stated period?

**Opinion / advice questions the bot must refuse (with an educational link):**

1. Should I buy HDFC ELSS Tax Saver Fund?
2. Which is better for me — HDFC Large Cap or HDFC Small Cap?
3. Is HDFC Balanced Advantage Fund a good choice for my retirement?
4. How should I split my portfolio between these five schemes?
5. When is a good time to enter a small cap fund?

**PII / out-of-policy inputs the bot must reject without storing:**

1. My PAN is ABCDE1234F — can you tell me the ELSS tax benefits?
2. Update my registered mobile number 9876543210.
3. My account number is 00123456789, where are my statements?

---

## 7. Success Criteria

**Functional (all must pass for the demo):**

| # | Criterion | Target |
|---|---|---|
| F1 | Answer accuracy on the fact set above | ≥ 90% of answers factually match the cited source |
| F2 | Citations | 100% of factual answers carry exactly one working official link |
| F3 | Answer length | 100% of answers ≤ 3 sentences |
| F4 | Refusal correctness | 100% of the 5 opinion questions refused, with an educational link |
| F5 | PII handling | 100% of the 3 PII probes rejected; nothing stored |
| F6 | Freshness line | "Last updated from sources:" present on every answer |
| F7 | No performance claims | 0 computed or compared return figures in any answer |
| F8 | Grounding | Answers only use retrieved chunks; no unsourced claims |

**Non-functional:**

| # | Criterion | Target |
|---|---|---|
| N1 | Latency | First token < 3 s; complete answer < 10 s on a laptop |
| N2 | Cold start | Ingestion is a one-time step; app restart does not re-ingest |
| N3 | Local run | `pip install` + two commands runs the app fully offline except the Groq call |
| N4 | Inspectability | Chunks dump to a readable `.txt`; chunking rationale documented |
| N5 | Secrets safety | No API key in the repo; `.env` git-ignored |
| N6 | Deployability | Deployed to Render free tier with a public URL, or a ≤3-min video as fallback |
| N7 | UX floor | Welcome line + 3 example questions + "Facts-only. No investment advice." visible |

**Demo-grade (judge-facing):**

- A reviewer with no prior context can read the README, install, and ask one
  question in under 10 minutes.
- The full RAG pipeline is inspectable end to end (chunks file → Chroma → retrieval).

---

## 8. Constraints

**Tooling / cost**

- **Free tier only.** No paid APIs. Groq free tier for the LLM; MiniLM and
  ChromaDB run locally at no cost. No hosted vector DB, no paid embedding service.
- Only one external paid-class dependency is forbidden; the Groq key is the sole secret.

**Runtime**

- **Runs locally** on a standard laptop, offline except for the Groq API call.
- Python-based, reproducible setup, pinned-ish dependencies.
- Ingestion and query are separable commands; ChromaDB persists to disk.

**Deployment**

- **Deployable to Render** (free web service) with a public URL; if Render cannot
  host it (e.g. model download size / cold start), a ≤3-min demo video is the
  accepted fallback.
- The deployed app must build the Chroma store on first start or ship a way to
  regenerate it, since ephemeral filesystems reset.

**Content & policy**

- **Public official sources only.** No screenshots of the app back-end; no
  third-party blogs as sources.
- **No PII**: never accept or store PAN, Aadhaar, account numbers, OTPs, emails,
  or phone numbers.
- **No performance claims**: do not compute or compare returns; link to the
  official factsheet when asked.
- **Clarity & transparency**: answers ≤ 3 sentences, always cite, always show
  "Last updated from sources:".

**Stack (fixed, from the brief)**

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local, no API key, 384-dim),
  used for both chunks and questions.
- Vector DB: ChromaDB, persisted to disk.
- LLM: Groq, key in `.env`, never committed.

---

## 9. Deliverables Checklist

- [ ] Working prototype link (Render) or ≤3-min demo video
- [ ] Source list (CSV/MD) of the URLs used
- [ ] README: setup steps, scope (AMC + schemes), known limits
- [ ] Sample Q&A file: 5–10 queries + answers + links
- [ ] Disclaimer snippet used in the UI
- [ ] `docs/PRD.md` (this document)
- [ ] `docs/architecture.md` + `docs/implementation.md`
- [ ] Chunking strategy note + inspectable `chunks.txt`
