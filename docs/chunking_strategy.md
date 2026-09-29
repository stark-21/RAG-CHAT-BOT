# Chunking Strategy — Mutual Fund FAQ Assistant

Phase 2 deliverable. The brief requires the strategy to be chosen **after
inspecting the real data**, with chunk size, overlap, and metadata specified and
justified. This document is that justification.

Implementation: `rag/chunking.py`. Inspectable output: `data/chunks/chunks.txt`.

---

## 1. What the data actually looks like

Five HDFC scheme pages, 80,235 characters of cleaned text. Inspecting
`data/raw/*.txt` showed three things that drove every decision below.

**(a) The useful content is organised as LABEL → VALUE pairs.**

```
Exit load
Exit load of 1% if redeemed within 1 year
Fund benchmark
NIFTY 100 Total Return Index
Total AUM
₹9,86,236.84 Cr
```

**(b) A large fraction of each page is site chrome, not content.** The page is
a JavaScript app; its navigation renders as plain text nodes that survive
`<script>`/`<nav>` stripping, and the entire footer is a link farm of futures,
option chains, and stock tickers.

**(c) Much of the remainder is data tables** — holdings (company / sector /
asset class / weight), sector breakdowns, NAV history.

---

## 2. Strategy: label-anchored, context-windowed chunks

Split on **detected label anchors** rather than at fixed character positions.
Each chunk keeps its label glued to the values that follow it.

### Why not fixed-size splitting

Fixed-size chunking breaks this corpus in three ways:

1. **It severs label from value.** A 450-character window reliably cuts between
   `Exit load` and `Exit load of 1% if redeemed within 1 year`, leaving a chunk
   whose text is `1% if redeemed within 1 year` — unanswerable, and the citation
   no longer matches the question asked.
2. **It mixes unrelated facts**, so a retrieval for "exit load" returns a chunk
   that also contains holdings weights and tax commentary. The useful sentence
   is buried in a low-similarity chunk.
3. **It wastes the corpus on noise.** Measured on the first run, 70% of chunks
   (201 of 285) were holdings and AUM tables, while the 6 fact types the PRD
   actually asks about shared the remaining 30%.

### Why label-anchored works here

Every answerable fact in this corpus is introduced by a short label line. Splitting
at those labels means a chunk is *by construction* one fact (or one tight group of
related facts) from one scheme, with the label attached. The chunk boundary and
the semantic boundary are the same boundary.

---

## 3. Parameters

| Parameter | Value | Why |
|---|---|---|
| Chunk size | **450 characters** | Longest answerable unit observed is a label plus 2–3 values, ~200–400 chars. 450 captures a full label group without pulling in the next unrelated label. Sized in characters, not tokens, because the budget must be enforced identically at ingest and display. |
| Overlap | **60 characters** | Only the tail of the previous chunk is carried forward. Enough to keep a value whose label sits on the previous window; small enough that overlap text does not dominate the embedding. |
| Split unit | **Whole lines** | Lines are never split mid-sentence, so a chunk is always readable and always quotable. |
| Minimum chunk | 25 characters | Below this a chunk is a nav fragment, not a fact. |

---

## 4. Metadata carried by every chunk

| Field | Purpose |
|---|---|
| `scheme` | HDFC Large Cap / Flexi Cap / ELSS / Small Cap / Balanced Advantage |
| `scheme_short` | Compact label for display |
| `category` | large_cap, flexi_cap, elss, small_cap, hybrid_balanced_advantage |
| `doc_type` | scheme_page (later: factsheet, KIM, SID, FAQ, fee page) |
| `section` | The label that anchored the chunk, e.g. "Exit load" |
| `source_url` | **The citation link shown to the user** |
| `source_tier` | `official` or `aggregator` — see §7 |
| `fetched_at` | Powers the "Last updated from sources:" line |
| `chunk_idx` | Stable ordering, for reproducibility |
| `char_count` | Audit trail for the size budget |

### The embed-text header

The text that gets embedded is **not** the chunk text. It is prefixed with:

```
[<scheme_short> | <section>]
<chunk text>
```

This is load-bearing. All five schemes have an "Exit load" section, so the bare
sentence `Exit load of 1% if redeemed within 1 year` embeds to nearly the same
vector in all five documents. Without the header, a question about HDFC Small
Cap can retrieve the Large Cap chunk at equal rank. With it, the scheme identity
is part of the vector, and Phase 5 can additionally filter on `scheme`
metadata. Test: `test_embed_text_carries_scheme_header`.

---

## 5. Noise control (three measured passes)

Cleaning is deliberately split between `rag/loader.py` (page level) and
`rag/chunking.py` (segment level).

| Pass | Rule | Measured effect |
|---|---|---|
| 1 | Strip chrome tags (`script`, `nav`, `footer`, …) | 453k → 17.7k chars |
| 2 | Boilerplate line blocklist | removes nav duplicates |
| 3 | Footer truncation at the breadcrumb sentinel `>` | **285 → 165 chunks** |
| 4 | Drop table sections (Assets, Holdings, Fund Size, …) | 165 → 85 chunks |
| 5 | Drop table cells (bare numbers, %, ₹, dates, `Equity`) | un-fragments label→value pairs |

Two of these were found by inspection, not by design, and both are worth
recording as traps:

- **The footer is not inside a `<footer>` tag.** It renders as plain text, so tag
  stripping cannot remove it. It was 80 of 165 chunks. Fixed by truncating at
  the first footer breadcrumb. A first attempt used weaker sentinels
  (`Blog`, `Products`, `Pricing`); those also appear in the *top* nav and
  truncated 80% of the corpus. Only `>` and `All rights reserved` are strong
  enough.
- **Real fact labels were being deleted as boilerplate.** `Expense ratio`,
  `Riskometer`, `Min. for SIP`, `High Risk` were in the nav blocklist. The
  "Expense ratio" section disappeared from the corpus entirely. Any label in
  that list that is also a fact must not be there.

---

## 6. Result

| Metric | Value |
|---|---|
| Cleaned corpus | 80,235 chars across 5 documents |
| Chunks | **90** |
| Chunk size | min 84, max 494, mean 316 chars |
| Sections captured | 12 distinct |
| Distribution | ~17–22 chunks per scheme |

Captured sections: `Overview` (20), `Tax implication` (11), `Exit load` (10),
`About` (10), `Minimum investments` (5), `Min. for SIP` (5), `Expense ratio` (5),
`Tax` (5), `Investment Objective` (5), `Fund benchmark` (5), `Total AUM` (5),
`Exit Load` (4).

Answerable today: exit load, expense ratio definition, minimum SIP, benchmark,
investment objective, AUM, tax implication, scheme description.

---

## 7. Known gaps in the current corpus

Measured against the PRD §6 fact questions, and stated plainly rather than
papered over:

| PRD question | Status | Cause |
|---|---|---|
| Exit load | Answerable | `Exit load` sections present |
| Expense ratio | Definition only | Page gives the *definition*, not the scheme's numeric ratio |
| Minimum SIP | Answerable | `Min. for SIP`, `Minimum investments` |
| Benchmark | Answerable | `Fund benchmark` |
| AUM | Answerable | `Total AUM` |
| **ELSS lock-in** | **Missing** | 0 occurrences of "lock" / "80C" in the ELSS page text |
| **Riskometer** | **Missing** | 0 occurrences of "riskometer"; risk is only implied as "Very High Risk" in prose |
| **Capital-gains statement download** | **Missing** | No statement-guide source registered |

These are **corpus gaps, not chunking bugs** — the text is not in the source
pages. Closing them requires adding sources (SID/KIM/factsheet PDFs, a SEBI or
AMFI statement guide), which is a `sources.py` change, not a chunker change.

### Source tier

The brief's groww.in URLs are the only reachable per-scheme sources:
`hdfcmf.com` is a parked domain ("this domain may be for sale") and
`hdfcassetmanagement.com` does not resolve. Every chunk therefore carries
`source_tier="aggregator"`, and `validate_registry()` fails the build while any
source is non-official.

**Citations from this corpus do not yet satisfy the "official sources only"
constraint.** Replacing a URL in `sources.py` with a real AMC/SEBI/AMFI page and
setting the tier is the only change needed — no chunker or downstream change.
