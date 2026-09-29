# Source List — Mutual Fund FAQ Assistant

Phase 2 deliverable: the exact URLs ingested, as required by the milestone brief
("Source list (CSV/MD) of the 5 URLs you used").

**Read the status column.** These are *not* official sources yet.

---

## Ingested sources (5)

| # | Scheme | Category | URL | Tier |
|---|---|---|---|---|
| 1 | HDFC Large Cap Fund – Direct Growth | large_cap | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth | **aggregator** |
| 2 | HDFC Equity (Flexi Cap) Fund – Direct Growth | flexi_cap | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth | **aggregator** |
| 3 | HDFC ELSS Tax Saver Fund – Direct Growth | elss | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth | **aggregator** |
| 4 | HDFC Small Cap Fund – Direct Growth | small_cap | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth | **aggregator** |
| 5 | HDFC Balanced Advantage Fund – Direct Growth | hybrid_balanced_advantage | https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth | **aggregator** |

Fetched: 2026-09-29. Snapshots saved to `data/raw/`.

> **Correction to the brief.** The brief's ELSS URL
> `…/hdfc-elss-tax-saver-fund-direct-growth` returns **HTTP 404** (the slug needs
> `-plan-`). The working URL is row 3 above.

## Why these are tier `aggregator`, not official

The brief requires official AMC/SEBI/AMFI pages and forbids third-party blogs,
but also supplies groww.in URLs. Phase 2 probed the official routes:

| Candidate | Result |
|---|---|
| `hdfcmf.com` | HTTP 200 but **parked domain** — "This domain may be for sale" |
| `hdfcassetmanagement.com` | **DNS does not resolve** |
| `amfiindia.com` | HTTP 200, reachable — no per-scheme fee/exit-load pages found |
| `sebi.gov.in` | HTTP 200, reachable — no per-scheme pages |
| `groww.in` (brief's URLs) | HTTP 200, ~17.7k chars of extractable text — **only viable per-scheme source** |

We ingest what works and record the tier honestly rather than labelling a
third-party site as official. `validate_registry()` currently returns 5
`NON-OFFICIAL SOURCE` errors, so the gap cannot silently reach a submission.

## To promote to `official`

For each row: replace `url` in `sources.py` with the real AMC factsheet / KIM /
SID / fee page, and set `SOURCE_TIER_BY_SCHEME[scheme] = TIER_OFFICIAL`.
Nothing in the chunker, store, or query path needs to change.

## Still needed (corpus gaps)

| Needed for | Document type | Status |
|---|---|---|
| ELSS 3-year lock-in, 80C | SID / KIM / tax section | not ingested |
| SEBI riskometer level | riskometer note | not ingested |
| Capital-gains statement download | AMC/SEBI statement guide | not registered |
| Scheme-specific expense ratio (numeric) | factsheet | not ingested |
