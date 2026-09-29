"""Scoped corpus registry — the single source of truth for scope.

Scope (PRD section 3): ONE AMC (HDFC Asset Management) and FIVE schemes, all
Direct - Growth plans.

Source tiers — READ THIS BEFORE PHASE 3
---------------------------------------
The brief hands us five `groww.in` scheme URLs but ALSO requires sources be
official AMC/SEBI/AMFI pages with no third-party blogs. Those conflict, and
Phase 2 measured the options:

  * `hdfcmf.com`          -> PARKED DOMAIN, "this domain may be for sale"
  * `hdfcassetmanagement.com` -> DNS does not resolve
  * `groww.in`            -> HTTP 200, ~17.7k chars of extractable text

So the only reachable per-scheme source is the aggregator. We ingest it and
record it honestly as source_tier="aggregator" rather than pretending it is
official. Every chunk and every citation carries that tier, so swapping in real
AMC pages later is a one-line change here and nothing downstream needs to move.

To promote a source to tier "official", replace its `url` with the real AMC /
SEBI / AMFI page and set its tier in SOURCE_TIER_BY_SCHEME.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --- Source tiers -----------------------------------------------------------

TIER_OFFICIAL = "official"
TIER_AGGREGATOR = "aggregator"


# --- Schemes in scope -------------------------------------------------------

LARGE_CAP = "HDFC Large Cap Fund - Direct Growth"
FLEXI_CAP = "HDFC Equity (Flexi Cap) Fund - Direct Growth"
ELSS = "HDFC ELSS Tax Saver Fund - Direct Growth"
SMALL_CAP = "HDFC Small Cap Fund - Direct Growth"
BALANCED_ADVANTAGE = "HDFC Balanced Advantage Fund - Direct Growth"


# --- Document types we expect to ingest -------------------------------------

DOC_TYPE_SCHEME_PAGE = "scheme_page"
DOC_TYPE_FACTSHEET = "factsheet"
DOC_TYPE_KIM = "kim"
DOC_TYPE_SID = "sid"
DOC_TYPE_FAQ = "scheme_faq"
DOC_TYPE_FEES = "fee_page"
DOC_TYPE_RISKOMETER = "riskometer"
DOC_TYPE_STATEMENT_GUIDE = "statement_guide"
DOC_TYPE_EDUCATION = "investor_education"

DOC_TYPES = (
    DOC_TYPE_SCHEME_PAGE,
    DOC_TYPE_FACTSHEET,
    DOC_TYPE_KIM,
    DOC_TYPE_SID,
    DOC_TYPE_FAQ,
    DOC_TYPE_FEES,
    DOC_TYPE_RISKOMETER,
    DOC_TYPE_STATEMENT_GUIDE,
)


@dataclass(frozen=True)
class SourceDoc:
    """One ingestible official document."""

    scheme: str
    doc_type: str
    url: str | None = None
    discovery_url: str | None = None
    description: str = ""

    @property
    def is_resolved(self) -> bool:
        return bool(self.url)


@dataclass(frozen=True)
class Scheme:
    """One mutual fund scheme in scope."""

    name: str
    category: str
    discovery_url: str
    docs: list[SourceDoc] = field(default_factory=list)

    @property
    def short_name(self) -> str:
        """Compact label used in metadata and citations."""
        return self.name.replace(" - Direct Growth", "")


def _scheme(name: str, category: str, url: str) -> Scheme:
    return Scheme(
        name=name,
        category=category,
        discovery_url=url,
        docs=[
            SourceDoc(
                scheme=name,
                doc_type=DOC_TYPE_SCHEME_PAGE,
                url=url,
                discovery_url=url,
                description="Scheme page: fees, exit load, min SIP, benchmark, riskometer, AUM",
            )
        ],
    )


# The five schemes in scope. Phase 2 confirmed these are the only reachable
# per-scheme sources, so they are registered as tier "aggregator" (see
# SOURCE_TIER_BY_SCHEME below and docs/sources.md).
SCHEMES: list[Scheme] = [
    _scheme(
        LARGE_CAP,
        "large_cap",
        "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
    ),
    _scheme(
        FLEXI_CAP,
        "flexi_cap",
        "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
    ),
    _scheme(
        ELSS,
        "elss",
        "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
    ),
    _scheme(
        SMALL_CAP,
        "small_cap",
        "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
    ),
    _scheme(
        BALANCED_ADVANTAGE,
        "hybrid_balanced_advantage",
        "https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth",
    ),
]

CATEGORY_BY_SCHEME: dict[str, str] = {s.name: s.category for s in SCHEMES}
SOURCE_TIER_BY_SCHEME: dict[str, str] = {s.name: TIER_AGGREGATOR for s in SCHEMES}

# Flat view of everything that should be ingested.
SOURCES: list[SourceDoc] = [doc for scheme in SCHEMES for doc in scheme.docs]

# Cross-cutting documents for the "how do I download a statement" class of
# question (PRD section 6, question 7). Unresolved: no official statement
# guide was reachable in Phase 2. The bot will deflect these questions.
GENERAL_SOURCES: list[SourceDoc] = [
    SourceDoc(
        scheme="ALL",
        doc_type=DOC_TYPE_STATEMENT_GUIDE,
        url=None,
        description="Official AMC/SEBI/AMFI guide to downloading statements and tax documents",
    ),
]

# Educational links used by the advice-refusal path (Phase 4). These are NOT
# part of the retrieved corpus - they are shown alongside a refusal so the user
# has somewhere sensible to go.
#
# These three roots were verified reachable during Phase 4. They are deliberately
# roots rather than deep links: specific investor-education pages change often,
# and a guessed deep link would 404 in front of a judge. Replace them with
# pinned deep links to SEBI Investor Charter / AMFI investor education before
# the final submission.
EDUCATIONAL_LINKS: dict[str, str | None] = {
    "sebi_investor_education": "https://www.sebi.gov.in/",
    "amfi_investor_education": "https://www.amfiindia.com/",
    "mutual_fund_basics": "https://groww.in/mutual-funds",
}


def validate_registry() -> list[str]:
    """Return a list of problems. Empty list means the registry is ready."""
    problems: list[str] = []

    if len(SCHEMES) != 5:
        problems.append(f"Expected 5 schemes, found {len(SCHEMES)}.")

    names = [s.name for s in SCHEMES]
    if len(set(names)) != len(names):
        problems.append("Duplicate scheme names in SCHEMES.")

    for scheme in SCHEMES:
        if not scheme.docs:
            problems.append(f"'{scheme.short_name}' has no documents registered.")
        for doc in scheme.docs:
            if not doc.is_resolved:
                problems.append(
                    f"Missing URL: {scheme.short_name} / {doc.doc_type}"
                )
            tier = SOURCE_TIER_BY_SCHEME.get(scheme.name)
            if tier != TIER_OFFICIAL:
                problems.append(
                    f"NON-OFFICIAL SOURCE: {scheme.short_name} / {doc.doc_type} "
                    f"is tier '{tier}'. Brief requires official AMC/SEBI/AMFI "
                    f"pages (PRD F2). Replace the URL before final submission."
                )

    for doc in GENERAL_SOURCES:
        if not doc.is_resolved:
            problems.append(f"Missing official URL: general / {doc.doc_type}")

    return problems


def resolve_sources() -> None:
    """Print the current state of the registry. Does not mutate anything."""
    print(f"Schemes in scope: {len(SCHEMES)}")
    for scheme in SCHEMES:
        print(f"\n  {scheme.short_name}  [{scheme.category}]")
        print(f"    discovery (not ingested): {scheme.discovery_url}")
        if not scheme.docs:
            print("    documents registered: NONE - add SourceDoc entries in Phase 2")
        for doc in scheme.docs:
            status = doc.url if doc.is_resolved else "UNRESOLVED"
            print(f"    - {doc.doc_type:<18} {status}")

    print(f"\nGeneral documents: {len(GENERAL_SOURCES)}")
    for doc in GENERAL_SOURCES:
        status = doc.url if doc.is_resolved else "UNRESOLVED"
        print(f"  - {doc.doc_type:<18} {status}")

    print(f"\nTotal ingestible documents: {len(SOURCES) + len(GENERAL_SOURCES)}")
    print(f"Resolved: {sum(1 for d in SOURCES + GENERAL_SOURCES if d.is_resolved)}")

    problems = validate_registry()
    print("\nValidation: " + ("OK" if not problems else f"{len(problems)} issue(s)"))
    for problem in problems:
        print(f"  - {problem}")


if __name__ == "__main__":
    resolve_sources()
