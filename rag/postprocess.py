"""Phase 5 - Post-processor.

The last line of defence. The prompt asks the model for a short, cited,
facts-only answer; this module guarantees it.

The brief's non-negotiables are enforced here in code, not trusted to the
model:

  * at most 3 sentences            (PRD F3)
  * exactly one citation link      (PRD F2)
  * "Last updated from sources:"   (PRD F6)
  * no returns/performance figures (PRD F7)
  * a refusal is never advice      (PRD F4)

Guard verdicts short-circuit the LLM entirely, so a refused question costs no
API call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.guard import Verdict
from rag.retrieval import ScoredChunk

FRESHNESS_PREFIX = "Last updated from sources:"

_SOURCE_LINE_RE = re.compile(r"^\s*source\s*:\s*(\S+)\s*$", re.IGNORECASE | re.MULTILINE)
_URL_RE = re.compile(r"https?://\S+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Phrases that signal a performance claim. Matched case-insensitively.
#
# These must stay narrow. The corpus legitimately contains percentages for
# exit load (1%), expense ratio (1.5%), AUM, and tax slabs, and flagging those
# would make the detector useless. So a bare number never triggers: a number
# only counts when it sits next to return/performance vocabulary, or when it is
# a comparative claim in its own right.
_PERFORMANCE_PATTERNS = (
    # Number + return vocabulary in either order: "22% annually", "CAGR is 18%",
    # "returned 22%", "18% return".
    r"\b\d+(\.\d+)?\s*%\s*(per\s+(annum|year)|p\.?a\.?|annually|"
    r"annualised|annualized|cagr|return(s|ed)?|growth|gain(s|ed)?)\b",
    r"\b(cagr|annualised|annualized|average\s+return)\b",
    r"\b\d+(\.\d+)?\s*%\s*(cagr|return|growth)\b",
    # Allow a couple of filler adjectives between the verb and the number:
    # "returned a healthy 12%", "grew nearly 3x".
    r"\b(return(s|ed|ing)?|gained|gain(s|ed)?|rose|grew|grown|lost|"
    r"appreciated|multiplied|doubled|up\s*\d*|down\s*\d*)\s+"
    r"([a-z]+\s+){0,3}?\d+(\.\d+)?\s*(?:%|x)(?!\w)",
    r"\b\d+(\.\d+)?\s*x\b",
    # Comparative claims: performance asserted without a number.
    r"\b(outperform(s|ed|ing)?|underperform(s|ed|ing)?|beat(s|ing)?|"
    r"top(s|ped|ping)?|trail(s|ed|ing)?|lag(s|ged|ging)?)\b",
    r"\b(best|top|leading|strong|weak|poor|consistent|stable)\s+"
    r"(perform(er|ance|ers)?|fund|scheme)\b",
    r"\brisk[\s-]?adjusted\s+return",
    r"\bhas\s+(grown|risen|returned|gained|multiplied|doubled)\b",
    r"\b(past|historical|since\s+inception)\s+returns?\b",
)
_PERFORMANCE_RE = re.compile("|".join(_PERFORMANCE_PATTERNS), re.IGNORECASE)


@dataclass
class Answer:
    """A fully-enforced response, ready for the UI."""

    text: str
    source_url: str = ""
    freshness: str = ""
    refused: bool = False
    refusal_reason: str = ""
    link: str = ""
    chunks: list[ScoredChunk] = field(default_factory=list)
    latency_s: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def is_cited(self) -> bool:
        return bool(self.source_url)


def split_sentences(text: str) -> list[str]:
    """Split into sentences, keeping terminal punctuation attached."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(cleaned) if s.strip()]


def extract_source_url(raw: str, chunks: list[ScoredChunk]) -> tuple[str, list[str]]:
    """Pull the citation out of the model output.

    Returns (url, warnings). Prefers a URL the model actually wrote, but only
    accepts it if it belongs to a retrieved chunk. An invented URL is dropped
    and the best in-corpus URL is used instead, because a citation the user
    cannot open is worse than no citation.
    """
    warnings: list[str] = []
    allowed = [c.source_url for c in chunks if c.source_url]

    match = _SOURCE_LINE_RE.search(raw or "")
    claimed = match.group(1).strip().rstrip(".,);") if match else None

    if not claimed:
        found = _URL_RE.findall(raw or "")
        claimed = found[0].rstrip(".,);") if found else None

    if claimed and claimed in allowed:
        return claimed, warnings

    if claimed:
        warnings.append(
            f"model cited a URL not in the retrieved context ({claimed}); "
            f"replaced with a source from the corpus"
        )

    if not allowed:
        return "", warnings

    best = min(chunks, key=lambda c: c.distance) if chunks else None
    return (best.source_url if best else allowed[0]), warnings


def strip_source_line(raw: str) -> str:
    """Remove any Source: line so it can be re-added canonically."""
    without = _SOURCE_LINE_RE.sub("", raw or "")
    return re.sub(r"\n{2,}", "\n", without).strip()


def enforce_sentence_limit(text: str, limit: int = 3) -> tuple[str, bool]:
    """Truncate to at most `limit` sentences. Returns (text, was_truncated)."""
    sentences = split_sentences(text)
    if len(sentences) <= limit:
        return " ".join(sentences), False
    return " ".join(sentences[:limit]), True


def contains_performance_claim(text: str) -> bool:
    return bool(_PERFORMANCE_RE.search(text or ""))


# Phrases that mark a decline to answer. A decline must not be cited: the
# retrieved chunk that happened to rank first did not support the answer, so
# linking to it points the user at an unrelated page.
#
# Note the shape. An earlier version enumerated literal phrasings
# ("not covered", "not provided") and broke the moment the model worded it as
# "is not among the schemes covered". Matching an enumerated list against
# free-form LLM output is guaranteed to miss phrasings. So this matches the
# *structure* of a decline instead: a negation, then coverage/availability
# vocabulary, within a short window.
_DECLINE_RE = re.compile(
    r"\b(i\s+(do\s+not|don't|dont|cannot|can't)\s+(know|have|find|answer)\b"
    r"|\b(no|not)\s+(such\s+)?(information|data|details?|record|mention)\b"
    r"|\b(cannot|can't)\s+(be\s+)?(answer|determine|confirm|verify)\b"
    # negation ... coverage vocabulary, within ~60 chars
    r"|\b(not|isn't|is\s+not|are\s+not|aren't|does\s+not|doesn't|don't)\b"
    r"[^.?!]{0,60}?\b(covered|provide[ds]?|available|mentioned|stated|listed|"
    r"included|present|found|known|disclosed|specified|among|part\s+of)\b"
    # "X is not among the schemes ..." / "not one of the schemes"
    r"|\bis\s+not\s+(one\s+of|among|part\s+of)\b"
    r"|\bnot\s+(in\s+this\s+(demo|corpus)|one\s+of\s+the)\b)",
    re.IGNORECASE,
)


def is_declined(text: str) -> bool:
    """True when the model declined rather than answered."""
    return bool(_DECLINE_RE.search(text or ""))


def refusal_answer(verdict: Verdict, chunks: list[ScoredChunk] | None = None) -> Answer:
    """Build an Answer from a guard verdict, without calling the LLM."""
    return Answer(
        text=verdict.message,
        source_url="",
        freshness="",
        refused=True,
        refusal_reason=verdict.rule or verdict.reason,
        link=verdict.link,
        chunks=chunks or [],
    )


def finalise(raw: str, chunks: list[ScoredChunk], freshness: str,
             latency_s: float = 0.0) -> Answer:
    """Turn raw model output into a compliant Answer."""
    warnings: list[str] = []

    url, url_warnings = extract_source_url(raw, chunks)
    warnings.extend(url_warnings)

    body = strip_source_line(raw)

    if contains_performance_claim(body):
        warnings.append(
            "performance language detected in the model's answer - review before "
            "showing (PRD F7)"
        )

    body, truncated = enforce_sentence_limit(body, limit=3)
    if truncated:
        warnings.append("answer exceeded 3 sentences and was truncated (PRD F3)")

    text = body.rstrip()
    if not text:
        text = "I don't know. That isn't covered by the official sources."
        warnings.append("model returned no usable text")

    # A decline gets no citation. See _DECLINE_RE for why.
    declined = is_declined(text)
    if declined and url:
        warnings.append(
            "model declined to answer; suppressed the citation, since no "
            "retrieved chunk supports a non-answer"
        )
        url = ""

    return Answer(
        text=text,
        source_url=url,
        freshness="" if declined else freshness,
        refused=declined,
        refusal_reason="declined: not covered by context" if declined else "",
        chunks=chunks,
        latency_s=latency_s,
        warnings=warnings,
    )


def render(answer: Answer) -> str:
    """Plain-text rendering, used by the CLI and the tests.

    A decline renders without a Source or freshness line, matching the rule
    enforced in finalise().
    """
    lines = [answer.text]
    if answer.source_url:
        lines.append(f"Source: {answer.source_url}")
    if answer.freshness:
        lines.append(f"{FRESHNESS_PREFIX} {answer.freshness}")
    if answer.link:
        lines.append(f"Learn more: {answer.link}")
    return "\n".join(lines)
