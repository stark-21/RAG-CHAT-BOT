"""Phase 4 - Guardrails.

Three gates, in order of cost. The first two need no retrieval and no LLM, so
they are instant and free.

  1. PII         reject outright, echo nothing
  2. Off-topic   reject questions with no mutual-fund subject
  3. Advice      reject buy/sell/allocate/personal-recommendation intent
  4. Grounding   (needs retrieval) "I don't know" when the corpus has no answer

Gates 1-3 run in `inspect()`. Gate 4 is `check_grounding()`, called by Phase 5
once chunks have been retrieved.

Every refusal is facts-only, carries an educational link, and never contains a
recommendation. The PII path additionally never echoes the detected value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import config
from sources import EDUCATIONAL_LINKS

# --- Verdicts ---------------------------------------------------------------

ACTION_ANSWER = "answer"
ACTION_REFUSE_ADVICE = "refuse_advice"
ACTION_REFUSE_PII = "refuse_pii"
ACTION_REFUSE_OFFTOPIC = "refuse_offtopic"
ACTION_REFUSE_DANGLING = "refuse_dangling"
ACTION_NO_CONTEXT = "no_context"


@dataclass(frozen=True)
class Verdict:
    """The guard's decision, with everything the UI needs to render it."""

    action: str
    reason: str = ""
    rule: str = ""
    message: str = ""
    link: str = ""

    @property
    def should_retrieve(self) -> bool:
        return self.action == ACTION_ANSWER

    @property
    def is_refusal(self) -> bool:
        return self.action != ACTION_ANSWER


# --- PII patterns -----------------------------------------------------------

_PII_PATTERNS = (
    ("pan", r"\b[A-Za-z]{5}[0-9]{4}[A-Za-z]\b"),
    ("aadhaar", r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b"),
    ("account_number", r"\b\d{9,18}\b"),
    ("otp", r"\b(otp|one time password|verification code)\b.{0,20}\b\d{4,6}\b"),
    ("email", r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
    ("phone", r"(?:\+?91[\s-]?)?\b[6-9]\d{9}\b"),
)

_PII_RES = tuple((name, re.compile(rx, re.IGNORECASE)) for name, rx in _PII_PATTERNS)

# --- Advice / personal-recommendation patterns ------------------------------
#
# Tuned against the PRD section 6 set. The risk here is over-blocking real
# questions: "good for tax saving" is a factual tax question, while
# "a good choice for my retirement" is a personal recommendation. These
# patterns must keep that distinction.

_ADVICE_PATTERNS = (
    r"\bshould\s+i\s+(buy|sell|invest|start|exit|hold|redeem|pick|choose|add)\b",
    r"\b(is|are)\s+it\s+(worth|a\s+good\s+(idea|time|buy))\b",
    r"\bwhich\s+(is|one|fund|scheme)?\s*(is\s+)?better\b",
    r"\bwhich\s+should\s+i\b",
    r"\bwhat\s+should\s+i\b",
    r"\bbest\s+(fund|scheme|option|amc|choice)\s+for\b",
    r"\brecommend\b",
    r"\bsuggest\b",
    r"\bworth\s+investing\b",
    r"\bgood\s+(choice|pick|option|fit)\s+for\s+(me|my)\b",
    r"\b(for\s+my|according\s+to\s+my)\s+(retirement|portfolio|goals?|horizon|profile)\b",
    r"\bhow\s+(should|do)\s+i\s+(allocate|split|divide|balance)\b",
    r"\bhow\s+much\s+(should|do)\s+i\s+invest\b",
    r"\bhow\s+should\s+i\s+(split|allocate|balance|plan)\b",
    r"\bwhen\s+(is|should|would)\s+(i|be)\s+(a\s+good\s+time|buy|sell|enter|start)\b",
    r"\bgood\s+time\s+to\s+(buy|sell|enter|start|invest)\b",
    r"\bsuitable\s+for\s+me\b",
    r"\bis\s+it\s+safe\s+to\s+invest\b",
    r"\bmy\s+(portfolio|allocation)\b",
)

_ADVICE_RE = re.compile("|".join(_ADVICE_PATTERNS), re.IGNORECASE)

# --- Mutual-fund vocabulary (for the off-topic gate) ------------------------

_MF_TERMS = (
    r"mutual\s*fund", r"\bfund\b", r"\bscheme\b", r"\bsip\b", r"\belss\b",
    r"\bnav\b", r"expense\s*ratio", r"exit\s*load", r"entry\s*load",
    r"aum\b", r"benchmark", r"riskometer", r"folio", r"\bamc\b", r"\bsid\b",
    r"\bkim\b", r"lumpsum", r"\bdividend\b", r"\btax\b", r"\btds\b",
    r"capital\s*gains?", r"lock[\s-]?in", r"redemption", r"invest",
    r"\bportfolio\b", r"hdfc", r"\bparag\b", r"\baxis\b", r"factsheet",
    r"risk[\s-]?ometer", r"\bhybrid\b", r"large\s*cap", r"small\s*cap",
    r"flexi\s*cap", r"equity\b", r"\bdebt\b", r"index\s*fund",
)

_MF_RE = re.compile("|".join(_MF_TERMS), re.IGNORECASE)


# --- Concept coverage --------------------------------------------------------
#
# (question pattern) -> (pattern that must appear in some retrieved chunk)
# Only concepts the PRD actually asks about. The point is to catch the case
# where the question is lexically similar to a chunk that does not contain the
# specific fact being asked for.

_CONCEPT_REQUIREMENTS = (
    ("lock-in",       r"lock[\s-]?in",            r"lock[\s-]?in|three\s+year|3\s+year"),
    ("80C tax",       r"\b80c\b|\b80\s*c\b",      r"\b80c\b|\b80\s*c\b"),
    ("riskometer",    r"riskometer|risk[\s-]?o?meter", r"riskometer|risk[\s-]?o?meter"),
    ("statement download", r"download|statement", r"statement|download"),
    ("exit load",     r"exit\s*load",             r"exit\s*load"),
    ("expense ratio", r"expense\s*ratio",         r"expense\s*ratio"),
    ("benchmark",     r"benchmark",               r"benchmark"),
    ("minimum SIP",   r"min\.?\s*(for\s*)?sip|minimum\s+sip", r"\bsip\b"),
    ("AUM",           r"\baum\b|fund\s*size",     r"\baum\b|fund\s*size"),
    ("NAV",           r"\bnav\b",                 r"\bnav\b"),
    ("tax treatment", r"\btax\b|\btds\b",         r"\btax\b|\btds\b"),
    ("dividend",      r"dividend",                r"dividend"),
    ("folio",         r"folio",                   r"folio"),
    ("fund manager",  r"fund\s*manager",          r"manager"),
    ("risk level",    r"risk\s*(level|rating|category)", r"risk"),
)

_CONCEPTS = tuple(
    (label, re.compile(q, re.IGNORECASE), re.compile(c, re.IGNORECASE))
    for label, q, c in _CONCEPT_REQUIREMENTS
)


def _mask_scheme_names(question: str) -> str:
    """Remove scheme names from the question before concept detection.

    Necessary because scheme names contain words that double as concepts: the
    ELSS scheme is literally called "HDFC ELSS **Tax** Saver Fund", so every
    question naming it would otherwise trip the "tax treatment" requirement and
    be refused even when the chunk does discuss tax.
    """
    from sources import SCHEMES

    masked = question
    for scheme in SCHEMES:
        for variant in (
            scheme.name,
            scheme.short_name,
            scheme.name.replace(" - Direct Growth", " Direct Plan Growth"),
        ):
            if variant:
                masked = re.sub(re.escape(variant), " ", masked, flags=re.IGNORECASE)
    return masked


def missing_concepts(question: str, retrieved_texts: list[str]) -> list[str]:
    """Concepts the question asks about that no retrieved chunk mentions.

    Returns human-readable labels, e.g. ["lock-in", "riskometer"]. An empty
    list means the retrieved context plausibly covers what was asked.
    """
    if not question or not retrieved_texts:
        return []

    ask_about = _mask_scheme_names(question)
    haystack = "\n".join(retrieved_texts).lower()
    missing: list[str] = []

    for label, question_re, content_re in _CONCEPTS:
        if question_re.search(ask_about) and not content_re.search(haystack):
            missing.append(label)

    return missing


# --- Detection primitives ---------------------------------------------------

def detect_pii(text: str) -> list[str]:
    """Return the names of PII types found. Never returns the values."""
    return [name for name, rx in _PII_RES if rx.search(text)]


def detect_advice(text: str) -> str | None:
    """Return the matched advice pattern, or None."""
    match = _ADVICE_RE.search(text)
    return match.group(0) if match else None


def is_mutual_fund_question(text: str) -> bool:
    """True when the question has mutual-fund subject matter."""
    return bool(_MF_RE.search(text))


# --- Messages ---------------------------------------------------------------

_ADVICE_MESSAGE = (
    "I only share published facts about these schemes, so I can't recommend one "
    "or suggest a course of action. For general investor education, see the link "
    "below."
)

_OFFTOPIC_MESSAGE = (
    "I can only answer questions about the mutual fund schemes in this demo's "
    "corpus. Ask me about fees, exit load, minimum SIP, lock-in, benchmark, "
    "riskometer, or AUM."
)

_PII_MESSAGE = (
    "I don't accept or store personal identifiers such as PAN, Aadhaar, account "
    "numbers, OTPs, emails, or phone numbers, so I can't process a message that "
    "contains one. Please re-send your question without any personal details."
)

_NO_CONTEXT_MESSAGE = (
    "I don't know. That isn't covered by the official sources in this corpus, so "
    "I won't guess. You can check the scheme's official page for the current "
    "details."
)

_DANGLING_MESSAGE = (
    "I need to know which scheme you mean before I can answer. Name the fund in "
    "your question, for example: \"What is the expense ratio of HDFC Small Cap "
    "Fund?\""
)


def names_corpus_scheme(question: str) -> bool:
    """True when the question names one of the schemes in the corpus."""
    from sources import SCHEMES

    lowered = (question or "").lower()
    for scheme in SCHEMES:
        candidates = (
            scheme.name,
            scheme.short_name,
            scheme.name.split(" - ")[0],
        )
        if any(c and c.lower() in lowered for c in candidates):
            return True
    return False


def has_dangling_reference(question: str) -> bool:
    """True when the question points at earlier context without naming it.

    "what about its fees?" is not off-topic, it is under-specified. Refusing it
    as off-topic tells the user the wrong thing about why they got no answer.

    The condition is "does not name a corpus scheme", NOT "is not a mutual-fund
    question". "and its benchmark?" contains "benchmark" yet is still
    under-specified, so testing for mutual-fund vocabulary let it through to a
    misleading "not covered by the official sources".
    """
    from rag.prompt import needs_resolution

    text = (question or "").strip()
    if not text:
        return False
    return needs_resolution(text) and not names_corpus_scheme(text)


def inspect_dangling(question: str) -> Verdict | None:
    """Refusal for an under-specified follow-up with no conversation context."""
    if not has_dangling_reference(question):
        return None
    return Verdict(
        action=ACTION_REFUSE_DANGLING,
        reason="question references earlier context that is not available",
        rule="dangling_reference",
        message=_DANGLING_MESSAGE,
    )


def _education_link() -> str:
    for key in ("sebi_investor_education", "amfi_investor_education",
                "mutual_fund_basics"):
        url = EDUCATIONAL_LINKS.get(key)
        if url:
            return url
    return ""


# --- Gate 1-3 ---------------------------------------------------------------

def inspect_pii(text: str) -> Verdict | None:
    """PII gate only. Returns a refusal Verdict, or None when clean.

    Used by the pipeline as a pre-pass on the raw question, so a follow-up
    rewrite can never pull an identifier into the pipeline. PII must be judged
    on what the user actually typed, not on a model-rewritten paraphrase of it.
    """
    pii_types = detect_pii((text or "").strip())
    if not pii_types:
        return None
    return Verdict(
        action=ACTION_REFUSE_PII,
        reason=f"personal identifier detected: {', '.join(pii_types)}",
        rule="pii",
        # Never echo the detected value.
        message=_PII_MESSAGE,
    )


def inspect(question: str, has_context: bool = True) -> Verdict:
    """Run the no-retrieval gates. Returns a Verdict.

    Order is deliberate: PII first, because nothing else should touch a
    message containing an identifier.

    `has_context` tells the gate whether earlier conversation is available to
    resolve a reference against. When it is False, a question that points at
    earlier context is refused as under-specified. That check sits AFTER advice,
    so "Should I buy it?" still gets the advice refusal and its education link
    rather than being told to name a fund.
    """
    text = (question or "").strip()
    if not text:
        return Verdict(
            action=ACTION_NO_CONTEXT,
            reason="empty question",
            rule="empty",
            message="Please ask a question about one of the schemes.",
        )

    pii_types = detect_pii(text)
    if pii_types:
        return Verdict(
            action=ACTION_REFUSE_PII,
            reason=f"personal identifier detected: {', '.join(pii_types)}",
            rule="pii",
            # Never echo the detected value.
            message=_PII_MESSAGE,
        )

    advice = detect_advice(text)
    if advice:
        return Verdict(
            action=ACTION_REFUSE_ADVICE,
            reason=f"advice/personal-recommendation intent: '{advice}'",
            rule="advice",
            message=_ADVICE_MESSAGE,
            link=_education_link(),
        )

    if not has_context:
        dangling = inspect_dangling(text)
        if dangling is not None:
            return dangling

    if not is_mutual_fund_question(text):
        return Verdict(
            action=ACTION_REFUSE_OFFTOPIC,
            reason="no mutual-fund subject matter in the question",
            rule="offtopic",
            message=_OFFTOPIC_MESSAGE,
            link=_education_link(),
        )

    return Verdict(action=ACTION_ANSWER)


# --- Gate 4 -----------------------------------------------------------------

def check_grounding(best_distance: float | None,
                    question: str = "",
                    retrieved_texts: list[str] | None = None,
                    threshold: float | None = None) -> Verdict:
    """Decide whether the retrieved context actually supports an answer.

    Phase 5 calls this after retrieval. Two independent checks run here:

    1. **Distance.** `best_distance` is the cosine distance of the closest
       chunk; lower is closer. Rejects off-topic questions.

    2. **Concept coverage.** Distance alone is NOT enough, and this is the
       measured reason. Asking "What is the lock-in period for HDFC ELSS Tax
       Saver Fund?" retrieves the ELSS *Tax implication* chunk at distance
       0.320 - comfortably inside the 0.65 threshold - because that chunk is
       lexically similar. But it discusses STCG/LTCG taxation and says nothing
       about the lock-in. The LLM would then either invent a lock-in period or
       quote the unrelated "within one year" tax rule.

       So when the question names a specific concept, at least one retrieved
       chunk must actually contain that concept. This is what makes the
       "I don't know" path real rather than decorative.

    **Caller contract.** `retrieved_texts` must be a list of context blocks
    that include each chunk's `section` label alongside its body text, e.g.
    ``f"{section}\\n{doc}"`` for every retrieved chunk - not just the body.
    The section label lives in the embed header, not in the stored document
    text, so a body-only haystack would report "expense ratio" as missing even
    when the Expense ratio section was retrieved. Build all blocks, not just
    the top one: a fact can legitimately live in chunk 3 of 5.
    """
    limit = config.CONFIG.score_threshold if threshold is None else threshold

    if best_distance is None:
        return Verdict(
            action=ACTION_NO_CONTEXT,
            reason="no chunks retrieved",
            rule="no_retrieval",
            message=_NO_CONTEXT_MESSAGE,
        )

    if best_distance > limit:
        return Verdict(
            action=ACTION_NO_CONTEXT,
            reason=(
                f"closest chunk at distance {best_distance:.3f} exceeds the "
                f"{limit:.2f} grounding threshold"
            ),
            rule="weak_retrieval",
            message=_NO_CONTEXT_MESSAGE,
        )

    missing = missing_concepts(question, retrieved_texts or [])
    if missing:
        return Verdict(
            action=ACTION_NO_CONTEXT,
            reason=(
                "retrieved context does not mention: " + ", ".join(missing)
            ),
            rule="concept_not_covered",
            message=_NO_CONTEXT_MESSAGE,
        )

    return Verdict(action=ACTION_ANSWER)
