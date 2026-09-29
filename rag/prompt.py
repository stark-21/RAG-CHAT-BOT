"""Phase 5 - Prompt builder.

Assembles the system prompt (the rules the model must follow) plus the
retrieved context and the question. Pure string work: no network, no vendor
SDK, no config lookups beyond the freshness string passed in.
"""

from __future__ import annotations

import re

from rag.retrieval import ScoredChunk

SYSTEM_PROMPT = """You are a facts-only assistant for a mutual fund \
FAQ demo. You answer strictly from the numbered CONTEXT blocks below, which \
were retrieved from official scheme pages.

Hard rules, in priority order:

1. FACTS ONLY. Never give investment advice, recommendations, opinions, or \
personalised guidance. If asked whether to buy, sell, hold, or how to \
allocate, refuse politely and point to investor education instead.
2. USE ONLY THE CONTEXT. If the context does not contain the answer, say "I \
don't know" and state that it is not covered by the official sources. Do not \
fill the gap from your own knowledge.
3. LENGTH. At most 3 sentences. No preamble, no restating the question, no \
closing pleasantries.
4. CITATION. End with exactly one source line in this exact format:
   Source: <the single most relevant URL from that context block>
   Use one URL only. Never invent, guess, or combine URLs.
5. NO PERFORMANCE CLAIMS. Do not state, compute, compare, estimate, or imply \
returns, CAGR, NAV history, or performance. If asked, say you do not provide \
performance figures and point to the official factsheet.
6. NO PII. Never request or repeat personal identifiers.
7. Attribute facts to the specific scheme named in the context block. If the \
question omits a scheme, name the scheme your answer refers to.

Write ONLY the answer text followed by the Source line. Do not add offers of \
further help, follow-up offers, disclaimers, or "let me know if" phrases. Do not \
restate the question.

CRITICAL - if you are declining to answer: when the context does not contain \
the answer, when the fund or scheme asked about is not present in the context, \
or when the question asks for performance figures, reply in one short sentence \
saying what specifically is not covered, naming the fund or fact the user asked \
about, and include **NO Source line at all**. A citation implies the linked page \
supports your answer. Linking to an unrelated scheme's page is worse than giving \
no link. This applies especially when the user names a fund that is not among the \
context blocks. For example: "Parag Parflex Fund is not among the schemes covered \
by these official sources." Do not reply with a bare "It is not covered"."""


NO_CONTEXT_SYSTEM_PROMPT = """You are a facts-only assistant for a mutual \
FAQ demo. You could not find supporting information in the official sources \
for that question. Say so plainly in at most 2 sentences. Do not answer from \
your own knowledge, do not guess, and do not offer advice."""


def format_context(chunks: list[ScoredChunk]) -> str:
    """Render retrieved chunks as numbered, labelled blocks.

    Deliberately terse. Every character here is billed against Groq's
    input-tokens-per-minute cap, and the free tier's cap is 7,000 while a
    grounded question costs ~1,450, so there is room for roughly four
    questions a minute. Two fields were removed for that reason:

      * `retrieved:` repeated the same date in all eight blocks. The freshness
        line the user sees comes from chunk metadata via
        `retrieval.freshness_date()`, not from the prompt.
      * `scheme:` carried the full "- Direct Growth" plan suffix, which
        `scheme_short` drops. The prompt needs the scheme named for
        attribution, and the short form names it unambiguously.

    The section label stays: it is not in the stored document text, and both
    the model's disambiguation and the Phase 4 concept check rely on it.
    """
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{index}] scheme: {chunk.scheme_short}\n"
            f"    section: {chunk.section}\n"
            f"    source: {chunk.source_url}\n"
            f"    text: {chunk.text}"
        )
    return "\n\n".join(blocks)


def build_user_message(question: str, chunks: list[ScoredChunk]) -> str:
    return (
        f"CONTEXT\n\n{format_context(chunks)}\n\n"
        f"QUESTION\n{question}\n"
    )


def build_messages(question: str, chunks: list[ScoredChunk]) -> list[dict]:
    """Full chat message list for a grounded answer."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(question, chunks)},
    ]


def build_no_context_messages(question: str) -> list[dict]:
    """Chat messages for the 'I don't know' path.

    No context is supplied, which is the point: the model has nothing to
    ground on, so the only permitted output is a refusal to answer.
    """
    return [
        {"role": "system", "content": NO_CONTEXT_SYSTEM_PROMPT},
        {"role": "user", "content": f"QUESTION\n{question}\n"},
    ]


# --- Follow-up rewriting ----------------------------------------------------

FOLLOWUP_SYSTEM_PROMPT = """You rewrite a user's follow-up question into a \
standalone question that can be understood without the conversation.

Resolve pronouns and references using the history. Rules:

1. Replace references with the specific thing they point to. If the user asked \
about HDFC Small Cap Fund and then says "what about its fees?", output "What is \
the expense ratio of HDFC Small Cap Fund?"
2. Do not add information that is not in the history or the question.
3. Do not answer the question. Rewrite it only.
4. Preserve the user's original intent and wording as much as possible. Expand \
references, do not rephrase for style.
5. If the question is already standalone, output it unchanged.

Output ONLY the rewritten question as a single line. No quotes, no label, no \
explanation, no punctuation beyond the question mark."""

_NEEDS_RESOLUTION_RE = re.compile(
    r"\b(its?|their|them|they|theirs|that|this|these|those|he|she|"
    r"such|also|another|one)\b"
    r"|\b(what|how)\s+(about|of)\b"
    r"|\band\s+(it|its|that)\b"
    r"|\bsame\s+(for|question|thing)\b"
    r"|\b(better|worse|instead)\b"
    r"|\band\s*\?*$",
    re.IGNORECASE,
)


def needs_resolution(question: str) -> bool:
    """Heuristic: does this question reference earlier context?

    Cheap by design. A rewrite costs an extra LLM round-trip, so it must only
    fire on questions that actually contain an unresolved reference.
    """
    return bool(_NEEDS_RESOLUTION_RE.search(question or ""))


def build_followup_messages(question: str, history: list[tuple[str, str]]) -> list[dict]:
    """Chat messages for the rewrite call.

    `history` is a list of (role, content) pairs, oldest first.
    """
    transcript = "\n".join(
        f"{'User' if role == 'user' else 'Assistant'}: {content}"
        for role, content in history
    )
    return [
        {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"CONVERSATION SO FAR\n{transcript}\n\n"
                f"FOLLOW-UP QUESTION\n{question}\n"
            ),
        },
    ]
