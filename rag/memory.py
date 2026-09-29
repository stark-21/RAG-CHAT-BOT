"""Phase 5 (extension) - Conversation memory.

Keeps a bounded window of the last 10 messages and uses it to rewrite follow-up
questions into standalone ones before retrieval.

    user: "What is the exit load of HDFC Small Cap Fund?"
    user: "what about its fees?"      -> "What is the expense ratio of HDFC Small
                                          Cap Fund?"

Three decisions worth knowing about:

1. **The rewrite runs before the off-topic gate, after the PII gate.** It has
   to: "what about its fees?" contains no mutual-fund vocabulary, so a guard
   would refuse it as off-topic before anything could resolve the reference.
   PII stays first, because a rewrite must never pull an identifier into the
   pipeline.

2. **Rewriting is not free, so it is conditional.** It costs an extra LLM
   round-trip. `needs_resolution()` fires only on a question that contains an
   unresolved reference AND does not name a scheme, so standalone questions pay
   nothing.

3. **The rewrite is fallible, so the original is kept.** If the call fails or
   returns something implausible, the question is used unchanged. A failed
   rewrite degrades to the behaviour that already existed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from rag import generator, prompt

DEFAULT_MAX_MESSAGES = 10
# A rewrite longer than this is not a rewrite of the question, it is a
# paragraph. Treat it as a failure and use the original.
MAX_REWRITE_CHARS = 220


@dataclass
class Turn:
    role: str
    content: str
    resolved: str = ""


@dataclass
class Conversation:
    """A bounded per-session transcript.

    In-memory only. Nothing here is written to disk, and PII never reaches it
    because a refused question is not recorded.
    """

    max_messages: int = DEFAULT_MAX_MESSAGES
    turns: deque[Turn] = field(default_factory=deque)

    def __post_init__(self) -> None:
        # default_factory produces an unbounded deque, so the bound has to be
        # applied here. Rebinding keeps the dataclass field contract intact.
        self.turns = deque(self.turns, maxlen=self.max_messages)

    # --- Recording ----------------------------------------------------------

    def record(self, role: str, content: str, resolved: str = "") -> None:
        self.turns.append(Turn(role=role, content=content, resolved=resolved))

    def record_exchange(self, question: str, answer: str, resolved: str = "") -> None:
        self.record("user", question, resolved=resolved)
        self.record("assistant", answer)

    def clear(self) -> None:
        self.turns.clear()

    def __len__(self) -> int:
        return len(self.turns)

    @property
    def is_empty(self) -> bool:
        return not self.turns

    # --- Reading ------------------------------------------------------------

    def history(self) -> list[tuple[str, str]]:
        return [(t.role, t.content) for t in self.turns]

    def last_user_question(self) -> str:
        for turn in reversed(self.turns):
            if turn.role == "user":
                return turn.content
        return ""

    def subjects(self) -> list[str]:
        """Corpus schemes named anywhere in the transcript, most recent first.

        Used as a fallback subject hint if the rewrite call is unavailable.
        """
        from sources import SCHEMES

        blob = "\n".join(t.content for t in self.turns).lower()
        found: list[str] = []
        for scheme in SCHEMES:
            for variant in (scheme.name, scheme.short_name, scheme.name.split(" - ")[0]):
                if variant and variant.lower() in blob:
                    found.append(scheme.name)
                    break
        return found

    # --- Rewriting ----------------------------------------------------------

    def should_resolve(self, question: str) -> bool:
        """True when the question references earlier context.

        Skipped when a scheme is named, because a question that already names
        its subject is standalone by definition.
        """
        if self.is_empty or not (question or "").strip():
            return False
        from sources import SCHEMES

        lowered = question.lower()
        for scheme in SCHEMES:
            if scheme.short_name.lower() in lowered or scheme.name.lower() in lowered:
                return False
            if scheme.name.split(" - ")[0].lower() in lowered:
                return False
        return prompt.needs_resolution(question)

    def resolve(self, question: str) -> tuple[str, bool]:
        """Return (question_to_use, was_rewritten)."""
        if not self.should_resolve(question):
            return question, False

        try:
            completion = generator.complete(
                prompt.build_followup_messages(question, self.history())
            )
            rewritten = _clean(completion.text)
        except generator.GeneratorError:
            rewritten = ""

        if not _plausible(rewritten, question):
            return self._fallback(question), False

        return rewritten, True

    def _fallback(self, question: str) -> str:
        """Append the most recent subject when the LLM rewrite is unavailable.

        Not as good as a real rewrite, but it turns "what about its fees?"
        into something retrievable instead of an unanswerable fragment.
        """
        subjects = self.subjects()
        if not subjects:
            return question
        return f"{question.rstrip('?.')} ({subjects[0]})?"


def _clean(raw: str) -> str:
    """Strip quotes, labels, and newlines the rewrite model may add."""
    text = (raw or "").strip()
    for prefix in ("rewritten:", "rewritten question:", "question:", "output:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
    text = text.strip('"').strip("'").strip()
    return " ".join(text.split())


def _plausible(rewritten: str, original: str) -> bool:
    if not rewritten:
        return False
    if len(rewritten) > MAX_REWRITE_CHARS:
        return False
    if len(rewritten.split()) < 2:
        return False
    # A rewrite that drops the question mark entirely usually means the model
    # returned commentary rather than a question.
    if original.strip().endswith("?") and not rewritten.endswith("?"):
        return False
    return True
