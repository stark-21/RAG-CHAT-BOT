"""End-to-end query path, wired once so the CLI and the UI cannot drift.

    PII gate -> follow-up rewrite -> guard -> retrieval -> grounding
             -> prompt -> generation -> postprocess

Refusals short-circuit before retrieval, and "I don't know" short-circuits
before the LLM. A refused question costs no API call.

The ordering of the first three steps is load-bearing:

* PII is checked on the RAW question. A rewrite must never pull an
  identifier into the pipeline.
* The follow-up rewrite runs BEFORE `guard.inspect()`, because
  `inspect()` includes the off-topic gate and "what about its fees?" has no
  mutual-fund vocabulary. Without the rewrite it would be refused as
  off-topic before its reference could be resolved.
* The grounding gate sees the RESOLVED question, so concept coverage is
  checked against "expense ratio" rather than "fees".
"""

from __future__ import annotations

from functools import lru_cache

import config
from rag import generator, guard, memory as memory_mod, postprocess, prompt, retrieval
from rag.guard import Verdict
from rag.memory import Conversation
from rag.postprocess import Answer


class QueryResult:
    """Everything a caller needs, including the retrieved chunks for display."""

    def __init__(self, answer: Answer, chunks: list, verdict: Verdict,
                 stage: str, latency_s: float = 0.0,
                 resolved_question: str = "", rewritten: bool = False):
        self.answer = answer
        self.chunks = chunks
        self.verdict = verdict
        self.stage = stage
        self.latency_s = latency_s
        self.resolved_question = resolved_question
        self.rewritten = rewritten

    @property
    def refused(self) -> bool:
        return self.answer.refused

    def clone(self, **overrides) -> "QueryResult":
        """A copy with selected fields replaced, for cache replay."""
        fields = {
            "answer": self.answer,
            "chunks": self.chunks,
            "verdict": self.verdict,
            "stage": self.stage,
            "latency_s": self.latency_s,
            "resolved_question": self.resolved_question,
            "rewritten": self.rewritten,
        }
        fields.update(overrides)
        return QueryResult(**fields)


# Bounded, because the corpus is small and the questions that recur are few: the
# three example buttons, and a follow-up that resolves back to something already
# asked. That is precisely the traffic that would otherwise spend the rate limit.
@lru_cache(maxsize=128)
def _cached_answer(resolved: str, has_context: bool) -> QueryResult:
    return _run(resolved, has_context)


def clear_cache() -> None:
    """Drop cached answers. Called when the store is rebuilt."""
    _cached_answer.cache_clear()


def answer_question(question: str, show_chunks: bool = False,
                    conversation: Conversation | None = None) -> QueryResult:
    """Run the full query path for one question.

    `conversation` is optional. Without it the assistant is stateless, which is
    the behaviour Phase 5 originally shipped.

    The LLM call is cached on the RESOLVED question, because everything after
    step 2 is a pure function of it: same resolved question, same chunks, same
    prompt, same answer, given a fixed corpus and a temperature of 0. The
    rewrite in step 2 still runs, so a follow-up costs its rewrite but not a
    second full generation. `has_context` is part of the key because the
    dangling-reference gate behaves differently with and without history.
    """
    raw = (question or "").strip()

    # 1. PII on the raw input, before any rewriting.
    pii_verdict = guard.inspect_pii(raw)
    if pii_verdict is not None and pii_verdict.is_refusal:
        return QueryResult(postprocess.refusal_answer(pii_verdict), [], pii_verdict,
                           "guard", 0.0, raw, False)

    # 2. Resolve follow-up references against the transcript.
    resolved, rewritten = raw, False
    if conversation is not None:
        resolved, rewritten = conversation.resolve(raw)

    # 3. Cheap gates: advice, dangling reference, off-topic. No retrieval, no API.
    has_context = conversation is not None and bool(conversation.subjects())
    verdict = guard.inspect(resolved, has_context=has_context)
    if verdict.is_refusal:
        if conversation is not None:
            conversation.record_exchange(raw, verdict.message, resolved)
        return QueryResult(postprocess.refusal_answer(verdict), [], verdict,
                           "guard", 0.0, resolved, rewritten)

    # 4 onwards: cached. Steps 1-3 stay outside the cache so the guards always
    # run against what the user actually typed.
    result = _cached_answer(resolved, has_context)

    if conversation is not None:
        # The cached run recorded the exchange against a throwaway conversation,
        # so the real one records it here. Same text, same resolved question.
        conversation.record_exchange(raw, result.answer.text, resolved)
    return result.clone(resolved_question=resolved, rewritten=rewritten)


def _run(resolved: str, has_context: bool) -> QueryResult:
    """Steps 4-7: retrieve, ground, generate, enforce. No conversation state."""
    # 4. Retrieve with the same embedder used at ingest.
    chunks = retrieval.retrieve(resolved)

    # 5. Grounding: distance + concept coverage.
    grounding = guard.check_grounding(
        retrieval.best_distance(chunks),
        resolved,
        retrieval.grounding_context(chunks),
    )
    if grounding.is_refusal:
        return QueryResult(
            postprocess.refusal_answer(grounding, chunks), chunks, grounding,
            "grounding", 0.0, resolved, False,
        )

    # 6. Generate.
    #
    # The grounding gate above judged the FULL top_k set, and that judgement is
    # what "I don't know" depends on, so it is not narrowed. Only the prompt is
    # narrowed, to LLM_CONTEXT_K blocks. See config for the measurement behind
    # that default.
    width = config.CONFIG.llm_context_k or len(chunks)
    messages = prompt.build_messages(resolved, chunks[:width])
    completion, elapsed = generator.timed_complete(messages)

    # 7. Enforce the contract.
    final = postprocess.finalise(
        completion.text, chunks, retrieval.freshness_date(chunks), elapsed
    )
    if completion.truncated:
        final.warnings.append(
            f"answer hit the {completion.max_tokens}-token cap "
            f"(finish_reason=length); may be incomplete"
        )
    return QueryResult(final, chunks, guard.inspect(resolved), "answered", elapsed,
                       resolved, False)
