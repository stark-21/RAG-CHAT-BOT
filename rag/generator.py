"""Phase 5 - Generator.

The ONLY module in rag/ that imports the Groq SDK. Swapping vendors means
changing this file alone.

**Reasoning-model budget.** The default model, `openai/gpt-oss-120b`, is a
reasoning model: its chain-of-thought is billed against `max_tokens` alongside
the visible answer. Measured on a factual query, `max_tokens=220` returned
`finish_reason="length"` with 220 completion tokens and an EMPTY content string,
because the reasoning consumed the entire budget. The failure is silent - no
exception, just a blank answer. `max_tokens` is therefore set well above what
the visible answer needs, and truncation is detected and retried rather than
passed downstream as a factually wrong-looking fragment.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import config

# Visible answer needs ~120 tokens. Reasoning needs far more, and the split
# varies per prompt, so this is deliberately generous.
DEFAULT_MAX_TOKENS = 2000
RETRY_MAX_TOKENS = 4000


class GeneratorError(RuntimeError):
    """Raised when the LLM call fails. The UI shows this, never a stack trace."""


@dataclass
class Completion:
    """Raw model result, including the bits that indicate silent failure."""

    text: str
    finish_reason: str
    completion_tokens: int
    reasoning: bool
    max_tokens: int
    retried: bool = False

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def usable(self) -> bool:
        return not self.is_empty


def _client():
    from groq import Groq

    if not config.has_groq_key():
        raise GeneratorError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return Groq(api_key=config.CONFIG.groq_api_key)


def _once(messages: list[dict], max_tokens: int) -> Completion:
    client = _client()
    try:
        response = client.chat.completions.create(
            model=config.CONFIG.llm_model,
            messages=messages,
            temperature=config.CONFIG.temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as a friendly error
        raise GeneratorError(f"Groq request failed: {exc}") from exc

    choice = response.choices[0]
    message = choice.message
    return Completion(
        text=(message.content or "").strip(),
        finish_reason=choice.finish_reason or "",
        completion_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
        reasoning=bool(getattr(message, "reasoning", None)),
        max_tokens=max_tokens,
    )


def complete(messages: list[dict]) -> Completion:
    """Send messages to Groq, retrying once if the answer was truncated away.

    Retry condition is deliberately narrow: truncated AND unusable. A truncated
    but non-empty answer is still readable and is returned so the caller can
    decide, rather than silently burning a second call.
    """
    budget = max(config.CONFIG.max_tokens, DEFAULT_MAX_TOKENS)
    result = _once(messages, budget)

    if result.truncated and not result.usable:
        result = _once(messages, max(budget, RETRY_MAX_TOKENS))
        result.retried = True

    if not result.usable:
        raise GeneratorError(
            f"{config.CONFIG.llm_model} returned an empty answer "
            f"(finish_reason={result.finish_reason or 'unknown'}, "
            f"{result.completion_tokens} tokens). Raise MAX_TOKENS, or pick a "
            f"non-reasoning model via GROQ_MODEL."
        )
    return result


def timed_complete(messages: list[dict]) -> tuple[Completion, float]:
    """complete() plus elapsed seconds, for the PRD N1 latency check."""
    started = time.time()
    result = complete(messages)
    return result, time.time() - started
