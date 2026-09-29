"""Phase 5 - Generator.

The ONLY module in rag/ that imports the Groq SDK. Swapping vendors means
changing this file alone.

**Client reuse.** The Groq client is built once per process. Constructing it
per call re-reads credentials and re-creates the underlying connection pool on
every question, which is pure latency on a small host.

**Rate limits are the normal case, not an exception.** Measured on the free
tier: the cap is 7,000 input tokens per minute, and one grounded question
costs ~1,450 of them, so the ceiling is roughly four questions a minute. A
demo, or one person clicking through the examples, hits that immediately.

A bare 429 used to surface as a hard "Groq request failed" error, and a blind
retry loop turned it into a 10-second stall before the same error. Three
things prevent that now:

  1. `_TokenWindow` counts input tokens over a trailing 60s window and delays a
     request that would push the process over the cap, so the limit is rarely
     reached in the first place.
  2. When it is reached anyway, the server's own advice is followed. Groq's
     429 body says "Please try again in 3.63 seconds" and names the tier's
     limit; both are parsed out and used, so the process self-calibrates to
     whatever tier the key is on instead of hard-coding a number.
  3. A configuration error (401, 404, 400) still fails immediately. Sleeping
     cannot fix a wrong key.

**Reasoning-model budget.** Reasoning models bill their chain-of-thought
against `max_tokens` alongside the visible answer, so they need a much larger
budget than the answer itself. The default model does not reason
(`config.CONFIG.reasoning_effort` is empty), so the budget is sized for the
visible answer only. If you point GROQ_MODEL at a reasoning model, raise
MAX_TOKENS and set GROQ_MODEL_REASONING_EFFORT=low.
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from functools import lru_cache

import config

# Visible answer needs ~120 tokens. With no reasoning in the default model this
# is already generous; kept as a floor so a small MAX_TOKENS cannot starve it.
DEFAULT_MAX_TOKENS = 400
RETRY_MAX_TOKENS = 900

# Statuses worth retrying: rate limiting and transient upstream failures.
# Everything else (401 bad key, 404 bad model, 400 bad request) is a
# configuration problem and must fail immediately, not after a minute of
# sleeping.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 6.0

# How long one question may spend waiting out rate limiting before the user is
# told to slow down.
#
# These two numbers are deliberately small and deliberately NOT the length of a
# token window. A window is 60s, so a limiter that waited for one would sit for
# a full minute and then send the request anyway and get the 429 - the exact
# "it takes forever and then fails" behaviour this replaces. Two short waits
# are better than one long one: the limiter smooths bursts of a few seconds,
# and the retry handles a genuinely exhausted quota with the server's own
# advice, so the user sees a result or a clear message inside ~12s.
MAX_PACING_WAIT_S = 3.0
MAX_TOTAL_WAIT_S = 12.0

# Conservative until the server tells us the real number. The free tier is
# 7,000; the 10% margin leaves room for the estimate being slightly high.
DEFAULT_ITPM_LIMIT = 7000
ITPM_SAFETY = 0.9
WINDOW_S = 60.0


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


# --- Rate limiting ----------------------------------------------------------
#
# Groq bills input tokens per minute, not requests, so the limiter counts
# tokens over a trailing 60-second window. The limit is not hard-coded: it
# starts at the free-tier value and is replaced by whatever the server reports
# in a 429, which is how a paid tier self-configures here.

_LIMIT_RE = re.compile(r"Limit\s+(\d[\d,]*)", re.IGNORECASE)
_RETRY_AFTER_RE = re.compile(
    r"(?:try again in|retry after)\s*([0-9]*\.?[0-9]+)\s*(?:s|sec|second)", re.IGNORECASE
)


class _TokenWindow:
    """Trailing-window input-token counter with a self-calibrating ceiling."""

    def __init__(self, limit: int = DEFAULT_ITPM_LIMIT):
        self._events: deque[tuple[float, int]] = deque()
        self._limit = limit
        self._lock = threading.Lock()

    @property
    def limit(self) -> int:
        with self._lock:
            return self._limit

    def observe_limit(self, message: str) -> None:
        """Adopt the tier's real limit, as stated in a 429 body."""
        match = _LIMIT_RE.search(message or "")
        if not match:
            return
        try:
            reported = int(match.group(1).replace(",", ""))
        except ValueError:
            return
        if reported > 0:
            with self._lock:
                self._limit = reported

    def _trim(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= WINDOW_S:
            self._events.popleft()

    def used(self) -> int:
        now = time.monotonic()
        with self._lock:
            self._trim(now)
            return sum(count for _, count in self._events)

    def reserve(self, tokens: int) -> float:
        """Block until `tokens` fit in the window. Returns seconds slept.

        Two properties matter and both are covered by tests:

        * The wait is capped at `MAX_PACING_WAIT_S`, which is far shorter than
          the window. The limiter smooths a burst of a few questions; it does
          not try to outlast a full minute of quota exhaustion, because the
          request would still be refused when the window rolled over.
        * The reservation is recorded even when the cap is hit. Dropping it
          would understate usage exactly when the process is already over its
          limit. The request then goes out, gets a 429, and `_call_with_retry`
          deals with it.
        """
        tokens = max(int(tokens), 1)
        budget = int(self.limit * ITPM_SAFETY)
        now = time.monotonic()
        deadline = now + MAX_PACING_WAIT_S
        slept = 0.0

        while True:
            with self._lock:
                self._trim(now)
                used = sum(count for _, count in self._events)
                if used + tokens <= budget or now >= deadline:
                    self._events.append((now, tokens))
                    return slept
                # Sleep until the oldest recorded tokens age out of the window,
                # but never past the pacing cap.
                wait = (WINDOW_S - (now - self._events[0][0]) + 0.25
                        if self._events else 0.5)
            wait = min(max(wait, 0.25), 1.0, deadline - now)
            if wait <= 0:
                with self._lock:
                    self._events.append((now, tokens))
                return slept
            time.sleep(wait)
            slept += wait
            now = time.monotonic()

    def forget(self) -> None:
        """Drop the reservation for a request that never reached the server."""
        with self._lock:
            if self._events:
                self._events.pop()


_LIMITER = _TokenWindow()


@lru_cache(maxsize=1)
def _client():
    """Build the Groq client once per process.

    Cached because constructing it re-reads credentials and re-creates the
    connection pool, which is pure latency on a request that should take half a
    second. The key itself is still checked on every call, in `_call`, so a
    missing key never slips through on a warm cache.
    """
    from groq import Groq

    return Groq(api_key=config.CONFIG.groq_api_key, max_retries=0)


def _status_of(exc: Exception) -> int | None:
    """HTTP status carried by a Groq SDK exception, or None if there isn't one."""
    for attribute in ("status_code", "http_status"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int):
            return value
    return None


def _estimated_tokens(messages: list[dict]) -> int:
    """Rough input-token count, used only to pace the rate limiter.

    Measured on the real prompt: 5,074 characters billed as 1,446 tokens, i.e.
    3.5 chars/token. Dividing by 3.3 errs a few percent high, which is the
    right direction: the limiter then paces slightly early instead of eating a
    429 and its recovery wait.
    """
    characters = sum(len(str(m.get("content") or "")) for m in messages)
    return int(characters / 3.3) + 64


def _retry_after_seconds(exc: Exception) -> float | None:
    """The server's own "try again in Ns", if it gave one."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        raw = headers.get("retry-after") or headers.get("Retry-After")
        if raw:
            try:
                return max(float(raw), 0.0)
            except (TypeError, ValueError):
                pass
    match = _RETRY_AFTER_RE.search(str(exc))
    if match:
        try:
            return max(float(match.group(1)), 0.0)
        except ValueError:
            return None
    return None


def _call(messages: list[dict], max_tokens: int) -> dict:
    """One chat-completions request, no retry. Returns the response."""
    # Checked here, not in _client(), so it holds even when the cached client
    # was built while a key was present.
    if not config.has_groq_key():
        raise GeneratorError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    payload = {
        "model": config.CONFIG.llm_model,
        "messages": messages,
        "temperature": config.CONFIG.temperature,
        "max_tokens": max_tokens,
    }
    if config.CONFIG.reasoning_effort:
        payload["reasoning_effort"] = config.CONFIG.reasoning_effort

    estimated = _estimated_tokens(messages)
    # Pace before sending, not after failing.
    _LIMITER.reserve(estimated)
    try:
        response = _client().chat.completions.create(**payload)
    except GeneratorError:
        _LIMITER.forget()
        raise
    except Exception as exc:  # noqa: BLE001 - re-raised as a friendly error
        # The request may or may not have been billed, but it definitely did
        # not succeed, so the reservation is released.
        _LIMITER.forget()
        raise GeneratorError(f"Groq request failed: {exc}") from exc
    return response


def _call_with_retry(messages: list[dict], max_tokens: int) -> tuple[dict, bool]:
    """Call, retrying transient failures. Returns (response, did_retry).

    Total time spent sleeping is capped at `MAX_TOTAL_WAIT_S`, so a question
    either comes back or reports, rather than stalling.
    """
    last_error: Exception | None = None
    advised_wait: float | None = None
    deadline = time.monotonic() + MAX_TOTAL_WAIT_S

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return _call(messages, max_tokens), attempt > 1
        except GeneratorError as exc:
            cause = exc.__cause__
            status = _status_of(cause) if cause is not None else None
            if status is None or status not in RETRYABLE_STATUS:
                raise
            last_error = exc

            # Learn the tier's real ceiling from the error it just gave us.
            _LIMITER.observe_limit(str(cause))

            if attempt == MAX_ATTEMPTS:
                break

            # Prefer the server's advice over a guess: it knows when the window
            # rolls over, and guessing low just burns another 429.
            server_advice = _retry_after_seconds(cause) if cause else None
            if server_advice is not None:
                advised_wait = server_advice
            delay = server_advice
            if delay is None:
                delay = min(BACKOFF_BASE_S * (2 ** (attempt - 1)), BACKOFF_CAP_S)
            delay = min(max(delay, 0.5), BACKOFF_CAP_S)
            remaining = deadline - time.monotonic()
            if delay >= remaining:
                break
            time.sleep(delay)

    hint = ""
    if advised_wait:
        hint = f" Try again in about {math.ceil(advised_wait)}s."
    raise GeneratorError(
        f"Groq is rate limiting this key and did not recover after "
        f"{MAX_ATTEMPTS} attempts. This tier allows {_LIMITER.limit:,} input "
        f"tokens per minute and a question here costs roughly "
        f"{_estimated_tokens(messages):,}, so only a few can be asked per "
        f"minute.{hint} ({last_error})"
    )


def _to_completion(response, max_tokens: int, retried: bool = False) -> Completion:
    choice = response.choices[0]
    message = choice.message
    return Completion(
        text=(message.content or "").strip(),
        finish_reason=choice.finish_reason or "",
        completion_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
        reasoning=bool(getattr(message, "reasoning", None)),
        max_tokens=max_tokens,
        retried=retried,
    )


def complete(messages: list[dict]) -> Completion:
    """Send messages to Groq, retrying once if the answer was truncated away.

    Retry condition is deliberately narrow: truncated AND unusable. A truncated
    but non-empty answer is still readable and is returned so the caller can
    decide, rather than silently burning a second call.
    """
    budget = max(config.CONFIG.max_tokens, DEFAULT_MAX_TOKENS)
    response, did_retry = _call_with_retry(messages, budget)
    result = _to_completion(response, budget, retried=did_retry)

    if result.truncated and not result.usable:
        response, did_retry = _call_with_retry(
            messages, max(budget, RETRY_MAX_TOKENS)
        )
        result = _to_completion(
            response, max(budget, RETRY_MAX_TOKENS), retried=True
        )

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
