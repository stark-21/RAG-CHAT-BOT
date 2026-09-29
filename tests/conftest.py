"""Shared test fixtures.

The seam is `rag.generator._call`, which is the single function that talks to
Groq. Stubbing it there rather than at `complete()` means everything between the
socket and the screen still runs for real: retry classification, token-window
pacing, response parsing, and all of `postprocess.finalise`, which is where the
citation and sentence-limit contract is actually enforced. Only the network is
removed.

**Why this file exists.** `tests/test_app.py` used to make live Groq calls, so
the suite's result depended on the provider's per-minute input-token quota. That
made the tests both slow and non-deterministic: they passed or failed depending
on what the other tests, or another process, had just spent. A suite that
reports a green build or a red build based on an external rate limit is not
testing anything. For tests of OUR code, the model is a collaborator we should
not depend on.
"""

from __future__ import annotations

import re

import pytest

# Answered facts, keyed by what the question is about. Enough for the UI
# contract: a cited answer, a benchmark, an AUM figure.
FACTS = {
    "exit load": "The exit load is 1% if redeemed within 1 year.",
    "benchmark": "The benchmark is the BSE 250 SmallCap Total Return Index.",
    "sip": "The minimum SIP amount is Rs 500 per month.",
    "aum": "The AUM is Rs 24,000 crore.",
    "nav": "The latest NAV is Rs 551.04.",
    "expense ratio": "The expense ratio is 1.57% plus GST.",
    "fund manager": "The fund manager is Dhruv Muchhal.",
}
DEFAULT_FACT = "The scheme facts are as recorded in the source page."


def _stub_response(text: str, finish: str = "stop"):
    """Minimal stand-in for a Groq chat-completions response object."""
    message = type("M", (), {"content": text, "reasoning": None})()
    choice = type("C", (), {"message": message, "finish_reason": finish})()
    usage = type("U", (), {"completion_tokens": 20})()
    return type("R", (), {"choices": [choice], "usage": usage})()


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace the Groq call with a deterministic one. Returns a call log."""
    from rag import generator

    calls: list[dict] = []

    def fake_call(messages: list[dict], max_tokens: int):
        calls.append({"messages": messages, "max_tokens": max_tokens})
        system = " ".join(m.get("content", "") for m in messages
                          if m.get("role") == "system")
        user = " ".join(m.get("content", "") for m in messages
                        if m.get("role") == "user")

        # The follow-up rewriter is a different prompt shape: it carries a
        # transcript plus a "FOLLOW-UP QUESTION" section and wants the
        # standalone question back, not an answer. Keying on that marker rather
        # than on position keeps the memory tests exercising the real branch.
        if "FOLLOW-UP QUESTION" in user:
            followup = user.split("FOLLOW-UP QUESTION", 1)[1]
            subject = _last_subject(messages)
            return _stub_response(
                f"What is the {_topic_of(followup)} of {subject}?")

        fact = DEFAULT_FACT
        for key, text in FACTS.items():
            if key in user.lower():
                fact = text
                break
        scheme = _scheme_in(user) or "hdfc-small-cap-fund-direct-growth"
        return _stub_response(
            f"{fact}\n\nSource: https://groww.in/mutual-funds/{scheme}")

    monkeypatch.setattr(generator, "_call", fake_call)
    return calls


def _topic_of(text: str) -> str:
    """Pull the topic out of a follow-up: "what about its benchmark?" -> "benchmark".

    A real rewriter returns a whole question; the tests only need the topic to
    survive, because the memory test asserts that the resolved question reached
    the right part of the corpus.
    """
    topic = text.strip().rstrip("?").strip()
    topic = re.sub(
        r"^(what(?:'s| is| about)|and|also|its|it's|it|the|a|an|does|do)\s+",
        "", topic, flags=re.IGNORECASE)
    return topic.strip() or "exit load"


def _last_subject(messages: list[dict]) -> str:
    """Pull the most recent fund name out of the transcript in the prompt."""
    blob = " ".join(m.get("content", "") for m in messages)
    names = re.findall(r"HDFC [A-Za-z ()]+?Fund", blob)
    return names[-1].strip() if names else "HDFC Small Cap Fund"


def _scheme_in(text: str) -> str | None:
    match = re.search(r"HDFC [A-Za-z ()]+?Fund", text)
    if not match:
        return None
    return match.group(0).lower().replace(" ", "-") + "-direct-growth"


@pytest.fixture(autouse=True, scope="session")
def _no_accidental_network_calls():
    """Fail the run if anything reaches the real provider.

    `rag.generator._client` is the only thing in the project that constructs a
    Groq client, and a live call has to go through it. Patching it to raise means
    a test that forgets the `fake_llm` fixture fails loudly and immediately,
    instead of quietly consuming the input-token quota and reporting a green run
    that proved nothing.
    """
    from rag import generator

    original = generator._client

    def forbidden():
        raise AssertionError(
            "A test tried to build a real Groq client, so it would make a live "
            "API call. Use the `fake_llm` fixture from tests/conftest.py. If you "
            "genuinely want a live check, write it in a separate, opt-in file "
            "and mark it, rather than letting it run in the normal suite."
        )

    generator._client = forbidden
    try:
        yield
    finally:
        generator._client = original
