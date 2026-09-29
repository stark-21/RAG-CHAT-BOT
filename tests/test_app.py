"""Phase 6 - Streamlit UI tests.

These use Streamlit's AppTest harness, which actually EXECUTES the app script.
A 200 from the server does not prove the app works: Streamlit renders over a
websocket, so the shell can serve fine while the app body raises on every run.
AppTest is what catches that.

These hit the real pipeline, so they make real Groq calls. The UI contract is
worth that cost, but it is why this file is separate from the unit tests.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

import config

TIMEOUT = 180
APP_PATH = config.BASE_DIR / "app.py"


def launch() -> AppTest:
    # Absolute path: AppTest resolves relative paths against the test module's
    # directory, not the project root.
    at = AppTest.from_file(str(APP_PATH), default_timeout=TIMEOUT).run()
    if at.exception:
        raise AssertionError("app raised on load: {}".format(
            at.exception[0].value))
    return at


def ask(at: AppTest, question: str) -> AppTest:
    at.chat_input[0].set_value(question).run()
    if at.exception:
        raise AssertionError("app raised on '{}': {}".format(
            question, at.exception[0].value))
    return at


def last_body(at: AppTest) -> str:
    return " ".join(x.value for x in at.chat_message[-1].markdown)


class TestFirstLoad:
    def test_renders_without_error(self):
        launch()

    def test_shows_facts_only_disclaimer(self):
        """PRD N7."""
        at = launch()
        assert any("Facts-only. No investment advice." in c.value
                   for c in at.caption)

    def test_shows_welcome_line(self):
        assert "Mutual Fund FAQ Assistant" in launch().title[0].value

    def test_shows_three_example_questions(self):
        """PRD N7."""
        examples = [b.label for b in launch().button
                    if b.label and "Clear" not in b.label]
        assert len(examples) == 3

    def test_has_clear_chat_button(self):
        assert any("Clear" in (b.label or "") for b in launch().button)

    def test_has_chat_input(self):
        assert launch().chat_input


class TestAnswering:
    def test_example_click_returns_cited_answer(self):
        at = launch()
        [b for b in at.button if "exit load" in b.label.lower()][0].click().run()
        assert not at.exception
        assert len(at.chat_message) == 2
        body = last_body(at)
        assert "Source:" in body
        assert "https://" in body

    def test_answer_shows_freshness_date(self):
        at = ask(launch(), "What is the exit load of HDFC Small Cap Fund?")
        captions = " ".join(c.value for c in at.chat_message[-1].caption)
        assert "Last updated from sources:" in captions

    def test_sources_expander_under_answer(self):
        at = ask(launch(), "What is the exit load of HDFC Small Cap Fund?")
        assert at.expander, "no Sources expander rendered"
        assert "Sources" in at.expander[0].label
        captions = " ".join(c.value for c in at.expander[0].caption)
        assert "distance" in captions, "expander must show chunk distances"
        assert "https://" in captions, "expander must show the source URL"

    def test_expander_only_for_grounded_answers(self):
        """A refusal has no chunks, so it must not show an empty expander."""
        at = ask(launch(), "What is the weather in Mumbai?")
        assert not at.expander


class TestHistoryAndMemory:
    def test_history_accumulates(self):
        at = launch()
        at = ask(at, "What is the exit load of HDFC Small Cap Fund?")
        at = ask(at, "What is the benchmark of HDFC Small Cap Fund?")
        assert len(at.chat_message) == 4

    def test_follow_up_resolves_against_history(self):
        """Memory must survive Streamlit's reruns."""
        at = ask(launch(), "What is the exit load of HDFC Small Cap Fund?")
        at = ask(at, "what about its benchmark?")
        captions = " ".join(c.value for c in at.chat_message[-1].caption)
        assert "interpreted as" in captions
        assert "BSE 250" in last_body(at) or "benchmark" in last_body(at).lower()

    def test_conversation_memory_bounded_to_ten(self):
        at = ask(launch(), "What is the exit load of HDFC Small Cap Fund?")
        for _ in range(4):
            at = ask(at, "What is the exit load of HDFC Large Cap Fund?")
        assert len(at.session_state.conversation) <= 10


class TestClearChat:
    def test_clears_transcript_and_memory(self):
        """Both must clear. A stale Conversation would let a later
        'what about its fees?' resolve against an invisible transcript."""
        at = ask(launch(), "What is the exit load of HDFC Small Cap Fund?")
        assert len(at.session_state.messages) > 0
        [b for b in at.button if "Clear" in b.label][0].click().run()
        assert not at.exception
        assert len(at.session_state.messages) == 0
        assert len(at.session_state.conversation) == 0
        assert len(at.chat_message) == 0

    def test_cleared_session_does_not_inherit_subject(self):
        at = ask(launch(), "What is the exit load of HDFC Small Cap Fund?")
        [b for b in at.button if "Clear" in b.label][0].click().run()
        at = ask(at, "what about its fees?")
        body = last_body(at)
        assert "which scheme you mean" in body.lower(), (
            "a cleared session must not remember a subject")


class TestRefusals:
    def test_advice_refused_with_education_link(self):
        at = ask(launch(), "Should I buy HDFC ELSS Tax Saver Fund?")
        body = last_body(at)
        assert "can't recommend" in body.lower()
        assert "sebi.gov.in" in body or "amfiindia" in body
        assert "Source:" not in body

    def test_pii_refused_and_not_echoed(self):
        """PRD: the PAN must not be echoed back or stored.

        Regression. The transcript is re-rendered on every Streamlit rerun, so
        storing the raw question meant the PAN was painted back onto the screen
        on the next run even though the assistant's reply was clean. Checking
        only the last message missed it; this checks every rendered element.
        """
        at = ask(launch(), "My PAN is ABCDE1234F")

        rendered = " ".join(
            [x.value for m in at.chat_message for x in m.markdown]
            + [c.value for m in at.chat_message for c in m.caption]
            + [w.value for w in at.markdown]
        )
        assert "ABCDE1234F" not in rendered, "PAN leaked into the rendered UI"

        stored = str(at.session_state.messages)
        assert "ABCDE1234F" not in stored, "PAN leaked into the transcript"
        assert "not stored" in rendered, "user should see why nothing was kept"

        assert "personal identifiers" in last_body(at)

    def test_offtopic_refused(self):
        at = ask(launch(), "What is the weather in Mumbai?")
        assert "can only answer questions" in last_body(at).lower()


class TestNoReingest:
    def test_persistent_store_is_reused(self):
        """Restarting the UI must not re-ingest (Phase 3 persistence)."""
        from rag.vectorstore import get_collection

        assert get_collection().count() > 0
