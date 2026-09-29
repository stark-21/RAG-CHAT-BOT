"""Conversation-memory tests.

The rewrite LLM call is mocked. These assert on OUR logic - window bounds,
when to rewrite, what counts as a usable rewrite, and the gate ordering - not
on model phrasing, which changes without notice.
"""

from __future__ import annotations

import pytest

from rag import memory as memory_mod, prompt
from rag.generator import Completion
from rag.memory import Conversation


def completion(text, **kw):
    return Completion(
        text=text,
        finish_reason=kw.get("finish", "stop"),
        completion_tokens=kw.get("tokens", 30),
        reasoning=False,
        max_tokens=2000,
    )


# --- window bound -----------------------------------------------------------

class TestWindow:
    def test_keeps_last_ten(self):
        conv = Conversation()
        for i in range(15):
            conv.record_exchange("q{}".format(i), "a{}".format(i))
        assert len(conv) == 10

    def test_evicts_oldest_first(self):
        conv = Conversation()
        for i in range(12):
            conv.record("user", "q{}".format(i))
        kept = [t.content for t in conv.turns]
        assert kept[0] == "q2"
        assert kept[-1] == "q11"
        assert "q0" not in kept

    def test_deque_has_maxlen(self):
        conv = Conversation()
        assert conv.turns.maxlen == 10

    def test_custom_window(self):
        conv = Conversation(max_messages=3)
        for i in range(6):
            conv.record("user", "q{}".format(i))
        assert len(conv) == 3

    def test_clear(self):
        conv = Conversation()
        conv.record_exchange("hi", "hello")
        conv.clear()
        assert conv.is_empty

    def test_history_shape(self):
        conv = Conversation()
        conv.record_exchange("q", "a")
        assert conv.history() == [("user", "q"), ("assistant", "a")]

    def test_last_user_question(self):
        conv = Conversation()
        conv.record_exchange("first", "a1")
        conv.record_exchange("second", "a2")
        assert conv.last_user_question() == "second"


# --- when to rewrite --------------------------------------------------------

class TestShouldResolve:
    def test_empty_conversation_never_rewrites(self):
        conv = Conversation()
        assert conv.should_resolve("what about its fees?") is False

    def test_pronoun_triggers(self):
        conv = Conversation()
        conv.record_exchange("Tell me about HDFC Small Cap Fund", "ok")
        for q in ("what about its fees?", "and its benchmark?", "is that good?",
                  "what about that one?", "how about it?"):
            assert conv.should_resolve(q) is True, q

    def test_named_scheme_does_not_rewrite(self):
        """A question naming its subject is standalone; do not pay for a call."""
        conv = Conversation()
        conv.record_exchange("Tell me about HDFC Small Cap Fund", "ok")
        assert conv.should_resolve(
            "What is the exit load of HDFC Small Cap Fund?") is False

    def test_blank_question(self):
        conv = Conversation()
        conv.record_exchange("hi", "hello")
        assert conv.should_resolve("   ") is False

    def test_resolve_returns_original_when_not_needed(self):
        conv = Conversation()
        conv.record_exchange("hi", "hello")
        text, rewritten = conv.resolve("What is the AUM of HDFC Large Cap Fund?")
        assert text == "What is the AUM of HDFC Large Cap Fund?"
        assert rewritten is False


class TestNeedsResolution:
    def test_detects_anaphora(self):
        for q in ("what about its fees?", "and its benchmark?", "is that right?",
                  "what about this fund?", "how about that?"):
            assert prompt.needs_resolution(q) is True, q

    def test_ignores_standalone(self):
        for q in ("What is the exit load of HDFC Small Cap Fund?",
                  "What is the minimum SIP?", "Tell me about the ELSS tax rules"):
            assert prompt.needs_resolution(q) is False, q


# --- rewrite quality --------------------------------------------------------

class TestCleanAndPlausible:
    def test_strips_quotes(self):
        assert memory_mod._clean('"What is the exit load?"') == "What is the exit load?"

    def test_strips_labels(self):
        assert memory_mod._clean("Rewritten: What is it?") == "What is it?"

    def test_collapses_whitespace(self):
        assert memory_mod._clean("What  is\n the  cost?") == "What is the cost?"

    def test_rejects_too_long(self):
        long_q = "What is " + ("the fee structure and its implications " * 8)
        assert memory_mod._plausible(long_q, "x?") is False

    def test_rejects_too_short(self):
        assert memory_mod._plausible("Why?", "why?") is False
        assert memory_mod._plausible("", "x?") is False

    def test_rejects_commentary_without_question_mark(self):
        assert memory_mod._plausible("I cannot determine that", "what?") is False

    def test_accepts_legitimate_rewrite(self):
        assert memory_mod._plausible(
            "What is the expense ratio of HDFC Small Cap Fund?",
            "what about its fees?") is True


# --- resolve, with the LLM mocked -----------------------------------------

class TestResolve:
    def test_uses_rewrite(self, monkeypatch):
        conv = Conversation()
        conv.record_exchange(
            "What is the exit load of HDFC Small Cap Fund?",
            "The exit load is 1%.")
        monkeypatch.setattr(
            memory_mod.generator, "complete",
            lambda msgs: completion("What is the expense ratio of HDFC Small Cap Fund?"),
        )
        text, rewritten = conv.resolve("what about its fees?")
        assert rewritten is True
        assert "HDFC Small Cap Fund" in text

    def test_falls_back_when_call_fails(self, monkeypatch):
        from rag.generator import GeneratorError

        conv = Conversation()
        conv.record_exchange("What is the exit load of HDFC Small Cap Fund?",
                             "The exit load is 1%.")

        def boom(_):
            raise GeneratorError("no key")

        monkeypatch.setattr(memory_mod.generator, "complete", boom)
        text, rewritten = conv.resolve("what about its fees?")
        assert rewritten is False
        # Fallback appends the most recent subject rather than losing the query.
        assert "HDFC Small Cap Fund" in text

    def test_falls_back_on_implausible_rewrite(self, monkeypatch):
        conv = Conversation()
        conv.record_exchange("What is the exit load of HDFC Small Cap Fund?",
                             "The exit load is 1%.")
        monkeypatch.setattr(
            memory_mod.generator, "complete",
            lambda msgs: completion("I cannot answer that."),
        )
        text, rewritten = conv.resolve("what about its fees?")
        assert rewritten is False
        assert "HDFC Small Cap Fund" in text

    def test_fallback_without_subject_is_original(self, monkeypatch):
        from rag.generator import GeneratorError

        conv = Conversation()
        conv.record_exchange("Tell me about mutual funds", "Sure.")
        monkeypatch.setattr(
            memory_mod.generator, "complete",
            lambda msgs: (_ for _ in ()).throw(GeneratorError("down")),
        )
        text, rewritten = conv.resolve("what about that?")
        assert rewritten is False
        assert text == "what about that?"


# --- prompt construction ----------------------------------------------------

class TestFollowupPrompt:
    def test_includes_transcript_and_question(self):
        msgs = prompt.build_followup_messages(
            "what about its fees?",
            [("user", "exit load?"), ("assistant", "1%")],
        )
        assert msgs[0]["role"] == "system"
        assert "User: exit load?" in msgs[1]["content"]
        assert "Assistant: 1%" in msgs[1]["content"]
        assert msgs[1]["content"].rstrip().endswith("what about its fees?")

    def test_system_prompt_forbids_answering(self):
        s = prompt.FOLLOWUP_SYSTEM_PROMPT.lower()
        assert "do not answer" in s
        assert "standalone" in s
        assert "output only" in s


# --- PII never enters memory ----------------------------------------------

class TestPIINotStored:
    def test_refused_question_is_not_recorded(self, monkeypatch):
        from rag import guard, pipeline

        conv = Conversation()
        r = pipeline.answer_question(
            "My PAN is ABCDE1234F, should I buy HDFC ELSS?",
            conversation=conv,
        )
        assert r.stage == "guard"
        assert r.answer.refused is True
        assert len(conv) == 0
        assert "ABCDE1234F" not in str(conv.history())

    def test_pii_gate_is_verbatim(self):
        from rag import guard

        assert guard.inspect_pii("pan ABCDE1234F") is not None
        assert guard.inspect_pii("What is the exit load?") is None


# --- transcript is bounded and cheap ---------------------------------------

class TestCost:
    def test_standalone_question_makes_no_llm_call(self, monkeypatch):
        def explode(_):
            raise AssertionError("rewrite should not have been called")

        monkeypatch.setattr(memory_mod.generator, "complete", explode)
        conv = Conversation()
        conv.record_exchange("What is the exit load of HDFC Small Cap Fund?", "1%")
        text, rewritten = conv.resolve(
            "What is the AUM of HDFC Large Cap Fund?")
        assert rewritten is False


class TestSubjects:
    def test_finds_scheme_in_transcript(self):
        conv = Conversation()
        conv.record_exchange("Tell me about HDFC Small Cap Fund", "ok")
        subjects = conv.subjects()
        assert any("Small Cap" in s for s in subjects)

    def test_empty_transcript_has_no_subjects(self):
        assert Conversation().subjects() == []


# --- dangling reference (no context to resolve against) --------------------
#
# "what about its fees?" with an empty transcript is under-specified, not
# off-topic. Telling the user it is off-topic misdescribes the problem.

class TestDanglingReference:
    def test_detected_when_no_scheme_named(self):
        from rag import guard

        for q in ("what about its fees?", "and its benchmark?", "is that good?",
                  "how about it?"):
            assert guard.has_dangling_reference(q) is True, q

    def test_not_detected_when_scheme_named(self):
        from rag import guard

        assert guard.has_dangling_reference(
            "What is the expense ratio of HDFC Small Cap Fund?") is False

    def test_mutual_fund_vocabulary_alone_is_not_enough(self):
        """Regression: 'and its benchmark?' contains a mutual-fund term.

        The first version tested for mutual-fund vocabulary, so this slipped
        through to a misleading "not covered by the official sources" instead
        of asking which fund was meant.
        """
        from rag import guard

        assert guard.is_mutual_fund_question("and its benchmark?") is True
        assert guard.has_dangling_reference("and its benchmark?") is True

    def test_names_corpus_scheme(self):
        from rag import guard

        assert guard.names_corpus_scheme("tell me about HDFC Small Cap Fund") is True
        assert guard.names_corpus_scheme("what about its fees?") is False

    def test_refused_with_dangling_message_when_no_context(self):
        from rag import guard

        v = guard.inspect("what about its fees?", has_context=False)
        assert v.action == guard.ACTION_REFUSE_DANGLING
        assert "which scheme" in v.message.lower()

    def test_allowed_once_context_exists(self):
        from rag import guard

        v = guard.inspect("and its benchmark?", has_context=True)
        assert v.action == guard.ACTION_ANSWER

    def test_advice_still_wins_over_dangling(self):
        """A PRD advice refusal must not be downgraded to 'name the fund'."""
        from rag import guard

        v = guard.inspect("Should I buy it?", has_context=False)
        assert v.action == guard.ACTION_REFUSE_ADVICE
        assert v.link, "advice refusal must keep its education link"

    def test_offtopic_still_wins_over_dangling(self):
        from rag import guard

        v = guard.inspect("What is the weather?", has_context=False)
        assert v.action == guard.ACTION_REFUSE_OFFTOPIC
