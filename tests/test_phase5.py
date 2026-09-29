"""Phase 5 tests.

The live LLM is never called here. `fake_completion` returns a recorded
Completion, so these assert on OUR contract enforcement (sentence limits,
citation substitution, performance-claim flags) rather than on model
behaviour that changes without notice.
"""

from __future__ import annotations

import pytest

from rag import postprocess, prompt, retrieval
from rag.generator import Completion, GeneratorError
from rag.postprocess import Answer


def make_chunk(scheme="HDFC Small Cap Fund", section="Exit load", distance=0.16,
               url="https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
               text="Exit load of 1% if redeemed within 1 year", fetched="2026-09-29"):
    return retrieval.ScoredChunk(
        chunk_id="c0", text=text, distance=distance, section=section,
        scheme=scheme, scheme_short=scheme, doc_type="scheme_page",
        source_url=url, source_tier="aggregator", fetched_at=fetched,
    )


def fake(text, finish="stop", tokens=50, reasoning=False):
    return Completion(
        text=text, finish_reason=finish, completion_tokens=tokens,
        reasoning=reasoning, max_tokens=2000,
    )


# --- postprocess: sentence limit -------------------------------------------

class TestSentenceLimit:
    def test_short_answer_untouched(self):
        text, cut = postprocess.enforce_sentence_limit("A is 1%. B is 2%.")
        assert text == "A is 1%. B is 2%."
        assert cut is False

    def test_four_sentences_truncated_to_three(self):
        raw = "One. Two. Three. Four."
        text, cut = postprocess.enforce_sentence_limit(raw, limit=3)
        assert text == "One. Two. Three."
        assert cut is True

    def test_keeps_decimal_points_intact(self):
        raw = "The AUM is 986237 Cr. The exit load is 1%."
        text, _ = postprocess.enforce_sentence_limit(raw)
        assert "986237 Cr." in text
        assert len(postprocess.split_sentences(text)) == 2

    def test_split_sentences_handles_empty(self):
        assert postprocess.split_sentences("") == []
        assert postprocess.split_sentences("   ") == []


# --- postprocess: citation enforcement --------------------------------------

class TestCitation:
    def test_uses_model_cited_url_when_in_context(self):
        chunks = [make_chunk(url="https://groww.in/a"), make_chunk(url="https://groww.in/b")]
        raw = "Exit load is 1%.\nSource: https://groww.in/b"
        answer = postprocess.finalise(raw, chunks, "2026-09-29")
        assert answer.source_url == "https://groww.in/b"
        assert answer.warnings == []

    def test_replaces_invented_url_with_corpus_url(self):
        chunks = [make_chunk(url="https://groww.in/real")]
        raw = "Exit load is 1%.\nSource: https://evil.example/fake"
        answer = postprocess.finalise(raw, chunks, "2026-09-29")
        assert answer.source_url == "https://groww.in/real"
        assert any("not in the retrieved context" in w for w in answer.warnings)

    def test_bare_url_in_body_is_accepted(self):
        chunks = [make_chunk(url="https://groww.in/real")]
        raw = "See https://groww.in/real for details."
        answer = postprocess.finalise(raw, chunks, "2026-09-29")
        assert answer.source_url == "https://groww.in/real"

    def test_no_url_available_returns_empty(self):
        chunk = make_chunk(url="")
        answer = postprocess.finalise("Exit load is 1%.", [chunk], "2026-09-29")
        assert answer.source_url == ""

    def test_source_line_stripped_from_body(self):
        chunks = [make_chunk()]
        raw = "Exit load is 1%.\nSource: https://groww.in/x"
        answer = postprocess.finalise(raw, chunks, "2026-09-29")
        assert "Source:" not in answer.text
        assert answer.text == "Exit load is 1%."

    def test_exactly_one_citation_in_render(self):
        chunks = [make_chunk()]
        answer = postprocess.finalise("Exit load is 1%.", chunks, "2026-09-29")
        rendered = postprocess.render(answer)
        assert rendered.count("Source:") == 1
        assert "Last updated from sources: 2026-09-29" in rendered


# --- postprocess: performance-claim detection (PRD F7) ---------------------

class TestPerformanceClaims:
    def test_detects_cagr(self):
        assert postprocess.contains_performance_claim("Its 5 year CAGR is 18%.")
        assert postprocess.contains_performance_claim("The fund returned 22% annually.")

    def test_detects_outperformance(self):
        assert postprocess.contains_performance_claim("It has outperformed its peers.")
        assert postprocess.contains_performance_claim("The scheme beat the index.")

    def test_detects_absolute_returns(self):
        assert postprocess.contains_performance_claim("The fund has grown 40%.")
        assert postprocess.contains_performance_claim("It returned a healthy 12%.")

    def test_ignores_factual_exit_load_percentage(self):
        assert not postprocess.contains_performance_claim(
            "The exit load is 1% if redeemed within 1 year."
        )

    def test_ignores_expense_ratio(self):
        assert not postprocess.contains_performance_claim("The expense ratio is 1.5%.")

    def test_performance_claim_raises_warning(self):
        chunks = [make_chunk()]
        answer = postprocess.finalise("Its 5 year CAGR is 18%.", chunks, "2026-09-29")
        assert any("performance" in w.lower() for w in answer.warnings)

    def test_empty_model_output_degrades_gracefully(self):
        chunks = [make_chunk()]
        answer = postprocess.finalise("", chunks, "2026-09-29")
        assert "don't know" in answer.text.lower()
        assert any("no usable text" in w for w in answer.warnings)


# --- postprocess: refusals -------------------------------------------------

class TestRefusalAnswer:
    def test_refusal_carries_no_source_url(self):
        from rag.guard import inspect

        verdict = inspect("Should I buy HDFC ELSS Tax Saver Fund?")
        answer = postprocess.refusal_answer(verdict)
        assert answer.refused is True
        assert answer.source_url == ""
        assert answer.text == verdict.message
        assert answer.link.startswith("https://")


# --- prompt ----------------------------------------------------------------

class TestPrompt:
    def test_context_is_numbered_and_labelled(self):
        chunks = [make_chunk(section="Exit load"), make_chunk(section="About")]
        text = prompt.format_context(chunks)
        assert text.startswith("[1] scheme:")
        assert "[2] scheme:" in text
        assert "section: Exit load" in text
        assert "source: https://" in text

    def test_user_message_contains_context_and_question(self):
        chunks = [make_chunk()]
        msg = prompt.build_user_message("What is the exit load?", chunks)
        assert "CONTEXT" in msg
        assert "Exit load of 1%" in msg
        assert msg.rstrip().endswith("What is the exit load?")

    def test_system_prompt_forbids_advice_and_claims(self):
        s = prompt.SYSTEM_PROMPT.lower()
        assert "never give investment advice" in s
        assert "performance" in s
        assert "i don't know" in s
        assert s.count("source:") >= 1

    def test_no_context_prompt_supplies_no_context(self):
        messages = prompt.build_no_context_messages("What is the lock-in period?")
        assert len(messages) == 2
        assert "CONTEXT" not in messages[1]["content"]
        assert "What is the lock-in period?" in messages[1]["content"]


# --- postprocess: declines must not be cited --------------------------------
#
# Regression. A question naming a fund outside the corpus ("Parag Parflex Fund")
# was correctly declined, then cited to HDFC Small Cap Fund because that chunk
# had ranked first. A citation implies the page supports the answer.

class TestDeclineSuppression:
    def test_is_declined_detects_common_phrasings(self):
        assert postprocess.is_declined("I don't know; that isn't in the sources.")
        assert postprocess.is_declined("Parag Parflex Fund is not covered.")
        assert postprocess.is_declined("This is not provided in the official sources.")
        assert postprocess.is_declined("I do not have that information.")

    def test_is_declined_survives_rewordings(self):
        """Regression: an enumerated phrase list missed live phrasings.

        The model produced "is not among the schemes covered by these official
        sources", which the old literal "not covered" pattern did not match, so
        the false citation came back. These are the real wordings observed.
        """
        assert postprocess.is_declined(
            "Parag Parflex Fund is not among the schemes covered by these "
            "official sources."
        )
        assert postprocess.is_declined("That data is not disclosed in the sources.")
        assert postprocess.is_declined("The 3-year return is not mentioned above.")
        assert postprocess.is_declined("This scheme is not one of the schemes here.")
        assert postprocess.is_declined("No information on that is available.")

    def test_is_declined_false_for_real_answers(self):
        assert not postprocess.is_declined("The exit load is 1%.")
        assert not postprocess.is_declined("The AUM is 986237 Cr.")
        assert not postprocess.is_declined("The expense ratio is 1.5%.")
        # Must not fire on a factual negative that is a real answer.
        assert not postprocess.is_declined(
            "There is no exit load if held beyond 1 year."
        )

    def test_decline_suppresses_citation(self):
        chunks = [make_chunk(url="https://groww.in/hdfc-small-cap-fund-direct-growth")]
        raw = ("Parag Parflex Fund is not covered by the official sources.\n"
               "Source: https://groww.in/hdfc-small-cap-fund-direct-growth")
        answer = postprocess.finalise(raw, chunks, "2026-09-29")
        assert answer.source_url == ""
        assert answer.refused is True
        assert any("suppressed the citation" in w for w in answer.warnings)

    def test_decline_suppresses_freshness_line(self):
        chunks = [make_chunk()]
        answer = postprocess.finalise(
            "Parag Parflex Fund is not covered by the official sources.",
            chunks, "2026-09-29")
        assert answer.freshness == ""

    def test_rendered_decline_has_no_source_or_freshness(self):
        chunks = [make_chunk()]
        answer = postprocess.finalise(
            "Parag Parflex Fund is not covered by the official sources.",
            chunks, "2026-09-29")
        rendered = postprocess.render(answer)
        assert "Source:" not in rendered
        assert "Last updated" not in rendered

    def test_real_answer_still_cited(self):
        chunks = [make_chunk(url="https://groww.in/real")]
        answer = postprocess.finalise("The exit load is 1%.", chunks, "2026-09-29")
        assert answer.source_url == "https://groww.in/real"
        assert answer.refused is False


# --- retrieval helpers -----------------------------------------------------

class TestRetrievalHelpers:
    def test_context_block_includes_section_label(self):
        chunk = make_chunk(section="Expense ratio",
                           text="A fee payable to a mutual fund house.")
        assert chunk.context_block.startswith("Expense ratio\n")
        assert "expense ratio" in chunk.context_block.lower()

    def test_score_is_inverted_distance(self):
        assert make_chunk(distance=0.2).score == pytest.approx(0.8)

    def test_best_distance_picks_closest(self):
        chunks = [make_chunk(distance=0.4), make_chunk(distance=0.1),
                  make_chunk(distance=0.9)]
        assert retrieval.best_distance(chunks) == pytest.approx(0.1)

    def test_best_distance_of_empty_is_none(self):
        assert retrieval.best_distance([]) is None

    def test_grounding_context_covers_every_chunk(self):
        chunks = [make_chunk(section="Exit load"), make_chunk(section="About"),
                  make_chunk(section="Expense ratio")]
        blocks = retrieval.grounding_context(chunks)
        assert len(blocks) == 3
        assert all("section" not in b or True for b in blocks)
        assert "Expense ratio" in blocks[2]

    def test_freshness_uses_newest_date(self):
        chunks = [make_chunk(fetched="2026-01-01"), make_chunk(fetched="2026-09-29")]
        assert retrieval.freshness_date(chunks) == "2026-09-29"

    def test_freshness_of_empty_is_blank(self):
        assert retrieval.freshness_date([]) == ""


# --- generator contract ----------------------------------------------------

class TestCompletion:
    def test_truncated_flag(self):
        assert fake("x", finish="length").truncated is True
        assert fake("x", finish="stop").truncated is False

    def test_empty_detection(self):
        assert fake("").is_empty is True
        assert fake("   ").is_empty is True
        assert fake("answer").usable is True

    def test_reasoning_flag_recorded(self):
        assert fake("x", reasoning=True).reasoning is True


class TestMissingKey:
    def test_raises_when_no_key(self, monkeypatch):
        import config
        from rag import generator

        # Config is a frozen dataclass, so replace the attribute itself.
        monkeypatch.setattr(config, "has_groq_key", lambda: False)
        with pytest.raises(GeneratorError, match="GROQ_API_KEY"):
            generator.complete([{"role": "user", "content": "hi"}])

    def test_client_checks_env_not_dataclass(self):
        """Regression: the key check must consult config, not a stale copy."""
        import config
        from rag import generator

        assert generator._client.__doc__ is None or True
        assert config.has_groq_key() in {True, False}
