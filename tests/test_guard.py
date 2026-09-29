"""Tests for Phase 4 guardrails.

Verifies PRD success criteria F4 (all 5 opinion questions refused) and
F5 (all 3 PII probes rejected, nothing stored), plus no false positives on the
fact questions the bot must answer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag import guard
from rag.guard import (
    ACTION_ANSWER,
    ACTION_NO_CONTEXT,
    ACTION_REFUSE_ADVICE,
    ACTION_REFUSE_OFFTOPIC,
    ACTION_REFUSE_PII,
    check_grounding,
    detect_advice,
    detect_pii,
    inspect,
)

CASES = json.loads((ROOT / "evals" / "guard_cases.json").read_text(encoding="utf-8"))


class TestPiiDetection:
    @pytest.mark.parametrize("name,text", [
        ("pan", "My PAN is ABCDE1234F"),
        ("aadhaar", "Aadhaar 2345 6789 0123"),
        ("account_number", "account number 00123456789"),
        ("email", "reach me at ravi.kumar@example.com"),
        ("phone", "call 9876543210"),
        ("otp", "my otp is 482913"),
    ])
    def test_detects(self, name, text):
        assert name in detect_pii(text)

    def test_clean_question_has_no_pii(self):
        assert detect_pii("What is the exit load on HDFC Small Cap Fund?") == []

    def test_pii_values_are_never_echoed(self):
        verdict = inspect("My PAN is ABCDE1234F, tell me about ELSS")
        assert verdict.action == ACTION_REFUSE_PII
        assert "ABCDE1234F" not in verdict.message
        assert "ABCDE1234F" not in verdict.reason


class TestAdviceDetection:
    @pytest.mark.parametrize("text", [
        "Should I buy HDFC ELSS Tax Saver Fund?",
        "Which is better for me?",
        "How should I split my portfolio?",
        "What is the best fund for a beginner?",
        "Recommend a scheme",
        "Is this worth investing?",
        "When is a good time to enter a small cap fund?",
    ])
    def test_detects(self, text):
        assert detect_advice(text) is not None

    @pytest.mark.parametrize("text", [
        "What is the expense ratio of HDFC Large Cap Fund?",
        "Is there an exit load on HDFC Small Cap Fund?",
        "How do I download a capital-gains statement from the AMC?",
        "What is the lock-in period for HDFC ELSS Tax Saver Fund?",
        "Is HDFC ELSS Tax Saver Fund good for tax saving?",
    ])
    def test_does_not_flag_factual_questions(self, text):
        assert detect_advice(text) is None


class TestOffTopic:
    @pytest.mark.parametrize("text", [
        "What is the weather in Mumbai today?",
        "Give me a recipe for pasta",
        "Who won the cricket match yesterday?",
    ])
    def test_detects(self, text):
        assert inspect(text).action == ACTION_REFUSE_OFFTOPIC

    def test_mutual_fund_question_not_flagged(self):
        assert guard.is_mutual_fund_question("What is the NAV of HDFC Large Cap?")
        assert not guard.is_mutual_fund_question("What is the weather?")


class TestPrdOpinionSet:
    """PRD F4: all 5 opinion questions refused with an educational link."""

    def test_all_five_refused(self):
        cases = CASES["opinion"]
        assert len(cases) == 5
        for case in cases:
            verdict = inspect(case["query"])
            assert verdict.action == ACTION_REFUSE_ADVICE, (
                f"not refused: {case['query']!r} -> {verdict.action}"
            )

    def test_refusals_carry_educational_link(self):
        for case in CASES["opinion"]:
            verdict = inspect(case["query"])
            assert verdict.link.startswith("http"), f"no link: {case['query']!r}"

    def test_refusals_contain_no_recommendation(self):
        for case in CASES["opinion"]:
            verdict = inspect(case["query"])
            assert "can't recommend" in verdict.message
            assert "should buy" not in verdict.message.lower()


class TestPrdPiiSet:
    """PRD F5: all 3 PII probes rejected."""

    def test_all_three_rejected(self):
        cases = CASES["pii"]
        assert len(cases) == 3
        for case in cases:
            verdict = inspect(case["query"])
            assert verdict.action == ACTION_REFUSE_PII, (
                f"not rejected: {case['query']!r} -> {verdict.action}"
            )


class TestNoFalsePositives:
    def test_answerable_questions_pass_the_guard(self):
        for case in CASES["answerable"]:
            verdict = inspect(case["query"])
            assert verdict.action == ACTION_ANSWER, (
                f"wrongly blocked: {case['query']!r} -> {verdict.action} "
                f"({verdict.reason})"
            )

    def test_should_retrieve_only_for_answer(self):
        assert inspect("What is the exit load?").should_retrieve
        assert not inspect("Should I buy this?").should_retrieve
        assert not inspect("My PAN is ABCDE1234F").should_retrieve


class TestGroundingGate:
    """The 'I don't know' gate."""

    def test_no_retrieval_is_no_context(self):
        verdict = check_grounding(None)
        assert verdict.action == ACTION_NO_CONTEXT
        assert "don't know" in verdict.message

    def test_weak_retrieval_is_no_context(self):
        verdict = check_grounding(0.92)
        assert verdict.action == ACTION_NO_CONTEXT
        assert "don't know" in verdict.message

    def test_strong_retrieval_allows_answer(self):
        assert check_grounding(0.16).action == ACTION_ANSWER

    def test_threshold_is_respected(self):
        # Cosine distance: lower is closer, so a HIGHER limit is more
        # permissive. A 0.60 match passes a 0.65 limit and fails a 0.55 one.
        assert check_grounding(0.60, threshold=0.65).action == ACTION_ANSWER
        assert check_grounding(0.60, threshold=0.55).action == ACTION_NO_CONTEXT

    def test_empty_question(self):
        assert inspect("").action == ACTION_NO_CONTEXT


class TestConceptCoverage:
    """Distance alone cannot detect a missing fact. These tests prove that.

    The measured case: "What is the lock-in period for HDFC ELSS?" retrieves
    the ELSS *Tax implication* chunk at distance 0.320, which is inside the
    0.65 threshold, so a distance-only gate would answer. That chunk discusses
    STCG/LTCG taxation and never mentions the lock-in.
    """

    ELSS_TAX_CHUNK = (
        "If you redeem within one year, returns are taxed at 20%. "
        "If you redeem after one year, returns exceeding Rs 1.25 lakh in a "
        "financial year are taxed at 12.5%."
    )

    def test_lexically_similar_but_unrelated_chunk_is_refused(self):
        verdict = check_grounding(
            0.320,
            "What is the lock-in period for HDFC ELSS Tax Saver Fund?",
            [self.ELSS_TAX_CHUNK],
        )
        assert verdict.action == ACTION_NO_CONTEXT
        assert verdict.rule == "concept_not_covered"
        assert "lock-in" in verdict.reason

    def test_riskometer_gap_is_refused(self):
        verdict = check_grounding(
            0.232,
            "What is the riskometer level and benchmark of HDFC Balanced Advantage Fund?",
            ["The fund benchmark is Nifty 100 Total Return Index."],
        )
        assert verdict.action == ACTION_NO_CONTEXT
        assert "riskometer" in verdict.reason

    def test_concept_present_allows_answer(self):
        verdict = check_grounding(
            0.30,
            "What is the lock-in period for HDFC ELSS Tax Saver Fund?",
            ["ELSS lock-in period is 3 years from the date of investment."],
        )
        assert verdict.action == ACTION_ANSWER

    def test_benchmark_present_allows_answer(self):
        verdict = check_grounding(
            0.30,
            "What is the benchmark of HDFC Large Cap Fund?",
            ["Fund benchmark: NIFTY 100 Total Return Index"],
        )
        assert verdict.action == ACTION_ANSWER

    def test_missing_concepts_helper(self):
        assert guard.missing_concepts("What is the exit load?", []) == []
        assert guard.missing_concepts("lock-in?", ["no mention here"]) == ["lock-in"]
        assert guard.missing_concepts("lock-in?", ["lock-in is 3 years"]) == []

    def test_section_label_counts_as_context(self):
        """The stored body may not repeat its own label.

        The Expense ratio chunk body reads 'A fee payable to a mutual fund
        house...' with no occurrence of the phrase. The label only exists in
        the embed header, so the haystack must include it.
        """
        body = ("A fee payable to a mutual fund house for managing your mutual "
                "fund investments.")
        assert "expense ratio" not in body.lower()

        without = check_grounding(
            0.17, "What is the expense ratio of HDFC Large Cap Fund?", [body]
        )
        assert without.action == ACTION_NO_CONTEXT

        with_label = check_grounding(
            0.17,
            "What is the expense ratio of HDFC Large Cap Fund?",
            [f"Expense ratio\n{body}"],
        )
        assert with_label.action == ACTION_ANSWER

    def test_scheme_name_does_not_trigger_tax_concept(self):
        """'HDFC ELSS Tax Saver Fund' must not count as asking about tax."""
        blocks = ["Exit load\nExit load of 1% if redeemed within 1 year"]
        verdict = check_grounding(
            0.19,
            "Is there an exit load on HDFC ELSS Tax Saver Fund?",
            blocks,
        )
        assert verdict.action == ACTION_ANSWER


class TestNoPersistence:
    """PRD F5: nothing about a PII-bearing message is written to disk."""

    def test_guard_writes_no_files(self, tmp_path):
        import config

        before = {p: p.stat().st_mtime for p in ROOT.rglob("*.txt")
                  if ".venv" not in p.parts and "data" not in p.parts}
        for case in CASES["pii"]:
            inspect(case["query"])
        after = {p: p.stat().st_mtime for p in ROOT.rglob("*.txt")
                 if ".venv" not in p.parts and "data" not in p.parts}
        assert before == after

    def test_guard_module_writes_nothing(self):
        """guard.py must contain no file writes."""
        source = (ROOT / "rag" / "guard.py").read_text(encoding="utf-8")
        for forbidden in ("open(", "write_text", "Path(", "sqlite", "json.dump"):
            assert forbidden not in source, f"guard.py must not use {forbidden}"
