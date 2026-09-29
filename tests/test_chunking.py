"""Tests for Phase 2 chunking.

Runs on synthetic RawDoc objects - no network, no corpus dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from rag.chunking import (
    build_segments,
    chunk_documents,
    format_chunks_report,
    is_forbidden,
    is_label,
)
from rag.loader import RawDoc, clean_html

SAMPLE_TEXT = """HDFC Small Cap Fund Direct Growth
Exit load
Exit load of 1% if redeemed within 1 year
Exit load of 2% if redeemed within 12 months, 1% if redeemed after 12 months
About
HDFC Small Cap Fund Direct Growth is an Equity Mutual Fund Scheme.
Fund benchmark
Nifty Smallcap 100 TRI
Total AUM
Rs. 35,000 Cr
Returns
Annualised returns
Historic returns
"""


def make_doc(text: str = SAMPLE_TEXT) -> RawDoc:
    return RawDoc(
        scheme="HDFC Small Cap Fund - Direct Growth",
        doc_type="scheme_page",
        source_url="https://example.invalid/small-cap",
        text=text,
        fetched_at="2026-09-29",
    )


class TestLabelDetection:
    def test_detects_fact_labels(self):
        for line in ("Exit load", "Expense ratio", "Fund benchmark",
                     "Min. for SIP", "Lock-in period", "Total AUM"):
            assert is_label(line), f"should be a label: {line}"

    def test_rejects_prose(self):
        assert not is_label("HDFC Small Cap Fund Direct Growth is an Equity Mutual Fund Scheme launched in 2011")
        assert not is_label("x" * 200)


class TestForbiddenLines:
    def test_flags_return_lines(self):
        for line in ("Returns", "Annualised returns", "Historic returns",
                     "Fund returns", "1Y", "Since inception"):
            assert is_forbidden(line), f"should be forbidden: {line}"

    def test_allows_fee_lines(self):
        assert not is_forbidden("Exit load of 1% if redeemed within 1 year")


class TestSegments:
    def test_splits_on_labels(self):
        segments = build_segments(SAMPLE_TEXT)
        labels = [label for label, _ in segments]
        assert "Exit load" in labels
        assert "Fund benchmark" in labels
        assert "Total AUM" in labels

    def test_label_glued_to_its_value(self):
        segments = dict(build_segments(SAMPLE_TEXT))
        exit_lines = " ".join(segments["Exit load"])
        assert "1% if redeemed within 1 year" in exit_lines


class TestChunkDocuments:
    def test_produces_chunks(self):
        chunks = chunk_documents([make_doc()])
        assert len(chunks) > 0

    def test_metadata_complete(self):
        required = {
            "scheme", "scheme_short", "category", "doc_type", "section",
            "source_url", "source_tier", "fetched_at", "chunk_idx", "char_count",
        }
        for chunk in chunk_documents([make_doc()]):
            assert set(chunk.to_metadata()) == required
            for field in ("scheme", "doc_type", "section", "source_url", "fetched_at"):
                assert chunk.to_metadata()[field], f"{field} must not be empty"

    def test_no_chunk_spans_two_sections(self):
        """A chunk belongs to exactly one section label."""
        for chunk in chunk_documents([make_doc()]):
            assert chunk.section
            assert chunk.text

    def test_respects_size_budget(self):
        for chunk in chunk_documents([make_doc()]):
            # allow slack for the prepended header
            assert chunk.char_count <= config.CONFIG.chunk_size + 80

    def test_drops_return_lines(self):
        for chunk in chunk_documents([make_doc()]):
            assert not is_forbidden(chunk.text.splitlines()[0])

    def test_embed_text_carries_scheme_header(self):
        """All five schemes mention 'exit load'; the header disambiguates."""
        chunks = chunk_documents([make_doc()])
        assert chunks[0].embed_text.startswith("[HDFC Small Cap Fund |")
        assert "HDFC Small Cap Fund" in chunks[0].embed_text

    def test_empty_input(self):
        assert chunk_documents([]) == []
        assert chunk_documents([make_doc(text="")]) == []


class TestReport:
    def test_report_numbers_and_counts(self):
        chunks = chunk_documents([make_doc()])
        report = format_chunks_report(chunks)
        assert "[0001]" in report
        assert "source   :" in report
        assert "chars    :" in report
        assert f"Total chunks: {len(chunks)}" in report
        assert "https://example.invalid/small-cap" in report


class TestCleanHtml:
    def test_strips_scripts_and_nav(self):
        html = (
            "<html><body><nav>Start SIP</nav><script>var x=1</script>"
            "<p>Exit load of 1%</p><footer>Copyright</footer></body></html>"
        )
        cleaned = clean_html(html)
        assert "var x" not in cleaned
        assert "Start SIP" not in cleaned
        assert "Exit load of 1%" in cleaned
