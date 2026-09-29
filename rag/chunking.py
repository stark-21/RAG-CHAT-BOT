"""Phase 2 - Chunking.

Strategy: LABEL-ANCHORED, CONTEXT-WINDOWED chunks.

Why this strategy (see docs/chunking_strategy.md for the full write-up):

The corpus is a set of scheme pages whose useful content is organised as
short LABEL -> VALUE blocks ("Exit load" -> "Exit load of 1% if redeemed
within 1 year"; "Fund benchmark" -> "Nifty 100 TRI"), surrounded by a large
amount of site navigation chrome.

  * Fixed-size splitting destroys these pairs, because the label and its value
    get separated - and the value alone ("1% if redeemed within 1 year") is
    unanswerable without the label.
  * Fixed-size splitting also produces chunks that mix unrelated facts, so
    retrieval returns noise alongside the answer.

So we split on detected label anchors instead, keep each label glued to the
lines that follow it, and cap each chunk at the configured character budget
with a small overlap. A short header ("Scheme | Section") is prepended to the
text that gets embedded, because all five schemes mention "exit load" and
without that header the vector cannot tell them apart.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import config

# Labels that introduce a fact. A line is treated as a section anchor when it
# is short, title-ish, and matches one of these patterns.
_LABEL_PATTERNS = (
    r"^expense ratio", r"^exit load", r"^entry load", r"^load$", r"^loads",
    r"^min\.? for sip", r"^minimum (sip|investment|amount|lumpsum)",
    r"^sip$", r"^lumpsum", r"^lock[- ]?in", r"^benchmark", r"^fund benchmark",
    r"^aum$", r"^total aum", r"^fund size", r"^nav\b", r"^riskometer",
    r"^risk[- ]?o?meter", r"^risk\b", r"^scheme type", r"^category",
    r"^fund category", r"^plan\b", r"^direct growth", r"^investment objective",
    r"^about\b", r"^fund manager", r"^tax", r"^tds\b", r"^capital gains",
    r"^statement", r"^how to download", r"^charges", r"^expense",
    r"^management fee", r"^other charges", r"^returns?\b", r"^performance\b",
    r"^holdings?\b", r"^portfolio\b", r"^top 10", r"^assets\b",
    r"^valuation", r"^sebi", r"^amfi", r"^disclaimer", r"^risk\b",
)

_LABEL_RE = re.compile("|".join(_LABEL_PATTERNS), re.IGNORECASE)

# Sentences that must never be emitted: they invite a performance claim, which
# the brief forbids (PRD F7).
_FORBIDDEN_PREFIXES = (
    r"^returns?\b", r"^annualised returns?$", r"^absolute returns?$",
    r"^historic returns?$", r"^fund returns?$", r"^performance\b",
    r"^1 ?y$", r"^3 ?y$", r"^5 ?y$", r"^since inception$",
)

_FORBIDDEN_RE = re.compile("|".join(_FORBIDDEN_PREFIXES), re.IGNORECASE)

# Sections that are pure data tables. Excluded because no in-scope question
# (PRD section 6) asks for holdings or portfolio weights, and they crowd out
# the facts that are asked about: measured at 201 of 285 chunks (70%) on the
# first run. See docs/chunking_strategy.md.
_TABLE_SECTIONS = (
    r"^assets?$", r"^holdings?$", r"^portfolio$", r"^top ?10", r"^sector",
    r"^fund size\s*\(?cr", r"^category average", r"^other details$",
    r"^return calculator", r"^tds calculator", r"^sip calculator",
    r"^fund ?composition", r"^debt\b", r"^equity\b", r"^asset allocation",
    r"^company\b", r"^nav history", r"^vintage", r"^returns?$",
)

_TABLE_SECTION_RE = re.compile("|".join(_TABLE_SECTIONS), re.IGNORECASE)

# Lines that are table cells rather than prose: bare numbers, percentages,
# currency, dates, asset-class markers, and rules. Dropping these is what
# un-fragments a fact - the first run produced orphans like "08 May 2015"
# and "01 Jan 2013" sitting between an "Exit load" label and its value.
_TABLE_CELL_RES = (
    r"^[\d,\.]+$",
    r"^[\d\.]+\s*%$",
    r"^rs\.?\s*[\d,\.]+\s*(cr|lakh|crore|rs\.?)?$",
    r"^\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}$",
    r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$",
    r"^(equity|debt|cash|cash & equivalent|others?)$",
    r"^[-=_*~•·|\s]+$",
    r"^\d+(st|nd|rd|th)$",
    # Contact-detail VALUES. The registrar block ships label/value pairs, and
    # stripping only the labels leaves phone/email/address values that bury
    # the fact the label introduced (e.g. the AUM figure).
    r"^\+?[\d\s\-–()]{8,}$",                    # phone
    r"^\S*@\S*$",                               # email, incl. obfuscated
    r"^\[?e-?mail",                             # "[email protected]"
    r"^https?://", r"^www\.",                   # urls
    r"^[\"“].*[\"”,]?$",                        # quoted postal address
)


def _is_table_cell(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    return any(re.match(p, stripped, re.IGNORECASE) for p in _TABLE_CELL_RES)


def is_table_section(section: str) -> bool:
    """True when a section label introduces a data table we do not index."""
    return bool(_TABLE_SECTION_RE.match(section.strip()))


@dataclass
class Chunk:
    """One retrievable unit."""

    chunk_id: str
    text: str
    embed_text: str
    scheme: str
    scheme_short: str
    category: str
    doc_type: str
    section: str
    source_url: str
    source_tier: str
    fetched_at: str
    chunk_idx: int
    char_count: int = 0
    metadata: dict = field(default_factory=dict)

    def to_metadata(self) -> dict:
        """Chroma-safe metadata (flat, scalar values only)."""
        return {
            "scheme": self.scheme,
            "scheme_short": self.scheme_short,
            "category": self.category,
            "doc_type": self.doc_type,
            "section": self.section,
            "source_url": self.source_url,
            "source_tier": self.source_tier,
            "fetched_at": self.fetched_at,
            "chunk_idx": self.chunk_idx,
            "char_count": self.char_count,
        }


def is_label(line: str, max_words: int = 8) -> bool:
    """True when a line looks like a section/fact label rather than prose."""
    if not line or len(line) > 70:
        return False
    if len(line.split()) > max_words:
        return False
    return bool(_LABEL_RE.match(line.strip()))


def is_forbidden(line: str) -> bool:
    """True for lines that would lead to a return/performance claim."""
    return bool(_FORBIDDEN_RE.match(line.strip()))


def build_segments(text: str) -> list[tuple[str, list[str]]]:
    """Split cleaned text into (section_label, lines) segments."""
    segments: list[tuple[str, list[str]]] = []
    current_label = "Overview"
    current: list[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if is_label(line):
            if current:
                segments.append((current_label, current))
            current_label = line
            current = []
            continue
        current.append(line)

    if current:
        segments.append((current_label, current))

    return segments


def _pack(lines: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    """Greedily pack lines into <=max_chars pieces, repeating overlap_chars of
    tail context at the start of the next piece."""
    pieces: list[str] = []
    current: list[str] = []
    length = 0
    tail: str = ""

    for line in lines:
        addition = len(line) + 1
        if current and length + addition > max_chars:
            pieces.append("\n".join(current))
            # Carry a slice of the tail forward so a fact split across the
            # boundary stays retrievable from either side.
            if overlap_chars > 0 and tail:
                carry = tail[-overlap_chars:]
                current = [carry]
                length = len(carry)
            else:
                current = []
                length = 0
        current.append(line)
        length += addition
        tail = "\n".join(current)

    if current:
        pieces.append("\n".join(current))
    return pieces


def chunk_documents(docs) -> list[Chunk]:
    """Turn cleaned RawDoc objects into metadata-rich chunks."""
    from sources import CATEGORY_BY_SCHEME, SOURCE_TIER_BY_SCHEME

    max_chars = config.CONFIG.chunk_size
    overlap_chars = config.CONFIG.chunk_overlap

    chunks: list[Chunk] = []
    idx = 0

    for doc in docs:
        if not doc.text.strip():
            continue

        category = CATEGORY_BY_SCHEME.get(doc.scheme, "unknown")
        tier = SOURCE_TIER_BY_SCHEME.get(doc.scheme, "unknown")
        scheme_short = doc.scheme.replace(" - Direct Growth", "")

        for section, lines in build_segments(doc.text):
            if is_table_section(section):
                continue

            kept = [
                ln for ln in lines
                if not is_forbidden(ln) and not _is_table_cell(ln)
            ]
            if not kept:
                continue

            for piece in _pack(kept, max_chars, overlap_chars):
                body = piece.strip()
                if len(body) < 25:
                    continue

                header = f"[{scheme_short} | {section}]"
                embed_text = f"{header}\n{body}"

                chunk_id = f"{scheme_short}__{doc.doc_type}__{idx:04d}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=body,
                        embed_text=embed_text,
                        scheme=doc.scheme,
                        scheme_short=scheme_short,
                        category=category,
                        doc_type=doc.doc_type,
                        section=section,
                        source_url=doc.source_url,
                        source_tier=tier,
                        fetched_at=doc.fetched_at,
                        chunk_idx=idx,
                        char_count=len(embed_text),
                    )
                )
                idx += 1

    return chunks


def format_chunks_report(chunks: list[Chunk]) -> str:
    """Render chunks.txt: numbered, with source and character count."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("CHUNK DUMP - Mutual Fund FAQ Assistant")
    lines.append("Strategy: label-anchored, context-windowed")
    lines.append(f"Total chunks: {len(chunks)}")
    lines.append(f"Chunk char budget: {config.CONFIG.chunk_size} "
                 f"(overlap {config.CONFIG.chunk_overlap})")
    lines.append("=" * 78)

    for chunk in chunks:
        lines.append("")
        lines.append("-" * 78)
        lines.append(f"[{chunk.chunk_idx + 1:04d}] {chunk.chunk_id}")
        lines.append(f"  source   : {chunk.source_url}")
        lines.append(f"  scheme   : {chunk.scheme}  [{chunk.category}]")
        lines.append(f"  section  : {chunk.section}")
        lines.append(f"  doc_type : {chunk.doc_type}   tier: {chunk.source_tier}")
        lines.append(f"  fetched  : {chunk.fetched_at}")
        lines.append(f"  chars    : {chunk.char_count}")
        lines.append("-" * 78)
        lines.append(chunk.text)

    lines.append("")
    lines.append("=" * 78)
    lines.append(f"END - {len(chunks)} chunks")
    lines.append("=" * 78)
    return "\n".join(lines)


def write_chunks_report(chunks: list[Chunk], path=None) -> str:
    """Write the inspectable chunk dump. Returns the path written."""
    target = path or config.CONFIG.chunks_file
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(format_chunks_report(chunks), encoding="utf-8")
    return str(target)
