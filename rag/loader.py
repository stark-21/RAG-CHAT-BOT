"""Phase 2 - Loading.

Fetches each registered source page, strips the page chrome, and saves the
cleaned raw text to data/raw/ so the corpus is reproducible and inspectable
without re-hitting the network.

This is the ONLY module in rag/ that performs network I/O.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

import config

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)

# Structural tags whose entire subtree is never content.
_CHROME_TAGS = (
    "script", "style", "noscript", "svg", "nav", "header", "footer",
    "aside", "form", "button", "iframe", "canvas",
)

# Site-chrome lines that survive tag stripping (the app shell renders nav as
# plain text nodes). Matched case-insensitively against the whole line.
_BOILERPLATE_PATTERNS = (
    r"^invest in (stocks|mutual funds)", r"^start sip$", r"^sip calculator$",
    r"^filter funds based on", r"^know about amcs", r"^holdings? \(?\d*\)?$",
    r"^returns? on your", r"^intraday$", r"^demat account$", r"^market today$",
    r"^stock screener$", r"^screener$", r"^share market", r"^live news",
    r"^stock events$", r"^news$", r"^login$", r"^sign ?up$", r"^search$",
    r"^bond yields", r"^ipo$", r"^etf$", r"^small ?cap$", r"^large ?cap$",
    r"^view all$", r"^show more$", r"^see all$", r"^compare$", r"^add to compare$",
    r"^discover\b", r"^learn\b", r"^tools?\b", r"^calculators?\b", r"^reviews?$",
    r"^ratings?\b", r"^more\b", r"^home$", r"^about us$", r"^contact\b",
    r"^download app$", r"^google play$", r"^app store$", r"^follow us$",
    r"^terms\b", r"^privacy\b", r"^disclaimer\b", r"^copyright\b",
    r"^categories?\b", r"^amcs?\b", r"^all funds\b", r"^shortlist\b",
    r"^returns?$", r"^rankings?\b", r"^performance\b.*$",
    r"^bharat[- ]?bond", r"^index funds?$", r"^debt\b.*$", r"^hybrid\b.*$",
    r"^returns? and rankings$",
    r"^buy$", r"^sell$", r"^invest now$", r"^add money$", r"^lumpsum$",
    r"^more info$", r"^details$",
    r"^\s*share\s*$", r"^\s*copy\s*$", r"^tap to", r"^scroll\b",
    # Registrar / contact block. Not FAQ content, and it buries the AUM value
    # that the "Total AUM" label introduces.
    r"^date of incorporation$", r"^phone$", r"^e-?mail$", r"^website$",
    r"^launch date$", r"^address$", r"^custodian$",
    r"^registrar", r"^fund house$", r"^amc$", r"^trustee$",
)

_BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE_PATTERNS), re.IGNORECASE)

# The page renders its footer as plain text nodes, not inside a <footer> tag,
# so tag stripping cannot remove it. On the first Phase 2 run this produced
# 80 of 165 chunks of link-farm noise (futures, option chains, tickers).
# We cut the document at the footer breadcrumb instead of trying to
# enumerate every nav label.
#
# Only STRONG sentinels belong here. Bare ">" is the footer's breadcrumb
# separator and is the first such line in the document. Weak candidates like
# "Blog" or "Products" also appear in the top nav, above real content, and
# were measured truncating 80% of the corpus.
_FOOTER_SENTINELS = (
    r"^>$",
    r"^groww\.? all rights reserved",
)

_FOOTER_RE = re.compile("|".join(_FOOTER_SENTINELS), re.IGNORECASE)


@dataclass
class RawDoc:
    """Cleaned text for one source document."""

    scheme: str
    doc_type: str
    source_url: str
    text: str
    fetched_at: str
    snapshot_path: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.text)


def clean_html(html: str) -> str:
    """Strip chrome from raw HTML and normalise to plain text."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(_CHROME_TAGS):
        tag.decompose()

    text = soup.get_text("\n")

    lines: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t\xa0]+", " ", raw_line).strip()
        if not line:
            continue
        if _FOOTER_RE.match(line):
            # Everything from here on is site chrome.
            break
        if _BOILERPLATE_RE.match(line):
            continue
        lines.append(line)

    # Collapse runs of duplicate lines produced by JS-rendered panels.
    deduped: list[str] = []
    for line in lines:
        if not deduped or deduped[-1] != line:
            deduped.append(line)

    return "\n".join(deduped)


def snapshot_name(scheme: str, doc_type: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", scheme.lower()).strip("_")[:48]
    return f"{slug}__{doc_type}.txt"


def snapshot_path(scheme: str, doc_type: str) -> Path:
    return config.CONFIG.raw_dir / snapshot_name(scheme, doc_type)


def fetch(url: str, timeout: int = 30, retries: int = 3) -> str:
    """GET a page with retries. Raises on final failure."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                url, timeout=timeout, headers={"User-Agent": USER_AGENT}
            )
            response.raise_for_status()
            return response.text
        except Exception as exc:  # noqa: BLE001 - re-raised below
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def load_source(source, refresh: bool = False) -> RawDoc:
    """Load one SourceDoc, preferring a saved snapshot unless refresh=True."""
    today = date.today().isoformat()
    snap = snapshot_path(source.scheme, source.doc_type)
    warnings: list[str] = []

    if snap.exists() and not refresh:
        text = snap.read_text(encoding="utf-8")
        fetched_at = today
        warnings.append("loaded from snapshot (offline)")
    else:
        html = fetch(source.url)
        text = clean_html(html)
        fetched_at = today
        config.CONFIG.raw_dir.mkdir(parents=True, exist_ok=True)
        snap.write_text(text, encoding="utf-8")

    if len(text) < 500:
        warnings.append(f"suspiciously short extraction ({len(text)} chars)")

    return RawDoc(
        scheme=source.scheme,
        doc_type=source.doc_type,
        source_url=source.url,
        text=text,
        fetched_at=fetched_at,
        snapshot_path=str(snap),
        warnings=warnings,
    )


def load_all(sources, refresh: bool = False) -> tuple[list[RawDoc], list[str]]:
    """Load every source. Returns (documents, fatal_errors)."""
    docs: list[RawDoc] = []
    errors: list[str] = []

    for source in sources:
        try:
            doc = load_source(source, refresh=refresh)
        except Exception as exc:  # noqa: BLE001 - collected, not raised
            errors.append(f"{source.scheme} / {source.doc_type}: {exc}")
            continue
        docs.append(doc)
        for warning in doc.warnings:
            print(f"  ! {doc.scheme} / {doc.doc_type}: {warning}")

    return docs, errors
