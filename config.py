"""Application configuration.

Loads .env once and exposes every tunable in one place, so no other module
needs to read environment variables or hard-code paths. Secrets are never
logged or printed by this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    # --- Secrets ---
    groq_api_key: str

    # --- Models ---
    llm_model: str
    embedding_model: str

    # --- Retrieval ---
    top_k: int
    score_threshold: float
    max_tokens: int
    temperature: float

    # --- Chunking (Phase 2 validates these against the real data) ---
    chunk_size: int
    chunk_overlap: int

    # --- Paths ---
    base_dir: Path
    data_dir: Path
    raw_dir: Path
    chroma_dir: Path
    chunks_dir: Path
    chunks_file: Path


def _build() -> Config:
    data_dir = BASE_DIR / "data"
    chunks_dir = data_dir / "chunks"
    return Config(
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        # llama-3.3-70b-versatile was retired by Groq (404). Verified live at
        # build time. Override with GROQ_MODEL for a different tier.
        llm_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
        or "openai/gpt-oss-120b",
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        # Raised from 5 to 8 after a measured retrieval miss. Each scheme has
        # 13-22 chunks and the scheme name appears in every chunk's embed
        # header, so a question like "What is the benchmark of HDFC Balanced
        # Advantage Fund?" is dominated by the scheme name: the real
        # "Fund benchmark" chunk (body: "NIFTY 50 Hybrid Composite Debt 50:50
        # Index") sat at distance 0.726, outside the top 5, and the bot
        # wrongly said "I don't know". At k=8 it is retrieved and all other
        # questions still resolve. See docs/chunking_strategy.md section 8.
        top_k=_int_env("TOP_K", 8),
        # Calibrated in Phase 3 against measured cosine distances (lower =
        # closer), not guessed. Answerable questions with the scheme named
        # score 0.149-0.492; corpus gaps 0.695; off-topic 0.787-0.924. At 0.65
        # the off-topic and known-gap questions are refused while every
        # answerable one passes. See docs/chunking_strategy.md section 7.
        score_threshold=_float_env("SCORE_THRESHOLD", 0.65),
        # The default model is a reasoning model: its chain-of-thought is
        # billed against max_tokens. At 220 the reasoning consumed the whole
        # budget and the visible answer came back empty (finish_reason=length).
        # Keep this comfortably above what the 3-sentence answer needs.
        max_tokens=_int_env("MAX_TOKENS", 2000),
        temperature=float(os.getenv("TEMPERATURE", "0.0")),
        chunk_size=_int_env("CHUNK_SIZE", 450),
        chunk_overlap=_int_env("CHUNK_OVERLAP", 60),
        base_dir=BASE_DIR,
        data_dir=data_dir,
        raw_dir=data_dir / "raw",
        chroma_dir=data_dir / "chroma",
        chunks_dir=chunks_dir,
        chunks_file=chunks_dir / "chunks.txt",
    )


CONFIG = _build()


def has_groq_key() -> bool:
    """True when a Groq key is present. Never exposes the value itself."""
    return bool(CONFIG.groq_api_key)


def missing_env() -> list[str]:
    """Names of required .env entries that are absent or empty."""
    missing: list[str] = []
    if not CONFIG.groq_api_key:
        missing.append("GROQ_API_KEY")
    if not CONFIG.llm_model:
        missing.append("GROQ_MODEL")
    return missing


def summary() -> dict:
    """Safe-to-print view of the config. Masks the API key (PRD N5)."""
    key = CONFIG.groq_api_key
    return {
        "groq_api_key": "loaded" if key else "MISSING (add it to .env)",
        "llm_model": CONFIG.llm_model,
        "embedding_model": CONFIG.embedding_model,
        "top_k": CONFIG.top_k,
        "chunk_size": CONFIG.chunk_size,
        "chunk_overlap": CONFIG.chunk_overlap,
        "chroma_dir": str(CONFIG.chroma_dir),
        "chunks_file": str(CONFIG.chunks_file),
    }


def ensure_dirs() -> None:
    """Create the data directories. Called by ingest and by the app at boot."""
    CONFIG.data_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.raw_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.chunks_dir.mkdir(parents=True, exist_ok=True)
