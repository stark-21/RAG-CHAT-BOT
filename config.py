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
    reasoning_effort: str
    embedding_model: str
    model_cache_dir: Path

    # --- Retrieval ---
    top_k: int
    llm_context_k: int
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
        # Model choice is a latency decision, measured on the real grounded
        # prompt (see docs/performance.md). The previous default,
        # openai/gpt-oss-120b, is a REASONING model: it spent 371-523 output
        # tokens and 1257-1641 characters of hidden chain-of-thought on a
        # 3-sentence factual answer, and measured 1.45-1.83s per turn.
        # qwen/qwen3.8-27b emits no reasoning at all and measured 0.44-0.66s
        # across the eight in-scope questions, with correct answers and correct
        # Source lines. Override with GROQ_MODEL for a different tier; if you
        # switch to a reasoning model, set GROQ_MODEL_REASONING_EFFORT=low.
        llm_model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip()
        or "qwen/qwen3.8-27b",
        # Only sent when set. Groq accepts "low"/"medium"/"high" on the
        # gpt-oss reasoning models; a non-reasoning model rejects the field, so
        # this stays empty by default.
        reasoning_effort=os.getenv("GROQ_MODEL_REASONING_EFFORT", "").strip().lower(),
        # The ONNX export of all-MiniLM-L6-v2. Identical vectors to the
        # sentence-transformers weights (measured cosine 1.0, max abs diff
        # 1.5e-7) at 0.22s load instead of 12.28s and ~130MB instead of
        # 762MB. See rag/embeddings.py for the measurement.
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "optimum/all-MiniLM-L6-v2"
        ),
        model_cache_dir=os.getenv("MODEL_CACHE_DIR", "").strip()
        or str(data_dir / "models"),
        # Raised from 5 to 8 after a measured retrieval miss. Each scheme has
        # 13-22 chunks and the scheme name appears in every chunk's embed
        # header, so a question like "What is the benchmark of HDFC Balanced
        # Advantage Fund?" is dominated by the scheme name: the real
        # "Fund benchmark" chunk (body: "NIFTY 50 Hybrid Composite Debt 50:50
        # Index") sat at distance 0.726, outside the top 5, and the bot
        # wrongly said "I don't know". At k=8 it is retrieved and all other
        # questions still resolve. See docs/chunking_strategy.md section 8.
        top_k=_int_env("TOP_K", 8),
        # How many retrieved chunks go into the PROMPT, as opposed to how many
        # are retrieved. These are separate concerns: the grounding gate always
        # sees the full top_k, and it is that gate which decides "I don't know".
        # The default (0) puts all of top_k in the prompt, which is what shipped
        # through Phase 5.
        #
        # Narrowing this is the available lever on throughput, because Groq caps
        # input tokens per minute (7,000 on the free tier) and the ceiling is
        # 7000 / tokens-per-question:
        #
        #     blocks   tokens   questions/minute
        #         8     1499                4
        #         6     1295                5
        #         5     1181                5
        #         4     1101                6
        #         3      988                7
        #
        # It is not the default because it is not free. Asking the same eight
        # in-scope questions at each width, one sample per cell:
        #
        #   * "benchmark of HDFC Small Cap Fund" is answered correctly with 8 and
        #     6 blocks, and answered "not covered by the sources provided" with
        #     5 and 4. The fact sits in block 6 or later.
        #   * "benchmark of HDFC Balanced Advantage Fund" is wrong at EVERY
        #     width, 8 included, so that one is a retrieval and corpus gap rather
        #     than a prompt-width one.
        #
        # A single wrong answer about a fund's benchmark is a worse outcome than
        # one fewer question a minute, so the default stays at the full set.
        # Lower this only if you have measured your own questions at the width
        # you intend to set.
        llm_context_k=_int_env("LLM_CONTEXT_K", 0) or _int_env("TOP_K", 8),
        # Calibrated in Phase 3 against measured cosine distances (lower =
        # closer), not guessed. Answerable questions with the scheme named
        # score 0.149-0.492; corpus gaps 0.695; off-topic 0.787-0.924. At 0.65
        # the off-topic and known-gap questions are refused while every
        # answerable one passes. See docs/chunking_strategy.md section 7.
        score_threshold=_float_env("SCORE_THRESHOLD", 0.65),
        # Sized for the VISIBLE answer only. It was 2000 because the old
        # default model was a reasoning model whose chain-of-thought is billed
        # against max_tokens: at 220 the reasoning ate the whole budget and the
        # visible answer came back empty (finish_reason=length). The default
        # model no longer emits reasoning, so 400 is generous for three
        # sentences plus a Source line, and a runaway generation is cut off
        # sooner.
        max_tokens=_int_env("MAX_TOKENS", 400),
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
        "reasoning_effort": CONFIG.reasoning_effort or "off (non-reasoning model)",
        "embedding_model": CONFIG.embedding_model,
        "model_cache_dir": str(CONFIG.model_cache_dir),
        "top_k": CONFIG.top_k,
        "llm_context_k": CONFIG.llm_context_k,
        "max_tokens": CONFIG.max_tokens,
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
