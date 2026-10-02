"""Phase 6 - FastAPI server for the Stitch "Calm Financial Clarity" UI.

    uvicorn server:app --host 0.0.0.0 --port $PORT
    # local:  python server.py

Serves the hand-built HTML/CSS/JS front end in `static/` and exposes the same
`rag.pipeline.answer_question()` call the Streamlit UI makes. The RAG pipeline
is untouched: this module is only an HTTP/JSON adapter around it. Nothing here
retrieves, prompts, generates, or post-processes.

Three responsibilities beyond routing:

* **Boot.** `boot.ensure_ready()` runs once per process in the lifespan hook,
  off the event loop. Until it finishes, `/api/health` reports `starting` and
  `/api/chat` refuses with 503, so no visitor waits on a vector-store build
  behind a request.
* **Sessions.** The design's left rail and its follow-up rewriting both need
  per-visitor conversation memory. `SessionStore` holds a `Conversation` and a
  lock per session id, in memory, bounded by count and age - the same lifetime
  rule the Streamlit version had, where memory died with the browser tab.
* **Redaction.** `guard.detect_pii()` runs before the question is echoed back,
  so a PAN never round-trips through the browser even though the pipeline
  already refuses it. The display string is a placeholder; the raw text is
  still what reaches the pipeline, which is what makes the refusal correct.

Every route is a sync `def`, so Starlette runs it in its worker threadpool.
`answer_question()` blocks for the length of an LLM call - up to ~12s when the
Groq rate limiter is pacing - and running that on the event loop would stall
every other visitor.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import secrets
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from rag import boot, guard
from rag.generator import GeneratorError
from rag.memory import Conversation
from rag.pipeline import answer_question

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Kept in step with app.EXAMPLE_QUESTIONS on purpose. app.py is the superseded
# Streamlit UI and still needs its own copy, because importing this module
# would drag FastAPI and the static mount into every Streamlit test run.
EXAMPLE_QUESTIONS = [
    "What is the exit load of HDFC Small Cap Fund?",
    "What is the minimum SIP amount for HDFC Large Cap Fund?",
    "What is the benchmark of HDFC Small Cap Fund?",
]

# Rendered in the header's second row and under the composer.
BANNER = (
    "Facts-only assistant. Answers are extracted strictly from official public "
    "documents and are not investment advice or recommendations."
)
FOOTER_NOTE = (
    "Mutual fund investments are subject to market risks. Read all scheme "
    "related documents carefully before investing."
)

# The left rail's "Official Sources We Query" block. Shown as a fixed list
# because it describes the deployment, not the query.
OFFICIAL_SOURCES = [
    "HDFC Asset Management scheme pages",
    "SEBI & AMFI statutory disclosures",
    "Scheme Information Documents",
]

CORPUS_NOTE = (
    "Answers are retrieved from the five HDFC Direct-Growth scheme pages in "
    "this demo's corpus. The assistant answers only what those sources cover "
    "and says so when they do not."
)

# Doc types come from sources.py; the UI shows the long form.
DOC_TYPE_LABELS = {
    "scheme_page": "Scheme Page",
    "factsheet": "Monthly Factsheet",
    "kim": "Key Information Memorandum",
    "sid": "Scheme Information Document",
    "faq": "FAQ",
    "fees": "Fees & Expenses",
    "riskometer": "Riskometer",
    "statement_guide": "Account Statement Guide",
    "education": "Investor Education",
}

# A visitor who never sends a message still holds one of these. Render's free
# tier runs a single process with no shared store, so this is a plain dict and
# it evaporates on restart - the same property the Streamlit session had.
MAX_SESSIONS = 200
SESSION_TTL_S = 2 * 60 * 60


class SessionStore:
    """Per-visitor `Conversation` plus the lock that serialises access to it.

    The lock matters: `answer_question()` mutates the conversation it is
    handed, so two requests from a double-clicking visitor would otherwise
    interleave records into one deque.
    """

    def __init__(self, max_sessions: int = MAX_SESSIONS,
                 ttl_s: float = SESSION_TTL_S) -> None:
        self._entries: OrderedDict[str, tuple[Conversation, float, threading.Lock]] = (
            OrderedDict()
        )
        self._max = max_sessions
        self._ttl = ttl_s
        self._lock = threading.Lock()

    def acquire(self, session_id: str) -> tuple[Conversation, threading.Lock]:
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(session_id)
            if entry is not None and now - entry[1] > self._ttl:
                self._entries.pop(session_id, None)
                entry = None
            if entry is None:
                entry = (Conversation(), now, threading.Lock())
                self._entries[session_id] = entry
            else:
                self._entries.move_to_end(session_id)
            self._evict(now)
            return entry[0], entry[2]

    def drop(self, session_id: str) -> None:
        """Forget a conversation. Used by "New inquiry"."""
        with self._lock:
            self._entries.pop(session_id, None)

    def _evict(self, now: float) -> None:
        """Drop expired entries, then the oldest ones over the cap. Caller holds the lock."""
        for key in [k for k, v in self._entries.items()
                    if now - v[1] > self._ttl]:
            del self._entries[key]
        while len(self._entries) > self._max:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


sessions = SessionStore()

# FastAPI reads this for the OpenAPI schema.
@asynccontextmanager
async def lifespan(_: FastAPI):
    """Warm the embedder and build the store before the first request.

    Boot already runs at BUILD time via `python -m rag.boot`, so this is the
    cheap warm path: reloading the ONNX model and opening the existing
    collection. It is blocking, and on a cold store it is the ~5s rebuild, so
    it goes to a thread rather than stalling the event loop - and it is paid
    here rather than inside somebody's first question.
    """
    await asyncio.to_thread(boot.ensure_ready)
    yield


app = FastAPI(
    title="Mutual Fund FAQ Assistant",
    description="Facts-only RAG assistant over HDFC mutual fund scheme pages.",
    version="1.0.0",
    lifespan=lifespan,
)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.svg", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(STATIC_DIR / "logo.svg", media_type="image/svg+xml")


@app.get("/api/health")
def health() -> dict:
    status = boot.status()
    return {
        "status": "ok" if status.ready else "starting",
        "ready": status.ready,
        "errors": list(status.errors),
        "chunks": status.chunks,
        "boot_seconds": round(status.total_seconds, 2),
        "sessions": len(sessions),
    }


@app.get("/api/meta")
def meta() -> dict:
    """Everything the shell renders before the first question is asked."""
    return {
        "title": "MF FAQ Assistant",
        "examples": EXAMPLE_QUESTIONS,
        "banner": BANNER,
        "footer_note": FOOTER_NOTE,
        "official_sources": OFFICIAL_SOURCES,
        "corpus_note": CORPUS_NOTE,
        "missing_env": config.missing_env(),
        "boot_errors": list(boot.status().errors),
    }


class ChatRequest(BaseModel):
    question: str = ""
    session_id: str = ""


class ResetRequest(BaseModel):
    session_id: str = ""


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    question = (request.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Ask a question first.")

    missing = config.missing_env()
    if missing:
        raise HTTPException(
            status_code=503,
            detail="Missing .env entries: {}. Add them and restart.".format(
                ", ".join(missing)),
        )

    status = boot.status()
    if not status.ready:
        raise HTTPException(
            status_code=503,
            detail="; ".join(status.errors) or "The assistant is still starting up.",
        )

    session_id = (request.session_id or "").strip() or secrets.token_urlsafe(12)
    conversation, lock = sessions.acquire(session_id)

    # Same rule as the Streamlit UI: the pipeline refuses PII anyway, but the
    # echo back to the browser is a separate surface, and a PAN rendered in the
    # transcript is a violation the user can see.
    pii = guard.detect_pii(question)
    display = question
    if pii:
        display = "[message contained a personal identifier and was not stored]"

    with lock:
        try:
            result = answer_question(question, conversation=conversation)
        except GeneratorError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "session_id": session_id,
        "question": display,
        "answer": {
            "text": result.answer.text,
            "source_url": result.answer.source_url,
            "freshness": result.answer.freshness,
            "link": result.answer.link,
            "refused": result.answer.refused,
            "refusal_reason": result.answer.refusal_reason,
            "warnings": list(result.answer.warnings),
        },
        "chunks": [_serialize_chunk(chunk) for chunk in result.chunks],
        "verdict": dataclasses.asdict(result.verdict),
        "stage": result.stage,
        "latency_s": round(result.latency_s, 2),
        "rewritten": result.rewritten,
        # The PII branch of the pipeline returns the raw question as the
        # resolved one. Redacting the echo is not enough - that value is on the
        # wire too, and a PAN in a JSON body is still a PAN in the browser's
        # network log. The UI only reads this field when `rewritten` is true,
        # which a PII refusal never is.
        "resolved_question": display if pii else result.resolved_question,
    }


@app.post("/api/reset")
def reset(request: ResetRequest) -> dict:
    """Start a fresh inquiry. Drops the conversation so the next question
    cannot resolve a follow-up against a transcript the user can no longer see."""
    session_id = (request.session_id or "").strip()
    if session_id:
        sessions.drop(session_id)
    return {"session_id": session_id or "", "status": "cleared"}


def _serialize_chunk(chunk) -> dict:
    """Flatten one retrieved chunk for the "Sources" disclosure.

    `distance` is sent rather than only `score` because the UI reports the
    cosine distance, and the pipeline's threshold is calibrated on it.
    """
    return {
        "chunk_id": chunk.chunk_id,
        "scheme": chunk.scheme,
        "scheme_short": chunk.scheme_short,
        "section": chunk.section,
        "doc_type": DOC_TYPE_LABELS.get(chunk.doc_type, chunk.doc_type),
        "source_url": chunk.source_url,
        "source_tier": chunk.source_tier,
        "fetched_at": chunk.fetched_at,
        "distance": round(chunk.distance, 3),
        "score": round(chunk.score, 3),
        "citation_label": chunk.citation_label,
        "text": chunk.text,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        log_level="info",
    )