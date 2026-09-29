"""Phase 6 - Streamlit chat UI.

    streamlit run app.py

Wires the Phase 5 pipeline into a chat interface: message history, a
"Sources" expander under every answer showing the exact chunks that were
retrieved, and a clear-chat button.

Streamlit reruns this whole script on every interaction, so all state lives in
st.session_state. Nothing is written to disk: the transcript and the
conversation memory are per-session and vanish when the browser tab closes.

`boot.ensure_ready()` runs once per process, guarded so reruns are free. It
warms the embedding model and builds the vector store if it is missing, which
is what lets a fresh deploy answer at all without a separate ingestion step.
See rag/boot.py.
"""

from __future__ import annotations

import time

import streamlit as st

import config
from rag import boot, guard, retrieval
from rag.generator import GeneratorError
from rag.memory import Conversation
from rag.pipeline import answer_question

EXAMPLE_QUESTIONS = [
    "What is the exit load of HDFC Small Cap Fund?",
    "What is the minimum SIP amount for HDFC Large Cap Fund?",
    "What is the benchmark of HDFC Small Cap Fund?",
]

DISCLAIMER = "Facts-only. No investment advice."

st.set_page_config(page_title="Mutual Fund FAQ Assistant", page_icon="📊",
                   layout="centered")

# Once per server process, not once per rerun. The work is ~1s on a warm store
# and ~5s on a cold one, and it has to happen before the first question either
# way - doing it lazily just moved the wait from page load to first answer.
if not st.session_state.get("_booted"):
    _started = time.time()
    with st.spinner("Preparing the assistant…"):
        _status = boot.ensure_ready()
    st.session_state["_booted"] = True
    st.session_state["_boot_seconds"] = time.time() - _started
    st.session_state["_boot_status"] = _status


# --- session state ---------------------------------------------------------

def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation" not in st.session_state:
        st.session_state.conversation = Conversation()
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = ""


def clear_chat() -> None:
    """Reset the transcript and the conversation memory together.

    Both must be cleared. Leaving the Conversation would let a later "what
    about its fees?" resolve against a transcript the user can no longer see.
    """
    st.session_state.messages = []
    st.session_state.conversation = Conversation()
    st.session_state.pending_question = ""


init_state()


# --- rendering -------------------------------------------------------------

def render_sources(chunks, limit: int = 3) -> None:
    """The 'Sources' expander under an answer: what was actually retrieved."""
    if not chunks:
        return
    with st.expander(f"Sources ({len(chunks)} chunks retrieved)"):
        for chunk in chunks[:limit]:
            st.markdown(
                "**{}** · {}".format(chunk.scheme_short, chunk.section)
            )
            st.caption("cosine distance {:.3f} (lower = closer)".format(
                chunk.distance))
            st.caption(chunk.source_url)
            body = chunk.text.replace("\n", " ").strip()
            st.write(body[:280] + ("…" if len(body) > 280 else ""))
            st.divider()
        if len(chunks) > limit:
            st.caption("… and {} more chunks retrieved".format(
                len(chunks) - limit))


def render_answer(answer, chunks, stage="", latency_s=None) -> None:
    """Render one answer. Single source of truth for answer presentation.

    Called both while the answer is being produced (so the spinner has
    something to sit next to) and when replaying history. Two copies of this
    would drift, and a drift here is a PRD violation the user can see.
    """
    st.markdown(answer["text"])

    if answer.get("source_url"):
        st.markdown("[🔗 Source: {0}]({0})".format(answer["source_url"]))
    if answer.get("freshness"):
        st.caption("Last updated from sources: {}".format(answer["freshness"]))
    if answer.get("link"):
        st.markdown("[Learn more: investor education]({})".format(answer["link"]))

    for warning in answer.get("warnings", []):
        st.caption("⚠ {}".format(warning))

    render_sources(chunks)

    meta = []
    if stage:
        meta.append(stage)
    if latency_s is not None:
        meta.append("{:.2f}s".format(latency_s))
    if meta:
        st.caption(" · ".join(meta))


def render_assistant(message: dict) -> None:
    with st.chat_message("assistant"):
        if message.get("resolved_question"):
            st.caption("interpreted as: {}".format(message["resolved_question"]))
        render_answer(message["answer"], message.get("chunks", []),
                      message.get("stage", ""), message.get("latency_s"))


def render_history() -> None:
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            render_assistant(message)


# --- asking ----------------------------------------------------------------

def ask(question: str) -> None:
    question = (question or "").strip()
    if not question:
        return

    # The transcript is re-rendered on every rerun, so a PII-bearing question
    # must never be stored verbatim. Detect first and store a placeholder; the
    # pipeline refuses it anyway, but the display layer is a separate risk and
    # a leaked PAN in the transcript is a PRD violation regardless of the
    # answer being clean.
    pii = guard.detect_pii(question)
    display = question
    if pii:
        display = "[message contained a personal identifier and was not stored]"

    st.session_state.messages.append({"role": "user", "content": display})
    with st.chat_message("user"):
        st.markdown(display)

    with st.chat_message("assistant"):
        with st.spinner("Searching the official sources…"):
            try:
                result = answer_question(
                    question, conversation=st.session_state.conversation
                )
            except GeneratorError as exc:
                st.error(str(exc))
                return

            payload = {
                "text": result.answer.text,
                "source_url": result.answer.source_url,
                "freshness": result.answer.freshness,
                "link": result.answer.link,
                "warnings": result.answer.warnings,
            }
            if result.rewritten:
                st.caption("interpreted as: {}".format(
                    result.resolved_question))
            render_answer(payload, result.chunks, result.stage, result.latency_s)

    # Stored only after a successful call, so a failed turn is not replayed as
    # if it had produced an answer. resolved_question is stored so the rewrite
    # indicator survives the next rerun.
    #
    # No st.rerun() here on purpose. The turn is already on screen and already
    # in session_state, so a rerun would render the identical answer a second
    # time and re-walk the whole transcript to do it - cost that grows with
    # every message in the conversation, for no change in what the user sees.
    st.session_state.messages.append({
        "role": "assistant",
        "answer": payload,
        "chunks": result.chunks,
        "stage": result.stage,
        "latency_s": result.latency_s,
        "resolved_question": result.resolved_question if result.rewritten else "",
    })


# --- page ------------------------------------------------------------------

st.title("📊 Mutual Fund FAQ Assistant")
st.caption(DISCLAIMER)
st.caption(
    "Answers are retrieved from the five HDFC Direct-Growth scheme pages in this "
    "demo's corpus. The assistant answers only what those sources cover and "
    "says so when they do not."
)

missing = config.missing_env()
if missing:
    st.error(
        "Missing .env entries: {}. Copy .env.example to .env and fill them "
        "in, then restart.".format(", ".join(missing))
    )
    st.stop()

# A boot that could not reach the store or the model means every question will
# come back as "I don't know". Say so, rather than letting the user find out.
_boot = st.session_state.get("_boot_status")
if _boot is not None and _boot.errors:
    for _error in _boot.errors:
        st.error("Startup problem: {}".format(_error))
    st.stop()

column, button_column = st.columns([5, 1])
with column:
    st.markdown("**Try one of these:**")
    example_row = st.columns(len(EXAMPLE_QUESTIONS))
    for slot, example in zip(example_row, EXAMPLE_QUESTIONS):
        if slot.button(example, key="ex_{}".format(example[:20])):
            st.session_state.pending_question = example
with button_column:
    st.markdown(" ")
    st.markdown(" ")
    if st.button("🗑️ Clear chat", on_click=clear_chat, use_container_width=True):
        st.rerun()

render_history()

typed = st.chat_input("Ask about fees, exit load, minimum SIP, benchmark, AUM, tax…")
question = st.session_state.pop("pending_question", "") or typed
if question:
    ask(question)
