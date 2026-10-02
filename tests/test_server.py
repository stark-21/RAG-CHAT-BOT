"""HTTP layer for server.py.

These tests stop at the adapter boundary. `answer_question` is stubbed, so what
is under test is exactly what server.py is responsible for: JSON shape, the
redaction that keeps a PAN out of the browser, session lifetime, and the mapping
from pipeline outcomes to status codes. The pipeline's own behaviour is covered
by test_guard.py, test_memory.py and test_phase5.py, and re-testing it here
would only duplicate them against a stub.

The `TestClient` is deliberately not used as a context manager, because that
triggers the lifespan hook and a real `boot.ensure_ready()`. Startup is
exercised separately by test_boot.py.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server  # noqa: E402
from rag.boot import BootStatus  # noqa: E402
from rag.generator import GeneratorError  # noqa: E402
from rag.guard import Verdict  # noqa: E402
from rag.memory import Conversation  # noqa: E402
from rag.pipeline import QueryResult  # noqa: E402
from rag.postprocess import Answer  # noqa: E402
from rag.retrieval import ScoredChunk  # noqa: E402


def make_chunk() -> ScoredChunk:
    return ScoredChunk(
        chunk_id="hdfc-small-cap__scheme_page__0007",
        text="Exit load: 1.00% if redeemed within 1 year from allotment.",
        distance=0.182,
        section="Exit load",
        scheme="HDFC Small Cap Fund - Direct Growth",
        scheme_short="HDFC Small Cap",
        doc_type="scheme_page",
        source_url="https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
        source_tier="aggregator",
        fetched_at="2025-02-28",
    )


def make_result(**overrides) -> QueryResult:
    chunk = make_chunk()
    answer = Answer(
        text="The exit load is 1.00% within 1 year.",
        source_url=chunk.source_url,
        freshness="2025-02-28",
        chunks=[chunk],
        latency_s=0.51,
    )
    fields = {
        "answer": answer,
        "chunks": [chunk],
        "verdict": Verdict("answer"),
        "stage": "answered",
        "latency_s": 0.51,
        "resolved_question": "",
        "rewritten": False,
    }
    fields.update(overrides)
    return QueryResult(**fields)


@pytest.fixture
def client(monkeypatch):
    """A client with boot reported as ready and the pipeline stubbed."""
    monkeypatch.setattr(
        server.boot, "status", lambda: BootStatus(model_seconds=0.2, chunks=90)
    )
    return TestClient(server.app)


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Replace `answer_question` and record the conversations it was handed.

    The stub records the exchange, because that is part of the contract being
    delegated: the real pipeline mutates the conversation it is given, and
    several tests below depend on that mutation happening.
    """
    calls: list[dict] = []

    def stub(question, show_chunks=False, conversation=None):
        result = make_result()
        calls.append({"question": question, "conversation": conversation})
        if conversation is not None:
            conversation.record_exchange(question, result.answer.text)
        return result

    monkeypatch.setattr(server, "answer_question", stub)
    return calls


class TestStaticShell:
    def test_index_is_served_at_the_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "MF FAQ Assistant" in response.text

    def test_stylesheet_and_script_are_served(self, client):
        assert client.get("/static/styles.css").status_code == 200
        assert client.get("/static/app.js").status_code == 200

    def test_favicon_is_the_logo(self, client):
        response = client.get("/favicon.svg")
        assert response.status_code == 200
        assert "svg" in response.headers["content-type"]

    def test_the_ui_only_talks_to_this_backend(self, client):
        """The whole point of the rewrite: no framework runtime in the request path.

        Asserted on where the script actually sends requests rather than on the
        word "streamlit", which a comment could reintroduce. Every fetch() must
        be same-origin and under /api/.
        """
        script = client.get("/static/app.js").text
        targets = re.findall(r'fetch\(\s*([^,)]+)', script)
        assert targets, "the script should be making requests"
        for target in targets:
            assert '"/api/' in target, target


class TestMeta:
    def test_returns_the_copy_the_shell_renders(self, client):
        payload = client.get("/api/meta").json()
        assert payload["examples"], "the composer needs at least one example"
        assert payload["banner"].startswith("Facts-only assistant")
        assert "market risks" in payload["footer_note"]
        assert payload["official_sources"]

    def test_every_example_is_a_string_the_backend_owns(self, client):
        for question in client.get("/api/meta").json()["examples"]:
            assert isinstance(question, str) and question.strip()


class TestHealth:
    def test_reports_starting_before_boot_completes(self, client, monkeypatch):
        monkeypatch.setattr(server.boot, "status",
                            lambda: BootStatus(errors=["no vector store"]))
        payload = client.get("/api/health").json()
        assert payload["status"] == "starting"
        assert payload["ready"] is False
        assert payload["errors"] == ["no vector store"]

    def test_reports_ok_once_boot_completes(self, client):
        payload = client.get("/api/health").json()
        assert payload["status"] == "ok"
        assert payload["chunks"] == 90

    def test_health_never_raises_so_a_cold_start_is_not_a_restart_loop(
        self, client, monkeypatch
    ):
        monkeypatch.setattr(
            server.boot, "status",
            lambda: BootStatus(errors=["model download failed"]),
        )
        assert client.get("/api/health").status_code == 200


class TestChat:
    def test_returns_the_fields_the_answer_card_needs(self, client, stub_pipeline):
        payload = client.post(
            "/api/chat",
            json={"question": "What is the exit load of HDFC Small Cap Fund?"},
        ).json()

        assert payload["answer"]["text"]
        assert payload["answer"]["source_url"].startswith("https://")
        assert payload["answer"]["freshness"] == "2025-02-28"
        assert payload["stage"] == "answered"
        assert payload["verdict"]["action"] == "answer"

    def test_echoes_the_question(self, client, stub_pipeline):
        payload = client.post(
            "/api/chat", json={"question": "What is the AUM of HDFC Small Cap Fund?"}
        ).json()
        assert payload["question"] == "What is the AUM of HDFC Small Cap Fund?"

    def test_serialises_chunks_with_the_fields_the_disclosure_shows(
        self, client, stub_pipeline
    ):
        chunk = client.post(
            "/api/chat", json={"question": "What is the exit load of HDFC Small Cap Fund?"}
        ).json()["chunks"][0]

        for field in ("chunk_id", "scheme_short", "section", "doc_type",
                      "source_url", "fetched_at", "distance", "citation_label",
                      "text"):
            assert chunk[field] not in (None, ""), field
        # doc_type is mapped to its long form for display.
        assert chunk["doc_type"] == "Scheme Page"

    def test_rounds_latency_for_the_footer_caption(self, client, stub_pipeline):
        payload = client.post(
            "/api/chat", json={"question": "What is the exit load of HDFC Small Cap Fund?"}
        ).json()
        assert payload["latency_s"] == 0.51

    def test_a_refusal_surfaces_its_verdict_and_no_citation(self, client, monkeypatch):
        verdict = Verdict("refuse_advice", rule="advice", message="No advice.")
        monkeypatch.setattr(
            server, "answer_question",
            lambda *a, **k: make_result(
                answer=Answer(text="No advice.", refused=True, chunks=[]),
                chunks=[], verdict=verdict, stage="guard", latency_s=0.0),
        )
        payload = client.post(
            "/api/chat", json={"question": "Which fund is best?"}
        ).json()
        assert payload["stage"] == "guard"
        assert payload["verdict"]["action"] == "refuse_advice"
        assert payload["answer"]["source_url"] == ""
        assert payload["chunks"] == []

    def test_empty_question_is_a_client_error(self, client, stub_pipeline):
        response = client.post("/api/chat", json={"question": "   "})
        assert response.status_code == 400
        assert stub_pipeline == []

    def test_issues_a_session_id_when_the_client_has_none(self, client, stub_pipeline):
        payload = client.post(
            "/api/chat", json={"question": "What is the exit load of HDFC Small Cap Fund?"}
        ).json()
        assert payload["session_id"]

    def test_generator_failure_is_503_not_a_500(self, client, monkeypatch):
        def boom(*args, **kwargs):
            raise GeneratorError("rate limited, retry in 12s")

        monkeypatch.setattr(server, "answer_question", boom)
        response = client.post(
            "/api/chat", json={"question": "What is the exit load of HDFC Small Cap Fund?"}
        )
        assert response.status_code == 503
        assert "rate limited" in response.json()["detail"]

    def test_a_failed_boot_is_503_before_any_llm_call(self, client, monkeypatch):
        monkeypatch.setattr(
            server.boot, "status", lambda: BootStatus(errors=["store is empty"])
        )
        called = []
        monkeypatch.setattr(
            server, "answer_question", lambda *a, **k: called.append(1)
        )
        response = client.post(
            "/api/chat", json={"question": "What is the exit load of HDFC Small Cap Fund?"}
        )
        assert response.status_code == 503
        assert called == []

    def test_a_missing_key_is_503_with_the_name_but_not_the_value(
        self, client, monkeypatch
    ):
        monkeypatch.setattr(server.config, "missing_env", lambda: ["GROQ_API_KEY"])
        response = client.post(
            "/api/chat", json={"question": "What is the exit load of HDFC Small Cap Fund?"}
        )
        assert response.status_code == 503
        assert "GROQ_API_KEY" in response.json()["detail"]
        assert "gsk_" not in response.json()["detail"]


class TestPii:
    """A PAN must not round-trip through the browser.

    The pipeline refuses it, but the echo back to the page is a separate
    surface: the question is rendered into the transcript before any answer
    exists, so redacting only the refusal would still leak it.
    """

    QUESTION = "My PAN is ABCDE1234F, what is the exit load of HDFC Small Cap Fund?"

    def test_the_echo_is_redacted(self, client, stub_pipeline):
        payload = client.post("/api/chat", json={"question": self.QUESTION}).json()
        assert payload["question"] != self.QUESTION
        assert "personal identifier" in payload["question"]

    def test_the_raw_question_still_reaches_the_pipeline(
        self, client, stub_pipeline
    ):
        """Redaction is for display only. Swapping it would break the refusal."""
        client.post("/api/chat", json={"question": self.QUESTION})
        assert stub_pipeline[0]["question"] == self.QUESTION

    def test_an_ordinary_question_is_not_redacted(self, client, stub_pipeline):
        payload = client.post(
            "/api/chat", json={"question": "What is the AUM of HDFC Small Cap Fund?"}
        ).json()
        assert payload["question"] == "What is the AUM of HDFC Small Cap Fund?"

    def test_the_identifier_appears_nowhere_in_the_response(self, client, monkeypatch):
        """Not just the echo. `resolved_question` is also on the wire.

        The PII branch of the pipeline returns the raw question as the resolved
        one, so a redacted echo alone would still leave the PAN in the JSON
        body - visible in the browser's network log and in any proxy in front
        of the app.
        """
        monkeypatch.setattr(
            server, "answer_question",
            lambda *a, **k: make_result(resolved_question=self.QUESTION),
        )
        body = client.post("/api/chat", json={"question": self.QUESTION}).text
        assert "ABCDE1234F" not in body


class TestSessions:
    def test_one_conversation_is_reused_across_turns(self, client, stub_pipeline):
        first = client.post(
            "/api/chat",
            json={"question": "What is the exit load of HDFC Small Cap Fund?",
                  "session_id": "s1"},
        ).json()
        client.post(
            "/api/chat",
            json={"question": "And its benchmark?", "session_id": "s1"},
        )
        assert stub_pipeline[0]["conversation"] is stub_pipeline[1]["conversation"]
        assert first["session_id"] == "s1"

    def test_different_visitors_get_different_conversations(
        self, client, stub_pipeline
    ):
        client.post("/api/chat", json={"question": "What is the AUM?", "session_id": "s1"})
        client.post("/api/chat", json={"question": "What is the AUM?", "session_id": "s2"})
        assert stub_pipeline[0]["conversation"] is not stub_pipeline[1]["conversation"]

    def test_reset_forgets_the_transcript(self, client, stub_pipeline):
        client.post("/api/chat", json={"question": "What is the AUM?", "session_id": "s1"})
        assert len(stub_pipeline[0]["conversation"]) > 0

        assert client.post("/api/reset", json={"session_id": "s1"}).status_code == 200

        client.post("/api/chat", json={"question": "What is the NAV?", "session_id": "s1"})
        assert stub_pipeline[1]["conversation"] is not stub_pipeline[0]["conversation"]
        # Only the new turn survives, so a later "and its exit load?" cannot
        # resolve against a transcript the user can no longer see.
        assert stub_pipeline[1]["conversation"].turns[0].content == "What is the NAV?"
        assert len(stub_pipeline[1]["conversation"]) == 2

    def test_reset_is_a_no_op_for_an_unknown_session(self, client):
        assert client.post("/api/reset", json={"session_id": "never-seen"}).status_code == 200

    def test_the_store_evicts_over_its_cap(self):
        store = server.SessionStore(max_sessions=3)
        for index in range(5):
            store.acquire("session-{}".format(index))
        assert len(store) == 3

    def test_the_store_expires_idle_sessions(self):
        store = server.SessionStore(ttl_s=0.0)
        first, _ = store.acquire("s1")
        first.record("user", "hello")
        _, lock = store.acquire("s1")
        assert lock is not None
        conversation, _ = store.acquire("s1")
        assert conversation is not first, "an expired session must not be resurrected"


class TestSessionLock:
    def test_each_session_hands_out_one_lock(self):
        store = server.SessionStore()
        _, first = store.acquire("s1")
        _, second = store.acquire("s1")
        assert first is second

    def test_different_sessions_do_not_share_a_lock(self):
        store = server.SessionStore()
        _, first = store.acquire("s1")
        _, second = store.acquire("s2")
        assert first is not second


class TestNoDirectPipelineImports:
    def test_the_server_calls_one_pipeline_entry_point(self):
        """One import surface, so the UI cannot drift from the CLI.

        `answer_question` is the only pipeline function the HTTP layer is
        allowed to reach for; everything else it uses is a plain dataclass.
        """
        source = (Path(server.__file__)).read_text(encoding="utf-8")
        assert "from rag.pipeline import answer_question" in source
        for leaked in ("retrieval.", "generator.complete", "postprocess.finalise",
                       "prompt.build_messages"):
            assert leaked not in source


def test_conversation_is_the_pipeline_type():
    """Guards against a drift where the server swaps in its own memory."""
    store = server.SessionStore()
    conversation, _ = store.acquire("s1")
    assert isinstance(conversation, Conversation)