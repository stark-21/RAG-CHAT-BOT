"""Startup tests.

Nothing here loads a model or touches Chroma: `embeddings.warm` and the store
build are stubbed. What is being tested is the startup ORDERING and CONCURRENCY
behaviour, which is what makes a cold deploy usable.
"""

from __future__ import annotations

import threading
import time

import pytest

from rag import boot


@pytest.fixture(autouse=True)
def _reset():
    """Boot memoises its result in a module global; each test wants a cold one."""
    boot._STATUS = None
    yield
    boot._STATUS = None


@pytest.fixture
def warm(monkeypatch):
    """Stub the embedder. Returns a list that records how often it was warmed."""
    from rag import embeddings

    calls = []

    def fake_warm():
        calls.append(1)

    monkeypatch.setattr(embeddings, "warm", fake_warm)
    return calls


def _store(count: int, monkeypatch):
    """Stub the Chroma collection to report `count` chunks."""
    from rag import vectorstore

    class FakeCollection:
        def count(self):
            return count

    monkeypatch.setattr(vectorstore, "get_collection", lambda: FakeCollection())


class TestIdempotence:
    def test_populated_store_is_reused_not_rebuilt(self, warm, monkeypatch):
        _store(90, monkeypatch)

        builds = []
        monkeypatch.setattr(boot, "_build_store",
                            lambda: builds.append(1) or (0, 0.0))

        result = boot.ensure_ready()

        assert result.ready
        assert result.chunks == 90
        assert result.store_built is False
        assert not builds, "an existing corpus must not be rebuilt on every start"

    def test_second_call_is_a_no_op(self, warm, monkeypatch):
        _store(90, monkeypatch)

        first = boot.ensure_ready()
        second = boot.ensure_ready()

        assert first is second, "the boot result should be memoised"
        assert len(warm) == 1, "the embedder is cached; warm() should run once"

    def test_status_is_readable_before_boot(self):
        s = boot.status()
        assert s.chunks == 0
        assert not s.ready


class TestStoreBuild:
    def test_empty_store_is_built_from_the_snapshots(self, warm, monkeypatch):
        _store(0, monkeypatch)
        monkeypatch.setattr(boot, "_build_store", lambda: (90, 4.2))

        result = boot.ensure_ready()

        assert result.ready
        assert result.store_built is True
        assert result.chunks == 90
        assert result.store_seconds == 4.2

    def test_build_clears_the_caches_computed_against_the_empty_store(
        self, warm, monkeypatch
    ):
        """Before the build, retrieval cached a miss for every question and
        the pipeline cached the resulting refusal. Both must be dropped."""
        from rag import pipeline, retrieval

        _store(0, monkeypatch)
        monkeypatch.setattr(boot, "_build_store", lambda: (90, 1.0))

        cleared = []
        monkeypatch.setattr(retrieval, "clear_cache",
                            lambda: cleared.append("retrieval"))
        monkeypatch.setattr(pipeline, "clear_cache",
                            lambda: cleared.append("pipeline"))

        boot.ensure_ready()

        assert cleared == ["retrieval", "pipeline"]


class TestConcurrentColdStart:
    def test_two_visitors_arriving_together_build_once(self, warm, monkeypatch):
        """Streamlit runs each session on its own thread and calls ensure_ready
        on every rerun, so a cold start is genuinely concurrent.

        Without the lock held across the work, both threads see an empty store
        and both build it: double the CPU, and a real chance of two writers
        racing on the same Chroma collection.
        """
        _store(0, monkeypatch)

        builds = []
        started = threading.Event()

        def slow_build():
            builds.append(1)
            started.set()
            time.sleep(0.15)          # the window a real build takes
            return 90, 0.15

        monkeypatch.setattr(boot, "_build_store", slow_build)

        results = []
        threads = [threading.Thread(target=lambda: results.append(boot.ensure_ready()))
                   for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(results) == 4
        assert len(builds) == 1, f"store built {len(builds)} times, expected once"
        assert all(r.ready and r.chunks == 90 for r in results)


class TestFailure:
    def test_model_failure_is_reported_not_raised(self, monkeypatch):
        from rag import embeddings

        def boom():
            raise RuntimeError("no such model on disk")

        monkeypatch.setattr(embeddings, "warm", boom)

        result = boot.ensure_ready()

        assert not result.ready
        assert any("embedding model" in e for e in result.errors)

    def test_store_failure_is_reported(self, warm, monkeypatch):
        from rag import vectorstore

        def boom():
            raise RuntimeError("chroma is locked")

        monkeypatch.setattr(vectorstore, "get_collection", boom)

        result = boot.ensure_ready()

        assert not result.ready
        assert any("vector store" in e for e in result.errors)

    def test_a_failed_boot_is_retried_next_time(self, warm, monkeypatch):
        """A transient failure must not be memoised as a permanent state, or
        the app would stay broken for the life of the process."""
        from rag import embeddings

        _store(90, monkeypatch)

        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("network hiccup")

        monkeypatch.setattr(embeddings, "warm", flaky)

        first = boot.ensure_ready()
        assert not first.ready

        second = boot.ensure_ready()
        assert second.ready, "boot must retry after a failure"
