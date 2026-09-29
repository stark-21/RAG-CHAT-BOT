"""Startup - get the app to a ready state before the first question.

Two things have to be true before the first question can be answered, and on a
fresh deploy neither of them is:

  1. The Chroma collection holds the corpus vectors.
  2. The embedding model is loaded.

**Why this exists.** `data/chroma/` is git-ignored, because it is a build
artifact. On a fresh clone - and on every Render deploy, since the free tier
has no persistent disk - the collection is created empty by
`get_or_create_collection`. Retrieval then returned nothing, `best_distance()`
was `None`, and every question was refused as "I don't know". The workaround
was to run `python ingest.py` in the platform's start command, which meant a
full load-chunk-embed-store pass on every single cold start, before the web
server was even listening.

Building here instead costs about a second, from the committed `data/raw/`
snapshots, and it happens inside the app rather than in front of it. Ingestion
is still idempotent and still available as a CLI; this is the same code path,
just triggered by an empty store instead of by a human.

Ordering note: the model is warmed before the store is built, because building
the store needs the embedder anyway. Doing it in this order means the cost is
paid once, not twice.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import config


@dataclass
class BootStatus:
    """What happened at startup. Safe to print; contains no secrets."""

    model_seconds: float = 0.0
    store_built: bool = False
    store_seconds: float = 0.0
    chunks: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.chunks > 0 and not self.errors

    @property
    def total_seconds(self) -> float:
        return self.model_seconds + self.store_seconds


_BOOT_LOCK = threading.Lock()
_STATUS: BootStatus | None = None


def status() -> BootStatus:
    """The boot result, or an empty status if boot has not run yet.

    Reads `_STATUS` without the boot lock on purpose: a lock-free read of a
    module global is atomic in CPython, and `status()` is only ever called for
    display, so it must never block behind an in-progress build.
    """
    return _STATUS or BootStatus()


def _build_store() -> tuple[int, float]:
    """Load -> chunk -> embed -> store from the committed snapshots.

    Loads from `data/raw/` rather than the network: those snapshots are in
    version control, so this is deterministic and does not depend on the
    source pages being reachable at deploy time.
    """
    from rag.chunking import chunk_documents
    from rag.embeddings import embed_documents
    from rag.loader import load_all
    from rag.vectorstore import count, get_collection, store_chunks
    import sources

    started = time.time()

    corpus = sources.SOURCES + [s for s in sources.GENERAL_SOURCES if s.is_resolved]
    docs, errors = load_all(corpus, refresh=False)
    if not docs:
        return 0, time.time() - started

    chunks = chunk_documents(docs)
    if not chunks:
        return 0, time.time() - started

    vectors = embed_documents([chunk.embed_text for chunk in chunks])
    store_chunks(chunks, vectors)

    # Re-read through the cached handle so the count is the store's, not the
    # writer's return value.
    return get_collection().count(), time.time() - started


def ensure_ready(verbose: bool = False) -> BootStatus:
    """Warm the embedder and build the vector store if it is empty.

    Idempotent and cheap when called again: an already-populated store is left
    alone and only the (cached, ~0.2s) model load is repeated. A no-op on the
    warm path is a few milliseconds, so app.py can call this on every rerun
    without a guard.

    The lock is held across the work, not just the status read. Streamlit runs
    each session in its own thread and calls this on every rerun, so two
    visitors arriving during a cold start would otherwise both see an empty
    store and both build it. The second caller now waits a few seconds and
    returns the first caller's result.
    """
    global _STATUS

    with _BOOT_LOCK:
        if _STATUS is not None and _STATUS.ready:
            return _STATUS
        result = BootStatus()

        from rag import embeddings
        from rag.vectorstore import get_collection

        try:
            started = time.time()
            embeddings.warm()
            result.model_seconds = time.time() - started
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            result.errors.append(f"embedding model unavailable: {exc}")

        if not result.errors:
            try:
                existing = get_collection().count()
                if existing > 0:
                    result.chunks = existing
                else:
                    count, elapsed = _build_store()
                    result.store_built = True
                    result.store_seconds = elapsed
                    result.chunks = count
                    # Anything cached against the empty store is now a miss:
                    # retrieval found nothing, and no answers were ever produced.
                    from rag import pipeline, retrieval

                    retrieval.clear_cache()
                    pipeline.clear_cache()
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                result.errors.append(f"vector store unavailable: {exc}")

        _STATUS = result

        if verbose:
            print(
                f"boot: model {result.model_seconds:.2f}s, "
                f"{'built' if result.store_built else 'reused'} store in "
                f"{result.store_seconds:.2f}s, {result.chunks} chunks"
                + (f", errors={result.errors}" if result.errors else "")
            )
        return result


def warm_in_background() -> threading.Thread:
    """Start `ensure_ready()` on a daemon thread and return it.

    Used by the CLI so the first keystroke does not wait on the model. The UI
    calls `ensure_ready()` synchronously instead: Streamlit reruns the script
    on every interaction, so work started on a thread that has not finished is
    work that gets started again.
    """
    thread = threading.Thread(target=ensure_ready, kwargs={"verbose": True},
                              daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    import sys

    print("Boot summary:")
    print(f"  config: {config.summary()}")
    boot = ensure_ready(verbose=True)
    print(f"  ready={boot.ready} chunks={boot.chunks} total={boot.total_seconds:.2f}s")
    for error in boot.errors:
        print(f"  ! {error}")
    # Non-zero on failure so a build step that depends on this fails loudly
    # instead of shipping a service that cannot answer anything.
    sys.exit(0 if boot.ready else 1)
