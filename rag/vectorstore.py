"""Phase 3 - Vector store.

Thin wrapper over ChromaDB's PersistentClient. The collection lives on disk at
data/chroma/ so ingestion runs once and is not repeated on every app restart
(PRD N2).

Cosine distance is used, which pairs correctly with the L2-normalised vectors
produced by rag/embeddings.py.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings

import config

COLLECTION_NAME = "mf_faq"


def _sanitize(metadata: dict) -> dict:
    """Chroma accepts only str/int/float/bool metadata values."""
    clean: dict = {}
    for key, value in metadata.items():
        if value is None:
            clean[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def get_client():
    """Open (or create) the persistent Chroma client."""
    chroma_dir = Path(config.CONFIG.chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def get_collection(client=None, create: bool = True):
    """Return the corpus collection, creating it if needed."""
    client = client or get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "description": "MF FAQ chunks"},
    )


def reset_collection(client=None):
    """Delete and recreate the collection. Makes ingest idempotent."""
    client = client or get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:  # noqa: BLE001 - collection may not exist yet
        pass
    return get_collection(client)


def store_chunks(chunks, vectors) -> int:
    """Write chunks + vectors + metadata into the collection. Returns count."""
    if not chunks:
        return 0

    collection = reset_collection()

    ids = [chunk.chunk_id for chunk in chunks]
    documents = [chunk.text for chunk in chunks]
    metadatas = [_sanitize(chunk.to_metadata()) for chunk in chunks]
    embeddings = [list(map(float, vector)) for vector in vectors]

    collection.add(
        ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
    )
    return collection.count()


def count() -> int:
    """Number of vectors currently persisted. Used by the restart check."""
    return get_collection().count()
