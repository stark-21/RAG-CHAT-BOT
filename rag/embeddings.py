"""Phase 3 - Embeddings.

Wraps `sentence-transformers/all-MiniLM-L6-v2`: a local, 384-dimensional model
that needs no API key.

There is exactly ONE model instance in the process, used for both corpus chunks
and user questions. That is not an optimisation, it is a correctness
requirement: the query vector must live in the same space as the document
vectors, or retrieval returns nonsense.
"""

from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np

import config

EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def get_model():
    """Load MiniLM once per process and reuse it."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.CONFIG.embedding_model)


def embed_documents(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed corpus chunks. L2-normalised so cosine == dot product."""
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    vectors = get_model().encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a single user question into the same space as the corpus."""
    vector = embed_documents([text])
    return vector[0]


def verify_space(query_vector: np.ndarray, doc_vector: np.ndarray) -> float:
    """Cosine similarity between a query and a document vector.

    Used by the Phase 3 checks: a correct pairing should score well above an
    unrelated pairing. Catches the classic failure of embedding queries with a
    different model than the corpus.
    """
    return float(np.dot(query_vector, doc_vector))
