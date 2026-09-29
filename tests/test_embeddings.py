"""Tests for Phase 3 embeddings and vector store.

The persistence tests need a built corpus. They skip cleanly if data/chroma
has not been built yet, so `pytest` works on a fresh clone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from rag import embeddings
from rag.vectorstore import COLLECTION_NAME, get_client, get_collection

CORPUS_READY = config.CONFIG.chroma_dir.exists() and (
    get_collection().count() > 0
) if config.CONFIG.chroma_dir.exists() else False

needs_corpus = pytest.mark.skipif(
    not CORPUS_READY, reason="run `python ingest.py` first"
)


class TestEmbedder:
    def test_produces_384_dimensions(self):
        vectors = embeddings.embed_documents(["exit load of 1%", "benchmark NIFTY"])
        assert vectors.shape == (2, embeddings.EMBEDDING_DIM)

    def test_vectors_are_l2_normalised(self):
        vectors = embeddings.embed_documents(["exit load of 1%"])
        assert np.isclose(float(np.linalg.norm(vectors[0])), 1.0, atol=1e-4)

    def test_documents_and_queries_share_one_model(self):
        """The classic RAG bug: different model for query vs corpus."""
        a = embeddings.embed_query("exit load")
        b = embeddings.embed_documents(["exit load"])[0]
        assert np.allclose(a, b, atol=1e-5)

    def test_related_text_is_closer_than_unrelated(self):
        q = embeddings.embed_query("What is the exit load?")
        near = embeddings.embed_documents(
            ["Exit load of 1% if redeemed within 1 year"]
        )[0]
        far = embeddings.embed_documents(["Recipes for pasta"])[0]
        assert embeddings.verify_space(q, near) > embeddings.verify_space(q, far)

    def test_empty_input(self):
        assert embeddings.embed_documents([]).shape == (0, embeddings.EMBEDDING_DIM)


@needs_corpus
class TestVectorStore:
    def test_collection_is_persistent(self):
        assert get_collection().count() > 0

    def test_store_lives_at_configured_directory(self):
        get_client()
        assert config.CONFIG.chroma_dir.exists()
        assert any(config.CONFIG.chroma_dir.iterdir())

    def test_uses_cosine_space(self):
        assert get_collection().metadata.get("hnsw:space") == "cosine"

    def test_metadata_round_trips(self):
        result = get_collection().get(limit=1, include=["metadatas"])
        meta = result["metadatas"][0]
        for field in ("scheme", "doc_type", "section", "source_url",
                      "source_tier", "fetched_at"):
            assert meta.get(field), f"{field} missing from stored metadata"

    def test_source_url_is_a_real_citation(self):
        """The citation must be a URL, since it is shown to the user."""
        result = get_collection().get(limit=5, include=["metadatas"])
        for meta in result["metadatas"]:
            assert meta["source_url"].startswith("http")

    def test_ids_are_unique(self):
        result = get_collection().get(include=[])
        ids = result["ids"]
        assert len(ids) == len(set(ids))


class TestConfig:
    def test_threshold_is_calibrated(self):
        assert 0.5 <= config.CONFIG.score_threshold <= 0.75

    def test_groq_key_never_in_summary(self):
        assert "gsk_" not in str(config.summary())
