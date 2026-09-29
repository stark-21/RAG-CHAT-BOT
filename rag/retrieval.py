"""Phase 5 - Retrieval.

Embeds the user's question with the SAME model instance used in Phase 3, then
pulls the top-k chunks from the persistent ChromaDB collection.

The shared model is not an optimisation. If the question were embedded with a
different model, its vector would not live in the corpus vector space and every
retrieval would be noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import config
from rag.embeddings import embed_query
from rag.vectorstore import get_collection


@dataclass
class ScoredChunk:
    """One retrieved chunk, with everything the answer needs."""

    chunk_id: str
    text: str
    distance: float
    section: str
    scheme: str
    scheme_short: str
    doc_type: str
    source_url: str
    source_tier: str
    fetched_at: str

    @property
    def score(self) -> float:
        """Similarity, where higher is closer. `1 - distance`."""
        return 1.0 - self.distance

    @property
    def context_block(self) -> str:
        """The block shown to the LLM and used by the grounding gate.

        The section label is included because it exists in the embed header but
        NOT in the stored document text. The Phase 4 concept check relies on it.
        """
        return f"{self.section}\n{self.text}"

    @property
    def citation_label(self) -> str:
        return f"{self.scheme_short} - {self.section}"


def _query(question: str, top_k: int) -> list[ScoredChunk]:
    vector = embed_query(question)

    result = get_collection().query(
        query_embeddings=[list(map(float, vector))],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    ids = (result.get("ids") or [[]])[0]

    chunks: list[ScoredChunk] = []
    for chunk_id, doc, meta, distance in zip(ids, documents, metadatas, distances):
        meta = meta or {}
        chunks.append(
            ScoredChunk(
                chunk_id=chunk_id,
                text=doc,
                distance=float(distance),
                section=meta.get("section", ""),
                scheme=meta.get("scheme", ""),
                scheme_short=meta.get("scheme_short", ""),
                doc_type=meta.get("doc_type", ""),
                source_url=meta.get("source_url", ""),
                source_tier=meta.get("source_tier", ""),
                fetched_at=meta.get("fetched_at", ""),
            )
        )
    return chunks


# Bounded so a long session cannot grow this without limit. Small on purpose:
# the wins are the repeat question and the follow-up that resolves to a
# question already asked, not general reuse.
@lru_cache(maxsize=64)
def _cached_query(question: str, top_k: int) -> tuple[ScoredChunk, ...]:
    return tuple(_query(question, top_k))


def clear_cache() -> None:
    """Drop cached retrieval results. Called after the store is rebuilt."""
    _cached_query.cache_clear()


def retrieve(question: str, k: int | None = None) -> list[ScoredChunk]:
    """Embed the question and return the top-k chunks, closest first.

    Cached on (question, k). The corpus is immutable for the life of the
    process, so two identical questions cannot retrieve differently, and
    returning a copy keeps a caller from mutating the cached entry.
    """
    top_k = k or config.CONFIG.top_k
    return list(_cached_query(question.strip(), top_k))


def best_distance(chunks: list[ScoredChunk]) -> float | None:
    """Cosine distance of the closest chunk, or None if nothing retrieved."""
    return min((c.distance for c in chunks), default=None)


def grounding_context(chunks: list[ScoredChunk]) -> list[str]:
    """Context blocks for the Phase 4 concept check.

    Every retrieved chunk, not just the top one - a fact can legitimately live
    in chunk 3 of 5.
    """
    return [chunk.context_block for chunk in chunks]


def freshness_date(chunks: list[ScoredChunk]) -> str:
    """Most recent fetched_at across retrieved chunks, for the freshness line."""
    dates = [c.fetched_at for c in chunks if c.fetched_at]
    return max(dates) if dates else ""
