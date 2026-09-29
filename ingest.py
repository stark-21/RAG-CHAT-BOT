"""Ingestion CLI - run once.

    load -> chunk -> embed -> store

Writes three things:
  * data/raw/         cleaned source snapshots      (Phase 2 loader)
  * data/chunks/      inspectable chunk dump        (Phase 2 chunker)
  * data/chroma/      persisted ChromaDB vectors    (this phase)
  * data/embeddings_preview.txt  first 5 vectors, first 10 dims each

Re-running is safe: the collection is dropped and rebuilt, so counts never
double up.

The app builds the store by itself when it finds it empty, so this is only
needed to refresh the corpus or to inspect the vector space. It is NOT needed
in a platform start command any more - that put a full ingest in front of every
cold start, delaying the first byte the web server could serve.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

import config
import sources
from rag.chunking import chunk_documents, write_chunks_report
from rag.embeddings import EMBEDDING_DIM, embed_documents, embed_query
from rag.loader import load_all
from rag.vectorstore import get_collection, store_chunks
from rag import retrieval

PREVIEW_PATH = None  # resolved from config at call time


def write_embedding_preview(chunks, vectors, count: int = 5, dims: int = 10) -> str:
    """Write the first N embeddings, first D dimensions each."""
    target = config.CONFIG.data_dir / "embeddings_preview.txt"
    lines: list[str] = []
    lines.append("=" * 74)
    lines.append("EMBEDDING PREVIEW - Mutual Fund FAQ Assistant")
    lines.append(f"model    : {config.CONFIG.embedding_model}")
    lines.append(f"dim      : {EMBEDDING_DIM}")
    lines.append(f"showing  : first {count} embeddings, first {dims} dimensions")
    lines.append(f"vectors  : {len(vectors)}")
    lines.append("=" * 74)

    for i in range(min(count, len(vectors))):
        vector = np.asarray(vectors[i])
        chunk = chunks[i]
        head = vector[:dims]

        lines.append("")
        lines.append("-" * 74)
        lines.append(f"[{i + 1}] {chunk.chunk_id}")
        lines.append(f"  scheme  : {chunk.scheme_short}   section: {chunk.section}")
        lines.append(f"  source  : {chunk.source_url}")
        lines.append(f"  text    : {chunk.text[:90].replace(chr(10), ' ')}...")
        lines.append(f"  norm    : {float(np.linalg.norm(vector)):.6f}  "
                     f"(L2-normalised)")
        lines.append("-" * 74)
        lines.append(f"  first {dims} dims: " + "  ".join(f"{v:+.5f}" for v in head))

    lines.append("")
    lines.append("=" * 74)
    lines.append("END")
    lines.append("=" * 74)

    target.write_text("\n".join(lines), encoding="utf-8")
    return str(target)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest the corpus (run once)")
    parser.add_argument("--refresh", action="store_true",
                        help="re-fetch sources instead of reusing snapshots")
    args = parser.parse_args()

    config.ensure_dirs()

    print("=" * 70)
    print("INGEST - load -> chunk -> embed -> store")
    print("=" * 70)

    # --- 1. Load ---
    corpus = sources.SOURCES + [s for s in sources.GENERAL_SOURCES if s.is_resolved]
    print(f"\n[1/4] Loading {len(corpus)} source document(s)...")
    docs, errors = load_all(corpus, refresh=args.refresh)
    for err in errors:
        print(f"  x {err}")
    if not docs:
        print("  No documents loaded. Aborting.")
        return 1
    print(f"      {len(docs)} loaded, {sum(d.char_count for d in docs):,} chars")

    # --- 2. Chunk ---
    print("\n[2/4] Chunking...")
    chunks = chunk_documents(docs)
    if not chunks:
        print("  No chunks produced. Aborting.")
        return 1
    chunks_path = write_chunks_report(chunks)
    print(f"      {len(chunks)} chunks -> {chunks_path}")

    # --- 3. Embed ---
    print(f"\n[3/4] Embedding with {config.CONFIG.embedding_model}...")
    started = time.time()
    vectors = embed_documents([chunk.embed_text for chunk in chunks])
    elapsed = time.time() - started
    print(f"      shape {vectors.shape} in {elapsed:.1f}s "
          f"({elapsed / max(len(chunks), 1) * 1000:.0f} ms/chunk)")

    preview_path = write_embedding_preview(chunks, vectors)
    print(f"      preview -> {preview_path}")

    # --- 4. Store ---
    print("\n[4/4] Storing in ChromaDB (persistent)...")
    stored = store_chunks(chunks, vectors)
    print(f"      directory : {config.CONFIG.chroma_dir}")
    print(f"      collection: {config.CONFIG.chroma_dir.name}/")
    print(f"      VECTORS STORED: {stored}")

    collection = get_collection()
    print(f"      collection count (re-read): {collection.count()}")

    # The app caches both retrieval results and generated answers, and ingest
    # changed what they would return, so anything cached against the old store
    # is now wrong.
    from rag import pipeline

    retrieval.clear_cache()
    pipeline.clear_cache()

    # --- Space sanity check ---
    print("\n" + "=" * 70)
    print("SPACE CHECK - query and corpus must share one vector space")
    print("=" * 70)
    probe = "exit load of HDFC Small Cap Fund"
    qv = embed_query(probe)
    best = collection.query(query_embeddings=[list(map(float, qv))], n_results=1)
    hit_id = best["ids"][0][0]
    hit_meta = best["metadatas"][0][0]
    hit_dist = best["distances"][0][0]
    print(f"  query      : {probe!r}")
    print(f"  top hit    : {hit_id}")
    print(f"  scheme     : {hit_meta.get('scheme_short')}")
    print(f"  section    : {hit_meta.get('section')}")
    print(f"  distance   : {hit_dist:.4f}  (cosine; lower is closer)")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
