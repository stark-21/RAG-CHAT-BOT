"""Phase 2 runner: load -> chunk -> inspect.

Usage:
    python run_phase2.py              # reuse snapshots in data/raw/ if present
    python run_phase2.py --refresh    # re-fetch every source
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import config
import sources
from rag.chunking import chunk_documents, write_chunks_report
from rag.loader import load_all


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2: loading & chunking")
    parser.add_argument("--refresh", action="store_true",
                        help="re-fetch sources instead of reusing snapshots")
    args = parser.parse_args()

    config.ensure_dirs()

    print("=" * 70)
    print("PHASE 2 - LOADING")
    print("=" * 70)

    problems = sources.validate_registry()
    if any("Missing" in p for p in problems):
        print("Registry has unresolved sources. Run `python sources.py`.")
    if any("NON-OFFICIAL" in p for p in problems):
        print(f"WARNING: {sum('NON-OFFICIAL' in p for p in problems)} source(s) are "
              f"tier 'aggregator', not official. See docs/sources.md.")

    corpus = sources.SOURCES + [s for s in sources.GENERAL_SOURCES if s.is_resolved]
    print(f"\nLoading {len(corpus)} source document(s)...")

    docs, errors = load_all(corpus, refresh=args.refresh)

    if errors:
        print(f"\n{len(errors)} source(s) failed:")
        for err in errors:
            print(f"  x {err}")

    if not docs:
        print("\nNo documents loaded. Nothing to chunk.")
        return 1

    total_chars = 0
    print(f"\n{'SCHEME':<40} {'CHARS':>8}")
    print("-" * 50)
    for doc in docs:
        total_chars += doc.char_count
        short = doc.scheme.replace(" - Direct Growth", "")
        print(f"{short:<40} {doc.char_count:>8,}")
    print("-" * 50)
    print(f"{'TOTAL':<40} {total_chars:>8,}")

    print()
    print("=" * 70)
    print("PHASE 2 - CHUNKING")
    print("=" * 70)

    chunks = chunk_documents(docs)
    print(f"Strategy   : label-anchored, context-windowed")
    print(f"Budget     : {config.CONFIG.chunk_size} chars, "
          f"overlap {config.CONFIG.chunk_overlap}")
    print(f"Chunks     : {len(chunks)}")

    if chunks:
        sizes = [c.char_count for c in chunks]
        print(f"Chunk size : min {min(sizes)}, max {max(sizes)}, "
              f"mean {sum(sizes) // len(sizes)}")

    per_scheme = Counter(c.scheme_short for c in chunks)
    print(f"\n{'SCHEME':<40} {'CHUNKS':>7}")
    print("-" * 50)
    for scheme, count in per_scheme.most_common():
        print(f"{scheme:<40} {count:>7}")
    print("-" * 50)
    print(f"{'TOTAL':<40} {len(chunks):>7}")

    per_section = Counter(c.section for c in chunks)
    print(f"\nDistinct sections captured: {len(per_section)}")
    for section, count in per_section.most_common(12):
        print(f"  {count:>3}  {section}")

    path = write_chunks_report(chunks)
    print(f"\nChunk dump written to: {path}")
    print(f"Raw text saved to   : {config.CONFIG.raw_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
