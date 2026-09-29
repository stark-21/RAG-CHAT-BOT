"""Phase 4 verification - runs the full guard eval set with real retrieval."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from rag.embeddings import embed_query
from rag.guard import ACTION_ANSWER, check_grounding, inspect
from rag.vectorstore import get_collection

CASES = json.loads(Path("evals/guard_cases.json").read_text(encoding="utf-8"))


def retrieve(question: str):
    v = embed_query(question)
    res = get_collection().query(
        query_embeddings=[list(map(float, v))], n_results=config.CONFIG.top_k
    )
    return res["distances"][0][0], res


def main() -> int:
    collection = get_collection()
    print("=" * 74)
    print("PHASE 4 VERIFICATION")
    print("=" * 74)
    print(f"corpus vectors : {collection.count()}")
    print(f"threshold      : {config.CONFIG.score_threshold} (cosine distance)")
    print()

    failures = 0

    for group in ("opinion", "pii", "offtopic"):
        cases = CASES[group]
        print(f"--- {group.upper()} ({len(cases)} cases) ---")
        for case in cases:
            v = inspect(case["query"])
            ok = v.action == case["expect"]
            failures += not ok
            link = f"  link={v.link[:38]}" if v.link else ""
            print(f"  [{'PASS' if ok else 'FAIL'}] {v.action:<18}{link}")
            print(f"         {case['query'][:68]}")
        print()

    print("--- ANSWERABLE (guard pass, then grounding gate) ---")
    for case in CASES["answerable"]:
        v = inspect(case["query"])
        if v.action != ACTION_ANSWER:
            failures += 1
            print(f"  [FAIL] guard blocked: {case['query'][:60]}")
            continue
        best, res = retrieve(case["query"])
        # Grounding haystack: section label + body for EVERY retrieved chunk.
        blocks = [
            f"{m.get('section', '')}\n{d}"
            for m, d in zip(res["metadatas"][0], res["documents"][0])
        ]
        g = check_grounding(best, case["query"], blocks)
        hit = res["metadatas"][0][0]
        status = "ANSWER" if g.action == ACTION_ANSWER else "I DON'T KNOW"
        print(f"  {best:.3f}  {status:<13} {hit['scheme_short'][:24]:<24} "
              f"{hit['section'][:18]:<18} | {case['query'][:34]}")

    print()
    print("=" * 74)
    print("FAILURES:", failures)
    print("=" * 74)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
