"""Interactive CLI for testing the chatbot.

    python cli.py                 # interactive
    python cli.py "your question" # one-shot

Shows the retrieved chunks for every question, so you can see *why* the bot
answered the way it did.
"""

from __future__ import annotations

import sys

import config
from rag import generator
from rag.memory import Conversation
from rag.pipeline import answer_question

BANNER = """
Mutual Fund FAQ Assistant - facts-only RAG demo
Type a question, or 'quit' to exit. 'clear' starts a new conversation.
Remembers the last 10 messages, so follow-ups work:
  > What is the exit load of HDFC Small Cap Fund?
  > what about its fees?
Try:  Should I buy HDFC ELSS Tax Saver Fund?     (refused: advice)
      What is the lock-in period for ELSS?       ("I don't know": not in corpus)
      What's the weather?                        (refused: off-topic)
"""


def show_chunks(chunks, limit: int = 3) -> None:
    if not chunks:
        return
    print("\n  -- retrieved chunks " + "-" * 46)
    for i, chunk in enumerate(chunks[:limit], start=1):
        print(f"  [{i}] {chunk.scheme_short} / {chunk.section}")
        print(f"      distance {chunk.distance:.3f}  (lower = closer)")
        print(f"      source   {chunk.source_url}")
        body = chunk.text.replace("\n", " ")
        print(f"      text     {body[:150]}{'...' if len(body) > 150 else ''}")
    if len(chunks) > limit:
        print(f"      ... and {len(chunks) - limit} more")


def handle(question: str, conversation: Conversation) -> None:
    print(f"\n> {question}")
    try:
        result = answer_question(question, conversation=conversation)
    except generator.GeneratorError as exc:
        print(f"\n  [error] {exc}")
        return

    # Show the rewrite, so the resolution is visible and checkable.
    if result.rewritten:
        print(f"\n  [memory] interpreted as: {result.resolved_question}")

    show_chunks(result.chunks)

    print("\n  " + "=" * 62)
    for line in result.answer.text.splitlines() or [""]:
        print(f"  {line}")
    if result.answer.source_url:
        print(f"  Source: {result.answer.source_url}")
    if result.answer.freshness:
        print(f"  Last updated from sources: {result.answer.freshness}")
    if result.answer.link:
        print(f"  Learn more: {result.answer.link}")
    print("  " + "=" * 62)

    print(f"  stage: {result.stage}   latency: {result.latency_s:.2f}s   "
          f"memory: {len(conversation)} messages")
    for warning in result.answer.warnings:
        print(f"  ! {warning}")


def main() -> int:
    missing = config.missing_env()
    if missing:
        print(f"Missing .env entries: {', '.join(missing)}")
        print("Copy .env.example to .env and fill them in.")
        return 1

    conversation = Conversation()

    if len(sys.argv) > 1:
        handle(" ".join(sys.argv[1:]), conversation)
        return 0

    print(BANNER)
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0
        if not question:
            continue
        if question.lower() in {"quit", "exit", ":q"}:
            print("Bye.")
            return 0
        if question.lower() in {"clear", "reset", "new"}:
            conversation.clear()
            print("  conversation cleared.")
            continue
        handle(question, conversation)


if __name__ == "__main__":
    sys.exit(main())
