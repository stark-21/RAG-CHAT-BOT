"""Retrieval-Augmented Generation package for the Mutual Fund FAQ Assistant.

Modules are added phase by phase (see docs/implementation.md):

    loader.py       Phase 2 - fetch + clean official pages
    chunking.py     Phase 2 - structure-aware split + metadata
    embeddings.py   Phase 3 - all-MiniLM-L6-v2
    vectorstore.py  Phase 3 - ChromaDB persistent store
    guard.py        Phase 4 - PII + advice-intent checks
    retrieval.py    Phase 5 - embed query + top-k
    prompt.py       Phase 5 - system rules + context assembly
    generator.py    Phase 5 - Groq call
    postprocess.py  Phase 5 - enforce citation/length/freshness

Import surface stays framework-free here: no Streamlit, no Groq SDK outside
generator.py, no network I/O outside loader.py.
"""
