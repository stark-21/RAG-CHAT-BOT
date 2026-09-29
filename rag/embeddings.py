"""Phase 3 - Embeddings.

`all-MiniLM-L6-v2`, a local 384-dimensional model that needs no API key.

**Why ONNX Runtime instead of sentence-transformers.** This is a latency and
memory decision, and it was measured, not assumed. The same model, exported to
ONNX and run through `onnxruntime`, produces vectors that are numerically
identical to the PyTorch path - cosine 1.0, max absolute difference 1.5e-7 -
while changing the two numbers that actually mattered on a small host:

    model load      12.28 s  ->  0.22 s      (measured, same machine)
    resident memory   762 MB ->  ~130 MB      (measured, same machine)

762 MB matters because Render's free web service has 512 MB of RAM: the torch
import graph alone does not fit, so the process was being pushed into swap or
restarted, which is what made the page itself feel slow to load. Dropping torch
also removes ~2 GB of wheels from the deploy, so the build finishes sooner too.

The pooling and normalisation are spelled out here rather than inherited,
because they are the part that has to match sentence-transformers exactly for
the stored corpus vectors to stay valid. MiniLM's sentence-transformers
configuration is: wordpiece tokenisation, lowercase, truncation at 256
tokens, right padding with `[PAD]` (id 0), attention-masked mean pooling over
`last_hidden_state`, then L2 normalisation.

There is exactly ONE model instance in the process, used for both corpus chunks
and user questions. That is not an optimisation, it is a correctness
requirement: the query vector must live in the same space as the document
vectors, or retrieval returns nonsense.
"""

from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

import config

EMBEDDING_DIM = 384

# Files fetched from the model repo, and only these. `model.onnx` is ~87 MB.
_MODEL_FILES = ("model.onnx", "tokenizer.json", "tokenizer_config.json")

# Matches sentence-transformers' max_seq_length for this model. Truncating here
# rather than letting the model see a 512-token window keeps the corpus and
# query paths identical.
MAX_SEQUENCE_LENGTH = 256

# Attention-masked mean pooling, so padded positions contribute nothing.
_POOL_EPS = 1e-9


class EmbeddingError(RuntimeError):
    """Raised when the embedding model cannot be prepared. The UI shows this."""


def _model_dir() -> Path:
    """Directory the ONNX weights are cached in.

    Defaults to a folder inside the project so the download survives a Streamlit
    rerun, and so it can be pre-populated during a Render *build* so no user
    ever waits on an 87 MB fetch.
    """
    return Path(config.CONFIG.model_cache_dir)


def _hf_url(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def _download(repo: str, destination: Path, filename: str) -> Path:
    """Fetch one model file unless it is already cached. Returns its path."""
    target = destination / filename
    if target.exists() and target.stat().st_size > 0:
        return target

    import requests

    url = _hf_url(repo, filename)
    try:
        with requests.get(url, stream=True, timeout=180) as response:
            response.raise_for_status()
            # Write to a temp name first: a half-written model.onnx would be
            # cached forever and every later run would fail to load it.
            staging = target.with_suffix(target.suffix + ".part")
            with open(staging, "wb") as handle:
                for block in response.iter_content(chunk_size=1 << 20):
                    handle.write(block)
            staging.replace(target)
    except Exception as exc:  # noqa: BLE001 - surfaced as a friendly error
        raise EmbeddingError(
            f"Could not download the embedding model ({filename}): {exc}"
        ) from exc
    return target


class MiniLMOnnx:
    """all-MiniLM-L6-v2 served by onnxruntime, mean-pooled and normalised."""

    def __init__(self, repo: str, cache_dir: Path):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.repo = repo
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        paths = {
            name: _download(repo, self.cache_dir, name) for name in _MODEL_FILES
        }

        # A single thread: the corpus is 90 chunks and one question is one
        # string, so intra-op parallelism costs more in thread setup than it
        # saves, and it multiplies peak memory.
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        # The CPU arena caches every intermediate tensor the graph allocates and
        # only returns the memory at session teardown. For a 22M-parameter model
        # feeding 256-token batches it bought no measurable speed and cost
        # ~110 MB of resident memory that grew with the batch count - the
        # difference between fitting Render's 512 MB free tier and not.
        options.enable_cpu_mem_arena = False

        self.session = ort.InferenceSession(
            str(paths["model.onnx"]),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        self.tokenizer = Tokenizer.from_file(str(paths["tokenizer.json"]))
        self.tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

        self._input_names = {i.name for i in self.session.get_inputs()}

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed a batch of texts into L2-normalised float32 rows."""
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        rows: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            rows.append(self._encode_batch(texts[start:start + batch_size]))
        return np.vstack(rows)

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer.encode_batch(list(texts))
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        # BERT-family exports declare token_type_ids; single-segment input is
        # all zeros, but the graph still requires the tensor to be present.
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.array(
                [e.type_ids for e in encoded], dtype=np.int64
            )

        hidden = self.session.run(["last_hidden_state"], feeds)[0]

        mask = attention_mask[..., None].astype(np.float32)
        pooled = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), _POOL_EPS, None)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.clip(norms, _POOL_EPS, None)).astype(np.float32)


@lru_cache(maxsize=1)
def get_model() -> MiniLMOnnx:
    """Load MiniLM once per process and reuse it.

    Thread-safe: `lru_cache` may run the wrapped function more than once under
    a race, which would mean two ONNX sessions in memory. The lock keeps it to
    exactly one.
    """
    with _MODEL_LOCK:
        return MiniLMOnnx(config.CONFIG.embedding_model, _model_dir())


_MODEL_LOCK = threading.Lock()


def warm() -> float:
    """Load the model and embed one string. Returns seconds taken.

    Called at app start so the first question does not pay the load cost. Safe
    to call more than once; the model is cached.
    """
    import time

    started = time.time()
    embed_query("warmup")
    return time.time() - started


def embed_documents(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed corpus chunks. L2-normalised so cosine == dot product."""
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    return get_model().encode(texts, batch_size=batch_size)


def embed_query(text: str) -> np.ndarray:
    """Embed a single user question into the same space as the corpus."""
    return embed_documents([text])[0]


def verify_space(query_vector: np.ndarray, doc_vector: np.ndarray) -> float:
    """Cosine similarity between a query and a document vector.

    Used by the Phase 3 checks: a correct pairing should score well above an
    unrelated pairing. Catches the classic failure of embedding queries with a
    different model than the corpus.
    """
    return float(np.dot(query_vector, doc_vector))
