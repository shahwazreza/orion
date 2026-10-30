"""Text embeddings for matching by meaning (memories, conversations, rules).

Runs locally through Ollama (nomic-embed-text, 768 dimensions). nomic expects
a task prefix: "search_document: " for things stored, "search_query: " for
the question being matched against them.
"""

from __future__ import annotations

import logging
from typing import Protocol

import httpx
import numpy as np

from core.net_guard import require_loopback_url

log = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    pass


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[np.ndarray]: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class OllamaEmbedder:
    def __init__(self, host: str = "http://127.0.0.1:11434", model: str = "nomic-embed-text",
                 timeout_s: float = 30.0, client: httpx.Client | None = None) -> None:
        require_loopback_url(host)
        self.host = host.rstrip("/")
        self.model = model
        self._client = client or httpx.Client(timeout=httpx.Timeout(timeout_s, connect=3.0),
                                              trust_env=False)
        # Memory, conversation and rule lookups embed the same request: compute it once.
        self._query_cache: dict[str, np.ndarray] = {}

    def _embed(self, inputs: list[str]) -> list[np.ndarray]:
        try:
            r = self._client.post(f"{self.host}/api/embed", json={"model": self.model, "input": inputs})
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc
        return [unit(np.asarray(v, dtype=np.float32)) for v in r.json()["embeddings"]]

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return self._embed([f"search_document: {t}" for t in texts]) if texts else []

    def embed_query(self, text: str) -> np.ndarray:
        hit = self._query_cache.get(text)
        if hit is None:
            hit = self._embed([f"search_query: {text}"])[0]
            if len(self._query_cache) > 64:
                self._query_cache.clear()
            self._query_cache[text] = hit
        return hit


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) or 1.0)


def to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def from_blob(b: bytes | None) -> np.ndarray | None:
    return None if b is None else np.frombuffer(b, dtype=np.float32)


def rank(query: np.ndarray, rows: list[tuple[object, bytes | None]], k: int = 5,
         min_score: float = 0.0) -> list[tuple[float, object]]:
    """Top ``k`` (score, item) by cosine similarity, from (item, embedding blob) rows."""
    scored = []
    for item, blob in rows:
        v = from_blob(blob)
        if v is not None:
            s = float(query @ v)
            if s >= min_score:
                scored.append((s, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]
