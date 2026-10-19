"""Pluggable embedding backends.

The default path is local-first and LLM-free. Heavy model dependencies
are lazy: `SentenceTransformerBackend` only imports sentence-transformers
when instantiated, and the pipeline degrades to BM25-only if embeddings
are unavailable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class EmbeddingBackend(Protocol):
    """Contract every embedding backend implements."""

    @property
    def dim(self) -> int: ...
    @property
    def name(self) -> str: ...
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerBackend:
    """Local dense embeddings via sentence-transformers (lazy import)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install it with `pip install rerankgate[local]`, or run "
                "the pipeline with use_vectors=False for BM25-only mode."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    @property
    def name(self) -> str:
        return f"sentence-transformers:{self._model_name}"

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vecs = self._model.encode(list(texts), normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]


class FakeEmbeddings:
    """Deterministic toy backend for tests — NOT for production use."""

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    @property
    def name(self) -> str:
        return "fake-test-embeddings"

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        import hashlib

        out: list[list[float]] = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            vec = [((h[i % len(h)] / 255.0) - 0.5) * 2 for i in range(self._dim)]
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            out.append([x / norm for x in vec])
        return out
