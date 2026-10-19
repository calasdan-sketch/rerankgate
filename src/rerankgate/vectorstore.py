"""Cosine-similarity vector store over chunk embeddings.

Uses numpy when available; falls back to pure Python otherwise.
Vectors are expected to be L2-normalized (cosine == dot product).
"""

from __future__ import annotations

import math

from rerankgate.documents import Chunk

try:
    import numpy as _np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _np = None  # type: ignore
    _HAS_NUMPY = False


class VectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def vectors(self) -> list[list[float]]:
        """The stored (normalized) vectors, in chunk order."""
        return [list(v) for v in self._vectors]

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have equal length")
        for v in vectors:
            if len(v) != self.dim:
                raise ValueError(f"expected dim {self.dim}, got {len(v)}")
        self._chunks.extend(chunks)
        self._vectors.extend([list(map(float, v)) for v in vectors])

    def search(self, query_vec: list[float], k: int = 100) -> list[tuple[Chunk, float]]:
        """Top-k (chunk, cosine similarity) pairs, highest first."""
        if not self._chunks:
            return []
        if _HAS_NUMPY:
            mat = _np.asarray(self._vectors, dtype=_np.float64)
            q = _np.asarray(query_vec, dtype=_np.float64)
            qn = _np.linalg.norm(q) or 1.0
            norms = _np.linalg.norm(mat, axis=1)
            norms[norms == 0] = 1.0
            sims = (mat @ q) / (norms * qn)
            order = _np.argsort(-sims, kind="stable")[:k]
            return [(self._chunks[i], float(sims[i])) for i in order]
        # Pure-python fallback.
        qn = math.sqrt(sum(x * x for x in query_vec)) or 1.0
        scored = []
        for ch, v in zip(self._chunks, self._vectors, strict=True):
            vn = math.sqrt(sum(x * x for x in v)) or 1.0
            sim = sum(a * b for a, b in zip(v, query_vec, strict=True)) / (vn * qn)
            scored.append((ch, sim))
        scored.sort(key=lambda p: p[1], reverse=True)
        return scored[:k]
