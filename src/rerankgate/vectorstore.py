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
        self._matrix = None  # cached numpy matrix of row-normalized vectors

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def vectors(self) -> list[list[float]]:
        """The stored (normalized) vectors, in chunk order."""
        return [list(v) for v in self._vectors]

    def validate(self, vectors: list[list[float]]) -> list[list[float]]:
        """Return float rows, or raise ValueError (wrong size, NaN/inf) without storing anything."""
        clean: list[list[float]] = []
        for v in vectors:
            if len(v) != self.dim:
                raise ValueError(f"expected dim {self.dim}, got {len(v)}")
            row = [float(x) for x in v]
            if not all(math.isfinite(x) for x in row):
                raise ValueError("embedding contains NaN or infinite values")
            clean.append(row)
        return clean

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have equal length")
        clean = self.validate(vectors)
        self._chunks.extend(chunks)
        self._vectors.extend(clean)
        self._matrix = None

    def remove_docs(self, doc_ids: set[str]) -> int:
        """Drop every chunk (and its vector) of the given documents; returns chunks removed."""
        keep = [i for i, c in enumerate(self._chunks) if c.doc_id not in doc_ids]
        removed = len(self._chunks) - len(keep)
        if removed:
            self._chunks = [self._chunks[i] for i in keep]
            self._vectors = [self._vectors[i] for i in keep]
            self._matrix = None
        return removed

    def search(self, query_vec: list[float], k: int = 100) -> list[tuple[Chunk, float]]:
        """Top-k (chunk, cosine similarity) pairs, highest first."""
        if not self._chunks or k <= 0:
            return []
        if len(query_vec) != self.dim:
            raise ValueError(f"query vector has dim {len(query_vec)}, expected {self.dim}")
        if _HAS_NUMPY:
            if self._matrix is None:
                mat = _np.asarray(self._vectors, dtype=_np.float64)
                norms = _np.linalg.norm(mat, axis=1)
                norms[norms == 0] = 1.0
                self._matrix = mat / norms[:, None]
            q = _np.asarray(query_vec, dtype=_np.float64)
            qn = _np.linalg.norm(q) or 1.0
            sims = (self._matrix @ q) / qn
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
