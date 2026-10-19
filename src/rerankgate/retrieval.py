"""Hybrid retrieval: BM25 + vector search fused with Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from rerankgate.bm25 import BM25Index
from rerankgate.documents import Chunk
from rerankgate.embeddings import EmbeddingBackend
from rerankgate.tokenize import tokenize
from rerankgate.vectorstore import VectorStore


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    sources: dict[str, float] = field(default_factory=dict)  # per-system raw scores


@dataclass
class RetrievalResult:
    chunks: list[ScoredChunk]
    metrics: dict[str, Any]


class HybridRetriever:
    """Fuses BM25 and vector rankings with RRF (rank-based, no score normalization)."""

    def __init__(
        self,
        bm25: BM25Index,
        vector_store: VectorStore | None = None,
        embeddings: EmbeddingBackend | None = None,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0,
    ) -> None:
        self.bm25 = bm25
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.degraded: str | None = None  # reason if running BM25-only

    @property
    def modes(self) -> list[str]:
        modes = ["bm25"]
        if self.vector_store is not None and self.embeddings is not None:
            modes.append("vector")
        return modes

    def retrieve(self, query: str, k: int = 100) -> RetrievalResult:
        t0 = perf_counter()
        rankings: list[tuple[str, list[tuple[Chunk, float]], float]] = []

        t_bm25 = perf_counter()
        bm25_hits = self.bm25.search(query, k=k)
        bm25_ms = (perf_counter() - t_bm25) * 1000
        rankings.append(("bm25", bm25_hits, self.bm25_weight))

        vector_ms = 0.0
        if self.vector_store is not None and self.embeddings is not None:
            t_vec = perf_counter()
            qvec = self.embeddings.embed([query])[0]
            vec_hits = self.vector_store.search(qvec, k=k)
            vector_ms = (perf_counter() - t_vec) * 1000
            rankings.append(("vector", vec_hits, self.vector_weight))

        # Reciprocal Rank Fusion over ranks (1-based).
        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        sources: dict[str, dict[str, float]] = {}
        for name, hits, weight in rankings:
            for rank, (ch, raw) in enumerate(hits, start=1):
                fused[ch.id] = fused.get(ch.id, 0.0) + weight / (self.rrf_k + rank)
                by_id[ch.id] = ch
                sources.setdefault(ch.id, {})[name] = raw

        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        chunks = [
            ScoredChunk(chunk=by_id[cid], score=s, sources=dict(sources[cid])) for cid, s in ordered
        ]
        total_ms = (perf_counter() - t0) * 1000
        metrics = {
            "modes": self.modes,
            "degraded": self.degraded,
            "k": k,
            "bm25_hits": len(bm25_hits),
            "bm25_ms": round(bm25_ms, 2),
            "vector_ms": round(vector_ms, 2),
            "total_ms": round(total_ms, 2),
            "fused_count": len(chunks),
            "query_terms": len(tokenize(query)),
        }
        return RetrievalResult(chunks=chunks, metrics=metrics)
