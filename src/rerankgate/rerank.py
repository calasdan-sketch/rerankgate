"""Rerankers: zero-dependency feature reranker (default) + cross-encoder (opt-in)."""

from __future__ import annotations

from time import perf_counter
from typing import Protocol

from rerankgate.retrieval import ScoredChunk
from rerankgate.tokenize import content_tokens, tokenize


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]: ...


class FeatureReranker:
    """Deterministic, explainable reranker. No model required.

    score = 0.5 * norm(fusion) + 0.3 * term_coverage + 0.2 * phrase_bonus
    """

    name = "feature"

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        t0 = perf_counter()
        q_terms = set(content_tokens(query))
        q_phrases = _phrases(query)
        max_fused = max((c.score for c in candidates), default=1.0) or 1.0

        rescored: list[ScoredChunk] = []
        for c in candidates:
            toks = set(tokenize(c.chunk.text))
            coverage = len(q_terms & toks) / len(q_terms) if q_terms else 0.0
            text_l = c.chunk.text.lower()
            phrase_bonus = 1.0 if q_phrases and any(p in text_l for p in q_phrases) else 0.0
            norm_fused = c.score / max_fused
            score = 0.5 * norm_fused + 0.3 * coverage + 0.2 * phrase_bonus
            rescored.append(
                ScoredChunk(
                    chunk=c.chunk,
                    score=score,
                    sources={
                        **c.sources,
                        "rerank": round(score, 4),
                        "coverage": round(coverage, 4),
                        "phrase": phrase_bonus,
                    },
                )
            )
        rescored.sort(key=lambda c: c.score, reverse=True)
        self.last_ms = round((perf_counter() - t0) * 1000, 2)
        return rescored[:k]


def _phrases(query: str) -> list[str]:
    toks = content_tokens(query)
    return [" ".join(toks[i : i + 2]) for i in range(len(toks) - 1)]


class CrossEncoderReranker:
    """Cross-encoder reranker (ms-marco MiniLM). Requires `rerankgate[local]`."""

    name = "cross-encoder"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install it with `pip install rerankgate[local]`."
            ) from exc
        self._model = CrossEncoder(model_name)
        self.last_ms = 0.0

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        import math

        if not candidates:
            return []
        t0 = perf_counter()
        pairs = [(query, c.chunk.text) for c in candidates]
        logits = self._model.predict(pairs)
        rescored = []
        for c, logit in zip(candidates, logits, strict=True):
            score = 1 / (1 + math.exp(-float(logit)))  # sigmoid to [0,1]
            rescored.append(
                ScoredChunk(chunk=c.chunk, score=score, sources={**c.sources, "rerank": score})
            )
        rescored.sort(key=lambda c: c.score, reverse=True)
        self.last_ms = round((perf_counter() - t0) * 1000, 2)
        return rescored[:k]
