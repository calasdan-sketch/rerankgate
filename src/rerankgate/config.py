"""Configuration dataclasses for rerankgate."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetryConfig:
    """Bounds for the query-rewrite retry loop."""

    max_retries: int = 2
    # Stop early if the best gate score does not improve by at least this much.
    min_improvement: float = 0.01
    # How many expansion terms the RM3-style rewriter may add per retry.
    expansion_terms: int = 5


@dataclass
class GateConfig:
    """Tunable knobs for the heuristic sufficiency gate.

    The gate score is a fixed weighted sum of interpretable features
    (see gate.py). Nothing is learned, so there is nothing to overfit;
    tune the threshold with `rerankgate eval` on your own labeled data.
    """

    threshold: float = 0.55
    w_top_score: float = 0.35
    w_coverage: float = 0.30
    w_margin: float = 0.20
    w_support: float = 0.15
    # Minimum fraction of (non-stopword) query terms that must appear
    # in the evidence before the gate can pass, regardless of score.
    min_coverage: float = 0.25
    # Minimum normalized top-1 rerank score before the gate can pass.
    min_top_score: float = 0.10

    def preset(self, name: str) -> GateConfig:
        """Return a copy with a named strictness preset applied."""
        presets = {
            # (threshold, min_coverage): stricter gates demand more evidence.
            "strict": (0.70, 0.35),
            "balanced": (0.55, 0.25),
            "lenient": (0.40, 0.15),
        }
        if name not in presets:
            raise ValueError(f"unknown gate preset {name!r}; choose from {sorted(presets)}")
        import dataclasses

        threshold, min_coverage = presets[name]
        return dataclasses.replace(self, threshold=threshold, min_coverage=min_coverage)


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    # Retrieval
    retrieve_k: int = 100  # candidates pulled from each retriever before fusion
    rerank_k: int = 10  # evidence chunks returned after reranking
    rrf_k: int = 60  # RRF rank constant
    bm25_weight: float = 1.0  # per-system RRF multipliers
    vector_weight: float = 1.0
    use_vectors: bool = True  # degraded to BM25-only if embeddings unavailable

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Gate + retry
    gate: GateConfig = field(default_factory=GateConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    gate_preset: str = "balanced"

    def effective_gate(self) -> GateConfig:
        return self.gate.preset(self.gate_preset)
