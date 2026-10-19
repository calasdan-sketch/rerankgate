"""Heuristic sufficiency gate: "is this retrieval enough to answer?".

Works WITHOUT any LLM call. Every decision carries machine-readable reason
codes plus a human explanation, so a rejection can be audited and the
threshold tuned with `rerankgate eval`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from rerankgate.config import GateConfig
from rerankgate.retrieval import ScoredChunk
from rerankgate.tokenize import content_tokens, tokenize

# Reason codes (stable strings; safe to match on in calling code).
PASS = "PASS"
LOW_TOP_SCORE = "LOW_TOP_SCORE"
LOW_COVERAGE = "LOW_COVERAGE"
NO_MARGIN = "NO_MARGIN"
NO_CONSENSUS = "NO_CONSENSUS"
NO_EVIDENCE = "NO_EVIDENCE"

_EXPLANATIONS = {
    PASS: "Retrieved evidence is sufficient to answer.",
    LOW_TOP_SCORE: "Best chunk score is below the minimum — nothing looks strongly relevant.",
    LOW_COVERAGE: "Too few query terms appear in the evidence — likely off-topic retrieval.",
    NO_MARGIN: "Top chunks score almost identically — no clear winner, ambiguous retrieval.",
    NO_CONSENSUS: "Top chunks don't corroborate each other — evidence is fragmented.",
    NO_EVIDENCE: "Retrieval returned no candidates at all.",
}


@dataclass
class GateFeatures:
    top_score: float = 0.0  # normalized top-1 rerank score [0,1]
    coverage: float = 0.0  # fraction of query content-terms in evidence
    margin: float = 0.0  # (s0 - s1) / s0 — decisiveness of the ranking
    support: float = 0.0  # mean pairwise token overlap of top-3 chunks
    depth: int = 0  # chunks above the score floor

    def to_dict(self) -> dict:
        return {
            "top_score": round(self.top_score, 4),
            "coverage": round(self.coverage, 4),
            "margin": round(self.margin, 4),
            "support": round(self.support, 4),
            "depth": self.depth,
        }


@dataclass
class GateDecision:
    passed: bool
    score: float
    features: GateFeatures
    reasons: list[str] = field(default_factory=list)
    explanation: str = ""
    threshold: float = 0.0
    ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "score": round(self.score, 4),
            "threshold": self.threshold,
            "reasons": self.reasons,
            "explanation": self.explanation,
            "features": self.features.to_dict(),
            "ms": self.ms,
        }


class SufficiencyGate:
    """Decide whether reranked evidence is sufficient to answer the query."""

    def __init__(self, config: GateConfig | None = None) -> None:
        self.config = config or GateConfig()

    def evaluate(self, query: str, evidence: list[ScoredChunk]) -> GateDecision:
        t0 = perf_counter()
        cfg = self.config
        features = self._features(query, evidence)
        reasons: list[str] = []

        if not evidence:
            return GateDecision(
                passed=False,
                score=0.0,
                features=features,
                reasons=[NO_EVIDENCE],
                explanation=_EXPLANATIONS[NO_EVIDENCE],
                threshold=cfg.threshold,
                ms=_ms(t0),
            )

        # Hard floors first: these fail regardless of the weighted score.
        if features.top_score < cfg.min_top_score:
            reasons.append(LOW_TOP_SCORE)
        if features.coverage < cfg.min_coverage:
            reasons.append(LOW_COVERAGE)

        score = (
            cfg.w_top_score * features.top_score
            + cfg.w_coverage * features.coverage
            + cfg.w_margin * features.margin
            + cfg.w_support * features.support
        )

        passed = score >= cfg.threshold and not reasons
        if passed:
            reasons = [PASS]
        else:
            # Soft diagnostics: explain *why* a near-miss failed.
            if features.margin < 0.05:
                reasons.append(NO_MARGIN)
            if features.support < 0.05 and features.depth >= 2:
                reasons.append(NO_CONSENSUS)
            reasons = reasons or [f"SCORE_BELOW_THRESHOLD ({score:.2f} < {cfg.threshold:.2f})"]

        explanation = " ".join(_EXPLANATIONS.get(r, r) for r in reasons)
        return GateDecision(
            passed=passed,
            score=round(score, 4),
            features=features,
            reasons=reasons,
            explanation=explanation,
            threshold=cfg.threshold,
            ms=_ms(t0),
        )

    def _features(self, query: str, evidence: list[ScoredChunk]) -> GateFeatures:
        q_terms = set(content_tokens(query))
        scores = [c.score for c in evidence]
        top = scores[0] if scores else 0.0
        # Normalize top score to [0,1] against the observed max for stability.
        norm_top = min(1.0, top)

        ev_toks: set[str] = set()
        for c in evidence:
            ev_toks |= set(tokenize(c.chunk.text))
        coverage = len(q_terms & ev_toks) / len(q_terms) if q_terms else 0.0

        margin = 0.0
        if len(scores) >= 2 and scores[0] > 0:
            margin = max(0.0, (scores[0] - scores[1]) / scores[0])

        support = 0.0
        top3 = evidence[:3]
        if len(top3) >= 2:
            pairs = 0
            acc = 0.0
            sets = [set(content_tokens(c.chunk.text)) for c in top3]
            for i in range(len(sets)):
                for j in range(i + 1, len(sets)):
                    a, b = sets[i], sets[j]
                    union = a | b
                    if union:
                        acc += len(a & b) / len(union)
                    pairs += 1
            support = acc / pairs if pairs else 0.0

        depth = sum(1 for s in scores if s >= 0.2)
        return GateFeatures(
            top_score=norm_top, coverage=coverage, margin=margin, support=support, depth=depth
        )


def _ms(t0: float) -> float:
    return round((perf_counter() - t0) * 1000, 2)
