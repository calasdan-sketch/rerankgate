"""Faithfulness judges: heuristic default, LLM-backed opt-in."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from rerankgate.retrieval import ScoredChunk
from rerankgate.tokenize import content_tokens

_SENT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class FaithfulnessReport:
    score: float  # mean per-sentence grounding, [0,1]
    unsupported: list[str] = field(default_factory=list)  # sentences below the floor
    per_sentence: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "unsupported": self.unsupported,
            "per_sentence": self.per_sentence,
        }


class Judge(Protocol):
    name: str

    def check(self, answer: str, evidence: list[ScoredChunk]) -> FaithfulnessReport: ...


class HeuristicJudge:
    """Ground every answer sentence in the evidence via token precision.

    A sentence's grounding is the largest fraction of its content tokens that
    appear in any single evidence chunk. A sentence is 'supported' when that
    fraction reaches the floor (0.35 by default). Heuristic — documented as
    such — but free, instant, and dependency-free.

    (rerankgate <= 0.2 used token F1 against the whole chunk, which punished
    short, correct sentences quoted from long chunks.)
    """

    name = "heuristic"

    def __init__(self, support_floor: float = 0.35) -> None:
        self.support_floor = support_floor

    def check(self, answer: str, evidence: list[ScoredChunk]) -> FaithfulnessReport:
        sentences = [s.strip() for s in _SENT.split(answer.strip()) if s.strip()]
        ev_sets = [set(content_tokens(c.chunk.text)) for c in evidence]
        per_sentence: list[dict] = []
        unsupported: list[str] = []
        for s in sentences:
            s_toks = set(content_tokens(s))
            best = 0.0
            for ev in ev_sets:
                if not s_toks or not ev:
                    continue
                best = max(best, len(s_toks & ev) / len(s_toks))
            per_sentence.append({"sentence": s, "grounding": round(best, 4)})
            if best < self.support_floor:
                unsupported.append(s)
        if per_sentence:
            score = sum(p["grounding"] for p in per_sentence) / len(per_sentence)
        else:
            score = 0.0
        return FaithfulnessReport(
            score=round(score, 4), unsupported=unsupported, per_sentence=per_sentence
        )


class LLMJudge:
    """Wrap any callable as a model-based judge: fn(answer, evidence_texts) -> float in [0,1]."""

    name = "llm"

    def __init__(self, fn: Callable[[str, list[str]], float]) -> None:
        self._fn = fn

    def check(self, answer: str, evidence: list[ScoredChunk]) -> FaithfulnessReport:
        import math

        try:
            score = float(self._fn(answer, [c.chunk.text for c in evidence]))
        except (TypeError, ValueError):
            score = 0.0  # unparseable judge output fails closed
        if not math.isfinite(score):
            score = 0.0  # NaN/inf must never count as "faithful"
        score = max(0.0, min(1.0, score))
        return FaithfulnessReport(score=round(score, 4))
