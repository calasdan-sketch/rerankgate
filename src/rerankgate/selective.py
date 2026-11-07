"""Selective-prediction metrics: answered-accuracy vs coverage, risk-coverage AUC.

``scores`` rank how willing the system is to answer (higher = answer); ``correct``
says whether answering that query would have been right. For an unanswerable query
answering is never correct. Ties are handled jointly (no arbitrary tie order).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def _validate(scores: Sequence[float], correct: Sequence[bool]) -> None:
    if len(scores) != len(correct):
        raise ValueError("scores and correct must have the same length")
    if not scores:
        raise ValueError("need at least one item")
    for s in scores:
        if isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s):
            raise ValueError(f"non-finite score {s!r}: fix or drop it before computing metrics")


def risk_coverage_curve(
    scores: Sequence[float], correct: Sequence[bool]
) -> list[tuple[float, float]]:
    """[(coverage, risk)] at each distinct score level, answering everything >= that level."""
    _validate(scores, correct)
    n = len(scores)
    order = sorted(range(n), key=lambda i: -scores[i])
    pts: list[tuple[float, float]] = []
    answered = wrong = 0
    i = 0
    while i < n:
        j = i
        while j < n and scores[order[j]] == scores[order[i]]:
            answered += 1
            wrong += 0 if correct[order[j]] else 1
            j += 1
        pts.append((answered / n, wrong / answered))
        i = j
    return pts


def aurc(scores: Sequence[float], correct: Sequence[bool]) -> float:
    """Area under the risk-coverage curve = mean selective risk over k = 1..n (lower is better).

    Reference points: a random ranking gives the overall error rate; an oracle gives
    the minimum achievable (``oracle_aurc``). Tie groups use their group-end risk.
    """
    _validate(scores, correct)
    n = len(scores)
    order = sorted(range(n), key=lambda i: -scores[i])
    total = 0.0
    wrong = 0
    i = 0
    answered = 0
    while i < n:
        j = i
        while j < n and scores[order[j]] == scores[order[i]]:
            answered += 1
            wrong += 0 if correct[order[j]] else 1
            j += 1
        total += (j - i) * wrong / answered
        i = j
    return total / n


def oracle_aurc(correct: Sequence[bool]) -> float:
    """AURC of a perfect ranker (all correct items first)."""
    n = len(correct)
    if n == 0:
        raise ValueError("need at least one item")
    right = sum(1 for c in correct if c)
    total, wrong = 0.0, 0
    for k in range(1, n + 1):
        if k > right:
            wrong += 1
        total += wrong / k
    return total / n


def risk_at_coverage(scores: Sequence[float], correct: Sequence[bool], coverage: float) -> float:
    """Selective risk when answering the top ``coverage`` fraction (0 < coverage <= 1).

    Takes the largest tie-group boundary not exceeding the target coverage; 0.0 if none.
    """
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    best = None
    for cov, risk in risk_coverage_curve(scores, correct):
        if cov <= coverage + 1e-12:
            best = risk
    return 0.0 if best is None else best


@dataclass
class AbstentionStats:
    n: int
    coverage: float  # fraction answered
    far: float  # answered | unanswerable
    answer_precision: float  # answerable-and-correct among answered
    abstention_precision: float  # unanswerable among abstained
    abstention_recall: float  # abstained among unanswerable
    answerable_recall: float  # answered among answerable (a.k.a. gate recall)

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in self.__dict__.items()}


def abstention_stats(
    answered: Sequence[bool], answerable: Sequence[bool], correct: Sequence[bool] | None = None
) -> AbstentionStats:
    if len(answered) != len(answerable) or not answered:
        raise ValueError("answered/answerable must be non-empty and equal length")
    if correct is None:
        correct = answerable
    n = len(answered)
    n_ans = sum(answered)
    unans = [i for i in range(n) if not answerable[i]]
    ans_q = [i for i in range(n) if answerable[i]]
    abst = [i for i in range(n) if not answered[i]]

    def ratio(a: int, b: int) -> float:
        return a / b if b else 0.0

    return AbstentionStats(
        n=n,
        coverage=n_ans / n,
        far=ratio(sum(1 for i in unans if answered[i]), len(unans)),
        answer_precision=ratio(sum(1 for i in range(n) if answered[i] and correct[i]), n_ans),
        abstention_precision=ratio(sum(1 for i in abst if not answerable[i]), len(abst)),
        abstention_recall=ratio(sum(1 for i in unans if not answered[i]), len(unans)),
        answerable_recall=ratio(sum(1 for i in ans_q if answered[i]), len(ans_q)),
    )
