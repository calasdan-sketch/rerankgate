"""ScoreGate: a calibrated abstain/answer decision for ANY retriever's raw scores.

You do not need the rerankgate pipeline. Feed the ranked relevance scores your own stack
returns (BM25, cosine, RRF, a Cohere/BGE/Jina/ms-marco reranker, a vector DB's distance)
for a labelled set of queries; ``fit`` turns them into a decision rule whose false-answer
rate on unanswerable queries is bounded (see ``calibration`` for the exact guarantee).

Scores from different retrievers/corpora are not comparable, which is why a single
"cutoff = 0.7" copied from a blog post fails. The conformal p-value that ``decide``
returns is: it says how many known-unanswerable queries scored at least as high.

Statistics (computed from the descending-sorted scores): ``top1``; ``gap`` = s1 - s2
(a clear winner); ``zgap`` = (s1 - mean(rest)) / (std(rest) + eps) needs >= 4 scores.
``statistic="auto"`` picks by AURC on half of the labelled data, then calibrates the
threshold on the other half, so the guarantee is not contaminated by the selection.
"""

from __future__ import annotations

import bisect
import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from rerankgate import calibration as cal
from rerankgate.selective import aurc

STATS = ("top1", "gap", "zgap")
INVALID = "INVALID_SCORE"
EMPTY = "NO_EVIDENCE"
PASS = "PASS"
BELOW = "BELOW_CALIBRATED_THRESHOLD"
UNCALIBRATED = "NOT_ENOUGH_CALIBRATION_DATA"


def statistic_value(
    scores: Sequence[float], stat: str, higher_is_better: bool = True
) -> float | None:
    """The scalar summary of one query's scores, or None if the input is unusable."""
    if stat not in STATS:
        raise ValueError(f"unknown statistic {stat!r}; choose from {STATS}")
    vals: list[float] = []
    for s in scores:
        if isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s):
            return None
        vals.append(float(s) if higher_is_better else -float(s))
    if not vals:
        return None
    vals.sort(reverse=True)
    if stat == "top1":
        return vals[0]
    if stat == "gap":
        return vals[0] - vals[1] if len(vals) > 1 else None  # needs a runner-up
    rest = vals[1:]
    if len(rest) < 3:
        return None  # not enough context for a z-score: treat as unusable -> abstain
    sd = statistics.pstdev(rest)
    return (vals[0] - statistics.fmean(rest)) / (sd + 1e-9)


@dataclass
class ScoreDecision:
    passed: bool
    reason: str
    statistic: float | None
    p_value: float  # conformal p-value that the query is unanswerable (1.0 = no evidence)
    threshold: float | None

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "statistic": self.statistic,
            "p_value": round(self.p_value, 4),
            "threshold": self.threshold,
        }


@dataclass
class ScoreGate:
    statistic: str
    higher_is_better: bool
    alpha: float
    delta: float | None
    tau: float | None  # None = cannot guarantee -> never pass
    neg_stats: list[float]
    n_calibration: int
    recall_holdout: float | None = None  # measured on held-out answerable, if any

    @classmethod
    def fit(
        cls,
        score_lists: Sequence[Sequence[float]],
        answerable: Sequence[bool],
        alpha: float = 0.05,
        delta: float | None = None,
        statistic: str = "auto",
        higher_is_better: bool = True,
    ) -> ScoreGate:
        if len(score_lists) != len(answerable):
            raise ValueError("score_lists and answerable must have the same length")
        cal._check_prob("alpha", alpha)
        if delta is not None:
            cal._check_prob("delta", delta)
        if statistic != "auto" and statistic not in STATS:
            raise ValueError(f"unknown statistic {statistic!r}")
        idx = list(range(len(score_lists)))
        sel = idx[0::2] if statistic == "auto" else []
        calib = idx[1::2] if statistic == "auto" else idx
        stat = statistic
        if statistic == "auto":
            best = None
            for st in STATS:
                vals = [statistic_value(score_lists[i], st, higher_is_better) for i in sel]
                if any(v is None for v in vals) or not vals:
                    continue
                a = aurc(vals, [answerable[i] for i in sel])
                if best is None or a < best[0] - 1e-12:
                    best = (a, st)
            stat = best[1] if best else "top1"
        neg, pos = [], []
        for i in calib:
            v = statistic_value(score_lists[i], stat, higher_is_better)
            if v is None:
                # Unusable input is always abstained on at decide() time, so it can never
                # cause a false answer; leaving it out of calibration keeps the bound valid.
                continue
            (pos if answerable[i] else neg).append(v)
        if not neg:
            raise ValueError("fit() needs unanswerable queries to calibrate on")
        tau = (
            cal.conformal_threshold(neg, alpha)
            if delta is None
            else cal.pac_threshold(neg, alpha, delta)
        )
        recall = cal.recall_at(pos, tau) if pos and math.isfinite(tau) else (0.0 if pos else None)
        return cls(
            statistic=stat,
            higher_is_better=higher_is_better,
            alpha=alpha,
            delta=delta,
            tau=tau if math.isfinite(tau) else None,
            neg_stats=sorted(neg),
            n_calibration=len(neg),
            recall_holdout=recall,
        )

    def decide(self, scores: Sequence[float]) -> ScoreDecision:
        if not len(scores):
            return ScoreDecision(False, EMPTY, None, 1.0, self.tau)
        v = statistic_value(scores, self.statistic, self.higher_is_better)
        if v is None:
            return ScoreDecision(False, INVALID, None, 1.0, self.tau)
        # neg_stats is sorted ascending: #{neg >= v} by bisection (O(log n), safe for huge sets)
        p = (1 + len(self.neg_stats) - bisect.bisect_left(self.neg_stats, v)) / (
            len(self.neg_stats) + 1
        )
        if self.tau is None:
            return ScoreDecision(False, UNCALIBRATED, v, p, None)
        ok = v > self.tau
        return ScoreDecision(ok, PASS if ok else BELOW, v, p, self.tau)

    def to_json(self) -> str:
        return json.dumps(
            {
                "format": 1,
                "statistic": self.statistic,
                "higher_is_better": self.higher_is_better,
                "alpha": self.alpha,
                "delta": self.delta,
                "tau": self.tau,
                "neg_stats": self.neg_stats,
                "n_calibration": self.n_calibration,
            }
        )

    @classmethod
    def from_json(cls, text: str) -> ScoreGate:
        try:
            d = json.loads(text)
            if d["format"] != 1 or d["statistic"] not in STATS:
                raise ValueError("unsupported format or statistic")
            tau = None if d["tau"] is None else float(d["tau"])
            neg = [float(x) for x in d["neg_stats"]]
            g = cls(
                statistic=d["statistic"],
                higher_is_better=bool(d["higher_is_better"]),
                alpha=float(d["alpha"]),
                delta=None if d["delta"] is None else float(d["delta"]),
                tau=tau,
                neg_stats=neg,
                n_calibration=int(d["n_calibration"]),
            )
        except (KeyError, TypeError, ValueError, RecursionError) as exc:
            raise ValueError(f"malformed ScoreGate file: {exc}") from exc
        if (tau is not None and not math.isfinite(tau)) or not all(math.isfinite(x) for x in neg):
            raise ValueError("malformed ScoreGate file: non-finite numbers")
        cal._check_prob("alpha", g.alpha)
        return g
