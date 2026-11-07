"""Distribution-free threshold selection with a false-answer-rate guarantee.

Setting: you hold a calibration set of *unanswerable* queries (ones the corpus
cannot answer) and the gate score each one received. A later unanswerable query
is treated as exchangeable with them (same source, same corpus, no drift).

Two guarantees, both for  FAR = P(gate passes | query is unanswerable):

* ``conformal_threshold`` (marginal): E[FAR] <= alpha, averaged over calibration
  draws. Needs n >= ceil(1/alpha) - 1 unanswerable calibration queries.
* ``pac_threshold`` (high probability): with probability >= 1 - delta over the
  draw of the calibration set, FAR <= alpha. Standard order-statistic / tolerance
  bound (Wilks 1941; Vovk 2012; Angelopoulos et al. 2021 "Learn then Test" for the
  general framework). Needs n >= ln(delta)/ln(1 - alpha) queries (59 for 5%/5%).

Both pick tau and pass iff ``score > tau``. If the calibration set is too small
for the requested guarantee, tau = +inf: the gate never passes (fail closed).
Ties only make the guarantee more conservative. NaN/inf scores raise ValueError.

What is NOT guaranteed: anything about answerable queries (recall is measured, not
promised), anything under distribution shift (use ``monitor``), or FAR on queries
unlike the calibration ones. Calibrate on queries that look like your traffic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

INF = math.inf


def _check_scores(scores: Sequence[float], what: str) -> list[float]:
    out: list[float] = []
    for s in scores:
        if isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s):
            raise ValueError(f"{what} contains a non-finite or non-numeric score: {s!r}")
        out.append(float(s))
    return out


def _check_prob(name: str, v: float, *, open_right: bool = True) -> None:
    ok = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    if not ok or not (0.0 < v < 1.0 if open_right else 0.0 < v <= 1.0):
        raise ValueError(f"{name} must be strictly between 0 and 1, got {v!r}")


def min_negatives(alpha: float, delta: float | None = None) -> int:
    """Smallest number of unanswerable calibration queries for which the gate can pass at all."""
    _check_prob("alpha", alpha)
    if delta is None:
        return max(1, math.ceil(1.0 / alpha) - 1)
    _check_prob("delta", delta)
    return max(1, math.ceil(math.log(delta) / math.log1p(-alpha)))


def conformal_threshold(neg_scores: Sequence[float], alpha: float) -> float:
    """tau such that ``pass iff score > tau`` has E[FAR] <= alpha (split conformal)."""
    _check_prob("alpha", alpha)
    s = sorted(_check_scores(neg_scores, "neg_scores"), reverse=True)
    n = len(s)
    j = math.floor(alpha * (n + 1) + 1e-12) - 1  # allow at most j calibration scores >= s
    if j < 0:
        return INF
    if j + 1 > n:
        return -INF
    return s[j]


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(Binomial(n, p) <= k), exact, stable for n up to ~1e5 (log-space terms)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    total = 0.0
    for i in range(k + 1):
        total += math.exp(
            math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
        )
    return min(1.0, total)


def pac_threshold(neg_scores: Sequence[float], alpha: float, delta: float) -> float:
    """tau such that, with probability >= 1 - delta over the calibration draw, FAR <= alpha."""
    _check_prob("alpha", alpha)
    _check_prob("delta", delta)
    s = sorted(_check_scores(neg_scores, "neg_scores"), reverse=True)
    n = len(s)
    # tau = (j+1)-th largest score; FAR(tau) ~ Beta(j+1, n-j) for continuous scores, so
    # we need P(Bin(n, alpha) <= j) <= delta. Take the largest such j (most permissive).
    j = -1
    lp, lq = math.log(alpha), math.log1p(-alpha)
    cum = 0.0
    for cand in range(n):  # incremental binomial CDF: O(j) not O(j^2)
        cum += math.exp(
            math.lgamma(n + 1)
            - math.lgamma(cand + 1)
            - math.lgamma(n - cand + 1)
            + cand * lp
            + (n - cand) * lq
        )
        if cum <= delta:
            j = cand
        else:
            break
    if j < 0:
        return INF
    return s[j]


def far_at(neg_scores: Sequence[float], tau: float) -> float:
    """Observed false-answer rate of ``pass iff score > tau`` on these negatives."""
    s = _check_scores(neg_scores, "neg_scores")
    if not s:
        return 0.0
    return sum(1 for x in s if x > tau) / len(s)


def recall_at(pos_scores: Sequence[float], tau: float) -> float:
    s = _check_scores(pos_scores, "pos_scores")
    if not s:
        return 0.0
    return sum(1 for x in s if x > tau) / len(s)


def conformal_p_value(neg_scores: Sequence[float], score: float) -> float:
    """Conformal p-value of 'this query is unanswerable': (1 + #{neg >= score}) / (n + 1).

    Small = the score is higher than almost every known-unanswerable query.
    Non-finite scores get p = 1.0 (never evidence of answerability).
    """
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        return 1.0
    s = _check_scores(neg_scores, "neg_scores")
    return (1 + sum(1 for x in s if x >= score)) / (len(s) + 1)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def nudge_up(tau: float) -> float:
    """Smallest float strictly above tau, so ``score >= nudge_up(tau)`` == ``score > tau``."""
    return math.nextafter(tau, INF)
