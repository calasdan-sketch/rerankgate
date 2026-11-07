"""Production drift monitor for the gate: abstain rate, score distribution, latency.

A threshold that was right at calibration time silently stops being right when the
corpus, the users or the embedding model change. This watches three things over a
rolling window and tells you *which* moved:

* ``ABSTAIN_RATE_SHIFT`` - two-proportion z-test of the window against the baseline
  (baseline sampling noise included, so a small baseline does not cause false alarms).
* ``SCORE_DRIFT`` - Population Stability Index of window scores over baseline deciles
  (rule of thumb used in credit-risk practice: < 0.1 stable, 0.1-0.25 moderate,
  > 0.25 major) plus the two-sample KS distance, reported alongside.
* ``LATENCY`` - p95 above a limit you set.
* ``INVALID_SCORES`` - NaN/inf/non-numeric scores were recorded. Always an alert; they
  also count as abstentions in the rate.

Standard library only; ``record`` is thread-safe. With fewer than ``min_n`` events in
the window the status is ``warming_up`` and no statistical alert fires (the stats would
be noise); invalid scores and latency still alert.
"""

from __future__ import annotations

import bisect
import json
import math
import threading
from collections import deque
from dataclasses import dataclass, field


def _finite(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return math.nan
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def psi(baseline_edges: list[float], baseline_frac: list[float], window: list[float]) -> float:
    """Population Stability Index of `window` against bins defined by `baseline_edges`.

    ``baseline_edges`` are the interior cut points (len = bins-1). Empty bins are floored
    at 1e-4 so the log is defined (standard practice).
    """
    bins = len(baseline_edges) + 1
    counts = [0] * bins
    for v in window:
        counts[bisect.bisect_right(baseline_edges, v)] += 1
    n = len(window)
    total = 0.0
    for b in range(bins):
        a = max(counts[b] / n, 1e-4)
        e = max(baseline_frac[b], 1e-4)
        total += (a - e) * math.log(a / e)
    return total


def ks_distance(a: list[float], b: list[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic."""
    if not a or not b:
        return 0.0
    sa, sb = sorted(a), sorted(b)
    i = j = 0
    d = 0.0
    while i < len(sa) and j < len(sb):
        x = min(sa[i], sb[j])
        while i < len(sa) and sa[i] <= x:
            i += 1
        while j < len(sb) and sb[j] <= x:
            j += 1
        d = max(d, abs(i / len(sa) - j / len(sb)))
    return d


@dataclass
class Baseline:
    """Reference behaviour captured at calibration time."""

    n: int
    abstain_rate: float
    edges: list[float]  # interior decile cut points of baseline scores
    scores: list[float] = field(default_factory=list)  # kept (capped) for the KS test

    @classmethod
    def from_events(cls, scores: list[float], answered: list[bool], bins: int = 10) -> Baseline:
        if len(scores) != len(answered) or len(scores) < 20:
            raise ValueError("a baseline needs >= 20 events with matching scores/answered")
        if not all(_finite(s) for s in scores):
            raise ValueError("baseline scores must all be finite")
        srt = sorted(float(s) for s in scores)
        edges = sorted({quantile(srt, k / bins) for k in range(1, bins)})
        return cls(
            n=len(scores),
            abstain_rate=sum(1 for a in answered if not a) / len(answered),
            edges=edges,
            scores=srt[:: max(1, len(srt) // 2000)],
        )

    def fractions(self) -> list[float]:
        counts = [0] * (len(self.edges) + 1)
        for v in self.scores:
            counts[bisect.bisect_right(self.edges, v)] += 1
        return [c / len(self.scores) for c in counts]

    def to_json(self) -> str:
        return json.dumps(
            {
                "n": self.n,
                "abstain_rate": self.abstain_rate,
                "edges": self.edges,
                "scores": self.scores,
            }
        )

    @classmethod
    def from_json(cls, text: str) -> Baseline:
        try:
            d = json.loads(text)
            b = cls(
                n=int(d["n"]),
                abstain_rate=float(d["abstain_rate"]),
                edges=[float(x) for x in d["edges"]],
                scores=[float(x) for x in d["scores"]],
            )
        except (KeyError, TypeError, ValueError, RecursionError) as exc:
            raise ValueError(f"malformed baseline: {exc}") from exc
        if not (0 <= b.abstain_rate <= 1) or b.n < 1 or not b.scores:
            raise ValueError("malformed baseline: out-of-range values")
        if not all(_finite(x) for x in b.edges + b.scores):
            raise ValueError("malformed baseline: non-finite numbers")
        return b


@dataclass
class MonitorReport:
    status: str  # "ok" | "warming_up" | "alert"
    n: int
    abstain_rate: float
    p50_score: float | None
    p95_latency_ms: float | None
    psi: float | None
    ks: float | None
    invalid: int
    alerts: list[str]

    def to_dict(self) -> dict:
        def r(x: float | None) -> float | None:
            return None if x is None or not math.isfinite(x) else round(x, 4)

        return {
            "status": self.status,
            "n": self.n,
            "abstain_rate": r(self.abstain_rate),
            "p50_score": r(self.p50_score),
            "p95_latency_ms": r(self.p95_latency_ms),
            "psi": r(self.psi),
            "ks": r(self.ks),
            "invalid": self.invalid,
            "alerts": self.alerts,
        }


class GateMonitor:
    def __init__(
        self,
        baseline: Baseline | None = None,
        window: int = 500,
        min_n: int = 100,
        z_alert: float = 3.0,
        psi_alert: float = 0.25,
        max_p95_latency_ms: float | None = None,
    ) -> None:
        if window < 10 or min_n < 10 or min_n > window:
            raise ValueError("need 10 <= min_n <= window")
        if not (_finite(z_alert) and z_alert > 0 and _finite(psi_alert) and psi_alert > 0):
            raise ValueError("z_alert and psi_alert must be positive finite numbers")
        self.baseline = baseline
        self.min_n = min_n
        self.z_alert, self.psi_alert = z_alert, psi_alert
        self.max_p95_latency_ms = max_p95_latency_ms
        self._lock = threading.Lock()
        self._events: deque[tuple[bool, float | None, float | None]] = deque(maxlen=window)

    def record(self, answered: bool, score: float, latency_ms: float | None = None) -> None:
        """Record one decision. Non-finite scores are stored as invalid and count as abstentions."""
        valid = _finite(score)
        lat = float(latency_ms) if _finite(latency_ms) else None
        with self._lock:
            self._events.append((bool(answered) and valid, float(score) if valid else None, lat))

    def check(self) -> MonitorReport:
        with self._lock:
            ev = list(self._events)
        n = len(ev)
        scores = [s for _, s, _ in ev if s is not None]
        invalid = sum(1 for _, s, _ in ev if s is None)
        lats = sorted(x for _, _, x in ev if x is not None)
        abstain = sum(1 for a, _, _ in ev if not a) / n if n else 0.0
        p95 = quantile(lats, 0.95) if lats else None
        alerts: list[str] = []
        if invalid:
            alerts.append(f"INVALID_SCORES: {invalid} of {n} events had a non-finite score")
        if (
            self.max_p95_latency_ms is not None
            and p95 is not None
            and p95 > self.max_p95_latency_ms
        ):
            alerts.append(f"LATENCY: p95 {p95:.1f}ms > limit {self.max_p95_latency_ms:g}ms")
        psi_v = ks_v = None
        warming = n < self.min_n
        if self.baseline is not None and not warming:
            b = self.baseline
            pooled = (abstain * n + b.abstain_rate * b.n) / (n + b.n)
            se = math.sqrt(pooled * (1 - pooled) * (1 / n + 1 / b.n))
            z = (
                (abstain - b.abstain_rate) / se
                if se > 0
                else (0.0 if abstain == b.abstain_rate else math.inf)
            )
            if abs(z) >= self.z_alert:
                alerts.append(
                    f"ABSTAIN_RATE_SHIFT: {abstain:.1%} vs baseline "
                    f"{b.abstain_rate:.1%} (z={z:+.1f})"
                )
            if len(scores) >= self.min_n:
                psi_v = psi(b.edges, b.fractions(), scores)
                ks_v = ks_distance(scores, b.scores)
                if psi_v >= self.psi_alert:
                    alerts.append(
                        f"SCORE_DRIFT: PSI {psi_v:.2f} >= {self.psi_alert:g} (KS {ks_v:.2f})"
                    )
        status = "alert" if alerts else ("warming_up" if warming else "ok")
        return MonitorReport(
            status=status,
            n=n,
            abstain_rate=abstain,
            p50_score=quantile(sorted(scores), 0.5) if scores else None,
            p95_latency_ms=p95,
            psi=psi_v,
            ks=ks_v,
            invalid=invalid,
            alerts=alerts,
        )
