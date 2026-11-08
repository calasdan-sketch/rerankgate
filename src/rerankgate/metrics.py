"""Evaluation aggregates: retrieval metrics + gate confusion matrix."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from rerankgate.calibration import wilson_interval


@dataclass
class EvalItem:
    query: str
    relevant_ids: list[str]  # chunk or doc ids considered relevant
    answerable: bool = True


def _match(ids: list[str], relevant: set[str]) -> bool:
    # A retrieved chunk matches if its id or its doc id is in the relevant set.
    for i in ids:
        if i in relevant or i.split("#c")[0] in relevant:
            return True
    return False


def _credited_ranks(ranked: list[str], relevant: set[str]) -> list[int]:
    """1-based ranks that credit a not-yet-credited relevant entry (each entry counts once).

    A document with ten relevant chunks retrieved is one relevant document, not ten, so
    nDCG and MAP can never exceed 1. A retrieved item credits at most one entry.
    """
    left = set(relevant)
    out: list[int] = []
    for rank, cid in enumerate(ranked, start=1):
        for key in (cid, cid.split("#c")[0]):
            if key in left:
                left.discard(key)
                out.append(rank)
                break
    return out


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG@k; 0.0 when nothing is relevant."""
    if not relevant or k <= 0:
        return 0.0
    dcg = sum(1.0 / math.log2(r + 1) for r in _credited_ranks(ranked[:k], relevant))
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-k slots holding a relevant chunk (k, not len(ranked), is the divisor)."""
    if k <= 0:
        return 0.0
    return sum(1 for i in ranked[:k] if i in relevant or i.split("#c")[0] in relevant) / k


def average_precision(ranked: list[str], relevant: set[str]) -> float:
    """Binary AP over the ranking; relevant entries never retrieved count as zero."""
    if not relevant:
        return 0.0
    credited = _credited_ranks(ranked, relevant)
    return sum((j + 1) / r for j, r in enumerate(credited)) / len(relevant)


@dataclass
class EvalReport:
    n: int = 0
    recall_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    map: float = 0.0
    false_answer_rate: float = 0.0  # of unanswerable items, fraction the gate let through
    false_answer_rate_ci95: tuple[float, float] = (0.0, 1.0)  # Wilson interval
    gate_recall_ci95: tuple[float, float] = (0.0, 1.0)
    gate_precision: float = 0.0  # of gate-PASS, fraction actually answerable
    gate_recall: float = 0.0  # of answerable, fraction gate passed
    abstention_rate: float = 0.0  # fraction verdict == insufficient
    answerable_rate: float = 0.0  # fraction verdict == answerable
    per_query: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "recall_at_k": {str(k): round(v, 4) for k, v in self.recall_at_k.items()},
            "mrr": round(self.mrr, 4),
            "ndcg_at_k": {str(k): round(v, 4) for k, v in self.ndcg_at_k.items()},
            "precision_at_k": {str(k): round(v, 4) for k, v in self.precision_at_k.items()},
            "map": round(self.map, 4),
            "false_answer_rate": round(self.false_answer_rate, 4),
            "false_answer_rate_ci95": [round(x, 4) for x in self.false_answer_rate_ci95],
            "gate_recall_ci95": [round(x, 4) for x in self.gate_recall_ci95],
            "gate_precision": round(self.gate_precision, 4),
            "gate_recall": round(self.gate_recall, 4),
            "abstention_rate": round(self.abstention_rate, 4),
            "answerable_rate": round(self.answerable_rate, 4),
            "per_query": self.per_query,
        }


def aggregate(
    items: list[EvalItem],
    results: list[dict],
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> EvalReport:
    """Aggregate per-query pipeline results into retrieval + gate metrics.

    Each result dict must have: `retrieved_ids` (ranked chunk ids),
    `verdict` ("answerable"|"insufficient"|"needs_review").
    """
    rep = EvalReport(n=len(items))
    if not items:
        return rep
    # Retrieval metrics are computed over answerable items only (those with
    # labeled relevant ids); the gate confusion matrix covers all items.
    retrievable = [(it, rs) for it, rs in zip(items, results, strict=True) if it.relevant_ids]
    hits = dict.fromkeys(ks, 0)
    rr_sum = 0.0
    tp = fp = fn = 0  # gate PASS vs labeled answerable
    abstentions = 0
    answerable_out = 0
    for item, res in zip(items, results, strict=True):
        verdict = res.get("verdict")
        passed = verdict == "answerable"
        if verdict == "insufficient":
            abstentions += 1
        if passed:
            answerable_out += 1
        if passed and item.answerable:
            tp += 1
        elif passed and not item.answerable:
            fp += 1
        elif not passed and item.answerable:
            fn += 1
        rep.per_query.append(
            {
                "query": item.query,
                "verdict": verdict,
                "expected_answerable": item.answerable,
                "retrieved_ids": res.get("retrieved_ids", [])[: max(ks)],
            }
        )
    ndcg_sum = dict.fromkeys(ks, 0.0)
    prec_sum = dict.fromkeys(ks, 0.0)
    ap_sum = 0.0
    for item, res in retrievable:
        rel = set(item.relevant_ids)
        ranked = res.get("retrieved_ids", [])
        ap_sum += average_precision(ranked, rel)
        for k in ks:
            ndcg_sum[k] += ndcg_at_k(ranked, rel, k)
            prec_sum[k] += precision_at_k(ranked, rel, k)
        for k in ks:
            if _match(ranked[:k], rel):
                hits[k] += 1
        for rank, cid in enumerate(ranked, start=1):
            if cid in rel or cid.split("#c")[0] in rel:
                rr_sum += 1.0 / rank
                break
    denom = len(retrievable) or 1
    rep.recall_at_k = {k: hits[k] / denom for k in ks}
    rep.mrr = rr_sum / denom
    rep.ndcg_at_k = {k: ndcg_sum[k] / denom for k in ks}
    rep.precision_at_k = {k: prec_sum[k] / denom for k in ks}
    rep.map = ap_sum / denom
    n_unans = sum(1 for it in items if not it.answerable)
    rep.false_answer_rate = fp / n_unans if n_unans else 0.0
    rep.false_answer_rate_ci95 = wilson_interval(fp, n_unans)
    rep.gate_recall_ci95 = wilson_interval(tp, tp + fn)
    rep.gate_precision = tp / (tp + fp) if (tp + fp) else 0.0
    rep.gate_recall = tp / (tp + fn) if (tp + fn) else 0.0
    rep.abstention_rate = abstentions / len(items)
    rep.answerable_rate = answerable_out / len(items)
    return rep
