"""Evaluation aggregates: retrieval metrics + gate confusion matrix."""

from __future__ import annotations

from dataclasses import dataclass, field


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


@dataclass
class EvalReport:
    n: int = 0
    recall_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
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
    for item, res in retrievable:
        rel = set(item.relevant_ids)
        ranked = res.get("retrieved_ids", [])
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
    rep.gate_precision = tp / (tp + fp) if (tp + fp) else 0.0
    rep.gate_recall = tp / (tp + fn) if (tp + fn) else 0.0
    rep.abstention_rate = abstentions / len(items)
    rep.answerable_rate = answerable_out / len(items)
    return rep
