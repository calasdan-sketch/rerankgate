"""Eval aggregation + faithfulness judge tests."""

import pytest

from rerankgate.judge import HeuristicJudge
from rerankgate.metrics import EvalItem, aggregate
from rerankgate.pipeline import ANSWERABLE, INSUFFICIENT


def _items():
    return [
        EvalItem(query="q1", relevant_ids=["doc-a#c0"], answerable=True),
        EvalItem(query="q2", relevant_ids=["doc-b#c0"], answerable=True),
        EvalItem(query="q3", relevant_ids=[], answerable=False),
    ]


def test_aggregate_perfect_run():
    results = [
        {"retrieved_ids": ["doc-a#c0", "doc-x#c0"], "verdict": ANSWERABLE},
        {"retrieved_ids": ["doc-b#c0"], "verdict": ANSWERABLE},
        {"retrieved_ids": [], "verdict": INSUFFICIENT},
    ]
    rep = aggregate(_items(), results)
    assert rep.recall_at_k[1] == 1.0
    assert rep.mrr == 1.0
    assert rep.gate_precision == 1.0
    assert rep.gate_recall == 1.0
    assert rep.abstention_rate == pytest.approx(1 / 3)


def test_aggregate_false_negative():
    # q1 answerable but gate abstained -> gate_recall < 1
    results = [
        {"retrieved_ids": ["doc-a#c0"], "verdict": INSUFFICIENT},
        {"retrieved_ids": ["doc-b#c0"], "verdict": ANSWERABLE},
        {"retrieved_ids": [], "verdict": INSUFFICIENT},
    ]
    rep = aggregate(_items(), results)
    assert rep.gate_recall == 0.5
    assert rep.gate_precision == 1.0


def test_aggregate_false_positive():
    # q3 unanswerable but gate passed -> gate_precision < 1
    results = [
        {"retrieved_ids": ["doc-a#c0"], "verdict": ANSWERABLE},
        {"retrieved_ids": ["doc-b#c0"], "verdict": ANSWERABLE},
        {"retrieved_ids": ["doc-z#c0"], "verdict": ANSWERABLE},
    ]
    rep = aggregate(_items(), results)
    assert rep.gate_precision == pytest.approx(2 / 3)
    assert rep.abstention_rate == 0.0


def test_aggregate_empty():
    rep = aggregate([], [])
    assert rep.n == 0


def test_aggregate_matches_doc_ids():
    items = [EvalItem(query="q", relevant_ids=["doc-a"], answerable=True)]
    results = [{"retrieved_ids": ["doc-a#c2"], "verdict": ANSWERABLE}]
    rep = aggregate(items, results)
    assert rep.recall_at_k[1] == 1.0


def test_heuristic_judge_supported(bm25_pipeline):
    retr = bm25_pipeline._retriever()
    res = retr.retrieve("rust borrow checker", k=3)
    answer = "Rust is a systems programming language focused on safety and speed."
    report = HeuristicJudge().check(answer, res.chunks)
    assert report.score > 0.35
    assert report.unsupported == []


def test_heuristic_judge_flags_unsupported(bm25_pipeline):
    retr = bm25_pipeline._retriever()
    res = retr.retrieve("rust borrow checker", k=3)
    report = HeuristicJudge().check("The moon is made of green cheese.", res.chunks)
    assert report.score < 0.35
    assert len(report.unsupported) == 1


def test_heuristic_judge_empty_answer(bm25_pipeline):
    retr = bm25_pipeline._retriever()
    res = retr.retrieve("rust", k=3)
    report = HeuristicJudge().check("", res.chunks)
    assert report.score == 0.0
