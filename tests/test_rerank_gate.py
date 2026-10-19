"""Reranker + sufficiency gate tests."""

from rerankgate.config import GateConfig
from rerankgate.gate import (
    LOW_COVERAGE,
    LOW_TOP_SCORE,
    NO_EVIDENCE,
    PASS,
    SufficiencyGate,
)
from rerankgate.rerank import FeatureReranker


def test_feature_reranker_prefers_coverage(bm25_pipeline):
    retr = bm25_pipeline._retriever()
    res = retr.retrieve("rust borrow checker", k=10)
    reranked = FeatureReranker().rerank("rust borrow checker", res.chunks, k=5)
    assert reranked[0].chunk.doc_id == "doc-rust"
    scores = [c.score for c in reranked]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_feature_reranker_k_limit(bm25_pipeline):
    retr = bm25_pipeline._retriever()
    res = retr.retrieve("language", k=10)
    assert len(FeatureReranker().rerank("language", res.chunks, k=2)) <= 2


def test_gate_passes_good_retrieval(bm25_pipeline):
    retr = bm25_pipeline._retriever()
    res = retr.retrieve("what is retrieval-augmented generation", k=10)
    reranked = FeatureReranker().rerank("what is retrieval-augmented generation", res.chunks, k=5)
    decision = SufficiencyGate(GateConfig().preset("lenient")).evaluate(
        "what is retrieval-augmented generation", reranked
    )
    assert decision.passed
    assert PASS in decision.reasons


def test_gate_fails_no_evidence():
    decision = SufficiencyGate().evaluate("anything", [])
    assert not decision.passed
    assert NO_EVIDENCE in decision.reasons


def test_gate_fails_off_topic_query(bm25_pipeline):
    retr = bm25_pipeline._retriever()
    res = retr.retrieve("xylophone concerto schedule", k=10)
    reranked = FeatureReranker().rerank("xylophone concerto schedule", res.chunks, k=5)
    decision = SufficiencyGate(GateConfig().preset("strict")).evaluate(
        "xylophone concerto schedule", reranked
    )
    assert not decision.passed
    assert decision.reasons  # at least one machine-readable reason
    assert decision.explanation


def test_gate_hard_floors_block_pass(bm25_pipeline):
    retr = bm25_pipeline._retriever()
    res = retr.retrieve("python", k=10)
    reranked = FeatureReranker().rerank("python", res.chunks, k=5)
    cfg = GateConfig(min_coverage=0.99, min_top_score=0.99, threshold=0.0)
    decision = SufficiencyGate(cfg).evaluate("python", reranked)
    assert not decision.passed
    assert LOW_COVERAGE in decision.reasons or LOW_TOP_SCORE in decision.reasons


def test_gate_presets_ordering():
    strict = GateConfig().preset("strict").threshold
    balanced = GateConfig().preset("balanced").threshold
    lenient = GateConfig().preset("lenient").threshold
    assert strict > balanced > lenient


def test_gate_unknown_preset_raises():
    import pytest

    with pytest.raises(ValueError):
        GateConfig().preset("nope")


def test_gate_decision_serializable(bm25_pipeline):
    retr = bm25_pipeline._retriever()
    res = retr.retrieve("python programming", k=10)
    reranked = FeatureReranker().rerank("python programming", res.chunks, k=5)
    d = SufficiencyGate().evaluate("python programming", reranked).to_dict()
    assert set(d) >= {"passed", "score", "threshold", "reasons", "explanation", "features", "ms"}


def test_presets_scale_min_coverage():
    strict_cov = GateConfig().preset("strict").min_coverage
    balanced_cov = GateConfig().preset("balanced").min_coverage
    lenient_cov = GateConfig().preset("lenient").min_coverage
    assert strict_cov > balanced_cov > lenient_cov


def test_cross_encoder_empty_candidates_guard():
    # guard is structural: empty input must short-circuit before touching the model
    import inspect

    from rerankgate.rerank import CrossEncoderReranker

    src = inspect.getsource(CrossEncoderReranker.rerank)
    assert "if not candidates" in src
