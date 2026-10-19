"""End-to-end pipeline tests: verdicts, retries, abstention, persistence."""

import json

import pytest

from rerankgate.config import PipelineConfig
from rerankgate.documents import Document
from rerankgate.pipeline import ANSWERABLE, INSUFFICIENT, NEEDS_REVIEW, RAGPipeline


def test_ask_answerable(bm25_pipeline):
    res = bm25_pipeline.ask("What is retrieval-augmented generation?")
    assert res.verdict == ANSWERABLE
    assert res.evidence
    assert res.gate.passed
    assert res.trace["attempts"] >= 1


def test_ask_abstains_on_garbage_query():
    cfg = PipelineConfig(use_vectors=False, gate_preset="strict")
    pipe = RAGPipeline(config=cfg)
    pipe.ingest([Document(id="d1", text="Python is a programming language. " * 10)])
    res = pipe.ask("xylophone concerto ticket prices")
    assert res.verdict == INSUFFICIENT
    assert not res.gate.passed
    assert res.gate.reasons
    # abstention still returns best-effort evidence + reasons, never hallucinates
    assert isinstance(res.evidence, list)


def test_retry_budget_is_honored():
    cfg = PipelineConfig(use_vectors=False)
    cfg.retry.max_retries = 1
    pipe = RAGPipeline(config=cfg)
    pipe.ingest([Document(id="d1", text="Completely unrelated content here. " * 20)])
    res = pipe.ask("quantum banana firmware")
    assert res.trace["attempts"] <= 2  # initial + at most 1 retry
    assert res.trace["retries_used"] <= 1


def test_retry_loop_terminates_without_improvement():
    cfg = PipelineConfig(use_vectors=False)
    cfg.retry.max_retries = 5
    pipe = RAGPipeline(config=cfg)
    pipe.ingest([Document(id="d1", text="Unrelated filler text about nothing. " * 20)])
    res = pipe.ask("obscure nonsense query terms")
    # early stop on no improvement must keep attempts well under the cap
    assert res.trace["attempts"] <= 6
    assert res.verdict == INSUFFICIENT


def test_rewriter_expands_query(bm25_pipeline):
    retr = bm25_pipeline._retriever()
    res = retr.retrieve("generation", k=10)
    rewritten = bm25_pipeline.rewriter.rewrite("generation", res.chunks)
    assert rewritten != "generation"
    assert "generation" in rewritten


def test_rewriter_noop_when_nothing_new(bm25_pipeline):
    assert bm25_pipeline.rewriter.rewrite("q", []) == "q"


def test_trace_has_stage_metrics(bm25_pipeline):
    res = bm25_pipeline.ask("Python programming paradigms")
    t = res.trace
    for key in (
        "modes",
        "attempts",
        "retries_used",
        "reranker",
        "judge",
        "total_ms",
        "chunk_count",
    ):
        assert key in t
    assert t["chunk_count"] == 3
    assert len(res.attempts) == t["attempts"]


def test_generator_and_faithfulness(bm25_pipeline):
    def gen(query, evidence):
        return " ".join(c.chunk.text for c in evidence[:1])

    res = bm25_pipeline.ask("What is retrieval-augmented generation?", generator=gen)
    assert res.answer is not None
    assert res.faithfulness is not None
    assert res.faithfulness.score >= 0.0


def test_unfaithful_answer_needs_review(bm25_pipeline):
    def bad_gen(query, evidence):
        return "The moon is made of green cheese and tastes like cheddar."

    res = bm25_pipeline.ask("What is retrieval-augmented generation?", generator=bad_gen)
    assert res.verdict == NEEDS_REVIEW
    assert res.faithfulness.score < bm25_pipeline.faithfulness_floor


def test_save_and_load_roundtrip(bm25_pipeline, tmp_path):
    store = str(tmp_path / "store")
    bm25_pipeline.save(store)
    loaded = RAGPipeline.load(store, config=PipelineConfig(use_vectors=False))
    res = loaded.ask("What is retrieval-augmented generation?")
    assert res.verdict == ANSWERABLE
    assert len(loaded.chunks) == 3


def test_load_missing_store_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        RAGPipeline.load(str(tmp_path / "nope"))


def test_result_serializable(bm25_pipeline):
    res = bm25_pipeline.ask("Rust memory safety")
    json.dumps(res.to_dict())  # must not raise


def test_ingest_counts_chunks(bm25_pipeline):
    n = bm25_pipeline.ingest([Document(id="extra", text="More text here. " * 30)])
    assert n >= 1
    assert len(bm25_pipeline.chunks) == 3 + n


def test_empty_query_abstains(bm25_pipeline):
    res = bm25_pipeline.ask("")
    assert res.verdict == INSUFFICIENT


def test_gate_override_escape_hatch():
    cfg = PipelineConfig(use_vectors=False, gate_preset="strict")
    pipe = RAGPipeline(config=cfg)
    pipe.ingest([Document(id="d1", text="Python is a programming language. " * 10)])
    res = pipe.ask("xylophone concerto ticket prices")
    assert res.verdict == INSUFFICIENT
    res2 = pipe.ask("xylophone concerto ticket prices", gate_override=True)
    assert res2.verdict == ANSWERABLE
    assert res2.trace["gate_overridden"] is True  # override is recorded, auditable


def test_override_recorded_when_not_needed(bm25_pipeline):
    res = bm25_pipeline.ask("What is retrieval-augmented generation?", gate_override=True)
    assert res.verdict == ANSWERABLE
    assert res.trace["gate_overridden"] is False
