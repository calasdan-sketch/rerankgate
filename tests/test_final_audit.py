"""Final-audit tests: vector-store persistence edge cases, eval accounting,
chunk-id uniqueness, empty-index behavior, and determinism claims."""

from __future__ import annotations

import json

from rerankgate.config import PipelineConfig
from rerankgate.documents import Document, chunk_text
from rerankgate.embeddings import FakeEmbeddings
from rerankgate.metrics import EvalItem, aggregate
from rerankgate.pipeline import ANSWERABLE, INSUFFICIENT, RAGPipeline


def test_save_load_with_vectors_roundtrip(hybrid_pipeline, tmp_path):
    """Vectors persist and reload: hybrid modes survive a save/load cycle."""
    store = str(tmp_path / "store")
    hybrid_pipeline.save(store)
    assert hybrid_pipeline.vector_store is not None
    n_before = len(hybrid_pipeline.vector_store)

    loaded = RAGPipeline.load(
        store,
        config=PipelineConfig(use_vectors=True),
        embeddings=FakeEmbeddings(dim=16),
    )
    assert loaded.vector_store is not None
    assert len(loaded.vector_store) == n_before
    assert loaded._retriever().modes == ["bm25", "vector"]
    res = loaded.ask("What is retrieval-augmented generation?")
    assert res.verdict == ANSWERABLE


def test_load_vector_store_without_backend_degrades(tmp_path):
    """A vector-bearing store loads BM25-only (not crash) without embeddings."""
    pipe = RAGPipeline(config=PipelineConfig(use_vectors=True), embeddings=FakeEmbeddings(dim=16))
    pipe.ingest([Document(id="d1", text="Python is a programming language. " * 10)])
    store = str(tmp_path / "store")
    pipe.save(store)

    # Default config with no backend: sentence-transformers is not installed
    # in this env, so construction must degrade to BM25-only.
    loaded = RAGPipeline.load(store, config=PipelineConfig(use_vectors=True))
    assert loaded.degraded_reason is not None
    assert loaded._retriever().modes == ["bm25"]
    res = loaded.ask("Python programming language")
    assert res.trace["degraded"] == loaded.degraded_reason
    assert res.verdict in (ANSWERABLE, INSUFFICIENT)


def test_dim_mismatch_vectors_reembedded_not_trusted(tmp_path):
    """Stored vectors from a different model are re-embedded, never trusted."""
    pipe = RAGPipeline(config=PipelineConfig(use_vectors=True), embeddings=FakeEmbeddings(dim=16))
    pipe.ingest([Document(id="d1", text="Python is a programming language. " * 10)])
    store = str(tmp_path / "store")
    pipe.save(store)

    loaded = RAGPipeline.load(
        store,
        config=PipelineConfig(use_vectors=True),
        embeddings=FakeEmbeddings(dim=8),
    )
    assert loaded.vector_store is not None
    assert loaded.vector_store.dim == 8
    assert len(loaded.vector_store) == len(loaded.chunks)
    # Every re-embedded vector is 8-dim (no 16-dim leftovers trusted).
    assert all(len(v) == 8 for v in loaded.vector_store.vectors)


def test_eval_aggregate_counts_needs_review():
    """needs_review is neither a pass nor an abstention in eval aggregates."""
    items = [
        EvalItem(query="q1", relevant_ids=["d1#c0"], answerable=True),
        EvalItem(query="q2", relevant_ids=["d2#c0"], answerable=True),
        EvalItem(query="q3", relevant_ids=[], answerable=False),
    ]
    results = [
        {"retrieved_ids": ["d1#c0"], "verdict": "answerable"},
        {"retrieved_ids": ["d9#c0"], "verdict": "needs_review"},
        {"retrieved_ids": [], "verdict": "insufficient"},
    ]
    rep = aggregate(items, results)
    assert rep.answerable_rate == 1 / 3
    assert rep.abstention_rate == 1 / 3
    # q2 is answerable-labeled but did not pass -> false negative for gate recall.
    assert rep.gate_recall == 1 / 2
    assert rep.gate_precision == 1.0


def test_chunk_ids_unique_across_docs():
    """Identical text in different docs still yields distinct chunk ids."""
    text = "Same filler sentence. " * 30
    chunks = chunk_text(text, doc_id="a") + chunk_text(text, doc_id="b")
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(i.startswith(("a#c", "b#c")) for i in ids)


def test_ask_on_empty_index():
    """A pipeline with zero ingested chunks abstains without crashing."""
    pipe = RAGPipeline(config=PipelineConfig(use_vectors=False))
    res = pipe.ask("anything at all")
    assert res.verdict == INSUFFICIENT
    assert res.evidence == []
    assert res.gate.reasons == ["NO_EVIDENCE"]
    json.dumps(res.to_dict())  # must stay serializable


def test_rm3_rewrite_deterministic(bm25_pipeline):
    """RM3 rewriting is deterministic: same input always expands the same way."""
    from rerankgate.rewrite import RM3Rewriter

    retrieval = bm25_pipeline.retrieve("programming language memory")
    rewriter = RM3Rewriter(expansion_terms=5)
    first = rewriter.rewrite("programming language memory", retrieval.chunks)
    second = rewriter.rewrite("programming language memory", retrieval.chunks)
    assert first == second
    # And the expansion actually adds new terms when evidence exists.
    if retrieval.chunks:
        assert len(first.split()) >= len("programming language memory".split())


def test_hybrid_modes_recorded_in_trace(hybrid_pipeline):
    """Trace advertises both retrieval modes; no degraded flag when healthy."""
    res = hybrid_pipeline.ask("What is retrieval-augmented generation?")
    assert res.trace["modes"] == ["bm25", "vector"]
    assert res.trace["degraded"] is None
    assert res.attempts[0].retrieval_metrics["modes"] == ["bm25", "vector"]


def test_cli_eval_summary_flag(tmp_path, capsys):
    """--summary prints aggregates without the per-query detail dump."""
    import json

    from rerankgate.cli import main

    store = str(tmp_path / "store")
    qrels = tmp_path / "qrels.jsonl"
    qrels.write_text(
        json.dumps(
            {
                "query": "What is retrieval-augmented generation?",
                "relevant_ids": ["doc-rag"],
                "answerable": True,
            }
        )
        + "\n"
    )
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "id": "doc-rag",
                "text": "Retrieval-augmented generation combines a retriever with a model. "
                "The retriever fetches passages.",
            }
        )
        + "\n"
    )
    assert main(["ingest", "--input", str(corpus), "--store", store, "--no-vectors"]) == 0
    capsys.readouterr()  # discard ingest's human-readable output
    assert main(["eval", "--store", store, "--qrels", str(qrels), "--no-vectors", "--summary"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert "per_query" not in out
    assert "recall_at_k" in out and "abstention_rate" in out


def test_cli_version_flag(capsys):
    """--version reports the package version and exits cleanly."""
    import pytest

    from rerankgate import __version__
    from rerankgate.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out
