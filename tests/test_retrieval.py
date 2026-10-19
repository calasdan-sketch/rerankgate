def test_hybrid_bm25_only_mode(bm25_pipeline):
    retr = bm25_pipeline._retriever()
    assert retr.modes == ["bm25"]
    res = retr.retrieve("what is retrieval-augmented generation", k=5)
    assert res.chunks
    assert res.chunks[0].chunk.doc_id == "doc-rag"
    assert res.metrics["modes"] == ["bm25"]
    assert "sources" in res.chunks[0].__dict__


def test_hybrid_vector_mode(hybrid_pipeline):
    retr = hybrid_pipeline._retriever()
    assert retr.modes == ["bm25", "vector"]
    res = retr.retrieve("programming language paradigms", k=5)
    assert res.chunks
    assert res.metrics["vector_ms"] >= 0


def test_rrf_fusion_combines_rankings(hybrid_pipeline):
    retr = hybrid_pipeline._retriever()
    res = retr.retrieve("borrow checker memory safety", k=5)
    ids = [c.chunk.id for c in res.chunks]
    assert "doc-rust#c0" in ids
    # every fused chunk records per-system raw scores
    for c in res.chunks:
        assert "bm25" in c.sources or "vector" in c.sources


def test_retrieve_empty_query_returns_empty(bm25_pipeline):
    res = bm25_pipeline.retrieve("", k=5)
    assert res.chunks == []


def test_retrieval_metrics_present(bm25_pipeline):
    res = bm25_pipeline.retrieve("python", k=5)
    m = res.metrics
    for key in ("modes", "k", "bm25_hits", "bm25_ms", "total_ms", "fused_count"):
        assert key in m


def test_degraded_flag_propagates():
    from rerankgate.config import PipelineConfig
    from rerankgate.pipeline import RAGPipeline

    pipe = RAGPipeline(config=PipelineConfig(use_vectors=True), embeddings=None)
    # sentence-transformers not installed in test env -> BM25-only degradation
    assert pipe.degraded_reason is not None or pipe.embeddings is not None
    res = pipe.retrieve("python")
    assert res.metrics["modes"] == ["bm25"] or "vector" in res.metrics["modes"]
