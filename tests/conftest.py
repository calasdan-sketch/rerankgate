"""Shared fixtures: tiny corpus + pipelines (no heavy deps)."""

import pytest

from rerankgate.config import PipelineConfig
from rerankgate.documents import Document
from rerankgate.embeddings import FakeEmbeddings
from rerankgate.pipeline import RAGPipeline

DOCS = [
    Document(
        id="doc-python",
        text=(
            "Python is a high-level programming language. "
            "It supports multiple programming paradigms including object-oriented "
            "and functional programming. Python's standard library is extensive."
        ),
        metadata={"title": "Python"},
    ),
    Document(
        id="doc-rust",
        text=(
            "Rust is a systems programming language focused on safety and speed. "
            "Its borrow checker enforces memory safety without a garbage collector."
        ),
        metadata={"title": "Rust"},
    ),
    Document(
        id="doc-rag",
        text=(
            "Retrieval-augmented generation combines a retriever with a language model. "
            "The retriever fetches relevant passages. The generator conditions on them. "
            "Reranking the retrieved passages improves answer quality substantially."
        ),
        metadata={"title": "RAG"},
    ),
]


@pytest.fixture
def docs():
    return list(DOCS)


@pytest.fixture
def bm25_pipeline(docs):
    cfg = PipelineConfig(use_vectors=False)
    pipe = RAGPipeline(config=cfg)
    pipe.ingest(docs)
    return pipe


@pytest.fixture
def hybrid_pipeline(docs):
    cfg = PipelineConfig(use_vectors=True)
    pipe = RAGPipeline(config=cfg, embeddings=FakeEmbeddings(dim=16))
    pipe.ingest(docs)
    return pipe
