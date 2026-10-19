"""rerankgate — agentic RAG quality gate.

Drop-in retrieval pipeline with built-in quality gates:
hybrid search (BM25 + vector) -> rerank -> sufficiency gate ->
query-rewrite retry or abstain. Local-first; the sufficiency gate
works without any LLM call by default.
"""

from rerankgate.config import GateConfig, PipelineConfig, RetryConfig
from rerankgate.documents import Document, chunk_text
from rerankgate.gate import GateDecision, SufficiencyGate
from rerankgate.pipeline import AnswerResult, RAGPipeline, Verdict
from rerankgate.retrieval import HybridRetriever, ScoredChunk

__all__ = [
    "AnswerResult",
    "Document",
    "GateConfig",
    "GateDecision",
    "HybridRetriever",
    "PipelineConfig",
    "RAGPipeline",
    "RetryConfig",
    "ScoredChunk",
    "SufficiencyGate",
    "Verdict",
    "chunk_text",
]

__version__ = "0.2.0"
