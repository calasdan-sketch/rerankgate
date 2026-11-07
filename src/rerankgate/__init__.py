"""rerankgate — agentic RAG quality gate.

Drop-in retrieval pipeline with built-in quality gates:
hybrid search (BM25 + vector) -> rerank -> sufficiency gate ->
query-rewrite retry or abstain. Local-first; the sufficiency gate
works without any LLM call by default.
"""

from rerankgate.abstain import AbstentionNotice, explain_abstention
from rerankgate.config import GateConfig, PipelineConfig, RetryConfig
from rerankgate.documents import Document, chunk_text
from rerankgate.gate import GateDecision, SufficiencyGate
from rerankgate.monitor import Baseline, GateMonitor
from rerankgate.pipeline import AnswerResult, ConformalResult, RAGPipeline, Verdict
from rerankgate.retrieval import HybridRetriever, ScoredChunk
from rerankgate.scoregate import ScoreGate
from rerankgate.store import StoreError

__all__ = [
    "AbstentionNotice",
    "AnswerResult",
    "Baseline",
    "ConformalResult",
    "Document",
    "GateConfig",
    "GateDecision",
    "GateMonitor",
    "HybridRetriever",
    "PipelineConfig",
    "RAGPipeline",
    "RetryConfig",
    "ScoreGate",
    "ScoredChunk",
    "StoreError",
    "SufficiencyGate",
    "Verdict",
    "chunk_text",
    "explain_abstention",
]

__version__ = "0.4.0"
