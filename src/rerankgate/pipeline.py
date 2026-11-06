"""RAGPipeline: retrieve -> rerank -> gate -> rewrite/retry -> verdict."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal

from rerankgate.bm25 import BM25Index
from rerankgate.config import PipelineConfig
from rerankgate.documents import Chunk, Document, chunk_text
from rerankgate.embeddings import EmbeddingBackend, SentenceTransformerBackend
from rerankgate.gate import GateDecision, SufficiencyGate
from rerankgate.judge import FaithfulnessReport, HeuristicJudge, Judge
from rerankgate.metrics import EvalItem, aggregate
from rerankgate.rerank import FeatureReranker, Reranker
from rerankgate.retrieval import HybridRetriever, RetrievalResult, ScoredChunk
from rerankgate.rewrite import RM3Rewriter
from rerankgate.store import DocumentStore
from rerankgate.vectorstore import VectorStore

ANSWERABLE = "answerable"
INSUFFICIENT = "insufficient"
NEEDS_REVIEW = "needs_review"

Verdict = Literal[ANSWERABLE, INSUFFICIENT, NEEDS_REVIEW]  # type: ignore[valid-type]


@dataclass
class AttemptRecord:
    query: str
    retrieval_metrics: dict
    gate: dict
    evidence_ids: list[str]
    rewrite: str | None = None


@dataclass
class AnswerResult:
    verdict: Verdict
    query: str
    evidence: list[ScoredChunk]
    gate: GateDecision
    attempts: list[AttemptRecord] = field(default_factory=list)
    answer: str | None = None
    faithfulness: FaithfulnessReport | None = None
    trace: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "query": self.query,
            "evidence": [
                {
                    "chunk_id": c.chunk.id,
                    "doc_id": c.chunk.doc_id,
                    "score": round(c.score, 4),
                    "text": c.chunk.text,
                    "sources": c.sources,
                }
                for c in self.evidence
            ],
            "gate": self.gate.to_dict(),
            "attempts": [
                {
                    "query": a.query,
                    "retrieval_metrics": a.retrieval_metrics,
                    "gate": a.gate,
                    "evidence_ids": a.evidence_ids,
                    "rewrite": a.rewrite,
                }
                for a in self.attempts
            ],
            "answer": self.answer,
            "faithfulness": self.faithfulness.to_dict() if self.faithfulness else None,
            "trace": self.trace,
        }


class RAGPipeline:
    """Drop-in retrieval pipeline with a sufficiency gate.

    If the retrieval wasn't good enough, the answer never gets generated:
    `ask()` returns verdict="insufficient" and abstains instead of
    handing weak evidence to an LLM.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        embeddings: EmbeddingBackend | None = None,
        reranker: Reranker | None = None,
        judge: Judge | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.bm25 = BM25Index()
        self.chunks: list[Chunk] = []
        self.reranker: Reranker = reranker or FeatureReranker()
        self.judge: Judge = judge or HeuristicJudge()
        self.gate = SufficiencyGate(self.config.effective_gate())
        self.rewriter = RM3Rewriter(expansion_terms=self.config.retry.expansion_terms)
        self.faithfulness_floor = 0.35

        self.embeddings: EmbeddingBackend | None = None
        self.vector_store: VectorStore | None = None
        self.degraded_reason: str | None = None
        self.reembedded_on_load = False
        if self.config.use_vectors:
            if embeddings is not None:
                self._enable_vectors(embeddings)
            else:
                try:
                    self._enable_vectors(SentenceTransformerBackend())
                except RuntimeError as exc:
                    self.degraded_reason = str(exc)
                    warnings.warn(
                        "rerankgate: embeddings unavailable, running BM25-only. "
                        "Install with `pip install rerankgate[local]` for hybrid mode.",
                        stacklevel=2,
                    )

    def _enable_vectors(self, backend: EmbeddingBackend) -> None:
        self.embeddings = backend
        self.vector_store = VectorStore(dim=backend.dim)

    # -- ingestion -----------------------------------------------------
    def ingest(self, documents: list[Document]) -> int:
        """Chunk, index, and embed documents. Returns chunk count added."""
        seen = {c.doc_id for c in self.chunks}
        batch: set[str] = set()
        for doc in documents:
            if not isinstance(doc.id, str) or not doc.id:
                raise ValueError(f"document id must be a non-empty string, got {doc.id!r}")
            if "#c" in doc.id:
                raise ValueError(f"document id {doc.id!r} must not contain '#c' (chunk-id marker)")
            if doc.id in seen or doc.id in batch:
                # Duplicate ids would produce colliding chunk ids and silently
                # merge two documents' scores during fusion.
                raise ValueError(f"duplicate document id {doc.id!r}")
            batch.add(doc.id)
        new_chunks: list[Chunk] = []
        for doc in documents:
            new_chunks.extend(
                chunk_text(
                    doc.text,
                    doc_id=doc.id,
                    size=self.config.chunk_size,
                    overlap=self.config.chunk_overlap,
                    metadata=doc.metadata,
                )
            )
        self.chunks.extend(new_chunks)
        self.bm25.add(new_chunks)
        if self.embeddings is not None and self.vector_store is not None:
            vecs = self.embeddings.embed([c.text for c in new_chunks])
            self.vector_store.add(new_chunks, vecs)
        return len(new_chunks)

    def save(self, path: str) -> None:
        vectors: list[list[float]] | None = None
        if self.vector_store is not None:
            vectors = self.vector_store.vectors
        DocumentStore(path).save(
            self.chunks,
            vectors,
            meta={"embedding": self.embeddings.name if self.embeddings else None},
        )

    @classmethod
    def load(
        cls,
        path: str,
        config: PipelineConfig | None = None,
        embeddings: EmbeddingBackend | None = None,
    ) -> RAGPipeline:
        """Load a persisted store and rebuild indexes."""
        store = DocumentStore(path)
        chunks, vectors, meta = store.load()
        pipe = cls(config=config, embeddings=embeddings)
        pipe.chunks = chunks
        pipe.bm25.add(chunks)
        if pipe.embeddings is None or pipe.vector_store is None:
            # No embedding backend in this process (e.g. sentence-transformers
            # missing): stay BM25-only. Retrieval still works; quality degrades
            # gracefully and the trace says so.
            return pipe
        stored_model = meta.get("embedding")
        same_model = stored_model is None or stored_model == pipe.embeddings.name
        if (
            vectors
            and same_model
            and len(vectors) == len(chunks)
            and len(vectors[0]) == pipe.embeddings.dim
        ):
            pipe.vector_store.add(chunks, vectors)
        elif chunks:
            # Different model (even with the same dimension), missing, or
            # unreadable vectors: re-embed. Vectors from another model are
            # never mixed with this model's query embeddings.
            pipe.vector_store = VectorStore(dim=pipe.embeddings.dim)
            pipe.vector_store.add(chunks, pipe.embeddings.embed([c.text for c in chunks]))
            pipe.reembedded_on_load = True
        return pipe

    # -- querying ------------------------------------------------------
    def _retriever(self) -> HybridRetriever:
        retr = HybridRetriever(
            self.bm25,
            vector_store=self.vector_store,
            embeddings=self.embeddings,
            rrf_k=self.config.rrf_k,
            bm25_weight=self.config.bm25_weight,
            vector_weight=self.config.vector_weight,
        )
        retr.degraded = self.degraded_reason
        return retr

    def retrieve(self, query: str, k: int | None = None) -> RetrievalResult:
        return self._retriever().retrieve(query, k=k or self.config.retrieve_k)

    def ask(
        self,
        query: str,
        generator: Callable[[str, list[ScoredChunk]], str] | None = None,
        gate_override: bool = False,
    ) -> AnswerResult:
        """Run retrieve -> rerank -> gate, with rewrite retries, then verdict.

        If `generator` is given and the gate passes, the generated answer is
        faithfulness-checked; a failing check flips the verdict to needs_review.

        `gate_override=True` is the escape hatch for suspected gate false
        negatives: the best evidence is returned as answerable regardless of
        the gate, and the override is recorded in the trace.
        """
        t0 = perf_counter()
        cfg = self.config
        attempts: list[AttemptRecord] = []
        best: tuple[float, list[ScoredChunk], GateDecision, str] | None = None
        current_query = query

        for attempt in range(cfg.retry.max_retries + 1):
            retrieval = self._retriever().retrieve(current_query, k=cfg.retrieve_k)
            reranked = self.reranker.rerank(current_query, retrieval.chunks, k=cfg.rerank_k)
            decision = self.gate.evaluate(current_query, reranked)
            attempts.append(
                AttemptRecord(
                    query=current_query,
                    retrieval_metrics=retrieval.metrics,
                    gate=decision.to_dict(),
                    evidence_ids=[c.chunk.id for c in reranked],
                )
            )
            prev_best = best[0] if best is not None else -1.0
            if best is None or decision.score > best[0]:
                best = (decision.score, reranked, decision, current_query)
            if decision.passed:
                break
            # Diminishing-returns early stop: don't burn retries going nowhere.
            improved = decision.score > prev_best + cfg.retry.min_improvement
            if len(attempts) >= 2 and not improved:
                break
            if attempt < cfg.retry.max_retries:
                rewritten = self.rewriter.rewrite(current_query, reranked)
                if rewritten == current_query:
                    break  # rewriter found nothing new; further retries are pointless
                attempts[-1].rewrite = rewritten
                current_query = rewritten

        assert best is not None
        _, evidence, decision, _ = best
        verdict: Verdict = ANSWERABLE if decision.passed else INSUFFICIENT
        overridden = False
        if gate_override and verdict == INSUFFICIENT:
            verdict = ANSWERABLE
            overridden = True

        answer: str | None = None
        faithfulness: FaithfulnessReport | None = None
        if verdict == ANSWERABLE and generator is not None:
            answer = generator(query, evidence)
            faithfulness = self.judge.check(answer, evidence)
            if faithfulness.score < self.faithfulness_floor:
                verdict = NEEDS_REVIEW

        total_ms = round((perf_counter() - t0) * 1000, 2)
        trace = {
            "modes": self._retriever().modes,
            "degraded": self.degraded_reason,
            "attempts": len(attempts),
            "retries_used": len(attempts) - 1,
            "reranker": self.reranker.name,
            "judge": self.judge.name,
            "gate_overridden": overridden,
            "total_ms": total_ms,
            "chunk_count": len(self.chunks),
        }
        return AnswerResult(
            verdict=verdict,
            query=query,
            evidence=evidence,
            gate=decision,
            attempts=attempts,
            answer=answer,
            faithfulness=faithfulness,
            trace=trace,
        )

    def evaluate(self, items: list[EvalItem]) -> dict:
        """Run the eval harness: retrieval metrics + gate confusion matrix."""
        results = []
        for item in items:
            res = self.ask(item.query)
            results.append(
                {
                    "retrieved_ids": [c.chunk.id for c in res.evidence],
                    "verdict": res.verdict,
                }
            )
        return aggregate(items, results).to_dict()

    def calibrate(
        self,
        items: list[EvalItem],
        thresholds: list[float] | None = None,
        target_precision: float | None = None,
    ) -> dict:
        """Sweep gate thresholds on labeled data and recommend one.

        Each threshold is a full re-run of `ask()` (retries depend on the
        threshold, so scores can't just be re-cut). Recommendation:

        - with `target_precision`: the threshold with the highest gate recall
          whose gate precision is >= the target (ties -> stricter threshold);
          `None` if no threshold reaches the target.
        - otherwise: the threshold with the best F1 of gate precision/recall
          (ties -> stricter threshold).

        Requires at least one answerable and one unanswerable item: without
        both, precision/recall trade-offs are undefined and this fails closed.
        """
        import dataclasses

        if not items:
            raise ValueError("calibrate() needs at least one labeled item")
        if not any(it.answerable for it in items) or all(it.answerable for it in items):
            raise ValueError(
                "calibrate() needs both answerable and unanswerable items to measure "
                "the precision/recall trade-off"
            )
        if target_precision is not None and not 0.0 < target_precision <= 1.0:
            raise ValueError("target_precision must be in (0, 1]")
        if thresholds is None:
            thresholds = [round(0.30 + 0.05 * i, 2) for i in range(11)]  # 0.30 .. 0.80
        if not thresholds or any(not 0.0 < t <= 1.0 for t in thresholds):
            raise ValueError("thresholds must be a non-empty list of values in (0, 1]")

        original = self.gate
        base = self.config.effective_gate()
        rows: list[dict] = []
        try:
            for t in sorted(set(thresholds)):
                self.gate = SufficiencyGate(dataclasses.replace(base, threshold=t))
                rep = self.evaluate(items)
                p, r = rep["gate_precision"], rep["gate_recall"]
                f1 = 2 * p * r / (p + r) if (p + r) else 0.0
                rows.append(
                    {
                        "threshold": t,
                        "gate_precision": p,
                        "gate_recall": r,
                        "f1": round(f1, 4),
                        "abstention_rate": rep["abstention_rate"],
                    }
                )
        finally:
            self.gate = original

        if target_precision is not None:
            ok = [row for row in rows if row["gate_precision"] >= target_precision]
            best = max(ok, key=lambda row: (row["gate_recall"], row["threshold"])) if ok else None
            rule = f"max recall with precision >= {target_precision}"
        else:
            best = max(rows, key=lambda row: (row["f1"], row["threshold"]))
            rule = "max F1"
        return {
            "n": len(items),
            "rule": rule,
            "current_threshold": base.threshold,
            "recommended_threshold": best["threshold"] if best else None,
            "sweep": rows,
        }
