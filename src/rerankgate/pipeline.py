"""RAGPipeline: retrieve -> rerank -> gate -> rewrite/retry -> verdict."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import threading
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal

from rerankgate.audit import AuditLog
from rerankgate.bm25 import BM25Index
from rerankgate.config import GateConfig, PipelineConfig
from rerankgate.documents import Chunk, Document, chunk_text
from rerankgate.embeddings import EmbeddingBackend, SentenceTransformerBackend
from rerankgate.gate import GateDecision, SufficiencyGate, effective_score
from rerankgate.injection import flag_instructions
from rerankgate.judge import FaithfulnessReport, HeuristicJudge, Judge
from rerankgate.metrics import EvalItem, aggregate
from rerankgate.monitor import GateMonitor
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
class ConformalResult:
    """Outcome of ``RAGPipeline.conformal_gate``."""

    alpha: float
    delta: float | None
    n_unanswerable: int
    tau: float | None  # None = +inf (cannot guarantee: gate never passes)
    threshold: float  # value for GateConfig.threshold
    achievable: bool
    min_unanswerable_needed: int
    calibration_far: float
    far_ci95: tuple[float, float]
    answerable_recall: float | None  # optimistic: measured on the calibration items

    @property
    def guarantee(self) -> str:
        if not self.achievable:
            return (
                f"NONE: {self.n_unanswerable} unanswerable calibration queries is below the "
                f"{self.min_unanswerable_needed} needed for alpha={self.alpha}"
                + (f", delta={self.delta}" if self.delta is not None else "")
                + "; the gate was set to never pass"
            )
        if self.delta is None:
            return (
                f"expected false-answer rate <= {self.alpha} on unanswerable "
                "queries like the calibration set"
            )
        return (
            f"with probability >= {1 - self.delta:g} over the calibration draw, false-answer "
            f"rate <= {self.alpha} on unanswerable queries like the calibration set"
        )

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "delta": self.delta,
            "n_unanswerable": self.n_unanswerable,
            "tau": self.tau,
            "threshold": self.threshold,
            "achievable": self.achievable,
            "min_unanswerable_needed": self.min_unanswerable_needed,
            "calibration_far": round(self.calibration_far, 4),
            "far_ci95": [round(x, 4) for x in self.far_ci95],
            "answerable_recall_on_calibration": (
                None if self.answerable_recall is None else round(self.answerable_recall, 4)
            ),
            "guarantee": self.guarantee,
        }


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


def _validate_ids(documents: list[Document], existing: set[str]) -> None:
    batch: set[str] = set()
    for doc in documents:
        if not isinstance(doc.id, str) or not doc.id:
            raise ValueError(f"document id must be a non-empty string, got {doc.id!r}")
        if "#c" in doc.id:
            raise ValueError(f"document id {doc.id!r} must not contain '#c' (chunk-id marker)")
        if doc.id in existing or doc.id in batch:
            # Duplicate ids would produce colliding chunk ids and silently
            # merge two documents' scores during fusion.
            raise ValueError(f"duplicate document id {doc.id!r}")
        batch.add(doc.id)


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
        self._lock = threading.RLock()  # guards the indexes against concurrent ingest/remove
        self.bm25 = BM25Index(stem=self.config.stemming)
        self.chunks: list[Chunk] = []
        self.reranker: Reranker = reranker or FeatureReranker(stem=self.config.stemming)
        self.judge: Judge = judge or HeuristicJudge()
        self.gate = self._make_gate(self.config.effective_gate())
        self.rewriter = RM3Rewriter(expansion_terms=self.config.retry.expansion_terms)
        self.faithfulness_floor = 0.35
        self.monitor: GateMonitor | None = None  # optional drift monitor (see monitor.py)
        self.audit: AuditLog | None = None  # optional tamper-evident decision log (audit.py)

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

    def _make_gate(self, cfg: GateConfig) -> SufficiencyGate:
        return SufficiencyGate(cfg, stem=self.config.stemming)

    def _enable_vectors(self, backend: EmbeddingBackend) -> None:
        self.embeddings = backend
        self.vector_store = VectorStore(dim=backend.dim)

    # -- identity ------------------------------------------------------
    def fingerprint(self) -> str:
        """SHA-256 of everything that changes gate SCORES (not the threshold itself).

        A threshold calibrated under one fingerprint is only meaningful under the same one:
        change the weights, chunking, stemming, reranker or retrieval modes and the same
        number means something else. ``conformal --out`` stores it; ``--gate-file`` checks it.
        """
        g = dataclasses.asdict(self.config.effective_gate())
        g.pop("threshold")
        spec = {
            "v": 1,
            "gate": g,
            "retry": dataclasses.asdict(self.config.retry),
            "chunk": [self.config.chunk_size, self.config.chunk_overlap],
            "fusion": [self.config.rrf_k, self.config.bm25_weight, self.config.vector_weight],
            "retrieve_k": self.config.retrieve_k,
            "rerank_k": self.config.rerank_k,
            "stemming": self.config.stemming,
            "quarantine": self.config.quarantine_suspect_evidence,
            "reranker": self.reranker.name,
            "modes": self._retriever().modes,
            "embedding": self.embeddings.name if self.embeddings else None,
        }
        blob = json.dumps(spec, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def corpus_digest(self) -> str:
        """SHA-256 over chunk ids and texts (order-independent)."""
        h = hashlib.sha256()
        for line in sorted(f"{c.id}\x00{c.text}" for c in self.chunks):
            h.update(hashlib.sha256(line.encode("utf-8", "replace")).digest())
        return h.hexdigest()

    # -- ingestion -----------------------------------------------------
    def ingest(self, documents: list[Document]) -> int:
        """Chunk, index, and embed documents. Returns chunk count added.

        Safe to call while other threads call ``ask()``: the whole update happens under a lock
        and an embedding failure leaves the index untouched.
        """
        with self._lock:
            return self._ingest(documents)

    def remove(self, doc_ids: list[str]) -> int:
        """Delete documents (all their chunks) from the indexes. Returns chunks removed.

        Unknown ids are ignored. Removal is incremental (tombstones, compacted when a
        quarter of the rows are dead). A gate threshold calibrated before a large change to
        the corpus may no longer be valid - recalibrate.
        """
        ids = set(doc_ids)
        with self._lock:
            removed = len(self.chunks)
            self.chunks = [c for c in self.chunks if c.doc_id not in ids]
            removed -= len(self.chunks)
            if removed:
                self.bm25.remove_docs(ids)
                if self.vector_store is not None:
                    self.vector_store.remove_docs(ids)
            return removed

    def upsert(self, documents: list[Document]) -> int:
        """Insert documents, replacing any existing document with the same id.

        All-or-nothing: the new documents are chunked, embedded and validated BEFORE anything
        is removed, so a bad id or a failing embedding backend leaves the index as it was.
        Returns the number of chunks added.
        """
        with self._lock:
            _validate_ids(documents, set())
            prepared = self._prepare(documents)
            self.remove([d.id for d in documents])
            return self._commit(*prepared)

    def _ingest(self, documents: list[Document]) -> int:
        _validate_ids(documents, {c.doc_id for c in self.chunks})
        return self._commit(*self._prepare(documents))

    def _prepare(self, documents: list[Document]) -> tuple[list[Chunk], list[list[float]] | None]:
        """Everything that can fail (chunking, embedding, vector checks); touches no index."""
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
        vecs = None
        if self.embeddings is not None and self.vector_store is not None:
            vecs = self.vector_store.validate(self.embeddings.embed([c.text for c in new_chunks]))
            if len(vecs) != len(new_chunks):
                raise ValueError("embedding backend returned the wrong number of vectors")
        return new_chunks, vecs

    def _commit(self, new_chunks: list[Chunk], vecs: list[list[float]] | None) -> int:
        if vecs is not None and self.vector_store is not None:
            self.vector_store.add(new_chunks, vecs)
        self.chunks.extend(new_chunks)
        self.bm25.add(new_chunks)
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
        with self._lock:
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
        best: tuple[tuple[bool, float], list[ScoredChunk], GateDecision, str] | None = None
        eff_max = -1.0
        suspect: list[str] = []
        current_query = query

        for attempt in range(cfg.retry.max_retries + 1):
            with self._lock:
                retrieval = self._retriever().retrieve(current_query, k=cfg.retrieve_k)
            # Expansion only widens the candidate pool. Rerank and gate against the
            # ORIGINAL query: the expansion terms come from the evidence itself, so
            # judging sufficiency on the expanded query is circular (it made
            # off-topic queries pass after a retry).
            candidates = retrieval.chunks
            for sc in candidates:
                if sc.chunk.id not in suspect and flag_instructions(sc.chunk.text):
                    suspect.append(sc.chunk.id)
            if self.config.quarantine_suspect_evidence:
                candidates = [c for c in candidates if c.chunk.id not in suspect]
            reranked = self.reranker.rerank(query, candidates, k=cfg.rerank_k)
            decision = self.gate.evaluate(query, reranked)
            attempts.append(
                AttemptRecord(
                    query=current_query,
                    retrieval_metrics=retrieval.metrics,
                    gate=decision.to_dict(),
                    evidence_ids=[c.chunk.id for c in reranked],
                )
            )
            prev_best = best[0][1] if best is not None else -1.0
            eff_max = max(eff_max, effective_score(decision))
            key = (decision.passed, decision.score)
            if best is None or key > best[0]:
                # a passing attempt always beats a failing one, even a higher-scored
                # failure (one that tripped a hard floor)
                best = (key, reranked, decision, current_query)
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
        used = {c.chunk.id for c in evidence}
        trace = {
            "modes": self._retriever().modes,
            "degraded": self.degraded_reason,
            "attempts": len(attempts),
            "retries_used": len(attempts) - 1,
            "reranker": self.reranker.name,
            "judge": self.judge.name,
            "gate_overridden": overridden,
            "gate_score": eff_max,
            "total_ms": total_ms,
            "chunk_count": len(self.chunks),
            "stemming": self.config.stemming,
            "suspect_evidence": [i for i in suspect if i in used],
            "suspect_quarantined": self.config.quarantine_suspect_evidence,
        }
        result = AnswerResult(
            verdict=verdict,
            query=query,
            evidence=evidence,
            gate=decision,
            attempts=attempts,
            answer=answer,
            faithfulness=faithfulness,
            trace=trace,
        )
        if self.monitor is not None:
            self.monitor.record(answered=decision.passed, score=eff_max, latency_ms=total_ms)
        if self.audit is not None:
            from rerankgate import __version__

            self.audit.record(result, self.fingerprint(), __version__)
        return result

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
        - `None` as well when every threshold has zero recall / zero F1.
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
                self.gate = self._make_gate(dataclasses.replace(base, threshold=t))
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
        if best is not None and (best["f1"] <= 0 or best["gate_recall"] <= 0):
            # No threshold lets a single answerable query through: any pick
            # would just reject everything. Fail closed with no recommendation.
            best = None
            rule += " (no threshold passes any answerable query)"
        return {
            "n": len(items),
            "rule": rule,
            "current_threshold": base.threshold,
            "recommended_threshold": best["threshold"] if best else None,
            "sweep": rows,
        }

    # -- conformal calibration ----------------------------------------
    def score_results(self, queries: list[str]) -> list[AnswerResult]:
        """Run `ask()` with a gate that never passes early, so each result's
        ``trace["gate_score"]`` is the threshold-free best effective score over the retry
        attempts. One calibrated threshold then applies to the exact deployed rule."""
        original, monitor = self.gate, self.monitor
        base = self.config.effective_gate()
        huge = base.w_top_score + base.w_coverage + base.w_margin + base.w_support + 1.0
        self.gate = self._make_gate(dataclasses.replace(base, threshold=huge))
        self.monitor = None
        try:
            return [self.ask(q) for q in queries]
        finally:
            self.gate, self.monitor = original, monitor

    def gate_scores(self, queries: list[str]) -> list[float]:
        return [float(r.trace["gate_score"]) for r in self.score_results(queries)]

    def conformal_gate(
        self,
        items: list[EvalItem],
        alpha: float = 0.05,
        delta: float | None = None,
        apply: bool = True,
    ) -> ConformalResult:
        """Pick the gate threshold with a finite-sample false-answer-rate guarantee.

        Uses the *unanswerable* items as calibration. ``delta=None``: expected
        false-answer rate <= alpha (split conformal). ``delta=0.05``: with 95%
        probability over the calibration draw, false-answer rate <= alpha. If there
        are too few unanswerable items the result says ``achievable=False`` and
        (with ``apply``) the gate is set to never pass. Recall on answerable items
        is measured on the same items, so treat it as optimistic; hold out a test
        split to report it (benchmarks/bench_gate.py does).
        """
        from rerankgate import calibration as cal

        if not items:
            raise ValueError("conformal_gate() needs labeled items")
        neg = [it.query for it in items if not it.answerable]
        pos = [it.query for it in items if it.answerable]
        if not neg:
            raise ValueError("conformal_gate() needs unanswerable items to calibrate on")
        neg_s = self.gate_scores(neg)
        pos_s = self.gate_scores(pos) if pos else []
        tau = (
            cal.conformal_threshold(neg_s, alpha)
            if delta is None
            else cal.pac_threshold(neg_s, alpha, delta)
        )
        base = self.config.effective_gate()
        cap = base.w_top_score + base.w_coverage + base.w_margin + base.w_support + 1.0
        achievable = math.isfinite(tau) or tau == -math.inf
        if tau == math.inf:
            threshold = cap  # nothing can reach it: fail closed
        elif tau == -math.inf or tau < 0:
            threshold = 1e-9  # only the hard floors apply
        else:
            threshold = cal.nudge_up(tau)
        need = cal.min_negatives(alpha, delta)
        res = ConformalResult(
            alpha=alpha,
            delta=delta,
            n_unanswerable=len(neg_s),
            tau=tau if math.isfinite(tau) else None,
            threshold=threshold,
            achievable=achievable,
            min_unanswerable_needed=need,
            calibration_far=cal.far_at(neg_s, tau),
            answerable_recall=cal.recall_at(pos_s, tau) if pos_s else None,
            far_ci95=cal.wilson_interval(sum(1 for s in neg_s if s > tau), len(neg_s)),
        )
        if apply:
            self.gate = self._make_gate(dataclasses.replace(base, threshold=threshold))
        return res
