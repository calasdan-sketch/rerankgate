# rerankgate

**Agentic RAG quality gate.** A drop-in retrieval pipeline with a decision point
between retrieval and generation:

```
query → hybrid retrieval (BM25 + vectors, RRF fusion)
      → rerank
      → sufficiency gate ── PASS ─→ evidence (+ optional generation & faithfulness check)
                           FAIL ─→ rewrite query (RM3, no LLM) → retry (≤2, early-stop)
                           exhausted → ABSTAIN with evidence + reasons
```

**The rule:** if the retrieval wasn't good enough, the answer never gets generated.
Local-first — the sufficiency gate works with zero LLM calls by default.

## Install

```bash
pip install rerankgate            # zero hard dependencies (BM25-only mode)
pip install "rerankgate[local]"   # + sentence-transformers, numpy (hybrid vectors)
```

## Quickstart

```python
from rerankgate import RAGPipeline, Document

pipe = RAGPipeline()  # degrades to BM25-only if embeddings aren't installed
pipe.ingest([
    Document(id="rag", text="Retrieval-augmented generation pairs a retriever..."),
    Document(id="bm25", text="BM25 is a classic probabilistic ranking function..."),
])

result = pipe.ask("What is retrieval-augmented generation?")
print(result.verdict)          # "answerable" | "insufficient" | "needs_review"
print(result.gate.reasons)     # e.g. ["PASS"] or ["LOW_COVERAGE", ...]
print(result.gate.explanation) # human-readable why

# Suspected false negative? Override it — the override is recorded in the trace.
result = pipe.ask("...", gate_override=True)

# Optional: run your own generator through the faithfulness check.
result = pipe.ask("...", generator=lambda q, ev: my_llm(q, [c.chunk.text for c in ev]))
# verdict flips to "needs_review" if the answer isn't grounded in the evidence.
```

Every `ask()` returns a full `trace`: per-stage timings, retrieval modes,
retry count, gate features, and whether the run was degraded.

## CLI

```bash
# JSONL corpus: {"id": ..., "text": ..., ...metadata}
rerankgate ingest --input corpus.jsonl --store ./store

rerankgate query --store ./store "What is RAG?"
rerankgate query --store ./store --gate-preset lenient --json "..."

# Evaluate retrieval + gate on labeled data, then tune the threshold.
# qrels JSONL: {"query": ..., "relevant_ids": [...], "answerable": true/false}
rerankgate eval --store ./store --qrels eval.jsonl
rerankgate eval --store ./store --qrels eval.jsonl --summary  # aggregates only
```

`rerankgate eval` reports recall@k, MRR (over answerable items), the gate
confusion matrix (precision/recall of PASS vs your labels), and the abstention
rate — use it to pick a threshold/preset instead of trusting the defaults.

## How the gate works

The sufficiency gate is a fixed weighted heuristic over interpretable features —
top rerank score, query-term coverage, ranking margin, and cross-chunk support —
plus hard floors on coverage and top score. Every decision carries stable
machine-readable reason codes (`PASS`, `LOW_TOP_SCORE`, `LOW_COVERAGE`,
`NO_MARGIN`, `NO_CONSENSUS`, `NO_EVIDENCE`) and a human explanation. Nothing is
learned, so there's nothing to overfit; tune it with the eval CLI.

Presets: `strict` (0.70) · `balanced` (0.55, default) · `lenient` (0.40).

## Design notes

- **Fusion:** Reciprocal Rank Fusion (k=60) — rank-based, no cross-system score
  normalization hacks. Per-system weights available in `PipelineConfig`.
- **Retries:** RM3-style pseudo-relevance feedback expands the query from the
  failed attempt's top docs — no LLM needed. Hard cap of 2 retries plus
  diminishing-returns early stop: the loop always terminates.
- **Abstention > hallucination:** on `insufficient` the pipeline returns
  best-effort evidence with gate reasons so the caller can decide — it never
  silently hands weak context to an LLM.
- **Pluggable:** `EmbeddingBackend`, `Reranker`, and `Judge` are protocols.
  Bring your own embeddings (OpenAI, Cohere…), a cross-encoder, or an
  LLM-as-judge without touching the pipeline.

## Limitations (honest)

- The heuristic gate is tuned for English keyword overlap. Synonym-heavy,
  cross-lingual, or highly abstract queries can under-score — use hybrid vectors
  and/or the `lenient` preset, and calibrate with `rerankgate eval`.
- The default faithfulness judge is token-overlap based: free and instant, but
  gameable by extractive copying. For adversarial settings, plug in `LLMJudge`.
- Default chunking is a sentence-aware character window, not semantic chunking;
  tables and structured text chunk naively.
- `pip install rerankgate[local]` pulls PyTorch (~2GB) via sentence-transformers —
  the price of local dense retrieval. BM25-only mode needs nothing.
- The gate threshold defaults are starting points, not tuned constants — they
  were sanity-checked on a toy corpus, not a benchmark. Calibrate on your data.
- Not thread-safe; one pipeline per thread. At 500K+ vectors, prefer the numpy
  path (installed with `[local]`) over the pure-Python fallback.

## License

MIT — Calas Automations Inc.
