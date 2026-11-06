# rerankgate

**Stop your RAG app from answering when retrieval found nothing useful — with
zero LLM calls and zero required dependencies.**

A drop-in retrieval pipeline with a decision point between retrieval and
generation:

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
pipe.ingest(
    [
        Document(id="rag", text="Retrieval-augmented generation pairs a retriever..."),
        Document(id="bm25", text="BM25 is a classic probabilistic ranking function..."),
    ]
)

result = pipe.ask("What is retrieval-augmented generation?")
print(result.verdict)  # "answerable" | "insufficient" | "needs_review"
print(result.gate.reasons)  # e.g. ["PASS"] or ["LOW_COVERAGE", ...]
print(result.gate.explanation)  # human-readable why

# Suspected false negative? Override it — the override is recorded in the trace.
result = pipe.ask("...", gate_override=True)

# Optional: run your own generator through the faithfulness check.
result = pipe.ask("...", generator=lambda q, ev: my_llm(q, [c.chunk.text for c in ev]))
# verdict flips to "needs_review" if the answer isn't grounded in the evidence.
```

Every `ask()` returns a full `trace`: per-stage timings, retrieval modes,
retry count, gate features, and whether the run was degraded.

## 60-second demo (no extra installs)

```bash
pip install rerankgate
cat > corpus.jsonl <<'EOF'
{"id": "rag", "text": "Retrieval-augmented generation pairs a retriever with a language model."}
{"id": "bread", "text": "Sourdough bread needs a starter, flour, water and patience."}
EOF
rerankgate ingest --input corpus.jsonl --store ./store --no-vectors
rerankgate query --store ./store --no-vectors "What is retrieval-augmented generation?"
rerankgate query --store ./store --no-vectors --gate-preset strict "xylophone concerto tickets"
```

The first query prints `verdict:  answerable`; the second prints
`verdict:  insufficient` with a reason code (here `NO_EVIDENCE`: no term matched) — that is the
gate refusing to hand off-topic context to a model. (Synthetic example corpus.)

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
rate.

### Calibrate the threshold on your data

```bash
rerankgate calibrate --store ./store --qrels eval.jsonl                     # best F1
rerankgate calibrate --store ./store --qrels eval.jsonl --target-precision 0.95
```

Sweeps thresholds 0.30–0.80, re-running the full pipeline at each (retries
depend on the threshold), prints precision / recall / F1 / abstention per
threshold and recommends one. Needs both answerable and unanswerable labeled
queries — it refuses to "calibrate" on one-sided data. From Python:
`pipe.calibrate(items, target_precision=0.95)`; then set
`PipelineConfig(gate=GateConfig(threshold=...))`.

### Use it as a CI check

```bash
rerankgate eval --store ./store --qrels eval.jsonl --summary \
    --fail-under-precision 0.9 --fail-under-recall 0.7 --fail-under-mrr 0.5
rerankgate query --store ./store --fail-on-insufficient "..."
```

Exit codes: `0` ok · `1` a quality check failed · `2` bad input or setup
(unreadable/empty corpus or qrels, malformed JSONL with the line number,
corrupt store). Empty inputs fail closed — an empty eval never "passes".

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

## Safety and robustness

- **Tamper-evident stores.** `meta.json` records SHA-256 checksums and counts;
  a modified, truncated or half-written store raises `StoreError` instead of
  serving wrong results. Files are written atomically (temp file + rename).
- **Vectors persist without numpy** (raw float32 via the stdlib), so custom
  embedding backends (e.g. a hosted API) work with zero extra installs.
  Stores written by 0.2 (`vectors.npy`) still load.
- **No model mixing.** A store embedded with one model is re-embedded when
  loaded with a different one — even if the vector size matches.
- **Terminal-safe output.** Corpus text is untrusted: the CLI escapes ANSI
  control sequences, bidi overrides and zero-width characters as `\uXXXX`.
- **Fail-closed defaults.** Duplicate document ids, NaN/inf embeddings, an
  LLM judge returning NaN or garbage, and nonsensical gate settings (threshold
  0, all weights 0) are rejected rather than silently passing.

## How it compares

Statements below are limited to what each project documents; follow the links.

| | Sufficiency gate before generation | LLM calls needed for the check | Notes |
|---|---|---|---|
| **rerankgate** | Yes — built in, with reason codes | **None** (LLM judge optional) | Zero required deps; BM25 + optional vectors, retries, eval + calibration CLI |
| LangGraph CRAG / Self-RAG recipes ([blog](https://www.langchain.com/blog/agentic-rag-with-langgraph)) | Yes — document grading node | Yes, an LLM grades each retrieved document | A pattern you assemble, not a package |
| Ragas faithfulness ([docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)) | No — offline evaluation metric | Yes (an HHEM classifier variant exists) | Evaluates answers after the fact |
| DeepEval contextual relevancy ([docs](https://deepeval.com/docs/metrics-contextual-relevancy)) | No — evaluation metric | LLM judge by default; a non-LLM "system_one" mode exists | Default threshold 0.5 |
| "Sufficient Context" (Google Research, [arXiv:2411.06037](https://arxiv.org/abs/2411.06037)) | Research method | Uses an LLM autorater | Reports selective generation raising the share of correct answers among those given by 2–10% |
| bm25s ([repo](https://github.com/xhluca/bm25s)) | No — lexical search only | — | Needs NumPy; author benchmarks show large speedups over rank-bm25 |
| FlashRank ([repo](https://github.com/PrithivirajDamodaran/FlashRank)) | No — reranker only | — | No torch; ~4MB default cross-encoder; pluggable here as a `Reranker` |

rerankgate is the gate *between* retrieval and generation; the evaluation
tools above are complementary (run them offline on what passes the gate).

## Performance

BM25 search uses an inverted index (0.3+): a query only touches chunks that
contain its terms. On a synthetic 20,000-chunk corpus, a 2-vCPU x86_64 cloud VM
measured 18.0 ms/query for the old full scan vs 0.57 ms/query for the inverted
index (~31x). Reproduce on your machine with
`python benchmarks/bench_bm25.py 20000`; numbers vary by hardware and corpus.
The numpy vector path now caches its normalized matrix instead of rebuilding
it per query.

## Limitations (honest)

- The heuristic gate is keyword-overlap based. Tokenization is Unicode-aware
  (accents, Cyrillic, Greek etc. work; Chinese/Japanese/Korean are split per
  character) but stopwords are English-only and there is no stemming, so
  synonym-heavy, cross-lingual or highly abstract queries can under-score — use
  hybrid vectors and/or the `lenient` preset, and run `rerankgate calibrate`.
  Thai and other unspaced scripts are not segmented.
- The default faithfulness judge measures how many of each answer sentence's
  content words appear in one evidence chunk: free and instant, but gameable by
  extractive copying and blind to negation. For adversarial settings, plug in
  `LLMJudge` (or an HHEM-style classifier).
- Default chunking is a sentence-aware character window, not semantic chunking;
  tables and structured text chunk naively.
- `pip install rerankgate[local]` pulls PyTorch (~2GB) via sentence-transformers —
  the price of local dense retrieval. BM25-only mode needs nothing.
- The gate threshold defaults are starting points, not tuned constants — they
  were sanity-checked on a toy corpus, not a benchmark. Calibrate on your data.
- Not thread-safe; one pipeline per thread. Everything is in memory: the store
  is loaded whole, and the vector search is brute force (no ANN index). At
  500K+ vectors, prefer the numpy path (installed with `[local]`) or an external
  vector DB.
- `calibrate` re-runs the pipeline once per threshold (11 runs by default) —
  cheap with BM25, slower with a hosted embedding API.

## License

MIT — Calas Automations Inc.
