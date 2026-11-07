# Changelog

## 0.4.0 - unreleased (expert round 2)

### Added
- `rerankgate conformal` / `RAGPipeline.conformal_gate()`: threshold with a finite-sample
  false-answer-rate guarantee (split conformal; or high-probability with `--delta`), fail
  closed (gate never passes) when there are too few unanswerable calibration queries.
  `--gate-file` on every command applies the saved threshold.
- `rerankgate.calibration` (exact binomial/order-statistic maths, p-values, Wilson).
- `ScoreGate`: the same calibrated decision for raw scores from any retriever/reranker,
  with conformal p-values and JSON save/load.
- `rerankgate.selective` + `rerankgate riskcov`: risk-coverage curve, AURC, abstention
  precision/recall, false-answer rate.
- `GateMonitor` / `Baseline`: thread-safe drift monitor (abstain-rate z-test, PSI, KS,
  latency p95, invalid scores); `pipe.monitor` records every `ask()`.
- `explain_abstention()`: what to show the user (kind, next action, missing terms,
  nearest sources labelled "does not answer", clarifying question), no LLM.
- `benchmarks/bench_gate.py`, `build_dataset.py`, committed data and `results.json`.
- `gate.effective_score()`, `RAGPipeline.score_results()`, `trace["gate_score"]`.

### Fixed
- HIGH: query-rewrite retries were circular. The reranker and gate were run on the
  *expanded* query, whose extra terms were taken from the evidence itself, so unanswerable
  queries passed after a retry (27 -> 119 of 223 on the benchmark). Both now use the
  original query; expansion only widens the candidate pool.
- A passing retry attempt could lose to an earlier, higher-scored attempt that had failed
  a hard floor; passing attempts now win.
- The gate compared the unrounded score but reported (and now calibrates on) the score
  rounded to 4 places; the comparison uses the rounded value.

### Changed
- Retry behaviour change as above: `retries_used` and verdicts on some borderline
  queries differ from 0.3 (fewer false passes).

## 0.3.0 — unreleased

### Added
- `rerankgate calibrate` / `RAGPipeline.calibrate()`: sweep gate thresholds on
  labeled data and recommend one (best F1, or max recall at a target precision).
- CI gating: `eval --fail-under-precision/--fail-under-recall/--fail-under-mrr`
  and `query --fail-on-insufficient`; documented exit codes 0/1/2.
- Zero-dependency vector persistence (`vectors.f32`, stdlib `array`), atomic
  writes, SHA-256 checksums and count checks; corrupt stores raise `StoreError`.
- Unicode-aware tokenization (NFKC + casefold; CJK split per character).
- `rerankgate.safeterm.clean()`; CLI output escapes control/bidi/zero-width chars.
- `benchmarks/bench_bm25.py`.

- Gate reason code `INVALID_SCORE` (non-finite or non-numeric reranker scores).

### Changed
- BM25 search uses an inverted index. Ids, ranking order and float scores are
  bit-identical to 0.2 (checked by a differential test). Speedup depends on the
  query mix: ~3x (very common terms) to ~39x (rare terms) on a synthetic
  20k-chunk benchmark; ~5x for a mix closest to natural questions.
- Numpy vector search caches the normalized matrix.
- Loading a store with a different embedding model (by name) re-embeds even when
  dimensions match; a BM25-only store loaded with a backend gets embedded.
- CLI reports bad/empty input files with line numbers and exit code 2, no traceback.

### Fixed (independent panel review)
- HIGH: a NaN/inf score from a custom reranker made the gate PASS
  (`min(1.0, nan) == 1.0`). It now fails closed with `INVALID_SCORE`.
- Store checks were skippable: a missing `meta.json` or a deleted checksum key
  disabled them. `meta.json` is now required, format-2 stores must carry
  counts and checksums, meta and chunk field types are validated, and NaN in a
  vector file raises `StoreError`. Docs now say "detects corruption" (the
  checksums are unkeyed, so they are not tamper-proof).
- `calibrate` recommended a threshold (exit 0) even when every threshold had
  zero recall; it now returns no recommendation and the CLI exits 1.
- Tokenization split words at combining marks (Hindi, Arabic, Hebrew); tokens
  now include Unicode marks, and CJK Extensions B+ are split per character.
- `LLMJudge` accepted `True` as 1.0, clamped out-of-range values (e.g. 7) to
  1.0, and crashed on huge ints; all of these now score 0.
- Inverted-index BM25 summed repeated query terms as `qtf * x`, which could
  flip the order of exactly tied documents; each occurrence is now added
  separately, matching 0.2 exactly.
- The speedup claim now reports the realistic range per query mix.

### Fixed
- A custom `GateConfig(threshold=...)` was silently overwritten by the default
  "balanced" preset. `PipelineConfig.gate_preset` now defaults to `None`
  (the `GateConfig` defaults are identical to "balanced").
- `HeuristicJudge` scored correct sentences quoted from long chunks as
  unsupported (F1 against the whole chunk). It now uses token precision.
- `LLMJudge` treated a NaN score as 1.0 (faithful). NaN/inf/unparseable now score 0.
- Hybrid stores could not be saved without numpy.
- `FaithfulnessReport.per_sentence` defaulted to a dict instead of a list.

### Breaking (deliberate)
- `ingest()` raises `ValueError` on duplicate document ids (previously produced
  colliding chunk ids and merged scores), empty ids, or ids containing `#c`.
- `GateConfig` values are validated (finite, non-negative, threshold > 0,
  weights not all zero).
- `HeuristicJudge` grounding numbers differ from 0.2 (precision, not F1).
- Token boundaries changed for non-ASCII text (e.g. "café" was "caf").
