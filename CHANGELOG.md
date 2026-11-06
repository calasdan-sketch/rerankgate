# Changelog

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

### Changed
- BM25 search uses an inverted index (same scores, ~31x faster on a synthetic
  20k-chunk benchmark); deterministic tie order.
- Numpy vector search caches the normalized matrix.
- Loading a store with a different embedding model (by name) re-embeds even when
  dimensions match; a BM25-only store loaded with a backend gets embedded.
- CLI reports bad/empty input files with line numbers and exit code 2, no traceback.

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
