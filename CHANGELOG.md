# Changelog

## 0.4.0 - unreleased (expert round 4, then rounds 3, 2 and the never-published 0.3.0 below)

PyPI has only 0.2.0; 0.3.0 was never uploaded, so the next release is 0.4.0 and contains every
section below. Not a breaking API change, but scores change (see "Changed" in the sections
below): re-calibrate any threshold you tuned on 0.2.

### Round 4: what a stranger who `pip install`s gets
Added
- `LICENSE` file (MIT, Calas Automations Inc.), shipped in the wheel and sdist. It was missing:
  an MIT licence needs its notice to travel with the code.
- `py.typed` (the package ships type hints; mypy now passes on `src`), `Typing :: Typed`,
  Python 3.10-3.13 classifiers, project URLs (repository, issues, changelog), SPDX
  `license = "MIT"` (build needs setuptools >= 77), `MANIFEST.in`.
- `rerankgate info --store DIR`, `ingest --json`, `ingest --input -` (stdin), and a `notice`
  block (what to tell the user) in `query --json` when the verdict is not `answerable`.
- `.github/workflows/ci.yml`: Linux/Windows/macOS x Python 3.10/3.13 tests, lint, mypy, and a
  build-install-run-quickstart job. Written but not yet run by anyone.
- `docs/ROUND4.md` (findings, 24-idea scored list, what was not tested), `PUBLISH-CHECKLIST.md`.

Fixed
- A corpus saved by Windows Notepad (UTF-8 BOM) was rejected as "invalid JSON"; gate files too.
- A non-UTF-8 console (Windows cp1252, C locale) crashed `query` with a traceback on any Greek,
  CJK or accented evidence. Output now degrades to `\uXXXX` escapes.
- A lone surrogate in a corpus (`"\ud800"`) crashed `ingest` with a traceback; now exit 2 with
  the line number (and `StoreError` from the library).
- `| head` printed `error: Broken pipe`; Ctrl-C printed a traceback. Now quiet (exit 141 / 130).
- An empty query was answered `insufficient` (exit 0); the CLI now rejects it (exit 2).
  `ask(None)` raises a clear `TypeError`.
- The library's "BM25-only" warning printed on top of the CLI's own note; the CLI hides the
  duplicate.
- `conformal --out` truncated an existing good gate file before writing, and wrote CRLF on
  Windows; it is now written atomically.
- Windows: `os.replace` of a store file that a scanner/reader has open is retried for about a
  second instead of failing (untested on Windows).
- First tokenization cost about 0.4-0.5 s on every CLI start (measured here, more on a busy machine): the combining-mark pattern was built by scanning all
  1.1M Unicode code points. It now scans only the blocks that contain marks (about
  6x faster: ~0.07 s), producing the identical pattern (tested against a full scan).
- 4 mypy errors in `monitor.py` / `scoregate.py` (no behaviour change).

### Round 3

### Added
- `PipelineConfig(stemming=True)` / `--stem`: opt-in light English stemming in BM25, reranker,
  gate and abstention notices (`tokenize.light_stem`). Off by default.
- `RAGPipeline.remove()` / `upsert()`: incremental, all-or-nothing document updates (BM25
  tombstones with compaction; results are bit-identical to a fresh build, checked by a churn
  test). `ingest`/`upsert` embed and validate before touching any index.
- Thread safety: one lock covers retrieval and index updates.
- `AuditLog` / `verify_audit` / `rerankgate audit-verify` / `query --audit-log`: hash-chained
  JSONL decision log (query stored as SHA-256 unless asked), anchorable head.
- `RAGPipeline.fingerprint()` / `corpus_digest()`; `conformal --out` stores them and
  `--gate-file` refuses a gate calibrated under a different scoring setup (exit 2).
- `rerankgate.injection.flag_instructions`, `trace["suspect_evidence"]`,
  `PipelineConfig(quarantine_suspect_evidence=...)`: heuristic tripwire for instruction-like
  retrieved text (documented as bypassable).
- `python -m rerankgate`.
- Benchmark: stemming variant and a singular/plural stress test in `results.json`.

### Fixed
- HIGH: `chunk_overlap >= chunk_size` made `chunk_text` loop forever on a long sentence; sizes
  and overlaps are now validated (`ValueError`).
- `chunk_overlap=0` carried the whole previous chunk forward, so chunks grew without bound.
- Chunk overlap started in the middle of a word (junk tokens at every chunk boundary).
- Gate margin and "consensus" counted duplicate passages as separate evidence: copies zeroed
  the margin (false abstention, misleading `NO_MARGIN`) and inflated corroboration. They now
  count once. Scores for corpora with duplicate text change slightly.
- Deeply nested JSON (corpus, store, gate file, baseline, ScoreGate file) raised
  `RecursionError` / a traceback instead of a clean error.
- A failing embedding backend during `ingest` could leave BM25 and the vector store out of step.

### Changed
- A gate file written before this version has no fingerprint: it still works, with a warning.
- Calibrated thresholds from earlier 0.4 builds can shift by the duplicate/chunking fixes above;
  re-run `conformal` (the committed benchmark moved the default-gate false-answer rate from
  13.2% to 12.3%).

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
