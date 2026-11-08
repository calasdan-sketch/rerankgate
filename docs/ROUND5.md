# Round 5: domain experts, properties, upgrade path

Branch `expert-2`. Rounds 1-4 (`README.md`, `CHANGELOG.md`, `docs/ROUND4.md`, `benchmarks/`) were read
first; only gaps were built.

## How this was run (honest)
No independent sub-agents were available, so one reviewer played each role in sequence, running
real commands. Different angles with a command behind every finding, not independent people.
Not tested: Windows, macOS, Python 3.10, real `sentence-transformers` hybrid mode, the GitHub
Actions workflow (never run), a read-only filesystem as non-root.

## Baseline (before any change)
- `pytest`: **321 passed** (104 s, 2-vCPU VM). `ruff`: clean. `mypy --ignore-missing-imports src`: clean.
- `benchmarks/bench_gate.py`: default gate 12.3% false answers / 99.2% recall; `conformal` 4.4% / 98.0%;
  with `delta` 2.5% / 93.4%; AURC 0.1714 (raw BM25 0.1733, oracle 0.1604).
- Registry: `pip download rerankgate==0.2.0 --no-deps` -> 0.2.0 is the only release.

## Author line
Already "Calas Automations Inc." in `authors` and `LICENSE` (round 4). Added `maintainers`
(name only, no email: none exists for this project), `rerankgate.__author__`, and a test pinning
all of them. There is no SARIF/JUnit output, no npm `package.json` and no generated-file header
in this project. Proof from a freshly built wheel: `unzip -p dist/*.whl '*/METADATA'` shows
`Author: Calas Automations Inc.` and `Maintainer: Calas Automations Inc.`; the wheel's
`rerankgate/__init__.py` has `__author__ = "Calas Automations Inc."`. Grep of the whole tree for
`Dan`, `DC Enterprises`, `calasdan` finds only real GitHub URLs. Whether the GitHub repository is
public is not something I can see; the README and metadata link to it as if it were. If it is
private those links 404 (see "Decide" below).

## Specialists and findings
| Role | Finding | Command that proved it |
|---|---|---|
| Information-retrieval researcher | BM25 matches a from-scratch naive Okapi implementation to 1e-9 on random corpora, with and without stemming, including after document removal (no defect). | `tests/test_round5.py::test_bm25_equals_naive_reference_before_and_after_removal` (30 seeds; first run as a 400-example hypothesis script) |
| Information-retrieval researcher | **Chunking cut words in half.** One long run without `.`/`!`/`?` (log line, CSV row, code) was cut every N characters: `configuration` became `co`+`uration`, so queries could never match it. | `chunk_text(" ".join(["configuration"]*60),"d",size=100,overlap=10)` -> words `co`, `uration`, `configur` |
| RAG engineer | **Chunking re-emitted earlier text.** "Short intro here." before a long sentence was stored 4 times (`d#c0, c2, c4, c6`), inflating the index and document frequencies. | same call with `"Short intro here. "+...` on the original code (git stash) |
| RAG engineer | Chunks could start/end with a tab or newline after an overlap. | invariants test on random text (19 of 60 seeds failed before the fix) |
| Statistician | `conformal_threshold`/`conformal_p_value` agree (pass implies p <= alpha), calibration budget holds on ties, and **simulated coverage is exactly as promised** (6000 draws, n=49, alpha=0.1: observed pass rate within 4 standard errors of 0.1). No defect. | `test_conformal_threshold_has_the_promised_coverage_by_simulation` |
| Statistician | `pac_threshold` equals a brute-force search over the exact binomial CDF on 60 random cases, and P(true FAR > alpha) <= delta by simulation (3000 draws). No defect. | `test_pac_threshold_matches_brute_force_and_holds_by_simulation` |
| Statistician | `wilson_interval` returned a lower bound ~3e-17 above the observed rate when k=0. Cosmetic, clamped. | property test over 300 random (k, n) |
| Statistician | **`ScoreGate.fit` failed on alternating labels.** `statistic="auto"` split by even/odd row; data whose label alternates with row order put every unanswerable query in the selection half -> "needs unanswerable queries". | `ScoreGate.fit(data, labels)` with `labels[i] = (i % 2 == 1)` |
| Statistician / security reviewer | **A hand-edited `ScoreGate` file could set `tau=-1e9` and pass everything**, voiding the guarantee while loading fine. `delta` was also unchecked. | `ScoreGate.from_json` of a file with edited `tau` (now rejected) |
| Search-relevance evaluator | `eval` had only recall@k and MRR: no nDCG, precision@k, MAP, and no uncertainty on the headline false-answer rate (0 false answers of 20 looks like "perfect"). Added, with a Wilson interval; a document counts once so nDCG/MAP cannot exceed 1. | `rerankgate eval --summary` before/after |
| Release engineer | Upgrade path: a store written by the **real 0.2.0 wheel** (no `format`, no checksum in `meta.json`) loads, queries identically (`score 0.657`), `info` works, and new documents can be ingested into it. No migration needed. Fixture committed. | `pip download rerankgate==0.2.0`; ingest with it; `rerankgate query/info` with this tree |
| Release engineer | sdist did not include JSON/JSONL under `tests/`, so the new golden and fixture files would be missing for anyone running the sdist's tests. `MANIFEST.in` fixed. | `tar tzf dist/*.tar.gz` |
| Mutation-testing expert | 16 hand-made mutants (flip `>=`/`>`, drop `-1`, drop a fail-closed check, remove a `break`, ...) run against the relevant tests: **15 killed**. First run left 3 alive and exposed missing tests for overlap size, double crediting in nDCG, and tombstone counts; tests added. One survivor remains: `sel += members[0::1]` (selection half overlapping the calibration half affects only which statistic is chosen, not a testable count). | `/tmp` scratch harness (not committed) |
| Competitor's maintainer (read-only) | LlamaIndex `SimilarityPostprocessor` (GitHub `main`, fetched today): `similarity_cutoff` default 0.0, plain `<` comparison, no calibration. FlashRank 0.2.10 and `rerankers` 0.10.0 (PyPI today) are rerankers only; `bm25s` 0.3.16 / `rank-bm25` 0.2.2 are lexical search only. None offers a calibrated, bounded abstain decision. NaN handling of LlamaIndex not re-verified this round. | `curl` of the raw GitHub file and the PyPI JSON API |
| Windows static review | Unchanged since round 4; new code uses `pathlib`/no shell. Still unrun on Windows. | code read |

## Scored ideas
Value, differentiation, effort, risk each 1-5.

| # | Idea | Value | Diff. | Effort | Risk | Status |
|---|---|---|---|---|---|---|
| 1 | Whitespace-aware hard split of long sentences | 5 | 2 | 2 | 2 | DONE |
| 2 | Stop re-emitting buffered text before long sentences | 5 | 1 | 1 | 1 | DONE |
| 3 | Strip stray tab/newline from chunk edges | 2 | 1 | 1 | 1 | DONE |
| 4 | Stratified deterministic split in `ScoreGate.fit` | 4 | 2 | 1 | 1 | DONE |
| 5 | Reject tampered/inconsistent `ScoreGate` files | 4 | 4 | 1 | 1 | DONE |
| 6 | nDCG@k, precision@k, MAP in `eval` | 4 | 2 | 2 | 1 | DONE |
| 7 | False-answer rate + Wilson interval in `eval` | 5 | 5 | 1 | 1 | DONE |
| 8 | Property/simulation tests of the guarantees | 5 | 4 | 2 | 1 | DONE |
| 9 | Differential BM25 test vs naive reference | 4 | 2 | 1 | 1 | DONE |
| 10 | Golden files for eval/query/gate-file output | 4 | 1 | 1 | 1 | DONE |
| 11 | Real 0.2.0 store as upgrade fixture | 4 | 1 | 1 | 1 | DONE |
| 12 | Pin author/maintainer/`__author__` | 3 | 1 | 1 | 1 | DONE |
| 13 | Clamp Wilson bounds | 1 | 1 | 1 | 1 | DONE |
| 14 | Ship test data files in the sdist | 2 | 1 | 1 | 1 | DONE |
| 15 | `hypothesis` as a dev dependency | 2 | 1 | 1 | 2 | NOT-DONE (seeded generators already cover it; no new dependency) |
| 16 | Persist the BM25 index in the store (ROUND4 #20) | 4 | 2 | 4 | 4 | NOT-DONE (store format change near the fail-closed checks) |
| 17 | Dense/hybrid + public BEIR benchmark (ROUND4 #21) | 5 | 5 | 5 | 2 | NOT-DONE (needs downloads; hosted embeddings cost money) |
| 18 | Windows/macOS/Python 3.10 run (ROUND4 #22) | 5 | 1 | 2 | 1 | NOT-DONE (no such machine; CI written) |
| 19 | Trusted-publishing release workflow (ROUND4 #23) | 3 | 1 | 2 | 3 | NOT-DONE (needs owner's PyPI setup; a publish path) |
| 20 | Cross-process locking for store and audit log (ROUND4 #24) | 3 | 2 | 4 | 3 | NOT-DONE (documented single writer) |
| 21 | ANN index for 500k+ vectors (ROUND4 #25) | 3 | 2 | 5 | 3 | NOT-DONE |
| 22 | Pin CI actions by commit SHA (ROUND4 #26) | 2 | 1 | 1 | 1 | NOT-DONE (cannot look up SHAs offline from here) |
| 23 | Paired bootstrap significance test for comparing two configs in `eval` | 4 | 4 | 3 | 2 | NOT-DONE |
| 24 | Graded relevance (nDCG with 0-3 labels) and qrels with TREC format import | 3 | 3 | 3 | 2 | NOT-DONE |
| 25 | Sentence-aware overlap using a real sentence segmenter for CJK/Thai | 3 | 2 | 4 | 3 | NOT-DONE |
| 26 | Mutation testing in CI (mutmut) | 2 | 1 | 3 | 2 | NOT-DONE (needs a dev dependency and long runtime) |

## What was built
Items 1-14; 187 + 3 new tests (`tests/test_round5.py`), 321 -> see "Pass B".

## Benchmark
Chunking fix moved the committed `benchmarks/results.json` (15 of 246 documents have a >500-char
sentence): default gate false answers 12.3% (unchanged) / recall 99.2% -> 98.7%; `conformal`
4.4% / 97.8%; with `delta` 2.5% / 93.3%; AURC 0.1716 (raw BM25 0.1738). Raw-cutoff row 8.6% -> 9.2%.
These are differences within the noise of one finite dataset; the guarantees (`conformal` mean
under 5%) still hold.

## Could not test
Windows, macOS, Python 3.10, hybrid mode with real embeddings, the CI workflow, a non-root
read-only filesystem, any upload (none attempted). Whether the GitHub repository is public.
