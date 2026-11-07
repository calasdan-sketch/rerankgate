# Round 4: the stranger's first hour

Branch `expert-2`. Focus: what someone gets from `pip install rerankgate`, not new algorithms.
Rounds 1-3 (gate, conformal, ScoreGate, monitor, audit log, stemming, upsert/remove) were read first
(`README.md`, `CHANGELOG.md`, `benchmarks/`); nothing from them was redone.

## How this was run (honest)
The environment had **no way to spawn independent sub-agents**, so one reviewer played every role
below in sequence. They are different angles with a command behind every finding, not independent
people. Windows, macOS, Python 3.10 and a non-root read-only filesystem were **not** tested.

## Baseline (before any change)
- `pytest`: **303 passed** (127 s on a loaded 2-vCPU VM); `ruff check`: clean; mypy: 4 errors.
- `python3 benchmarks/bench_gate.py`: reproduces the committed `results.json` byte for byte
  (about 18 s CPU; wall time 56-69 s on this noisy VM).
- Registry: `pip index versions rerankgate` -> **0.2.0 only** (PyPI JSON agrees: author
  "Calas Automations Inc.", MIT, only a Homepage URL). The repo is 0.4.0.

## Specialists and findings
| Role | Finding | Command that proved it |
|---|---|---|
| Release engineer | **No LICENSE file** in repo, wheel or sdist (MIT needs its notice to travel). | `ls LICENSE*`; `unzip -l dist0/*.whl` |
| Release engineer | No `py.typed`; only a Homepage URL; no Python-version classifiers; legacy `License ::` classifier. | `unzip -p dist0/*.whl '*/METADATA'` |
| Release engineer | sdist shipped tests but not CHANGELOG/LICENSE/benchmarks; README links to `benchmarks/` would 404 on PyPI. | `tar tzf dist0/*.tar.gz` |
| Release engineer | Three CHANGELOG headings all say "0.4.0 - unreleased" plus a never-uploaded 0.3.0; PyPI has 0.2.0. | `grep -n '^## ' CHANGELOG.md` |
| First-time user (clean venv, path with spaces) | README quick start works from the built wheel on 3.11/3.12/3.13. But plain `pip install rerankgate` would fetch **0.2.0**, which lacks these commands. | `qs_verify.sh` per version; `pip index versions` |
| First-time user | A Windows-Notepad corpus (UTF-8 BOM, CRLF) was rejected: "invalid JSON (Unexpected UTF-8 BOM)". | `printf '\xef\xbb\xbf{...}\r\n' > c.jsonl; rerankgate ingest --input c.jsonl ...` |
| Windows/locale reviewer | Any Greek/CJK/accented evidence on a cp1252 console -> `UnicodeEncodeError` traceback. | `PYTHONIOENCODING=cp1252 rerankgate query ...` |
| Hostile-input fuzzer | Lone surrogate (`"\ud800"`) in a corpus -> traceback from `ingest`. | `rerankgate ingest --input sur.jsonl` |
| Hostile-input fuzzer | `ask(None)` -> cryptic `normalize() argument 2 must be str`. Empty query accepted (exit 0). | `python fz.py` (scratch fuzz script) |
| CI/CD integrator | `\| head` prints `error: Broken pipe`; Ctrl-C traceback; `--out` truncated an existing gate file before computing the new one; no stdin corpus; ingest has no `--json`; `query --json` has no abstention notice. | `rerankgate query ... \| head`; code read of `cmd_conformal` |
| Performance engineer | Every CLI start paid ~0.4-0.5 s scanning all 1.1M Unicode code points to build the token regex. 50k-doc store load ~4 s (BM25 rebuild dominates: `bm25.add` 9 s under cProfile). | `cProfile` of `RAGPipeline.load` |
| Supply-chain reviewer | Zero runtime dependencies (nothing to audit). Wheel has no tests, secrets or personal paths (grep for home paths and key patterns on tracked files: none). The `[local]` extra is unpinned above (`sentence-transformers>=2.2`); CI actions are pinned by tag, not commit SHA. | `pip list` in clean venv; grep |
| Competitor's maintainer | LlamaIndex `SimilarityPostprocessor` still keeps NaN scores (`score < cutoff` is False for NaN), default cutoff 0.0; FlashRank still described as a reranker only. Neither offers a calibrated, bounded gate. | fetched `postprocessor/node.py` (GitHub `main`) and the FlashRank PyPI summary; LangGraph/Ragas/DeepEval rows **not** re-fetched (unverified) |
| Type-check reviewer | `py.typed` would have been a lie: mypy had 4 errors. | `mypy --ignore-missing-imports src` |
| Windows static review | No `shell=True`, no `signal`, no POSIX-only calls (`O_NOFOLLOW`/`O_BINARY` guarded with `getattr`); all `open()` use explicit encodings. Risks: `os.replace` onto a file another process holds open raises `PermissionError` on Windows; text-mode `--out` writes CRLF. | `grep -n 'open(\|os.replace\|shell=\|signal' src/rerankgate/*.py` |

## Scored ideas
Value, differentiation, effort, risk each 1-5 (high value/differentiation good; high effort/risk bad).

| # | Idea | Value | Diff. | Effort | Risk | Status |
|---|---|---|---|---|---|---|
| 1 | Ship a `LICENSE` file in repo and artifacts | 5 | 1 | 1 | 1 | DONE |
| 2 | `py.typed` + make mypy clean | 3 | 2 | 2 | 1 | DONE |
| 3 | Full metadata: URLs, classifiers, SPDX licence, setuptools>=77 | 3 | 1 | 1 | 2 | DONE |
| 4 | Accept UTF-8 BOM / CRLF in corpus, qrels, gate file | 4 | 1 | 1 | 1 | DONE |
| 5 | Never crash on unencodable console output (`backslashreplace`) | 4 | 2 | 1 | 1 | DONE |
| 6 | Lone surrogate -> clean error (CLI) and `StoreError` (library) | 3 | 1 | 1 | 1 | DONE |
| 7 | Quiet exit on broken pipe (141) and Ctrl-C (130) | 3 | 1 | 1 | 1 | DONE |
| 8 | Reject empty query (exit 2); clear `TypeError` for non-str | 3 | 1 | 1 | 1 | DONE |
| 9 | `ingest --input -` (stdin) | 3 | 2 | 1 | 1 | DONE |
| 10 | `rerankgate info` (size, mode, model, digests) | 3 | 2 | 1 | 1 | DONE |
| 11 | `ingest --json` | 3 | 1 | 1 | 1 | DONE |
| 12 | `notice` block in `query --json` (what to show the user) | 4 | 4 | 1 | 1 | DONE |
| 13 | CI workflow: Linux/Windows/macOS x 3.10/3.13, lint, mypy, build+install+quickstart | 5 | 2 | 2 | 1 | DONE (never run) |
| 14 | Cut ~0.4 s startup (token regex scan) | 3 | 1 | 1 | 2 | DONE |
| 15 | Atomic `conformal --out` | 3 | 1 | 1 | 1 | DONE |
| 16 | Retry locked `os.replace` on Windows | 3 | 1 | 1 | 2 | DONE (untested on Windows) |
| 17 | Test that version/licence/author/URLs agree across pyproject, `__version__`, CHANGELOG, LICENSE | 4 | 1 | 1 | 1 | DONE |
| 18 | README: `>=0.4` install, GitHub links, Windows/limits, exit codes | 4 | 1 | 1 | 1 | DONE |
| 19 | PUBLISH-CHECKLIST.md for the owner | 4 | 1 | 1 | 1 | DONE |
| 20 | Persist the BM25 index in the store (skip re-tokenizing on load; 50k docs ~4 s) | 4 | 2 | 4 | 4 | NOT-DONE (store format change; risks the fail-closed checks) |
| 21 | Dense/hybrid + public-dataset (BEIR/MS MARCO) benchmark | 5 | 5 | 5 | 2 | NOT-DONE (no offline data; hosted embeddings cost money) |
| 22 | Real Windows / macOS / Python 3.10 run | 5 | 1 | 2 | 1 | NOT-DONE (no such machine; CI job written) |
| 23 | Trusted-publishing (OIDC) release workflow | 3 | 1 | 2 | 3 | NOT-DONE (needs owner's PyPI setup; would be a publish path) |
| 24 | Cross-process locking for the audit log / store writers | 3 | 2 | 4 | 3 | NOT-DONE (documented single writer) |
| 25 | ANN index for 500k+ vectors | 3 | 2 | 5 | 3 | NOT-DONE (README points to external vector DBs) |
| 26 | Pin CI actions by commit SHA | 2 | 1 | 1 | 1 | NOT-DONE (could not look up SHAs; tags used) |

## What was built
Everything marked DONE, with 18 new tests in `tests/test_round4.py` (303 -> 321). No scoring code
changed, so the benchmark is unchanged.

## Pass B: attacking the result
Re-ran, from the **freshly built wheel installed non-editable** into new venvs (3.11, 3.12, 3.13,
working directory with spaces):
- every README quick-start and CLI command (`ingest`, `query`, `--gate-preset strict`,
  `--fail-on-insufficient` -> exit 1, `info`, `ingest --json`, the Python quick start,
  `gate_override`, a generator that fails faithfulness -> `needs_review`, `explain_abstention`);
- hostile inputs: BOM+CRLF via stdin, cp1252 console, lone surrogate, UTF-16 file, bad gate file,
  empty qrels, empty query, 4 concurrent CLI processes on one store;
- the full suite against the installed wheel on 3.11 and 3.12: 321 passed each (321 on the
  checkout under 3.13).

Found in this pass: the README claim that the benchmark takes "~10 s" was wrong (about 18-20 s of
CPU) - fixed. A 200 KB query passed as a command-line argument fails in the shell ("Argument list
too long"), not in the tool; use the Python API for such inputs.

## Benchmark
Unchanged: `benchmarks/results.json` is byte-identical before and after (default gate 12.3% false
answers / 99.2% recall; `conformal` 4.4% / 98.0%; with `delta` 2.5% / 93.4%). Startup: first
tokenization ~0.4-0.5 s -> ~0.07 s. Package: the wheel holds the 25 modules, `py.typed`, LICENSE
and metadata only; `twine check` passes for wheel and sdist.

## Could not test
- **Windows** (any version), macOS, and Python **3.10** (not installed): reviewed statically only.
- Read-only filesystem as a non-root user (the sandbox runs as root, so permissions are ignored).
- Hybrid mode with real `sentence-transformers` (a ~2 GB download): only the fake embedder.
- The GitHub Actions workflow has never run; the action versions are tags I believe exist (unverified).
- No upload, tag or push was done; no TestPyPI rehearsal.
- LangGraph / Ragas / DeepEval / bm25s / arXiv comparison rows were not re-fetched.
