"""rerankgate CLI: ingest | query | eval | calibrate.

Exit codes: 0 ok · 1 a quality check failed (``--fail-on-insufficient``,
``--fail-under-*``) · 2 bad input or setup (unreadable/empty file, corrupt store).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator

from rerankgate.config import PipelineConfig
from rerankgate.documents import Document
from rerankgate.metrics import EvalItem
from rerankgate.pipeline import RAGPipeline
from rerankgate.safeterm import clean

EXIT_OK = 0
EXIT_CHECK_FAILED = 1
EXIT_BAD_INPUT = 2


class InputError(Exception):
    """User-facing input problem; reported without a traceback."""


def _read_jsonl(path: str, what: str) -> Iterator[tuple[int, dict]]:
    try:
        f = open(path, encoding="utf-8")
    except OSError as exc:
        raise InputError(f"cannot open {what} {path!r}: {exc.strerror or exc}") from exc
    with f:
        try:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError as exc:
                    raise InputError(f"{what} line {lineno}: invalid JSON ({exc.msg})") from exc
                if not isinstance(obj, dict):
                    raise InputError(f"{what} line {lineno}: expected a JSON object")
                yield lineno, obj
        except UnicodeDecodeError as exc:
            raise InputError(f"{what} is not valid UTF-8: {exc}") from exc


def _pipeline_for(args: argparse.Namespace, *, for_ingest: bool = False) -> RAGPipeline:
    cfg = PipelineConfig()
    if args.gate_preset:
        cfg.gate_preset = args.gate_preset
    if args.max_retries is not None:
        if args.max_retries < 0:
            raise InputError("--max-retries must be >= 0")
        cfg.retry.max_retries = args.max_retries
    if args.no_vectors:
        cfg.use_vectors = False
    if for_ingest:
        return RAGPipeline(config=cfg)
    return RAGPipeline.load(args.store, config=cfg)


def cmd_ingest(args: argparse.Namespace) -> int:
    docs: list[Document] = []
    for lineno, obj in _read_jsonl(args.input, "corpus"):
        text = obj.get("text", "")
        if not isinstance(text, str):
            raise InputError(f"corpus line {lineno}: 'text' must be a string")
        meta = {k: v for k, v in obj.items() if k not in ("id", "text")}
        raw_id = obj.get("id")
        doc_id = f"doc-{lineno}" if raw_id is None else str(raw_id)
        docs.append(Document(id=doc_id, text=text, metadata=meta))
    if not docs:
        raise InputError(f"corpus {args.input!r} has no documents; nothing to ingest")
    pipe = _pipeline_for(args, for_ingest=True)
    try:
        n_chunks = pipe.ingest(docs)
    except ValueError as exc:
        raise InputError(str(exc)) from exc
    if n_chunks == 0:
        raise InputError("every document was empty; refusing to write an empty store")
    pipe.save(args.store)
    print(f"ingested {len(docs)} documents -> {n_chunks} chunks -> {clean(args.store)}")
    if pipe.degraded_reason:
        print("note: BM25-only mode (no embeddings installed)")
    return EXIT_OK


def cmd_query(args: argparse.Namespace) -> int:
    pipe = _pipeline_for(args)
    result = pipe.ask(args.query)
    out = result.to_dict()
    rc = EXIT_OK
    if args.fail_on_insufficient and out["verdict"] != "answerable":
        rc = EXIT_CHECK_FAILED
    if args.json:
        # JSON escapes control characters itself; ensure_ascii also escapes
        # bidi/zero-width characters, so the output is terminal-safe.
        print(json.dumps(out, indent=2))
        return rc
    print(f"verdict:  {out['verdict']}")
    print(
        f"gate:     {'PASS' if out['gate']['passed'] else 'FAIL'} "
        f"(score {out['gate']['score']} vs threshold {out['gate']['threshold']})"
    )
    print(f"reasons:  {clean(', '.join(out['gate']['reasons']))}")
    print(
        f"attempts: {out['trace']['attempts']}  "
        f"modes: {','.join(out['trace']['modes'])}  "
        f"total: {out['trace']['total_ms']}ms"
    )
    if out["trace"]["degraded"]:
        print("degraded: BM25-only (embeddings unavailable)")
    print("evidence:")
    for ev in out["evidence"][:10]:
        snippet = clean(ev["text"][:120])
        print(f"  [{ev['score']:.3f}] {clean(ev['chunk_id'])}: {snippet}...")
    return rc


def _load_items(path: str) -> list[EvalItem]:
    items: list[EvalItem] = []
    for lineno, obj in _read_jsonl(path, "qrels"):
        query = obj.get("query")
        if not isinstance(query, str) or not query.strip():
            raise InputError(f"qrels line {lineno}: 'query' must be a non-empty string")
        rel = obj.get("relevant_ids", [])
        if not isinstance(rel, list) or not all(isinstance(r, str) for r in rel):
            raise InputError(f"qrels line {lineno}: 'relevant_ids' must be a list of strings")
        answerable = obj.get("answerable", True)
        if not isinstance(answerable, bool):
            raise InputError(f"qrels line {lineno}: 'answerable' must be true or false")
        items.append(EvalItem(query=query, relevant_ids=list(rel), answerable=answerable))
    if not items:
        raise InputError(f"qrels {path!r} has no items; an empty eval proves nothing")
    return items


def cmd_eval(args: argparse.Namespace) -> int:
    items = _load_items(args.qrels)
    pipe = _pipeline_for(args)
    report = pipe.evaluate(items)
    failures = []
    for flag, key in (
        ("fail_under_precision", "gate_precision"),
        ("fail_under_recall", "gate_recall"),
        ("fail_under_mrr", "mrr"),
    ):
        floor = getattr(args, flag)
        if floor is not None and report[key] < floor:
            failures.append(f"{key} {report[key]} < {floor}")
    if failures:
        report["failed_checks"] = failures
    if args.summary:
        report = {k: v for k, v in report.items() if k != "per_query"}
    print(json.dumps(report, indent=2))
    if failures:
        print("FAILED: " + "; ".join(failures), file=sys.stderr)
        return EXIT_CHECK_FAILED
    return EXIT_OK


def cmd_calibrate(args: argparse.Namespace) -> int:
    items = _load_items(args.qrels)
    pipe = _pipeline_for(args)
    try:
        report = pipe.calibrate(items, target_precision=args.target_precision)
    except ValueError as exc:
        raise InputError(str(exc)) from exc
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"{'threshold':>9}  {'precision':>9}  {'recall':>6}  {'f1':>6}  {'abstain':>7}")
        for row in report["sweep"]:
            mark = "  <- recommended" if row["threshold"] == report["recommended_threshold"] else ""
            print(
                f"{row['threshold']:>9.2f}  {row['gate_precision']:>9.3f}  "
                f"{row['gate_recall']:>6.3f}  {row['f1']:>6.3f}  "
                f"{row['abstention_rate']:>7.3f}{mark}"
            )
        rec = report["recommended_threshold"]
        print(f"rule: {report['rule']}  (n={report['n']}, current={report['current_threshold']})")
        print(f"recommended threshold: {rec if rec is not None else 'none reaches the target'}")
    return EXIT_OK if report["recommended_threshold"] is not None else EXIT_CHECK_FAILED


def _fraction(value: str) -> float:
    try:
        v = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from exc
    if not 0.0 <= v <= 1.0:
        raise argparse.ArgumentTypeError(f"{value} must be between 0 and 1")
    return v


def build_parser() -> argparse.ArgumentParser:
    import rerankgate

    p = argparse.ArgumentParser(prog="rerankgate", description="Agentic RAG quality gate")
    p.add_argument("--version", action="version", version=f"%(prog)s {rerankgate.__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def _common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--gate-preset", choices=["strict", "balanced", "lenient"], default=None)
        sp.add_argument("--max-retries", type=int, default=None)
        sp.add_argument("--no-vectors", action="store_true", help="force BM25-only mode")

    pi = sub.add_parser("ingest", help="ingest a JSONL corpus into a store")
    pi.add_argument("--input", required=True, help="JSONL with {id, text, ...metadata}")
    pi.add_argument("--store", required=True, help="output store directory")
    _common(pi)
    pi.set_defaults(func=cmd_ingest)

    pq = sub.add_parser("query", help="ask a question against a store")
    pq.add_argument("--store", required=True)
    pq.add_argument("query")
    pq.add_argument("--json", action="store_true")
    pq.add_argument(
        "--fail-on-insufficient",
        action="store_true",
        help="exit 1 unless the verdict is 'answerable' (for scripts and CI)",
    )
    _common(pq)
    pq.set_defaults(func=cmd_query)

    pe = sub.add_parser("eval", help="evaluate retrieval + gate on labeled qrels")
    pe.add_argument("--store", required=True)
    pe.add_argument("--qrels", required=True, help="JSONL with {query, relevant_ids, answerable}")
    pe.add_argument(
        "--summary",
        action="store_true",
        help="print aggregate metrics only, without the per-query detail",
    )
    pe.add_argument("--fail-under-precision", type=_fraction, default=None, metavar="X")
    pe.add_argument("--fail-under-recall", type=_fraction, default=None, metavar="X")
    pe.add_argument("--fail-under-mrr", type=_fraction, default=None, metavar="X")
    _common(pe)
    pe.set_defaults(func=cmd_eval)

    pc = sub.add_parser("calibrate", help="sweep gate thresholds on qrels and recommend one")
    pc.add_argument("--store", required=True)
    pc.add_argument("--qrels", required=True, help="JSONL with {query, relevant_ids, answerable}")
    pc.add_argument(
        "--target-precision",
        type=_fraction,
        default=None,
        metavar="X",
        help="recommend the highest-recall threshold with gate precision >= X",
    )
    pc.add_argument("--json", action="store_true")
    _common(pc)
    pc.set_defaults(func=cmd_calibrate)
    return p


def main(argv: list[str] | None = None) -> int:
    from rerankgate.store import StoreError

    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except InputError as exc:
        print(f"rerankgate: error: {clean(exc)}", file=sys.stderr)
        return EXIT_BAD_INPUT
    except (StoreError, OSError) as exc:  # FileNotFoundError is an OSError
        print(f"rerankgate: error: {clean(exc)}", file=sys.stderr)
        return EXIT_BAD_INPUT


if __name__ == "__main__":
    sys.exit(main())
