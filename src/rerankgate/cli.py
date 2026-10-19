"""rerankgate CLI: ingest | query | eval."""

from __future__ import annotations

import argparse
import json
import sys

from rerankgate.config import PipelineConfig
from rerankgate.documents import Document
from rerankgate.metrics import EvalItem
from rerankgate.pipeline import RAGPipeline


def _pipeline_for(args: argparse.Namespace, *, for_ingest: bool = False) -> RAGPipeline:
    cfg = PipelineConfig()
    if args.gate_preset:
        cfg.gate_preset = args.gate_preset
    if args.max_retries is not None:
        cfg.retry.max_retries = args.max_retries
    if args.no_vectors:
        cfg.use_vectors = False
    if for_ingest:
        return RAGPipeline(config=cfg)
    return RAGPipeline.load(args.store, config=cfg)


def cmd_ingest(args: argparse.Namespace) -> int:
    docs: list[Document] = []
    with open(args.input, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            meta = {k: v for k, v in obj.items() if k not in ("id", "text")}
            docs.append(Document(id=str(obj.get("id", f"doc-{i}")), text=text, metadata=meta))
    pipe = _pipeline_for(args, for_ingest=True)
    n_chunks = pipe.ingest(docs)
    pipe.save(args.store)
    print(f"ingested {len(docs)} documents -> {n_chunks} chunks -> {args.store}")
    if pipe.degraded_reason:
        print("note: BM25-only mode (no embeddings installed)")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    pipe = _pipeline_for(args)
    result = pipe.ask(args.query)
    out = result.to_dict()
    if args.json:
        print(json.dumps(out, indent=2))
        return 0
    print(f"verdict:  {out['verdict']}")
    print(
        f"gate:     {'PASS' if out['gate']['passed'] else 'FAIL'} "
        f"(score {out['gate']['score']} vs threshold {out['gate']['threshold']})"
    )
    print(f"reasons:  {', '.join(out['gate']['reasons'])}")
    print(
        f"attempts: {out['trace']['attempts']}  "
        f"modes: {','.join(out['trace']['modes'])}  "
        f"total: {out['trace']['total_ms']}ms"
    )
    if out["trace"]["degraded"]:
        print("degraded: BM25-only (embeddings unavailable)")
    print("evidence:")
    for ev in out["evidence"][:10]:
        snippet = ev["text"][:120].replace("\n", " ")
        print(f"  [{ev['score']:.3f}] {ev['chunk_id']}: {snippet}...")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    pipe = _pipeline_for(args)
    items: list[EvalItem] = []
    with open(args.qrels, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(
                EvalItem(
                    query=obj["query"],
                    relevant_ids=list(obj.get("relevant_ids", [])),
                    answerable=bool(obj.get("answerable", True)),
                )
            )
    report = pipe.evaluate(items)
    if args.summary:
        report = {k: v for k, v in report.items() if k != "per_query"}
    print(json.dumps(report, indent=2))
    return 0


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
    _common(pe)
    pe.set_defaults(func=cmd_eval)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
