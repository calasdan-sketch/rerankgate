"""Committed, re-runnable benchmark: does the gate stop false answers, and what does it cost?

    python benchmarks/bench_gate.py            # prints a table, writes benchmarks/results.json

Data: benchmarks/data/{corpus,queries}.jsonl (see build_dataset.py; synthetic labels over
real stdlib/numpy/scipy docstring text; BM25-only retrieval because the benchmark must run
with zero installs). Procedure: score every query once with a threshold-free gate, then
repeat R random 50/50 calibration/test splits; each method sets its threshold on the
calibration half and is judged on the test half. Reported: mean over splits.

Methods
  cutoff-F1      raw top-1 BM25 score with a fixed cutoff tuned for F1 on the calibration
                 half - the best a "similarity cutoff / score_threshold" user can do
  heuristic      rerankgate default gate (threshold 0.55), no calibration
  conformal      rerankgate + conformal_threshold(alpha)             (expected FAR <= alpha)
  conformal-PAC  rerankgate + pac_threshold(alpha, delta)            (FAR <= alpha w.p. 1-delta)
  ScoreGate      retriever-agnostic gate on raw BM25 score lists (auto statistic)
"""

from __future__ import annotations

import json
import pathlib
import random
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))

from rerankgate import Document, PipelineConfig, RAGPipeline  # noqa: E402
from rerankgate import calibration as cal  # noqa: E402
from rerankgate.scoregate import ScoreGate  # noqa: E402
from rerankgate.selective import abstention_stats, aurc, oracle_aurc  # noqa: E402

ALPHA, DELTA, R = 0.05, 0.10, 200


def load():
    docs = [json.loads(x) for x in open(ROOT / "data/corpus.jsonl", encoding="utf-8")]
    qs = [json.loads(x) for x in open(ROOT / "data/queries.jsonl", encoding="utf-8")]
    return docs, qs


def best_f1_cutoff(vals, labels):
    best = (-1.0, 0.0)
    for t in sorted(set(vals)):
        tp = sum(1 for v, y in zip(vals, labels, strict=True) if v >= t and y)
        fp = sum(1 for v, y in zip(vals, labels, strict=True) if v >= t and not y)
        fn = sum(1 for v, y in zip(vals, labels, strict=True) if v < t and y)
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best[0]:
            best = (f1, t)
    return best[1]


def main() -> None:
    docs, qs = load()
    out = run(docs, qs, stem=False)
    print("\n-- same benchmark with PipelineConfig(stemming=True) --")
    out["variants"] = {"stemming": run(docs, qs, stem=True)}
    out["inflected_queries"] = inflected(docs, qs)
    print("\n-- queries with every long word flipped singular/plural (default threshold 0.55) --")
    print(json.dumps(out["inflected_queries"], indent=2))
    (ROOT / "results.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


def flip_number(query: str) -> str:
    """Flip singular/plural on every word of 5+ letters: 'handles' <-> 'handle'. Deterministic."""
    out = []
    for w in query.split():
        core = w.strip(".,?!;:()\"'")
        if len(core) >= 5 and core.isascii() and core.isalpha():
            new = core[:-1] if core.endswith("s") and not core.endswith("ss") else core + "s"
            w = w.replace(core, new)
        out.append(w)
    return " ".join(out)


def inflected(docs, qs) -> dict:
    """Answerable / unanswerable queries after a morphological perturbation, default gate."""
    res = {}
    for name, stem in (("default", False), ("stemming", True)):
        pipe = RAGPipeline(PipelineConfig(use_vectors=False, stemming=stem))
        pipe.ingest([Document(id=d["id"], text=d["text"]) for d in docs])
        a_ok = a_n = u_bad = u_n = 0
        for q in qs:
            passed = pipe.ask(flip_number(q["query"])).gate.passed
            if q["answerable"]:
                a_n += 1
                a_ok += passed
            else:
                u_n += 1
                u_bad += passed
        res[name] = {
            "answerable_answered": round(a_ok / a_n, 4),
            "unanswerable_answered": round(u_bad / u_n, 4),
        }
    return res


def run(docs, qs, stem: bool) -> dict:
    cfg = PipelineConfig(use_vectors=False, stemming=stem)
    pipe = RAGPipeline(cfg)
    pipe.ingest([Document(id=d["id"], text=d["text"]) for d in docs])
    results = pipe.score_results([q["query"] for q in qs])
    gate_s = [float(r.trace["gate_score"]) for r in results]
    answerable = [q["answerable"] for q in qs]
    correct = [
        a and any(c.chunk.doc_id in set(q["relevant_ids"]) for c in r.evidence[:3])
        for q, r, a in zip(qs, results, answerable, strict=True)
    ]
    bm25_lists = [[s for _, s in pipe.bm25.search(q["query"], k=10)] for q in qs]
    top_bm25 = [lst[0] if lst else 0.0 for lst in bm25_lists]
    thr_heur = pipe.config.effective_gate().threshold

    methods = ["cutoff-F1", "heuristic", "conformal", "conformal-PAC", "ScoreGate"]
    acc = {m: {"far": [], "recall": [], "prec": [], "cov": []} for m in methods}
    rng = random.Random(7)
    idx = list(range(len(qs)))
    for _ in range(R):
        rng.shuffle(idx)
        cal_i, test_i = idx[: len(idx) // 2], idx[len(idx) // 2 :]
        neg = [gate_s[i] for i in cal_i if not answerable[i]]
        decisions = {}
        cut = best_f1_cutoff([top_bm25[i] for i in cal_i], [answerable[i] for i in cal_i])
        decisions["cutoff-F1"] = [top_bm25[i] >= cut for i in test_i]
        decisions["heuristic"] = [gate_s[i] >= thr_heur for i in test_i]
        tau = cal.conformal_threshold(neg, ALPHA)
        decisions["conformal"] = [gate_s[i] > tau for i in test_i]
        tau = cal.pac_threshold(neg, ALPHA, DELTA)
        decisions["conformal-PAC"] = [gate_s[i] > tau for i in test_i]
        sg = ScoreGate.fit(
            [bm25_lists[i] for i in cal_i], [answerable[i] for i in cal_i], alpha=ALPHA
        )
        decisions["ScoreGate"] = [sg.decide(bm25_lists[i]).passed for i in test_i]
        ans_t = [answerable[i] for i in test_i]
        cor_t = [correct[i] for i in test_i]
        for m in methods:
            st = abstention_stats(decisions[m], ans_t, cor_t)
            acc[m]["far"].append(st.far)
            acc[m]["recall"].append(st.answerable_recall)
            acc[m]["prec"].append(st.answer_precision)
            acc[m]["cov"].append(st.coverage)

    out = {
        "dataset": {
            "corpus_docs": len(docs),
            "queries": len(qs),
            "answerable": sum(answerable),
            "unanswerable": len(qs) - sum(answerable),
        },
        "alpha": ALPHA,
        "delta": DELTA,
        "splits": R,
        "aurc": {
            "gate_score": round(aurc(gate_s, correct), 4),
            "raw_top1_bm25": round(aurc(top_bm25, correct), 4),
            "oracle": round(oracle_aurc(correct), 4),
            "no_gate": round(1 - sum(correct) / len(correct), 4),
        },
        "methods": {},
    }
    print(f"{'method':14} {'FAR':>6} {'FAR>a %':>8} {'recall':>7} {'precision':>9} {'coverage':>8}")
    for m in methods:
        far = acc[m]["far"]
        row = {
            "far_mean": round(statistics.fmean(far), 4),
            "splits_far_above_alpha": round(sum(1 for f in far if f > ALPHA) / R, 3),
            "answerable_recall": round(statistics.fmean(acc[m]["recall"]), 4),
            "answer_precision": round(statistics.fmean(acc[m]["prec"]), 4),
            "coverage": round(statistics.fmean(acc[m]["cov"]), 4),
        }
        out["methods"][m] = row
        print(
            f"{m:14} {row['far_mean']:6.3f} {row['splits_far_above_alpha']:8.2f} "
            f"{row['answerable_recall']:7.3f} {row['answer_precision']:9.3f} {row['coverage']:8.3f}"
        )
    print("AURC (lower is better):", out["aurc"])
    return out


if __name__ == "__main__":
    main()
