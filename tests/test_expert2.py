"""Expert round 2: conformal calibration, selective metrics, ScoreGate, monitor, abstention UX."""

from __future__ import annotations

import json
import math
import pathlib
import random
import threading

import pytest

from rerankgate import Document, PipelineConfig, RAGPipeline
from rerankgate import calibration as cal
from rerankgate.abstain import AMBIGUOUS, ERROR, NO_MATCH, PARTIAL, explain_abstention
from rerankgate.cli import main
from rerankgate.metrics import EvalItem
from rerankgate.monitor import Baseline, GateMonitor, ks_distance, psi
from rerankgate.scoregate import ScoreGate
from rerankgate.selective import (
    abstention_stats,
    aurc,
    oracle_aurc,
    risk_at_coverage,
    risk_coverage_curve,
)

DATA = pathlib.Path(__file__).resolve().parent.parent / "benchmarks" / "data"


# ---------------------------------------------------------------- calibration math
def test_binom_cdf_matches_bruteforce():
    for n, p in [(10, 0.3), (50, 0.05), (200, 0.5)]:
        for k in (0, 1, 5, n // 2, n - 1, n):
            ref = sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))
            assert cal.binom_cdf(k, n, p) == pytest.approx(ref, abs=1e-9)


def test_min_negatives_and_too_few_fail_closed():
    assert cal.min_negatives(0.05) == 19
    assert cal.min_negatives(0.05, 0.05) == 59
    assert cal.conformal_threshold([0.1] * 18, 0.05) == math.inf  # needs 19
    assert cal.conformal_threshold([0.1] * 19, 0.05) < math.inf
    assert cal.pac_threshold([0.1] * 58, 0.05, 0.05) == math.inf
    assert cal.pac_threshold([0.1] * 59, 0.05, 0.05) < math.inf
    assert cal.conformal_threshold([], 0.5) == math.inf


@pytest.mark.parametrize("bad", [math.nan, math.inf, "x", None, True])
def test_non_finite_calibration_scores_raise(bad):
    with pytest.raises(ValueError):
        cal.conformal_threshold([0.1, 0.2, bad], 0.5)
    with pytest.raises(ValueError):
        cal.pac_threshold([0.1, bad], 0.5, 0.5)


@pytest.mark.parametrize("a", [0, 1, -0.1, 1.5, math.nan, True])
def test_bad_alpha_rejected(a):
    with pytest.raises(ValueError):
        cal.conformal_threshold([0.1] * 50, a)


def test_monte_carlo_conformal_expected_far_bounded():
    """Uniform scores: true FAR of 'pass iff s > tau' is exactly 1 - tau."""
    rng = random.Random(1)
    alpha, n, trials = 0.1, 99, 4000
    fars = []
    for _ in range(trials):
        tau = cal.conformal_threshold([rng.random() for _ in range(n)], alpha)
        fars.append(1 - tau)
    assert sum(fars) / trials <= alpha + 0.004  # MC error ~0.0007; guarantee is <= alpha
    assert sum(fars) / trials >= alpha - 0.02  # and it is not wastefully conservative


def test_monte_carlo_pac_failure_rate_below_delta():
    rng = random.Random(2)
    alpha, delta, n, trials = 0.1, 0.1, 150, 4000
    bad = 0
    for _ in range(trials):
        tau = cal.pac_threshold([rng.random() for _ in range(n)], alpha, delta)
        if 1 - tau > alpha:
            bad += 1
    assert bad / trials <= delta + 0.02  # MC error ~0.005


def test_guarantee_conservative_with_tied_scores():
    rng = random.Random(3)
    alpha, trials = 0.1, 2000
    tot = 0.0
    for _ in range(trials):
        draw = [rng.randint(0, 9) / 10 for _ in range(60)]  # heavy ties
        tau = cal.conformal_threshold(draw, alpha)
        true_far = sum(1 for v in range(10) if v / 10 > tau) / 10
        tot += true_far
    assert tot / trials <= alpha


def test_p_value_and_nonfinite():
    neg = [0.1, 0.2, 0.3, 0.4]
    assert cal.conformal_p_value(neg, 0.5) == pytest.approx(1 / 5)
    assert cal.conformal_p_value(neg, 0.0) == 1.0
    assert cal.conformal_p_value(neg, math.nan) == 1.0
    assert cal.conformal_p_value(neg, math.inf) == 1.0


def test_wilson_and_nudge():
    lo, hi = cal.wilson_interval(0, 100)
    assert lo == 0.0 and 0.0 < hi < 0.05
    assert cal.wilson_interval(0, 0) == (0.0, 1.0)
    t = 0.5957
    assert not (t >= cal.nudge_up(t)) and (cal.nudge_up(t) > t)


# ---------------------------------------------------------------- selective metrics
def test_aurc_oracle_and_random_reference():
    correct = [True, True, True, False, False]
    perfect = [0.9, 0.8, 0.7, 0.2, 0.1]
    worst = [0.1, 0.2, 0.3, 0.8, 0.9]
    assert aurc(perfect, correct) == pytest.approx(oracle_aurc(correct))
    assert aurc(worst, correct) > aurc(perfect, correct)
    assert aurc([0.5] * 5, correct) == pytest.approx(0.4)  # one tie group = overall error


def test_risk_coverage_ties_grouped_and_curve():
    pts = risk_coverage_curve([0.9, 0.5, 0.5, 0.1], [True, True, False, False])
    assert pts == [(0.25, 0.0), (0.75, pytest.approx(1 / 3)), (1.0, 0.5)]
    assert risk_at_coverage([0.9, 0.5, 0.5, 0.1], [True, True, False, False], 0.5) == 0.0


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_selective_rejects_non_finite(bad):
    with pytest.raises(ValueError):
        aurc([0.1, bad], [True, False])
    with pytest.raises(ValueError):
        aurc([], [])


def test_abstention_stats():
    st = abstention_stats([True, True, False, False], [True, False, True, False])
    assert st.far == 0.5 and st.answerable_recall == 0.5
    assert st.abstention_precision == 0.5 and st.abstention_recall == 0.5
    assert st.coverage == 0.5


# ---------------------------------------------------------------- pipeline integration
def _bench_pipe(retries: int = 2) -> tuple[RAGPipeline, list[dict]]:
    docs = [json.loads(x) for x in open(DATA / "corpus.jsonl", encoding="utf-8")]
    qs = [json.loads(x) for x in open(DATA / "queries.jsonl", encoding="utf-8")]
    cfg = PipelineConfig(use_vectors=False)
    cfg.retry.max_retries = retries
    p = RAGPipeline(cfg)
    p.ingest([Document(id=d["id"], text=d["text"]) for d in docs])
    return p, qs


def test_gate_and_reranker_use_original_query_on_retries():
    """Regression: expansion terms come from the evidence, so judging the gate on the
    expanded query was circular and let off-topic queries pass after a retry."""
    seen: list[str] = []

    class Spy:
        name = "spy"

        def rerank(self, query, candidates, k):
            seen.append(query)
            return candidates[:k]

    p = RAGPipeline(PipelineConfig(use_vectors=False), reranker=Spy())
    p.ingest(
        [
            Document(id="a", text="alpha beta gamma delta epsilon zeta eta theta " * 5),
            Document(id="b", text="iota kappa lambda mu nu xi omicron pi rho sigma " * 5),
        ]
    )
    res = p.ask("alpha quokka")
    assert res.trace["attempts"] >= 1
    assert set(seen) == {"alpha quokka"}


def test_retries_do_not_inflate_false_answers():
    p0, qs = _bench_pipe(0)
    p2, _ = _bench_pipe(2)
    neg = [q["query"] for q in qs if not q["answerable"]]
    f0 = sum(p0.ask(q).verdict == "answerable" for q in neg)
    f2 = sum(p2.ask(q).verdict == "answerable" for q in neg)
    assert f2 <= f0 + 3  # was +63 before the fix (27 -> 119 of 223)


def test_conformal_gate_matches_deployed_decisions_exactly():
    p, qs = _bench_pipe(2)
    items = [EvalItem(q["query"], q["relevant_ids"], q["answerable"]) for q in qs]
    res = p.conformal_gate(items, alpha=0.05, delta=0.1)
    assert res.achievable and res.tau is not None
    neg = [it for it in items if not it.answerable]
    passes = sum(p.ask(it.query).verdict == "answerable" for it in neg)
    assert passes / len(neg) == pytest.approx(res.calibration_far)
    assert res.calibration_far <= 0.05
    assert res.answerable_recall is not None and res.answerable_recall > 0.5


def test_conformal_gate_too_few_negatives_never_passes():
    p, qs = _bench_pipe(0)
    ans = [q for q in qs if q["answerable"]][:30]
    neg = [q for q in qs if not q["answerable"]][:5]
    items = [EvalItem(q["query"], q["relevant_ids"], q["answerable"]) for q in ans + neg]
    res = p.conformal_gate(items, alpha=0.05)
    assert not res.achievable and res.tau is None
    assert "NONE" in res.guarantee
    for q in ans[:5]:  # even an exact-topic query is refused: fail closed
        assert p.ask(q["query"]).verdict != "answerable"


def test_conformal_gate_input_validation():
    p, _ = _bench_pipe(0)
    with pytest.raises(ValueError):
        p.conformal_gate([])
    with pytest.raises(ValueError):
        p.conformal_gate([EvalItem("alpha", ["a"], True)])


def test_gate_scores_restore_state_and_monitor():
    p, _ = _bench_pipe(0)
    p.monitor = GateMonitor(window=50, min_n=10)
    gate_before = p.gate
    p.gate_scores(["getpass prompt password"])
    assert p.gate is gate_before and p.monitor.check().n == 0
    p.ask("getpass prompt password")
    assert p.monitor.check().n == 1


def test_passing_attempt_preferred_over_higher_scored_floor_failure():
    # decision tuple ordering: (passed, score) - a pass wins even with a lower weighted score
    assert (True, 0.6) > (False, 0.9)


# ---------------------------------------------------------------- ScoreGate
def _synthetic_lists(rng, n_pos, n_neg):
    lists, labels = [], []
    for _ in range(n_pos):
        lists.append(
            sorted([rng.gauss(2.0, 0.7)] + [rng.gauss(0.5, 0.5) for _ in range(5)], reverse=True)
        )
        labels.append(True)
    for _ in range(n_neg):
        lists.append(sorted([rng.gauss(0.6, 0.5) for _ in range(6)], reverse=True))
        labels.append(False)
    return lists, labels


def test_scoregate_monte_carlo_far():
    rng = random.Random(4)
    alpha = 0.1
    fars = []
    for _ in range(60):
        tl, tlab = _synthetic_lists(rng, 150, 150)
        g = ScoreGate.fit(tl, tlab, alpha=alpha, statistic="top1")
        vl, vlab = _synthetic_lists(rng, 0, 1500)
        fars.append(sum(g.decide(x).passed for x in vl) / len(vl))
    assert sum(fars) / len(fars) <= alpha + 0.015


def test_scoregate_auto_roundtrip_and_fail_closed():
    rng = random.Random(5)
    lists, labels = _synthetic_lists(rng, 120, 120)
    g = ScoreGate.fit(lists, labels, alpha=0.1)
    assert g.statistic in ("top1", "gap", "zgap")
    g2 = ScoreGate.from_json(g.to_json())
    x = lists[0]
    assert g2.decide(x).to_dict() == g.decide(x).to_dict()
    assert not g.decide([]).passed and g.decide([]).reason == "NO_EVIDENCE"
    for bad in ([math.nan, 1.0], [math.inf], ["a"], [None, 2.0], [True, 0.2]):
        d = g.decide(bad)
        assert not d.passed and d.reason == "INVALID_SCORE" and d.p_value == 1.0


def test_scoregate_lower_is_better_distances():
    rng = random.Random(6)
    lists, labels = _synthetic_lists(rng, 100, 100)
    dist = [[-v for v in lst] for lst in lists]  # distances: smaller is better
    g = ScoreGate.fit(dist, labels, alpha=0.1, statistic="top1", higher_is_better=False)
    assert (
        g.decide(dist[0]).passed
        == ScoreGate.fit(lists, labels, 0.1, statistic="top1").decide(lists[0]).passed
    )


def test_scoregate_too_few_negatives_never_passes():
    lists = [[3.0, 1.0]] * 10 + [[0.2, 0.1]] * 5
    g = ScoreGate.fit(lists, [True] * 10 + [False] * 5, alpha=0.05, statistic="top1")
    assert g.tau is None
    d = g.decide([100.0, 1.0])
    assert not d.passed and d.reason == "NOT_ENOUGH_CALIBRATION_DATA"


def test_scoregate_bad_inputs_and_tampered_file():
    with pytest.raises(ValueError):
        ScoreGate.fit([[1.0]], [True, False])
    with pytest.raises(ValueError):
        ScoreGate.fit([[1.0]] * 3, [True] * 3, statistic="top1")  # no negatives
    with pytest.raises(ValueError):
        ScoreGate.fit([[1.0]] * 3, [True, False, False], statistic="nope")
    for text in ('{"format":1}', "not json", '{"format":2}'):
        with pytest.raises(ValueError):
            ScoreGate.from_json(text)
    g = ScoreGate.fit(
        [[2.0]] * 30 + [[0.1]] * 30, [True] * 30 + [False] * 30, 0.1, statistic="top1"
    )
    d = json.loads(g.to_json())
    d["tau"] = "NaN"
    with pytest.raises(ValueError):
        ScoreGate.from_json(json.dumps(d))


# ---------------------------------------------------------------- monitor
def _feed(mon, rng, n, p_answer, mu, lat=5.0):
    for _ in range(n):
        s = rng.gauss(mu, 0.1)
        mon.record(answered=rng.random() < p_answer, score=s, latency_ms=lat)


def _baseline(rng):
    sc = [rng.gauss(0.5, 0.1) for _ in range(1000)]
    ans = [rng.random() < 0.6 for _ in sc]
    return Baseline.from_events(sc, ans)


def test_monitor_warmup_then_ok_then_alerts():
    rng = random.Random(7)
    mon = GateMonitor(_baseline(rng), window=400, min_n=100)
    _feed(mon, rng, 50, 0.6, 0.5)
    assert mon.check().status == "warming_up"
    _feed(mon, rng, 350, 0.6, 0.5)
    assert mon.check().status == "ok"
    _feed(mon, rng, 400, 0.2, 0.5)  # abstain rate shifts (answers drop)
    r = mon.check()
    assert r.status == "alert" and any(a.startswith("ABSTAIN_RATE_SHIFT") for a in r.alerts)
    _feed(mon, rng, 400, 0.6, 0.8)  # score distribution shifts
    r = mon.check()
    assert any(a.startswith("SCORE_DRIFT") for a in r.alerts) and r.psi > 0.25 and r.ks > 0.5


def test_monitor_stationary_false_alarm_rate_is_low():
    rng = random.Random(8)
    base = _baseline(rng)
    alarms = 0
    for _ in range(100):
        mon = GateMonitor(base, window=300, min_n=100)
        _feed(mon, rng, 300, 0.6, 0.5)
        alarms += mon.check().status == "alert"
    assert alarms <= 3


def test_monitor_invalid_scores_and_latency_always_alert():
    mon = GateMonitor(window=20, min_n=10, max_p95_latency_ms=50)
    mon.record(True, math.nan, 10)
    mon.record(True, "x", 10)  # type: ignore[arg-type]
    mon.record(True, math.inf, 500)
    r = mon.check()
    assert r.invalid == 3 and r.abstain_rate == 1.0  # invalid counts as abstain
    assert any(a.startswith("INVALID_SCORES") for a in r.alerts)
    assert any(a.startswith("LATENCY") for a in r.alerts)


def test_monitor_thread_safety_and_window_cap():
    mon = GateMonitor(window=100, min_n=10)

    def work():
        for i in range(1000):
            mon.record(i % 2 == 0, 0.5, 1.0)

    ts = [threading.Thread(target=work) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert mon.check().n == 100


def test_baseline_roundtrip_and_malformed():
    rng = random.Random(9)
    b = _baseline(rng)
    b2 = Baseline.from_json(b.to_json())
    assert b2.edges == b.edges and b2.abstain_rate == b.abstain_rate
    for text in ('{"n":1}', '{"n":5,"abstain_rate":2,"edges":[0.1],"scores":[0.2]}'):
        with pytest.raises(ValueError):
            Baseline.from_json(text)
    with pytest.raises(ValueError):
        Baseline.from_events([0.1] * 10, [True] * 10)
    with pytest.raises(ValueError):
        Baseline.from_events([math.nan] * 30, [True] * 30)
    with pytest.raises(ValueError):
        GateMonitor(window=5)


def test_psi_ks_identity():
    rng = random.Random(10)
    a = [rng.random() for _ in range(2000)]
    edges = [i / 10 for i in range(1, 10)]
    assert psi(edges, [0.1] * 10, a) < 0.05
    assert ks_distance(a, a) == 0.0 and ks_distance(a, [x + 1 for x in a]) == 1.0


# ---------------------------------------------------------------- abstention UX
def _tiny():
    p = RAGPipeline(PipelineConfig(use_vectors=False))
    p.ingest(
        [
            Document(
                id="d1",
                text="Sourdough bread needs a starter flour water and patience.",
                metadata={"title": "Bread basics"},
            ),
            Document(
                id="d2", text="Retrieval augmented generation pairs a retriever with a model."
            ),
        ]
    )
    return p


def test_explain_none_when_answerable():
    assert explain_abstention(_tiny().ask("sourdough starter flour")) is None


def test_explain_no_match_and_message_never_echoes_evidence():
    r = _tiny().ask("xylophone concerto tickets")
    n = explain_abstention(r)
    assert n.kind == NO_MATCH and n.action == "search" and n.nearest_sources == []
    assert "Sourdough" not in n.message


def test_explain_partial_lists_missing_terms():
    r = _tiny().ask("sourdough starter quantum chromodynamics lagrangian")
    n = explain_abstention(r)
    assert n.kind == PARTIAL and "sourdough" in n.matched_terms
    assert "quantum" in n.missing_terms and n.action == "clarify"
    assert n.nearest_sources and n.nearest_sources[0]["title"] == "Bread basics"
    assert "does not answer" in n.nearest_sources[0]["note"]


def test_explain_ambiguous_branch():
    r = _tiny().ask("bread")
    r.verdict = "insufficient"
    r.gate.reasons = ["NO_MARGIN"]
    assert explain_abstention(r).kind == AMBIGUOUS


def test_explain_error_on_invalid_score():
    class NaNRerank:
        name = "nan"

        def rerank(self, q, c, k):
            from rerankgate.retrieval import ScoredChunk

            return [ScoredChunk(chunk=c[0].chunk, score=math.nan)]

    p = RAGPipeline(PipelineConfig(use_vectors=False), reranker=NaNRerank())
    p.ingest([Document(id="d", text="alpha beta gamma delta")])
    n = explain_abstention(p.ask("alpha"))
    assert n.kind == ERROR and n.action == "retry_later"


def test_explain_escapes_untrusted_text():
    p = RAGPipeline(PipelineConfig(use_vectors=False))
    p.ingest([Document(id="evil\x1b[31m", text="alpha beta gamma", metadata={"title": "T‮X"})])
    n = explain_abstention(p.ask("alpha zebra \x1b[2Jwombat"))
    blob = json.dumps(n.to_dict(), ensure_ascii=True)
    assert "\x1b" not in n.message and "‮" not in n.message
    assert "\\u001b" in blob or "\\\\u001b" in blob


# ---------------------------------------------------------------- CLI
@pytest.fixture(scope="module")
def store(tmp_path_factory):
    d = tmp_path_factory.mktemp("st")
    assert (
        main(["ingest", "--input", str(DATA / "corpus.jsonl"), "--store", str(d), "--no-vectors"])
        == 0
    )
    return str(d)


def test_cli_conformal_writes_gate_file_and_query_uses_it(store, tmp_path, capsys):
    out = tmp_path / "gate.json"
    rc = main(
        [
            "conformal",
            "--store",
            store,
            "--qrels",
            str(DATA / "queries.jsonl"),
            "--no-vectors",
            "--alpha",
            "0.05",
            "--delta",
            "0.1",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    g = json.loads(out.read_text())
    assert g["threshold"] > 0 and g["calibration"]["achievable"]
    capsys.readouterr()
    assert (
        main(
            [
                "query",
                "--store",
                store,
                "--no-vectors",
                "--gate-file",
                str(out),
                "--json",
                "getpass prompt password echo",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["gate"]["threshold"] == pytest.approx(g["threshold"])


def test_cli_conformal_unachievable_exits_1(store, tmp_path):
    q = tmp_path / "q.jsonl"
    q.write_text(
        '{"query":"getpass prompt","relevant_ids":["getpass"],"answerable":true}\n'
        '{"query":"xylophone","relevant_ids":[],"answerable":false}\n'
    )
    assert main(["conformal", "--store", store, "--qrels", str(q), "--no-vectors"]) == 1


@pytest.mark.parametrize(
    "content", ["{}", '{"threshold": -1}', '{"threshold": "x"}', "nope", '{"threshold": true}']
)
def test_cli_bad_gate_file_exits_2(store, tmp_path, content):
    f = tmp_path / "g.json"
    f.write_text(content)
    assert main(["query", "--store", store, "--no-vectors", "--gate-file", str(f), "x"]) == 2
    assert (
        main(
            [
                "query",
                "--store",
                store,
                "--no-vectors",
                "--gate-file",
                str(tmp_path / "missing"),
                "x",
            ]
        )
        == 2
    )


def test_cli_conformal_bad_alpha(store):
    with pytest.raises(SystemExit):
        main(["conformal", "--store", store, "--qrels", "x", "--alpha", "1"])


def test_cli_riskcov(store, capsys):
    assert (
        main(["riskcov", "--store", store, "--qrels", str(DATA / "queries.jsonl"), "--no-vectors"])
        == 0
    )
    rep = json.loads(capsys.readouterr().out)
    assert rep["aurc_oracle"] <= rep["aurc"] < rep["aurc_no_gate"]


def test_explain_ignores_needs_review():
    r = _tiny().ask("sourdough starter flour")
    r.verdict = "needs_review"
    assert explain_abstention(r) is None


def test_scoregate_p_value_matches_reference():
    neg = sorted(random.Random(11).random() for _ in range(500))
    g = ScoreGate(
        statistic="top1",
        higher_is_better=True,
        alpha=0.1,
        delta=None,
        tau=0.9,
        neg_stats=neg,
        n_calibration=500,
    )
    for v in (0.0, 0.3, neg[10], 0.99, 2.0):
        assert g.decide([v]).p_value == pytest.approx(cal.conformal_p_value(neg, v))
