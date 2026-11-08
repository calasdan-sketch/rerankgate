"""Round 5: property tests, differential tests, golden files, upgrade path, author pin."""

import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path

import pytest

import rerankgate
from rerankgate import calibration as cal
from rerankgate.bm25 import BM25Index
from rerankgate.cli import main
from rerankgate.documents import Chunk, chunk_text
from rerankgate.metrics import (
    EvalItem,
    aggregate,
    average_precision,
    ndcg_at_k,
    precision_at_k,
)
from rerankgate.scoregate import ScoreGate
from rerankgate.tokenize import tokenize

try:  # tomllib is Python 3.11+; requires-python is 3.10
    import tomllib
except ImportError:  # pragma: no cover
    tomllib = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AUTHOR = "Calas Automations Inc."


# --- author line ---------------------------------------------------------------------------


def test_author_string_is_pinned():
    assert rerankgate.__author__ == AUTHOR
    if tomllib is None or not (ROOT / "pyproject.toml").exists():
        pytest.skip("needs a checkout and tomllib")
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"]
    assert meta["authors"] == [{"name": AUTHOR}]
    assert meta["maintainers"] == [{"name": AUTHOR}]  # name only: no invented email
    lic = (ROOT / "LICENSE").read_text("utf-8")
    assert f"Copyright (c) 2026 {AUTHOR}" in lic


def test_installed_metadata_names_the_author():
    from importlib.metadata import PackageNotFoundError, metadata

    try:
        md = metadata("rerankgate")
    except PackageNotFoundError:  # pragma: no cover
        pytest.skip("package not installed")
    assert AUTHOR in (md.get("Author") or md.get("Author-email") or "")


# --- seeded generators (no extra dependency) -------------------------------------------------


def _scores(rng: random.Random, n: int, ties: bool) -> list[float]:
    if ties:
        return [rng.randint(0, 6) / 6 for _ in range(n)]
    return [rng.random() for _ in range(n)]


@pytest.mark.parametrize("seed", range(40))
def test_conformal_threshold_obeys_its_budget_on_calibration_scores(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 80)
    alpha = rng.choice([0.01, 0.05, 0.1, 0.2, 0.5])
    neg = _scores(rng, n, ties=seed % 2 == 0)
    tau = cal.conformal_threshold(neg, alpha)
    # at most floor(alpha*(n+1)) calibration scores sit above tau (ties only reduce it)
    assert sum(1 for v in neg if v > tau) <= math.floor(alpha * (n + 1) + 1e-9)
    for x in (rng.random() for _ in range(25)):
        # p-value and threshold agree: passing means the conformal p-value is within alpha
        if x > tau:
            assert cal.conformal_p_value(neg, x) <= alpha + 1e-9


def test_conformal_threshold_has_the_promised_coverage_by_simulation():
    """Fresh exchangeable negatives pass at rate <= alpha, averaged over calibration draws."""
    rng = random.Random(2024)
    n, alpha, trials = 49, 0.1, 6000
    passed = 0
    for _ in range(trials):
        neg = [rng.random() for _ in range(n)]
        tau = cal.conformal_threshold(neg, alpha)
        passed += rng.random() > tau
    rate = passed / trials
    # exact expectation is floor(alpha*(n+1))/(n+1) = 5/50 = 0.1; allow 4 standard errors
    assert rate <= alpha + 4 * math.sqrt(alpha * (1 - alpha) / trials)
    assert rate >= 0.07  # and it is not trivially conservative (would hide a bug the other way)


def test_pac_threshold_matches_brute_force_and_holds_by_simulation():
    rng = random.Random(7)
    for _ in range(60):
        n = rng.randint(1, 70)
        alpha = rng.choice([0.05, 0.1, 0.2, 0.3])
        delta = rng.choice([0.05, 0.1, 0.25])
        neg = [rng.random() for _ in range(n)]
        s = sorted(neg, reverse=True)
        best = -1
        for j in range(n):
            if cal.binom_cdf(j, n, alpha) <= delta:
                best = j
        expect = math.inf if best < 0 else s[best]
        assert cal.pac_threshold(neg, alpha, delta) == expect
    # high-probability claim: P(true FAR > alpha) <= delta  (true FAR of uniform = 1 - tau)
    bad, trials, n, alpha, delta = 0, 3000, 59, 0.1, 0.1
    for _ in range(trials):
        tau = cal.pac_threshold([rng.random() for _ in range(n)], alpha, delta)
        if (1.0 - tau) > alpha:
            bad += 1
    assert bad / trials <= delta + 4 * math.sqrt(delta * (1 - delta) / trials)


def test_too_few_negatives_never_pass():
    assert cal.conformal_threshold([0.3] * 5, 0.05) == math.inf
    assert cal.pac_threshold([0.3] * 10, 0.05, 0.05) == math.inf


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, "0.5", None])
def test_calibration_rejects_non_finite_or_non_numeric(bad):
    with pytest.raises(ValueError):
        cal.conformal_threshold([0.1, 0.2, bad], 0.5)


def test_wilson_interval_properties():
    rng = random.Random(3)
    for _ in range(300):
        n = rng.randint(1, 200)
        k = rng.randint(0, n)
        lo, hi = cal.wilson_interval(k, n)
        assert 0.0 <= lo <= k / n <= hi <= 1.0
    assert cal.wilson_interval(0, 0) == (0.0, 1.0)


# --- BM25: differential test against a naive reference -------------------------------------

_WORDS = ["alpha", "beta", "gamma", "Café", "日本語", "x1", "the", "server", "servers", "restart"]


def _reference_search(chunks, query, stem, k1=1.5, b=0.75):
    docs = [tokenize(c.text, stem) for c in chunks]
    n = len(docs)
    avg = sum(map(len, docs)) / n if n else 0.0
    df: Counter = Counter()
    for d in docs:
        df.update(set(d))
    out = []
    for i, d in enumerate(docs):
        tf = Counter(d)
        score = 0.0
        for t in tokenize(query, stem):
            f = tf.get(t, 0)
            if not f or not df.get(t):
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            norm = 1 - b + b * (len(d) / avg) if avg else 1.0
            score += idf * (f * (k1 + 1)) / (f + k1 * norm)
        if score > 0:
            out.append((chunks[i].id, score))
    out.sort(key=lambda x: -x[1])  # stable: ties keep insertion order
    return out


def _same(got, exp):
    return len(got) == len(exp) and all(
        a[0] == b[0] and abs(a[1] - b[1]) < 1e-9 for a, b in zip(got, exp, strict=True)
    )


@pytest.mark.parametrize("seed", range(30))
def test_bm25_equals_naive_reference_before_and_after_removal(seed):
    rng = random.Random(seed)
    stem = seed % 2 == 0
    texts = [
        " ".join(rng.choice(_WORDS) for _ in range(rng.randint(0, 12)))
        for _ in range(rng.randint(1, 15))
    ]
    chunks = [Chunk(id=f"d{i}#c0", doc_id=f"d{i}", text=t, position=0) for i, t in enumerate(texts)]
    ix = BM25Index(stem=stem)
    ix.add(chunks)
    q = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(1, 4)))
    got = [(c.id, s) for c, s in ix.search(q, k=100)]
    assert _same(got, _reference_search(chunks, q, stem))
    gone = {f"d{i}" for i in rng.sample(range(len(chunks)), rng.randint(0, len(chunks)))}
    ix.remove_docs(gone)
    rest = [c for c in chunks if c.doc_id not in gone]
    got = [(c.id, s) for c, s in ix.search(q, k=100)]
    assert _same(got, _reference_search(rest, q, stem) if rest else [])


# --- chunking ------------------------------------------------------------------------------


def test_hard_split_keeps_words_whole():
    """A long run with no sentence punctuation (log line, CSV row) used to be cut mid-word."""
    text = " ".join(["configuration"] * 60)
    chunks = chunk_text(text, "d", size=100, overlap=10)
    words = {w for c in chunks for w in c.text.split()}
    assert words == {"configuration"}
    assert max(len(c.text) for c in chunks) <= 100
    assert len(chunks) > 1
    # a query for the word must find it in every chunk
    ix = BM25Index()
    ix.add(chunks)
    assert len(ix.search("configuration", k=100)) == len(chunks)


def test_text_before_a_long_sentence_is_emitted_once():
    """The buffered short sentence used to be re-emitted before every piece of the long one."""
    text = "Short intro here. " + " ".join(f"word{i}" for i in range(40)) + ". Tail sentence."
    chunks = chunk_text(text, "d", size=80, overlap=10)
    assert sum(c.text.startswith("Short intro here.") for c in chunks) == 1
    assert len({c.text for c in chunks}) == len(chunks)
    assert {f"word{i}" for i in range(40)} <= {w.strip(".") for c in chunks for w in c.text.split()}
    assert chunks[-1].text.endswith("Tail sentence.")


def test_hard_split_overlaps_by_about_the_requested_amount():
    words = [f"w{i:03d}" for i in range(120)]  # 4-char words, one unpunctuated run
    text = " ".join(words)
    with_overlap = chunk_text(text, "d", size=60, overlap=20)
    no_overlap = chunk_text(text, "d", size=60, overlap=0)
    for a, b in zip(with_overlap, with_overlap[1:], strict=False):
        shared = set(a.text.split()) & set(b.text.split())
        assert 2 <= len(shared) <= 5, (a.text, b.text)  # 20 chars of 5-char slots, word aligned
    for a, b in zip(no_overlap, no_overlap[1:], strict=False):
        assert not set(a.text.split()) & set(b.text.split())


def test_giant_token_still_falls_back_to_a_character_cut():
    chunks = chunk_text("x" * 1000, "d", size=100, overlap=10)
    assert max(len(c.text) for c in chunks) == 100
    assert "".join(c.text[:90] for c in chunks[:-1]) in "x" * 1000


@pytest.mark.parametrize("seed", range(60))
def test_chunking_invariants_on_random_text(seed):
    rng = random.Random(seed)
    alphabet = "ab cd\n\t. ,é日"
    text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 400)))
    size = rng.randint(1, 60)
    overlap = rng.randint(0, size - 1)
    chunks = chunk_text(text, "d", size=size, overlap=overlap)
    assert chunks == chunk_text(text, "d", size=size, overlap=overlap)  # deterministic
    assert [c.position for c in chunks] == list(range(len(chunks)))
    assert all(c.text and c.text == c.text.strip() for c in chunks)
    # no content is invented: every character of every chunk comes from the input
    assert set("".join(c.text for c in chunks)) <= set(text) | {" "}
    # every whole word that fits in a chunk survives intact in some chunk
    for w in set(text.split()):
        if len(w) <= size // 2:
            assert any(w in c.text for c in chunks), (w, size, overlap)


# --- ScoreGate -----------------------------------------------------------------------------


def _data(n=120, seed=1, flip=False):
    rng = random.Random(seed)
    lists, labels = [], []
    for i in range(n):
        ans = i % 2 == 1  # labels alternate with row order: the old even/odd split broke on this
        top = rng.random() + (1.0 if ans else 0.0)
        lists.append([top, rng.random(), rng.random(), rng.random()])
        labels.append((not ans) if flip else ans)
    return lists, labels


@pytest.mark.parametrize("flip", [False, True])
def test_scoregate_fit_works_when_labels_alternate_with_row_order(flip):
    lists, labels = _data(flip=flip)
    gate = ScoreGate.fit(lists, labels, alpha=0.1)
    assert gate.n_calibration >= 20
    assert gate.tau is not None


def test_scoregate_selection_and_calibration_halves_are_disjoint_and_even():
    lists, labels = _data(n=200, seed=2)
    gate = ScoreGate.fit(lists, labels, alpha=0.1)  # statistic="auto": half selects, half calibrates
    assert gate.n_calibration == labels.count(False) // 2  # exactly half the negatives
    assert ScoreGate.fit(lists, labels, alpha=0.1, statistic="top1").n_calibration == 100


def test_scoregate_split_is_deterministic_and_does_not_peek():
    lists, labels = _data(seed=5)
    a = ScoreGate.fit(lists, labels, alpha=0.1)
    b = ScoreGate.fit(lists, labels, alpha=0.1)
    assert a.to_json() == b.to_json()


def test_scoregate_round_trip_and_tamper_detection():
    lists, labels = _data(seed=9)
    gate = ScoreGate.fit(lists, labels, alpha=0.1, statistic="top1")
    back = ScoreGate.from_json(gate.to_json())
    assert back.tau == gate.tau and back.neg_stats == gate.neg_stats
    d = json.loads(gate.to_json())
    d["tau"] = -1e9  # a hand-edited file that would pass everything
    with pytest.raises(ValueError, match="threshold does not match"):
        ScoreGate.from_json(json.dumps(d))
    d = json.loads(gate.to_json())
    d["n_calibration"] += 1
    with pytest.raises(ValueError):
        ScoreGate.from_json(json.dumps(d))
    d = json.loads(gate.to_json())
    d["delta"] = 7
    with pytest.raises(ValueError):
        ScoreGate.from_json(json.dumps(d))


def test_scoregate_unachievable_gate_survives_a_round_trip_and_never_passes():
    gate = ScoreGate.fit([[0.9, 0.1], [0.2, 0.1]], [True, False], alpha=0.05, statistic="top1")
    assert gate.tau is None
    back = ScoreGate.from_json(gate.to_json())
    assert back.decide([100.0, 1.0]).passed is False


# --- retrieval metrics ---------------------------------------------------------------------


def test_ndcg_precision_map_known_values():
    rel = {"a", "b"}
    ranked = ["x#c0", "a#c0", "y#c0", "b#c0"]
    assert ndcg_at_k(ranked, rel, 1) == 0.0
    expect = (1 / math.log2(3) + 1 / math.log2(5)) / (1 / math.log2(2) + 1 / math.log2(3))
    assert ndcg_at_k(ranked, rel, 4) == pytest.approx(expect)
    assert precision_at_k(ranked, rel, 4) == 0.5
    assert precision_at_k(ranked, rel, 8) == 0.25  # k is the divisor
    assert average_precision(ranked, rel) == pytest.approx((1 / 2 + 2 / 4) / 2)
    assert ndcg_at_k([], rel, 5) == 0.0 and average_precision([], rel) == 0.0
    assert ndcg_at_k(ranked, set(), 5) == 0.0


def test_an_item_listed_by_chunk_id_and_doc_id_is_credited_once():
    ranked = ["a#c0"]
    rel = {"a#c0", "a"}
    assert ndcg_at_k(ranked, rel, 1) == pytest.approx(1.0)
    assert average_precision(ranked, rel) <= 0.5 + 1e-12


def test_many_chunks_of_one_relevant_doc_cannot_push_ndcg_or_map_above_one():
    ranked = [f"a#c{i}" for i in range(5)] + ["b#c0"]
    for k in (1, 3, 6):
        assert ndcg_at_k(ranked, {"a", "b"}, k) <= 1.0 + 1e-12
    assert average_precision(ranked, {"a", "b"}) <= 1.0 + 1e-12


@pytest.mark.parametrize("seed", range(30))
def test_ranking_metrics_stay_in_unit_interval(seed):
    rng = random.Random(seed)
    docs = [f"d{i}" for i in range(6)]
    ranked = [f"{rng.choice(docs)}#c{rng.randint(0, 2)}" for _ in range(rng.randint(0, 12))]
    rel = set(rng.sample(docs, rng.randint(1, 4)))
    for k in (1, 3, 5, 10):
        assert 0.0 <= ndcg_at_k(ranked, rel, k) <= 1.0 + 1e-12
        assert 0.0 <= precision_at_k(ranked, rel, k) <= 1.0
    assert 0.0 <= average_precision(ranked, rel) <= 1.0 + 1e-12
    perfect = [f"{d}#c0" for d in sorted(rel)]
    assert ndcg_at_k(perfect, rel, 10) == pytest.approx(1.0)


def test_eval_report_has_false_answer_rate_with_interval():
    items = [EvalItem("q1", ["a"], True), EvalItem("q2", [], False), EvalItem("q3", [], False)]
    results = [
        {"verdict": "answerable", "retrieved_ids": ["a#c0"]},
        {"verdict": "answerable", "retrieved_ids": ["z#c0"]},  # false answer
        {"verdict": "insufficient", "retrieved_ids": []},
    ]
    rep = aggregate(items, results).to_dict()
    assert rep["false_answer_rate"] == 0.5
    lo, hi = rep["false_answer_rate_ci95"]
    assert lo < 0.5 < hi
    assert rep["ndcg_at_k"]["1"] == 1.0 and rep["map"] == 1.0
    assert aggregate([], []).to_dict()["false_answer_rate_ci95"] == [0.0, 1.0]


# --- upgrade path: a store written by the registry version (0.2.0) -----------------------------


def test_store_written_by_0_2_0_still_loads_queries_and_accepts_new_documents(tmp_path, capsys):
    src = HERE / "fixtures" / "store_v0_2_0"
    store = tmp_path / "old store"
    shutil.copytree(src, store)
    rc = main(["query", "--store", str(store), "--no-vectors", "--json", "how to restart server"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["verdict"] == "answerable"
    assert out["evidence"][0]["doc_id"] == "d1"
    rc = main(["info", "--store", str(store)])
    info = json.loads(capsys.readouterr().out)
    assert rc == 0 and info["documents"] == 3
    extra = tmp_path / "more.jsonl"
    extra.write_text('{"id":"d4","text":"Passwords reset from the account page."}\n')
    assert main(["ingest", "--input", str(extra), "--store", str(store), "--no-vectors"]) == 0
    capsys.readouterr()
    rc = main(["query", "--store", str(store), "--no-vectors", "--json", "reset my password"])
    assert json.loads(capsys.readouterr().out)["evidence"][0]["doc_id"] == "d4"


# --- golden files: output formats must not drift silently -----------------------------------------


def _golden_pipeline_files(tmp_path):
    corpus = tmp_path / "c.jsonl"
    qrels = tmp_path / "q.jsonl"
    docs = [
        ("cat", "Cats are small carnivorous mammals kept as pets. Cats purr when content."),
        ("dog", "Dogs are loyal domesticated mammals. Dogs bark and enjoy walks."),
        ("tea", "Green tea is brewed from unoxidised leaves at low temperature."),
        ("sql", "A SQL index speeds up lookups on a table column at the cost of writes."),
    ]
    corpus.write_text("".join(json.dumps({"id": i, "text": t}) + "\n" for i, t in docs))
    rows = [
        ("why do cats purr", ["cat"], True),
        ("how to brew green tea", ["tea"], True),
        ("what does a sql index do", ["sql"], True),
        ("quantum chromodynamics lattice gauge", [], False),
        ("medieval siege engines catapult", [], False),
    ]
    qrels.write_text(
        "".join(
            json.dumps({"query": q, "relevant_ids": r, "answerable": a}) + "\n" for q, r, a in rows
        )
    )
    return corpus, qrels


def _golden(name):
    return json.loads((HERE / "golden" / name).read_text("utf-8"))


def _strip_timing(obj):
    if isinstance(obj, dict):
        return {k: _strip_timing(v) for k, v in obj.items() if k not in ("ms", "total_ms")}
    if isinstance(obj, list):
        return [_strip_timing(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 6)  # last-digit libm differences must not fail a golden file
    return obj


def test_golden_eval_summary(tmp_path, capsys):
    corpus, qrels = _golden_pipeline_files(tmp_path)
    store = tmp_path / "s"
    assert main(["ingest", "--input", str(corpus), "--store", str(store), "--no-vectors"]) == 0
    capsys.readouterr()
    rc = main(["eval", "--store", str(store), "--qrels", str(qrels), "--summary", "--no-vectors"])
    got = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert got == _golden("eval_summary.json")


def test_golden_query_json_shape(tmp_path, capsys):
    corpus, _ = _golden_pipeline_files(tmp_path)
    store = tmp_path / "s"
    main(["ingest", "--input", str(corpus), "--store", str(store), "--no-vectors"])
    capsys.readouterr()
    main(["query", "--store", str(store), "--no-vectors", "--json", "why do cats purr"])
    got = _strip_timing(json.loads(capsys.readouterr().out))
    got["trace"] = sorted(got["trace"])  # timings vary; the key set must not
    got["attempts"] = [sorted(a) for a in got["attempts"]]
    assert got == _golden("query_shape.json")


def test_golden_conformal_gate_file_keys(tmp_path, capsys):
    corpus, qrels = _golden_pipeline_files(tmp_path)
    store = tmp_path / "s"
    main(["ingest", "--input", str(corpus), "--store", str(store), "--no-vectors"])
    out = tmp_path / "gate.json"
    main(
        [
            "conformal",
            "--store",
            str(store),
            "--qrels",
            str(qrels),
            "--no-vectors",
            "--alpha",
            "0.5",
            "--out",
            str(out),
        ]
    )
    capsys.readouterr()
    data = json.loads(out.read_text("utf-8"))
    assert sorted(data) == _golden("gate_file_keys.json")["top"]
    assert sorted(data["calibration"]) == _golden("gate_file_keys.json")["calibration"]
