"""Tests for the 0.3.0 round: correctness fixes, zero-dep persistence,
inverted-index BM25, calibration, CLI hardening, Unicode tokenization."""

from __future__ import annotations

import json
import os
import random

import pytest

from rerankgate.bm25 import BM25Index
from rerankgate.cli import main
from rerankgate.config import GateConfig, PipelineConfig
from rerankgate.documents import Chunk, Document
from rerankgate.embeddings import FakeEmbeddings
from rerankgate.judge import HeuristicJudge, LLMJudge
from rerankgate.metrics import EvalItem
from rerankgate.pipeline import ANSWERABLE, RAGPipeline
from rerankgate.safeterm import clean
from rerankgate.store import DocumentStore, StoreError
from rerankgate.tokenize import tokenize

# --------------------------------------------------------------- config / gate


def test_custom_threshold_not_overwritten_by_default_preset():
    cfg = PipelineConfig(use_vectors=False, gate=GateConfig(threshold=0.9))
    assert cfg.effective_gate().threshold == 0.9
    assert RAGPipeline(config=cfg).gate.config.threshold == 0.9


def test_explicit_preset_still_applies():
    cfg = PipelineConfig(use_vectors=False, gate_preset="strict")
    assert cfg.effective_gate().threshold == 0.70


def test_default_gate_equals_balanced():
    a = PipelineConfig().effective_gate()
    b = GateConfig().preset("balanced")
    assert (a.threshold, a.min_coverage) == (b.threshold, b.min_coverage)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": 0.0},
        {"threshold": float("nan")},
        {"min_coverage": -0.1},
        {"w_top_score": 0, "w_coverage": 0, "w_margin": 0, "w_support": 0},
    ],
)
def test_bad_gate_config_fails_closed(kwargs):
    with pytest.raises(ValueError):
        RAGPipeline(config=PipelineConfig(use_vectors=False, gate=GateConfig(**kwargs)))


# ---------------------------------------------------------------------- judge


def test_judge_accepts_verbatim_sentence_from_long_chunk():
    filler = " ".join(f"Sentence {i} covers topic{i} and detail{i} at length." for i in range(25))
    pipe = RAGPipeline(config=PipelineConfig(use_vectors=False, chunk_size=5000))
    pipe.ingest([Document(id="d", text=filler + " The Eiffel Tower is in Paris, France.")])
    res = pipe.ask(
        "Where is the Eiffel Tower?",
        generator=lambda q, ev: "The Eiffel Tower is in Paris, France.",
    )
    assert res.verdict == ANSWERABLE
    assert res.faithfulness.unsupported == []


def test_judge_still_flags_fabrication(bm25_pipeline):
    ev = bm25_pipeline._retriever().retrieve("rust borrow checker", k=3).chunks
    rep = HeuristicJudge().check("Penguins invented quantum basketball in Antarctica.", ev)
    assert rep.unsupported


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "not a number", None])
def test_llm_judge_bad_output_fails_closed(bad):
    assert LLMJudge(lambda a, e: bad).check("x", []).score == 0.0


def test_faithfulness_report_default_is_list():
    from rerankgate.judge import FaithfulnessReport

    assert FaithfulnessReport(score=0.0).per_sentence == []


# ------------------------------------------------------------------ ingestion


def test_duplicate_doc_ids_rejected(bm25_pipeline):
    with pytest.raises(ValueError, match="duplicate"):
        bm25_pipeline.ingest([Document(id="doc-rust", text="again")])
    pipe = RAGPipeline(config=PipelineConfig(use_vectors=False))
    with pytest.raises(ValueError, match="duplicate"):
        pipe.ingest([Document(id="a", text="x"), Document(id="a", text="y")])
    assert pipe.chunks == []  # nothing half-ingested


@pytest.mark.parametrize("bad_id", ["", "a#c1"])
def test_bad_doc_ids_rejected(bad_id):
    pipe = RAGPipeline(config=PipelineConfig(use_vectors=False))
    with pytest.raises(ValueError):
        pipe.ingest([Document(id=bad_id, text="hello world")])


# ---------------------------------------------------------------- persistence


def test_vectors_persist_without_numpy(hybrid_pipeline, tmp_path):
    store = tmp_path / "s"
    hybrid_pipeline.save(str(store))
    assert (store / "vectors.f32").exists()
    assert not (store / "vectors.npy").exists()
    loaded = RAGPipeline.load(str(store), embeddings=FakeEmbeddings(dim=16))
    assert not loaded.reembedded_on_load
    a = hybrid_pipeline.vector_store.vectors
    b = loaded.vector_store.vectors
    assert len(a) == len(b)
    assert all(
        abs(x - y) < 1e-6 for va, vb in zip(a, b, strict=True) for x, y in zip(va, vb, strict=False)
    )


class _OtherModel(FakeEmbeddings):
    @property
    def name(self) -> str:
        return "some-other-model"


def test_same_dim_different_model_is_reembedded(hybrid_pipeline, tmp_path):
    store = str(tmp_path / "s")
    hybrid_pipeline.save(store)
    loaded = RAGPipeline.load(store, embeddings=_OtherModel(dim=16))
    assert loaded.reembedded_on_load


def test_bm25_only_store_gets_vectors_when_backend_present(bm25_pipeline, tmp_path):
    store = str(tmp_path / "s")
    bm25_pipeline.save(store)
    loaded = RAGPipeline.load(store, embeddings=FakeEmbeddings(dim=16))
    assert len(loaded.vector_store) == len(loaded.chunks)


def test_tampered_chunks_fail_closed(bm25_pipeline, tmp_path):
    store = tmp_path / "s"
    bm25_pipeline.save(str(store))
    f = store / "chunks.jsonl"
    f.write_text(f.read_text().replace("Python", "Pyth0n"))
    with pytest.raises(StoreError, match="checksum"):
        RAGPipeline.load(str(store), config=PipelineConfig(use_vectors=False))


def test_truncated_vectors_fail_closed(hybrid_pipeline, tmp_path):
    store = tmp_path / "s"
    hybrid_pipeline.save(str(store))
    v = store / "vectors.f32"
    v.write_bytes(v.read_bytes()[:-4])
    with pytest.raises(StoreError, match="bytes"):
        RAGPipeline.load(str(store), embeddings=FakeEmbeddings(dim=16))


def test_missing_vector_file_fails_closed(hybrid_pipeline, tmp_path):
    store = tmp_path / "s"
    hybrid_pipeline.save(str(store))
    os.unlink(store / "vectors.f32")
    with pytest.raises(StoreError):
        RAGPipeline.load(str(store), embeddings=FakeEmbeddings(dim=16))


@pytest.mark.parametrize(
    "line",
    ["not json", "[1,2]", '{"id": "x"}', '{"id": 1, "doc_id": "d", "text": "t", "position": 0}'],
)
def test_malformed_chunk_lines_fail_closed(tmp_path, line):
    (tmp_path / "chunks.jsonl").write_text(line + "\n")
    (tmp_path / "meta.json").write_text('{"chunk_count": 1}')  # legacy 0.2 meta
    with pytest.raises(StoreError, match="line 1"):
        DocumentStore(str(tmp_path)).load()


def test_unknown_chunk_fields_ignored(tmp_path):
    row = {"id": "d#c0", "doc_id": "d", "text": "t", "position": 0, "__class__": "evil"}
    (tmp_path / "chunks.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "meta.json").write_text('{"chunk_count": 1}')  # legacy 0.2 meta
    chunks, _, _ = DocumentStore(str(tmp_path)).load()
    assert chunks[0].id == "d#c0"


def test_resave_without_vectors_removes_stale_file(hybrid_pipeline, bm25_pipeline, tmp_path):
    store = tmp_path / "s"
    hybrid_pipeline.save(str(store))
    bm25_pipeline.save(str(store))
    assert not (store / "vectors.f32").exists()
    assert not [p for p in os.listdir(store) if p.startswith(".tmp-")]


def test_nan_embeddings_rejected():
    class NaNEmb(FakeEmbeddings):
        def embed(self, texts):
            return [[float("nan")] * self.dim for _ in texts]

    pipe = RAGPipeline(config=PipelineConfig(), embeddings=NaNEmb(dim=4))
    with pytest.raises(ValueError, match="NaN"):
        pipe.ingest([Document(id="d", text="hello")])


# ----------------------------------------------------------------------- BM25


def _naive_bm25(index: BM25Index, query: str) -> dict[str, float]:
    """Reference full-scan scorer (the 0.2 algorithm) for equivalence checks."""
    out = {}
    q = tokenize(query)
    for i, ch in enumerate(index._chunks):
        tf, dl = index._term_freqs[i], index._doc_lens[i]
        norm = 1 - index.b + index.b * dl / index._avg_len
        s = 0.0
        for t in q:
            f = tf.get(t, 0)
            if f:
                s += index._idf(t) * f * (index.k1 + 1) / (f + index.k1 * norm)
        if s > 0:
            out[ch.id] = s
    return out


def test_inverted_index_matches_full_scan():
    rng = random.Random(7)
    vocab = [f"w{i}" for i in range(60)]
    chunks = [
        Chunk(id=f"d{i}#c0", doc_id=f"d{i}", text=" ".join(rng.choices(vocab, k=30)), position=0)
        for i in range(200)
    ]
    idx = BM25Index()
    idx.add(chunks)
    for _ in range(25):
        q = " ".join(rng.choices(vocab, k=4))
        got = {c.id: s for c, s in idx.search(q, k=1000)}
        want = _naive_bm25(idx, q)
        assert got.keys() == want.keys()
        assert all(abs(got[k] - want[k]) < 1e-9 for k in want)


def test_bm25_ties_are_deterministic():
    idx = BM25Index()
    idx.add([Chunk(id=f"d{i}#c0", doc_id=f"d{i}", text="same text", position=0) for i in range(5)])
    assert [c.id for c, _ in idx.search("same", k=5)] == [f"d{i}#c0" for i in range(5)]
    assert idx.search("same", k=0) == []


# --------------------------------------------------------------- tokenization


def test_unicode_tokens():
    assert tokenize("Café CAFÉ café") == ["café", "café", "café"]
    assert tokenize("Straße") == tokenize("STRASSE")
    assert tokenize("ＡＢＣ１２３") == ["abc123"]  # full-width folds to ASCII
    assert tokenize("Привет мир") == ["привет", "мир"]


def test_cjk_split_per_character():
    assert tokenize("東京タワー") == ["東", "京", "タ", "ワ", "ー"]
    assert tokenize("RAG检索") == ["rag", "检", "索"]


def test_non_english_retrieval_now_works():
    pipe = RAGPipeline(config=PipelineConfig(use_vectors=False))
    pipe.ingest(
        [
            Document(id="fr", text="La tour Eiffel se trouve à Paris. Elle a été construite."),
            Document(id="zh", text="长城是中国古代的军事防御工程。"),
        ]
    )
    assert pipe.retrieve("Où se trouve la tour Eiffel ?").chunks[0].chunk.doc_id == "fr"
    assert pipe.retrieve("长城在哪里").chunks[0].chunk.doc_id == "zh"


# ---------------------------------------------------------------- calibration


def _labeled():
    return [
        EvalItem("What is retrieval-augmented generation?", ["doc-rag"], True),
        EvalItem("Rust borrow checker memory safety", ["doc-rust"], True),
        EvalItem("Python programming paradigms", ["doc-python"], True),
        EvalItem("xylophone concerto ticket prices", [], False),
        EvalItem("sourdough starter hydration", [], False),
    ]


def test_calibrate_sweep_and_recommendation(bm25_pipeline):
    before = bm25_pipeline.gate
    rep = bm25_pipeline.calibrate(_labeled())
    assert bm25_pipeline.gate is before  # gate restored
    assert len(rep["sweep"]) == 11
    assert rep["recommended_threshold"] in [r["threshold"] for r in rep["sweep"]]
    recalls = [r["gate_recall"] for r in rep["sweep"]]
    assert recalls == sorted(recalls, reverse=True)  # stricter never passes more


def test_calibrate_target_precision(bm25_pipeline):
    rep = bm25_pipeline.calibrate(_labeled(), target_precision=1.0)
    rec = rep["recommended_threshold"]
    if rec is not None:
        row = next(r for r in rep["sweep"] if r["threshold"] == rec)
        assert row["gate_precision"] >= 1.0


@pytest.mark.parametrize(
    "items",
    [[], [EvalItem("q", ["a"], True)], [EvalItem("q", [], False)]],
)
def test_calibrate_fails_closed_on_one_sided_labels(bm25_pipeline, items):
    with pytest.raises(ValueError):
        bm25_pipeline.calibrate(items)


# ------------------------------------------------------------------------ CLI


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def cli_store(tmp_path):
    corpus = _write(
        tmp_path / "c.jsonl",
        [
            {"id": "rag", "text": "Retrieval-augmented generation pairs a retriever with a model."},
            {"id": "bread", "text": "Sourdough bread needs a starter, flour and water."},
        ],
    )
    store = str(tmp_path / "store")
    assert main(["ingest", "--input", corpus, "--store", store, "--no-vectors"]) == 0
    return store


def test_cli_escapes_terminal_control_and_bidi(tmp_path, capsys):
    evil = "Bad \x1b[2J\x1b[31mred\x07 text ‮evil​ here."
    corpus = _write(tmp_path / "c.jsonl", [{"id": "x\x1b]0;t\x07", "text": evil}])
    store = str(tmp_path / "s")
    assert main(["ingest", "--input", corpus, "--store", store, "--no-vectors"]) == 0
    capsys.readouterr()
    main(["query", "--store", store, "--no-vectors", "bad red text"])
    out = capsys.readouterr().out
    for ch in ("\x1b", "\x07", "‮", "​"):
        assert ch not in out
    assert "\\u202e" in out and "\\u001b" in out


def test_clean_helper():
    assert clean("a\r\nb\tc") == "a  b c"
    assert clean("⁦x⁩") == "\\u2066x\\u2069"
    assert clean("ok é 東") == "ok é 東"


@pytest.mark.parametrize(
    "content,needle",
    [
        ('{"id": 1, "text": "ok"}\n{broken\n', "line 2"),
        ("", "no documents"),
        ('"just a string"\n', "line 1"),
        ('{"id": "a", "text": 5}\n', "must be a string"),
        ('{"id": "a", "text": "x"}\n{"id": "a", "text": "y"}\n', "duplicate"),
    ],
)
def test_cli_ingest_bad_input_exit_2(tmp_path, capsys, content, needle):
    f = tmp_path / "c.jsonl"
    f.write_text(content, encoding="utf-8")
    rc = main(["ingest", "--input", str(f), "--store", str(tmp_path / "s"), "--no-vectors"])
    assert rc == 2
    assert needle in capsys.readouterr().err
    assert not (tmp_path / "s" / "chunks.jsonl").exists()


def test_cli_missing_files_exit_2(tmp_path, capsys):
    assert main(["ingest", "--input", str(tmp_path / "nope"), "--store", str(tmp_path)]) == 2
    assert main(["query", "--store", str(tmp_path / "nope"), "--no-vectors", "q"]) == 2
    assert "Traceback" not in capsys.readouterr().err


def test_cli_empty_qrels_fail_closed(cli_store, tmp_path, capsys):
    q = tmp_path / "q.jsonl"
    q.write_text("\n")
    assert main(["eval", "--store", cli_store, "--qrels", str(q), "--no-vectors"]) == 2
    assert "no items" in capsys.readouterr().err


def test_cli_bad_qrels_types(cli_store, tmp_path, capsys):
    q = _write(tmp_path / "q.jsonl", [{"query": "x", "answerable": "yes"}])
    assert main(["eval", "--store", cli_store, "--qrels", q, "--no-vectors"]) == 2


def test_cli_eval_fail_under(cli_store, tmp_path, capsys):
    q = _write(
        tmp_path / "q.jsonl",
        [
            {"query": "What is retrieval-augmented generation?", "relevant_ids": ["rag"]},
            {"query": "xylophone concerto", "relevant_ids": [], "answerable": False},
        ],
    )
    args = ["eval", "--store", cli_store, "--qrels", q, "--no-vectors", "--summary"]
    assert main(args + ["--fail-under-mrr", "0.5"]) == 0
    capsys.readouterr()
    assert main(args + ["--fail-under-recall", "1.0", "--fail-under-precision", "1.0"]) in (0, 1)
    capsys.readouterr()
    # An impossible floor must fail.
    q2 = _write(tmp_path / "q2.jsonl", [{"query": "xylophone", "relevant_ids": ["rag"]}])
    assert main(["eval", "--store", cli_store, "--qrels", q2, "--no-vectors",
                 "--fail-under-recall", "1.0"]) == 1  # fmt: skip
    assert "FAILED" in capsys.readouterr().err


def test_cli_fail_on_insufficient(cli_store, capsys):
    base = ["query", "--store", cli_store, "--no-vectors", "--fail-on-insufficient"]
    assert main(base + ["What is retrieval-augmented generation?"]) == 0
    assert main(base + ["--gate-preset", "strict", "xylophone concerto tickets"]) == 1


def test_cli_calibrate(cli_store, tmp_path, capsys):
    q = _write(
        tmp_path / "q.jsonl",
        [
            {"query": "What is retrieval-augmented generation?", "relevant_ids": ["rag"]},
            {"query": "xylophone concerto tickets", "relevant_ids": [], "answerable": False},
        ],
    )
    rc = main(["calibrate", "--store", cli_store, "--qrels", q, "--no-vectors", "--json"])
    rep = json.loads(capsys.readouterr().out)
    assert rc == 0 and rep["recommended_threshold"] is not None
    rc = main(["calibrate", "--store", cli_store, "--qrels", q, "--no-vectors"])
    assert "recommended threshold" in capsys.readouterr().out


def test_cli_calibrate_one_sided_exit_2(cli_store, tmp_path, capsys):
    q = _write(tmp_path / "q.jsonl", [{"query": "What is RAG?", "relevant_ids": ["rag"]}])
    assert main(["calibrate", "--store", cli_store, "--qrels", q, "--no-vectors"]) == 2


def test_cli_negative_retries_exit_2(cli_store):
    assert main(["query", "--store", cli_store, "--no-vectors", "--max-retries", "-1", "q"]) == 2


def test_cli_fraction_validation(cli_store, tmp_path):
    with pytest.raises(SystemExit):
        main(["eval", "--store", cli_store, "--qrels", "x", "--fail-under-mrr", "2"])


def test_legacy_npy_store_without_numpy_reembeds(tmp_path):
    """A 0.2-format store (vectors.npy, no checksums) still loads."""
    row = {"id": "d#c0", "doc_id": "d", "text": "Python is a language.", "position": 0}
    (tmp_path / "chunks.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "meta.json").write_text(
        json.dumps({"chunk_count": 1, "has_vectors": True, "dim": 16, "embedding": "x"})
    )
    (tmp_path / "vectors.npy").write_bytes(b"\x93NUMPY-not-really")
    try:
        import numpy  # noqa: F401
    except ImportError:
        loaded = RAGPipeline.load(str(tmp_path), embeddings=FakeEmbeddings(dim=16))
        assert loaded.reembedded_on_load
        assert len(loaded.vector_store) == 1
    else:  # pragma: no cover - numpy env: a garbage .npy must not load silently
        with pytest.raises(Exception):  # noqa: B017
            RAGPipeline.load(str(tmp_path), embeddings=FakeEmbeddings(dim=16))


def test_cli_store_path_is_a_file_exit_2(tmp_path, capsys):
    corpus = _write(tmp_path / "c.jsonl", [{"id": "a", "text": "hello world"}])
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    assert main(["ingest", "--input", corpus, "--store", str(blocker), "--no-vectors"]) == 2


# ------------------------------------------------- panel review fixes (round 1b)


class _ConstReranker:
    name = "const"

    def __init__(self, value):
        self.value = value

    def rerank(self, query, candidates, k):
        from rerankgate.retrieval import ScoredChunk

        return [ScoredChunk(c.chunk, self.value, c.sources) for c in candidates[:k]]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_rerank_score_fails_closed(docs, bad):
    from rerankgate.gate import INVALID_SCORE

    pipe = RAGPipeline(config=PipelineConfig(use_vectors=False), reranker=_ConstReranker(bad))
    pipe.ingest(docs)
    res = pipe.ask("What is retrieval-augmented generation?")
    assert res.verdict == "insufficient"
    assert res.gate.reasons == [INVALID_SCORE]
    json.dumps(res.to_dict())


def test_gate_rejects_bool_and_nonnumeric_scores(bm25_pipeline):
    from rerankgate.gate import INVALID_SCORE, SufficiencyGate
    from rerankgate.retrieval import ScoredChunk

    ev = bm25_pipeline.retrieve("rust").chunks
    for bad in (True, "0.9", None):
        tampered = [ScoredChunk(c.chunk, bad, c.sources) for c in ev]
        assert SufficiencyGate().evaluate("rust", tampered).reasons == [INVALID_SCORE]


def _fresh_store(pipe, tmp_path):
    store = tmp_path / "s"
    pipe.save(str(store))
    return store


def test_missing_meta_json_fails_closed(bm25_pipeline, tmp_path):
    store = _fresh_store(bm25_pipeline, tmp_path)
    os.unlink(store / "meta.json")
    with open(store / "chunks.jsonl", "a") as f:
        f.write(json.dumps({"id": "x#c0", "doc_id": "x", "text": "forged", "position": 0}) + "\n")
    with pytest.raises(StoreError, match="meta.json is missing"):
        DocumentStore(str(store)).load()


@pytest.mark.parametrize("key", ["chunks_sha256", "chunk_count", "dim", "has_vectors"])
def test_format2_requires_integrity_fields(bm25_pipeline, tmp_path, key):
    store = _fresh_store(bm25_pipeline, tmp_path)
    meta = json.loads((store / "meta.json").read_text())
    del meta[key]
    (store / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(StoreError, match="missing"):
        DocumentStore(str(store)).load()


def test_format2_with_vectors_requires_vector_checksum(hybrid_pipeline, tmp_path):
    store = _fresh_store(hybrid_pipeline, tmp_path)
    meta = json.loads((store / "meta.json").read_text())
    del meta["vectors_sha256"]
    (store / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(StoreError, match="vectors_sha256"):
        DocumentStore(str(store)).load()


@pytest.mark.parametrize(
    "patch",
    [
        {"format": "2"},
        {"format": 99},
        {"chunk_count": "3"},
        {"has_vectors": "yes"},
        {"dim": -1},
        {"chunks_sha256": "abc"},
    ],
)
def test_meta_field_types_validated(bm25_pipeline, tmp_path, patch):
    store = _fresh_store(bm25_pipeline, tmp_path)
    meta = {**json.loads((store / "meta.json").read_text()), **patch}
    (store / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(StoreError):
        DocumentStore(str(store)).load()


@pytest.mark.parametrize(
    "row",
    [
        {"id": "d#c0", "doc_id": "d", "text": "t", "position": "x"},
        {"id": "d#c0", "doc_id": None, "text": "t", "position": 0},
        {"id": "d#c0", "doc_id": "d", "text": "t", "position": True},
        {"id": "d#c0", "doc_id": "d", "text": "t", "position": -1},
    ],
)
def test_chunk_field_types_validated(tmp_path, row):
    (tmp_path / "chunks.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "meta.json").write_text('{"chunk_count": 1}')
    with pytest.raises(StoreError, match="line 1"):
        DocumentStore(str(tmp_path)).load()


def test_nan_in_vector_file_is_store_error(hybrid_pipeline, tmp_path):
    import hashlib
    import struct

    store = _fresh_store(hybrid_pipeline, tmp_path)
    data = bytearray((store / "vectors.f32").read_bytes())
    data[0:4] = struct.pack("<f", float("nan"))
    (store / "vectors.f32").write_bytes(bytes(data))
    meta = json.loads((store / "meta.json").read_text())
    meta["vectors_sha256"] = hashlib.sha256(bytes(data)).hexdigest()
    (store / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(StoreError, match="NaN"):
        DocumentStore(str(store)).load()


def test_calibrate_no_recommendation_when_nothing_passes(bm25_pipeline):
    items = [
        EvalItem("zebra quantum lattice", ["doc-rag"], True),
        EvalItem("xylophone concerto tickets", [], False),
    ]
    rep = bm25_pipeline.calibrate(items)
    assert all(r["gate_recall"] == 0 for r in rep["sweep"])
    assert rep["recommended_threshold"] is None


def test_cli_calibrate_nothing_passes_exit_1(cli_store, tmp_path, capsys):
    q = _write(
        tmp_path / "q.jsonl",
        [
            {"query": "zebra quantum lattice", "relevant_ids": ["rag"]},
            {"query": "xylophone concerto", "relevant_ids": [], "answerable": False},
        ],
    )
    assert main(["calibrate", "--store", cli_store, "--qrels", q, "--no-vectors"]) == 1
    assert "none" in capsys.readouterr().out


@pytest.mark.parametrize(
    "text,expected",
    [
        ("हिन्दी भाषा", ["हिन्दी", "भाषा"]),
        ("العَرَبِيَّة لغة", ["العَرَبِيَّة", "لغة"]),
        ("שָׁלוֹם עוֹלָם", ["שָׁלוֹם", "עוֹלָם"]),
        ("𠀀𠀁x", ["𠀀", "𠀁", "x"]),  # CJK Extension B
        ("𰀀", ["𰀀"]),  # CJK Extension G
    ],
)
def test_combining_marks_and_cjk_extensions(text, expected):
    assert tokenize(text) == expected


def test_hindi_retrieval_end_to_end():
    pipe = RAGPipeline(config=PipelineConfig(use_vectors=False))
    pipe.ingest(
        [
            Document(id="hi", text="हिन्दी भारत की एक प्रमुख भाषा है।"),
            Document(id="en", text="Python is a programming language."),
        ]
    )
    assert pipe.retrieve("हिन्दी भाषा").chunks[0].chunk.doc_id == "hi"


@pytest.mark.parametrize("bad", [True, False, 7, -0.5, 1.0000001, 10**400, "7"])
def test_llm_judge_rejects_out_of_contract_values(bad):
    assert LLMJudge(lambda a, e: bad).check("x", []).score == 0.0


@pytest.mark.parametrize("good", [0, 0.0, 0.42, 1, "0.9"])
def test_llm_judge_accepts_valid_scores(good):
    assert LLMJudge(lambda a, e: good).check("x", []).score == round(float(good), 4)


def _bm25_v02_search(index: BM25Index, query: str, k: int = 100):
    """Verbatim copy of the rerankgate 0.2 full-scan BM25 search."""
    q_terms = tokenize(query)
    if not q_terms or not index._chunks:
        return []
    idfs = {t: index._idf(t) for t in set(q_terms)}
    scored = []
    for i, ch in enumerate(index._chunks):
        tf = index._term_freqs[i]
        dl = index._doc_lens[i]
        norm = 1 - index.b + index.b * (dl / index._avg_len) if index._avg_len else 1.0
        score = 0.0
        for t in q_terms:
            f = tf.get(t, 0)
            if f and idfs[t]:
                score += idfs[t] * (f * (index.k1 + 1)) / (f + index.k1 * norm)
        if score > 0:
            scored.append((ch, score))
    scored.sort(key=lambda p: p[1], reverse=True)
    return scored[:k]


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_bm25_bit_identical_to_v02_including_order(seed):
    """Differential test: same ids, same order, same float scores (==, no tolerance)."""
    rng = random.Random(seed)
    vocab = [f"w{i}" for i in range(40)]
    chunks = []
    for i in range(300):
        n = rng.randint(1, 25)
        chunks.append(Chunk(f"d{i}#c0", f"d{i}", " ".join(rng.choices(vocab, k=n)), 0))
    # Exact duplicates force ties between different documents.
    chunks += [Chunk(f"dup{i}#c0", f"dup{i}", chunks[i].text, 0) for i in range(30)]
    idx = BM25Index()
    idx.add(chunks)
    for _ in range(1500):
        q = " ".join(rng.choices(vocab, k=rng.randint(1, 6)))  # repeats common
        for k in (5, 100, 1000):
            got = [(c.id, s) for c, s in idx.search(q, k=k)]
            want = [(c.id, s) for c, s in _bm25_v02_search(idx, q, k=k)]
            assert got == want, q
