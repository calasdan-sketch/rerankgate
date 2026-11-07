"""Round 4: what a stranger who pip-installs gets (packaging, Windows-ish inputs, consoles)."""

import io
import json
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest
import tomllib

import rerankgate
from rerankgate import PipelineConfig, RAGPipeline
from rerankgate import cli as cli_mod
from rerankgate import store as store_mod
from rerankgate import tokenize as tok
from rerankgate.cli import main

try:  # tomllib is Python 3.11+; requires-python is 3.10
    import tomllib
except ImportError:  # pragma: no cover
    tomllib = None

ROOT = Path(__file__).resolve().parent.parent
HAVE_CHECKOUT = (ROOT / "pyproject.toml").exists() and (ROOT / "CHANGELOG.md").exists()
needs_checkout = pytest.mark.skipif(not HAVE_CHECKOUT, reason="not running from a source checkout")


@needs_checkout
@pytest.mark.skipif(tomllib is None, reason="needs tomllib (Python 3.11+)")
def test_version_is_consistent_everywhere():
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert meta["project"]["version"] == rerankgate.__version__
    top = re.search(r"^## (\d+\.\d+\.\d+)", (ROOT / "CHANGELOG.md").read_text("utf-8"), re.M)
    assert top and top.group(1) == rerankgate.__version__


@needs_checkout
@pytest.mark.skipif(tomllib is None, reason="needs tomllib (Python 3.11+)")
def test_packaging_files_present_and_declared():
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert (ROOT / "LICENSE").read_text("utf-8").startswith("MIT License")
    assert "Calas Automations Inc." in (ROOT / "LICENSE").read_text("utf-8")
    assert (ROOT / "src" / "rerankgate" / "py.typed").exists()
    assert meta["license"] == "MIT" and meta["license-files"] == ["LICENSE"]
    assert meta["authors"][0]["name"] == "Calas Automations Inc."
    assert meta["urls"]["Repository"] == "https://github.com/calasdan-sketch/rerankgate"
    assert meta["dependencies"] == []  # zero hard dependencies is a headline claim
    assert "Typing :: Typed" in meta["classifiers"]
    assert "Programming Language :: Python :: 3.10" in meta["classifiers"]  # = requires-python


def test_combining_mark_regex_matches_a_full_unicode_scan():
    ranges, start = [], None
    for cp in range(sys.maxunicode + 2):
        mark = cp <= sys.maxunicode and unicodedata.category(chr(cp)).startswith("M")
        if mark and start is None:
            start = cp
        elif not mark and start is not None:
            ranges.append(f"{tok._esc(start)}-{tok._esc(cp - 1)}")
            start = None
    expected = r"(?:[^\W_]|[" + "".join(ranges) + "])+"
    assert tok._token_re().pattern == expected


def _corpus(tmp_path, raw: bytes) -> str:
    p = tmp_path / "corpus.jsonl"
    p.write_bytes(raw)
    return str(p)


def test_ingest_accepts_utf8_bom_and_crlf(tmp_path, capsys):
    raw = b'\xef\xbb\xbf{"id":"a","text":"hello world"}\r\n{"id":"b","text":"crlf line"}\r\n'
    rc = main(
        [
            "ingest",
            "--input",
            _corpus(tmp_path, raw),
            "--store",
            str(tmp_path / "s"),
            "--no-vectors",
        ]
    )
    assert rc == 0
    assert "ingested 2 documents" in capsys.readouterr().out


def test_ingest_from_stdin(tmp_path, monkeypatch, capsys):
    class FakeStdin(io.TextIOWrapper):
        def fileno(self):  # route to a real file
            return fd

    p = tmp_path / "in.jsonl"
    p.write_bytes(b'{"id":"a","text":"from standard input"}\n')
    fd = os.open(p, os.O_RDONLY)
    try:
        monkeypatch.setattr(sys, "stdin", FakeStdin(io.BytesIO(b"")))
        rc = main(["ingest", "--input", "-", "--store", str(tmp_path / "s"), "--no-vectors"])
    finally:
        os.close(fd)
    assert rc == 0
    assert "ingested 1 documents" in capsys.readouterr().out


def test_utf16_corpus_gets_a_plain_error(tmp_path, capsys):
    raw = '{"id":"a","text":"hi"}\n'.encode("utf-16")
    rc = main(
        [
            "ingest",
            "--input",
            _corpus(tmp_path, raw),
            "--store",
            str(tmp_path / "s"),
            "--no-vectors",
        ]
    )
    assert rc == 2
    assert "UTF-16" in capsys.readouterr().err


def test_lone_surrogate_is_a_clean_error_not_a_traceback(tmp_path, capsys):
    raw = b'{"id":"s","text":"lone \\ud800 surrogate"}\n'
    rc = main(
        [
            "ingest",
            "--input",
            _corpus(tmp_path, raw),
            "--store",
            str(tmp_path / "s"),
            "--no-vectors",
        ]
    )
    err = capsys.readouterr().err
    assert rc == 2 and "unpaired surrogate" in err and "line 1" in err
    assert not (tmp_path / "s").exists()


def test_library_save_rejects_surrogates_with_store_error(tmp_path):
    from rerankgate import Document

    pipe = RAGPipeline(config=PipelineConfig(use_vectors=False))
    pipe.ingest([Document(id="s", text="bad \ud800 text")])
    with pytest.raises(store_mod.StoreError, match="surrogate"):
        pipe.save(str(tmp_path / "s"))


def test_empty_query_is_an_input_error(tmp_path, capsys):
    p = _corpus(tmp_path, b'{"id":"a","text":"hello world"}\n')
    main(["ingest", "--input", p, "--store", str(tmp_path / "s"), "--no-vectors"])
    capsys.readouterr()
    assert main(["query", "--store", str(tmp_path / "s"), "--no-vectors", "   "]) == 2
    assert "query is empty" in capsys.readouterr().err


def test_ask_rejects_non_string_query():
    with pytest.raises(TypeError, match="query must be a string"):
        RAGPipeline(config=PipelineConfig(use_vectors=False)).ask(None)  # type: ignore[arg-type]


def test_cp1252_console_does_not_crash(tmp_path):
    """A Windows-style console cannot encode Greek/CJK: output must be escaped, not a traceback."""
    corpus = tmp_path / "u.jsonl"
    corpus.write_text('{"id":"u","text":"Greek ΚΑΛΑΣ and 日本語"}\n', encoding="utf-8")
    env = {**os.environ, "PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"}
    base = [sys.executable, "-m", "rerankgate"]
    store = str(tmp_path / "s")
    r = subprocess.run(
        [*base, "ingest", "--input", str(corpus), "--store", store, "--no-vectors"],
        capture_output=True,
        env=env,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr
    r = subprocess.run(
        [*base, "query", "--store", store, "--no-vectors", "ΚΑΛΑΣ"],
        capture_output=True,
        env=env,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr
    assert b"Traceback" not in r.stderr and b"\\u" in r.stdout


def test_broken_pipe_exits_quietly(monkeypatch, tmp_path):
    def boom(_args):
        raise BrokenPipeError

    monkeypatch.setattr(cli_mod, "cmd_query", boom)
    parser = cli_mod.build_parser()
    parser.set_defaults()
    monkeypatch.setattr(cli_mod, "build_parser", lambda: _with_func(parser, boom))
    assert main(["query", "--store", "x", "q"]) == cli_mod.EXIT_BROKEN_PIPE


def _with_func(parser, fn):
    orig = parser.parse_args

    def parse(argv=None):
        ns = orig(argv)
        ns.func = fn
        return ns

    parser.parse_args = parse  # type: ignore[method-assign]
    return parser


def test_keyboard_interrupt_exits_130_without_traceback(monkeypatch, capsys):
    parser = cli_mod.build_parser()

    def interrupt(_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_mod, "build_parser", lambda: _with_func(parser, interrupt))
    assert main(["query", "--store", "x", "q"]) == 130
    assert "interrupted" in capsys.readouterr().err


def test_conformal_out_is_written_whole_and_survives_a_failed_run(tmp_path, capsys):
    items = []
    for i in range(60):
        items.append({"query": f"zzunknown{i} qqq", "relevant_ids": [], "answerable": False})
    items.append({"query": "hello world", "relevant_ids": ["a"], "answerable": True})
    qrels = tmp_path / "q.jsonl"
    qrels.write_text("\n".join(json.dumps(i) for i in items), encoding="utf-8")
    corpus = _corpus(tmp_path, b'{"id":"a","text":"hello world"}\n')
    store = str(tmp_path / "s")
    main(["ingest", "--input", corpus, "--store", store, "--no-vectors"])
    out = tmp_path / "gate.json"
    out.write_text("previous good gate", encoding="utf-8")
    rc = main(
        ["conformal", "--store", store, "--no-vectors", "--qrels", str(qrels), "--out", str(out)]
    )
    capsys.readouterr()
    assert rc in (0, 1)
    data = json.loads(out.read_text("utf-8"))
    assert "threshold" in data and "fingerprint" in data["calibration"]
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".tmp-")]


def test_replace_retries_a_locked_file_on_windows_only(monkeypatch, tmp_path):
    calls = {"n": 0}
    real = os.replace

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("locked")
        return real(src, dst)

    monkeypatch.setattr(store_mod.os, "replace", flaky)
    monkeypatch.setattr(store_mod.time, "sleep", lambda s: None)
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("x")
    monkeypatch.setattr(store_mod.os, "name", "nt")
    store_mod._replace(str(a), str(b))
    assert b.read_text() == "x" and calls["n"] == 3

    calls["n"] = 0
    monkeypatch.setattr(store_mod.os, "name", "posix")  # a real permission error must surface
    a.write_text("y")
    with pytest.raises(PermissionError):
        store_mod._replace(str(a), str(b))
    assert calls["n"] == 1


def test_gate_file_with_bom_loads(tmp_path):
    p = tmp_path / "g.json"
    p.write_bytes(b"\xef\xbb\xbf" + json.dumps({"threshold": 0.6}).encode())
    assert cli_mod._gate_file(str(p))[0] == 0.6


def test_ingest_json_info_and_query_notice(tmp_path, capsys):
    corpus = _corpus(
        tmp_path,
        b'{"id":"a","text":"Retrieval-augmented generation pairs a retriever with a model."}\n',
    )
    store = str(tmp_path / "s")
    assert main(["ingest", "--input", corpus, "--store", store, "--no-vectors", "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["documents"] == 1 and summary["chunks"] == 1 and summary["bm25_only"] is True

    assert main(["info", "--store", store, "--no-vectors"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["documents"] == 1 and info["has_vectors"] is False
    assert info["rerankgate_version"] == rerankgate.__version__
    assert len(info["corpus_digest"]) == 64

    assert main(["query", "--store", store, "--no-vectors", "--json", "xylophone tickets"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["verdict"] == "insufficient" and out["notice"]["kind"]
    assert main(["query", "--store", store, "--no-vectors", "--json", "retrieval generation"]) == 0
    assert json.loads(capsys.readouterr().out)["notice"] is None


def test_info_on_missing_store_is_a_plain_error(tmp_path, capsys):
    assert main(["info", "--store", str(tmp_path / "nope")]) == 2
    assert "no store" in capsys.readouterr().err
