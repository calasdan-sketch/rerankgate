"""CLI end-to-end: ingest -> query -> eval."""

import json
import os

import pytest

from rerankgate.cli import main

CORPUS = [
    {
        "id": "c1",
        "text": "Retrieval-augmented generation pairs a retriever with a generator. " * 8,
        "title": "RAG",
    },
    {
        "id": "c2",
        "text": "Sourdough bread needs a starter, flour, water, and patience. " * 8,
        "title": "Bread",
    },
]


@pytest.fixture
def store(tmp_path):
    corpus_file = tmp_path / "corpus.jsonl"
    corpus_file.write_text("\n".join(json.dumps(d) for d in CORPUS))
    store_dir = str(tmp_path / "store")
    rc = main(["ingest", "--input", str(corpus_file), "--store", store_dir, "--no-vectors"])
    assert rc == 0
    assert os.path.exists(os.path.join(store_dir, "chunks.jsonl"))
    return store_dir


def test_cli_query_answerable(store, capsys):
    rc = main(
        ["query", "--store", store, "--no-vectors", "What is retrieval-augmented generation?"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "verdict:  answerable" in out
    assert "gate:" in out


def test_cli_query_json(store, capsys):
    rc = main(
        [
            "query",
            "--store",
            store,
            "--no-vectors",
            "--json",
            "What is retrieval-augmented generation?",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "answerable"
    assert payload["evidence"]


def test_cli_query_abstains_off_topic(store, capsys):
    rc = main(
        [
            "query",
            "--store",
            store,
            "--no-vectors",
            "--gate-preset",
            "strict",
            "xylophone concerto tickets",
        ]
    )
    assert rc == 0
    assert "verdict:  insufficient" in capsys.readouterr().out


def test_cli_eval(store, tmp_path, capsys):
    qrels = tmp_path / "qrels.jsonl"
    qrels.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "query": "What is retrieval-augmented generation?",
                        "relevant_ids": ["c1"],
                        "answerable": True,
                    }
                ),
                json.dumps(
                    {"query": "xylophone concerto tickets", "relevant_ids": [], "answerable": False}
                ),
            ]
        )
    )
    rc = main(["eval", "--store", store, "--no-vectors", "--qrels", str(qrels)])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["n"] == 2
    assert "recall_at_k" in report
    assert "gate_precision" in report
    assert "abstention_rate" in report
