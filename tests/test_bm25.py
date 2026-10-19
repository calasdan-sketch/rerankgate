from rerankgate.bm25 import BM25Index
from rerankgate.documents import Document, chunk_text


def _index():
    docs = [
        Document(id="a", text="The quick brown fox jumps over the lazy dog."),
        Document(id="b", text="A fast rust-colored fox leaps across the sleepy hound."),
        Document(id="c", text="Quantum computing uses qubits and superposition."),
    ]
    chunks = []
    for d in docs:
        chunks.extend(chunk_text(d.text, doc_id=d.id))
    idx = BM25Index()
    idx.add(chunks)
    return idx


def test_bm25_ranks_relevant_first():
    idx = _index()
    hits = idx.search("fox jumps", k=3)
    assert hits
    assert hits[0][0].doc_id == "a"


def test_bm25_empty_query():
    idx = _index()
    assert idx.search("", k=3) == []


def test_bm25_no_match_terms():
    idx = _index()
    assert idx.search("xylophone zebra", k=3) == []


def test_bm25_respects_k():
    idx = _index()
    assert len(idx.search("fox", k=1)) == 1


def test_bm25_scores_descending():
    idx = _index()
    hits = idx.search("fox", k=3)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
