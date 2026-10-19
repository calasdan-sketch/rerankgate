from rerankgate.documents import chunk_text


def test_chunking_empty():
    assert chunk_text("", doc_id="d") == []
    assert chunk_text("   ", doc_id="d") == []


def test_chunking_deterministic():
    text = "First sentence. Second sentence here. Third one is longer than the others."
    a = chunk_text(text, doc_id="d", size=40, overlap=5)
    b = chunk_text(text, doc_id="d", size=40, overlap=5)
    assert [c.text for c in a] == [c.text for c in b]
    assert [c.id for c in a] == [f"d#c{i}" for i in range(len(a))]


def test_chunking_respects_size():
    text = " ".join(f"Sentence number {i} is here." for i in range(30))
    chunks = chunk_text(text, doc_id="d", size=100, overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 100 + 60  # sentence granularity may overshoot slightly


def test_chunking_long_sentence_hard_split():
    text = "x" * 300
    chunks = chunk_text(text, doc_id="d", size=100, overlap=10)
    assert len(chunks) >= 2
    assert all(len(c.text) <= 100 for c in chunks)


def test_chunking_metadata_propagated():
    chunks = chunk_text("Hello world. Bye world.", doc_id="d", metadata={"title": "T"})
    assert chunks and all(c.metadata["title"] == "T" for c in chunks)
