"""Documents and deterministic chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Document:
    """A source document before chunking."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def title(self) -> str:
        return str(self.metadata.get("title", ""))


@dataclass
class Chunk:
    """A retrievable unit produced by chunking a Document."""

    id: str  # f"{doc_id}#c{n}"
    doc_id: str
    text: str
    position: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _word_tail(text: str, n: int) -> str:
    """The last ``n`` characters of ``text``, without a leading half-word."""
    if n <= 0:
        return ""
    tail = text[-n:]
    if text[-n - 1 : -n].isspace() or not tail:
        return tail.lstrip()
    cut = tail.find(" ")
    return tail[cut + 1 :] if cut != -1 else ""


def _hard_split(sent: str, size: int, overlap: int) -> tuple[str, str]:
    """Cut one over-long sentence: (piece of at most ``size`` chars, the rest).

    Prefers to cut at whitespace so words stay whole (a word sliced in two can never be
    matched by a query: log lines, CSV rows and code without sentence punctuation hit
    this). The rest restarts about ``overlap`` characters before the cut, also at a word
    start. With no whitespace in the window (one giant token) it cuts at ``size`` as before.
    """
    window = sent[: size + 1]
    cut = max(window.rfind(" "), window.rfind("\n"), window.rfind("\t"))
    if cut < max(1, size // 2):
        return sent[:size], sent[size - overlap :]
    piece = sent[:cut].rstrip()
    start = max(1, cut - overlap)
    if not sent[start - 1].isspace():
        nxt = next((i for i in range(start, cut + 1) if sent[i].isspace()), cut)
        start = nxt + 1
    return piece, sent[start:].lstrip()


def chunk_text(
    text: str,
    doc_id: str,
    size: int = 500,
    overlap: int = 50,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Split text into sentence-aware sliding-window chunks.

    Deterministic: same input always yields the same chunks.
    """
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"chunk size must be a positive integer, got {size!r}")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 <= overlap < size:
        raise ValueError(f"chunk overlap must be an integer in [0, size), got {overlap!r}")
    text = text.strip()
    if not text:
        return []
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        sentences = [text]

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0
    n = 0

    def flush() -> None:
        nonlocal n, buf, buf_len
        body = " ".join(buf).strip()  # an overlap tail may start/end on a tab or newline
        if body:
            chunks.append(
                Chunk(
                    id=f"{doc_id}#c{n}",
                    doc_id=doc_id,
                    text=body,
                    position=n,
                    metadata=dict(metadata or {}),
                )
            )
            n += 1

    for sent in sentences:
        # A single over-long sentence becomes its own chunk (hard-split).
        while len(sent) > size:
            flush()
            buf, buf_len = [], 0  # already emitted: without this it was emitted again per piece
            piece, rest = _hard_split(sent, size, overlap)
            if piece.strip():
                chunks.append(
                    Chunk(
                        id=f"{doc_id}#c{n}",
                        doc_id=doc_id,
                        text=piece.strip(),
                        position=n,
                        metadata=dict(metadata or {}),
                    )
                )
                n += 1
            sent = rest.lstrip()
        if buf_len + len(sent) + 1 > size and buf:
            flush()
            # overlap: carry the tail of the previous chunk forward
            tail = " ".join(buf)
            buf = [_word_tail(tail, overlap)] if len(tail) > overlap else []
            buf = [b for b in buf if b]
            buf_len = sum(len(b) + 1 for b in buf)
        buf.append(sent)
        buf_len += len(sent) + 1
    flush()
    return chunks
