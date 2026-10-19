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
        if buf:
            chunks.append(
                Chunk(
                    id=f"{doc_id}#c{n}",
                    doc_id=doc_id,
                    text=" ".join(buf),
                    position=n,
                    metadata=dict(metadata or {}),
                )
            )
            n += 1

    for sent in sentences:
        # A single over-long sentence becomes its own chunk (hard-split).
        while len(sent) > size:
            flush()
            chunks.append(
                Chunk(
                    id=f"{doc_id}#c{n}",
                    doc_id=doc_id,
                    text=sent[:size],
                    position=n,
                    metadata=dict(metadata or {}),
                )
            )
            n += 1
            sent = sent[size - overlap :]
        if buf_len + len(sent) + 1 > size and buf:
            flush()
            # overlap: carry the tail of the previous chunk forward
            tail = " ".join(buf)
            buf = [tail[-overlap:]] if len(tail) > overlap else []
            buf_len = sum(len(b) + 1 for b in buf)
        buf.append(sent)
        buf_len += len(sent) + 1
    flush()
    return chunks
