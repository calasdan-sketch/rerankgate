"""Document store: JSONL persistence for chunks + optional numpy vectors."""

from __future__ import annotations

import json
import os
from dataclasses import asdict

from rerankgate.documents import Chunk
from rerankgate.vectorstore import _HAS_NUMPY


class DocumentStore:
    """Persist chunks as JSONL; vectors as .npy when numpy is available."""

    def __init__(self, path: str) -> None:
        self.path = path

    @property
    def chunks_file(self) -> str:
        return os.path.join(self.path, "chunks.jsonl")

    @property
    def vectors_file(self) -> str:
        return os.path.join(self.path, "vectors.npy")

    @property
    def meta_file(self) -> str:
        return os.path.join(self.path, "meta.json")

    def save(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]] | None = None,
        meta: dict | None = None,
    ) -> None:
        os.makedirs(self.path, exist_ok=True)
        with open(self.chunks_file, "w", encoding="utf-8") as f:
            for ch in chunks:
                f.write(json.dumps(asdict(ch)) + "\n")
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "chunk_count": len(chunks),
                    "has_vectors": vectors is not None,
                    "dim": len(vectors[0]) if vectors else 0,
                    **(meta or {}),
                },
                f,
            )
        if vectors is not None:
            if not _HAS_NUMPY:
                raise RuntimeError("numpy is required to persist vectors")
            import numpy as np

            np.save(self.vectors_file, np.asarray(vectors, dtype=np.float32))

    def load(self) -> tuple[list[Chunk], list[list[float]] | None, dict]:
        if not os.path.exists(self.chunks_file):
            raise FileNotFoundError(f"no store at {self.path!r}: run `rerankgate ingest` first")
        chunks: list[Chunk] = []
        with open(self.chunks_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(Chunk(**json.loads(line)))
        meta: dict = {}
        if os.path.exists(self.meta_file):
            with open(self.meta_file, encoding="utf-8") as f:
                meta = json.load(f)
        vectors: list[list[float]] | None = None
        if meta.get("has_vectors") and os.path.exists(self.vectors_file):
            if not _HAS_NUMPY:
                raise RuntimeError("store has vectors but numpy is not installed")
            import numpy as np

            vectors = np.load(self.vectors_file).astype(float).tolist()
        return chunks, vectors, meta
