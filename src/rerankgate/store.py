"""Document store: JSONL chunks + dependency-free float32 vectors.

Layout of a store directory:

- ``chunks.jsonl`` — one chunk per line
- ``vectors.f32``  — raw little-endian float32, ``chunk_count * dim`` values
  (written with the stdlib ``array`` module: no numpy needed)
- ``meta.json``    — counts, dim, embedding backend name, SHA-256 of both files

Every file is written atomically (temp file + ``os.replace``), and ``load()``
verifies counts, sizes and checksums, failing closed with ``StoreError`` on any
mismatch instead of silently serving a half-written or tampered store.
Stores written by rerankgate <= 0.2 (``vectors.npy``) still load when numpy is
installed.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from array import array
from dataclasses import asdict, fields

from rerankgate.documents import Chunk

STORE_FORMAT = 2
_CHUNK_FIELDS = {f.name for f in fields(Chunk)}
_REQUIRED_FIELDS = {"id", "doc_id", "text", "position"}


class StoreError(ValueError):
    """The store on disk is missing, corrupt, or inconsistent."""


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_write(path: str, data: bytes) -> None:
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=os.path.basename(path))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _floats_to_bytes(vectors: list[list[float]]) -> bytes:
    arr = array("f")
    for v in vectors:
        arr.extend(v)
    if sys.byteorder != "little":  # pragma: no cover - big-endian hosts
        arr.byteswap()
    return arr.tobytes()


def _bytes_to_floats(data: bytes, dim: int) -> list[list[float]]:
    arr = array("f")
    arr.frombytes(data)
    if sys.byteorder != "little":  # pragma: no cover
        arr.byteswap()
    flat = arr.tolist()
    return [flat[i : i + dim] for i in range(0, len(flat), dim)]


class DocumentStore:
    """Persist chunks as JSONL and vectors as raw float32."""

    def __init__(self, path: str) -> None:
        self.path = path

    @property
    def chunks_file(self) -> str:
        return os.path.join(self.path, "chunks.jsonl")

    @property
    def vectors_file(self) -> str:
        return os.path.join(self.path, "vectors.f32")

    @property
    def legacy_vectors_file(self) -> str:
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
        if vectors is not None and len(vectors) != len(chunks):
            raise StoreError(f"{len(vectors)} vectors for {len(chunks)} chunks")
        dim = len(vectors[0]) if vectors else 0
        if vectors and any(len(v) != dim for v in vectors):
            raise StoreError("vectors have inconsistent dimensions")
        os.makedirs(self.path, exist_ok=True)

        chunk_bytes = "".join(
            json.dumps(asdict(ch), ensure_ascii=False) + "\n" for ch in chunks
        ).encode("utf-8")
        _atomic_write(self.chunks_file, chunk_bytes)
        info = {
            "format": STORE_FORMAT,
            "chunk_count": len(chunks),
            "has_vectors": bool(vectors),
            "dim": dim,
            "chunks_sha256": hashlib.sha256(chunk_bytes).hexdigest(),
        }
        if vectors:
            vec_bytes = _floats_to_bytes(vectors)
            _atomic_write(self.vectors_file, vec_bytes)
            info["vectors_sha256"] = hashlib.sha256(vec_bytes).hexdigest()
        elif os.path.exists(self.vectors_file):
            os.unlink(self.vectors_file)  # never leave stale vectors behind
        # meta.json is written last: it is the commit marker for the store.
        _atomic_write(self.meta_file, json.dumps({**(meta or {}), **info}).encode("utf-8"))

    def load(self) -> tuple[list[Chunk], list[list[float]] | None, dict]:
        if not os.path.exists(self.chunks_file):
            raise FileNotFoundError(f"no store at {self.path!r}: run `rerankgate ingest` first")
        meta: dict = {}
        if os.path.exists(self.meta_file):
            try:
                with open(self.meta_file, encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, ValueError) as exc:
                raise StoreError(f"unreadable meta.json: {exc}") from exc
            if not isinstance(meta, dict):
                raise StoreError("meta.json must contain a JSON object")

        expected_sha = meta.get("chunks_sha256")
        if expected_sha and _sha256(self.chunks_file) != expected_sha:
            raise StoreError("chunks.jsonl checksum mismatch (corrupt or modified store)")

        chunks: list[Chunk] = []
        with open(self.chunks_file, encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                chunks.append(_parse_chunk(line, lineno))
        if "chunk_count" in meta and meta["chunk_count"] != len(chunks):
            raise StoreError(
                f"meta.json says {meta['chunk_count']} chunks, chunks.jsonl has {len(chunks)}"
            )

        vectors: list[list[float]] | None = None
        if meta.get("has_vectors"):
            vectors = self._load_vectors(meta, len(chunks))
        return chunks, vectors, meta

    def _load_vectors(self, meta: dict, n_chunks: int) -> list[list[float]] | None:
        dim = meta.get("dim")
        if os.path.exists(self.vectors_file):
            if not isinstance(dim, int) or dim <= 0:
                raise StoreError("meta.json has vectors but no valid 'dim'")
            size = os.path.getsize(self.vectors_file)
            if size != n_chunks * dim * 4:
                raise StoreError(
                    f"vectors.f32 is {size} bytes, expected {n_chunks * dim * 4} "
                    f"({n_chunks} chunks x {dim} dims x 4 bytes)"
                )
            with open(self.vectors_file, "rb") as f:
                data = f.read()
            expected = meta.get("vectors_sha256")
            if expected and hashlib.sha256(data).hexdigest() != expected:
                raise StoreError("vectors.f32 checksum mismatch (corrupt or modified store)")
            return _bytes_to_floats(data, dim)
        if os.path.exists(self.legacy_vectors_file):
            try:
                import numpy as np
            except ImportError:
                # Legacy store, no numpy: vectors are unreadable here. The
                # pipeline re-embeds (if it has a backend) or runs BM25-only.
                return None
            vectors = np.load(self.legacy_vectors_file, allow_pickle=False).astype(float).tolist()
            if len(vectors) != n_chunks:
                raise StoreError(f"vectors.npy has {len(vectors)} rows for {n_chunks} chunks")
            return vectors
        raise StoreError("meta.json says the store has vectors, but no vector file was found")


def _parse_chunk(line: str, lineno: int) -> Chunk:
    try:
        obj = json.loads(line)
    except ValueError as exc:
        raise StoreError(f"chunks.jsonl line {lineno}: invalid JSON ({exc})") from exc
    if not isinstance(obj, dict):
        raise StoreError(f"chunks.jsonl line {lineno}: expected an object")
    missing = _REQUIRED_FIELDS - obj.keys()
    if missing:
        raise StoreError(f"chunks.jsonl line {lineno}: missing fields {sorted(missing)}")
    obj = {k: v for k, v in obj.items() if k in _CHUNK_FIELDS}
    if not isinstance(obj["text"], str) or not isinstance(obj["id"], str):
        raise StoreError(f"chunks.jsonl line {lineno}: 'id' and 'text' must be strings")
    if not isinstance(obj.get("metadata", {}), dict):
        raise StoreError(f"chunks.jsonl line {lineno}: 'metadata' must be an object")
    return Chunk(**obj)
