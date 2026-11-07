"""Document store: JSONL chunks + dependency-free float32 vectors.

Layout of a store directory:

- ``chunks.jsonl`` — one chunk per line
- ``vectors.f32``  — raw little-endian float32, ``chunk_count * dim`` values
  (written with the stdlib ``array`` module: no numpy needed)
- ``meta.json``    — counts, dim, embedding backend name, SHA-256 of both files

Every file is written atomically (temp file + ``os.replace``), and ``load()``
verifies counts, sizes, field types and checksums, failing closed with
``StoreError`` instead of serving a half-written or corrupted store.

The checksums are plain SHA-256 with no secret key: they detect accidental
corruption (crashes, truncation, disk errors, bad hand edits), NOT deliberate
tampering — anyone who can write the files can recompute them. Protect the
store directory with filesystem permissions if that matters.

``meta.json`` is required (every rerankgate version writes it). Stores written
by rerankgate <= 0.2 (no ``format`` key, ``vectors.npy``) still load, without
checksum verification; format 2+ stores must carry their checksums.
"""

from __future__ import annotations

import hashlib
import json
import math
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
        if not os.path.exists(self.meta_file):
            raise StoreError(
                "meta.json is missing: every rerankgate store has one, so this store is "
                "incomplete or corrupt — re-run `rerankgate ingest`"
            )
        try:
            with open(self.meta_file, encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError, RecursionError) as exc:
            raise StoreError(f"unreadable meta.json: {exc}") from exc
        if not isinstance(meta, dict):
            raise StoreError("meta.json must contain a JSON object")
        _validate_meta(meta)

        expected_sha = meta.get("chunks_sha256")
        if expected_sha is not None and _sha256(self.chunks_file) != expected_sha:
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
            if expected is not None and hashlib.sha256(data).hexdigest() != expected:
                raise StoreError("vectors.f32 checksum mismatch (corrupt or modified store)")
            vectors = _bytes_to_floats(data, dim)
            _check_finite(vectors, "vectors.f32")
            return vectors
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
            _check_finite(vectors, "vectors.npy")
            return vectors
        raise StoreError("meta.json says the store has vectors, but no vector file was found")


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _validate_meta(meta: dict) -> None:
    fmt = meta.get("format", 1)
    if not _is_int(fmt) or fmt < 1:
        raise StoreError(f"meta.json: 'format' must be a positive integer, got {fmt!r}")
    if fmt > STORE_FORMAT:
        raise StoreError(
            f"store format {fmt} is newer than this rerankgate supports ({STORE_FORMAT}); upgrade"
        )
    if "chunk_count" in meta and (not _is_int(meta["chunk_count"]) or meta["chunk_count"] < 0):
        raise StoreError("meta.json: 'chunk_count' must be a non-negative integer")
    if "has_vectors" in meta and not isinstance(meta["has_vectors"], bool):
        raise StoreError("meta.json: 'has_vectors' must be true or false")
    if "dim" in meta and (not _is_int(meta["dim"]) or meta["dim"] < 0):
        raise StoreError("meta.json: 'dim' must be a non-negative integer")
    for key in ("chunks_sha256", "vectors_sha256"):
        v = meta.get(key)
        if v is not None and not (
            isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)
        ):
            raise StoreError(f"meta.json: {key!r} must be a SHA-256 hex digest")
    if fmt >= 2:
        # Format 2+ always writes these; a missing one means a damaged store.
        required = ["chunk_count", "has_vectors", "dim", "chunks_sha256"]
        if meta.get("has_vectors"):
            required.append("vectors_sha256")
        missing = [k for k in required if k not in meta]
        if missing:
            raise StoreError(f"meta.json (format {fmt}) is missing {missing}")


def _check_finite(vectors: list[list[float]], name: str) -> None:
    for row in vectors:
        for x in row:
            if not math.isfinite(x):
                raise StoreError(f"{name} contains NaN or infinite values")


def _parse_chunk(line: str, lineno: int) -> Chunk:
    try:
        obj = json.loads(line)
    except (ValueError, RecursionError) as exc:
        raise StoreError(f"chunks.jsonl line {lineno}: invalid JSON ({exc})") from exc
    if not isinstance(obj, dict):
        raise StoreError(f"chunks.jsonl line {lineno}: expected an object")
    missing = _REQUIRED_FIELDS - obj.keys()
    if missing:
        raise StoreError(f"chunks.jsonl line {lineno}: missing fields {sorted(missing)}")
    obj = {k: v for k, v in obj.items() if k in _CHUNK_FIELDS}
    for key in ("id", "doc_id", "text"):
        if not isinstance(obj[key], str):
            raise StoreError(f"chunks.jsonl line {lineno}: {key!r} must be a string")
    if not _is_int(obj["position"]) or obj["position"] < 0:
        raise StoreError(f"chunks.jsonl line {lineno}: 'position' must be a non-negative integer")
    if not isinstance(obj.get("metadata", {}), dict):
        raise StoreError(f"chunks.jsonl line {lineno}: 'metadata' must be an object")
    return Chunk(**obj)
