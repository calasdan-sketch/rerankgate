"""Tamper-evident decision log: one JSON line per ``ask()``, hash-chained.

Why: when an answer was withheld (or given) an auditor asks "what did the system decide, on
what evidence, under which settings?". Each record carries the verdict, gate score and
threshold, reason codes, the ids of the evidence used and a fingerprint of the scoring
configuration. Records are chained: each stores the SHA-256 of the previous record, so an
edited, deleted or reordered line is detected by :func:`verify`.

What this is not: a security boundary. Anyone who can write the file can rewrite the whole
chain and recompute every hash. To make rewriting detectable, copy ``AuditLog.head`` to
somewhere the writer cannot reach (another host, a ticket, a signed commit) and pass it to
``verify(path, expected_head=...)``. Queries are user data: by default only their SHA-256 is
stored; pass ``store_queries=True`` to keep the text as well. Single-process writer (a lock
serialises threads; two processes appending to one file can interleave and break the chain).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

GENESIS = "0" * 64
FORMAT = "rerankgate-audit-1"


def _canon(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _digest(record: dict) -> str:
    body = {k: v for k, v in record.items() if k != "hash"}
    return hashlib.sha256(_canon(body)).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass
class VerifyReport:
    ok: bool
    records: int
    head: str
    error: str | None = None
    bad_line: int | None = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "records": self.records,
            "head": self.head,
            "error": self.error,
            "bad_line": self.bad_line,
        }


def verify(path: str, expected_head: str | None = None) -> VerifyReport:
    """Walk the chain. Fails closed: any unreadable, malformed or out-of-order line is an error."""
    prev, n = GENESIS, 0
    try:
        f = open(path, "rb")
    except OSError as exc:
        return VerifyReport(False, 0, GENESIS, f"cannot open {path!r}: {exc.strerror or exc}")
    with f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.rstrip(b"\n")
            if not line.strip():
                return VerifyReport(False, n, prev, "blank line inside the log", lineno)
            if not raw.endswith(b"\n"):
                return VerifyReport(False, n, prev, "last record is truncated", lineno)
            try:
                rec = json.loads(line)
            except (ValueError, RecursionError):
                return VerifyReport(False, n, prev, "line is not valid JSON", lineno)
            if not isinstance(rec, dict) or rec.get("format") != FORMAT:
                return VerifyReport(False, n, prev, "not a rerankgate audit record", lineno)
            if rec.get("seq") != n + 1:
                return VerifyReport(
                    False, n, prev, f"sequence break (got {rec.get('seq')})", lineno
                )
            if rec.get("prev") != prev:
                return VerifyReport(False, n, prev, "chain broken: prev hash mismatch", lineno)
            if rec.get("hash") != _digest(rec):
                return VerifyReport(
                    False, n, prev, "record content does not match its hash", lineno
                )
            prev, n = rec["hash"], n + 1
    if expected_head is not None and expected_head != prev:
        return VerifyReport(False, n, prev, "head does not match the anchored value")
    return VerifyReport(True, n, prev)


class AuditLog:
    """Append-only hash-chained JSONL log. Use ``pipe.audit = AuditLog("decisions.jsonl")``."""

    def __init__(
        self,
        path: str,
        store_queries: bool = False,
        clock: Callable[[], str] = _utc_now,
    ) -> None:
        self.path = path
        self.store_queries = store_queries
        self._clock = clock
        self._lock = threading.Lock()
        self._seq = 0
        self._head = GENESIS
        if os.path.exists(path) and os.path.getsize(path) > 0:
            rep = verify(path)
            if not rep.ok:
                # never append to a log that does not verify: that would launder a bad chain
                raise ValueError(f"audit log {path!r} fails verification: {rep.error}")
            self._seq, self._head = rep.records, rep.head

    @property
    def head(self) -> str:
        """Hash of the newest record - anchor this outside the log's reach."""
        return self._head

    def record(self, result, fingerprint: str, version: str) -> dict:
        """Append one record for an ``AnswerResult``; returns it."""
        g = result.gate
        with self._lock:
            body = {
                "format": FORMAT,
                "seq": self._seq + 1,
                "ts": self._clock(),
                "prev": self._head,
                "version": version,
                "fingerprint": fingerprint,
                "query_sha256": hashlib.sha256(result.query.encode("utf-8", "replace")).hexdigest(),
                "verdict": result.verdict,
                "gate": {
                    "passed": g.passed,
                    "score": g.score,
                    "threshold": g.threshold,
                    "reasons": list(g.reasons),
                },
                "gate_overridden": bool(result.trace.get("gate_overridden")),
                "retries_used": result.trace.get("retries_used"),
                "evidence": [c.chunk.id for c in result.evidence],
                "suspect_evidence": list(result.trace.get("suspect_evidence", [])),
            }
            if self.store_queries:
                body["query"] = result.query
            body["hash"] = _digest(body)
            line = _canon(body) + b"\n"
            # one O_APPEND write per record keeps a crash from leaving half a line
            fd = os.open(
                self.path,
                os.O_WRONLY
                | os.O_APPEND
                | os.O_CREAT
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                view = memoryview(line)
                while view:  # os.write may write less than asked
                    view = view[os.write(fd, view) :]
                os.fsync(fd)
            finally:
                os.close(fd)
            self._seq += 1
            self._head = body["hash"]
            return body
