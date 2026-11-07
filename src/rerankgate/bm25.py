"""Pure-stdlib BM25 (Okapi) index over chunks, backed by an inverted index.

Queries only touch the postings of their own terms, so query cost scales with
the number of chunks that *contain* a query term rather than with corpus size.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter

from rerankgate.documents import Chunk
from rerankgate.tokenize import tokenize


class BM25Index:
    """In-memory BM25 index. Deterministic; rebuild from chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, stem: bool = False) -> None:
        self.stem = stem
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk | None] = []  # None = removed (tombstone)
        self._live = 0
        self._doc_freq: Counter[str] = Counter()
        self._term_freqs: list[Counter[str]] = []
        self._doc_lens: list[int] = []
        self._avg_len: float = 0.0
        # term -> list of (chunk index, term frequency)
        self._postings: dict[str, list[tuple[int, int]]] = {}

    def __len__(self) -> int:
        return self._live

    def add(self, chunks: list[Chunk]) -> None:
        for ch in chunks:
            toks = tokenize(ch.text, self.stem)
            tf = Counter(toks)
            idx = len(self._chunks)
            self._chunks.append(ch)
            self._term_freqs.append(tf)
            self._doc_lens.append(len(toks))
            self._live += 1
            for term, f in tf.items():
                self._doc_freq[term] += 1
                self._postings.setdefault(term, []).append((idx, f))
        self._refresh_avg()

    def _refresh_avg(self) -> None:
        # tombstoned rows have length 0, so the sum is over live chunks only
        self._avg_len = sum(self._doc_lens) / self._live if self._live else 0.0

    def remove_docs(self, doc_ids: set[str]) -> int:
        """Remove every chunk of the given documents; returns how many were removed.

        Removed rows are tombstoned (skipped at search time) and the index is compacted once
        a quarter of its rows are dead, so removal costs O(removed chunks) amortised.
        Scores and ordering afterwards equal those of an index built without them.
        """
        removed = 0
        for idx, ch in enumerate(self._chunks):
            if ch is None or ch.doc_id not in doc_ids:
                continue
            for term in self._term_freqs[idx]:
                self._doc_freq[term] -= 1
                if self._doc_freq[term] <= 0:
                    del self._doc_freq[term]
            self._chunks[idx] = None
            self._term_freqs[idx] = Counter()
            self._doc_lens[idx] = 0
            self._live -= 1
            removed += 1
        if removed:
            self._refresh_avg()
            dead = len(self._chunks) - self._live
            if dead > 1000 and dead * 4 > len(self._chunks):
                self.rebuild([c for c in self._chunks if c is not None])
        return removed

    def rebuild(self, chunks: list[Chunk]) -> None:
        """Replace the whole index with ``chunks``."""
        self._live = 0
        self._chunks = []
        self._doc_freq = Counter()
        self._term_freqs = []
        self._doc_lens = []
        self._postings = {}
        self._avg_len = 0.0
        self.add(chunks)

    def _idf(self, term: str) -> float:
        n = self._live
        df = self._doc_freq.get(term, 0)
        # Standard BM25 IDF with +1 smoothing floor; unseen terms get ~0.
        return math.log(1 + (n - df + 0.5) / (df + 0.5)) if df else 0.0

    def search(self, query: str, k: int = 100) -> list[tuple[Chunk, float]]:
        """Return top-k (chunk, score) pairs, highest score first.

        Ties are broken by insertion order, so results are deterministic.
        """
        q_terms = tokenize(query, self.stem)
        if not q_terms or not self._live or k <= 0:
            return []
        # Each query-term contribution is computed once per (term, chunk), then
        # added once per *occurrence*, in query order, with exactly the same
        # float expression as the 0.2 full scan. Floating-point addition is
        # not associative, so this is what keeps scores and ranking order
        # bit-identical to the old path (repeated terms included).
        contrib: dict[str, dict[int, float]] = {}
        for t in set(q_terms):
            idf = self._idf(t)
            if not idf:
                continue
            per_chunk: dict[int, float] = {}
            for idx, f in self._postings.get(t, ()):
                if self._chunks[idx] is None:
                    continue
                dl = self._doc_lens[idx]
                norm = 1 - self.b + self.b * (dl / self._avg_len) if self._avg_len else 1.0
                per_chunk[idx] = idf * (f * (self.k1 + 1)) / (f + self.k1 * norm)
            contrib[t] = per_chunk
        scores: dict[int, float] = {}
        for t in q_terms:
            for idx, c in contrib.get(t, {}).items():
                scores[idx] = scores.get(idx, 0.0) + c
        # Same order as the old stable sort: score desc, then insertion order.
        top = heapq.nsmallest(k, ((-s, i) for i, s in scores.items() if s > 0))
        return [(self._chunks[i], -neg) for neg, i in top]  # type: ignore[misc]

    def term_stats(self, chunk_idx: int) -> Counter[str]:
        return self._term_freqs[chunk_idx]

    def chunks(self) -> list[Chunk]:
        return [c for c in self._chunks if c is not None]
