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

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._doc_freq: Counter[str] = Counter()
        self._term_freqs: list[Counter[str]] = []
        self._doc_lens: list[int] = []
        self._avg_len: float = 0.0
        # term -> list of (chunk index, term frequency)
        self._postings: dict[str, list[tuple[int, int]]] = {}

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk]) -> None:
        for ch in chunks:
            toks = tokenize(ch.text)
            tf = Counter(toks)
            idx = len(self._chunks)
            self._chunks.append(ch)
            self._term_freqs.append(tf)
            self._doc_lens.append(len(toks))
            for term, f in tf.items():
                self._doc_freq[term] += 1
                self._postings.setdefault(term, []).append((idx, f))
        total = sum(self._doc_lens)
        self._avg_len = total / len(self._doc_lens) if self._doc_lens else 0.0

    def _idf(self, term: str) -> float:
        n = len(self._chunks)
        df = self._doc_freq.get(term, 0)
        # Standard BM25 IDF with +1 smoothing floor; unseen terms get ~0.
        return math.log(1 + (n - df + 0.5) / (df + 0.5)) if df else 0.0

    def search(self, query: str, k: int = 100) -> list[tuple[Chunk, float]]:
        """Return top-k (chunk, score) pairs, highest score first.

        Ties are broken by insertion order, so results are deterministic.
        """
        q_terms = tokenize(query)
        if not q_terms or not self._chunks or k <= 0:
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
        return [(self._chunks[i], -neg) for neg, i in top]

    def term_stats(self, chunk_idx: int) -> Counter[str]:
        return self._term_freqs[chunk_idx]

    def chunks(self) -> list[Chunk]:
        return list(self._chunks)
