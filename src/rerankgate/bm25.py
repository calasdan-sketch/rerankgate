"""Pure-stdlib BM25 (Okapi) index over chunks."""

from __future__ import annotations

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

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk]) -> None:
        for ch in chunks:
            toks = tokenize(ch.text)
            tf = Counter(toks)
            self._chunks.append(ch)
            self._term_freqs.append(tf)
            self._doc_lens.append(len(toks))
            for term in tf:
                self._doc_freq[term] += 1
        total = sum(self._doc_lens)
        self._avg_len = total / len(self._doc_lens) if self._doc_lens else 0.0

    def _idf(self, term: str) -> float:
        n = len(self._chunks)
        df = self._doc_freq.get(term, 0)
        # Standard BM25 IDF with +1 smoothing floor; unseen terms get ~0.
        return math.log(1 + (n - df + 0.5) / (df + 0.5)) if df else 0.0

    def search(self, query: str, k: int = 100) -> list[tuple[Chunk, float]]:
        """Return top-k (chunk, score) pairs, highest score first."""
        q_terms = tokenize(query)
        if not q_terms or not self._chunks:
            return []
        idfs = {t: self._idf(t) for t in set(q_terms)}
        scored: list[tuple[Chunk, float]] = []
        for i, ch in enumerate(self._chunks):
            tf = self._term_freqs[i]
            dl = self._doc_lens[i]
            norm = 1 - self.b + self.b * (dl / self._avg_len) if self._avg_len else 1.0
            score = 0.0
            for t in q_terms:
                f = tf.get(t, 0)
                if f and idfs[t]:
                    score += idfs[t] * (f * (self.k1 + 1)) / (f + self.k1 * norm)
            if score > 0:
                scored.append((ch, score))
        scored.sort(key=lambda p: p[1], reverse=True)
        return scored[:k]

    def term_stats(self, chunk_idx: int) -> Counter[str]:
        return self._term_freqs[chunk_idx]

    def chunks(self) -> list[Chunk]:
        return list(self._chunks)
