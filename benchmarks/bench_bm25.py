"""Reproducible BM25 query-latency benchmark on a synthetic corpus.

    python benchmarks/bench_bm25.py [n_chunks]

The corpus is random words drawn from a Zipf-like vocabulary, so numbers show
relative scaling on *your* machine, not quality. Compares the inverted-index
search against a full-scan scorer (the algorithm used in rerankgate <= 0.2).
"""

from __future__ import annotations

import random
import sys
import time

from rerankgate.bm25 import BM25Index
from rerankgate.documents import Chunk
from rerankgate.tokenize import tokenize


def full_scan(index: BM25Index, query: str, k: int = 100):
    q = tokenize(query)
    idfs = {t: index._idf(t) for t in set(q)}
    out = []
    for i, ch in enumerate(index._chunks):
        tf, dl = index._term_freqs[i], index._doc_lens[i]
        norm = 1 - index.b + index.b * dl / index._avg_len
        s = 0.0
        for t in q:
            f = tf.get(t, 0)
            if f and idfs[t]:
                s += idfs[t] * f * (index.k1 + 1) / (f + index.k1 * norm)
        if s > 0:
            out.append((ch, s))
    out.sort(key=lambda p: p[1], reverse=True)
    return out[:k]


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    rng = random.Random(0)
    vocab = [f"term{i}" for i in range(20000)]
    weights = [1 / (i + 1) for i in range(len(vocab))]
    chunks = [
        Chunk(f"d{i}#c0", f"d{i}", " ".join(rng.choices(vocab, weights, k=80)), 0) for i in range(n)
    ]
    idx = BM25Index()
    t = time.perf_counter()
    idx.add(chunks)
    build = time.perf_counter() - t
    queries = [" ".join(rng.choices(vocab[50:5000], k=4)) for _ in range(30)]

    def timed(fn):
        t = time.perf_counter()
        for q in queries:
            fn(idx, q)
        return (time.perf_counter() - t) / len(queries) * 1000

    new_ms = timed(lambda i, q: i.search(q, k=100))
    old_ms = timed(full_scan)
    print(f"chunks={n}  index build={build:.2f}s")
    print(f"full scan (<=0.2):     {old_ms:8.2f} ms/query")
    print(f"inverted index (0.3):  {new_ms:8.2f} ms/query  ({old_ms / new_ms:.1f}x)")


if __name__ == "__main__":
    main()
