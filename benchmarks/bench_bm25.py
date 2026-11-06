"""Reproducible BM25 query-latency benchmark on a synthetic corpus.

    python benchmarks/bench_bm25.py [n_chunks]

The corpus is random words drawn from a Zipf-like vocabulary, so numbers show
relative scaling on *your* machine, not quality. Compares the inverted-index
search against a full-scan scorer (the algorithm used in rerankgate <= 0.2).

The speedup depends heavily on the query mix: the inverted index only skips
chunks that contain none of the query's terms. Three query sets are timed:

- rare:     4 terms from ranks 50-5000 (specific keyword queries)
- mixed:    2 very frequent terms (ranks 0-50, standing in for words like
            "what", "is", "the" — BM25 keeps stopwords) + 2 rarer terms;
            closest to a natural-language question
- frequent: 4 terms from ranks 0-50 (worst case)
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
    query_sets = {
        "rare": [" ".join(rng.choices(vocab[50:5000], k=4)) for _ in range(30)],
        "mixed": [
            " ".join(rng.choices(vocab[:50], k=2) + rng.choices(vocab[50:5000], k=2))
            for _ in range(30)
        ],
        "frequent": [" ".join(rng.choices(vocab[:50], k=4)) for _ in range(30)],
    }

    def timed(fn, queries):
        t = time.perf_counter()
        for q in queries:
            fn(idx, q)
        return (time.perf_counter() - t) / len(queries) * 1000

    print(f"chunks={n}  index build={build:.2f}s")
    print(f"{'queries':>9}  {'full scan (<=0.2)':>18}  {'inverted (0.3)':>15}  speedup")
    for name, queries in query_sets.items():
        old_ms = timed(full_scan, queries)
        new_ms = timed(lambda i, q: i.search(q, k=100), queries)
        print(f"{name:>9}  {old_ms:>15.2f} ms  {new_ms:>12.2f} ms  {old_ms / new_ms:6.1f}x")


if __name__ == "__main__":
    main()
