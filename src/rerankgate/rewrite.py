"""Query rewriting without an LLM: RM3-style pseudo-relevance feedback.

Takes the top documents of a failed attempt and expands the query with
high-value terms from them — the classic relevance-feedback trick,
deterministic and free.
"""

from __future__ import annotations

from collections import Counter

from rerankgate.retrieval import ScoredChunk
from rerankgate.tokenize import STOPWORDS, tokenize


class RM3Rewriter:
    """Expand the query with discriminative terms from top failed-attempt docs."""

    def __init__(self, expansion_terms: int = 5, top_docs: int = 3) -> None:
        self.expansion_terms = expansion_terms
        self.top_docs = top_docs

    def rewrite(self, query: str, evidence: list[ScoredChunk]) -> str:
        """Return an expanded query; returns the original if nothing to add."""
        q_terms = set(tokenize(query))
        term_counts: Counter[str] = Counter()
        for sc in evidence[: self.top_docs]:
            for t in tokenize(sc.chunk.text):
                if t not in STOPWORDS and t not in q_terms and len(t) > 2:
                    term_counts[t] += 1
        if not term_counts:
            return query
        # Prefer terms that appear across multiple top docs (consensus terms).
        ranked = sorted(term_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        new_terms = [t for t, _ in ranked[: self.expansion_terms]]
        if not new_terms:
            return query
        return query.rstrip() + " " + " ".join(new_terms)
