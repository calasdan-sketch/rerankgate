"""What the user sees when the system abstains - without calling an LLM.

An abstention is a product moment, not just ``None``. ``explain_abstention`` turns a
non-answerable ``AnswerResult`` into: why (in plain words), which of the user's terms
were and were not found, the closest sources *labelled as not answering*, one concrete
next step (search / clarify / escalate), and a ready-to-show message.

Everything shown is derived from the user's own query terms and document ids/titles;
no evidence text is echoed (weak context must not look like an answer). All strings
pass through ``safeterm.clean`` since queries and titles are untrusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rerankgate.safeterm import clean
from rerankgate.tokenize import content_tokens, tokenize

NO_MATCH = "no_match"  # nothing on this topic in the knowledge base
PARTIAL = "partial"  # some of the question is covered, some is not
AMBIGUOUS = "ambiguous"  # on topic but no clear best source
ERROR = "error"  # the scoring itself failed: not the user's fault


@dataclass
class AbstentionNotice:
    kind: str
    action: str  # "search" | "clarify" | "escalate" | "retry_later"
    message: str
    matched_terms: list[str] = field(default_factory=list)
    missing_terms: list[str] = field(default_factory=list)
    nearest_sources: list[dict] = field(default_factory=list)
    clarifying_question: str | None = None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "action": self.action,
            "message": self.message,
            "matched_terms": self.matched_terms,
            "missing_terms": self.missing_terms,
            "nearest_sources": self.nearest_sources,
            "clarifying_question": self.clarifying_question,
        }


def _ordered_terms(query: str) -> list[str]:
    seen: list[str] = []
    for t in content_tokens(query):
        if t not in seen:
            seen.append(t)
    return seen


def explain_abstention(result, max_sources: int = 3, max_terms: int = 5) -> AbstentionNotice | None:
    """Return a notice for an abstained result, or ``None`` if the result was answerable."""
    if result.verdict != "insufficient":
        return (
            None  # answerable, or needs_review (an unfaithful answer, not a retrieval abstention)
        )
    reasons = list(result.gate.reasons)
    terms = _ordered_terms(result.query)
    ev_tokens: set[str] = set()
    for c in result.evidence:
        ev_tokens |= set(tokenize(c.chunk.text))
    matched = [clean(t) for t in terms if t in ev_tokens][:max_terms]
    missing = [clean(t) for t in terms if t not in ev_tokens][:max_terms]
    sources = []
    for c in result.evidence[:max_sources]:
        title = c.chunk.metadata.get("title") if isinstance(c.chunk.metadata, dict) else None
        sources.append(
            {
                "doc_id": clean(c.chunk.doc_id),
                "title": clean(title) if isinstance(title, str) else None,
                "note": "closest match found - it does not answer the question",
            }
        )

    if "INVALID_SCORE" in reasons:
        return AbstentionNotice(
            kind=ERROR,
            action="retry_later",
            message=(
                "I couldn't check my sources reliably just now, so I won't guess. "
                "Please try again or ask a person."
            ),
        )
    if "NO_EVIDENCE" in reasons or not terms or not matched:
        return AbstentionNotice(
            kind=NO_MATCH,
            action="search",
            message=(
                "I couldn't find anything about this in the knowledge base, so I won't guess. "
                "Try different keywords, or ask a person."
            ),
            missing_terms=missing,
            nearest_sources=[],
        )
    if missing and len(missing) >= max(1, len(terms) // 3):
        q = (
            f"I found material about {', '.join(matched)} but nothing on "
            f"{', '.join(missing)}. Can you say more about {missing[0]}?"
        )
        return AbstentionNotice(
            kind=PARTIAL,
            action="clarify",
            message=(
                f"I only found part of what you asked: it covers {', '.join(matched)} "
                f"but not {', '.join(missing)}. I won't fill the gap with a guess."
            ),
            matched_terms=matched,
            missing_terms=missing,
            nearest_sources=sources,
            clarifying_question=q,
        )
    return AbstentionNotice(
        kind=AMBIGUOUS,
        action="clarify",
        message=(
            "I found related material but no single source that clearly answers this. "
            "Could you narrow the question?"
        ),
        matched_terms=matched,
        missing_terms=missing,
        nearest_sources=sources,
        clarifying_question="Which of these are you asking about?" if sources else None,
    )
