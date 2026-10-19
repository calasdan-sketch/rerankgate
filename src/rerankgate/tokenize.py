"""Shared tokenization and stopwords."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """
    a an the and or but if then else when while of at by for with about into through
    during before after above below to from up down in out on off over under again
    further once here there all any both each few more most other some such no nor
    not only own same so than too very can will just don should now is are was were
    be been being have has had having do does did doing would could ought i you he
    she it we they them his her its our their this that these those am as it s t
    what which who whom how why where when
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens."""
    return _TOKEN.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokens with stopwords removed."""
    return [t for t in tokenize(text) if t not in STOPWORDS]
