"""Shared tokenization and stopwords.

Unicode-aware: text is NFKC-normalized and case-folded, so "Café", "CAFÉ" and
"café" are one token, and full-width digits/letters match their ASCII forms.
Chinese/Japanese ideographs, kana and Hangul syllables are split into single
characters (Thai and other unspaced scripts are not handled), which gives BM25 something to match
on instead of one giant "word" per sentence. This is a pragmatic baseline, not a
linguistic segmenter.
"""

from __future__ import annotations

import re
import unicodedata

# Letters/digits in any script (no underscore).
_TOKEN = re.compile(r"[^\W_]+")

# CJK ideographs, kana, and Hangul syllables: tokenized one character at a time.
_CJK = re.compile(
    "["
    "぀-ゟ"  # Hiragana
    "゠-ヿ"  # Katakana
    "㐀-䶿"  # CJK Extension A
    "一-鿿"  # CJK Unified Ideographs
    "가-힯"  # Hangul syllables
    "豈-﫿"  # CJK Compatibility Ideographs
    "]"
)

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


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def tokenize(text: str) -> list[str]:
    """Case-folded word tokens in any script; CJK split per character."""
    out: list[str] = []
    for tok in _TOKEN.findall(_normalize(text)):
        if _CJK.search(tok):
            # Split CJK runs into characters, keep embedded non-CJK runs whole.
            buf = ""
            for ch in tok:
                if _CJK.match(ch):
                    if buf:
                        out.append(buf)
                        buf = ""
                    out.append(ch)
                else:
                    buf += ch
            if buf:
                out.append(buf)
        else:
            out.append(tok)
    return out


def content_tokens(text: str) -> list[str]:
    """Tokens with stopwords removed."""
    return [t for t in tokenize(text) if t not in STOPWORDS]
