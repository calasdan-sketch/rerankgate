"""Shared tokenization and stopwords.

Unicode-aware: tokens are runs of letters, numbers and combining marks (so
Devanagari, Arabic and Hebrew words stay whole); text is NFKC-normalized and
case-folded, so "Café", "CAFÉ" and
"café" are one token, and full-width digits/letters match their ASCII forms.
Chinese/Japanese ideographs, kana and Hangul syllables are split into single
characters (Thai and other unspaced scripts are not handled), which gives BM25 something to match
on instead of one giant "word" per sentence. This is a pragmatic baseline, not a
linguistic segmenter.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from functools import lru_cache


@lru_cache(maxsize=1)
def _token_re() -> re.Pattern[str]:
    r"""Token = run of letters (L*), numbers (N*) and combining marks (M*).

    `[^\W_]` alone covers letters and numbers but not combining marks, which
    would shred Devanagari, Arabic with harakat, Hebrew with niqqud, etc.
    The mark ranges are built once, lazily, from this Python's Unicode data.
    """
    ranges: list[str] = []
    start = None
    for cp in range(sys.maxunicode + 1):
        is_mark = unicodedata.category(chr(cp)).startswith("M")
        if is_mark and start is None:
            start = cp
        elif not is_mark and start is not None:
            ranges.append(f"{_esc(start)}-{_esc(cp - 1)}")
            start = None
    if start is not None:  # pragma: no cover - no marks at the very top
        ranges.append(f"{_esc(start)}-{_esc(sys.maxunicode)}")
    return re.compile(r"(?:[^\W_]|[" + "".join(ranges) + "])+")


def _esc(cp: int) -> str:
    return f"\\U{cp:08x}"


# CJK ideographs, kana, and Hangul syllables: tokenized one character at a time.
_CJK = re.compile(
    r"["
    r"぀-ゟ"  # Hiragana
    r"゠-ヿ"  # Katakana
    r"㐀-䶿"  # CJK Extension A
    r"一-鿿"  # CJK Unified Ideographs
    r"가-힯"  # Hangul syllables
    r"豈-﫿"  # CJK Compatibility Ideographs
    r"\U00020000-\U000323af"  # CJK Extensions B-H/I + Compatibility Supplement
    r"]"
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
    for tok in _token_re().findall(_normalize(text)):
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
