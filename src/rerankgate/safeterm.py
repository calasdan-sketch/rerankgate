"""Make untrusted text safe to print to a terminal.

Corpus text, chunk ids and queries come from files the user may not control.
Raw control characters (ANSI escape sequences, carriage returns, backspaces)
can rewrite what the terminal shows, and bidi overrides / zero-width
characters can make text read differently from what it is ("Trojan Source").
`clean()` replaces every Unicode control (Cc) and format (Cf) character —
which covers all of those — and lone surrogates with a visible ``\\uXXXX`` escape.
"""

from __future__ import annotations

import unicodedata


def clean(text: object, *, keep_newlines: bool = False) -> str:
    """Return `text` with control/format characters escaped visibly."""
    s = str(text)
    out: list[str] = []
    for ch in s:
        if keep_newlines and ch == "\n":
            out.append(ch)
            continue
        if ch in ("\n", "\t", "\r"):
            out.append(" ")
            continue
        if unicodedata.category(ch) in ("Cc", "Cf", "Cs"):
            cp = ord(ch)
            out.append(f"\\u{cp:04x}" if cp <= 0xFFFF else f"\\U{cp:08x}")
            continue
        out.append(ch)
    return "".join(out)
