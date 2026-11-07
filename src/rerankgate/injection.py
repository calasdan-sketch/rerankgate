"""A tripwire for instruction-like text inside retrieved documents (indirect prompt injection).

Retrieved text is untrusted: a poisoned page can say "ignore previous instructions ...". This
module flags the crudest, most common phrasings so you can log them, review them, or drop the
chunk before it reaches a model. It is a HEURISTIC: it is trivially bypassed by rephrasing,
other languages, or spacing tricks, and it is not a defence. Treat a hit as a signal, never a
silent "clean" as safety. Matching is on NFKC-normalised, case-folded text with whitespace and
zero-width characters collapsed, so spacing and full-width tricks do not hide the common
phrasings.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# ruff: noqa: E501  (the patterns read best on one line each)
_PATTERNS: dict[str, re.Pattern[str]] = {
    name: re.compile(rx)
    for name, rx in {
        "ignore_instructions": r"\b(ignore|disregard|forget|override)\b.{0,30}\b(previous|prior|above|earlier|all|any|your|the)\b.{0,20}\b(instructions?|prompts?|rules|context|guidelines)\b",
        "role_override": r"\b(you are now|from now on you|act as|pretend (to be|you are)|new instructions?:)",
        "reveal_prompt": r"\b(reveal|print|show|repeat|leak)\b.{0,25}\b(system|hidden|initial)\b.{0,10}\b(prompt|instructions?|message)\b",
        "conceal_from_user": r"\b(do not|don't|never)\b.{0,15}\b(tell|inform|mention|reveal)\b.{0,15}\b(the )?(user|human)\b",
        "chat_template_tokens": r"(<\|im_start\|>|<\|im_end\|>|<\|system\|>|\[/?inst\]|<<sys>>|^\s*system\s*:)",
        "exfiltrate": r"\b(send|post|upload|email|exfiltrate)\b.{0,40}\b(to|at)\b.{0,10}(https?://|www\.)",
    }.items()
}
_MAX_SCAN = 20_000
_INVISIBLE = re.compile("[​-‏⁠⁦-⁩‪-‮﻿­]")


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).casefold()
    t = _INVISIBLE.sub("", t)
    return re.sub(r"\s+", " ", t)


@lru_cache(maxsize=8192)
def _flag(text: str) -> tuple[str, ...]:
    t = _norm(text[:_MAX_SCAN])  # bounded work on hostile giant chunks
    return tuple(name for name, rx in _PATTERNS.items() if rx.search(t))


def flag_instructions(text: str) -> list[str]:
    """Names of the heuristic patterns that match ``text`` (empty list = no hit, NOT safe)."""
    return list(_flag(text))
