"""Build the committed, seeded benchmark corpus + labelled queries.

Corpus  : docstrings of ~60 Python standard-library modules (real technical English).
Answerable queries   : 3-5 content words sampled from a *different* sentence of a doc
                       than its title line (so it is not a trivial name match).
Unanswerable queries : (a) "foreign-domain": same style, built from numpy/scipy/sklearn
                       docstrings that are NOT in the corpus (shares generic vocabulary such
                       as "array", "return", "function" -> hard); (b) "off-topic": hand-written
                       everyday questions; (c) "hybrid": half the words from an in-corpus doc,
                       half from a foreign doc (partial overlap, the classic RAG failure).

These are SYNTHETIC labels (we know which doc a query was drawn from), not human relevance
judgements. The generated files are committed so results do not depend on the Python or
library versions of whoever re-runs the benchmark. Regenerate with:
    python benchmarks/build_dataset.py
"""

from __future__ import annotations

import importlib
import inspect
import json
import pathlib
import random
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from rerankgate.tokenize import content_tokens  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "data"
SEED = 20261107

STD = (
    "argparse bisect calendar collections colorsys configparser contextlib copy csv dataclasses "
    "datetime decimal difflib email enum fnmatch fractions functools getpass glob gzip "
    "hashlib heapq "
    "hmac html http imaplib inspect io ipaddress itertools json locale logging math mimetypes "
    "operator pathlib pickle platform pprint queue random re secrets shlex shutil smtplib socket "
    "sqlite3 statistics string struct subprocess tarfile tempfile textwrap threading time timeit "
    "traceback typing unicodedata urllib uuid warnings weakref zipfile zlib"
).split()
FOREIGN = [
    "numpy.linalg", "numpy.fft", "numpy.random", "numpy.polynomial", "scipy.optimize",
    "scipy.stats", "scipy.signal", "scipy.sparse", "scipy.interpolate", "scipy.integrate",
    "sklearn.cluster", "sklearn.linear_model", "sklearn.metrics", "sklearn.preprocessing",
    "sklearn.ensemble", "sklearn.decomposition", "skimage.filters", "skimage.morphology",
]  # fmt: skip
OFF_TOPIC = [
    "how long should I roast a chicken", "who won the world cup in 2010",
    "best hiking trails near Banff", "how do I train for a half marathon",
    "what is the capital of Mongolia", "recipe for gluten free banana bread",
    "how to repot a fiddle leaf fig", "symptoms of seasonal allergies",
    "cheap flights to Lisbon in March", "how does compound interest work on a mortgage",
    "tips for sleeping on a long flight", "what rhymes with orange",
    "how do vaccines train the immune system", "how to tune a ukulele",
    "history of the Roman aqueducts", "how to negotiate a rent increase",
    "what causes the northern lights", "how many calories in an avocado",
    "best way to learn to swim as an adult", "how to remove a red wine stain",
    "origin of the word quarantine", "how deep is the Mariana trench",
    "what to pack for a camping trip", "how do tides work",
    "can dogs eat grapes", "how to write a wedding toast",
    "difference between baking soda and baking powder", "how do I fold a fitted sheet",
    "why is the sky blue at noon", "how to start a vegetable garden in clay soil",
]  # fmt: skip
_SENT = re.compile(r"(?<=[.!?])\s+")


def _docs(mods: list[str], limit_per_mod: int) -> list[dict]:
    out: list[dict] = []
    for name in mods:
        try:
            mod = importlib.import_module(name)
        except Exception:  # noqa: BLE001 - optional libs
            continue
        items = [(name, inspect.getdoc(mod) or "")]
        for attr in sorted(dir(mod)):
            if attr.startswith("_"):
                continue
            obj = getattr(mod, attr, None)
            if inspect.isfunction(obj) or inspect.isclass(obj):
                if getattr(obj, "__module__", "").split(".")[0] != name.split(".")[0]:
                    continue
                items.append((f"{name}.{attr}", inspect.getdoc(obj) or ""))
        taken = 0
        for did, doc in items:
            text = re.sub(r"\s+", " ", doc).strip()
            if len(text.split()) < 25:
                continue
            out.append({"id": did, "text": text[:900]})
            taken += 1
            if taken >= limit_per_mod:
                break
    return out


_DF: dict[str, int] = {}


def _sample_query(text: str, rng: random.Random) -> str | None:
    sents = [s for s in _SENT.split(text) if len(content_tokens(s)) >= 4]
    if len(sents) < 2:
        return None
    sent = rng.choice(sents[1:])  # not the title line
    words = []
    for w in re.findall(r"[A-Za-z][A-Za-z\-]+", sent):
        if content_tokens(w) and w.lower() not in words:
            words.append(w.lower())
    if len(words) < 4:
        return None
    # keep the rarer words (a person asks with the distinctive terms, not "the call")
    words.sort(key=lambda w: (_DF.get(w, 0), rng.random()))
    k = rng.randint(3, min(5, len(words)))
    pick = set(words[:k])
    ordered = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]+", sent) if w.lower() in pick]
    seen: list[str] = []
    for w in ordered:
        if w.lower() not in seen:
            seen.append(w.lower())
    return " ".join(seen)


def main() -> None:
    rng = random.Random(SEED)
    corpus = _docs(STD, 6)
    foreign = _docs(FOREIGN, 6)
    for d in corpus + foreign:
        for w in set(re.findall(r"[a-z][a-z\-]+", d["text"].lower())):
            _DF[w] = _DF.get(w, 0) + 1
    cids = {d["id"] for d in corpus}
    assert not cids & {d["id"] for d in foreign}
    qs: list[dict] = []
    for d in rng.sample(corpus, min(len(corpus), 300)):
        q = _sample_query(d["text"], rng)
        if q:
            qs.append(
                {"query": q, "relevant_ids": [d["id"]], "answerable": True, "kind": "answerable"}
            )
    n_ans = len(qs)
    foreign_q = []
    for d in rng.sample(foreign, min(len(foreign), 140)):
        q = _sample_query(d["text"], rng)
        if q:
            foreign_q.append(
                {"query": q, "relevant_ids": [], "answerable": False, "kind": "foreign"}
            )
    qs += foreign_q
    for q in OFF_TOPIC:
        qs.append({"query": q, "relevant_ids": [], "answerable": False, "kind": "off_topic"})
    hyb = 0
    while hyb < 100:
        a, b = rng.choice(corpus), rng.choice(foreign)
        qa, qb = _sample_query(a["text"], rng), _sample_query(b["text"], rng)
        if not qa or not qb:
            continue
        wa, wb = qa.split(), qb.split()
        q = " ".join(wa[: max(2, len(wa) // 2)] + wb[: max(2, len(wb) // 2)])
        qs.append({"query": q, "relevant_ids": [], "answerable": False, "kind": "hybrid"})
        hyb += 1
    rng.shuffle(qs)
    OUT.mkdir(exist_ok=True)
    with open(OUT / "corpus.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for d in corpus:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(OUT / "queries.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for q in qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"corpus={len(corpus)} answerable={n_ans} unanswerable={len(qs) - n_ans}")


if __name__ == "__main__":
    main()
