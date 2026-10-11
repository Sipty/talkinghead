"""Retake markers: the spoken tells that precede a restatement.

Tiers come from the observed hit rate on the v10 source (see the v11 spec):
"sorry" and "hold on" marked a real retake in 5 of 6 occurrences, while
"I mean" marked one in 0 of 4 -- every occurrence was discourse filler or a
definition ("by vibecode, I mean, apply strong engineering practices"). So the
guidelines' default-to-cut bias applies to STRONG markers only; on WEAK markers
the standing "when unsure, keep" asymmetry stands.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

STRONG: tuple[str, ...] = (
    # Explicit announcements. On the v10 source he says all three of
    # "scrap that part", "I will do a retake here" and "I'll correct myself"
    # inside one 14 s stretch at 25:33 -- an unmissable retake that a
    # marker list without them scores as ordinary speech, because the only
    # other tell in the passage is a bare "actually".
    "let me say that again", "i will do a retake", "do a retake", "correct myself",
    "scratch that", "scrap that", "start over", "no actually", "no wait",
    "hold on", "sorry",
)
WEAK: tuple[str, ...] = ("i mean", "actually")
_PUNCT = re.compile(r"[^a-z0-9']+")


@dataclass(frozen=True)
class Candidate:
    word: int      # index of the marker's FIRST word
    marker: str
    tier: str      # "strong" | "weak"


def norm(text: str) -> str:
    """Lowercase, punctuation stripped: the transcript writes 'sorry,' and 'Sorry'."""
    return _PUNCT.sub("", text.lower())


def find_markers(words: list) -> list[Candidate]:
    """Every marker occurrence in the word stream, longest phrase winning."""
    toks = [norm(w.text) for w in words]
    phrases = [(p, "strong") for p in STRONG] + [(p, "weak") for p in WEAK]
    phrases.sort(key=lambda pt: -len(pt[0].split()))
    out: list[Candidate] = []
    taken: set[int] = set()
    for i in range(len(toks)):
        if i in taken:
            continue
        for phrase, tier in phrases:
            parts = phrase.split()
            if toks[i:i + len(parts)] == parts:
                out.append(Candidate(i, phrase, tier))
                taken.update(range(i, i + len(parts)))
                break
    return sorted(out, key=lambda c: c.word)
