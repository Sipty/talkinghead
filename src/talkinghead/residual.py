"""Disfluencies the finished render still contains, found on its own transcript.

Scribe on the raw recording merges a stutter into one word ("I, I, I" comes back as one "I"), so the
source transcript misses many of them. The render's transcript is what the viewer hears, so this module
scans it for three things: filler sounds, a 1-3 word phrase said twice in a row, and a word restarted
after a fragment. `scan` is a pure function of the words, so the output pass and the final check see the
same findings. `to_source` maps a finding's output span back to the main source through the plan.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Sequence

from .timeline import Plan, Segment

# A filler sound, spelled out or stretched ("uh", "umm", "hmm", "mm").
FILLER = re.compile(r"^(?:u+h+m*|u+m+|e+r+m*|h+m+|m{2,})$")
# Longest repeated phrase looked for, in words.
MAX_REPEAT = 3
# A word restarted without a hyphen: its stem is 2-4 letters and the next word starts with it.
RESTART_STEM = (2, 4)
HYPHENS = ("-", "\u2013", "\u2014")
# Output times closer than this count as the same instant when mapping through the plan.
EPS = 1e-6


def norm(text: str) -> str:
    """Lower case, curly apostrophes straightened, everything but letters, digits and apostrophes dropped."""
    return re.sub(r"[^a-z0-9']", "", text.lower().replace("\u2019", "'"))


@dataclass(frozen=True)
class Finding:
    """One disfluency in the output: the words a cut would remove, and the word that stays after them.

    `first`..`last` are indices into the transcript; `heard` is the removed words as written. `start`..`end`
    is the output span a removal takes: from the first removed word to the instant the kept word begins.
    `follows` is False when nothing is kept after the span (a filler at the very end), so its end is the
    filler's own end.
    """

    kind: str  # "filler" | "repeat" | "restart"
    first: int
    last: int
    start: float
    end: float
    follows: bool
    heard: str  # the removed words as the transcript wrote them
    key: str  # identity across renders: the kind and the normalised words, so a re-render still matches

    def to_dict(self) -> dict:
        return asdict(self)


def _span_text(words: Sequence, lo: int, hi: int) -> str:
    return " ".join(words[i].text.strip() for i in range(lo, hi + 1))


def _same_speaker(words: Sequence, idx: list[int]) -> bool:
    return len({words[i].speaker for i in idx}) <= 1


def _repeat_at(words: Sequence, toks: list[tuple[int, str]], p: int) -> int:
    """How many words the phrase starting at token `p` repeats (0 when it does not repeat)."""
    for n in range(1, MAX_REPEAT + 1):
        if p + 2 * n > len(toks):
            break
        first = [toks[k][1] for k in range(p, p + n)]
        second = [toks[k][1] for k in range(p + n, p + 2 * n)]
        if first == second and _same_speaker(words, [toks[k][0] for k in range(p, p + 2 * n)]):
            return n
    return 0


def _restart_at(words: Sequence, toks: list[tuple[int, str]], p: int) -> bool:
    """Whether token `p` is a fragment that the next token restarts: "adver- advertising", "F- Facebook"."""
    if p + 1 >= len(toks):
        return False
    stem, nxt = toks[p][1], toks[p + 1][1]
    if not stem or not nxt.startswith(stem):
        return False
    if words[toks[p][0]].text.strip().endswith(HYPHENS):
        return True
    return RESTART_STEM[0] <= len(stem) <= RESTART_STEM[1] and len(nxt) > len(stem)


def _finding(words: Sequence, kind: str, toks: list[tuple[int, str]], p: int, n: int) -> Finding:
    """The finding whose removed words are toks[p]..toks[p+n-1]; the kept word is the next token, if any."""
    first, last = toks[p][0], toks[p + n - 1][0]
    heard = _span_text(words, first, last)
    # A filler is identified by its own sound; a repeat or restart also by the copy that stays after it.
    copies = {"filler": 0, "restart": 1, "repeat": n}[kind]
    if p + n < len(toks):
        end, follows = words[toks[p + n][0]].start, True
    else:
        end, follows = words[last].end, False
    key = " ".join([kind] + [toks[k][1] for k in range(p, min(len(toks), p + n + copies))])
    return Finding(kind, first, last, words[first].start, end, follows, heard, key)


def scan(words: Sequence) -> list[Finding]:
    """Every filler, repeated phrase and restart in the transcript, in time order.

    A repeat removes its first copy and the gap before the second copy, so chained repeats ("I, I, I") give
    adjacent findings that do not overlap. Words are compared after normalisation; punctuation-only tokens
    are skipped. A repeat never crosses a speaker change.
    """
    toks = [(i, norm(w.text)) for i, w in enumerate(words)]
    toks = [(i, t) for i, t in toks if t]
    out: list[Finding] = []
    p = 0
    while p < len(toks):
        t = toks[p][1]
        if FILLER.match(t):
            out.append(_finding(words, "filler", toks, p, 1))
            p += 1
            continue
        n = _repeat_at(words, toks, p)
        if n:
            out.append(_finding(words, "repeat", toks, p, n))
            p += n
            continue
        if _restart_at(words, toks, p):
            out.append(_finding(words, "restart", toks, p, 1))
        p += 1
    return out


def to_source(plan: Plan, start: float, end: float) -> tuple[float, float] | None:
    """The main-source span of an output span, or None when the mapping is ambiguous.

    A span may cross a join. Where the join runs forward in the source (a cut sits between the two sides), the
    span's two pieces meet through material the plan already removed, so one source span covers both pieces and
    the removed gap in between. A join that runs backward (a moved block) has no such span: the source order breaks,
    so the caller reports the finding instead of cutting. None also when the span leaves the main source.
    """
    spans: list[tuple[Segment, float]] = []  # (segment, output time where it starts)
    offset = 0.0
    for seg in plan.segments:
        stop = offset + seg.length
        if stop > start + EPS and offset < end - EPS:
            spans.append((seg, offset))
        offset = stop
    if not spans or any(seg.source != plan.main for seg, _ in spans):
        return None
    first, first_at = spans[0]
    last, last_at = spans[-1]
    if first_at > start + EPS or last_at + last.length < end - EPS:
        return None
    if any(b.start < a.end - EPS for (a, _), (b, _) in zip(spans, spans[1:])):
        return None
    return first.start + (start - first_at), last.start + (end - last_at)
