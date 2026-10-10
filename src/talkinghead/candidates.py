"""Candidates: every span a judge may be asked about is generated here, by
code, with a stable id. The judge answers per id; it never invents a span.

Reuses the calibrated detectors that now sit beside this module (derive:
pauses, fillers, restarts; noise: smacks/breaths/throat clears). Moans
(bursts) are NOT candidates: they are constraints for edge placement (see
edges.py).
"""

from __future__ import annotations

from dataclasses import dataclass

from .derive import derive_cuts
from .semantic import Sentence
from .transcribe import Word

KIND_PREFIX = {"silence": "P", "filler": "F", "restart": "R", "noise": "N"}


@dataclass(frozen=True)
class Candidate:
    id: str
    start: float
    end: float
    kind: str
    text: str = ""

    @property
    def length(self) -> float:
        return self.end - self.start


def sentence_groups(
    words: list[Word], split_pause: float | None = None, burst_last: list[bool] | None = None
) -> list[list[int]]:
    """Word indices per sentence: a word carrying . ! ? ends a sentence, and so
    does a gap of >= split_pause before the next word (the ASR forgets
    punctuation in rehearsal; on the real source it glued 77 s into one
    "sentence", which made the true start unselectable). When `burst_last`
    is given, a pause split is only allowed after a word whose voice actually
    stops: a stamp gap inside a continuous burst is ASR slop, not a pause."""
    groups: list[list[int]] = []
    current: list[int] = []
    for i, w in enumerate(words):
        current.append(i)
        ends = any(e in w.text for e in (".", "!", "?"))
        if not ends and split_pause is not None and i + 1 < len(words):
            ends = words[i + 1].start - w.end >= split_pause and (burst_last is None or burst_last[i])
        if ends:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def sentence_table(
    words: list[Word], split_pause: float | None = None, burst_last: list[bool] | None = None
) -> list[Sentence]:
    """Sentences with zero-padded ids S0000...; the unit of LLM judgment."""
    return [
        Sentence(sid=f"S{n:04d}", start=words[g[0]].start, end=words[g[-1]].end,
                 text=" ".join(words[i].text for i in g))
        for n, g in enumerate(sentence_groups(words, split_pause, burst_last))
    ]


def _number(cands: list[Candidate]) -> list[Candidate]:
    counters: dict[str, int] = {}
    out: list[Candidate] = []
    for c in sorted(cands, key=lambda c: (c.start, c.end, c.kind)):
        n = counters.get(c.kind, 0)
        counters[c.kind] = n + 1
        out.append(Candidate(f"{KIND_PREFIX[c.kind]}{n:04d}", c.start, c.end, c.kind, c.text))
    return out


def machinery_candidates(
    duration: float,
    words: list[Word],
    silences: list[tuple[float, float]],
    *,
    pause_above: float = 0.35,
    pause: object | None = None,
    min_filler_dur: float = 0.15,
) -> list[Candidate]:
    """Pause trims, fillers and restarts from the calibrated derive pass.

    derive_cuts merges touching spans across reasons; we keep its output
    span-for-span and label each by the reason it reports, so an id points
    at exactly one removable span.
    """
    cuts = derive_cuts(
        duration, words, silences=silences, pause_above=pause_above, pause=pause,
        min_filler_dur=min_filler_dur, pad=0.0,
    )
    cands = [
        Candidate("", c.start, c.end, c.reason if c.reason in KIND_PREFIX else "silence", c.text)
        for c in cuts
    ]
    return _number(cands)


def noise_candidates(noise_cuts) -> list[Candidate]:
    """Mouth-noise spans (from noise.detect_mouth_noises) as candidates."""
    return _number([Candidate("", c.start, c.end, "noise", c.text) for c in noise_cuts])


def cold_open_candidates(sents: list[Sentence], *, window_sec: float = 300.0) -> list[Sentence]:
    """Every sentence starting inside the head window can be the true start.

    The judge reads the whole transcript anyway; filtering candidates by a
    pause heuristic once left the real opener off the list and the judge said
    so in its answer. Sentences are already split at pauses (sentence_groups)."""
    return [s for s in sents if s.start <= window_sec]


def outro_candidates(sents: list[Sentence], duration: float, *, window_sec: float = 120.0) -> list[Sentence]:
    """Sentences ending inside the tail window: where the video could end."""
    return [s for s in sents if s.end >= duration - window_sec]
