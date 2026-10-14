"""Removed words leave no audible fragment: the cut edge moves to a real boundary, or the word is given back."""

import numpy as np

from talkinghead.audio import FRAME_SEC, Envelope, RefinedWord
from talkinghead.edges import EdgeContext, take_removed_whole
from talkinghead.timeline import Cut

FLOOR = -50.0
LOUD = -20.0


def _w(text: str, s: float, e: float) -> RefinedWord:
    return RefinedWord(text, s, e, s, e, 1.0)


def _levels(spans: list[tuple[float, float, float]]) -> Envelope:
    db = np.full(int(10.0 / FRAME_SEC), FLOOR)
    for s, e, level in spans:
        db[int(s / FRAME_SEC): int(e / FRAME_SEC)] = level
    return Envelope(db, np.zeros_like(db), FRAME_SEC, 16000)


def _ctx(words, env) -> EdgeContext:
    return EdgeContext(duration=10.0, env=env, floor=FLOOR, words=list(words), moans=[], voices=[],
                       voice_guard=True, protected_spans=[], pad=0.05, quiet_above_floor_db=6.0,
                       flow_min_kept=2.0)


def _settle(cuts: list[Cut], words: list[RefinedWord], env: Envelope):
    """Runs the fragment step with the word at index 1 ("um") removed; returns (cuts, ctx)."""
    ctx = _ctx(words, env)
    out = take_removed_whole(cuts, ctx, {id(words[1])})
    return out, ctx


def test_gap_case_moves_the_edge_into_the_silence_so_the_whole_word_goes() -> None:
    # "keep" ends at 1.0; silence runs to 1.2 where "um" (1.2-1.6) starts. The cut begins at 1.3, so 100 ms of
    # "um" stays audible. The edge moves into the silence; "keep" is not touched.
    words = [_w("keep", 0.5, 1.0), _w("um", 1.2, 1.6)]
    out, ctx = _settle([Cut(1.3, 1.6, "filler", "um")], words, _levels([(0.5, 1.0, LOUD), (1.2, 1.6, LOUD)]))
    assert len(out) == 1 and 1.0 < out[0].start <= 1.2 + 1e-9
    assert out[0].end == 1.6
    assert ctx.fragments["gap"] == 1 and ctx.given_back == []


def test_touching_case_places_the_edge_in_the_valley_between_the_words() -> None:
    # "keep" (0.5-1.2) touches "um" (1.2-1.6): no silence, only a dip at 1.18-1.20 between them.
    words = [_w("keep", 0.5, 1.2), _w("um", 1.2, 1.6)]
    out, ctx = _settle([Cut(1.3, 1.6, "filler", "um")], words,
                       _levels([(0.5, 1.18, LOUD), (1.21, 1.6, LOUD)]))
    assert len(out) == 1 and 1.17 <= out[0].start <= 1.21
    assert ctx.fragments["touch"] == 1 and ctx.given_back == []


def test_no_valley_gives_the_word_back_whole_and_counts_it() -> None:
    # Speech runs unbroken from "keep" into "um": no quiet point, so no boundary leaves "um" fragment-free.
    words = [_w("keep", 0.5, 1.2), _w("um", 1.2, 1.6)]
    out, ctx = _settle([Cut(1.3, 1.6, "filler", "um")], words, _levels([(0.5, 1.6, LOUD)]))
    assert out == []
    assert ctx.fragments["given_back"] == 1 and ctx.given_back == [(1.2, 1.6)]


def test_a_boundary_that_would_clip_the_kept_word_gives_the_word_back() -> None:
    # The only dip lies 0.3 s inside "keep": taking it would clip far more than the 10% tolerance.
    words = [_w("keep", 0.5, 1.2), _w("um", 1.2, 1.6)]
    env = _levels([(0.5, 0.9, LOUD), (0.9, 0.95, FLOOR), (0.95, 1.6, LOUD)])
    out, ctx = _settle([Cut(1.3, 1.6, "filler", "um")], words, env)
    assert out == []
    assert ctx.fragments["given_back"] == 1 and ctx.given_back == [(1.2, 1.6)]
