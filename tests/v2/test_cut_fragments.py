"""Removed words leave no audible fragment: the cut edge moves to a real boundary, or the word is given back."""

import numpy as np

from talkinghead import edges as E
from talkinghead.audio import FRAME_SEC, Envelope, RefinedWord
from talkinghead.edges import SPEAKER_KEY, EdgeContext, finalize_cuts, take_removed_whole
from talkinghead.timeline import Cut

FLOOR = -50.0
LOUD = -20.0


def _w(text: str, s: float, e: float) -> RefinedWord:
    return RefinedWord(text, s, e, s, e, 1.0)


def _ws(text: str, s: float, e: float, asr_s: float, asr_e: float, speaker: str) -> RefinedWord:
    return RefinedWord(text, s, e, asr_s, asr_e, 1.0, speaker=speaker)


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


def test_same_speaker_asr_overlap_resolves_to_the_word_edge_or_gives_the_word_back() -> None:
    # "so" (S0, refined 0.5-1.3) has an ASR stamp reaching 1.32, so its jitter overlaps "um" (S0, 1.3-1.7).
    # The cut starts at 1.36, leaving 60 ms of "um" in the output. The overlap is 20 ms of "so" (within 10%):
    # the edge moves to the word's own edge, and no scrap remains.
    words = [_ws("so", 0.5, 1.3, 0.5, 1.32, "S0"), _ws("um", 1.3, 1.7, 1.3, 1.7, "S0")]
    out, ctx = _settle([Cut(1.36, 1.7, "filler", "um")], words, _levels([(0.5, 1.7, LOUD)]))
    assert ctx.fragments["overlap"] == 1 and ctx.given_back == []
    assert len(out) == 1 and out[0].start <= 1.3 + 1e-9

    # Overlap beyond the tolerance (200 ms of "so", 20% of its extent): no edge fits, so "um" goes back whole.
    words = [_ws("so", 0.5, 1.3, 0.5, 1.5, "S0"), _ws("um", 1.3, 1.7, 1.3, 1.7, "S0")]
    out, ctx = _settle([Cut(1.36, 1.7, "filler", "um")], words, _levels([(0.5, 1.7, LOUD)]))
    assert ctx.fragments["given_back"] == 1 and ctx.given_back == [(1.3, 1.7)]
    assert not any(c.end > 1.3 and c.start < 1.7 for c in out)


def test_a_word_with_a_far_kept_word_before_it_is_cut_whole_at_a_quiet_point_in_the_gap() -> None:
    # "so" ends at 1.0; silence runs to 1.9 where "um" (1.9-2.3) starts. The cut begins at 2.0, so 100 ms of "um"
    # stays audible. The gap is 0.9 s: the edge moves to its quiet point, and "so" is not touched.
    words = [_w("so", 0.5, 1.0), _w("um", 1.9, 2.3)]
    out, ctx = _settle([Cut(2.0, 2.3, "filler", "um")], words, _levels([(0.5, 1.0, LOUD), (1.9, 2.3, LOUD)]))
    assert len(out) == 1 and 1.0 < out[0].start <= 1.9 + 1e-9
    assert not any(c.end > 0.5 and c.start < 1.0 for c in out)
    assert ctx.fragments["gap"] == 1 and ctx.given_back == []


def test_refine_join_guard_flags_a_cut_that_leaves_removed_audio() -> None:
    # refine_joins rejects a join move whose cut leaves a removed word's audio in the output (_leaves_fragment).
    word = _w("um", 1.3, 1.7)
    assert E._leaves_fragment(Cut(1.4, 1.7, "filler", "um"), [word])      # 100 ms of "um" left in the output
    assert not E._leaves_fragment(Cut(1.3, 1.7, "filler", "um"), [word])  # the whole word is cut
    assert not E._leaves_fragment(Cut(1.33, 1.67, "filler", "um"), [word])  # a splice margin under 40 ms


def test_a_filler_fused_into_another_speakers_silence_cut_still_removes_its_word() -> None:
    # The silence cut belongs to S0 and overlaps the filler "uh" (S1, 1.2-1.5), which its own cut removes. The
    # filler is not another speaker's word to spare: the silence cut must not be trimmed back and keep it.
    words = [_ws("so", 0.5, 1.0, 0.5, 1.0, "S0"), _ws("uh", 1.2, 1.5, 1.2, 1.5, "S1"),
             _ws("next", 1.9, 2.4, 1.9, 2.4, "S0")]
    silence = Cut(1.0, 1.9, "silence", "", {SPEAKER_KEY: "S0"})
    filler = Cut(1.2, 1.5, "filler", "uh")
    ctx = _ctx(words, _levels([(0.5, 2.4, LOUD)]))
    out = finalize_cuts([silence, filler], ctx)
    covered = sum(max(0.0, min(1.5, c.end) - max(1.2, c.start)) for c in out)
    assert covered >= 0.25   # "uh" (0.3 s) goes out up to the splice margin
