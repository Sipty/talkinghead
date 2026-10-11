"""Edge placement: the one post-processing pass every cut list goes through.
Never clip a kept word; never leave a join inside a moan; land joins in
genuine quiet; no rapid-cut pileups."""

import numpy as np
import pytest

from talkinghead import edges, joins
from talkinghead.audio import FRAME_SEC, Envelope, RefinedWord
from talkinghead.edges import (
    EdgeContext,
    finalize_cuts,
    keep_splice_lead_in,
    protect_words,
    refine_joins,
    removed_words,
    settle_filler_edges,
    snap_to_valleys,
    steer_out_of_moans,
    sweep_to_quiet,
)
from talkinghead.timeline import Cut, Plan, Segment, SourceInfo

FLOOR = -50.0


def _env(seconds: float = 10.0, excursions: list[tuple[float, float]] = ()) -> Envelope:
    db = np.full(int(seconds / FRAME_SEC), FLOOR)
    for s, e in excursions:
        db[int(s / FRAME_SEC): int(e / FRAME_SEC)] = FLOOR + 7.0
    return Envelope(db, np.zeros_like(db), FRAME_SEC, 16000)


def _w(text: str, s: float, e: float) -> RefinedWord:
    return RefinedWord(text, s, e, s, e, 1.0)


def _ctx(words, moans=(), env=None, pad=0.05, voices=(), voice_guard=True, protected=()) -> EdgeContext:
    return EdgeContext(duration=10.0, env=env or _env(), floor=FLOOR, words=list(words),
                       moans=list(moans), voices=list(voices), voice_guard=voice_guard,
                       protected_spans=list(protected),
                       pad=pad, quiet_above_floor_db=6.0, flow_min_kept=2.0)


def test_silence_cut_retreats_from_every_touched_word() -> None:
    words = [_w("a", 0.9, 1.2), _w("b", 1.8, 2.3)]
    out = protect_words([Cut(1.0, 2.0, "silence")], _ctx(words))
    assert out == [Cut(pytest.approx(1.25), pytest.approx(1.75), "silence")]


def test_silence_cut_inside_a_word_is_dropped() -> None:
    assert protect_words([Cut(1.0, 1.1, "silence")], _ctx([_w("a", 0.9, 1.2)])) == []


def test_word_cut_snaps_outward_over_majority_covered_words_and_away_from_the_rest() -> None:
    words = [_w("keep", 0.5, 1.0), _w("um", 1.05, 1.4), _w("so", 1.45, 1.9), _w("next", 1.95, 2.5)]
    # Retake span given by ASR stamps: covers "um" fully, 60% of "so", 10% of "next".
    out = protect_words([Cut(1.05, 1.72, "retake", "um so")], _ctx(words))
    assert out[0].start == pytest.approx(1.05)
    assert out[0].start >= 1.0 + 0.03 - 1e-9          # never within splice_pad of "keep"
    assert out[0].end == pytest.approx(1.9)            # "so" removed whole
    assert out[0].end <= 1.95 - 0.03 + 1e-9            # "next" untouched


def test_join_never_lands_inside_a_moan_in_a_gap() -> None:
    words = [_w("a", 0.5, 1.0), _w("b", 3.5, 4.0)]
    out = steer_out_of_moans([Cut(1.1, 3.0, "silence")], _ctx(words, moans=[(2.9, 3.2)]))
    assert out[0].end == pytest.approx(3.25)  # swallowed the moan plus pad; still clear of "b"


def test_join_retreats_when_the_moan_is_glued_to_a_kept_word() -> None:
    words = [_w("a", 0.5, 1.0), _w("b", 3.5, 4.0)]
    out = steer_out_of_moans([Cut(1.1, 3.0, "silence")], _ctx(words, moans=[(2.9, 3.6)]))
    assert out[0].end == pytest.approx(2.85)  # the moan stays whole in the kept audio


def test_sweep_extends_over_weak_excursions_but_not_into_words() -> None:
    words = [_w("a", 1.0, 2.0), _w("b", 5.0, 6.0)]
    env = _env(excursions=[(3.1, 3.2), (4.97, 5.0)])
    out = sweep_to_quiet([Cut(2.5, 3.0, "silence")], _ctx(words, env=env))
    assert out[0].end == pytest.approx(3.2, abs=FRAME_SEC)
    out = sweep_to_quiet([Cut(2.5, 4.9, "silence")], _ctx(words, env=env))
    assert out[0].end <= 5.0 - 0.05 + 1e-9


def test_finalize_fuses_rapid_cuts_only_across_speechless_gaps() -> None:
    words = [_w("a", 0.0, 0.5), _w("b", 4.0, 4.5), _w("mid", 6.2, 6.5), _w("c", 9.0, 9.5)]
    cuts = [Cut(1.0, 2.0, "silence"), Cut(2.5, 3.5, "silence"), Cut(5.0, 6.0, "silence"), Cut(7.0, 8.0, "silence")]
    out = finalize_cuts(cuts, _ctx(words))
    assert [(round(c.start, 2), round(c.end, 2)) for c in out] == [(1.0, 3.5), (5.0, 6.0), (7.0, 8.0)]


def test_finalize_is_idempotent() -> None:
    words = [_w("a", 0.9, 1.2), _w("b", 1.8, 2.3), _w("c", 5.0, 5.5)]
    ctx = _ctx(words, moans=[(3.0, 3.2)], env=_env(excursions=[(2.4, 2.5)]))
    once = finalize_cuts([Cut(1.0, 2.0, "silence"), Cut(2.6, 3.1, "silence")], ctx)
    twice = finalize_cuts(once, ctx)
    assert twice == once


def test_finalize_drops_mechanical_slivers_below_the_minimum_useful_cut() -> None:
    # A 30 ms trim buys nothing and costs a join, whatever asked for it; a
    # filler shrunk to a sliver by its neighbours' pads is left in (the
    # accepted error) rather than cut.
    words = [_w("a", 0.0, 0.5), _w("b", 4.0, 4.5)]
    ctx = _ctx(words)
    ctx.min_cut = 0.08
    out = finalize_cuts([Cut(1.0, 1.03, "silence"), Cut(2.0, 2.03, "filler", "uh"), Cut(3.0, 3.5, "silence")], ctx)
    assert [(c.start, c.end) for c in out] == [(3.0, 3.5)]


def test_finalize_snaps_cut_edges_to_the_video_frame_grid_by_shrinking() -> None:
    # 30 fps: frames are 33.33 ms. Start rounds later, end rounds earlier, so a
    # snapped cut never removes audio the plan meant to keep; segments become
    # whole frames and the render cannot drift.
    words = [_w("a", 0.0, 0.5), _w("b", 4.0, 4.5)]
    ctx = _ctx(words)
    ctx.fps = 30
    (c,) = finalize_cuts([Cut(1.01, 2.99, "silence")], ctx)
    assert c.start == pytest.approx(31 / 30) and c.end == pytest.approx(89 / 30)
    assert abs(c.start * 30 - round(c.start * 30)) < 1e-6 and abs(c.end * 30 - round(c.end * 30)) < 1e-6


def test_zero_length_word_inside_a_word_cut_is_removed() -> None:
    # A token refined to a point (s == e, as whisper emits) carries no audio:
    # it can never be majority-covered, so left "kept" it would pin a phantom
    # hole mid-cut and split the head trim.
    words = [_w("a", 0.5, 1.0), _w("you.", 2.1, 2.1), _w("b", 3.5, 4.0)]
    out = protect_words([Cut(0.0, 3.0, "cold-open")], _ctx(words))
    assert out == [Cut(pytest.approx(0.0), pytest.approx(3.0), "cold-open")]


# --- waveform voice guard: audio the transcript does not cover ---

def test_silence_cut_splits_around_voiced_audio_the_transcript_never_named() -> None:
    """The reference recording had a 0.16 s utterance with no word anywhere in
    the transcript, so a pause trim spanning it was the only thing that could
    have eaten it. The waveform, not the word list, keeps it."""
    out = protect_words([Cut(4.0, 5.2, "silence")], _ctx([], voices=[(4.5, 4.7)], pad=0.15))
    assert [(round(c.start, 3), round(c.end, 3)) for c in out] == [(4.0, 4.35), (4.85, 5.2)]


def test_silence_cut_retreats_when_the_unnamed_voice_touches_its_edge() -> None:
    out = protect_words([Cut(4.0, 5.2, "silence")], _ctx([], voices=[(4.05, 4.2)], pad=0.15))
    assert [(round(c.start, 2), round(c.end, 2)) for c in out] == [(4.35, 5.2)]


def test_editorial_cut_is_reported_not_split_by_an_unnamed_voice() -> None:
    """A content cut is a decision about content: splitting it would ship a
    fragment of the passage the judge chose to remove. What it removes is
    measured instead (verify.unclassified_voice)."""
    out = protect_words([Cut(4.0, 5.2, "content")], _ctx([], voices=[(4.5, 4.7)], pad=0.15))
    assert [(round(c.start, 2), round(c.end, 2)) for c in out] == [(4.0, 5.2)]


def test_voice_guard_bounds_the_quiet_sweep() -> None:
    # The excursion runs to 4.6 and no word stops the sweep: without the unnamed
    # voice the edge would extend into it.
    env = _env(excursions=[(3.8, 4.6)])
    out = sweep_to_quiet([Cut(3.0, 3.5, "silence")], _ctx([], voices=[(4.0, 4.4)], pad=0.05, env=env))
    assert out[0].end == pytest.approx(3.95)


def test_voice_guard_off_restores_transcript_only_edges() -> None:
    out = protect_words([Cut(4.0, 5.2, "silence")], _ctx([], voices=[(4.5, 4.7)], pad=0.15, voice_guard=False))
    assert [(round(c.start, 2), round(c.end, 2)) for c in out] == [(4.0, 5.2)]


def test_finalize_never_fuses_cuts_across_voiced_audio_no_word_covers() -> None:
    """flow_guard fuses cuts that would leave too little speechless audio
    between them. Run at the step that actually ships, because fusing across
    unclassified voice deletes the very span the guard just kept."""
    cuts = [Cut(1.0, 2.0, "silence"), Cut(2.5, 3.5, "silence")]
    assert [(c.start, c.end) for c in finalize_cuts(cuts, _ctx([]))] == [(1.0, 3.5)]
    out = finalize_cuts(cuts, _ctx([], voices=[(2.2, 2.4)]))
    assert [(c.start, c.end) for c in out] == [(1.0, 2.0), (2.5, 3.5)]


# --- word-removing cuts: valley-placed edges and spliced-back quiet ---

LOUD = -20.0


def _levels(spans: list[tuple[float, float, float]]) -> Envelope:
    """Floor-level envelope with the given (start, end, dBFS) spans painted over it, in order."""
    db = np.full(int(10.0 / FRAME_SEC), FLOOR)
    for s, e, level in spans:
        db[round(s / FRAME_SEC): round(e / FRAME_SEC)] = level
    return Envelope(db, np.zeros_like(db), FRAME_SEC, 16000)


def _on_grid(t: float) -> bool:
    return abs(t * 30 - round(t * 30)) < 1e-6


def _speech_words(mid: str = "so") -> list[RefinedWord]:
    """Continuous speech 0.5-2.4 s: kept `mid`, a stamped filler, kept "next"."""
    return [_w(mid, 0.5, 1.3), _w("um", 1.3, 1.7), _w("next", 1.7, 2.4)]


def test_filler_edges_land_on_the_energy_valleys_the_stamps_missed() -> None:
    # The stamps put the cut 60 ms before the real dips; both edges end on them.
    env = _levels([(0.5, 2.4, LOUD), (1.35, 1.39, FLOOR), (1.61, 1.65, FLOOR)])
    (c,) = finalize_cuts([Cut(1.3, 1.7, "filler", "um")], _ctx(_speech_words(), env=env))
    assert c.start == pytest.approx(41 / 30) and c.end == pytest.approx(49 / 30)
    assert _on_grid(c.start) and _on_grid(c.end)


def test_mid_sentence_cut_takes_the_deepest_dip_and_keeps_no_extra_quiet() -> None:
    env = _levels([(0.5, 2.4, LOUD), (1.32, 1.35, LOUD - 15), (1.39, 1.415, FLOOR)])
    (c,) = finalize_cuts([Cut(1.3, 1.7, "content", "um")], _ctx(_speech_words(), env=env))
    assert c.start == pytest.approx(42 / 30)          # deepest, not the nearer shallow dip

    # No punctuation before the cut: real quiet on both sides is not handed back.
    words = [_w("so", 0.5, 1.0), _w("um", 1.5, 1.9), _w("next", 2.4, 3.0)]
    env = _levels([(0.5, 1.0, LOUD), (1.5, 1.9, LOUD), (2.4, 3.0, LOUD)])
    (c,) = finalize_cuts([Cut(1.0, 2.4, "retake", "um")], _ctx(words, env=env))
    assert c.start - 1.0 < 0.05 and 2.4 - c.end < 0.1   # only the splice lead-in (at most 60 ms) is handed back


def test_cut_after_a_completed_sentence_keeps_a_real_eighty_millisecond_tail() -> None:
    words = [_w("one.", 1.0, 1.5), _w("so", 1.9, 2.4), _w("two", 2.8, 3.3)]
    env = _levels([(1.0, 1.5, LOUD), (1.9, 2.4, LOUD), (2.8, 3.3, LOUD)])
    for reason in ("retake", "content"):
        ctx = _ctx(words, env=env)
        # Edges the earlier passes would leave right beside the kept word.
        (c,) = finalize_cuts([Cut(1.5, 2.8, reason, "so")], ctx)
        assert c.start >= 1.5 + 0.08 - 1e-9, reason
        assert c.start < 1.9                           # still in the pause, not in "so"
        assert _on_grid(c.start) and _on_grid(c.end)
        assert [w.text for w in removed_words(c, ctx.words)] == ["so"]
        assert finalize_cuts([c], ctx) == [c]          # re-placing settled edges moves nothing


def test_sentence_tail_is_a_floor_and_leaves_longer_natural_pauses_alone() -> None:
    words = [_w("one.", 1.0, 1.5), _w("so", 2.5, 3.0), _w("two", 3.4, 3.9)]
    env = _levels([(1.0, 1.5, LOUD), (2.5, 3.0, LOUD), (3.4, 3.9, LOUD)])
    (c,) = finalize_cuts([Cut(2.5, 3.4, "content", "so")], _ctx(words, env=env))
    assert c.start >= 2.4                              # the 1 s pause before "so" stays


def test_cut_after_a_clause_end_still_gets_the_clause_quiet_target() -> None:
    words = [_w("one,", 1.0, 1.5), _w("so", 1.9, 2.4), _w("two", 2.8, 3.3)]
    env = _levels([(1.0, 1.5, LOUD), (1.9, 2.4, LOUD), (2.8, 3.3, LOUD)])
    (c,) = finalize_cuts([Cut(1.5, 2.8, "content", "so")], _ctx(words, env=env))
    after_left, before_right = c.start - 1.5, 2.8 - c.end
    assert after_left + before_right == pytest.approx(0.18, abs=0.04)
    assert after_left > before_right                   # about 60/40


def test_retake_inside_a_sentence_joins_straight_to_the_correction() -> None:
    # "Claude Code [have it,] has it": 0.6 s of quiet on each side of the
    # abandoned attempt must not become a pause between "Code" and "has".
    words = [_w("Code", 0.8, 1.3), _w("have", 1.9, 2.2), _w("it,", 2.2, 2.5), _w("has", 3.1, 3.4)]
    env = _levels([(0.8, 1.3, LOUD), (1.9, 2.5, LOUD), (3.1, 3.4, LOUD)])
    for kept_before in ("Code", "Code,"):
        words[0] = _w(kept_before, 0.8, 1.3)
        ctx = _ctx(words, env=env)
        (c,) = finalize_cuts([Cut(1.9, 2.5, "retake", "have it,")], ctx)
        gap = (c.start - 1.3) + (3.1 - c.end)
        assert gap <= 2 * (0.03 + 1 / 30) + 1e-9, kept_before
        assert c.start >= 1.3 + 0.03 - 1e-9 and c.end <= 3.1 - 0.03 + 1e-9   # phonemes of both kept words stay
        assert [w.text for w in removed_words(c, ctx.words)] == ["have", "it,"]
        assert _on_grid(c.start) and _on_grid(c.end)


def test_retake_valley_never_moves_into_the_abandoned_word() -> None:
    # Quiet gap at -45 dB, a deeper dip 80 ms inside the first removed word and
    # another inside its last 80 ms: neither may become a join, or a fragment
    # of "have" survives in front of the correction.
    words = [_w("Code", 0.8, 1.3), _w("have", 1.9, 2.3), _w("has", 2.9, 3.2)]
    env = _levels([(0.8, 1.3, LOUD), (1.3, 1.9, -45.0), (1.9, 2.3, LOUD), (1.98, 2.02, FLOOR),
                   (2.2, 2.24, FLOOR), (2.3, 2.9, -45.0), (2.9, 3.2, LOUD)])
    for reason in ("retake", "restart"):
        (c,) = snap_to_valleys([Cut(1.9, 2.3, reason, "have")], _ctx(words, env=env))
        assert c.start <= 1.9 + 1e-9 and c.end >= 2.3 - 1e-9, reason


def test_retake_after_a_sentence_end_keeps_its_tail_and_the_natural_pause() -> None:
    words = [_w("Done.", 0.8, 1.3), _w("So", 2.0, 2.3), _w("So", 2.9, 3.2), _w("next", 3.2, 3.6)]
    env = _levels([(0.8, 1.3, LOUD), (2.0, 2.3, LOUD), (2.9, 3.6, LOUD)])
    (c,) = finalize_cuts([Cut(2.0, 2.3, "retake", "So")], _ctx(words, env=env))
    assert c.start >= 1.3 + 0.08 - 1e-9
    assert c.start - 1.3 > 0.5                         # a pause before the abandoned "So" is not a repair seam


def test_splice_quiet_is_bounded_by_the_quiet_the_source_has() -> None:
    # Only 50 ms of real quiet follows the kept word before the removed word's
    # voice starts: nothing is invented and the removed voice is not kept.
    words = [_w("one.", 1.0, 1.5), _w("so", 1.55, 2.4), _w("two", 2.8, 3.3)]
    env = _levels([(1.0, 1.5, LOUD), (1.55, 2.4, LOUD), (2.8, 3.3, LOUD)])
    (c,) = finalize_cuts([Cut(1.5, 2.8, "retake", "so")], _ctx(words, env=env))
    assert c.start <= 1.55 + 1e-9


def test_valley_snap_never_moves_an_edge_into_a_kept_words_voiced_run() -> None:
    # Deeper dips sit inside the kept words: the edges must stay outside them.
    env = _levels([(0.5, 2.4, LOUD), (1.22, 1.26, FLOOR), (1.75, 1.79, FLOOR)])
    (c,) = finalize_cuts([Cut(1.3, 1.7, "retake", "um")], _ctx(_speech_words(), env=env))
    assert c.start >= 1.3 + 0.03 - 1e-9 and c.end <= 1.7 - 0.03 + 1e-9


def test_valley_snap_never_uncovers_a_removed_word() -> None:
    # A deep dip lies far inside the filler: taking it would keep most of the word.
    env = _levels([(0.5, 2.4, LOUD), (1.61, 1.65, FLOOR)])
    ctx = _ctx(_speech_words(), env=env)
    ctx.splice_valley_window = 0.5
    (c,) = finalize_cuts([Cut(1.3, 1.7, "content", "um")], ctx)
    assert c.start <= 1.5 + 1e-9                       # at least half of "um" stays inside the cut
    assert [w.text for w in removed_words(c, ctx.words)] == ["um"]


def test_silence_cuts_keep_the_shared_pad_and_ignore_valleys_and_splice_quiet() -> None:
    words = [_w("a.", 0.5, 1.0), _w("b", 3.0, 3.5)]
    env = _levels([(0.5, 1.0, LOUD), (3.0, 3.5, LOUD), (1.3, 1.35, LOUD)])
    (c,) = finalize_cuts([Cut(1.0, 3.0, "silence")], _ctx(words, env=env, pad=0.15))
    assert c.start == pytest.approx(35 / 30) and c.end == pytest.approx(85 / 30)


def test_filler_in_continuous_voiced_speech_is_not_cut() -> None:
    env = _levels([(0.5, 2.4, LOUD)])
    ctx = _ctx(_speech_words(), env=env)
    ctx.filler_keep_voiced_max = 0.5                   # the 0.4 s "um" counts as short here
    assert finalize_cuts([Cut(1.3, 1.7, "filler", "um")], ctx) == []
    assert ctx.dropped_fillers == 1


def test_filler_with_a_dip_on_only_one_side_is_still_dropped() -> None:
    env = _levels([(0.5, 2.4, LOUD), (1.35, 1.39, FLOOR)])
    ctx = _ctx(_speech_words(), env=env)
    ctx.filler_keep_voiced_max = 0.5
    assert finalize_cuts([Cut(1.3, 1.7, "filler", "um")], ctx) == []


def test_a_filler_longer_than_the_limit_is_cut_despite_voiced_edges() -> None:
    env = _levels([(0.5, 2.4, LOUD)])
    ctx = _ctx(_speech_words(), env=env)
    ctx.filler_keep_voiced_max = 0.2
    assert len(finalize_cuts([Cut(1.3, 1.7, "filler", "um")], ctx)) == 1
    assert ctx.dropped_fillers == 0


def test_retake_with_voiced_edges_is_still_cut() -> None:
    env = _levels([(0.5, 2.4, LOUD)])
    ctx = _ctx(_speech_words(), env=env)
    assert len(finalize_cuts([Cut(1.3, 1.7, "retake", "um")], ctx)) == 1
    assert ctx.dropped_fillers == 0


def test_a_filler_with_voiced_edges_is_cut_by_default() -> None:
    # The default removes every filler; sparing voiced ones is opt-in.
    env = _levels([(0.5, 2.4, LOUD)])
    ctx = _ctx(_speech_words(), env=env)
    assert len(finalize_cuts([Cut(1.3, 1.7, "filler", "um")], ctx)) == 1
    assert ctx.dropped_fillers == 0


def test_edge_pass_uses_the_verifiers_voiced_edge_predicate() -> None:
    assert edges.edge_state is joins.edge_state


# --- protected audio events: whole laughs survive every cut ---

LAUGH = (2.013, 2.787)     # off the 30 fps grid on both ends


def _kept(cuts: list[Cut], duration: float = 10.0) -> list[tuple[float, float]]:
    """What the cuts leave of the source, as rendered: the complement of the cuts."""
    out, cursor = [], 0.0
    for c in sorted(cuts, key=lambda c: c.start):
        if c.start > cursor:
            out.append((cursor, c.start))
        cursor = max(cursor, c.end)
    if cursor < duration:
        out.append((cursor, duration))
    return out


def _survives(cuts: list[Cut], span: tuple[float, float]) -> bool:
    return any(s <= span[0] + 1e-9 and e >= span[1] - 1e-9 for s, e in _kept(cuts))


def _laugh_env() -> Envelope:
    """A loud 200 ms core inside a 774 ms laugh whose onset and decay sit near the floor."""
    return _levels([(0.3, 0.8, LOUD), (2.013, 2.3, -47.0), (2.3, 2.5, LOUD), (2.5, 2.787, -47.0),
                    (4.5, 5.0, LOUD)])


@pytest.mark.parametrize("reason", ["silence", "noise", "filler", "retake", "restart", "content",
                                    "join", "cold-open", "outro"])
def test_a_protected_laugh_survives_every_cut_reason_whole_after_frame_snapping(reason: str) -> None:
    words = [_w("so", 0.3, 0.8), _w("next", 4.5, 5.0)]
    ctx = _ctx(words, env=_laugh_env(), protected=[LAUGH])
    out = finalize_cuts([Cut(1.0, 4.0, reason, "x")], ctx)
    assert _survives(out, LAUGH)
    assert len(out) == 2 and out[0].end <= LAUGH[0] and out[1].start >= LAUGH[1]
    assert all(_on_grid(c.start) and _on_grid(c.end) for c in out)


def test_without_the_protected_span_the_same_cut_removes_the_laugh() -> None:
    words = [_w("so", 0.3, 0.8), _w("next", 4.5, 5.0)]
    out = finalize_cuts([Cut(1.0, 4.0, "silence")], _ctx(words, env=_laugh_env()))
    assert not _survives(out, LAUGH)


def test_a_cut_inside_or_equal_to_a_protected_event_is_dropped() -> None:
    ctx = _ctx([], protected=[(2.0, 2.8)])
    assert protect_words([Cut(2.1, 2.5, "noise")], ctx) == []
    assert protect_words([Cut(2.0, 2.8, "content")], ctx) == []


def test_a_retake_beside_a_laugh_keeps_its_quiet_onset_and_decay() -> None:
    # The laugh's loud core is 1.5-1.8 s; the protected event is the whole
    # 1.1-2.2 s. Without it the join goes to the quiet right behind the core.
    words = [_w("so", 0.5, 1.0), _w("um", 2.3, 2.6), _w("next", 3.0, 3.5)]
    env = _levels([(0.5, 1.0, LOUD), (1.1, 1.5, -47.0), (1.5, 1.8, LOUD), (1.8, 2.2, -47.0),
                   (2.3, 2.6, LOUD), (2.7, 2.95, -47.0), (3.0, 3.5, LOUD)])
    cut = Cut(2.3, 2.6, "retake", "um")
    (bare,) = finalize_cuts([cut], _ctx(words, env=env))
    assert bare.start < 2.2 and bare.end > 2.7        # the unprotected join eats both tails

    ahead, behind = (1.1, 2.2), (2.7, 2.95)
    (c,) = finalize_cuts([cut], _ctx(words, env=env, protected=[ahead, behind]))
    assert c.start >= ahead[1] and c.end <= behind[0]
    assert [w.text for w in removed_words(c, words)] == ["um"]


def test_overlapping_protected_events_and_voices_are_one_kept_span() -> None:
    # A laugh reported twice plus the unnamed voice of its tail: the cut around
    # them lands after all of it, not in the middle.
    ctx = _ctx([], protected=[(1.0, 1.6), (1.4, 2.0)], voices=[(1.9, 2.3)])
    out = protect_words([Cut(0.9, 3.5, "silence")], ctx)
    assert _survives(out, (1.0, 2.3))
    assert out[-1].start == pytest.approx(2.35)


def test_cuts_either_side_of_a_protected_event_are_never_fused_across_it() -> None:
    before, after = Cut(0.2, 0.9, "silence"), Cut(2.5, 3.5, "silence")
    fused = finalize_cuts([before, after], _ctx([]))
    assert len(fused) == 1                              # nothing between them: the pileup rule fuses
    kept = finalize_cuts([before, after], _ctx([], protected=[(1.0, 1.6), (1.4, 2.0)]))
    assert len(kept) == 2 and _survives(kept, (1.0, 2.0))


# --- join refinement for hand-authored plans, and filler voiced runs ---


def _two_segment_plan(cut: Cut, duration: float = 10.0) -> Plan:
    source = SourceInfo("main", "rec.mov", "sha", duration)
    return Plan(sources={"main": source}, main="main", cuts=[cut],
                segments=[Segment("main", 0.0, cut.start), Segment("main", cut.end, duration)])


def test_refine_joins_moves_an_edge_off_a_voiced_onset_into_the_quiet_before_it() -> None:
    # Quiet gap 0.9-1.6 then "next" at 1.6: the join was placed on the onset.
    words = [_w("so", 0.5, 0.9), _w("next", 1.6, 2.4)]
    env = _levels([(0.5, 0.9, LOUD), (1.6, 2.4, LOUD)])
    ctx = _ctx(words, env=env)
    plan = _two_segment_plan(Cut(0.9, 1.6, "content", ""))
    refined, (record,) = refine_joins(plan, ctx)
    assert record["moved"] is True
    assert record["cost_after"] < record["cost_before"]
    assert 1.5 < refined.segments[1].start < 1.6        # moved into the quiet, not past "next"'s pad
    assert [s.source for s in refined.segments] == ["main", "main"]


def test_refine_joins_keeps_a_join_already_in_quiet_unchanged() -> None:
    words = [_w("so", 0.5, 0.9), _w("next", 1.6, 2.4)]
    env = _levels([(0.5, 0.9, LOUD), (1.6, 2.4, LOUD)])
    plan = _two_segment_plan(Cut(1.0, 1.2, "content", ""))
    refined, (record,) = refine_joins(plan, _ctx(words, env=env))
    assert record["moved"] is False
    assert refined.segments == plan.segments


def test_refine_joins_keeps_both_kept_words_inside_their_pads() -> None:
    words = [_w("so", 0.5, 0.9), _w("next", 1.6, 2.4)]
    env = _levels([(0.5, 0.9, LOUD), (1.6, 2.4, LOUD)])
    ctx = _ctx(words, env=env)
    refined, _ = refine_joins(_two_segment_plan(Cut(0.9, 1.6, "content", "")), ctx)
    assert refined.segments[0].end >= 0.9 - ctx.splice_valley_window - 1e-9
    assert refined.segments[1].start >= 1.6 - ctx.splice_valley_window - 1e-9
    assert refined.segments[1].start <= 1.6 - ctx.splice_pad + 1e-9


def test_a_filler_on_voiced_audio_takes_its_voiced_run_into_the_cut() -> None:
    # "um" is voiced from 1.1 (quiet before it, 1.0-1.1). The cut starts 0.1 s into that run.
    words = [_w("so", 0.5, 1.0), _w("um", 1.2, 1.7), _w("next", 2.0, 2.4)]
    env = _levels([(0.5, 1.0, LOUD), (1.1, 2.4, LOUD)])
    ctx = _ctx(words, env=env)
    (settled,) = settle_filler_edges([Cut(1.2, 1.7, "filler", "um")], ctx)
    assert settled.start == pytest.approx(1.1, abs=1 / 30)
    assert not joins.edge_state(env, FLOOR, settled.start, "left")["voiced"]
    assert [w.text for w in removed_words(settled, ctx.words)] == ["um"]
    assert ctx.dropped_fillers == 0


def test_a_short_voiced_filler_is_dropped_only_when_the_drop_limit_allows_it() -> None:
    # Continuous speech: nothing quiet beside "um" to walk into, so its edges stay voiced.
    env = _levels([(0.5, 2.4, LOUD)])
    ctx = _ctx(_speech_words(), env=env)
    assert ctx.filler_keep_voiced_max == 0.0
    assert len(settle_filler_edges([Cut(1.3, 1.7, "filler", "um")], ctx)) == 1   # default keeps every filler
    ctx.filler_keep_voiced_max = 0.5
    assert settle_filler_edges([Cut(1.3, 1.7, "filler", "um")], ctx) == []
    assert ctx.dropped_fillers == 1


# --- splice lead-in: real source in front of a kept onset ---


def test_lead_in_moves_a_close_cut_end_back_through_quiet_to_the_onset() -> None:
    # Speech resumes 10 ms after the cut ends, with quiet before it: the end moves
    # back until 60 ms of real source precedes the onset.
    ctx = _ctx([_w("so", 0.5, 0.9), _w("next", 2.01, 2.5)])
    (c,) = keep_splice_lead_in([Cut(1.0, 2.0, "silence")], ctx)
    assert c.start == pytest.approx(1.0)
    assert 2.01 - c.end >= ctx.splice_lead_in - 1e-9
    assert _on_grid(c.end)


def test_lead_in_stops_at_voiced_audio_when_the_quiet_runs_out_first() -> None:
    # Voiced audio ends at 1.94 and the frame grid cannot reach the target: the end
    # comes back as far as the quiet goes, never into the voiced frames.
    ctx = _ctx([_w("next", 2.01, 2.5)], env=_levels([(1.5, 1.94, LOUD)]))
    (c,) = keep_splice_lead_in([Cut(1.0, 2.0, "silence")], ctx)
    assert 1.94 <= c.end < 2.0
    assert 2.01 - c.end < ctx.splice_lead_in


def test_lead_in_never_re_exposes_a_removed_word_before_the_onset() -> None:
    # "x" is removed up to 1.95: the quiet after it is handed back, the word is not.
    words = [_w("x", 1.2, 1.95), _w("next", 2.01, 2.5)]
    ctx = _ctx(words)
    (c,) = keep_splice_lead_in([Cut(1.0, 2.0, "content", "x")], ctx)
    assert 1.95 <= c.end < 2.0
    assert [w.text for w in removed_words(c, ctx.words)] == ["x"]


def test_finalize_gives_a_silence_cut_its_lead_in_after_the_frame_snap() -> None:
    # protect_words leaves 50 ms between the cut and "b"; the planner hands back quiet to reach 60 ms.
    (c,) = finalize_cuts([Cut(1.0, 2.0, "silence")], _ctx([_w("a", 0.0, 0.9), _w("b", 2.0, 2.5)]))
    assert 2.0 - c.end >= 0.06 - 1e-9
    assert _on_grid(c.start) and _on_grid(c.end)


def test_refine_joins_gives_the_incoming_speech_a_lead_in_through_quiet() -> None:
    # The join sits 10 ms before "next": the incoming segment starts earlier, through quiet only.
    words = [_w("so", 0.5, 0.9), _w("next", 1.61, 2.4)]
    env = _levels([(0.5, 0.9, LOUD), (1.61, 2.4, LOUD)])
    refined, _ = refine_joins(_two_segment_plan(Cut(0.9, 1.6, "content", "")), _ctx(words, env=env))
    assert refined.segments[0].end < refined.segments[1].start <= 1.61 - 0.06 + 1e-9
    assert [s.source for s in refined.segments] == ["main", "main"]


def test_refine_joins_never_lets_the_lead_in_re_expose_a_removed_word() -> None:
    # A retake whose removed "x" ends at 1.58, 20 ms before "next": the join stays at the word's end.
    words = [_w("so", 0.5, 0.9), _w("x", 1.0, 1.58), _w("next", 1.62, 2.4)]
    env = _levels([(0.5, 0.9, LOUD), (1.0, 1.58, LOUD), (1.62, 2.4, LOUD)])
    refined, _ = refine_joins(_two_segment_plan(Cut(0.9, 1.6, "retake", "x")), _ctx(words, env=env))
    assert 1.58 - 1e-9 <= refined.segments[1].start <= 1.6 + 1e-9
    assert [w.text for w in removed_words(refined.cuts[0], words)] == ["x"]
