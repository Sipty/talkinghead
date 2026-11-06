"""Natural jump cuts: no pause-only cut of min_pause_cut seconds or less, no kept segment under min_segment_sec,
the output opens on its first word, and the retake prompt starts an abandoned attempt at the sentence its restart
repeats."""

import pytest

from talkinghead import verify
from talkinghead.audio import RefinedWord
from talkinghead.config import Config
from talkinghead.derive import derive_cuts
from talkinghead.edges import EdgeContext, _kept_between, enforce_min_segments, trim_opening
from talkinghead.prompts import retake_prompt
from talkinghead.stages import polish
from talkinghead.timeline import Cut, Plan, SourceInfo
from talkinghead.transcribe import Word
from talkinghead.verify import dead_air_excess

from .test_polish import W, _ctx

# --- VAD pause trims (derive) ----------------------------------------------------------------------------------------

def _pause_cuts(pause: float, cfg: Config) -> list:
    """One mid-sentence silence of `pause` seconds between two words. A mid pause keeps at most pause_cap_mid (0.40)."""
    start = 1.0
    ws = [Word(text="hello", start=0.0, end=start, probability=1.0),
          Word(text="world", start=start + pause, end=start + pause + 1.0, probability=1.0)]
    return derive_cuts(start + pause + 1.0, ws, silences=[(start, start + pause)], pause=cfg, pad=0.04)


def test_a_pause_trimmed_by_exactly_min_pause_cut_is_kept_whole() -> None:
    # A 1.00 s mid pause keeps 0.40 s, so the trim would remove exactly 0.60 s: no cut.
    assert _pause_cuts(1.0, Config()) == []


def test_a_pause_trimmed_by_more_than_min_pause_cut_is_cut_to_its_boundary_cap() -> None:
    cuts = _pause_cuts(1.01, Config())
    assert len(cuts) == 1
    assert cuts[0].end - cuts[0].start == pytest.approx(0.61)   # 1.01 s pause less the 0.40 s kept


def test_a_short_pause_is_kept_whole_under_the_default() -> None:
    assert _pause_cuts(0.7, Config()) == []                      # the old policy would have trimmed 0.31 s


def test_min_pause_cut_zero_restores_every_trim_the_policy_asks_for() -> None:
    cuts = _pause_cuts(0.7, Config(min_pause_cut=0.0))
    assert len(cuts) == 1
    assert cuts[0].end - cuts[0].start == pytest.approx(0.7 - 0.3875)   # kept = pause_keep + ratio * (raw - above)


# --- Gap tightening (polish) -----------------------------------------------------------------------------------------

def _sentence_gap(gap: float) -> list[Cut]:
    """Two words either side of a `gap`-second silence after a sentence end (cap polish_gap_sentence, 0.30)."""
    words = [W("alpha.", 0.0, 1.0), W("beta", 1.0 + gap, 2.0 + gap)]
    return polish.gap_cuts(_ctx(words), [], [])


def test_a_gap_whose_trim_is_exactly_min_pause_cut_is_kept_whole() -> None:
    assert _sentence_gap(0.9) == []                              # 0.90 - 0.30 kept = 0.60 removed


def test_a_gap_trimmed_by_more_than_min_pause_cut_is_capped_to_its_boundary() -> None:
    cuts = _sentence_gap(0.91)
    assert len(cuts) == 1
    assert cuts[0].length == pytest.approx(0.61)


# --- Kept-segment minimum (edges) ------------------------------------------------------------------------------------

def _edge_ctx(words: list[RefinedWord] | None = None, min_segment: float = 0.5, **kw) -> EdgeContext:
    return EdgeContext(duration=10.0, env=None, floor=-50.0, words=words or [], min_segment=min_segment, **kw)


def _spans(cuts: list[Cut]) -> list[tuple[float, float]]:
    return [(round(a, 3), round(b, 3)) for a, b in _kept_between(cuts, 10.0)]


def test_a_room_tone_sliver_is_absorbed_into_the_neighbouring_cut() -> None:
    ctx = _edge_ctx()
    out = enforce_min_segments([Cut(1.0, 3.0, "retake"), Cut(3.2, 5.0, "retake")], ctx)
    assert _spans(out) == [(0.0, 1.0), (5.0, 10.0)]
    assert ctx.min_segment_fixes["absorbed"] == 1


def test_a_kept_segment_of_049_seconds_is_repaired_and_050_is_left_alone() -> None:
    repaired = enforce_min_segments([Cut(1.0, 3.0, "retake"), Cut(3.49, 5.0, "retake")], _edge_ctx())
    assert _spans(repaired) == [(0.0, 1.0), (5.0, 10.0)]
    untouched_ctx = _edge_ctx()
    untouched = enforce_min_segments([Cut(1.0, 3.0, "retake"), Cut(3.5, 5.0, "retake")], untouched_ctx)
    assert len(untouched) == 2
    assert untouched_ctx.min_segment_fixes["absorbed"] == 0


def test_a_sliver_holding_a_kept_word_restores_the_adjacent_pause() -> None:
    ctx = _edge_ctx([W("hi", 3.05, 3.15)])
    out = enforce_min_segments([Cut(1.0, 3.0, "retake"), Cut(3.2, 5.0, "silence")], ctx)
    assert _spans(out) == [(0.0, 1.0), (3.0, 10.0)]              # 'hi' now sits in a kept run past the pause
    assert ctx.min_segment_fixes["restored"] == 1


def test_a_sliver_with_a_word_between_two_word_cuts_is_reported_and_not_reopened() -> None:
    ctx = _edge_ctx([W("hi", 3.05, 3.15)])
    out = enforce_min_segments([Cut(1.0, 3.0, "retake"), Cut(3.2, 5.0, "filler")], ctx)
    assert len(out) == 2                                         # the abandoned attempt is not reopened
    assert ctx.min_segment_fixes["unfixed"] == [[3.0, 3.2]]


def test_a_hand_cut_is_never_changed_by_the_sliver_repair() -> None:
    hand = Cut(1.0, 3.0, "retake", "", {"hand": True})
    ctx = _edge_ctx()
    out = enforce_min_segments([hand, Cut(3.2, 5.0, "retake")], ctx)
    assert any(c.provenance.get("hand") and (c.start, c.end) == (1.0, 3.0) for c in out)
    assert ctx.min_segment_fixes["unfixed"] == [[3.0, 3.2]]


def test_min_segment_zero_disables_the_repair() -> None:
    cuts = [Cut(1.0, 3.0, "retake"), Cut(3.2, 5.0, "retake")]
    assert enforce_min_segments(cuts, _edge_ctx(min_segment=0.0)) == cuts


# --- Output opening (edges) ------------------------------------------------------------------------------------------

def _plan_with(cuts: list[Cut]) -> Plan:
    return Plan.from_cuts(SourceInfo("main", "/v/x.mp4", "sha", 10.0), cuts)


def _open_ctx(words: list[RefinedWord], **kw) -> EdgeContext:
    return EdgeContext(duration=10.0, env=None, floor=-50.0, words=words, pad=0.15, min_segment=0.0, **kw)


def test_silence_before_the_first_word_of_the_output_is_cut_back_to_the_pad() -> None:
    plan = _plan_with([Cut(0.0, 2.0, "cold-open")])
    out = trim_opening(plan, _open_ctx([W("hi", 2.74, 3.0)]))
    assert out.segments[0].start == pytest.approx(2.59)
    assert [(round(c.start, 2), round(c.end, 2)) for c in out.cuts] == [(0.0, 2.59)]


def test_the_opening_trim_follows_a_reordered_block_that_now_leads_the_output() -> None:
    plan = _plan_with([Cut(5.0, 6.0, "retake")])
    plan.segments.insert(0, plan.segments.pop(-1))              # a later block moved to the front
    word = W("open", 7.0, 7.4)
    out = trim_opening(plan, _open_ctx([word]))
    assert out.segments[0].start == pytest.approx(6.85)
    assert out.segments[0].end == plan.segments[0].end


def test_head_silence_holding_a_laugh_is_left_alone() -> None:
    plan = _plan_with([Cut(0.0, 2.0, "cold-open")])
    out = trim_opening(plan, _open_ctx([W("hi", 2.74, 3.0)], protected_spans=[(2.2, 2.5)]))
    assert out is plan


def test_a_head_word_already_near_the_start_is_untouched() -> None:
    plan = _plan_with([])
    assert trim_opening(plan, _open_ctx([W("hi", 0.2, 0.5)])) is plan


# --- verify check ----------------------------------------------------------------------------------------------------

def test_verify_fails_a_plan_with_a_kept_segment_under_the_minimum() -> None:
    result = verify._min_segment(_plan_with([Cut(1.0, 3.0, "retake"), Cut(3.49, 10.0, "retake")]),
                                 {"min_segment_sec": 0.5})
    assert not result.ok
    assert result.severity == "fatal"


def test_verify_passes_a_kept_segment_of_exactly_the_minimum() -> None:
    result = verify._min_segment(_plan_with([Cut(1.0, 3.0, "retake"), Cut(3.5, 10.0, "retake")]),
                                 {"min_segment_sec": 0.5})
    assert result.ok


# --- dead-air allowance ----------------------------------------------------------------------------------------------

def test_a_pause_the_policy_keeps_whole_is_not_dead_air() -> None:
    # Mid run: cap 0.40 + min_pause_cut 0.60 + slack 0.10 = 1.10 s allowed before any excess.
    cfg = Config()
    assert dead_air_excess([(1.0, 2.0)], [], cfg)[0]["excess"] == 0.0
    assert dead_air_excess([(1.0, 2.2)], [], cfg)[0]["excess"] == pytest.approx(0.1)


# --- config ----------------------------------------------------------------------------------------------------------

def test_defaults_and_validation() -> None:
    cfg = Config()
    assert cfg.min_pause_cut == 0.6
    assert cfg.min_segment_sec == 0.5
    with pytest.raises(ValueError):
        Config(min_pause_cut=-0.1).validate()


# --- retake prompt ---------------------------------------------------------------------------------------------------

def test_the_retake_prompt_starts_an_abandoned_attempt_at_the_sentence_a_restart_repeats() -> None:
    prompt = retake_prompt("[w0001] And then you should have presentations.", None)
    assert "Where the abandoned attempt starts" in prompt
    assert "And then you should have presentations." in prompt
    assert "the range starts at its first word, not at the restart" in prompt
