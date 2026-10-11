"""Behavior tests for talkinghead.derive.derive_cuts: silence, filler, and restart rules."""

import pytest

from talkinghead.config import Config
from talkinghead.derive import derive_cuts, pause_boundary, pause_cap
from talkinghead.transcribe import Word


def words(*spans: tuple[str, float, float]) -> list[Word]:
    """Build words from (text, start, end) tuples."""
    return [Word(text=t, start=s, end=e, probability=1.0) for t, s, e in spans]


# --- Silence cuts from whisper word gaps (test-only fallback; CLI uses VAD) ---

def test_silence_gap_above_pause_above_is_cut() -> None:
    ws = words(("hello", 0.0, 1.0), ("world", 2.2, 3.0))
    cuts = derive_cuts(3.0, ws, pause_above=0.6)
    silence = [c for c in cuts if c.reason == "silence"]
    assert len(silence) == 1
    assert silence[0].start == pytest.approx(1.0)
    assert silence[0].end == pytest.approx(2.2)


def test_silence_gap_below_pause_above_not_cut() -> None:
    ws = words(("hello", 0.0, 1.0), ("world", 1.3, 2.0))
    assert derive_cuts(2.0, ws, pause_above=0.6) == []


def test_leading_silence_cut() -> None:
    ws = words(("hello", 1.0, 2.0))
    cuts = derive_cuts(2.5, ws, pause_above=0.6, pad=0.04)
    assert len(cuts) == 1
    assert cuts[0].reason == "silence"
    assert cuts[0].start == pytest.approx(0.0)
    assert cuts[0].end == pytest.approx(1.0 - 0.04)


def test_trailing_silence_cut() -> None:
    ws = words(("hello", 0.0, 1.0))
    cuts = derive_cuts(2.0, ws, pause_above=0.6, pad=0.04)
    assert len(cuts) == 1
    assert cuts[0].reason == "silence"
    assert cuts[0].start == pytest.approx(1.0 + 0.04)
    assert cuts[0].end == pytest.approx(2.0)


# --- Filler cuts: hesitation tokens long enough to disrupt flow ---

def test_long_filler_is_cut() -> None:
    # "uh" spans 0.3s: cut, padded by filler_pad.
    # Gaps around it are small (< pause_above), so no silence cuts interfere.
    ws = words(("hello", 0.0, 0.5), ("uh", 0.55, 0.85), ("world", 1.2, 1.8))
    cuts = derive_cuts(1.8, ws, pad=0.04)
    assert len(cuts) == 1
    assert cuts[0].reason == "filler"
    assert cuts[0].text == "uh"
    assert cuts[0].start == pytest.approx(0.55 - 0.06)
    assert cuts[0].end == pytest.approx(0.85 + 0.06)


def test_short_filler_between_words_is_cut_by_default() -> None:
    # Every named filler goes by default, however short: the padded cut takes it.
    ws = words(("hello", 0.0, 0.5), ("uh", 0.5, 0.6), ("world", 0.6, 1.2))
    cuts = derive_cuts(1.2, ws)
    assert [(c.reason, c.text) for c in cuts] == [("filler", "uh")]


def test_short_filler_below_a_configured_floor_is_kept() -> None:
    # min_filler_dur restores the older policy: a 0.1s "uh" under a 0.15s floor stays.
    ws = words(("hello", 0.0, 0.5), ("uh", 0.5, 0.6), ("world", 0.6, 1.2))
    assert derive_cuts(1.2, ws, min_filler_dur=0.15) == []


def test_long_filler_after_gap_only_is_cut() -> None:
    # No pause before the filler but the word is long enough: still cut.
    ws = words(("hello", 0.0, 0.5), ("um", 0.5, 0.8), ("world", 1.1, 1.5))
    cuts = derive_cuts(1.5, ws)
    assert [c.reason for c in cuts] == ["filler"]
    assert cuts[0].text == "um"
    assert cuts[0].start == pytest.approx(0.44)
    assert cuts[0].end == pytest.approx(0.86)


def test_multiword_filler_phrase_is_kept() -> None:
    # "you know" is fluent speech, not a hesitation token: never cut outright.
    ws = words(
        ("hello", 0.0, 0.5),
        ("you", 0.55, 0.65),
        ("know,", 0.65, 0.85),
        ("world", 1.1, 1.5),
    )
    assert derive_cuts(1.5, ws, pad=0.04) == []


def test_filler_cut_edges_snap_to_silence() -> None:
    # Filler at 0.9-1.2 with a silence boundary 0.05s before its padded start:
    # the start snaps onto the boundary.
    ws = words(("hello", 0.0, 0.5), ("uh", 0.9, 1.2), ("world", 1.6, 2.0))
    cuts = derive_cuts(2.0, ws, silences=[(0.95, 1.4)], pad=0.04)
    fillers = [c for c in cuts if c.reason == "filler"]
    assert len(fillers) == 1
    assert fillers[0].start == pytest.approx(0.95)  # snapped to silence start


def test_isolated_filler_cut_anchored_to_vad_burst() -> None:
    # ASR clips the hesitation's edges; the cut must cover the full voiced burst.
    ws = words(("Uh", 1.0, 1.3))
    cuts = derive_cuts(2.5, ws, silences=[(0.2, 0.95), (1.35, 2.4)], pad=0.04)
    assert any(c.start <= 0.95 and c.end >= 1.35 for c in cuts)


def test_mid_speech_filler_uses_word_time_fallback() -> None:
    # Filler inside a long burst (1.6s > filler span + 0.3): cut by word time,
    # not the whole burst (which would also remove real speech).
    ws = words(("hello", 0.0, 0.5), ("uh", 0.9, 1.2), ("world", 1.6, 2.0))
    cuts = derive_cuts(2.0, ws, silences=[(1.6, 2.0)], pad=0.04)
    fillers = [c for c in cuts if c.reason == "filler"]
    assert len(fillers) == 1
    assert fillers[0].start == pytest.approx(0.84)  # word start - filler_pad
    assert fillers[0].end == pytest.approx(1.26)  # word end + filler_pad, not burst end 1.6


# --- Restart detection ---

def test_restart_cut_removes_earlier_sentence() -> None:
    ws = words(
        ("let", 0.0, 0.2),
        ("me", 0.2, 0.4),
        ("explain", 0.4, 0.8),
        ("this.", 0.8, 1.0),
        ("let", 1.1, 1.3),
        ("me", 1.3, 1.5),
        ("explain", 1.5, 1.9),
        ("this.", 1.9, 2.1),
    )
    cuts = derive_cuts(2.5, ws)
    restarts = [c for c in cuts if c.reason == "restart"]
    assert len(restarts) == 1
    assert restarts[0].start == pytest.approx(0.0)  # padded start clipped at 0
    assert restarts[0].end == pytest.approx(1.0 + 0.04)
    assert restarts[0].text == "let me explain this."


def test_restart_not_triggered_below_similarity_threshold() -> None:
    ws = words(
        ("the", 0.0, 0.2),
        ("cat", 0.2, 0.4),
        ("sat", 0.4, 0.6),
        ("on", 0.6, 0.8),
        ("the", 0.8, 1.0),
        ("mat.", 1.0, 1.2),
        ("a", 1.2, 1.4),
        ("wholly", 1.4, 1.6),
        ("different", 1.6, 1.8),
        ("sentence", 1.8, 2.0),
        ("spoken", 2.0, 2.2),
        ("now.", 2.2, 2.4),
    )
    assert derive_cuts(2.5, ws, pause_above=0.6) == []


def test_short_first_sentence_never_counts_as_restart() -> None:
    # Fewer than 4 words in the earlier sentence: no restart even at similarity 1.0.
    ws = words(
        ("hi.", 0.0, 0.2),
        ("hi.", 0.2, 0.4),
    )
    assert derive_cuts(0.4, ws) == []


def test_overlapping_cuts_are_merged() -> None:
    # Filler cut (0.94, 1.46) overlaps the silence gap (1.4, 5.0): one fused span.
    ws = words(("hello", 0.0, 0.5), ("uh", 0.6, 1.0), ("world", 5.0, 5.9))
    cuts = derive_cuts(6.0, ws, pad=0.04)
    assert len(cuts) == 1
    assert cuts[0].reason == "filler"  # earlier (sorted) span's reason wins
    assert cuts[0].start == pytest.approx(0.54)
    assert cuts[0].end == pytest.approx(5.0)


def test_cuts_are_clipped_to_duration() -> None:
    # Restart-style padded start goes negative and must clip to 0.
    ws = words(("a", 0.0, 0.2), ("b", 0.2, 0.4), ("c", 0.4, 0.6), ("d.", 0.6, 0.8))
    cuts = derive_cuts(1.5, ws, pad=0.04)
    assert all(c.start >= 0.0 and c.end <= 1.5 for c in cuts)


def test_empty_words_whole_video_is_silence() -> None:
    cuts = derive_cuts(10.0, [])
    assert len(cuts) == 1
    assert cuts[0].reason == "silence"
    assert cuts[0].start == pytest.approx(0.0)
    assert cuts[0].end == pytest.approx(10.0)


def test_empty_words_short_video_no_cut() -> None:
    assert derive_cuts(0.5, [], pause_above=0.6) == []


# --- VAD-based silence cuts (silences= provided: waveform ground truth) ---

def test_vad_interior_silence_trimmed_to_boundary_cap() -> None:
    # Mid-sentence 2 s pause keeps its 0.40 s cap, split across both edges.
    ws = words(("hello", 0.0, 1.0), ("world", 4.0, 5.0))
    cuts = derive_cuts(5.0, ws, silences=[(1.0, 3.0)], pad=0.04)
    assert len(cuts) == 1
    assert cuts[0].reason == "silence"
    assert cuts[0].start == pytest.approx(1.2)
    assert cuts[0].end == pytest.approx(2.8)


def test_vad_trailing_silence_cut_reaches_duration() -> None:
    ws = words(("hello", 0.0, 1.0))
    cuts = derive_cuts(3.0, ws, silences=[(1.2, 3.0)], pad=0.04)
    assert len(cuts) == 1
    assert cuts[0].reason == "silence"
    assert cuts[0].start == pytest.approx(1.24)
    assert cuts[0].end == pytest.approx(3.0)


def test_vad_trailing_speech_preserved_even_if_whisper_ended_early() -> None:
    # Regression: whisper dropped a quiet trailing utterance ("yeah"), so its
    # last word ends at 1.0. VAD proves audio continues to 3.0 -> no trailing cut.
    ws = words(("hello", 0.0, 1.0))
    cuts = derive_cuts(3.0, ws, silences=[(1.2, 1.9)], pad=0.04)
    silence = [c for c in cuts if c.reason == "silence"]
    assert len(silence) == 1
    # 0.7 s raw keeps 0.30 + 0.25 * (0.7 - 0.35) = 0.3875 s, centred.
    assert silence[0].start == pytest.approx(1.39375)
    assert silence[0].end == pytest.approx(1.70625)


def test_vad_short_silence_below_pause_above_not_cut() -> None:
    ws = words(("hello", 0.0, 1.0), ("world", 2.0, 3.0))
    assert derive_cuts(3.0, ws, silences=[(1.0, 1.5)], pause_above=0.6) == []


# --- Glued fillers (whisper merged filler + pause into the next word) ---

def test_glued_filler_chunk_is_cut_with_its_pause() -> None:
    # "an" spans 1.0-2.2 but the audio is [uhh][0.6s pause][an]; probe hears "uhm".
    ws = words(("hello", 0.0, 0.5), ("an", 1.0, 2.2))
    cuts = derive_cuts(2.5, ws, silences=[(1.4, 2.0)], pad=0.04, probe=lambda s, e: "uhm")
    fillers = [c for c in cuts if c.reason == "filler"]
    assert len(fillers) == 1
    assert fillers[0].start == pytest.approx(0.96)
    assert fillers[0].end == pytest.approx(1.96)


def test_glued_real_word_chunk_is_not_cut() -> None:
    # Same structure but the pre-silence audio is a real word ("that"): probe
    # says so, and only the (long enough) silence itself is trimmed.
    ws = words(("video", 0.0, 1.0), ("that", 0.5, 4.0))
    cuts = derive_cuts(4.5, ws, silences=[(1.4, 3.0)], pad=0.04, probe=lambda s, e: "that")
    assert [c.reason for c in cuts] == ["silence"]
    assert cuts[0].start == pytest.approx(1.6)   # 1.6 s raw, "that" is mid: keeps 0.40
    assert cuts[0].end == pytest.approx(2.8)


def test_glued_filler_silence_below_max_filler_gap_is_kept() -> None:
    ws = words(("hello", 0.0, 0.5), ("an", 1.0, 2.0))
    # Internal silence 1.2-1.4 is only 0.2s < max_filler_gap 0.3: not glue.
    assert derive_cuts(2.5, ws, silences=[(1.2, 1.4)], probe=lambda s, e: "uhm") == []

# --- Interior pause policy: kept length by raw length and boundary ---

def _kept(text: str, raw: float, *, end: float = 1.0) -> float:
    """Kept pause length for a VAD silence of `raw` seconds after `text`."""
    ws = words((text, 0.0, end), ("next", 1.0 + raw, 2.0 + raw))
    cuts = derive_cuts(3.0 + raw, ws, silences=[(1.0, 1.0 + raw)], pad=0.0)
    silence = [c for c in cuts if c.reason == "silence"]
    return raw if not silence else raw - (silence[0].end - silence[0].start)


def test_kept_pause_grows_with_raw_length_until_the_cap() -> None:
    kept = [round(_kept("and", raw), 6) for raw in (0.4, 0.5, 0.6, 1.2, 2.0)]
    assert kept == sorted(kept)
    assert kept[0] < kept[1] < kept[2] < kept[3] == kept[-1] == pytest.approx(0.40)


def test_kept_pause_per_boundary() -> None:
    assert _kept("and", 1.2) == pytest.approx(0.40)        # mid: cap 0.40
    assert _kept("done.", 1.25) == pytest.approx(0.525)    # sentence: growth below cap 0.55
    assert _kept("done.", 3.0) == pytest.approx(0.80)      # topic
    assert _kept("so,", 2.0) == pytest.approx(0.45)        # clause


def test_kept_pause_never_exceeds_raw_minus_margin() -> None:
    ws = words(("done.", 0.0, 1.0), ("next", 1.25, 2.0))
    cuts = derive_cuts(3.0, ws, silences=[(1.0, 1.25)], pause_above=0.2, pad=0.0)
    assert 0.25 - (cuts[0].end - cuts[0].start) == pytest.approx(0.23)


def test_pause_at_exactly_pause_above_is_untouched() -> None:
    ws = words(("done.", 0.0, 1.0), ("next", 1.5, 2.0))
    assert derive_cuts(3.0, ws, silences=[(1.0, 1.5)], pause_above=0.5, pad=0.0) == []


def test_boundary_word_is_the_last_one_that_started_before_the_silence() -> None:
    # "done." is stamped to 1.6, inside the silence 1.0-3.0; it still starts
    # first, so the pause is a sentence boundary.
    ws = words(("so", 0.0, 0.5), ("done.", 0.5, 1.6), ("next", 2.4, 4.0))
    cuts = derive_cuts(5.0, ws, silences=[(1.0, 2.4)], pad=0.0)
    assert 1.4 - (cuts[0].end - cuts[0].start) == pytest.approx(0.55)  # sentence cap, not mid 0.40


def test_boundary_and_cap_classification() -> None:
    cfg = Config()
    assert pause_boundary("done?", 1.0, cfg) == "sentence"
    assert pause_boundary("done!", 1.5, cfg) == "topic"
    assert pause_boundary('done."', 1.0, cfg) == "sentence"
    assert pause_boundary("so;", 3.0, cfg) == "clause"
    assert pause_boundary("so-", 0.5, cfg) == "clause"
    assert pause_boundary("and", 3.0, cfg) == "mid"
    assert pause_cap("topic", cfg) > pause_cap("sentence", cfg) > pause_cap("clause", cfg) > pause_cap("mid", cfg)
