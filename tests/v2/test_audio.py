"""Audio analysis primitives: numpy envelope (parity with the calibrated
pure-Python one), noise floor, and acoustic word-boundary refinement."""

import math

import numpy as np
import pytest

from talkinghead.audio import (
    FRAME_SEC,
    Envelope,
    envelope,
    filter_masked,
    noise_floor,
    refine_word_bounds,
    unclassified_voice,
)
from talkinghead.noise import VAD_FRAME_SEC
from talkinghead.noise import _envelope as reference_envelope
from talkinghead.transcribe import Word

SR = 16000


def test_envelope_matches_the_calibrated_reference_implementation() -> None:
    rng = np.random.default_rng(7)
    samples = rng.integers(-3000, 3000, size=SR * 2, dtype=np.int16)
    samples[1000:1400] = 0                      # digital silence -> -120 dB frame
    samples[5000:5010] = 0                      # zeros inside a frame: sign carries over
    env = envelope(samples, SR, FRAME_SEC)
    ref_db, ref_zcr = reference_envelope(samples.tolist(), SR, FRAME_SEC)
    assert len(env.db) == len(ref_db)
    assert np.allclose(env.db, ref_db, atol=1e-9)
    assert np.allclose(env.zcr, ref_zcr, atol=1e-12)


def test_noise_floor_is_the_25th_percentile_and_guards_implausible_audio() -> None:
    db = np.linspace(-80.0, -20.0, 100)
    assert noise_floor(db) == pytest.approx(np.sort(db)[25])
    with pytest.raises(RuntimeError, match="implausible noise floor"):
        noise_floor(np.full(100, -3.0))


def _tone(n: int, amp: int = 8000) -> np.ndarray:
    t = np.arange(n) / SR
    return (amp * np.sin(2 * math.pi * 220 * t)).astype(np.int16)


def _speech_like(spans: list[tuple[float, float]], seconds: float = 6.0) -> np.ndarray:
    """Quiet floor (+-8) with loud tone bursts exactly on the given spans."""
    x = np.random.default_rng(1).integers(-8, 9, size=int(seconds * SR)).astype(np.int16)
    for s, e in spans:
        a, b = int(s * SR), int(e * SR)
        x[a:b] = _tone(b - a)
    return x


def test_refine_word_bounds_snaps_to_acoustic_onset_and_offset() -> None:
    # True voice spans; ASR stamps the start 60 ms late and the end 120 ms early.
    true = [(1.000, 1.500), (2.000, 2.400), (3.000, 3.800)]
    env = envelope(_speech_like(true), SR, FRAME_SEC)
    words = [Word(f"w{i}", s + 0.06, e - 0.12, 1.0) for i, (s, e) in enumerate(true)]
    refined = refine_word_bounds(words, env)
    for r, (s, e) in zip(refined, true):
        assert r.start == pytest.approx(s, abs=FRAME_SEC * 2)
        assert r.end == pytest.approx(e, abs=FRAME_SEC * 2)
        assert r.confidence == 1.0


def test_refine_word_bounds_never_crosses_a_neighbouring_word() -> None:
    # Two words with no quiet between them: one continuous burst 1.0-2.0.
    env = envelope(_speech_like([(1.0, 2.0)]), SR, FRAME_SEC)
    words = [Word("a", 1.05, 1.40, 1.0), Word("b", 1.45, 1.90, 1.0)]
    a, b = refine_word_bounds(words, env)
    assert a.end <= b.start
    assert a.start == pytest.approx(1.0, abs=FRAME_SEC * 2)   # onset is unambiguous
    assert b.end == pytest.approx(2.0, abs=FRAME_SEC * 2)     # offset is unambiguous
    # The shared boundary had no acoustic evidence: ASR stamps kept, low confidence.
    assert a.end == pytest.approx(1.40) and b.start == pytest.approx(1.45)
    assert a.confidence == 0.75 and b.confidence == 0.75


def test_refine_word_bounds_extends_over_a_voiced_tail_rather_than_clipping_it() -> None:
    # The word runs 1.0-1.4, then a quieter voiced tail (a hesitation) to 1.7.
    x = _speech_like([(1.0, 1.4)])
    a, b = int(1.4 * SR), int(1.7 * SR)
    x[a:b] = _tone(b - a, amp=1200)
    env = envelope(x, SR, FRAME_SEC)
    (r,) = refine_word_bounds([Word("w", 1.05, 1.30, 1.0)], env)
    assert r.end == pytest.approx(1.7, abs=FRAME_SEC * 2)


def test_envelope_exposes_time_helpers() -> None:
    env = envelope(np.zeros(SR, dtype=np.int16), SR, FRAME_SEC)
    assert isinstance(env, Envelope)
    assert env.frame_sec == FRAME_SEC
    assert env.frame_at(0.5) == int(0.5 / FRAME_SEC)
    assert env.duration == pytest.approx(1.0)


def test_refine_word_bounds_unfolds_trailing_silence_from_sentence_final_words() -> None:
    # Parakeet folds trailing silence into sentence-final words ("Hello." stamped
    # 15.12-20.80 with an acoustic end at 15.80). The refined end must be where
    # the voice stops, not where the stamp says.
    env = envelope(_speech_like([(1.0, 1.7), (6.5, 7.0)], seconds=8.0), SR, FRAME_SEC)
    words = [Word("Hello.", 1.0, 6.0, 1.0), Word("Are", 6.5, 7.0, 1.0)]
    hello, _ = refine_word_bounds(words, env)
    assert hello.end == pytest.approx(1.7, abs=FRAME_SEC * 2)
    assert hello.confidence == 1.0


def test_refine_word_bounds_does_not_split_a_word_at_a_plosive_closure() -> None:
    # A 60 ms closure inside the word (a "t") dips to the floor but the word continues.
    x = _speech_like([(1.0, 1.3), (1.36, 1.7)])
    env = envelope(x, SR, FRAME_SEC)
    (w,) = refine_word_bounds([Word("later", 1.05, 1.6, 1.0)], env)
    assert w.end == pytest.approx(1.7, abs=FRAME_SEC * 2)


def test_refine_word_bounds_handles_an_early_start_stamp() -> None:
    # Parakeet stamped "Hello." from 15.12 while the voice runs 15.46-15.82 (real
    # data). The refined span must be the voiced burst, never a zero-length word
    # sitting in the quiet before it.
    env = envelope(_speech_like([(1.46, 1.82), (2.6, 3.0)], seconds=5.0), SR, FRAME_SEC)
    hello, nxt = refine_word_bounds([Word("Hello.", 1.12, 2.5, 1.0), Word("Let's", 2.6, 3.0, 1.0)], env)
    assert hello.start == pytest.approx(1.46, abs=FRAME_SEC * 2)
    assert hello.end == pytest.approx(1.82, abs=FRAME_SEC * 2)
    assert hello.end - hello.start > 0.3


def test_refine_word_bounds_keeps_stamps_for_ghost_words_in_silence() -> None:
    env = envelope(_speech_like([(1.0, 1.5)], seconds=6.0), SR, FRAME_SEC)
    real, ghost = refine_word_bounds([Word("a", 1.0, 1.5, 1.0), Word("No,", 4.0, 4.2, 1.0)], env)
    assert (ghost.start, ghost.end) == (4.0, 4.2) and ghost.confidence == 0.0


def test_refine_word_bounds_never_yields_negative_or_zero_length_words() -> None:
    # Real-data failure: a word stamped BEFORE the burst it attaches to, being the
    # burst's first but not last word, got start=burst start and end=own stamp end.
    env = envelope(_speech_like([(2.0, 3.0), (4.0, 4.5)], seconds=6.0), SR, FRAME_SEC)
    words = [Word("Are", 1.70, 1.90, 1.0), Word("you", 2.30, 2.50, 1.0), Word("tired", 2.50, 3.10, 1.0),
             Word("of", 4.0, 4.5, 1.0)]
    refined = refine_word_bounds(words, env)
    for r in refined:
        assert r.end - r.start >= 0.05 - 1e-9, r
    for a, b in zip(refined, refined[1:]):
        assert a.end <= b.start + 1e-9
    rng = np.random.default_rng(3)
    for _ in range(50):
        stamps = np.sort(rng.uniform(0.5, 5.5, size=8))
        words = [Word(f"w{i}", float(s), float(s) + float(rng.uniform(0.05, 0.4)), 1.0) for i, s in enumerate(stamps)]
        refined = refine_word_bounds(words, env)
        assert all(r.end - r.start >= 0.05 - 1e-9 for r in refined)
        assert all(a.end <= b.start + 1e-9 for a, b in zip(refined, refined[1:]))


def test_refined_words_know_whether_they_end_a_burst() -> None:
    env = envelope(_speech_like([(1.0, 2.0), (3.0, 3.5)], seconds=5.0), SR, FRAME_SEC)
    a, b, c = refine_word_bounds([Word("a", 1.0, 1.4, 1.0), Word("b", 1.5, 2.0, 1.0), Word("c", 3.0, 3.5, 1.0)], env)
    assert not a.burst_last and b.burst_last and c.burst_last


# --- waveform arbitration: voiced, speech-level audio no word covers ---

def _with_fragment(amp: int) -> tuple[Envelope, list]:
    """Two stamped words with a separate burst between them at `amp`."""
    x = _speech_like([(1.0, 1.4), (2.0, 2.4)])
    a, b = int(1.6 * SR), int(1.84 * SR)
    x[a:b] = _tone(b - a, amp=amp)
    env = envelope(x, SR, FRAME_SEC)
    words = refine_word_bounds([Word("a", 1.0, 1.4, 1.0), Word("b", 2.0, 2.4, 1.0)], env)
    return env, words


def test_unclassified_voice_finds_a_spoken_fragment_no_stamp_covers() -> None:
    """The reference recording's defect in miniature: an utterance at speech
    level between two stamped words that the transcript never carried."""
    env, words = _with_fragment(amp=3000)
    assert unclassified_voice(env, noise_floor(env.db), words) == [
        (pytest.approx(1.6, abs=FRAME_SEC * 2), pytest.approx(1.84, abs=FRAME_SEC * 2))]


def test_unclassified_voice_ignores_audio_far_below_speech_level() -> None:
    # A breath at -38 dBFS clears the noise floor but is not speech: only the
    # level gate keeps a quiet recording's room tone out of the guard.
    env, words = _with_fragment(amp=400)
    assert unclassified_voice(env, noise_floor(env.db), words) == []


def test_unclassified_voice_ignores_stamped_words_and_already_claimed_audio() -> None:
    env, words = _with_fragment(amp=3000)
    # Covered by a word's own acoustic span: the transcript knows it.
    assert unclassified_voice(env, noise_floor(env.db), words, exclude=[]) != []
    assert unclassified_voice(env, noise_floor(env.db), words, exclude=[(1.6, 1.84)]) == []
    covered = [Word("a", 1.0, 1.4, 1.0), Word("drop", 1.6, 1.84, 1.0), Word("b", 2.0, 2.4, 1.0)]
    assert unclassified_voice(env, noise_floor(env.db), refine_word_bounds(covered, env)) == []


def test_filter_masked_drops_only_spans_a_speech_model_calls_non_speech() -> None:
    # A model is the only thing allowed to demote a guarded span, and it has to
    # deny the whole span: any frame it calls speech keeps the span guarded.
    spans = [(1.6, 1.84), (4.0, 4.4)]
    mask = [False] * 200
    assert filter_masked(spans, mask) == []
    for i in range(int(4.0 / VAD_FRAME_SEC), int(4.4 / VAD_FRAME_SEC) + 1):
        mask[i] = True
    assert filter_masked(spans, mask) == [(4.0, 4.4)]
