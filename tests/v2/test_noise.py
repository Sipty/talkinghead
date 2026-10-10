"""Tests for talkinghead.noise: the VAD-gated mouth-noise pass.

Audio is synthesized, not loaded: a low uniform-noise floor with sine "speech"
bursts, an alternating-sign smack (high ZCR), and a low-frequency breath, all
at the levels the spec's calibration table describes. Both gap paths are
covered - a synthetic speech_mask and the envelope-only fallback.
"""

from __future__ import annotations

import math
import random
import wave
from array import array
from pathlib import Path

import pytest

from talkinghead import noise
from talkinghead.noise import (
    VAD_FRAME_SEC,
    detect_mouth_noises,
    silero_speech_mask,
)
from talkinghead.transcribe import Word

SR = 16000
DURATION = 7.0

# (kind, start, end) - the ground truth every test below is written against.
SPEECH = [(0.0, 1.0), (2.0, 3.0), (4.3, 5.3), (5.51, 6.51)]
GAP_SMACK = (1.5, 1.55)  # 0.5s of quiet before, 0.45s after: cuttable
BREATH = (3.5, 3.8)  # 0.3s, low ZCR, ~12dB over floor: cuttable
EMBEDDED_SMACK = (5.38, 5.43)  # only 0.08s of quiet either side: must be kept

EVENTS = (
    [("speech", s, e) for s, e in SPEECH]
    + [("smack", *GAP_SMACK), ("breath", *BREATH), ("smack", *EMBEDDED_SMACK)]
)


def build_audio(events=EVENTS, duration: float = DURATION, seed: int = 7) -> array:
    """16-bit mono samples: a quiet noise floor with events painted over it.

    Levels are chosen so the classifier sees the spec's signatures: speech is
    ~42dB over the floor (far too loud for breath/throat) and 1s long (too
    long for a smack); the smack is ~17dB over the floor with ZCR 1.0; the
    breath is ~12dB over the floor with ZCR 0.04.
    """
    rng = random.Random(seed)
    samples = array("h", [rng.randint(-30, 30) for _ in range(int(duration * SR))])
    for kind, t0, t1 in events:
        start = int(t0 * SR)
        for i in range(start, int(t1 * SR)):
            k = i - start
            if kind == "speech":
                value = 3000 * math.sin(2 * math.pi * 150 * k / SR)
            elif kind == "smack":
                # Sign flips every sample, so ZCR lands at 1.0 (spec: > 0.4).
                value = (1 if k % 2 == 0 else -1) * rng.uniform(60, 180)
            else:  # breath: 400 Hz, ZCR 0.05, quiet
                value = 98 * math.sin(2 * math.pi * 400 * k / SR)
            samples[i] = int(value)
    return samples


def speech_mask(spans=SPEECH, duration: float = DURATION) -> list[bool]:
    """A synthetic Silero-shaped mask: one flag per VAD_FRAME_SEC of audio."""
    frames = int(duration / VAD_FRAME_SEC)
    return [
        any(s <= i * VAD_FRAME_SEC < e for s, e in spans)
        for i in range(frames)
    ]


def kinds(cuts) -> list[str]:
    return [c.text.removeprefix("mouth noise: ") for c in cuts]


def overlaps(cut, span: tuple[float, float]) -> bool:
    return cut.start < span[1] and span[0] < cut.end


# --- the VAD-mask path ---

def test_gap_smack_and_breath_are_cut() -> None:
    cuts = detect_mouth_noises(build_audio(), [], speech_mask=speech_mask())
    assert kinds(cuts) == ["smack", "breath"]
    assert all(c.reason == "noise" for c in cuts)
    assert cuts[0].start == pytest.approx(GAP_SMACK[0] - noise.DEFAULT_PAD)
    assert cuts[0].end == pytest.approx(GAP_SMACK[1] + noise.DEFAULT_PAD)


def test_embedded_smack_between_close_words_is_skipped() -> None:
    """Safety rule 2: <150ms of true quiet on a side means leave it alone."""
    cuts = detect_mouth_noises(build_audio(), [], speech_mask=speech_mask())
    assert not any(overlaps(c, EMBEDDED_SMACK) for c in cuts)


def test_speech_bursts_are_never_cut() -> None:
    cuts = detect_mouth_noises(build_audio(), [], speech_mask=speech_mask())
    assert not any(overlaps(c, span) for c in cuts for span in SPEECH)


def test_mask_marking_everything_as_speech_suppresses_all_cuts() -> None:
    """The gate is the VAD: bursts outside a speech gap are not candidates."""
    all_speech = [True] * int(DURATION / VAD_FRAME_SEC)
    assert detect_mouth_noises(build_audio(), [], speech_mask=all_speech) == []


def test_mask_marking_only_the_smack_gap_cuts_only_the_smack() -> None:
    mask = speech_mask(spans=SPEECH + [BREATH])
    assert kinds(detect_mouth_noises(build_audio(), [], speech_mask=mask)) == ["smack"]


# --- the envelope-only fallback path ---

def test_falls_back_to_envelope_gaps_without_silero(monkeypatch, capsys) -> None:
    monkeypatch.setattr(noise, "silero_speech_mask", lambda samples, sample_rate: None)
    cuts = detect_mouth_noises(build_audio(), [])
    assert kinds(cuts) == ["smack", "breath"]
    assert "silero-vad unavailable" in capsys.readouterr().err


def test_fallback_still_skips_the_embedded_smack(monkeypatch) -> None:
    monkeypatch.setattr(noise, "silero_speech_mask", lambda samples, sample_rate: None)
    cuts = detect_mouth_noises(build_audio(), [])
    assert not any(overlaps(c, EMBEDDED_SMACK) for c in cuts)


# --- safety rule 1: word spans bound the edges, never prove the gap ---

def test_cut_stops_short_of_the_next_word_onset() -> None:
    words = [Word("next", 1.56, 2.0, 1.0)]
    cuts = detect_mouth_noises(build_audio(), words, speech_mask=speech_mask())
    assert cuts[0].end == pytest.approx(1.56)


def test_word_span_swallowing_the_burst_does_not_block_the_cut() -> None:
    """The throat-clear case: ASR swallowed the noise into a word span, but
    the envelope shows quiet, so the cut still stands (spec safety rule 1)."""
    words = [Word("previous,", 0.9, 1.9, 1.0)]
    cuts = detect_mouth_noises(build_audio(), words, speech_mask=speech_mask())
    assert kinds(cuts) == ["smack", "breath"]


# --- entry points ---

def test_wav_file_path_gives_the_same_cuts(tmp_path: Path) -> None:
    wav = tmp_path / "clip.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(build_audio().tobytes())
    from_file = detect_mouth_noises(str(wav), [], speech_mask=speech_mask())
    from_samples = detect_mouth_noises(build_audio(), [], speech_mask=speech_mask())
    assert from_file == from_samples


def test_empty_audio_returns_no_cuts() -> None:
    assert detect_mouth_noises(array("h", []), [], speech_mask=[]) == []


def test_implausible_floor_fails_loudly() -> None:
    # Regression guard for the past wrong-file bug: a pass measured the noise
    # floor on the golden master instead of its actual input, silently
    # under-detecting everything. An envelope whose 25th-percentile "floor"
    # sits near full scale cannot come from real talking-head audio, so the
    # detector must fail loudly instead of emitting garbage cuts.
    loud = array("h", [18000, -18000, 17000, -17000] * 4000)  # ~1 s near full scale
    with pytest.raises(RuntimeError, match="implausible noise floor"):
        detect_mouth_noises(loud, [], speech_mask=[True] * (len(loud) // 512))


def test_realistic_floor_still_detects() -> None:
    # The guard must not fire on normal synthetic audio (floor near -50 dBFS).
    cuts = detect_mouth_noises(build_audio(), [], speech_mask=speech_mask())
    assert kinds(cuts) == ["smack", "breath"]


# --- the real model, when it is installed ---

def test_silero_mask_shape_and_gating() -> None:
    pytest.importorskip("silero_vad")
    samples = build_audio()
    mask = silero_speech_mask(samples, SR)
    if mask is None:
        pytest.skip("silero-vad present but onnxruntime/model unavailable")
    assert len(mask) == len(samples) // 512  # 512 samples = one 32ms VAD frame
    assert all(isinstance(v, bool) for v in mask)
    # Synthetic sines are not speech to Silero, so every burst sits in a gap
    # and the pass behaves like the fallback: same two cuts, no crash.
    assert kinds(detect_mouth_noises(samples, [])) == ["smack", "breath"]


def test_silero_mask_is_none_for_unsupported_sample_rate() -> None:
    assert silero_speech_mask(array("h", [0] * 1024), 44100) is None
