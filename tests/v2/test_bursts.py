"""Tests for talkinghead.bursts: the voiced-burst pass for ASR-blind vocals.

Audio is synthesized like test_noise.py: a low uniform-noise floor with
sine "speech" (150 Hz, ~42 dB over the floor) and sine "hum" (130 Hz, ~14 dB
over the floor - low-frequency, well under speech level, matching the >=
-31 dBFS residual energy measured in the v7 audit). The signature scenario
is the v7 master's absorbed "this": a 1.72 s word span holding a short loud
core and a quieter voiced hum.

The real-world regression case is replayed at the audited timestamps:
word span 155.22-156.94 ("this", 1.72 s) with a moan inside.
"""

from __future__ import annotations

import math
import random
import wave
from array import array
from pathlib import Path

import pytest

from talkinghead.bursts import detect_voiced_bursts
from talkinghead.transcribe import Word

SR = 16000
DURATION = 8.0

# 6 words before, absorbed span ("this", 1.72 s), 6 words after: enough
# neighbours for a stable median (SPAN_NEIGHBORS = 6 on each side).
NEIGHBORS_BEFORE = [(i * 0.5, i * 0.5 + 0.4) for i in range(6)]
NEIGHBORS_AFTER = [(4.8 + i * 0.5, 4.8 + i * 0.5 + 0.4) for i in range(6)]
ABSORB_SPAN = (3.0, 4.72)
ABSORB_CORE = (3.2, 3.7)  # the actual spoken word, at speech level
ABSORB_HUM = (4.0, 4.6)  # the moan whisper glued into the span


def absorbed_words() -> list[Word]:
    starts = [s for s, _ in NEIGHBORS_BEFORE] + [ABSORB_SPAN[0]] + [s for s, _ in NEIGHBORS_AFTER]
    ends = [e for _, e in NEIGHBORS_BEFORE] + [ABSORB_SPAN[1]] + [e for _, e in NEIGHBORS_AFTER]
    texts = ["filler"] * 6 + ["this"] + ["filler"] * 6
    return [Word(t, s, e, 1.0) for t, s, e in zip(texts, starts, ends)]


# (kind, start, end): speech = loud 150 Hz; hum = quiet 130 Hz (voiced, low
# ZCR); loud = a burst at word-core level (indistinguishable: must be kept);
# click = alternating-sign burst (smack-like ZCR: not this pass's finding).
def build_audio(
    events: list[tuple[str, float, float]],
    duration: float = DURATION,
    seed: int = 11,
) -> array:
    rng = random.Random(seed)
    samples = array("h", [rng.randint(-30, 30) for _ in range(int(duration * SR))])
    for kind, t0, t1 in events:
        start = int(t0 * SR)
        for i in range(start, int(t1 * SR)):
            k = i - start
            if kind == "speech":
                value = 3000 * math.sin(2 * math.pi * 150 * k / SR)
            elif kind == "hum":
                value = 120 * math.sin(2 * math.pi * 130 * k / SR)
            elif kind == "loud":
                value = 3000 * math.sin(2 * math.pi * 130 * k / SR)
            else:  # click: sign flips every sample, ZCR 1.0
                value = (1 if k % 2 == 0 else -1) * rng.uniform(60, 180)
            samples[i] = int(value)
    return samples


def absorbed_events(hum_kind: str = "hum") -> list[tuple[str, float, float]]:
    return (
        [("speech", s, e) for s, e in NEIGHBORS_BEFORE + NEIGHBORS_AFTER]
        + [("speech", *ABSORB_CORE), (hum_kind, *ABSORB_HUM)]
    )


def test_absorbed_hum_in_long_span_is_detected() -> None:
    cuts = detect_voiced_bursts(build_audio(absorbed_events()), absorbed_words())
    assert len(cuts) == 1
    c = cuts[0]
    assert c.reason == "noise"
    assert "absorbed" in c.text
    # The burst, not the word: starts after the split dip, ends inside the span.
    assert c.start == pytest.approx(ABSORB_HUM[0] - 0.02, abs=0.05)
    assert c.end == pytest.approx(ABSORB_HUM[1] + 0.02, abs=0.05)
    # The word core is never touched.
    assert not (c.start < ABSORB_CORE[1] and ABSORB_CORE[0] < c.end)


def test_real_world_absorption_case_is_detected() -> None:
    """The v7 audit case: 'this' at 155.22-156.94 (1.72 s) holding a moan."""
    before = [(149.0 + i * 0.5, 149.0 + i * 0.5 + 0.4) for i in range(6)]
    after = [(157.2 + i * 0.5, 157.2 + i * 0.5 + 0.4) for i in range(6)]
    words = [Word("w", s, e, 1.0) for s, e in before]
    words.append(Word("this", 155.22, 156.94, 1.0))
    words += [Word("w", s, e, 1.0) for s, e in after]
    events = (
        [("speech", s, e) for s, e in before + after]
        + [("speech", 155.42, 155.92), ("hum", 156.22, 156.82)]
    )
    cuts = detect_voiced_bursts(build_audio(events, duration=161.0), words)
    assert len(cuts) == 1
    c = cuts[0]
    assert 156.22 - 0.05 <= c.start and c.end <= 156.94
    assert not (c.start < 155.92 and 155.42 < c.end)  # core protected


def test_clean_speech_yields_no_findings() -> None:
    spans = [(i * 0.5, i * 0.5 + 0.4) for i in range(10)]
    events = [("speech", s, e) for s, e in spans]
    words = [Word("w", s, e, 1.0) for s, e in spans]
    assert detect_voiced_bursts(build_audio(events, duration=5.5), words) == []


def test_absorbed_burst_as_loud_as_the_word_core_is_skipped() -> None:
    # A vocalization at word-core level cannot be told apart from the word.
    cuts = detect_voiced_bursts(build_audio(absorbed_events("loud")), absorbed_words())
    assert cuts == []


def test_absorbed_burst_with_smack_zcr_is_skipped() -> None:
    # Clicking bursts are detect_mouth_noises' finding, not a voiced one.
    cuts = detect_voiced_bursts(build_audio(absorbed_events("click")), absorbed_words())
    assert cuts == []


def test_long_span_without_interior_burst_is_skipped() -> None:
    # Absorption signature alone (1.72 s span) with quiet inside the span:
    # nothing voiced to cut.
    events = [("speech", s, e) for s, e in NEIGHBORS_BEFORE + NEIGHBORS_AFTER]
    events.append(("speech", 3.2, 3.7))
    cuts = detect_voiced_bursts(build_audio(events), absorbed_words())
    assert cuts == []


def test_wav_file_path_gives_the_same_cuts(tmp_path: Path) -> None:
    samples = build_audio(absorbed_events())
    wav = tmp_path / "clip.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(samples.tobytes())
    assert detect_voiced_bursts(str(wav), absorbed_words()) == detect_voiced_bursts(
        samples, absorbed_words()
    )


def test_empty_audio_or_words_return_nothing() -> None:
    assert detect_voiced_bursts(array("h", []), absorbed_words()) == []
    assert detect_voiced_bursts(build_audio(absorbed_events()), []) == []


def test_implausible_floor_fails_loudly() -> None:
    loud = array("h", [18000, -18000, 17000, -17000] * 16000)  # 4 s near full scale
    with pytest.raises(RuntimeError, match="implausible noise floor"):
        detect_voiced_bursts(loud, absorbed_words())


# --- finding 2: hums inside inter-word gaps ----------------------------

GAP_WORDS = [Word("before", 1.0, 1.5, 1.0), Word("after", 2.2, 2.7, 1.0)]
GAP_HUM = (1.6, 2.0)
GAP_SILENCES = [(1.5, 2.2)]  # detect_silences calls the whole gap quiet


def gap_events(hum_kind: str = "hum") -> list[tuple[str, float, float]]:
    return [
        ("speech", 1.0, 1.5),
        ("speech", 2.2, 2.7),
        (hum_kind, *GAP_HUM),
    ]


def test_gap_hum_inside_silence_is_detected() -> None:
    cuts = detect_voiced_bursts(build_audio(gap_events(), duration=4.0), GAP_WORDS,
                                silences=GAP_SILENCES)
    assert len(cuts) == 1
    c = cuts[0]
    assert c.reason == "noise"
    assert GAP_WORDS[0].end <= c.start and c.end <= GAP_WORDS[1].start
    assert GAP_HUM[0] - 0.05 <= c.start <= GAP_HUM[0]
    assert GAP_HUM[1] <= c.end <= GAP_HUM[1] + 0.05


def test_gap_hum_not_in_silences_is_skipped() -> None:
    # detect_silences calls the gap speech (empty list): the witness fails,
    # so the burst is left alone - quiet real speech may live there.
    cuts = detect_voiced_bursts(build_audio(gap_events(), duration=4.0), GAP_WORDS, silences=[])
    assert cuts == []


def test_gap_hum_without_silences_falls_back_to_level_gate() -> None:
    # Quiet hum: under floor+GAP_HUM_MAX_DB, so still cuttable without a
    # detect_silences witness.
    cuts = detect_voiced_bursts(build_audio(gap_events(), duration=4.0), GAP_WORDS)
    assert len(cuts) == 1
    # A hum at speech level without the witness is indistinguishable from
    # quiet speech whisper missed: left alone.
    loud = detect_voiced_bursts(build_audio(gap_events("loud"), duration=4.0), GAP_WORDS)
    assert loud == []


def test_short_gaps_are_never_scanned() -> None:
    # 0.1 s gaps between normal words: inside co-articulation, no findings
    # even with hum energy present elsewhere.
    events = [("speech", 1.0, 1.5), ("speech", 1.6, 2.1), ("hum", 2.5, 2.9)]
    words = [Word("a", 1.0, 1.5, 1.0), Word("b", 1.6, 2.1, 1.0), Word("c", 3.0, 3.4, 1.0)]
    assert detect_voiced_bursts(build_audio(events, duration=4.0), words, silences=[]) == []