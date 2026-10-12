"""Voice signatures and voice-first speaker reconciliation, on synthetic audio. No network."""

from __future__ import annotations

import numpy as np

from talkinghead import voice
from talkinghead.transcribe import PCM_RATE, ScribeChunk, Word, _reconcile_speakers, _voice_mapping

S = PCM_RATE
FRAMES = 4100  # 41 s at 100 frames per second
HOST = 5.0 + 2.0 * np.linspace(-1.0, 1.0, voice.MEL_BANDS)  # one voice's log-mel profile
BROTHER = 10.0 - HOST  # mirrored about 5: a clearly different voice
# Two requests over 0-41 s: the first hears 0-25 s, the second 15-40 s; the overlap is 15-25 s.
FIRST = ScribeChunk(0, 25 * S, 0, 20 * S)
SECOND = ScribeChunk(15 * S, 40 * S, 20 * S, 40 * S)


def _mel(spans: list[tuple[float, float, np.ndarray]]) -> np.ndarray:
    rng = np.random.default_rng(0)
    mel = np.zeros((FRAMES, voice.MEL_BANDS), np.float32)
    for start, end, vec in spans:
        idx = voice.frames_in(start, end, FRAMES)
        mel[idx] = vec + rng.normal(0, 0.05, (idx.size, voice.MEL_BANDS))
    return mel


def _part(chunk: ScribeChunk, turns: list[tuple[str, float, float, int, str]]):
    """Words for one request. Each turn: (label, start, end, word count, one-letter text prefix), recording clock."""
    shift = chunk.request_start / S
    words = []
    for label, t0, t1, n, prefix in turns:
        step = (t1 - t0) / n
        for i in range(n):
            start = t0 + i * step
            words.append(Word(f"{prefix}{i}", start - shift, start + 0.8 * step - shift, 1.0, label))
    return chunk, sorted(words, key=lambda w: w.start), []


def _labels(part) -> dict[str, str]:
    return {w.text[0]: w.speaker for w in part[1]}


def test_log_mel_has_one_row_per_10_ms_and_forty_bands() -> None:
    assert voice.log_mel(np.zeros(S)).shape == (98, voice.MEL_BANDS)
    assert voice.log_mel(np.zeros(10)).shape == (0, voice.MEL_BANDS)


def test_frames_in_selects_frames_whose_centre_lies_in_the_span() -> None:
    idx = voice.frames_in(1.0, 2.0, FRAMES)
    assert idx[0] * voice.FRAME_HOP + voice.FRAME_WIN / 2 >= 1.0 * S
    assert idx[-1] * voice.FRAME_HOP + voice.FRAME_WIN / 2 <= 2.0 * S


def test_a_signature_survives_a_loudness_offset_and_tells_two_voices_apart() -> None:
    spans = [(0, 5, HOST), (5, 10, BROTHER)]
    a = voice.chunk_signatures(_mel(spans), [(0, 5, "x"), (5, 10, "y")])
    b = voice.chunk_signatures(_mel([(s, e, v + 3.0) for s, e, v in spans]), [(0, 5, "x"), (5, 10, "y")])
    assert set(a) == {"x", "y"}
    assert voice.similarity(a["x"][0], b["x"][0]) > 0.99
    assert voice.similarity(a["x"][0], a["y"][0]) < -0.9


def test_a_label_without_enough_exclusive_frames_has_no_signature() -> None:
    mel = _mel([(0, 5, HOST)])
    assert voice.chunk_signatures(mel, [(0, 0.1, "x")]) == {}


def test_voice_mapping_picks_the_closest_global_and_reports_the_margin() -> None:
    local = {"a": -HOST, "b": HOST}
    globals_ = {"S0": HOST, "S1": -HOST}
    mapping, margin = _voice_mapping(["a", "b"], ["S0", "S1"], local, globals_)
    assert mapping == {"a": "S1", "b": "S0"}
    assert margin > 0.5  # the runner-up leaves one label unmatched and scores 1.0


def test_voice_mapping_has_no_margin_when_two_voices_are_indistinguishable() -> None:
    same = {"a": HOST, "b": HOST}
    _, margin = _voice_mapping(["a", "b"], ["S0", "S1"], same, {"S0": HOST, "S1": -HOST})
    assert margin < 1e-6


def test_voices_repair_a_swapped_second_request_when_no_words_overlap_in_text() -> None:
    # Physical voices: host at 0-10, 15-20, 25-30, 35-40; brother at 10-15, 20-25, 30-35.
    # The first request calls the host S0; the second calls the host S1 (swapped), with different words.
    mel = _mel([(0, 10, HOST), (10, 15, BROTHER), (15, 20, HOST), (20, 25, BROTHER),
                (25, 30, HOST), (30, 35, BROTHER), (35, 40, HOST)])
    first = _part(FIRST, [("S0", 0, 10, 25, "a"), ("S1", 10, 15, 25, "b"),
                          ("S0", 15, 20, 25, "c"), ("S1", 20, 25, 25, "g")])
    second = _part(SECOND, [("S1", 15, 20, 25, "d"), ("S0", 20, 25, 25, "e"),
                            ("S1", 25, 30, 25, "f"), ("S0", 30, 35, 25, "h"), ("S1", 35, 40, 25, "i")])
    out = _reconcile_speakers([first, second], max_speakers=2, mel=mel)
    assert _labels(out[0]) == {"a": "S0", "b": "S1", "c": "S0", "g": "S1"}
    assert _labels(out[1]) == {"d": "S0", "e": "S1", "f": "S0", "h": "S1", "i": "S0"}


def test_text_breaks_a_tie_when_the_voices_cannot_be_told_apart() -> None:
    mel = _mel([(0, 41, HOST)])  # both labels sound the same
    first = _part(FIRST, [("S0", 15, 20, 25, "t"), ("S1", 20, 25, 25, "u")])
    second = _part(SECOND, [("S1", 15, 20, 25, "t"), ("S0", 20, 25, 25, "u")])
    out = _reconcile_speakers([first, second], max_speakers=2, mel=mel)
    assert _labels(out[1]) == {"t": "S0", "u": "S1"}


def test_without_a_recording_log_mel_the_text_path_is_unchanged() -> None:
    first = _part(FIRST, [("S0", 15, 20, 25, "t"), ("S1", 20, 25, 25, "u")])
    second = _part(SECOND, [("S1", 15, 20, 25, "t"), ("S0", 20, 25, 25, "u")])
    out = _reconcile_speakers([first, second], max_speakers=2)
    assert _labels(out[1]) == {"t": "S0", "u": "S1"}
