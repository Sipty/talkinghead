"""Scribe speakers: parsed per word, kept through every copy, one label set across chunks. No network."""

from __future__ import annotations

import json

import numpy as np

from talkinghead import transcribe as module
from talkinghead.audio import envelope, refine_word_bounds
from talkinghead.transcribe import (
    PCM_RATE,
    SCRIBE_VERSION,
    AudioEvent,
    ScribeChunk,
    Transcript,
    Word,
    _merge_chunks,
    _reconcile_speakers,
    parse_openrouter_response,
)

S = PCM_RATE
# Two requests over a 40 s recording: the first hears 0-25 s, the second 15-40 s. The overlap is 15-25 s.
FIRST = ScribeChunk(0, 25 * S, 0, 20 * S)
SECOND = ScribeChunk(15 * S, 40 * S, 20 * S, 40 * S)
# (text, start, end, speaker) on the recording's clock. A and B both talk inside the overlap.
RECORDING = [
    ("zeta", 2.0, 2.4, "A"),
    ("alpha", 16.0, 16.4, "A"),
    ("beta", 17.0, 17.4, "A"),
    ("gamma", 19.0, 19.4, "B"),
    ("delta", 21.0, 21.4, "B"),
    ("eta", 30.0, 30.4, "B"),
]
# Only A talks inside the overlap.
LONE_OVERLAP = [
    ("zeta", 2.0, 2.4, "A"),
    ("theta", 5.0, 5.4, "B"),
    ("alpha", 16.0, 16.4, "A"),
    ("beta", 17.0, 17.4, "A"),
    ("eta", 30.0, 30.4, "B"),
]


def _answer(chunk: ScribeChunk, heard: list[tuple[str, float, float, str]], raw: dict[str, str]):
    """What one request returns: the words inside its window, on its own clock, under that request's labels."""
    shift = chunk.request_start / S
    lo, hi = chunk.request_start / S, chunk.request_end / S
    words = [Word(text, s - shift, e - shift, 1.0, raw[who])
             for text, s, e, who in heard if lo <= (s + e) / 2 < hi]
    return chunk, words, []


def _speakers(parts) -> list[dict[str, str | None]]:
    return [{w.text: w.speaker for w in words} for _, words, _ in parts]


def test_openrouter_words_carry_their_speaker_as_s_n_with_the_label_preferred() -> None:
    native = {"words": [
        {"word": "hello", "start": 0.1, "end": 0.4, "confidence": 0.9, "speaker": 0, "speaker_label": "speaker_0"},
        {"word": "yes", "start": 0.5, "end": 0.7, "speaker": 1},
        {"word": "ok", "start": 0.8, "end": 0.9, "speaker_label": "speaker_2"},
        {"word": "clash", "start": 0.95, "end": 1.0, "speaker": 1, "speaker_label": "speaker_0"},
        {"word": "no", "start": 1.0, "end": 1.2, "speaker": True},
        {"word": "hm", "start": 1.3, "end": 1.5},
        {"type": "audio_event", "word": "[laughs]", "start": 1.6, "end": 2.0, "speaker": 0},
    ]}
    words, events = parse_openrouter_response(native, duration=3.0)
    assert [w.speaker for w in words] == ["S0", "S1", "S2", "S0", None, None]
    assert [e.text for e in events] == ["[laughs]"]


def test_speakers_round_trip_and_cache_entries_without_them_still_load() -> None:
    t = Transcript("scribe", SCRIBE_VERSION, 2.0,
                   [Word("a", 0.0, 1.0, 1.0, "S1"), Word("b", 1.0, 2.0, 1.0)], [AudioEvent("[sighs]", 0.2, 0.6)])
    d = t.to_dict()
    assert d["words"][0]["sp"] == "S1" and "sp" not in d["words"][1]
    back = Transcript.from_dict(json.loads(json.dumps(d)))
    assert [w.speaker for w in back.words] == ["S1", None]
    legacy = {"backend": "scribe", "version": "old", "duration": 2.0,
              "words": [{"w": "a", "s": 0.0, "e": 1.0, "p": 1.0}], "events": []}
    assert Transcript.from_dict(legacy).words[0].speaker is None


def test_the_scribe_body_asks_for_speakers_and_the_version_says_so() -> None:
    body = json.loads(module._openrouter_body(b"RIFF"))
    assert body["diarize"] is True and body["response_format"] == "verbose_json"
    assert "diarize=true" in SCRIBE_VERSION


def test_refinement_keeps_the_speaker() -> None:
    env = envelope(np.zeros(PCM_RATE * 2), PCM_RATE)
    (r,) = refine_word_bounds([Word("hi", 0.5, 0.9, 1.0, "S1")], env)
    assert r.speaker == "S1"


def test_labels_swapped_in_the_second_request_map_back_to_the_first() -> None:
    parts = [_answer(FIRST, RECORDING, {"A": "S0", "B": "S1"}),
             _answer(SECOND, RECORDING, {"A": "S1", "B": "S0"})]
    first, second = _speakers(_reconcile_speakers(parts))
    assert first == {"zeta": "S0", "alpha": "S0", "beta": "S0", "gamma": "S1", "delta": "S1"}
    assert second == {"alpha": "S0", "beta": "S0", "gamma": "S1", "delta": "S1", "eta": "S1"}


def test_a_lone_overlap_speaker_pairs_with_the_one_leftover_label() -> None:
    parts = [_answer(FIRST, LONE_OVERLAP, {"A": "S0", "B": "S1"}),
             _answer(SECOND, LONE_OVERLAP, {"A": "S1", "B": "S0"})]
    _, second = _speakers(_reconcile_speakers(parts))
    assert second == {"alpha": "S0", "beta": "S0", "eta": "S1"}


def test_a_third_speaker_in_the_second_request_gets_a_new_label() -> None:
    recording = [*RECORDING, ("kappa", 30.5, 30.9, "C")]
    parts = [_answer(FIRST, recording, {"A": "S0", "B": "S1"}),
             _answer(SECOND, recording, {"A": "S0", "B": "S1", "C": "S2"})]
    _, second = _speakers(_reconcile_speakers(parts))
    assert second == {"alpha": "S0", "beta": "S0", "gamma": "S1", "delta": "S1", "eta": "S1", "kappa": "S2"}


def test_the_merged_timeline_keeps_each_words_global_speaker() -> None:
    parts = _reconcile_speakers([_answer(FIRST, RECORDING, {"A": "S0", "B": "S1"}),
                                 _answer(SECOND, RECORDING, {"A": "S1", "B": "S0"})])
    words, _ = _merge_chunks(parts, 40 * S)
    assert [(w.text, w.speaker) for w in words] == [
        ("zeta", "S0"), ("alpha", "S0"), ("beta", "S0"), ("gamma", "S1"), ("delta", "S1"), ("eta", "S1")]
