"""Scribe speakers: parsed per word, kept through every copy, one label set across chunks. No network."""

from __future__ import annotations

import json

import numpy as np
import pytest

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
    words, events = parse_openrouter_response(native, duration=3.0, diarized=True)
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


def test_the_diarized_body_asks_for_speakers_and_the_version_says_so() -> None:
    body = json.loads(module._openrouter_body(b"RIFF", diarize=True))
    assert body["diarize"] is True and body["response_format"] == "verbose_json"
    assert "diarize=true" in module.SCRIBE_DIARIZED_VERSION


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


def test_undiarized_transcripts_keep_the_cache_key_they_have_always_had() -> None:
    # Changing these strings re-keys every cached transcript of that kind.
    assert module.SCRIBE_VERSION == "scribe_v2;word;events;diarize=false;verbatim;parse=1"
    assert module.ScribeTranscriber().version == "scribe_v2;word;events;diarize=false;verbatim;parse=1"
    assert module.ScribeTranscriber(diarize=True).version == "scribe_v2;word;events;diarize=true;verbatim;parse=2"


def test_an_undiarized_request_body_is_byte_for_byte_the_old_one() -> None:
    assert module._openrouter_body(b"RIFF") == (
        b'{"model": "elevenlabs/scribe-v2", "input_audio": {"data": "UklGRg==", "format": "wav"}, '
        b'"response_format": "verbose_json", "timestamp_granularities": ["word"]}')


def test_the_diarized_body_adds_only_the_diarize_flag() -> None:
    off = json.loads(module._openrouter_body(b"RIFF"))
    assert json.loads(module._openrouter_body(b"RIFF", diarize=True)) == {**off, "diarize": True}


def test_undiarized_answers_are_parsed_without_speakers() -> None:
    native = {"words": [{"word": "a", "start": 0.0, "end": 0.5, "speaker": 0, "speaker_label": "speaker_0"}]}
    words, _ = parse_openrouter_response(native, duration=1.0)
    assert words[0].speaker is None


def test_the_words_digest_ignores_speakers_only_when_none_are_present() -> None:
    import hashlib

    from talkinghead.pipeline import _words_sha
    plain = [Word("a", 0.1234, 0.5, 1.0), Word("b", 0.6, 0.9, 1.0)]
    old = hashlib.sha256(json.dumps([(w.text, round(w.start, 3), round(w.end, 3)) for w in plain]).encode())
    assert _words_sha(plain) == old.hexdigest()[:16]
    labelled = [Word("a", 0.1234, 0.5, 1.0, "S0"), Word("b", 0.6, 0.9, 1.0, "S1")]
    assert _words_sha(labelled) != old.hexdigest()[:16]


def test_the_config_writes_no_diarize_key_while_off_and_reads_it_from_toml(tmp_path) -> None:
    from talkinghead.config import Config, load_config
    assert "diarize" not in Config().to_dict()
    assert "diarize_speakers" not in Config().to_dict()
    assert Config(diarize=True).to_dict()["diarize"] is True
    f = tmp_path / "talkinghead.toml"
    f.write_text('transcriber = "scribe"\ndiarize = true\n')
    assert load_config(f).diarize is True


def test_diarize_reaches_only_the_scribe_transcriber(monkeypatch) -> None:
    from talkinghead.cli import build_parser, build_transcriber
    monkeypatch.setenv("OPENROUTER_API_KEY", "route-test-key")
    assert build_transcriber("scribe").version == module.SCRIBE_VERSION
    assert build_transcriber("scribe", diarize=True).version == module.SCRIBE_DIARIZED_VERSION
    with pytest.raises(ValueError, match="--transcriber scribe"):
        build_transcriber("whisper", diarize=True)
    assert build_parser().parse_args(["run", "v.mov", "--diarize"]).diarize is True
    assert build_parser().parse_args(["run", "v.mov"]).diarize is False


def _turns(backchannel: bool) -> list[tuple[str, float, float, str]]:
    """A talks 0-20 s, B talks 20-25 s, A talks on from 25 s. With backchannel, X (a third raw label)
    sits between A's words from 30 s, so it is next to A and never next to B."""
    heard = [(f"a{i}", 0.2 * i, 0.2 * i + 0.1, "A") for i in range(100)]
    heard += [(f"b{i}", 20.0 + 0.2 * i, 20.0 + 0.2 * i + 0.1, "B") for i in range(25)]
    for k in range(75):
        t = 25.0 + 0.2 * k
        if backchannel and t >= 30.0 and k % 4 == 3:
            heard.append((f"x{k}", t, t + 0.1, "X"))
        else:
            heard.append((f"c{k}", t, t + 0.1, "A"))
    return heard


def test_a_one_word_label_is_absorbed_into_its_nearest_speaker_and_never_becomes_one() -> None:
    heard = [*_turns(backchannel=False), ("stella", 10.2, 10.4, "S")]
    parts = [_answer(FIRST, heard, {"A": "S0", "B": "S1", "S": "S2"}),
             _answer(SECOND, heard, {"A": "S1", "B": "S0", "S": "S2"})]
    out = _reconcile_speakers(parts)
    assert _speakers(out)[0]["stella"] == "S0"
    assert {w.speaker for _, words, _ in out for w in words} == {"S0", "S1"}
    assert {w.speaker for w in out[1][1] if w.text.startswith("c")} == {"S0"}
    assert {w.speaker for w in out[1][1] if w.text.startswith("b")} == {"S1"}


def test_with_a_speaker_bound_a_third_label_folds_onto_the_speaker_it_sits_next_to_least() -> None:
    heard = _turns(backchannel=True)
    parts = [_answer(FIRST, heard, {"A": "S0", "B": "S1", "X": "S2"}),
             _answer(SECOND, heard, {"A": "S1", "B": "S0", "X": "S2"})]
    bounded = _reconcile_speakers(parts, max_speakers=2)
    assert {w.speaker for w in bounded[1][1] if w.text.startswith("x")} == {"S1"}
    assert {w.speaker for _, words, _ in bounded for w in words} == {"S0", "S1"}
    unbounded = _reconcile_speakers(parts)
    assert {w.speaker for w in unbounded[1][1] if w.text.startswith("x")} == {"S2"}


def test_the_speaker_bound_needs_diarize_and_shows_in_the_version() -> None:
    from talkinghead.config import Config
    with pytest.raises(ValueError):
        Config(diarize_speakers=2).validate()
    assert Config(diarize=True, diarize_speakers=2).to_dict()["diarize_speakers"] == 2
    assert "diarize_speakers" not in Config(diarize=True).to_dict()
    bounded = module.ScribeTranscriber(diarize=True, diarize_speakers=2)
    assert bounded.version == module.SCRIBE_DIARIZED_VERSION + ";speakers=2"
