"""Transcriber interface: any backend, transcribed once per audio hash."""

import array
import io
import json
import math
import subprocess
import urllib.error
import urllib.request
import wave
from pathlib import Path

import pytest

from talkinghead.store import Cache
from talkinghead.transcribe import (
    AudioEvent,
    AVMismatch,
    FakeTranscriber,
    Transcript,
    Word,
    av_durations,
    check_av_mismatch,
    prepare_audio,
    transcribe_cached,
)


def test_transcribe_cached_calls_the_backend_once_per_audio_hash(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "cache")
    backend = FakeTranscriber([Word("a", 0.0, 0.5, 1.0)], duration=1.0)
    t1 = transcribe_cached(cache, "wavsha", backend, "unused.wav")
    t2 = transcribe_cached(cache, "wavsha", backend, "unused.wav")
    assert backend.calls == 1
    assert t1 == t2 and t1.backend == "fake"
    other = FakeTranscriber([Word("b", 0.0, 0.5, 1.0)], duration=1.0, name="other")
    transcribe_cached(cache, "wavsha", other, "unused.wav")
    assert other.calls == 1


def _synthetic_video(path: Path, video_sec: float, audio_sec: float) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=15:duration={video_sec}",
         "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={audio_sec}",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(path)],
        check=True,
    )


def test_av_durations_and_mismatch_guard(tmp_path: Path) -> None:
    ok = tmp_path / "ok.mp4"
    _synthetic_video(ok, 3.0, 3.0)
    v, a = av_durations(str(ok))
    assert v == pytest.approx(3.0, abs=0.2) and a == pytest.approx(3.0, abs=0.2)
    check_av_mismatch(str(ok))  # no raise

    skew = tmp_path / "skew.mp4"
    _synthetic_video(skew, 3.0, 6.0)
    with pytest.raises(AVMismatch, match="audio"):
        check_av_mismatch(str(skew), tolerance=1.0)
    check_av_mismatch(str(skew), tolerance=1.0, allow=True)  # explicit override


def test_prepare_audio_extracts_16k_mono_wav_and_hashes_it(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    _synthetic_video(clip, 2.0, 2.0)
    wav, sha = prepare_audio(str(clip), tmp_path / "work")
    assert Path(wav).is_file() and wav.endswith(".wav")
    assert len(sha) == 64
    wav2, sha2 = prepare_audio(str(clip), tmp_path / "work2")
    assert sha2 == sha  # extraction is deterministic, so the cache key is stable


def test_forced_extraction_replaces_a_stale_workdir_wav(tmp_path: Path) -> None:
    """collect_facts measures `final.mp4`; a wav left by an earlier render must
    never be reused by existence. force=True re-extracts, and the deterministic
    extraction means the hash is unchanged when the render itself is."""
    clip = tmp_path / "clip.mp4"
    _synthetic_video(clip, 2.0, 2.0)
    work = tmp_path / "work"
    work.mkdir()
    stale = b"stale bytes from an earlier render"
    (work / "audio-16k.wav").write_bytes(stale)
    wav, _sha = prepare_audio(str(clip), work)          # default: reused as-is
    assert Path(wav).read_bytes() == stale
    wav2, sha2 = prepare_audio(str(clip), work, force=True)
    assert Path(wav2).read_bytes() != stale
    _, fresh_sha = prepare_audio(str(clip), tmp_path / "fresh", force=True)
    assert sha2 == fresh_sha


def _write_pcm_wav(path: Path, samples: list[int], sr: int = 16000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(array.array("h", samples).tobytes())


def test_silence_threshold_is_invariant_to_track_composition(tmp_path: Path) -> None:
    """Regression: the old noise-referenced threshold (2x the 15th-percentile
    frame RMS) sank as the track grew. On the owner's recording the same first
    6 minutes returned 178 silence ranges alone but 38 inside the full
    31-minute file, which is what left the full cook loosely cut. The
    speech-referenced threshold must return the same silences for the same
    audio whether it is detected alone or embedded in a much longer, quieter
    track."""
    from talkinghead.transcribe import detect_silences

    sr = 16000

    def speech(seconds: float) -> list[int]:
        return [int(3000 * math.sin(2 * math.pi * 440 * i / sr)) for i in range(int(seconds * sr))]

    tail = [-80] * int(0.3 * sr)   # soft word tails: above the noise, under speech
    pause = [-40] * int(0.2 * sr)  # inter-word pause
    deep = [-10] * int(0.5 * sr)   # room tone much quieter than the prefix's

    prefix = []
    for _ in range(6):
        prefix += speech(0.5) + tail + pause
    prefix_dur = 6.0

    # The long track is mostly deep quiet, but keeps enough speech bursts
    # that the 90th percentile still lands on speech (as in the real
    # recording, where speech stays near the top decile however long the
    # file is). It opens with a burst so the last prefix pause does not fuse
    # into the appended quiet.
    long_track = prefix + speech(0.5)
    for _ in range(8):
        long_track += deep * 9 + speech(0.5)

    solo_wav, long_wav = tmp_path / "solo.wav", tmp_path / "long.wav"
    _write_pcm_wav(solo_wav, prefix)
    _write_pcm_wav(long_wav, long_track)

    solo = [s for s in detect_silences(str(solo_wav)) if s[1] <= prefix_dur + 1e-6]
    both = [s for s in detect_silences(str(long_wav)) if s[1] <= prefix_dur + 1e-6]
    assert len(solo) == len(both) >= 5, (solo, both)
    for (s1, e1), (s2, e2) in zip(solo, both):
        assert s1 == pytest.approx(s2, abs=0.12) and e1 == pytest.approx(e2, abs=0.12)


# ---------------------------------------------------------------- Scribe v2


def _tone_wav(path: Path, seconds: float = 2.0, sr: int = 16000, amp: int = 3000) -> None:
    _write_pcm_wav(path, [int(amp * math.sin(2 * math.pi * 300 * i / sr)) for i in range(int(seconds * sr))], sr)


NATIVE = {
    "language_code": "eng",
    "audio_duration_secs": 2.0,
    "words": [
        {"text": "Hello", "start": 0.1, "end": 0.5, "type": "word", "logprob": -0.1},
        {"text": " ", "start": 0.5, "end": 0.6, "type": "spacing", "logprob": 0.0},
        {"text": "[laughs]", "start": 0.6, "end": 1.2, "type": "audio_event", "logprob": -0.2},
        {"text": "world.", "start": 1.3, "end": 1.8, "type": "word", "logprob": 0.0},
    ],
}


def _saved(tmp_path: Path, wav: Path, native: dict = NATIVE):
    from talkinghead.transcribe import SavedScribeTranscriber

    response = tmp_path / "response.json"
    response.write_text(json.dumps(native))
    return SavedScribeTranscriber(response=response, source_audio=wav)


def test_scribe_parse_separates_words_from_events_and_skips_spacing() -> None:
    from talkinghead.transcribe import parse_scribe_response

    words, events = parse_scribe_response(NATIVE, duration=2.0)
    assert [(w.text, w.start, w.end) for w in words] == [("Hello", 0.1, 0.5), ("world.", 1.3, 1.8)]
    assert 0.0 < words[0].probability < 1.0 and words[1].probability == 1.0
    assert [(e.text, e.start, e.end) for e in events] == [("[laughs]", 0.6, 1.2)]


def test_scribe_parse_refuses_stamps_outside_the_audio() -> None:
    from talkinghead.transcribe import parse_scribe_response

    with pytest.raises(ValueError, match="other audio"):
        parse_scribe_response(NATIVE, duration=1.0)


def test_saved_scribe_replays_for_the_same_pcm_whatever_the_wav_header(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    _tone_wav(source)
    saved = _saved(tmp_path, source)
    # ffmpeg without bitexact adds a LIST chunk: a different file, the same PCM.
    rewrapped = tmp_path / "rewrapped.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(source), "-c:a", "pcm_s16le", str(rewrapped)],
                   check=True)
    assert rewrapped.read_bytes() != source.read_bytes()
    got = saved.transcribe(str(rewrapped))
    assert [w.text for w in got.words] == ["Hello", "world."]
    assert [e.text for e in got.events] == ["[laughs]"]
    assert got.duration == pytest.approx(2.0)


def test_saved_scribe_source_may_be_the_original_the_pipeline_downmixes(tmp_path: Path) -> None:
    original = tmp_path / "stereo48.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=300:sample_rate=48000:duration=2",
                    "-ac", "2", str(original)], check=True)
    prepared, _ = prepare_audio(str(original), tmp_path / "work")
    saved = _saved(tmp_path, original)
    assert saved.transcribe(prepared).words


def test_saved_scribe_rejects_audio_that_differs_in_one_sample(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    _tone_wav(source)
    saved = _saved(tmp_path, source)
    with wave.open(str(source), "rb") as w:
        samples = array.array("h", w.readframes(w.getnframes()))
    samples[12345] += 1
    other = tmp_path / "other.wav"
    _write_pcm_wav(other, list(samples))
    with pytest.raises(ValueError, match="does not match"):
        saved.transcribe(str(other))


def test_saved_scribe_rejects_a_response_for_other_audio_at_construction(tmp_path: Path) -> None:
    short = tmp_path / "short.wav"
    _tone_wav(short, seconds=1.0)
    with pytest.raises(ValueError):
        _saved(tmp_path, short)


def test_saved_scribe_cache_never_replays_one_response_for_another(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    _tone_wav(source)
    other_native = {**NATIVE, "words": [{"text": "Different", "start": 0.2, "end": 0.9, "type": "word"}]}
    first = _saved(tmp_path, source)
    second_dir = tmp_path / "second"
    second_dir.mkdir()
    second = _saved(second_dir, source, other_native)
    assert first.version != second.version
    cache = Cache(tmp_path / "cache")
    a = transcribe_cached(cache, "same-audio-sha", first, str(source))
    b = transcribe_cached(cache, "same-audio-sha", second, str(source))
    assert [w.text for w in a.words] == ["Hello", "world."]
    assert [w.text for w in b.words] == ["Different"] and b.events == []


def test_cached_transcript_keeps_events_and_legacy_entries_load_without_them(tmp_path: Path) -> None:
    events = [AudioEvent("[sighs]", 1.0, 2.5)]
    backend = FakeTranscriber([Word("a", 0.0, 0.5, 1.0)], duration=3.0, events=events)
    cache = Cache(tmp_path / "cache")
    first = transcribe_cached(cache, "sha", backend, "unused.wav")
    again = transcribe_cached(cache, "sha", backend, "unused.wav")
    assert backend.calls == 1
    assert first.events == again.events == events
    legacy = Transcript.from_dict({"backend": "b", "version": "1", "duration": 1.0, "words": []})
    assert legacy.events == []



def test_live_scribe_without_a_key_fails_loudly_and_never_calls_out(tmp_path: Path, monkeypatch) -> None:
    from talkinghead.transcribe import ScribeTranscriber

    wav = tmp_path / "a.wav"
    _tone_wav(wav)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("network used without a key"))
    with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
        ScribeTranscriber().transcribe(str(wav))


def test_live_scribe_http_failure_reports_status_without_the_key(tmp_path: Path, monkeypatch) -> None:
    from talkinghead.transcribe import ScribeTranscriber

    wav = tmp_path / "a.wav"
    _tone_wav(wav)

    def deny(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, io.BytesIO(b'{"detail":"invalid"}'))

    monkeypatch.setenv("ELEVENLABS_API_KEY", "secret-do-not-print")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(urllib.request, "urlopen", deny)
    with pytest.raises(RuntimeError) as err:
        ScribeTranscriber().transcribe(str(wav))
    assert "401" in str(err.value) and "secret-do-not-print" not in str(err.value)


# ----------------------------------------------- Scribe routes and fallback

OPENROUTER_NATIVE = {
    "words": [
        {"word": "Hello", "start": 0.1, "end": 0.5, "confidence": 0.9},
        {"word": "[laughs]", "start": 0.6, "end": 1.2, "type": "audio_event", "confidence": 0.99},
        {"word": "world.", "start": 1.3, "end": 1.8, "type": "word"},
    ],
}
ROUTE_ELEVENLABS_NATIVE = {
    "words": [
        {"text": "Hello", "start": 0.1, "end": 0.5, "type": "word", "logprob": 0.0},
        {"text": "world.", "start": 1.3, "end": 1.8, "type": "word", "logprob": 0.0},
    ],
}


def _route_env(monkeypatch, *, elevenlabs: bool, openrouter: bool) -> None:
    for name, present in (("ELEVENLABS_API_KEY", elevenlabs), ("OPENROUTER_API_KEY", openrouter)):
        if present:
            monkeypatch.setenv(name, "route-test-key")
        else:
            monkeypatch.delenv(name, raising=False)


def test_openrouter_parse_reads_untyped_words_events_and_confidence() -> None:
    from talkinghead.transcribe import parse_openrouter_response

    words, events = parse_openrouter_response(OPENROUTER_NATIVE, duration=2.0)
    assert [(w.text, w.start, w.end) for w in words] == [("Hello", 0.1, 0.5), ("world.", 1.3, 1.8)]
    assert words[0].probability == pytest.approx(0.9) and words[1].probability == 1.0
    assert [(e.text, e.start, e.end) for e in events] == [("[laughs]", 0.6, 1.2)]


def test_elevenlabs_parse_still_refuses_an_untyped_item() -> None:
    from talkinghead.transcribe import parse_scribe_response

    with pytest.raises(ValueError, match="unknown type"):
        parse_scribe_response({"words": [{"text": "x", "start": 0.0, "end": 0.1}]}, duration=2.0)


def test_scribe_uses_elevenlabs_when_its_key_is_set(tmp_path: Path, monkeypatch) -> None:
    from talkinghead import transcribe as module
    from talkinghead.transcribe import ScribeTranscriber

    wav = tmp_path / "a.wav"
    _tone_wav(wav)
    _route_env(monkeypatch, elevenlabs=True, openrouter=True)
    monkeypatch.setattr(module, "_elevenlabs_request", lambda path, key: json.dumps(ROUTE_ELEVENLABS_NATIVE).encode())
    monkeypatch.setattr(module, "_openrouter_request", lambda *a: pytest.fail("fallback used while ElevenLabs answered"))
    got = ScribeTranscriber().transcribe(str(wav))
    assert got.version.endswith(";route=elevenlabs")
    assert [w.text for w in got.words] == ["Hello", "world."]


def test_scribe_falls_back_to_openrouter_when_elevenlabs_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    from talkinghead import transcribe as module
    from talkinghead.transcribe import ProviderUnavailable, ScribeTranscriber

    wav = tmp_path / "a.wav"
    _tone_wav(wav)
    _route_env(monkeypatch, elevenlabs=True, openrouter=True)

    def down(path, key):
        raise ProviderUnavailable("ElevenLabs Scribe request failed: HTTP 429")

    monkeypatch.setattr(module, "_elevenlabs_request", down)
    monkeypatch.setattr(module, "_openrouter_request", lambda path, key: json.dumps(OPENROUTER_NATIVE).encode())
    got = ScribeTranscriber().transcribe(str(wav))
    assert got.version.endswith(";route=openrouter")
    assert [w.text for w in got.words] == ["Hello", "world."]
    assert [e.text for e in got.events] == ["[laughs]"]


def test_scribe_without_an_elevenlabs_key_goes_straight_to_openrouter(tmp_path: Path, monkeypatch) -> None:
    from talkinghead import transcribe as module
    from talkinghead.transcribe import ScribeTranscriber

    wav = tmp_path / "a.wav"
    _tone_wav(wav)
    _route_env(monkeypatch, elevenlabs=False, openrouter=True)
    monkeypatch.setattr(module, "_elevenlabs_request", lambda *a: pytest.fail("ElevenLabs called without a key"))
    monkeypatch.setattr(module, "_openrouter_request", lambda path, key: json.dumps(OPENROUTER_NATIVE).encode())
    assert ScribeTranscriber().transcribe(str(wav)).version.endswith(";route=openrouter")


def test_scribe_never_retries_a_malformed_response_on_the_other_route(tmp_path: Path, monkeypatch) -> None:
    from talkinghead import transcribe as module
    from talkinghead.transcribe import ScribeTranscriber

    wav = tmp_path / "a.wav"
    _tone_wav(wav)
    _route_env(monkeypatch, elevenlabs=True, openrouter=True)
    bad = {"words": [{"type": "word", "text": "x", "start": 5.0, "end": 6.0}]}
    monkeypatch.setattr(module, "_elevenlabs_request", lambda path, key: json.dumps(bad).encode())
    monkeypatch.setattr(module, "_openrouter_request", lambda *a: pytest.fail("malformed response retried on OpenRouter"))
    with pytest.raises(ValueError, match="lies outside"):
        ScribeTranscriber().transcribe(str(wav))


def test_scribe_reports_both_routes_when_both_fail_and_keeps_keys_out(tmp_path: Path, monkeypatch) -> None:
    from talkinghead import transcribe as module
    from talkinghead.transcribe import ProviderUnavailable, ScribeTranscriber

    wav = tmp_path / "a.wav"
    _tone_wav(wav)
    _route_env(monkeypatch, elevenlabs=True, openrouter=True)

    def el_down(path, key):
        raise ProviderUnavailable("ElevenLabs Scribe request failed: HTTP 401")

    def or_down(path, key):
        raise ProviderUnavailable("OpenRouter Scribe request failed: HTTP 402")

    monkeypatch.setattr(module, "_elevenlabs_request", el_down)
    monkeypatch.setattr(module, "_openrouter_request", or_down)
    with pytest.raises(ProviderUnavailable) as err:
        ScribeTranscriber().transcribe(str(wav))
    message = str(err.value)
    assert "HTTP 401" in message and "HTTP 402" in message and "route-test-key" not in message


def test_scribe_refuses_to_run_without_either_key(tmp_path: Path, monkeypatch) -> None:
    from talkinghead import transcribe as module
    from talkinghead.transcribe import ScribeTranscriber

    wav = tmp_path / "a.wav"
    _tone_wav(wav)
    _route_env(monkeypatch, elevenlabs=False, openrouter=False)
    monkeypatch.setattr(module, "_openrouter_request", lambda *a: pytest.fail("network used without a key"))
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        ScribeTranscriber().transcribe(str(wav))


def test_the_cache_keeps_the_route_that_answered(tmp_path: Path, monkeypatch) -> None:
    from talkinghead import transcribe as module
    from talkinghead.transcribe import ScribeTranscriber

    wav = tmp_path / "a.wav"
    _tone_wav(wav)
    cache = Cache(tmp_path / "cache")
    _route_env(monkeypatch, elevenlabs=False, openrouter=True)
    monkeypatch.setattr(module, "_openrouter_request", lambda path, key: json.dumps(OPENROUTER_NATIVE).encode())
    first = transcribe_cached(cache, "sha", ScribeTranscriber(), str(wav))
    assert first.version.endswith(";route=openrouter")
    monkeypatch.setattr(module, "_openrouter_request", lambda *a: pytest.fail("cache miss for the same audio"))
    again = transcribe_cached(cache, "sha", ScribeTranscriber(), str(wav))
    assert again.version == first.version


def test_scribe_with_only_an_elevenlabs_key_calls_elevenlabs_and_never_runs_a_subprocess(
        tmp_path: Path, monkeypatch) -> None:
    from talkinghead import transcribe as module
    from talkinghead.transcribe import SCRIBE_ENDPOINT, ScribeTranscriber

    wav = tmp_path / "a.wav"
    _tone_wav(wav)
    _route_env(monkeypatch, elevenlabs=True, openrouter=False)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("a subprocess ran"))
    sent: list = []

    def answer(request, timeout=None):
        sent.append(request)
        return io.BytesIO(json.dumps(ROUTE_ELEVENLABS_NATIVE).encode())

    monkeypatch.setattr(urllib.request, "urlopen", answer)
    got = ScribeTranscriber().transcribe(str(wav))
    assert got.version.endswith(";route=elevenlabs")
    assert [w.text for w in got.words] == ["Hello", "world."]
    assert sent[0].full_url == SCRIBE_ENDPOINT and sent[0].get_header("Xi-api-key") == "route-test-key"


@pytest.mark.parametrize("key", ["private-test-key\n", "private-test-key\r\n", "private\nkey", "private-key-\u0100"])
def test_invalid_key_format_is_rejected_without_disclosing_it(tmp_path: Path, monkeypatch, key: str) -> None:
    from talkinghead import transcribe as module
    from talkinghead.transcribe import ScribeTranscriber

    wav = tmp_path / "a.wav"
    _tone_wav(wav)
    monkeypatch.setenv("ELEVENLABS_API_KEY", key)
    monkeypatch.setattr(module, "SCRIBE_ENDPOINT", "http://127.0.0.1:1/")
    with pytest.raises(RuntimeError, match="single ASCII token") as err:
        ScribeTranscriber().transcribe(str(wav))
    assert key not in str(err.value)
    assert "private" not in str(err.value)
