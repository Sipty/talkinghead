"""Scribe chunking: the plan, the seams, the merge and the requests. No network: the HTTP call is replaced."""

from __future__ import annotations

import io
import json
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from talkinghead import transcribe as module
from talkinghead.transcribe import (
    OPENROUTER_BODY_LIMIT_BYTES,
    PCM_RATE,
    SCRIBE_BODY_BUDGET_BYTES,
    SCRIBE_MAX_PARALLEL,
    SCRIBE_VERSION,
    WAV_HEADER_BYTES,
    ScribeChunk,
    ScribeRequestError,
    ScribeTranscriber,
    plan_scribe_chunks,
)

S = PCM_RATE
# Small limits keep the tests short: a 25 s request with 5 s of context per side.
# The first and last cores own 20 s each, the middle cores 15 s.
MAX_FRAMES = 25 * S
OVERLAP = 5 * S


def _clamp_seam(lo: int, hi: int, target: int) -> int:
    return min(max(target, lo), hi)


def _plan(n: int, choose=_clamp_seam) -> list[ScribeChunk]:
    return plan_scribe_chunks(n, max_request_frames=MAX_FRAMES, overlap_frames=OVERLAP, choose_seam=choose)


def _assert_valid(chunks: list[ScribeChunk], n: int) -> None:
    assert chunks[0].core_start == 0 and chunks[-1].core_end == n
    for left, right in zip(chunks, chunks[1:]):
        assert left.core_end == right.core_start
    for c in chunks:
        assert c.core_end > c.core_start
        assert c.request_start == max(0, c.core_start - OVERLAP)
        assert c.request_end == min(n, c.core_end + OVERLAP)
        assert c.request_end - c.request_start <= MAX_FRAMES


@pytest.mark.parametrize("seconds, count", [(10, 1), (25, 1), (26, 2), (40, 2), (41, 3), (55, 3), (56, 4)])
def test_the_plan_uses_the_fewest_requests_that_fit(seconds: int, count: int) -> None:
    n = seconds * S
    chunks = _plan(n)
    assert len(chunks) == count
    _assert_valid(chunks, n)


@pytest.mark.parametrize("pick", ["lowest", "highest"])
def test_every_seam_inside_the_offered_bounds_keeps_each_request_within_the_limit(pick: str) -> None:
    n = 50 * S
    chunks = _plan(n, lambda lo, hi, target: lo if pick == "lowest" else hi)
    assert len(chunks) == 3
    _assert_valid(chunks, n)


def test_a_seam_outside_the_offered_bounds_is_refused() -> None:
    with pytest.raises(ValueError, match="outside"):
        _plan(50 * S, lambda lo, hi, target: hi + 1)


def test_the_seams_aim_at_the_even_split() -> None:
    targets: list[int] = []

    def record(lo: int, hi: int, target: int) -> int:
        targets.append(target)
        return _clamp_seam(lo, hi, target)

    _plan(50 * S, record)
    assert targets == [50 * S // 3, 2 * 50 * S // 3]


def test_the_real_budget_fits_a_53_minute_recording_in_three_requests() -> None:
    chunks = plan_scribe_chunks(
        3222 * S, max_request_frames=module._max_request_frames(),
        overlap_frames=module._overlap_frames(), choose_seam=_clamp_seam)
    assert len(chunks) == 3


def test_the_largest_request_body_is_within_the_budget_and_the_gateway_limit() -> None:
    frames = module._max_request_frames()
    body = module._openrouter_body(module._pcm_wav(bytes(2 * frames)))
    assert SCRIBE_BODY_BUDGET_BYTES - 8 <= len(body) <= SCRIBE_BODY_BUDGET_BYTES < OPENROUTER_BODY_LIMIT_BYTES
    assert len(module._pcm_wav(b"")) == WAV_HEADER_BYTES


def _speech(seconds: float, silences: list[tuple[float, float]]) -> np.ndarray:
    """A 300 Hz tone with a little noise, zeroed inside each silence."""
    rng = np.random.default_rng(11)
    t = np.arange(int(seconds * S)) / S
    x = np.round(3000 * np.sin(2 * np.pi * 300 * t) + rng.integers(-100, 101, size=t.size)).astype(np.int16)
    for a, b in silences:
        x[int(a * S): int(b * S)] = 0
    return x


def _write(path: Path, samples: np.ndarray) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(S)
        w.writeframes(samples.astype("<i2").tobytes())
    return path


def test_a_seam_goes_in_the_longest_silence_of_the_window(tmp_path: Path) -> None:
    wav = _write(tmp_path / "speech.wav", _speech(20.0, [(8.0, 8.5), (10.0, 12.0), (15.0, 15.3)]))
    seam = module._SeamFinder(wav)(7 * S, 13 * S, 8 * S + S // 4)
    assert abs(seam / S - 11.0) <= 0.05


def test_with_no_silence_in_the_window_the_seam_goes_at_the_quietest_block(tmp_path: Path) -> None:
    samples = _speech(20.0, [])
    samples[10 * S: 10 * S + S // 20] //= 10  # a 50 ms dip, far above the silence threshold
    wav = _write(tmp_path / "dip.wav", samples)
    seam = module._SeamFinder(wav)(9 * S, 11 * S, 10 * S)
    assert abs(seam / S - 10.025) <= 0.01


def _truth() -> tuple[list[tuple[str, float, float]], list[tuple[str, float, float]]]:
    """Words and one laugh on a 50 s recording. "Hold" straddles the first seam (16.67 s)
    and the laugh straddles the second (31.67 s)."""
    words: list[tuple[str, float, float]] = []
    start, i = 0.2, 0
    while start + 0.4 < 49.5:
        end = start + 0.4
        if end <= 16.2 or 17.2 <= start < 31.0 or start >= 32.2:
            words.append((f"w{i}", start, end))
        start, i = start + 0.9, i + 1
    words.append(("Hold", 16.5, 16.9))
    words.sort(key=lambda w: w[1])
    return words, [("[laughs]", 31.5, 31.9)]


def _scribe_stand_in(monkeypatch: pytest.MonkeyPatch, pcm: bytes, words, events, *,
                     fail_from: float | None = None, bad_from: float | None = None,
                     log: list | None = None, delay: float = 0.0) -> dict:
    """Stands in for OpenRouter. It hears only its request's audio and answers with what lies wholly inside it."""
    lock = threading.Lock()
    active = {"now": 0, "peak": 0}

    def answer(wav_bytes: bytes, key: str) -> bytes:
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            sent = w.readframes(w.getnframes())
        offset = pcm.find(sent)
        assert offset >= 0 and offset % 2 == 0, "a request must carry a slice of the recording"
        start = offset / 2 / S
        end = start + len(sent) / 2 / S
        with lock:
            active["now"] += 1
            active["peak"] = max(active["peak"], active["now"])
            if log is not None:
                log.append((start, end))
        try:
            time.sleep(delay)
            if fail_from is not None and start >= fail_from:
                raise ScribeRequestError("OpenRouter Scribe request failed: HTTP 502")
            items = [{"word": t, "start": round(s - start, 6), "end": round(e - start, 6), "confidence": 0.9}
                     for t, s, e in words if start <= s and e <= end]
            items += [{"type": "audio_event", "word": t, "start": round(s - start, 6), "end": round(e - start, 6)}
                      for t, s, e in events if start <= s and e <= end]
            if bad_from is not None and start >= bad_from:
                items.append({"word": "ghost", "start": end - start + 1.0, "end": end - start + 1.5})
            return json.dumps({"words": items}).encode()
        finally:
            with lock:
                active["now"] -= 1

    monkeypatch.setattr(module, "_openrouter_request", answer)
    return active


@pytest.fixture
def small_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "route-test-key")
    monkeypatch.setattr(module, "_max_request_frames", lambda: MAX_FRAMES)
    monkeypatch.setattr(module, "_overlap_frames", lambda: OVERLAP)
    monkeypatch.setattr(module, "_SeamFinder", lambda wav: _clamp_seam)


def _recording(tmp_path: Path, seconds: float) -> tuple[Path, bytes]:
    rng = np.random.default_rng(3)
    samples = rng.integers(-20000, 20000, size=int(seconds * S), dtype=np.int16)
    return _write(tmp_path / "rec.wav", samples), samples.astype("<i2").tobytes()


def test_a_recording_split_into_three_comes_back_as_one_timeline(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_limits: None) -> None:
    wav, pcm = _recording(tmp_path, 50.0)
    words, events = _truth()
    log: list = []
    _scribe_stand_in(monkeypatch, pcm, words, events, log=log)
    got = ScribeTranscriber().transcribe(str(wav))
    assert len(log) == 3
    assert got.version == f"{SCRIBE_VERSION};route=openrouter;chunks=3"
    assert [w.text for w in got.words] == [t for t, _, _ in words]
    for w, (_, s, e) in zip(got.words, words):
        assert w.start == pytest.approx(s, abs=1e-5) and w.end == pytest.approx(e, abs=1e-5)
    assert [e.text for e in got.events] == ["[laughs]"]
    assert got.events[0].start == pytest.approx(31.5, abs=1e-5)


def test_audio_within_one_request_is_sent_once_and_carries_no_chunk_count(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_limits: None) -> None:
    wav, pcm = _recording(tmp_path, 10.0)
    log: list = []
    _scribe_stand_in(monkeypatch, pcm, [("hello", 1.0, 1.4)], [], log=log)
    got = ScribeTranscriber().transcribe(str(wav))
    assert log == [(0.0, 10.0)]
    assert got.version == f"{SCRIBE_VERSION};route=openrouter"
    assert [w.text for w in got.words] == ["hello"]


def test_a_failed_request_fails_the_whole_transcription(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_limits: None) -> None:
    wav, pcm = _recording(tmp_path, 50.0)
    words, events = _truth()
    _scribe_stand_in(monkeypatch, pcm, words, events, fail_from=11.0)
    with pytest.raises(ScribeRequestError, match=r"HTTP 502 \(chunk \d of 3\)"):
        ScribeTranscriber().transcribe(str(wav))


def test_a_malformed_answer_fails_the_whole_transcription(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_limits: None) -> None:
    wav, pcm = _recording(tmp_path, 50.0)
    words, events = _truth()
    _scribe_stand_in(monkeypatch, pcm, words, events, bad_from=0.0)
    with pytest.raises(ValueError, match=r"lies outside.*\(chunk \d of 3\)"):
        ScribeTranscriber().transcribe(str(wav))


def test_chunk_requests_never_exceed_the_parallel_cap(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_limits: None) -> None:
    wav, pcm = _recording(tmp_path, 50.0)
    words, events = _truth()
    active = _scribe_stand_in(monkeypatch, pcm, words, events, delay=0.05)
    ScribeTranscriber().transcribe(str(wav))
    assert 1 <= active["peak"] <= SCRIBE_MAX_PARALLEL


def test_seams_land_in_the_longest_silence_near_their_targets(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "route-test-key")
    monkeypatch.setattr(module, "_max_request_frames", lambda: MAX_FRAMES)
    monkeypatch.setattr(module, "_overlap_frames", lambda: OVERLAP)
    samples = _speech(50.0, [(15.5, 16.5), (30.2, 30.7)])
    wav = _write(tmp_path / "talk.wav", samples)
    log: list = []
    _scribe_stand_in(monkeypatch, samples.astype("<i2").tobytes(), [], [], log=log)
    ScribeTranscriber().transcribe(str(wav))
    # Seams fall at 16.0 s and 30.45 s, and each request runs 5 s past its core.
    expected = [(0.0, 21.0), (11.0, 35.45), (25.45, 50.0)]
    for (s, e), (es, ee) in zip(sorted(log), expected):
        assert s == pytest.approx(es, abs=0.05) and e == pytest.approx(ee, abs=0.05)
