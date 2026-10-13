"""Voice isolation: latency measurement, dropout repair, the plan record and the memo keys it must leave alone."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from talkinghead import isolate as I
from talkinghead.config import ISOLATE_KEYS, Config
from talkinghead.pipeline import STAGE_CONFIG_KEYS, _stage_config
from talkinghead.timeline import Cut, Plan, SourceInfo
from talkinghead.transcribe import Word, prepare_audio
from talkinghead.verify import ISOLATION_RESIDUAL_MAX, _isolation_alignment, _isolation_words

RATE = I.RATE


def _noise(seconds: float, seed: int = 0) -> np.ndarray:
    """Stationary stereo noise: every stretch is voiced, so any window correlates."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((round(seconds * RATE), 2)) * 0.2).astype(np.float32)


def _speechlike(seconds: float, seed: int = 0) -> np.ndarray:
    """Stereo noise in 0.8 s bursts with pauses: loud, varying, and easy to correlate."""
    rng = np.random.default_rng(seed)
    n = round(seconds * RATE)
    gate = np.zeros(n)
    t = 0
    while t < n:
        on = round(rng.uniform(0.5, 0.9) * RATE)
        gate[t:t + on] = 1.0
        t += on + round(rng.uniform(0.2, 0.5) * RATE)
    return (rng.standard_normal((n, 2)) * gate[:, None] * 0.2).astype(np.float32)


def _delayed(orig: np.ndarray, lag: int, tail: int) -> np.ndarray:
    return np.vstack([np.zeros((lag, orig.shape[1]), np.float32), orig, np.zeros((tail, orig.shape[1]), np.float32)])


def _single_segment_plan(seconds: float) -> Plan:
    src = SourceInfo(id="main", path="/nowhere/master.mov", sha256="deadbeef", duration=seconds)
    return Plan.from_cuts(src, [])


def test_latency_is_measured_on_a_synthetic_delay():
    orig = _speechlike(60.0)
    raw = _delayed(orig, 3664, RATE)
    lag, points = I.measure_lag(orig, raw)
    assert lag == 3664
    assert len(points) == I.LAG_POINTS
    assert all(p.lag == 3664 and p.corr > 0.9 for p in points)


def test_a_latency_that_changes_across_the_recording_stops_the_run():
    orig = _noise(60.0)
    half = len(orig) // 2
    zeros = lambda n: np.zeros((n, 2), np.float32)  # noqa: E731
    # The first half comes out 3664 samples late, the second 7364: no single trim fits both.
    raw = np.vstack([zeros(3664), orig[:half], zeros(3700), orig[half:], zeros(RATE)])
    with pytest.raises(I.IsolationError, match="not constant"):
        I.measure_lag(orig, raw)


def test_a_track_that_does_not_follow_the_source_is_refused():
    orig = _noise(60.0)
    other = _noise(60.0, seed=7)
    with pytest.raises(I.IsolationError, match="does not follow"):
        I.measure_lag(orig, _delayed(other, 3664, RATE))


def test_a_dropout_word_is_flagged_and_spliced_back_with_the_original(tmp_path):
    orig = _noise(20.0, seed=3)
    # Eight words, each 1.2 s long, 2 s apart; the isolated track keeps 60% of the level, except word 5.
    words = [Word(f"w{i}", 0.5 + i * 2.0, 1.7 + i * 2.0, 1.0) for i in range(8)]
    iso = orig * 0.6
    dead = words[5]
    iso[round(dead.start * RATE):round(dead.end * RATE)] *= 0.02
    plan = _single_segment_plan(20.0)

    repaired, gain = I.find_dropouts(orig, iso, plan, words, floor_db=-120.0)

    assert [r["text"] for r in repaired] == ["w5"]
    assert gain == pytest.approx(0.6, rel=0.05)
    out = tmp_path / "track.wav"
    I.write_track(orig, iso, 0, mix=1.0, regions=I._regions(repaired), gain=gain, out=out)
    track = np.asarray(I._memmap(out))
    xf = round(I.XFADE_SEC * RATE)
    inside = slice(round(dead.start * RATE) + xf // 2, round(dead.end * RATE) - xf // 2)
    # Inside the dropout the track is the original, scaled to the isolated level.
    assert np.allclose(track[inside], orig[inside] * gain, atol=1e-5)
    # Away from it the isolated audio is kept as it was.
    clean = slice(round(words[1].start * RATE), round(words[1].end * RATE))
    assert np.allclose(track[clean], iso[clean], atol=1e-5)


def test_crossfades_are_equal_power_and_stay_within_their_ramp():
    regions = [(48000, 96000)]
    w = I._orig_weight(regions, 0, 144000)
    xf = round(I.XFADE_SEC * RATE)
    assert w.min() >= 0.0 and w.max() == pytest.approx(1.0)
    assert w[48000 - xf // 2 - 1] == 0.0 and w[48000 + xf // 2] == pytest.approx(1.0)
    gi, go = np.cos(np.pi / 2 * w), np.sin(np.pi / 2 * w)
    assert np.allclose(gi ** 2 + go ** 2, 1.0)


def test_isolation_knobs_leave_every_editorial_memo_key_alone():
    off, on = Config(), Config(isolate_voice=True, isolate_mix=0.5)
    ctx_off, ctx_on = SimpleNamespace(config=off), SimpleNamespace(config=on)
    for stage in STAGE_CONFIG_KEYS:
        assert _stage_config(ctx_on, stage) == _stage_config(ctx_off, stage), stage
    # The whole-config stages (visuals and the rest) drop the knobs too, so toggling them re-keys nothing.
    assert _stage_config(ctx_on, "visuals") == _stage_config(ctx_off, "visuals")
    assert all(k not in _stage_config(ctx_on, "visuals")["config"] for k in ISOLATE_KEYS)


def test_isolation_knobs_are_absent_from_provenance_until_enabled():
    assert not any(k in Config().to_dict() for k in ISOLATE_KEYS)
    d = Config(isolate_voice=True, isolate_mix=0.25).to_dict()
    assert d["isolate_voice"] is True and d["isolate_mix"] == 0.25


def test_isolate_mix_must_be_a_fraction():
    with pytest.raises(ValueError, match="isolate_mix"):
        Config(isolate_voice=True, isolate_mix=1.5).validate()


def test_plan_without_isolation_serialises_as_before_and_round_trips_with_it():
    plan = _single_segment_plan(10.0)
    assert "isolation" not in plan.to_json()
    plan.isolation = {"mix": 1.0, "lag_samples": 3664, "track": {"path": "/x.wav", "sha256": "ab"}}
    again = Plan.from_json(plan.to_json())
    assert again.isolation == plan.isolation


def test_timing_reads_the_track_only_for_the_main_source():
    plan = _single_segment_plan(10.0)
    assert I.timing_audio(plan, "main") == "/nowhere/master.mov"
    plan.isolation = {"track": {"path": "/track.wav", "sha256": "ab"}}
    assert I.timing_audio(plan, "main") == "/track.wav"


def test_isolation_alignment_fails_on_a_residual_past_one_sample():
    plan = _single_segment_plan(10.0)
    plan.isolation = {"lag_samples": 3664}
    rows = lambda *off: [{"at": float(i), "residual_samples": d, "corr": 0.95, "qualified": True}  # noqa: E731
                         for i, d in enumerate(off)]
    ok = {"isolation": {"lag_samples": 3664, "residual": rows(1, 0, 1, 0), "max_abs_residual": 1}}
    bad = {"isolation": {"lag_samples": 3664, "residual": rows(2, 0, 0, 0), "max_abs_residual": 2}}
    assert _isolation_alignment(plan, ok).ok
    result = _isolation_alignment(plan, bad)
    assert not result.ok and result.severity == "fatal"
    assert ISOLATION_RESIDUAL_MAX == 1


def test_isolation_words_warns_when_the_render_lost_speech():
    plan = _single_segment_plan(10.0)
    short = _isolation_words(plan, {"isolation": {"rendered_words": 70, "expected_words": 100}})
    assert not short.ok and short.severity == "warn"
    assert _isolation_words(plan, {"isolation": {"rendered_words": 98, "expected_words": 100}}).ok


def test_isolation_checks_are_the_alignment_and_word_checks():
    from talkinghead.verify import CHECKS, ISOLATION_CHECKS

    assert [c.__name__ for c in ISOLATION_CHECKS] == ["_isolation_alignment", "_isolation_words"]
    assert not set(ISOLATION_CHECKS) & set(CHECKS)


def test_a_cut_word_is_not_a_repair_candidate():
    orig = _noise(10.0, seed=5)
    plan = Plan.from_cuts(SourceInfo(id="main", path="/x", sha256="s", duration=10.0),
                          [Cut(2.0, 4.0, "content")])
    words = [Word("gone", 2.2, 3.0, 1.0), Word("kept", 5.0, 5.6, 1.0)]
    iso = orig * 0.5
    repaired, gain = I.find_dropouts(orig, iso, plan, words, floor_db=-120.0)
    assert repaired == [] and gain == pytest.approx(0.5, rel=0.05)


@pytest.mark.skipif(sys.platform != "darwin", reason="AUSoundIsolation is an Apple audio unit")
def test_end_to_end_on_a_speech_recording(tmp_path):
    """The real helper: its latency is trimmed, the track sits on the source timeline, and a rebuild reproduces it.

    Needs a recording of real speech (AUSoundIsolation keeps speech, not synthetic noise): set
    TALKINGHEAD_ISOLATION_SAMPLE to a WAV or video of 60 s or more, or the test is skipped.
    """
    import os

    sample = os.environ.get("TALKINGHEAD_ISOLATION_SAMPLE")
    if not sample:
        pytest.skip("set TALKINGHEAD_ISOLATION_SAMPLE to a recording of speech to run the helper end to end")
    master = tmp_path / "master.wav"
    I.decode_input(sample, I._channels(sample), master)
    _, frames, _ = I._wav_data(master)
    plan = Plan.from_cuts(SourceInfo(id="main", path=str(master), sha256=I.sha256_file(master),
                                     duration=frames / RATE), [])
    scratch = tmp_path / "scratch"
    record = I.ensure(plan, [], floor_db=-120.0, mix=1.0, track=tmp_path / "isolated.wav",
                      cache_root=tmp_path / "cache", scratch=scratch)
    plan.isolation = record
    facts = I.verify_facts(plan, expected_words=0, rendered_words=0, scratch=scratch)
    assert facts["max_abs_residual"] <= ISOLATION_RESIDUAL_MAX
    rebuilt = I.audio_path(plan, cache_root=tmp_path / "cache", scratch=scratch)
    assert rebuilt == str(tmp_path / "isolated.wav")
    (tmp_path / "isolated.wav").unlink()
    assert I.audio_path(plan, cache_root=tmp_path / "cache", scratch=scratch) == rebuilt
    assert I.sha256_file(rebuilt) == record["track"]["sha256"]


def test_isolation_re_keys_the_render_and_nothing_editorial(tmp_path, monkeypatch):
    """Turning isolation on re-renders the audio but re-judges and re-transcribes nothing editorial.

    The unit itself is stubbed here (its output on a six-second tone is not speech): the track is the original
    with zero latency, so the wiring is what is under test: the record, the render through the track, the checks.
    """
    from talkinghead.judge import FakeJudge
    from talkinghead.pipeline import run_pipeline
    from talkinghead.store import Cache

    from .test_pipeline import PlanAwareTranscriber, _answers, _cfg, _clip

    def unit_stub(plan, words, *, floor_db, mix, track, cache_root, scratch):
        src = plan.sources[plan.main]
        scratch.mkdir(parents=True, exist_ok=True)
        wav = I.decode_input(src.path, I._channels(src.path), scratch / "input.wav")
        orig = I._memmap(wav)
        digest = I.write_track(orig, orig, 0, mix=mix, regions=[], gain=1.0, out=track)
        return {"source_sha256": src.sha256, "helper": "stub", "version": I.ISOLATE_VERSION, "mix": mix,
                "rate": RATE, "channels": orig.shape[1], "lag_samples": 0, "lag_points": [], "gain": 1.0,
                "repaired": [], "track": {"path": str(track), "sha256": digest, "frames": orig.shape[0]}}

    def residual_stub(plan, *, expected_words, rendered_words, scratch):
        # The six-second clip is shorter than the latency measurement needs; its track is the original, so 0 is exact.
        return {"lag_samples": 0, "residual": [{"at": 0.0, "residual_samples": 0, "corr": 1.0, "qualified": True}],
                "max_abs_residual": 0, "expected_words": expected_words, "rendered_words": rendered_words}

    def silence_stub(source, source_sha, *, mix, cache_root, scratch):
        # The analysis track: the 16 kHz source (no unit on a six-second tone). Silence detection reads it as usual.
        return Path(prepare_audio(source, scratch)[0])

    class Recording(PlanAwareTranscriber):
        def __init__(self, run_dir):
            super().__init__(run_dir)
            self.wavs: list[str] = []

        def transcribe(self, wav_path):
            self.wavs.append(wav_path)
            return super().transcribe(wav_path)

    monkeypatch.setattr(I, "ensure", unit_stub)
    monkeypatch.setattr(I, "verify_facts", residual_stub)
    monkeypatch.setattr(I, "silence_track", silence_stub)
    clip = tmp_path / "clip.mp4"
    _clip(clip)
    out = tmp_path / "out"
    cache = Cache(tmp_path / "cache")
    first = run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out),
                         judge=FakeJudge(_answers()), cache=cache, guidelines="G")
    plain_plan = Plan.from_json(first.plan_path.read_text())
    assert not plain_plan.isolation

    keys = lambda: {p.stem: json.loads(p.read_text())["key"]  # noqa: E731
                    for p in (out / "stages").glob("*.json") if ".superseded" not in p.name}
    before = keys()
    strict = FakeJudge([])  # any editorial call would raise StopIteration
    iso_cfg = Config(**{**_cfg().__dict__, "isolate_voice": True, "isolate_mix": 1.0})
    recorder = Recording(out)
    second = run_pipeline(str(clip), out, iso_cfg, transcriber=recorder, judge=strict,
                          cache=cache, guidelines="G")
    assert strict.calls == 0
    # The judged transcript is the original-audio render; the isolated render is transcribed for its own count only.
    assert any("judge" in w for w in recorder.wavs) and any("verify" in w for w in recorder.wavs)
    assert second.facts["isolated_words"] is not None
    # Only the pauses memo re-keys on isolation (its silence now comes from the isolated track); the editorial
    # stages above it keep their memo keys, so a cook that turns isolation on replays them.
    after = keys()
    assert after["pauses"] != before["pauses"]
    for stage in ("structure", "retakes", "content"):
        assert after[stage] == before[stage], stage
    plan = Plan.from_json(second.plan_path.read_text())
    assert plan.isolation and plan.isolation["track"]["sha256"]
    assert [(c.start, c.end, c.reason) for c in plan.cuts] == \
        [(c.start, c.end, c.reason) for c in plain_plan.cuts]
    assert "## Isolation" in second.report.read_text()
    assert second.facts["isolation"]["max_abs_residual"] == 0
    assert "isolation_alignment" in {c.name for c in second.checks}
    assert second.ok


def test_off_macos_the_unit_is_refused_before_any_work(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(I.IsolationError, match="needs macOS"):
        I.preflight(tmp_path)
    assert not (tmp_path / "bin").exists()


def test_the_cli_takes_the_isolation_flags():
    from talkinghead.cli import build_parser

    args = build_parser().parse_args(["run", "x.mov", "--isolate-voice", "--isolate-mix", "0.5"])
    assert args.isolate_voice is True and args.isolate_mix == 0.5
    plain = build_parser().parse_args(["run", "x.mov"])
    assert plain.isolate_voice is False and plain.isolate_mix is None


def test_a_word_spanning_a_phrase_is_not_repaired():
    """A stamp over a run of speech is a bad timing, not a dropout: it is never spliced."""
    orig = _noise(20.0, seed=9)
    long_word = Word("phrase", 2.0, 2.0 + I.REPAIR_MAX_SEC + 0.5, 1.0)
    plan = _single_segment_plan(20.0)
    iso = orig * 0.01
    repaired, _ = I.find_dropouts(orig, iso, plan, [long_word], floor_db=-120.0)
    assert repaired == []


def test_frame_percentiles_read_the_room_tone_and_count_gated_frames():
    rng = np.random.default_rng(4)
    # Room tone at about -40 dBFS RMS for 8 s, then 2 s of digital silence.
    tone = (rng.standard_normal(8 * RATE) * 0.01).astype(np.float32)
    x = np.concatenate([tone, np.zeros(2 * RATE, np.float32)])
    got = I.frame_percentiles(x)
    assert got["p10"] < -100  # digital silence sits in the low tail
    assert got["gated_share"] == pytest.approx(0.2, abs=0.01)
    assert got["p50"] == pytest.approx(-40.0, abs=0.5)  # median is the tone, not the silence
    stereo = np.stack([x, x], axis=1)
    assert I.frame_percentiles(stereo) == got  # two equal channels measure as one
    with pytest.raises(I.IsolationError, match="too short"):
        I.frame_percentiles(np.zeros(10, np.float32))


def _shifted_track(orig, lag):
    return np.vstack([np.zeros((lag, orig.shape[1]), np.float32), orig])[:len(orig)]


def test_a_short_recording_with_a_known_delay_is_measured_without_a_stub():
    """Recordings of a few seconds still get a latency: the windows shrink to fit (no stubbing of measure_lag)."""
    for seconds in (4.0, 12.0):
        orig = _noise(seconds, seed=21)
        raw = _delayed(orig, 3664, RATE)
        lag, points = I.measure_lag(orig, raw)
        assert lag == 3664, seconds
        assert len(points) == I.LAG_POINTS


def test_a_recording_too_short_for_any_window_is_refused():
    with pytest.raises(I.IsolationError, match="too short"):
        I.measure_lag(_noise(0.5), _delayed(_noise(0.5), 3664, RATE))


def test_a_silent_check_window_is_not_an_alignment_failure():
    """Correctly aligned speech with a silent stretch: the silent point is discounted, the track passes."""
    from talkinghead.verify import _isolation_alignment

    orig = _noise(20.0, seed=31)
    orig[round(9.0 * RATE):round(13.0 * RATE)] = 0.0  # silence across the middle of the recording
    track = orig.copy()
    window = round(2.0 * RATE)
    silent_start = round(9.5 * RATE)  # a check window wholly inside the silence
    loud_starts = [round(s * RATE) for s in (1.0, 2.0, 3.0, 5.0, 15.0)]
    rows = I.residual_lags(orig, track, window, [silent_start, *loud_starts])
    assert not rows[0]["qualified"] and rows[0]["corr"] < I.LAG_STRONG_CORR  # the silent point is recognised
    assert all(r["qualified"] and r["residual_samples"] == 0 for r in rows[1:])
    plan = _single_segment_plan(20.0)
    plan.isolation = {"lag_samples": 3664}
    result = _isolation_alignment(plan, {"isolation": {"lag_samples": 3664, "residual": rows,
                                                       "max_abs_residual": 0}})
    assert result.ok, result.detail


def test_a_real_offset_on_qualified_points_still_fails_the_alignment():
    from talkinghead.verify import _isolation_alignment

    orig = _noise(20.0, seed=32)
    track = _shifted_track(orig, 3)  # the track is three samples late everywhere
    window, starts = I._window_layout(orig)
    rows = I.residual_lags(orig, track, window, starts)
    assert all(r["qualified"] for r in rows)
    plan = _single_segment_plan(20.0)
    plan.isolation = {"lag_samples": 3664}
    result = _isolation_alignment(plan, {"isolation": {"lag_samples": 3664, "residual": rows,
                                                       "max_abs_residual": 3}})
    assert not result.ok and result.severity == "fatal"


def test_a_rebuild_from_the_record_writes_the_same_bytes_with_a_repair_and_an_odd_gain(tmp_path, monkeypatch):
    """A missing track is rebuilt from the plan's record: the repaired samples must come back identical.

    The unit is replaced by a prepared raw track whose level is 0.61234567 of the original, so the median gain
    has more than four decimals, and one word is a dropout that gets spliced.
    """
    orig = _noise(30.0, seed=41)
    raw = _delayed(orig * np.float32(0.61234567), 3664, RATE)
    raw[round(10.0 * RATE):round(10.8 * RATE)] *= np.float32(0.01)  # the dropout
    master = tmp_path / "master.wav"
    master.write_bytes(I._wav_header(2, len(orig)) + np.ascontiguousarray(orig, dtype="<f4").tobytes())
    raw_file = tmp_path / "raw.wav"
    raw_file.write_bytes(I._wav_header(2, len(raw)) + np.ascontiguousarray(raw, dtype="<f4").tobytes())
    monkeypatch.setattr(I, "raw_isolation", lambda wav_in, sha, cache: raw_file)

    words = [Word("w", 10.0, 10.8, 1.0)] + [Word(f"v{i}", 1.0 + i * 3.0, 2.0 + i * 3.0, 1.0) for i in range(8)]
    plan = Plan.from_cuts(SourceInfo(id="main", path=str(master), sha256=I.sha256_file(master), duration=30.0), [])
    track = tmp_path / "isolated.wav"
    record = I.ensure(plan, words, floor_db=-120.0, mix=1.0, track=track, cache_root=tmp_path / "cache",
                      scratch=tmp_path / "scratch")
    assert [r["text"] for r in record["repaired"]] == ["w"]
    assert record["gain"] == pytest.approx(0.6123, abs=1e-4)  # the unrounded median is 0.6123457
    plan.isolation = record
    first = I.sha256_file(track)
    track.unlink()
    I.audio_path(plan, cache_root=tmp_path / "cache", scratch=tmp_path / "scratch")
    assert I.sha256_file(track) == first == record["track"]["sha256"]


def _room_tone_master(tmp_path, seconds=30.0):
    """Speech bursts over a 4 s stretch of room tone at about -34 dBFS, as a master WAV and its raw unit output.

    The unit keeps the speech and gates the room tone to near-digital silence (what AUSoundIsolation does between
    phrases on the restaurant recording), delayed by the measured latency.
    """
    rng = np.random.default_rng(51)
    n = round(seconds * RATE)
    speech = (rng.standard_normal((n, 2)) * 0.2).astype(np.float32)
    gate = np.zeros(n, np.float32)
    t = 0
    while t < n:  # speech-like bursts, 0.6 s on, 0.4 s off, everywhere except the room-tone stretch
        on = round(0.6 * RATE)
        gate[t:t + on] = 1.0
        t += on + round(0.4 * RATE)
    room = slice(round(12.0 * RATE), round(16.0 * RATE))
    gate[room] = 0.0
    orig = speech * gate[:, None]
    orig[room] = (rng.standard_normal((room.stop - room.start, 2)) * 0.02).astype(np.float32)  # about -34 dBFS
    raw = orig.copy()
    raw[room] *= np.float32(1e-4)  # the unit's gated room tone
    raw = _delayed(raw, 3664, RATE)
    master = tmp_path / "master.wav"
    master.write_bytes(I._wav_header(2, len(orig)) + np.ascontiguousarray(orig, dtype="<f4").tobytes())
    raw_file = tmp_path / "raw.wav"
    raw_file.write_bytes(I._wav_header(2, len(raw)) + np.ascontiguousarray(raw, dtype="<f4").tobytes())
    return master, raw_file, room


def test_room_tone_the_unit_gates_is_silence_on_the_isolated_track(tmp_path, monkeypatch):
    """Root of the render_silence regression: the pauses stage must see the isolated track's silence.

    The original's room tone reads as speech level, so detect_silences keeps it; the isolated track's gated room
    tone is silence, and detect_silences reports it. The pauses stage reads the isolated silences.
    """
    from talkinghead.transcribe import detect_silences

    master, raw_file, room = _room_tone_master(tmp_path)
    monkeypatch.setattr(I, "raw_isolation", lambda wav_in, sha, cache_root: raw_file)
    cache_root = tmp_path / "cache"
    original = I.silence_track(str(master), I.sha256_file(master), mix=0.0, cache_root=cache_root,
                               scratch=tmp_path / "s0")
    isolated = I.silence_track(str(master), I.sha256_file(master), mix=1.0, cache_root=cache_root,
                               scratch=tmp_path / "s1")
    assert Path(original).is_file() and Path(isolated).is_file()

    def covered(wav):
        spans = detect_silences(str(wav))
        lo, hi = room.start / RATE, room.stop / RATE
        return sum(max(0.0, min(e, hi) - max(s, lo)) for s, e in spans)

    assert covered(original) < 0.5  # the original's room tone is not silence to the detector
    assert covered(isolated) > 3.0  # the isolated track's room tone is silence: most of its 4 seconds


def test_isolated_silence_track_is_cached_and_16k_mono(tmp_path, monkeypatch):
    import wave

    master, raw_file, _ = _room_tone_master(tmp_path, seconds=20.0)
    monkeypatch.setattr(I, "raw_isolation", lambda wav_in, sha, cache_root: raw_file)
    path = I.silence_track(str(master), I.sha256_file(master), mix=1.0, cache_root=tmp_path / "c",
                           scratch=tmp_path / "s")
    with wave.open(str(path)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
        assert abs(w.getnframes() - 20 * 16000) <= 16000 // 10
    again = I.silence_track(str(master), I.sha256_file(master), mix=1.0, cache_root=tmp_path / "c",
                            scratch=tmp_path / "s2")
    assert again == path  # content-addressed: the same source, helper and mix reuse the file
