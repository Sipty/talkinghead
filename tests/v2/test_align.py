"""Alignment onto the video master: offset and drift on synthetic signals,
rotation decisions, and remux smoke tests on generated fixtures.

No real footage and no network: fixtures are ffmpeg-generated speech-like
noise in tmp_path, and the estimator tests never touch ffmpeg at all.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from talkinghead.align import (
    ANALYSIS_SR,
    AlignError,
    AlignResult,
    _measure_arrays,
    _remux_command,
    _samples,
    align,
    analyze,
)
from talkinghead.color import hdr_codec, hdr_pix_fmt

SR = ANALYSIS_SR


def speech_signal(duration_s: float, seed: int = 0) -> np.ndarray:
    """Speech-like signal: broadband bursts with a random pause pattern.

    Deliberately non-periodic: the envelope must not make cross-correlation
    ambiguous the way sine modulation would.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * SR)
    env = np.ones(n)
    # Random pauses (0.5-3 s) every 2-8 s, like a person breathing.
    pos = 0
    while pos < n:
        pos += int(rng.uniform(2.0, 8.0) * SR)
        gap = int(rng.uniform(0.5, 3.0) * SR)
        env[pos: pos + gap] = 0.0
        pos += gap
    # Fast syllable flutter on top.
    t = np.arange(n) / SR
    env *= 0.6 + 0.4 * np.sin(2 * np.pi * 2 * t + rng.uniform(0, 6.28))
    return env * rng.standard_normal(n) * 0.3


def make_pair_arrays(video_len: float, audio_len: float, offset: float,
                     drift: float = 0.0, seed: int = 1) -> tuple[np.ndarray, np.ndarray, float]:
    """In-memory pair with the fixture semantics.

    A hears speech from its t=0; B contains speech from speech-time |offset|
    (offset > 0 means B joined late). With drift, B's clock runs fast by that
    factor: B time t holds content A has at t*(1+drift).
    """
    total = max(video_len, abs(offset) + audio_len + 1.0)
    speech = speech_signal(total, seed=seed)
    if offset < 0:
        # A joined late: its t=0 holds speech-time |offset|.
        a = speech[int(abs(offset) * SR): int(abs(offset) * SR) + int(video_len * SR)].copy()
    else:
        a = speech[: int(video_len * SR)].copy()
    lo = int(offset * SR) if offset > 0 else 0
    n_b = int(audio_len * SR)
    if drift:
        # Stretch a speech slice by (1+drift) so B's clock runs fast.
        seg = speech[lo: lo + n_b + int(n_b * drift) + 2]
        idx = np.linspace(0, seg.size - 1, n_b)
        b = np.interp(idx, np.arange(seg.size), seg)
    else:
        b = speech[lo: lo + n_b].copy()
    return a, b, offset


# ------------------------------------------------------------- estimator


def test_offset_exact_no_drift():
    a, b, expected = make_pair_arrays(video_len=120, audio_len=100, offset=13.7)
    result = _measure_arrays(a, b)
    assert abs(result.offset - expected) < 0.01
    assert abs(result.drift) < 50e-6
    assert result.confidence is not None


def test_offset_negative_obs_started_early():
    # The external recording begins before the video master's content does.
    a, b, expected = make_pair_arrays(video_len=120, audio_len=140, offset=-15.0)
    result = _measure_arrays(a, b)
    assert abs(result.offset - expected) < 0.01


def test_drift_detected_and_offset_unbiased():
    # 60 ppm drift over a 100 s recording: the lag at the far end shifts ~6 ms.
    drift = 60e-6
    a, b, expected = make_pair_arrays(video_len=120, audio_len=100, offset=13.7, drift=drift)
    result = _measure_arrays(a, b)
    assert abs(result.drift - drift) < 30e-6
    assert abs(result.offset - expected) < 0.05


def test_quiet_windows_dont_break_alignment():
    a, b, expected = make_pair_arrays(video_len=120, audio_len=100, offset=13.7, seed=2)
    # Insert 43 s of silence into b so whole GCC-PHAT windows land inside it.
    b[int(35 * SR): int(78 * SR)] = 0.0
    result = _measure_arrays(a, b)
    assert abs(result.offset - expected) < 0.01
    assert result.windows_used < result.windows_total


def test_identical_silence_reports_no_confidence():
    # Two silent tracks share no signal: the coarse peak is not positive, and a
    # caller must be told not to trust the returned offset.
    z = np.zeros(int(60 * SR))
    assert _measure_arrays(z, z).confidence is None


def test_repeated_pattern_still_finds_match():
    # Adversarial: a 10 s period signal makes naive envelope correlation ambiguous.
    speech = speech_signal(30.0, seed=3)
    tiled = np.tile(speech, 5)
    a = np.zeros(int(7.0 * SR) + tiled.size + SR)
    a[int(7.0 * SR): int(7.0 * SR) + tiled.size] = tiled
    result = _measure_arrays(a, tiled)
    assert abs(result.offset - 7.0) < 0.05


# --------------------------------------------------------------- rotation


def _result(**kw) -> AlignResult:
    base = dict(offset_sec=3.0, drift_ppm=0.0, confidence=10.0,
                video_duration=60.0, audio_duration=50.0)
    base.update(kw)
    return AlignResult(**base)


def _cmd(result: AlignResult, *, fix_drift: bool = False) -> list[str]:
    return _remux_command(Path("v.mov"), Path("a.mp4"), Path("o.mov"), result, fix_drift=fix_drift)


def _graph(result: AlignResult, *, fix_drift: bool = False) -> str:
    cmd = _cmd(result, fix_drift=fix_drift)
    return cmd[cmd.index("-filter_complex") + 1]


def test_no_rotation_keeps_the_stream_copy():
    cmd = _cmd(_result(video_info={"rotation": 0, "pix_fmt": "yuv420p10le"}))
    assert "-c:v" in cmd and cmd[cmd.index("-c:v") + 1] == "copy"
    assert "hevc_videotoolbox" not in cmd  # a copied stream needs no codec args
    assert "-display_rotation" not in cmd  # nothing to strip
    assert "[v]" not in cmd[cmd.index("-filter_complex") + 1]


def test_rotation_minus_ninety_bakes_in_clockwise_upright():
    cmd = _cmd(_result(video_info={"rotation": -90, "pix_fmt": "yuv420p10le",
                                   "color_transfer": "arib-std-b67"}))
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "transpose=1" in graph and "transpose=2" not in graph
    for flag, value in (("-c:v", hdr_codec()), ("-profile:v", "main10"),
                        ("-pix_fmt", hdr_pix_fmt(hdr_codec())), ("-tag:v", "hvc1"),
                        ("-color_trc", "arib-std-b67")):
        assert cmd[cmd.index(flag) + 1] == value
    assert "-display_rotation" in cmd  # input side data must not ride through


def test_rotation_plus_ninety_maps_to_counter_clockwise():
    cmd = _cmd(_result(video_info={"rotation": 90, "pix_fmt": "yuv420p"}))
    assert "transpose=2" in cmd[cmd.index("-filter_complex") + 1]
    assert cmd[cmd.index("-c:v") + 1] == "libx264"


def test_rotation_180_maps_to_the_flip_pair():
    cmd = _cmd(_result(video_info={"rotation": 180, "pix_fmt": "yuv420p"}))
    assert "hflip,vflip" in cmd[cmd.index("-filter_complex") + 1]


def test_unsupported_rotation_is_an_error():
    with pytest.raises(AlignError, match="unsupported display rotation"):
        _cmd(_result(video_info={"rotation": 45, "pix_fmt": "yuv420p"}))


def test_offset_sign_picks_delay_or_trim():
    graph = _graph(_result(offset_sec=3.25))
    assert "adelay=3250:all=1" in graph and "atrim" not in graph
    graph = _graph(_result(offset_sec=-3.25))
    assert "atrim=start=3.250000" in graph and "adelay" not in graph
    assert "apad" in graph


def test_fix_drift_stretches_by_the_measured_clock_ratio():
    assert "atempo" not in _graph(_result(drift_ppm=60.0))  # opt-in only
    assert "atempo=0.999940004" in _graph(_result(drift_ppm=60.0), fix_drift=True)  # 1/(1+60ppm)
    graph = _graph(_result(drift_ppm=60.0, offset_sec=-3.0), fix_drift=True)
    # The stretch must come after the trim and before the pad/delay.
    assert graph.index("atempo") > graph.index("atrim") > -1


# ------------------------------------------------------------ end to end


def _pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()


def _write_av(path: Path, video_src: str, samples: np.ndarray, duration: float) -> None:
    """Mux a lavfi video source with raw PCM audio, both trimmed to `duration`."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", video_src,
         "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-",
         "-c:v", "libx264", "-preset", "ultrafast",
         "-c:a", "aac", "-t", f"{duration}", str(path)],
        input=_pcm16(samples[: int(duration * SR)]), check=True,
    )


def _attach_audio(src: Path, dst: Path, samples: np.ndarray, duration: float) -> None:
    """Copy `src`'s video stream untouched and give it `samples` as its own mic track."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src),
         "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-",
         "-c:v", "copy", "-c:a", "aac", "-t", f"{duration}", str(dst)],
        input=_pcm16(samples[: int(duration * SR)]), check=True,
    )


def _make_pair(tmp_path: Path, video_len: float, audio_len: float, offset: float,
               seed: int = 1) -> tuple[Path, Path]:
    """(master, external): the master heard the speech from its t=0, the external
    recording joined `offset` seconds later (negative: it started early)."""
    speech = speech_signal(max(video_len, abs(offset) + audio_len) + 2.0, seed=seed)
    video = tmp_path / "master.mp4"
    _write_av(video, f"testsrc=duration={video_len}:size=320x240:rate=30",
              speech[int(max(-offset, 0.0) * SR):], video_len)
    audio = tmp_path / "external.mp4"
    _write_av(audio, f"color=black:duration={audio_len}:size=320x240:rate=30",
              speech[int(max(offset, 0.0) * SR):], audio_len)
    return video, audio


def _probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height,side_data_list",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"][0]


def _duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def _track_durations(path: Path) -> tuple[float, float]:
    """(video, audio) track durations: a gap in one track shows up here, but
    not in the container duration."""
    def one(stream: str) -> float:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", stream, "-show_entries", "stream=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True).stdout
        return float(out.strip().splitlines()[0])

    return one("v:0"), one("a:0")


def _video_frames_md5(path: Path) -> str:
    """MD5 of the decoded video frames: identical bits iff the stream was copied."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "md5", "-"],
        capture_output=True, text=True, check=True).stdout
    return out.strip().split("=")[-1]


def _audio(path: Path, workdir: Path) -> np.ndarray:
    return _samples(path, workdir)


def _corr_at_offset(out_audio: np.ndarray, src_audio: np.ndarray, offset: float) -> float:
    """Correlation of the loudest source second against the output `offset` later.

    The loudest window is used on purpose: constant (silent) segments make
    corrcoef NaN, and the synthetic speech has silence gaps.
    """
    w = SR
    energy = np.convolve(src_audio ** 2, np.ones(w), mode="valid")
    i_src = int(np.argmax(energy))
    i_out = i_src + int(round(offset * SR))
    return float(np.corrcoef(out_audio[i_out: i_out + w], src_audio[i_src: i_src + w])[0, 1])


def test_analyze_measures_without_rendering(tmp_path: Path) -> None:
    video, audio = _make_pair(tmp_path, video_len=24.0, audio_len=18.0, offset=4.0)
    result = analyze(video, audio)
    assert result.output is None
    assert abs(result.offset_sec - 4.0) < 0.02
    assert result.video_duration == pytest.approx(24.0, abs=0.2)
    assert result.summary().startswith("offset +")
    assert result.to_json()["offset_sec"] == round(result.offset_sec, 6)
    # Nothing but the fixtures was written.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["external.mp4", "master.mp4"]


def test_align_remux_keeps_the_picture_and_lands_the_audio(tmp_path: Path) -> None:
    video, audio = _make_pair(tmp_path, video_len=36.0, audio_len=32.0, offset=6.0)
    out = tmp_path / "master_aligned.mov"
    result = align(video, audio, out)
    assert result.output == out
    assert abs(result.offset_sec - 6.0) < 0.02

    assert _duration(out) == pytest.approx(36.0, abs=0.2)
    probed = _probe(out)
    assert probed["codec_name"] == "h264"
    # No display rotation on the source, so the video stream is untouched:
    # the decoded frames are bit-identical to the master's.
    assert _video_frames_md5(out) == _video_frames_md5(video)
    assert not any("rotation" in sd for sd in probed.get("side_data_list") or [])

    corr = _corr_at_offset(_audio(out, tmp_path / "probe-out"), _audio(audio, tmp_path / "probe-in"), 6.0)
    assert corr > 0.9, f"audio not at the measured offset: corr={corr:.3f}"


def test_align_pads_the_tail_when_the_audio_runs_out(tmp_path: Path) -> None:
    video, audio = _make_pair(tmp_path, video_len=30.0, audio_len=18.0, offset=5.0)
    out = tmp_path / "out.mov"
    align(video, audio, out)
    samples = _audio(out, tmp_path / "probe")
    # Audio ends at 23 s on the output timeline; the tail must be near silence,
    # but speech must be present right up to that point.
    assert np.sqrt(np.mean(samples[int(25 * SR):] ** 2)) < 1e-3
    assert np.sqrt(np.mean(samples[int(15 * SR): int(22 * SR)] ** 2)) > 1e-3


def test_align_trims_the_head_when_the_recording_started_early(tmp_path: Path) -> None:
    """A recording that joined early is trimmed and rebased onto the output
    timeline. Leaving its timestamps alone put a `offset`-second hole with no
    audio under the head of the picture, and left the track shorter than the
    video, which the run then refused to cut."""
    video, audio = _make_pair(tmp_path, video_len=24.0, audio_len=32.0, offset=-6.0)
    out = tmp_path / "master.mov"
    result = align(video, audio, out)
    assert abs(result.offset_sec + 6.0) < 0.02

    v_dur, a_dur = _track_durations(out)
    assert v_dur == pytest.approx(24.0, abs=0.2)
    assert a_dur == pytest.approx(v_dur, abs=0.15)

    # The kept audio is the external recording from its 6 s mark, on time.
    corr = _corr_at_offset(_audio(out, tmp_path / "probe-out"), _audio(audio, tmp_path / "probe-in"), -6.0)
    assert corr > 0.9, f"audio not at the measured offset: corr={corr:.3f}"


def test_align_bakes_display_rotation_into_upright_pixels(tmp_path: Path) -> None:
    """Master stores upright 320x180 sideways (transposed + display_rotation),
    which is how iPhone portrait-sensor masters look to ffmpeg."""
    video_len, audio_len, offset = 24.0, 20.0, 5.0
    speech = speech_signal(offset + audio_len + 1.0)
    sideways, rotated, master = tmp_path / "sideways.mp4", tmp_path / "rotated.mp4", tmp_path / "master.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"testsrc=duration={video_len}:size=320x180:rate=30",
         "-vf", "transpose=1", "-an", "-c:v", "libx264", "-preset", "ultrafast", str(sideways)],
        check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-display_rotation", "90", "-i", str(sideways),
                    "-c", "copy", str(rotated)], check=True)
    # Give the master its own mic track (speech from its t=0).
    _attach_audio(rotated, master, speech, video_len)
    audio = tmp_path / "external.mp4"
    _write_av(audio, f"color=black:duration={audio_len}:size=320x180:rate=30",
              speech[int(offset * SR):], audio_len)

    out = tmp_path / "out.mov"
    align(master, audio, out)
    probed = _probe(out)
    assert (probed["width"], probed["height"]) == (320, 180)
    assert probed["codec_name"] == "h264"  # SDR source -> libx264 path
    assert not any("rotation" in sd for sd in probed.get("side_data_list") or [])
    assert _video_frames_md5(out) != _video_frames_md5(rotated)  # re-encoded

    # Frames must match the upright reference, not the sideways storage.
    def gray(path: Path, t: float) -> np.ndarray:
        return np.frombuffer(subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1",
             "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout,
            np.uint8).astype(float)

    ref = tmp_path / "ref.raw"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30",
         "-ss", "10", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", str(ref)],
        capture_output=True, check=True)
    got, want = gray(out, 10.0), np.frombuffer(ref.read_bytes(), np.uint8).astype(float)
    assert got.size == want.size
    assert np.abs(got - want).mean() < 8.0, f"output not upright: diff={np.abs(got - want).mean():.2f}"

    corr = _corr_at_offset(_audio(out, tmp_path / "probe-out"), _audio(audio, tmp_path / "probe-in"), offset)
    assert corr > 0.9, f"audio misaligned after rotation: corr={corr:.3f}"


def test_refuses_output_equal_to_an_input(tmp_path: Path) -> None:
    video, audio = tmp_path / "v.mov", tmp_path / "a.mp4"
    for out in (video, audio):
        with pytest.raises(AlignError, match="must differ"):
            align(video, audio, out)


def test_missing_input_raises_align_error(tmp_path: Path) -> None:
    with pytest.raises(AlignError, match="video not found"):
        analyze(tmp_path / "nope.mov", tmp_path / "nope.mp4")
