"""Render: picture and audio in separate passes, muxed by stream copy.

The split is not cosmetic. ffmpeg 9.0.1 emits digitally silent audio from a
single combined `concat=v=1:a=1` graph for some source regions, so the picture
and the audio are encoded apart and the audio is copied onto the picture --
identical trims and encodes, so the bytes are the ones a single pass owed.
"""

import json
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

from talkinghead.color import ColorProfile, decode_args, hdr_codec, hdr_pix_fmt, without_hwaccel
from talkinghead.pipeline import render_memo_key
from talkinghead.render import (
    AUDIO_RATE,
    RENDER_MIN_TIMEOUT_SEC,
    RENDERER_VERSION,
    RenderSettings,
    _handles,
    _join_gain_db,
    _run,
    _run_decoding,
    _timeout,
    build_commands,
    frame_size,
    join_audio,
    render_audio_plan,
    render_picture,
    render_plan,
)
from talkinghead.timeline import Cut, Plan, Segment, SourceInfo
from talkinghead.verify import _loudness, measure_lufs
from talkinghead.visuals.render import Overlays, _even
from talkinghead.visuals.schedule import Placement

HLG = ColorProfile(primaries="bt2020", transfer="arib-std-b67", matrix="bt2020nc",
                   range="tv", pix_fmt="yuv420p10le", hdr=True)


def _clip(path: Path, seconds: float = 6.0, pix_fmt: str | None = None) -> None:
    """An H.264 clip with audio. libx264 keeps testsrc's RGB as 4:4:4 unless `pix_fmt` says otherwise."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=30:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}",
         "-c:v", "libx264", "-preset", "ultrafast", *(["-pix_fmt", pix_fmt] if pix_fmt else []),
         "-c:a", "aac", str(path)],
        check=True,
    )


def _hlg_clip(path: Path, seconds: float = 3.0) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=30:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}",
         "-vf", "format=yuv420p10le,"
                "scale=out_color_matrix=bt2020nc:out_transfer=arib-std-b67:"
                "out_primaries=bt2020:out_range=tv",
         "-c:v", hdr_codec(), "-profile:v", "main10", "-pix_fmt", hdr_pix_fmt(hdr_codec()),
         "-b:v", "20M", "-tag:v", "hvc1", *HLG.tags,
         "-c:a", "aac", "-b:a", "128k", str(path)],
        check=True,
    )


def _probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,duration,sample_rate,height,pix_fmt,profile,"
         "color_primaries,color_transfer,color_space,nb_frames",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


def _video(path: Path) -> dict:
    return next(s for s in _probe(path)["streams"] if s["codec_type"] == "video")


def _audio(path: Path) -> dict:
    return next(s for s in _probe(path)["streams"] if s["codec_type"] == "audio")


def _frame(path: Path, t: float) -> bytes:
    return subprocess.check_output(
        ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
         "-pix_fmt", "rgb24", "-f", "rawvideo", "-"])


def _mae(a: bytes, b: bytes) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def _volume_max(path: Path) -> float:
    err = subprocess.run(["ffmpeg", "-v", "info", "-nostats", "-i", str(path),
                          "-af", "volumedetect", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    line = next(ln for ln in err.splitlines() if "max_volume" in ln)
    return float(line.split("max_volume:")[1].split("dB")[0])


@pytest.mark.parametrize("fps", [0, 30])
def test_render_removes_the_planned_time(tmp_path: Path, fps: int) -> None:
    src = tmp_path / "src.mp4"
    _clip(src)
    source = SourceInfo("main", str(src), "x", 6.0)
    plan = Plan.from_cuts(source, [Cut(1.0, 2.0, "silence"), Cut(4.0, 4.5, "filler")])
    out = tmp_path / "out.mp4"
    render_plan(plan, str(out), RenderSettings(crf=28, preset="ultrafast", fps=fps))
    info = _probe(out)
    assert float(info["format"]["duration"]) == pytest.approx(4.5, abs=0.15)
    audio = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert audio["sample_rate"] == "48000"


def test_render_demo_tier_downscales(tmp_path: Path) -> None:
    src = tmp_path / "src.mp4"
    _clip(src, 3.0)
    plan = Plan.from_cuts(SourceInfo("main", str(src), "x", 3.0), [Cut(1.0, 1.5, "silence")])
    out = tmp_path / "demo.mp4"
    render_plan(plan, str(out), RenderSettings(crf=28, preset="ultrafast", scale="-2:120"))
    assert int(_video(out)["height"]) == 120


def test_render_with_no_cuts_keeps_the_full_length(tmp_path: Path) -> None:
    src = tmp_path / "src.mp4"
    _clip(src, 2.0)
    plan = Plan.from_cuts(SourceInfo("main", str(src), "x", 2.0), [])
    out = tmp_path / "same.mp4"
    render_plan(plan, str(out), RenderSettings(crf=28, preset="ultrafast"))
    assert out.is_file()
    assert float(_probe(out)["format"]["duration"]) == pytest.approx(2.0, abs=0.1)


def test_render_refuses_a_plan_that_removes_everything(tmp_path: Path) -> None:
    plan = Plan.from_cuts(SourceInfo("main", "/nonexistent.mp4", "x", 2.0), [Cut(0.0, 2.0, "outro")])
    with pytest.raises(RuntimeError, match="nothing to render"):
        render_plan(plan, str(tmp_path / "x.mp4"), RenderSettings())


# -- the audio master, judged on its samples -------------------------------------
def _write_wav(path: Path, x: np.ndarray) -> None:
    pcm = np.clip(np.round(x * 32767), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(AUDIO_RATE)
        w.writeframes(pcm.tobytes())


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768


def _rms_db(x: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(x * x)))


def _crest_db(x: np.ndarray) -> float:
    return 20 * np.log10(np.max(np.abs(x)) / np.sqrt(np.mean(x * x)))


def _peak_db(x: np.ndarray) -> float:
    return 20 * np.log10(np.max(np.abs(x)))


def _master(tmp_path: Path, recording: np.ndarray) -> Path:
    src = tmp_path / "recording.wav"
    _write_wav(src, recording)
    plan = Plan.from_cuts(SourceInfo("main", str(src), "x", len(recording) / AUDIO_RATE), [])
    out = tmp_path / "master.wav"
    render_audio_plan(plan, str(out), RenderSettings(fps=0))
    return out


def test_the_master_keeps_the_recordings_dynamics_and_crest_factor(tmp_path: Path) -> None:
    t = np.arange(4 * AUDIO_RATE) / AUDIO_RATE
    tone = np.sin(2 * np.pi * 220 * t) + 0.6 * np.sin(2 * np.pi * 330 * t + 1.0)
    recording = np.where(t >= 2.0, 0.1, 0.01) * tone          # quiet, then loud: 20 dB apart
    master = _read_wav(_master(tmp_path, recording))
    quiet, loud = slice(int(0.5 * AUDIO_RATE), int(1.5 * AUDIO_RATE)), slice(int(2.5 * AUDIO_RATE), int(3.5 * AUDIO_RATE))
    # One fixed gain: the loud part stays exactly as far above the quiet part, and its peaks keep their rise over RMS.
    gap_in = _rms_db(recording[loud]) - _rms_db(recording[quiet])
    gap_out = _rms_db(master[loud]) - _rms_db(master[quiet])
    assert gap_out == pytest.approx(gap_in, abs=0.2)
    assert _crest_db(master[loud]) == pytest.approx(_crest_db(recording[loud]), abs=0.2)


def test_a_hot_peak_caps_the_master_and_the_quiet_master_is_reported_without_a_cause(tmp_path: Path) -> None:
    t = np.arange(4 * AUDIO_RATE) / AUDIO_RATE
    recording = 0.1 * np.sin(2 * np.pi * 220 * t)
    recording[2 * AUDIO_RATE:2 * AUDIO_RATE + 96] = 0.95       # a 2 ms hot peak above the bed
    out = _master(tmp_path, recording)
    assert _peak_db(_read_wav(out)) <= -1.4                    # the true-peak ceiling holds
    result = _loudness(None, {"lufs": measure_lufs(str(out)), "lufs_target": -16.0})
    assert result.severity == "warn" and not result.ok         # quiet, reported, not a failed cook
    assert "quieter than target by" in result.detail and "compression were applied" in result.detail


def test_a_master_with_headroom_reaches_the_target_and_passes_the_loudness_check(tmp_path: Path) -> None:
    t = np.arange(4 * AUDIO_RATE) / AUDIO_RATE
    out = _master(tmp_path, 0.02 * np.sin(2 * np.pi * 220 * t))
    assert measure_lufs(str(out)) == pytest.approx(-16.0, abs=1.0)
    assert _loudness(None, {"lufs": measure_lufs(str(out)), "lufs_target": -16.0}).ok


def test_a_short_hot_click_is_still_peak_capped(tmp_path: Path) -> None:
    recording = np.zeros(AUDIO_RATE)                           # one second: too short to gate an integrated level
    recording[24000:24096] = 0.95
    assert _peak_db(_read_wav(_master(tmp_path, recording))) <= -1.4


def test_silence_renders_as_exact_silence(tmp_path: Path) -> None:
    master = _read_wav(_master(tmp_path, np.zeros(AUDIO_RATE)))
    assert master.size == AUDIO_RATE and not np.any(master)



# -- a plain render's output ---------------------------------------------------
def test_a_plain_render_writes_one_video_and_one_audio_stream_of_the_plan_length(tmp_path: Path) -> None:
    src = tmp_path / "src.mp4"
    _clip(src, 2.0)
    plan = Plan.from_cuts(SourceInfo("main", str(src), "x", 2.0), [])
    out = tmp_path / "out.mp4"
    render_plan(plan, str(out), RenderSettings(crf=28, preset="ultrafast"))
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type:format=duration",
         "-of", "json", str(out)], capture_output=True, text=True, check=True).stdout)
    assert sorted(s["codec_type"] for s in probe["streams"]) == ["audio", "video"]
    assert float(probe["format"]["duration"]) == pytest.approx(2.0, abs=0.1)


def test_many_frame_grid_cuts_keep_the_exact_frames(tmp_path: Path) -> None:
    # More cuts than one `select` expression holds, all on the frame grid: the output
    # must be exactly the kept source frames in order, so neither the chunking nor a
    # closed-interval off-by-one can drop, repeat or shift a frame.
    src = tmp_path / "src.mp4"
    # Flat frames whose luma is the frame index, one level per frame (240 frames).
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "nullsrc=size=64x64:rate=30:duration=8,geq=lum='N':cb=128:cr=128",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=8",
         "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", "-c:a", "aac", str(src)],
        check=True)
    cuts = [Cut((5 * k + 2) / 30, (5 * k + 4) / 30, "silence") for k in range(45)]
    plan = Plan.from_cuts(SourceInfo("main", str(src), "x", 8.0), cuts)
    out = tmp_path / "out.mp4"
    render_plan(plan, str(out), RenderSettings(crf=18, preset="ultrafast", fps=30))

    # The raw Y plane, not gray: a gray conversion rescales the limited-range luma.
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(out), "-fps_mode", "passthrough",
                                   "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"])
    frames = np.frombuffer(raw, np.uint8).reshape(-1, 64 * 64 * 3 // 2)[:, :64 * 64]
    shown = [round(f.mean()) for f in frames]
    removed = {5 * k + d for k in range(45) for d in (2, 3)}
    assert shown == [n for n in range(240) if n not in removed]


def test_reordered_segments_keep_the_exact_frames(tmp_path: Path) -> None:
    # A reordered plan reads each run through its own seeked input. Seeking must land on
    # the same frame grid as an unseeked decode: no frame dropped, repeated or shifted.
    src = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "nullsrc=size=64x64:rate=30:duration=8,geq=lum='N':cb=128:cr=128",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=8",
         "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", "-g", "300", "-c:a", "aac", str(src)],
        check=True)
    spans = [(180, 210), (30, 60), (120, 150), (6, 18)]
    plan = Plan(
        sources={"main": SourceInfo("main", str(src), "x", 8.0)}, main="main", cuts=[],
        segments=[Segment("main", a / 30, b / 30) for a, b in spans],
    )
    out = tmp_path / "out.mp4"
    render_plan(plan, str(out), RenderSettings(crf=18, preset="ultrafast", fps=30, audio_filter="anull"))

    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(out), "-fps_mode", "passthrough",
                                   "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"])
    frames = np.frombuffer(raw, np.uint8).reshape(-1, 64 * 64 * 3 // 2)[:, :64 * 64]
    assert [round(f.mean()) for f in frames] == [n for a, b in spans for n in range(a, b)]


def test_short_video_tail_preserves_next_run_sync(tmp_path: Path) -> None:
    src = tmp_path / "src.mp4"
    _clip(src, 2.0)
    plan = Plan(
        sources={"main": SourceInfo("main", str(src), "x", 2 + 1 / 30)},
        main="main", cuts=[],
        segments=[Segment("main", 0.0, 2 + 1 / 30), Segment("main", 0.0, 1.0)],
    )
    out = tmp_path / "out.mp4"
    render_plan(plan, str(out), RenderSettings(crf=18, preset="ultrafast", audio_filter="anull"))
    assert int(_video(out)["nb_frames"]) == 91
    assert float(_audio(out)["duration"]) == pytest.approx(91 / 30, abs=0.025)
    assert _mae(_frame(src, 0.0), _frame(out, 61 / 30)) < 3.0


def test_a_render_leaves_no_intermediates_behind(tmp_path: Path) -> None:
    src = tmp_path / "src.mp4"
    _clip(src, 2.0)
    plan = Plan.from_cuts(SourceInfo("main", str(src), "x", 2.0), [])
    out = tmp_path / "out.mp4"
    render_plan(plan, str(out), RenderSettings(crf=28, preset="ultrafast"))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.mp4", "src.mp4"]


def test_a_plan_mixing_hdr_and_sdr_sources_is_refused(tmp_path: Path) -> None:
    hdr, sdr = tmp_path / "hdr.mp4", tmp_path / "sdr.mp4"
    _hlg_clip(hdr, 1.0)
    _clip(sdr, 1.0)
    plan = Plan(
        sources={"h": SourceInfo("h", str(hdr), "x", 1.0), "s": SourceInfo("s", str(sdr), "x", 1.0)},
        main="h", cuts=[],
        segments=[Segment("h", 0.0, 0.5), Segment("s", 0.0, 0.5)],
    )
    with pytest.raises(RuntimeError, match="mixes HDR and SDR"):
        build_commands(plan, str(tmp_path / "out.mp4"), RenderSettings(), "master.m4a")


def test_a_plan_cannot_relabel_pq_as_hlg(monkeypatch, tmp_path: Path) -> None:
    pq = ColorProfile(primaries="bt2020", transfer="smpte2084", matrix="bt2020nc",
                      range="tv", pix_fmt="yuv420p10le", hdr=True)
    profiles = {"hlg.mov": HLG, "pq.mov": pq}
    monkeypatch.setattr("talkinghead.render.probe_color", profiles.__getitem__)
    plan = Plan(
        sources={name: SourceInfo(name, path, "x", 1.0)
                 for name, path in (("hlg", "hlg.mov"), ("pq", "pq.mov"))},
        main="hlg", cuts=[],
        segments=[Segment("hlg", 0.0, 0.5), Segment("pq", 0.0, 0.5)],
    )
    with pytest.raises(RuntimeError):
        render_plan(plan, str(tmp_path / "out.mp4"), RenderSettings())


# -- colour ------------------------------------------------------------------
def test_hdr_source_renders_ten_bit_hlg_with_equivalent_colour(tmp_path: Path) -> None:
    src = tmp_path / "src.mp4"
    _hlg_clip(src, 3.0)
    plan = Plan.from_cuts(SourceInfo("main", str(src), "x", 3.0), [Cut(1.0, 1.5, "silence")])
    out = tmp_path / "out.mp4"
    render_plan(plan, str(out), RenderSettings(fps=30))

    video = _video(out)
    assert video["pix_fmt"] == "yuv420p10le"
    assert video["profile"] == "Main 10"
    assert (video["color_primaries"], video["color_transfer"], video["color_space"]) == \
        ("bt2020", "arib-std-b67", "bt2020nc")
    assert int(video["nb_frames"]) == 75                       # 2.5 s at 30 fps
    assert float(video["duration"]) == pytest.approx(2.5, abs=0.05)
    assert float(_probe(out)["format"]["duration"]) == pytest.approx(float(_audio(out)["duration"]), abs=0.05)
    assert _volume_max(out) > -30                              # the audio survived the split
    # Colour, not just tags: an output frame decodes to the same picture the
    # source's own frame does.
    assert _mae(_frame(src, 2.0), _frame(out, 1.0)) < 6.0


def test_sdr_source_still_renders_eight_bit_sdr(tmp_path: Path) -> None:
    src = tmp_path / "src.mp4"
    _clip(src, 2.0)
    plan = Plan.from_cuts(SourceInfo("main", str(src), "x", 2.0), [])
    out = tmp_path / "out.mp4"
    render_plan(plan, str(out), RenderSettings(crf=28, preset="ultrafast"))
    video = _video(out)
    assert video["pix_fmt"] == "yuv420p"
    assert video["codec_name"] == "h264"
    assert video.get("color_transfer") != "arib-std-b67"


def test_a_multi_segment_plan_keeps_audible_audio(tmp_path: Path) -> None:
    """The defect the split passes exist for: five joins, and the audio is there."""
    src = tmp_path / "src.mp4"
    _clip(src, 8.0)
    cuts = [Cut(1.0, 1.4, "silence"), Cut(2.2, 2.6, "filler"), Cut(3.4, 3.8, "filler"),
            Cut(4.6, 5.0, "silence"), Cut(5.8, 6.2, "filler")]
    plan = Plan.from_cuts(SourceInfo("main", str(src), "x", 8.0), cuts)
    out = tmp_path / "out.mp4"
    render_plan(plan, str(out), RenderSettings(crf=28, preset="ultrafast"))
    assert _volume_max(out) > -30
    assert float(_video(out)["duration"]) == pytest.approx(float(_audio(out)["duration"]), abs=0.05)


def test_the_renderer_version_is_exported_for_the_render_memo() -> None:
    assert isinstance(RENDERER_VERSION, int) and RENDERER_VERSION >= 2


# -- crossfaded joins --------------------------------------------------------
def _sine(seconds: float, hz: float = 1000.0) -> np.ndarray:
    t = np.arange(round(seconds * AUDIO_RATE)) / AUDIO_RATE
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)[:, None]


def _rms_db(x: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)


def test_a_join_inside_continuous_audio_has_no_level_dip() -> None:
    # 100 whole periods removed, so the sides are in phase and only the fade shape is tested.
    src = {"main": _sine(2.0)}
    segs = [Segment("main", 0.0, 1.0), Segment("main", 1.1, 2.0)]
    out = join_audio(src, segs, 0.020)
    window = AUDIO_RATE // 500  # 2 ms
    levels = [_rms_db(out[i:i + window]) for i in range(AUDIO_RATE // 4, len(out) - AUDIO_RATE // 4, window)]
    assert min(levels) > _rms_db(src["main"]) - 3.0


def test_the_joined_length_is_the_rounded_sum_of_segment_lengths() -> None:
    src = {"main": _sine(3.0, 440.0)}
    segs = [Segment("main", 0.0333, 0.9011), Segment("main", 1.0417, 1.2), Segment("main", 1.7777, 2.9)]
    out = join_audio(src, segs, 0.020)
    assert len(out) == round(sum(s.end - s.start for s in segs) * AUDIO_RATE)


def test_handles_stay_inside_the_source_and_within_half_the_removed_span() -> None:
    rate = AUDIO_RATE
    segs = [Segment("a", 0.0, 0.5), Segment("a", 0.51, 1.0), Segment("a", 1.0, 1.5), Segment("a", 1.5, 2.0)]
    lengths = [round((s.end - s.start) * rate) for s in segs]
    starts = [round(s.start * rate) for s in segs]
    half = _handles(segs, lengths, starts, {"a": 2 * rate}, round(0.020 * rate))
    assert half[0] <= (starts[1] - starts[0] - lengths[0]) // 2   # 10 ms removed -> at most 5 ms
    assert half[1] == 0                                             # contiguous: nothing removed
    assert half[2] == 0
    short = _handles([Segment("a", 0.0, 0.004), Segment("a", 0.5, 1.0)], [192, 24000],
                     [0, 24000], {"a": 2 * rate}, round(0.020 * rate))
    assert short[0] == 96                                           # half the short segment


def test_handles_never_read_past_the_source_end() -> None:
    rate = AUDIO_RATE
    segs = [Segment("a", 0.0, 1.0), Segment("b", 0.5, 1.0)]
    half = _handles(segs, [rate, rate // 2], [0, rate // 2], {"a": rate + 100, "b": rate}, round(0.020 * rate))
    assert half == [100]


def test_the_crossfade_length_is_part_of_the_render_memo_key(tmp_path: Path) -> None:
    plan = Plan.from_cuts(SourceInfo("main", "/x.mp4", "sha", 6.0), [Cut(1.0, 2.0, "silence")])
    a = render_memo_key(plan, RenderSettings(crossfade_sec=0.020))
    b = render_memo_key(plan, RenderSettings(crossfade_sec=0.040))
    assert a != b


def test_a_rendered_join_keeps_the_audio_length_and_the_level(tmp_path: Path) -> None:
    src = tmp_path / "src.mp4"
    _clip(src, 3.0)
    plan = Plan.from_cuts(SourceInfo("main", str(src), "x", 3.0), [Cut(1.0, 1.1, "filler")])
    out = tmp_path / "out.mp4"
    render_plan(plan, str(out), RenderSettings(crf=28, preset="ultrafast"))
    assert float(_audio(out)["duration"]) == pytest.approx(2.9, abs=0.06)
    assert _volume_max(out) > -30


# --- audio-only render: the same join and mastering, a WAV out, no picture ---

def _tone_wav(path: Path, seconds: float = 6.0) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", f"sine=frequency=440:sample_rate=16000:duration={seconds}", "-c:a", "pcm_s16le", str(path)],
        check=True)


def test_an_audio_plan_renders_a_48k_pcm_wav_of_the_planned_length(tmp_path: Path) -> None:
    src = tmp_path / "src.wav"
    _tone_wav(src)
    plan = Plan.from_cuts(SourceInfo("main", str(src), "x", 6.0), [Cut(1.0, 1.4, "silence"), Cut(3.0, 3.5, "filler")])
    out = tmp_path / "out.wav"
    render_audio_plan(plan, str(out), RenderSettings(fps=0))
    probed = _probe(out)
    assert [s["codec_type"] for s in probed["streams"]] == ["audio"]
    assert probed["streams"][0]["codec_name"] == "pcm_s16le" and probed["streams"][0]["sample_rate"] == "48000"
    assert float(probed["format"]["duration"]) == pytest.approx(plan.output_duration, abs=0.01)
    assert -30 < _volume_max(out) <= -1.0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.wav", "src.wav"]


def test_a_failed_audio_render_leaves_the_previous_file_and_no_debris(tmp_path: Path) -> None:
    src = tmp_path / "src.wav"
    _tone_wav(src)
    plan = Plan.from_cuts(SourceInfo("main", str(src), "x", 6.0), [Cut(1.0, 1.4, "silence")])
    out = tmp_path / "out.wav"
    out.write_bytes(b"the last good render")
    src.write_bytes(b"not audio any more")
    with pytest.raises(RuntimeError):
        render_audio_plan(plan, str(out), RenderSettings(fps=0))
    assert out.read_bytes() == b"the last good render"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.wav", "src.wav"]


# -- ffmpeg failure modes ----------------------------------------------------
def test_a_missing_ffmpeg_gets_an_install_hint(monkeypatch) -> None:
    def missing(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "ffmpeg")
    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(RuntimeError, match="ffmpeg not found; install ffmpeg"):
        _run(["ffmpeg", "-version"], 5.0)


def test_a_stalled_ffmpeg_is_stopped_and_says_to_rerun(monkeypatch) -> None:
    def stalled(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
    monkeypatch.setattr(subprocess, "run", stalled)
    with pytest.raises(RuntimeError, match="stalled.*rerun the same command"):
        _run(["ffmpeg", "-version"], 5.0)


def test_the_pass_timeout_scales_with_the_output_length() -> None:
    short = Plan.from_cuts(SourceInfo("main", "/x.mp4", "x", 60.0), [])
    long = Plan.from_cuts(SourceInfo("main", "/x.mp4", "x", 3600.0), [])
    assert _timeout(short) == RENDER_MIN_TIMEOUT_SEC
    assert _timeout(long) == pytest.approx(3 * 3600.0)


# -- hardware decode ---------------------------------------------------------
needs_hwaccel = pytest.mark.skipif(not decode_args("yuv420p"), reason="this ffmpeg has no VideoToolbox decode")


def _graph_frames(cmd: list[str], pix_fmt: str) -> tuple[bytes, str]:
    """The picture pass's filtered frames before the encode, and its debug log.

    The encoder is left out on purpose: decode is the only thing hardware changes, and
    comparing raw frames does not depend on the encoder being deterministic.
    """
    head = cmd[:cmd.index("-map")]
    head[head.index("-v") + 1] = "debug"
    proc = subprocess.run([*head, "-map", "[vout]", "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"],
                          capture_output=True, check=True)
    return proc.stdout, proc.stderr.decode(errors="replace")


@needs_hwaccel
@pytest.mark.parametrize("hdr", [False, True])
def test_hardware_decode_gives_the_cut_the_software_frames(tmp_path: Path, hdr: bool) -> None:
    """VideoToolbox is only a speed-up: the cut's frames are the software decode's, bit for bit.

    The log check keeps the comparison honest -- a hardware decoder that quietly fell
    back to software would compare equal without testing anything.
    """
    src = tmp_path / "src.mp4"
    _hlg_clip(src, 3.0) if hdr else _clip(src, 3.0, pix_fmt="yuv420p")
    plan = Plan.from_cuts(SourceInfo("main", str(src), "x", 3.0),
                          [Cut(0.5, 1.0, "silence"), Cut(2.0, 2.2, "filler")])
    cmd = build_commands(plan, str(tmp_path / "out.mp4"), RenderSettings(), "master.m4a")[0]
    assert cmd[cmd.index("-hwaccel") + 1] == "videotoolbox"
    pix_fmt = "yuv420p10le" if hdr else "yuv420p"
    hard, log = _graph_frames(cmd, pix_fmt)
    soft, _ = _graph_frames(without_hwaccel(cmd), pix_fmt)
    assert "videotoolbox_vld" in log
    assert hard and hard == soft


@needs_hwaccel
def test_every_reordered_input_decodes_on_the_hardware(tmp_path: Path) -> None:
    """A plan that revisits a source gives each run its own seeked input; each one gets the flag."""
    src = tmp_path / "src.mp4"
    _clip(src, 4.0, pix_fmt="yuv420p")
    source = SourceInfo("main", str(src), "x", 4.0)
    plan = Plan(sources={"main": source}, main="main", cuts=[],
                segments=[Segment("main", 2.0, 3.0), Segment("main", 0.5, 1.5)])
    cmd = build_commands(plan, str(tmp_path / "out.mp4"), RenderSettings(), "master.m4a")[0]
    assert cmd.count("-i") == 2 and cmd.count("-hwaccel") == 2
    hard, _ = _graph_frames(cmd, "yuv420p")
    soft, _ = _graph_frames(without_hwaccel(cmd), "yuv420p")
    assert hard and hard == soft


def test_a_444_source_decodes_in_software(tmp_path: Path) -> None:
    """Found by this suite: VideoToolbox subsamples 4:4:4 chroma itself and lands one level off
    the software decode, so such a source keeps the software path and its exact frames."""
    src = tmp_path / "src.mp4"
    _clip(src, 2.0)
    plan = Plan.from_cuts(SourceInfo("main", str(src), "x", 2.0), [Cut(0.5, 1.0, "silence")])
    cmd = build_commands(plan, str(tmp_path / "out.mp4"), RenderSettings(), "master.m4a")[0]
    assert "-hwaccel" not in cmd


def test_a_pass_the_hardware_cannot_decode_reruns_in_software(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """A real ffmpeg failure, not a mock: an unknown hwaccel makes the picture pass exit
    nonzero. The render still completes, and the command it reports is the one that ran."""
    monkeypatch.setattr("talkinghead.render.decode_args", lambda pix_fmt: ["-hwaccel", "no-such-hwaccel"])
    src = tmp_path / "src.mp4"
    _clip(src, 2.0)
    plan = Plan.from_cuts(SourceInfo("main", str(src), "x", 2.0), [Cut(0.5, 1.0, "silence")])
    out = tmp_path / "out.mp4"
    cmd = render_plan(plan, str(out), RenderSettings(crf=28, preset="ultrafast"))
    assert float(_probe(out)["format"]["duration"]) == pytest.approx(1.5, abs=0.1)
    assert "-hwaccel" not in cmd
    assert "software decode" in capsys.readouterr().err


def test_a_pass_that_also_fails_in_software_still_raises(tmp_path: Path) -> None:
    """The rerun is for hardware decode only: a pass that fails in software too raises as before."""
    cmd = ["ffmpeg", "-y", "-v", "error", "-nostdin", "-hwaccel", "no-such-hwaccel",
           "-i", str(tmp_path / "missing.mp4"), "-f", "null", "-"]
    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        _run_decoding(cmd, 30.0)


def test_a_voiced_join_gain_matches_the_incoming_side_then_eases_back_to_unity() -> None:
    # Loud sine 0.4-0.9 s, quieter sine 1.1-1.6 s, silence around: a 6 dB step across the cut.
    rate = AUDIO_RATE
    x = np.zeros((2 * rate, 1), dtype=np.float32)
    tt = np.arange(rate // 2) / rate
    x[int(0.4 * rate):int(0.9 * rate), 0] = 0.5 * np.sin(2 * np.pi * 440 * tt)
    x[int(1.1 * rate):int(1.6 * rate), 0] = 0.25 * np.sin(2 * np.pi * 440 * tt)
    out = join_audio({"a": x}, [Segment("a", 0.4, 0.9), Segment("a", 1.1, 1.6)], 0.020)
    join = round(0.5 * rate)                                 # output index of the join
    pre, post = _rms_db(x[int(0.7 * rate):int(0.9 * rate)]), _rms_db(x[int(1.3 * rate):int(1.5 * rate)])
    just_after = out[join + round(0.02 * rate):join + round(0.04 * rate)]
    assert _rms_db(just_after) == pytest.approx(pre, abs=2.0)  # boosted to the outgoing level at the join
    settled = out[join + round(0.4 * rate):join + round(0.5 * rate)]
    assert _rms_db(settled) == pytest.approx(post, abs=0.5)    # unity again after the ramp


def test_a_join_between_two_room_tone_sides_keeps_unity_gain() -> None:
    rng = np.random.default_rng(0)
    x = (0.001 * rng.standard_normal((AUDIO_RATE, 1))).astype(np.float32)
    assert _join_gain_db(x, AUDIO_RATE // 4, x, AUDIO_RATE // 2) == 0.0


# --- the AAC master and the picture render on their own ---

def _packet_md5(path: Path) -> str:
    """The MD5 of a file's first audio stream's packets, which survives a stream-copy mux."""
    return subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-c", "copy", "-f", "md5", "-"],
                          capture_output=True, text=True, check=True).stdout.strip()


def _cut_plan(tmp_path: Path, seconds: float = 4.0) -> Plan:
    src = tmp_path / "src.mp4"
    _clip(src, seconds)
    return Plan.from_cuts(SourceInfo("main", str(src), "x", seconds), [Cut(0.5, 1.0, "silence"), Cut(2.0, 2.2, "filler")])


def test_a_video_render_muxes_the_given_master_unchanged(tmp_path: Path) -> None:
    plan = _cut_plan(tmp_path)
    master = tmp_path / "master.m4a"
    render_audio_plan(plan, str(master), RenderSettings(preset="ultrafast"))
    out = tmp_path / "out.mp4"
    cmd = render_plan(plan, str(out), RenderSettings(crf=28, preset="ultrafast"), audio_file=str(master))
    assert _packet_md5(out) == _packet_md5(master)
    assert " && " in cmd and "master.m4a" in cmd and "aac" not in cmd
    assert sorted(p.name for p in tmp_path.iterdir()) == ["master.m4a", "out.mp4", "src.mp4"]


def test_a_video_render_without_a_master_has_the_same_master_as_a_separate_render(tmp_path: Path) -> None:
    plan = _cut_plan(tmp_path)
    settings = RenderSettings(crf=28, preset="ultrafast")
    master = tmp_path / "master.m4a"
    render_audio_plan(plan, str(master), settings)
    out = tmp_path / "out.mp4"
    render_plan(plan, str(out), settings)
    assert _packet_md5(out) == _packet_md5(master)
    assert _audio(master)["codec_name"] == "aac" and _audio(master)["sample_rate"] == "48000"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["master.m4a", "out.mp4", "src.mp4"]


def test_a_supplied_master_at_the_automatic_path_survives_success_and_failure(tmp_path: Path) -> None:
    plan = _cut_plan(tmp_path)
    settings = RenderSettings(crf=28, preset="ultrafast")
    out = tmp_path / "out.mp4"
    master = tmp_path / "out.mp4.audio.m4a"
    render_audio_plan(plan, str(master), settings)
    before = master.read_bytes()
    render_plan(plan, str(out), settings, audio_file=str(master))
    assert master.read_bytes() == before
    bad = RenderSettings(crf=28, preset="no-such-preset")
    out.unlink()
    with pytest.raises(Exception):
        render_plan(plan, str(out), bad, audio_file=str(master))
    assert master.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.mp4.audio.m4a", "src.mp4"]


def test_the_m4a_master_snaps_segment_edges_to_the_frame_grid(tmp_path: Path) -> None:
    src = tmp_path / "src.mp4"
    _clip(src, 4.0)
    plan = Plan(sources={"main": SourceInfo("main", str(src), "x", 4.0)}, main="main", cuts=[],
                segments=[Segment("main", 0.0, 1.0135), Segment("main", 2.0135, 3.0135)])
    snapped, free = tmp_path / "snapped.m4a", tmp_path / "free.m4a"
    render_audio_plan(plan, str(snapped), RenderSettings(fps=30))
    render_audio_plan(plan, str(free), RenderSettings(fps=0))

    def samples(path: Path) -> float:
        return float(_probe(path)["format"]["duration"]) * AUDIO_RATE

    # Off-grid edges move by up to half a frame each: the grid render is not the free one.
    assert abs(samples(snapped) - samples(free)) > 100


def test_a_picture_only_render_has_no_audio_and_the_frames_of_the_video(tmp_path: Path) -> None:
    plan = _cut_plan(tmp_path)
    settings = RenderSettings(crf=28, preset="ultrafast")
    picture, video = tmp_path / "picture.mp4", tmp_path / "video.mp4"
    render_picture(plan, str(picture), settings)
    render_plan(plan, str(video), settings)
    assert [s["codec_type"] for s in _probe(picture)["streams"]] == ["video"]
    assert _video(picture)["nb_frames"] == _video(video)["nb_frames"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["picture.mp4", "src.mp4", "video.mp4"]


def test_an_audio_render_to_an_unsupported_suffix_is_refused_before_any_work(tmp_path: Path) -> None:
    src = tmp_path / "src.wav"
    _tone_wav(src)
    plan = Plan.from_cuts(SourceInfo("main", str(src), "x", 6.0), [])
    with pytest.raises(ValueError, match="m4a"):
        render_audio_plan(plan, str(tmp_path / "out.flac"), RenderSettings(fps=0))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["src.wav"]


# --- overlays composited inside the cut's one picture encode ---

def _white_card(path: Path) -> Path:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=white:s=100x60",
                    "-vf", "format=rgb24", "-frames:v", "1", str(path)], check=True)
    return path


def _luma_block(path: Path, t: float, x: int, y: int, *, size: tuple[int, int] = (320, 240), half: int = 4) -> float:
    """The mean 8-bit luma of a small block around (x, y) in the frame at `t`."""
    w, h = size
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
                                   "-pix_fmt", "gray", "-f", "rawvideo", "-"])
    rows = [raw[(j * w + x - half):(j * w + x + half)] for j in range(y - half, y + half)]
    return sum(sum(r) for r in rows) / sum(len(r) for r in rows)


def _luma10(path: Path, t: float, x: int, y: int, width: int = 320) -> int:
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
                                   "-pix_fmt", "yuv420p10le", "-f", "rawvideo", "-"])
    return int.from_bytes(raw[(y * width + x) * 2:(y * width + x) * 2 + 2], "little")


def _card_overlays(card: Path, start: float = 0.2, end: float = 2.6, scale: float = 0.3) -> Overlays:
    return Overlays((Placement("card", str(card), start, end, scale),))


def test_a_card_lands_in_its_corner_inside_the_cut_and_leaves_the_rest_of_the_picture_alone_overlay_in_the_cut(
        tmp_path: Path) -> None:
    plan = _cut_plan(tmp_path)                                    # 4 s source, 0.7 s cut: 3.3 s out
    settings = RenderSettings(crf=18, preset="ultrafast")
    card = _white_card(tmp_path / "card.png")
    bare, drawn, audio = tmp_path / "bare.mp4", tmp_path / "drawn.mp4", tmp_path / "master.m4a"
    render_audio_plan(plan, str(audio), settings)
    render_plan(plan, str(bare), settings, audio_file=str(audio))
    render_plan(plan, str(drawn), settings, audio_file=str(audio), overlays=_card_overlays(card))

    # 0.3 * 320 = 96 wide, top-right with margin 40: x 184..280, y 40..98. The window midpoint is 1.4 s.
    assert _luma_block(drawn, 1.4, 230, 65) > 225
    assert abs(_luma_block(drawn, 1.4, 60, 190) - _luma_block(bare, 1.4, 60, 190)) <= 8     # outside the card
    assert abs(_luma_block(drawn, 3.0, 230, 65) - _luma_block(bare, 3.0, 230, 65)) <= 8     # outside the window


def test_an_overlaid_render_runs_one_video_encode_and_keeps_the_frames_and_the_master_one_encode(
        tmp_path: Path, monkeypatch) -> None:
    import talkinghead.render as render

    plan = _cut_plan(tmp_path)
    settings = RenderSettings(crf=28, preset="ultrafast")
    card = _white_card(tmp_path / "card.png")
    master, bare, drawn = tmp_path / "master.m4a", tmp_path / "bare.mp4", tmp_path / "drawn.mp4"
    render_audio_plan(plan, str(master), settings)
    render_plan(plan, str(bare), settings, audio_file=str(master))
    ran: list[list[str]] = []
    real = render._tool
    monkeypatch.setattr(render, "_tool", lambda cmd, timeout: (ran.append(list(cmd)), real(cmd, timeout))[1])
    render_plan(plan, str(drawn), settings, audio_file=str(master), overlays=_card_overlays(card))

    # The size probe decodes one frame of the source to measure it; it writes no part of the render.
    ffmpegs = [c for c in ran if c[0] == "ffmpeg" and "ffv1" not in c]
    assert len(ffmpegs) == 2
    assert sum("-c:v" in c for c in ffmpegs) == 1
    mux = next(c for c in ffmpegs if "-c:v" not in c)
    assert mux[mux.index("-c") + 1] == "copy"
    assert _video(drawn)["nb_frames"] == _video(bare)["nb_frames"]
    assert _packet_md5(drawn) == _packet_md5(master)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bare.mp4", "card.png", "drawn.mp4", "master.m4a", "src.mp4"]


def test_a_card_on_an_hlg_source_keeps_main_10_and_reads_at_graphics_white_hlg_overlay(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        hdr_codec()
    except RuntimeError:
        pytest.skip("this ffmpeg has no 10-bit HEVC encoder")
    src = tmp_path / "src.mp4"
    _hlg_clip(src, 4.0)
    plan = Plan.from_cuts(SourceInfo("main", str(src), "x", 4.0), [Cut(3.5, 4.0, "silence")])
    card = _white_card(tmp_path / "card.png")
    out = tmp_path / "out.mp4"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    render_plan(plan, str(out), RenderSettings(fps=30, preset="ultrafast"), overlays=_card_overlays(card, 0.2, 2.6))
    assert list(scratch.iterdir()) == []

    video = _video(out)
    assert video["pix_fmt"] == "yuv420p10le" and video["profile"] == "Main 10"
    assert (video["color_primaries"], video["color_transfer"], video["color_space"]) == \
        ("bt2020", "arib-std-b67", "bt2020nc")
    assert abs(_luma10(out, 1.4, 230, 70) - 721) <= 25


def test_an_hlg_overlay_render_that_fails_leaves_no_cube_directory(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        hdr_codec()
    except RuntimeError:
        pytest.skip("this ffmpeg has no 10-bit HEVC encoder")
    src = tmp_path / "src.mp4"
    _hlg_clip(src, 4.0)
    plan = Plan.from_cuts(SourceInfo("main", str(src), "x", 4.0), [Cut(3.5, 4.0, "silence")])
    card = _white_card(tmp_path / "card.png")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    import talkinghead.render as render
    seen: list[list[Path]] = []

    def boom(cmd: list[str], timeout: float) -> str:
        seen.append(list(scratch.iterdir()))
        raise RuntimeError("encode failed")

    monkeypatch.setattr(render, "_run_decoding", boom)
    with pytest.raises(RuntimeError, match="encode failed"):
        render_picture(plan, str(tmp_path / "out.mp4"), RenderSettings(fps=30, preset="ultrafast"),
                       overlays=_card_overlays(card, 0.2, 2.6))
    assert len(seen) == 1 and len(seen[0]) == 1          # the cube directory existed during the encode
    assert list(scratch.iterdir()) == []


def test_frame_size_is_the_ffprobe_size_of_the_render_and_a_card_scales_with_it_scaled_overlay(
        tmp_path: Path) -> None:
    plan = _cut_plan(tmp_path)
    card = _white_card(tmp_path / "card.png")
    for scale, expected in ((None, (320, 240)), ("-2:120", (160, 120))):
        out = tmp_path / "out.mp4"
        settings = RenderSettings(crf=28, preset="ultrafast", scale=scale)
        assert frame_size(plan, settings) == expected
        render_picture(plan, str(out), settings)
        video = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                         "stream=width,height", "-of", "csv=p=0", str(out)], text=True)
        assert tuple(int(v) for v in video.strip().split(",")) == expected

    settings = RenderSettings(crf=28, preset="ultrafast", scale="-2:120")
    out = tmp_path / "scaled.mp4"
    render_picture(plan, str(out), settings, overlays=_card_overlays(card, 0.2, 2.6, 0.5))
    width = _even(0.5 * 160)                                      # 80 px, top-right, margin 40: x 40..120
    assert _luma_block(out, 1.4, 80, 50, size=(160, 120), half=3) > 225
    assert _luma_block(out, 1.4, 120 - 3 + 8, 50, size=(160, 120), half=3) < 225      # right of the card's edge
    assert _luma_block(out, 1.4, 40 + 3, 50, size=(160, 120), half=3) > 225           # inside its left edge
    assert width == 80


def test_frame_size_follows_a_non_square_source_pixel_aspect_in_the_scale_expression(tmp_path: Path) -> None:
    src = tmp_path / "sar.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=320x240:r=30:d=1",
                    "-vf", "setsar=2", "-c:v", "libx264", "-preset", "ultrafast", str(src)], check=True)
    plan = Plan.from_cuts(SourceInfo("main", str(src), "x", 1.0), [Cut(0.8, 1.0, "silence")])
    settings = RenderSettings(crf=28, preset="ultrafast", scale="iw*sar:ih")
    out = tmp_path / "out.mp4"
    render_picture(plan, str(out), settings)
    probed = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                      "stream=width,height", "-of", "csv=p=0", str(out)], text=True)
    assert frame_size(plan, settings) == tuple(int(v) for v in probed.strip().split(",")) == (640, 240)


def test_frame_size_of_a_rotated_non_square_source_is_the_rendered_size(tmp_path: Path) -> None:
    flat = tmp_path / "flat.mp4"
    src = tmp_path / "rot.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=320x240:r=30:d=1",
                    "-vf", "setsar=2", "-c:v", "libx264", "-preset", "ultrafast", str(flat)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", "90", "-i", str(flat),
                    "-c", "copy", str(src)], check=True)
    plan = Plan.from_cuts(SourceInfo("main", str(src), "x", 1.0), [Cut(0.8, 1.0, "silence")])
    settings = RenderSettings(crf=28, preset="ultrafast", scale="iw*sar:ih")
    out = tmp_path / "out.mp4"
    render_picture(plan, str(out), settings)
    probed = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                      "stream=width,height", "-of", "csv=p=0", str(out)], text=True)
    assert frame_size(plan, settings) == tuple(int(v) for v in probed.strip().split(",")[:2])


def test_no_overlays_and_an_empty_set_build_the_commands_the_render_always_had(tmp_path: Path) -> None:
    plan = _cut_plan(tmp_path)
    settings = RenderSettings(crf=28, preset="ultrafast")
    base = build_commands(plan, str(tmp_path / "o.mp4"), settings, "m.m4a")
    assert build_commands(plan, str(tmp_path / "o.mp4"), settings, "m.m4a", overlays=None) == base
    assert build_commands(plan, str(tmp_path / "o.mp4"), settings, "m.m4a", overlays=Overlays(())) == base
    assert "[vbase]" not in " ".join(base[0])


def test_overlays_on_an_hdr_plan_without_a_cube_are_refused(tmp_path: Path) -> None:
    try:
        hdr_codec()
    except RuntimeError:
        pytest.skip("this ffmpeg has no 10-bit HEVC encoder")
    src = tmp_path / "src.mp4"
    _hlg_clip(src, 2.0)
    plan = Plan.from_cuts(SourceInfo("main", str(src), "x", 2.0), [])
    card = _white_card(tmp_path / "card.png")
    with pytest.raises(ValueError, match="cube"):
        build_commands(plan, str(tmp_path / "o.mp4"), RenderSettings(), "m.m4a", overlays=_card_overlays(card))
