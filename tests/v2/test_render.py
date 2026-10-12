"""Render: picture and audio in separate passes, muxed by stream copy.

The split is not cosmetic. ffmpeg 9.0.1 emits digitally silent audio from a
single combined `concat=v=1:a=1` graph for some source regions, so the picture
and the audio are encoded apart and the audio is copied onto the picture --
identical trims and encodes, so the bytes are the ones a single pass owed.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from talkinghead.color import ColorProfile, hdr_codec, hdr_pix_fmt
from talkinghead.pipeline import render_memo_key
from talkinghead.render import (
    AUDIO_RATE,
    RENDER_MIN_TIMEOUT_SEC,
    RENDERER_VERSION,
    RenderSettings,
    _handles,
    _join_gain_db,
    _run,
    _timeout,
    build_commands,
    join_audio,
    render_audio_plan,
    render_plan,
)
from talkinghead.timeline import Cut, Plan, Segment, SourceInfo

HLG = ColorProfile(primaries="bt2020", transfer="arib-std-b67", matrix="bt2020nc",
                   range="tv", pix_fmt="yuv420p10le", hdr=True)


def _clip(path: Path, seconds: float = 6.0) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=30:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(path)],
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
def test_render_removes_the_planned_time_and_applies_the_audio_chain_once(tmp_path: Path, fps: int) -> None:
    src = tmp_path / "src.mp4"
    _clip(src)
    source = SourceInfo("main", str(src), "x", 6.0)
    plan = Plan.from_cuts(source, [Cut(1.0, 2.0, "silence"), Cut(4.0, 4.5, "filler")])
    out = tmp_path / "out.mp4"
    cmd = render_plan(plan, str(out), RenderSettings(crf=28, preset="ultrafast", fps=fps))
    info = _probe(out)
    assert float(info["format"]["duration"]) == pytest.approx(4.5, abs=0.15)
    audio = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert audio["sample_rate"] == "48000"
    assert cmd.count("loudnorm") == 1


def test_render_demo_tier_downscales(tmp_path: Path) -> None:
    src = tmp_path / "src.mp4"
    _clip(src, 3.0)
    plan = Plan.from_cuts(SourceInfo("main", str(src), "x", 3.0), [Cut(1.0, 1.5, "silence")])
    out = tmp_path / "demo.mp4"
    render_plan(plan, str(out), RenderSettings(crf=28, preset="ultrafast", scale="-2:120"))
    assert int(_video(out)["height"]) == 120


def test_render_with_no_cuts_still_runs_the_chain(tmp_path: Path) -> None:
    src = tmp_path / "src.mp4"
    _clip(src, 2.0)
    plan = Plan.from_cuts(SourceInfo("main", str(src), "x", 2.0), [])
    out = tmp_path / "same.mp4"
    cmd = render_plan(plan, str(out), RenderSettings(crf=28, preset="ultrafast"))
    assert out.is_file() and "loudnorm" in cmd


def test_render_refuses_a_plan_that_removes_everything(tmp_path: Path) -> None:
    plan = Plan.from_cuts(SourceInfo("main", "/nonexistent.mp4", "x", 2.0), [Cut(0.0, 2.0, "outro")])
    with pytest.raises(RuntimeError, match="nothing to render"):
        render_plan(plan, str(tmp_path / "x.mp4"), RenderSettings())


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
        build_commands(plan, str(tmp_path / "out.mp4"), RenderSettings())


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
    cmd = render_audio_plan(plan, str(out), RenderSettings(fps=0))
    assert "loudnorm" in cmd
    probed = _probe(out)
    assert [s["codec_type"] for s in probed["streams"]] == ["audio"]
    assert probed["streams"][0]["codec_name"] == "pcm_s16le" and probed["streams"][0]["sample_rate"] == "48000"
    assert float(probed["format"]["duration"]) == pytest.approx(plan.output_duration, abs=0.01)
    assert _volume_max(out) > -30
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
