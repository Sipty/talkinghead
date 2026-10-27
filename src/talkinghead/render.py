"""Render frame-aligned picture and crossfaded audio separately, then stream-copy mux.

An audio-only plan renders through the same join and mastering to a WAV (`render_audio_plan`).

Audio is joined in numpy (equal-power crossfade over removed-audio handles), not in
an ffmpeg filtergraph: a ~200-link graph is known to hang.
Separate passes avoid ffmpeg's combined-concat audio dropout. Source HDR is
preserved, and a completed mux replaces the destination atomically.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from . import isolate
from .color import (
    HDR_PIX_FMT,
    ColorProfile,
    decode_args,
    encode_args,
    probe_color,
    without_hwaccel,
    write_sdr_to_hdr_cube,
)
from .store import DEFAULT_CACHE_DIR
from .timeline import Plan, Segment
from .visuals.render import Overlays, overlay_graph, overlay_inputs

# Bumped whenever this module's pixels or audio would differ for an identical
# plan and settings. Version 4: the picture is one CFR decode per source run
# filtered by `select`, replacing per-segment trim branches. Version 5: voiced
# joins are gain-matched (join_audio). Version 6: the master is one fixed,
# true-peak-limited gain (`_loudness_gain_db`) instead of loudnorm, whose dynamic
# mode time-varies the level whenever linear gain would break the peak ceiling.
RENDERER_VERSION = 6

# The master is the joined audio times one fixed gain. No compressor, EQ, highpass or dynamic normalisation: the only
# shaping is the fades in join_audio (program edges and every join). Shown in the report's Audio section.
MASTER_CHAIN_NOTE = ("one fixed gain (no compressor, EQ, highpass or dynamic normalisation); "
                     "fades only at program edges and joins")
# Raw audio is decoded and joined at the output rate.
AUDIO_RATE = 48000
# Program start/end fade, so the first and last sample cannot click.
EDGE_FADE_SEC = 0.010
# Voiced-to-voiced joins: the incoming side is gain-matched to the outgoing level
# and eases back to unity over GAIN_RAMP_SEC, the correction capped at
# GAIN_MATCH_CAP_DB. Joins with a side at room tone keep unity gain.
GAIN_MATCH_CAP_DB = 6.0
GAIN_RAMP_SEC = 0.12
GAIN_ROOM_WINDOW_SEC = 1.25
GAIN_VOICED_ABOVE_ROOM_DB = 12.0
GAIN_BLOCK_SEC = 0.005
# ffmpeg's expression parser fails above 99 functions, so the removed spans are
# split across chained `select` filters of at most this many terms each.
SELECT_TERMS = 40
# Cover sub-frame differences between source-track and requested segment ends.
PAD_SEC = 0.1
# Each ffmpeg pass is stopped after the longer of these, so a stalled encoder
# fails with a message instead of hanging the run. Generous: the failure this
# guards against is a stall, not a slow encode.
RENDER_MIN_TIMEOUT_SEC = 600.0
RENDER_TIMEOUT_PER_OUTPUT_SEC = 3.0
# A plan that revisits a source (reordered blocks) gives every run its own input, seeked this
# far before the run: one shared decode feeding a later branch would buffer frames unboundedly.
SEEK_MARGIN_SEC = 2.0


@dataclass
class RenderSettings:
    crf: int = 18
    crossfade_sec: float = 0.020      # equal-power crossfade at every join
    preset: str = "medium"
    scale: str | None = None          # e.g. "-2:480" for the demo tier
    fps: int = 30
    # The final pass's filter. None measures the fixed gain from the joined audio (the cook's path); a string runs verbatim.
    audio_filter: str | None = None
    loudness_lufs: float = -16.0      # integrated loudness the gain aims at
    true_peak_dbtp: float = -1.5      # the gain never takes the true peak past this
    strategy: str = "concat"


def _parts(out_path: str) -> tuple[str, str, str]:
    """The temporary files a video render writes beside its output: picture, audio master, mux."""
    return f"{out_path}.picture.mp4", f"{out_path}.audio.m4a", f"{out_path}.mux.mp4"


def _snap(segments: list[Segment], fps: float) -> list[Segment]:
    """Segment edges onto the frame grid (plan.json stores milliseconds; the
    plan was already snapped, this removes the rounding)."""
    out = []
    for s in segments:
        a, b = round(s.start * fps) / fps, round(s.end * fps) / fps
        if b > a:
            out.append(Segment(s.source, a, b))
    return out


def plan_color(plan: Plan, order: list[str]) -> ColorProfile:
    """The colour of the whole render; every source in it must agree.

    Concatenating an HDR source with an SDR one is not a render, it is a colour
    error: whichever single profile we encoded would be wrong for the other
    half, so this stops with the disagreement named instead.
    """
    profiles = {sid: probe_color(plan.sources[sid].path) for sid in order}
    families = {p.hdr for p in profiles.values()}
    if len(families) > 1:
        detail = ", ".join(f"{sid}={p.transfer}/{p.pix_fmt}" for sid, p in profiles.items())
        raise RuntimeError(f"plan mixes HDR and SDR sources, one encode cannot be true "
                           f"for both: {detail}")
    encodings = {(p.primaries, p.transfer, p.matrix, p.range) for p in profiles.values()}
    if len(encodings) > 1:
        detail = ", ".join(
            f"{sid}={p.primaries}/{p.transfer}/{p.matrix}/{p.range}"
            for sid, p in profiles.items()
        )
        raise RuntimeError(f"plan sources use incompatible color profiles: {detail}")
    return profiles[order[0]]


def _runs(segments: list[Segment]) -> list[list[Segment]]:
    """Consecutive segments from the same source, in plan order."""
    runs: list[list[Segment]] = []
    for s in segments:
        if runs and runs[-1][0].source == s.source and s.start >= runs[-1][-1].end:
            runs[-1].append(s)
        else:
            runs.append([s])
    return runs


def _picture_runs(segments: list[Segment], fps: int | None) -> list[list[Segment]]:
    return _runs(segments) if fps else [[s] for s in segments]


def _run_inputs(segments: list[Segment], order: list[str], fps: int | None) -> list[tuple[str, float]] | None:
    """(source id, input seek) per picture run, or None when one unseeked input per source serves.

    A plan that comes back to a source (a later run of a source already left) gets one
    input per picture run, seeked to whole frames before it, so every decode is consumed
    in order. Plans that visit each source once render exactly as before.
    """
    sources = [run[0].source for run in _runs(segments)]
    if len(set(sources)) == len(sources):
        return None
    out = []
    for run in _picture_runs(segments, fps):
        t = max(0.0, run[0].start - SEEK_MARGIN_SEC)
        out.append((run[0].source, math.floor(t * fps) / fps if fps else t))
    return out


def _video_graph(segments: list[Segment], input_index: dict[str, int],
                 settings: RenderSettings, profile: ColorProfile,
                 run_inputs: list[tuple[int, float]] | None = None, out: str = "[vout]") -> str:
    """Select kept CFR frames per source run, avoiding per-segment branch fan-out.

    `run_inputs` gives each run its (input index, input seek); by default a run reads its
    source's shared, unseeked input. Times inside a run's chain are relative to its seek.
    """
    fps = settings.fps
    # Frame times sit on the k/fps grid after `fps`; comparing against edges moved
    # half a frame earlier makes each span half-open on that grid, so a kept
    # segment yields exactly length*fps frames (a closed `between` drops one more).
    half = 0.5 / fps if fps else 0.0
    runs = _picture_runs(segments, fps)
    if run_inputs is None:
        run_inputs = [(input_index[run[0].source], 0.0) for run in runs]
    parts: list[str] = []
    for i, (run, (idx, o)) in enumerate(zip(runs, run_inputs)):
        if not fps:
            s = run[0]
            parts.append(f"[{idx}:v]trim=start={s.start - o:.6f}:end={s.end - o:.6f},"
                         f"setpts=PTS-STARTPTS[v{i}]")
            continue
        gaps = [(0.0, run[0].start - o)] if run[0].start > o else []
        gaps += [(a.end - o, b.start - o) for a, b in zip(run, run[1:]) if b.start > a.end]
        terms = [f"gte(t,{a - half:.6f})*lt(t,{b - half:.6f})" for a, b in gaps]
        # A finite trim stops decoding at the run endpoint instead of discarding to EOF.
        selects = [f"select='not({'+'.join(terms[j:j + SELECT_TERMS])})'"
                   for j in range(0, len(terms), SELECT_TERMS)]
        chain = ",".join([f"fps={fps}", f"tpad=stop_mode=clone:stop_duration={PAD_SEC}",
                          f"trim=end={run[-1].end - o:.6f}",
                          *selects, f"setpts=N/({fps}*TB)"])
        parts.append(f"[{idx}:v]{chain}[v{i}]")
    parts.append("".join(f"[v{i}]" for i in range(len(runs)))
                 + f"concat=n={len(runs)}:v=1:a=0[vcat]")
    vchain = []
    if settings.scale:
        vchain.append(f"scale={settings.scale}")
    # p010le for HDR: the 10 bits are the point of the profile, and scale
    # above already carries them.
    vchain.append(f"format={HDR_PIX_FMT if profile.hdr else 'yuv420p'}")
    parts.append(f"[vcat]{','.join(vchain)}{out}")
    return ";".join(parts)


def _tool(cmd: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    """Run an ffmpeg-suite command; a missing binary or a stall gets a message the user can act on."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as e:
        raise RuntimeError(f"{cmd[0]} not found; install ffmpeg (e.g. brew install ffmpeg)") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"{cmd[0]} stalled: no result after {timeout:.0f} s. The output was not replaced; "
            "rerun the same command to retry") from e


def _channels(path: str, timeout: float) -> int:
    out = _tool(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                 "stream=channels", "-of", "csv=p=0", path], timeout)
    try:
        return max(1, int(out.stdout.strip().splitlines()[0]))
    except (ValueError, IndexError):
        raise RuntimeError(f"no audio stream in {path}") from None


def _decode(path: str, channels: int, out_pcm: str, timeout: float) -> np.ndarray:
    """One source's whole audio as float32 (frames, channels) at the output rate."""
    _run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-i", path, "-vn", "-map", "0:a:0",
          "-ac", str(channels), "-ar", str(AUDIO_RATE), "-f", "f32le", out_pcm], timeout)
    return np.memmap(out_pcm, dtype=np.float32, mode="r").reshape(-1, channels)


def _take(src: np.ndarray, start: int, count: int) -> np.ndarray:
    """`count` frames from `start`; anything outside the source reads as silence."""
    out = np.zeros((count, src.shape[1]), dtype=np.float32)
    lo, hi = max(start, 0), min(start + count, len(src))
    if hi > lo:
        out[lo - start:hi - start] = src[lo:hi]
    return out


def _handles(segments: list[Segment], lengths: list[int], starts: list[int],
             source_frames: dict[str, int], crossfade: int) -> list[int]:
    """Half-length of the crossfade at each join, in frames.

    Each side reads `h` frames of removed audio past its edge and the two
    overlap, so the join spans 2h frames centred on the cut. h is clamped to:
    half the requested crossfade; the source bounds; half the removed span
    (a handle never reaches the neighbour's own audio); and half of each
    adjacent segment (both of a segment's joins must fit in its body).
    A jump backwards in the same source (a reordered plan) has no removed span
    between its edges, so only the other bounds apply: its handles are real
    source audio on each side of the seam and the crossfade still holds.
    """
    out = []
    for i in range(len(segments) - 1):
        a, b = segments[i], segments[i + 1]
        end_a = starts[i] + lengths[i]
        room = min(crossfade // 2, lengths[i] // 2, lengths[i + 1] // 2,
                   source_frames[a.source] - end_a, starts[i + 1])
        if a.source == b.source and starts[i + 1] >= end_a:
            room = min(room, (starts[i + 1] - end_a) // 2)
        out.append(max(room, 0))
    return out


def _mono_window(src: np.ndarray, start: int, count: int) -> np.ndarray:
    """Mono samples of `count` frames from `start` (silence outside the source)."""
    return _take(src, start, count).mean(axis=1)


def _source_window(src: np.ndarray, start: int, end: int) -> np.ndarray:
    """Mono samples of the real source in [start, end). Never zero padding: room tone is read from actual audio."""
    lo, hi = max(0, start), min(len(src), end)
    return src[lo:hi].mean(axis=1) if hi > lo else np.zeros(0)


def _block_db(mono: np.ndarray) -> np.ndarray:
    """RMS dBFS of consecutive GAIN_BLOCK_SEC blocks: the envelope the join grader reads."""
    block = round(GAIN_BLOCK_SEC * AUDIO_RATE)
    n = len(mono) // block
    if n == 0:
        return np.zeros(0)
    b = mono[: n * block].reshape(n, block).astype(np.float64)
    return 20 * np.log10(np.sqrt(np.mean(b * b, axis=1)) + 1e-12)


def _join_gain_db(out_src: np.ndarray, out_end: int, in_src: np.ndarray, in_start: int) -> float:
    """Gain (dB) matching the incoming side to the outgoing level at one join.

    Each side is read 20-200 ms from the join, as the join grader reads it. The
    gain is zero unless both sides are voiced above the local room tone, so
    quiet joins keep unity gain.
    """
    rate = AUDIO_RATE
    room_n = round(GAIN_ROOM_WINDOW_SEC * rate)
    pre = _block_db(_mono_window(out_src, out_end - round(0.2 * rate), round(0.18 * rate)))
    post = _block_db(_mono_window(in_src, in_start + round(0.02 * rate), round(0.18 * rate)))
    if not len(pre) or not len(post):
        return 0.0
    room = float(np.percentile(np.concatenate([
        _block_db(_source_window(out_src, out_end - room_n, out_end + room_n)),
        _block_db(_source_window(in_src, in_start - room_n, in_start + room_n)),
    ]), 25))
    pre_db, post_db = float(pre.mean()), float(post.mean())
    if min(pre_db, post_db) <= room + GAIN_VOICED_ABOVE_ROOM_DB:
        return 0.0
    return float(np.clip(pre_db - post_db, -GAIN_MATCH_CAP_DB, GAIN_MATCH_CAP_DB))


def join_audio(sources: dict[str, np.ndarray], segments: list[Segment], crossfade_sec: float) -> np.ndarray:
    """Overlap-add the segments with equal-power crossfades at every join.

    The result is exactly round(sum(lengths) * rate) frames: output boundaries
    are rounded from the running total, not per segment, so rounding never
    accumulates. Program start and end get a short fade so they cannot click.
    A voiced-to-voiced join gain-matches its incoming side to the outgoing level,
    the correction easing back to unity over GAIN_RAMP_SEC.
    """
    if not segments:
        raise RuntimeError("nothing to render: the plan removes the whole video")
    cum = 0.0
    bounds = [0]
    for s in segments:
        cum += s.end - s.start
        bounds.append(round(cum * AUDIO_RATE))
    lengths = [bounds[i + 1] - bounds[i] for i in range(len(segments))]
    starts = [round(s.start * AUDIO_RATE) for s in segments]
    frames = {sid: len(a) for sid, a in sources.items()}
    half = _handles(segments, lengths, starts, frames, round(crossfade_sec * AUDIO_RATE))
    gains = [_join_gain_db(sources[segments[i - 1].source], round(segments[i - 1].end * AUDIO_RATE),
                           sources[segments[i].source], starts[i]) for i in range(1, len(segments))]
    channels = next(iter(sources.values())).shape[1]
    out = np.zeros((bounds[-1], channels), dtype=np.float32)
    edge = round(EDGE_FADE_SEC * AUDIO_RATE)
    for i, s in enumerate(segments):
        n = lengths[i]
        if n == 0:
            continue
        hl = half[i - 1] if i > 0 else 0
        hr = half[i] if i < len(half) else 0
        chunk = _take(sources[s.source], starts[i] - hl, n + hl + hr)
        if i > 0 and gains[i - 1]:
            # Chunk index hl is the join. Full gain through the crossfade, then unity.
            after = np.maximum(np.arange(len(chunk)) - hl, 0)
            ramp = np.clip(1.0 - after / (GAIN_RAMP_SEC * AUDIO_RATE), 0.0, 1.0)
            chunk = chunk * np.power(10.0, gains[i - 1] * ramp / 20.0).astype(np.float32)[:, None]
        env = np.ones(len(chunk), dtype=np.float32)
        if hl:
            ramp = np.sin(np.pi / 2 * (np.arange(2 * hl) + 0.5) / (2 * hl))
            env[:2 * hl] = ramp
        else:
            k = min(edge, n // 2) if i == 0 else 0
            if k:
                env[:k] = np.sin(np.pi / 2 * (np.arange(k) + 0.5) / k)
        if hr:
            ramp = np.cos(np.pi / 2 * (np.arange(2 * hr) + 0.5) / (2 * hr))
            env[len(env) - 2 * hr:] = ramp
        else:
            k = min(edge, n // 2) if i == len(segments) - 1 else 0
            if k:
                env[len(env) - k:] = np.cos(np.pi / 2 * (np.arange(k) + 0.5) / k)
        out[bounds[i] - hl:bounds[i + 1] + hr] += chunk * env[:, None]
    return out


def _audio_sources(plan: Plan) -> dict[str, str]:
    """The file each source's audio is read from: the isolated track for the main source when isolation is on.

    The track is verified (or rebuilt from the plan's record) once here, so one render reads one checked file.
    """
    return {sid: (isolate.audio_path(plan, cache_root=DEFAULT_CACHE_DIR) if plan.isolation and sid == plan.main
                  else plan.sources[sid].path) for sid in _source_order(plan)}


def _write_joined_audio(plan: Plan, order: list[str], segments: list[Segment],
                        settings: RenderSettings, pcm: str, scratch: list[str], audio: dict[str, str]) -> None:
    timeout = _timeout(plan)
    channels = _channels(audio[order[0]], timeout)
    sources = {sid: _decode(audio[sid], channels, tmp, timeout) for sid, tmp in zip(order, scratch)}
    join_audio(sources, segments, settings.crossfade_sec).tofile(pcm)


def _source_order(plan: Plan) -> list[str]:
    order: list[str] = []
    for s in plan.segments:
        if s.source not in order:
            order.append(s.source)
    return order


def _loudness_gain_db(pcm: str, channels: int, settings: RenderSettings, timeout: float) -> float:
    """The one fixed gain (dB) toward the loudness target, stopped by the true-peak ceiling.

    Measured with loudnorm's analysis on the joined audio, before any filter. The gain is the lower of the gain to
    target and the gain the true-peak ceiling allows, and is never compressed up to the target. Where the cap is the
    lower one the master lands quieter; verification reports the measured shortfall, not its cause.
    Silence (no peak and no level) takes unity gain. A signal too short for an integrated measurement has no level
    to aim at, so it is only attenuated when its finite peaks exceed the ceiling; it is never boosted. NaN or
    infinite-peak measurements are an error, not silence.
    """
    cmd = ["ffmpeg", "-v", "info", "-nostdin", "-f", "f32le", "-ar", str(AUDIO_RATE), "-ac", str(channels), "-i", pcm,
           "-af", f"loudnorm=I={settings.loudness_lufs}:TP={settings.true_peak_dbtp}:print_format=json",
           "-f", "null", "-"]
    proc = _tool(cmd, timeout)
    blocks = re.findall(r"\{[^{}]*\}", proc.stderr or "")
    if proc.returncode != 0 or not blocks:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-15:])
        raise RuntimeError(f"could not measure the loudness of the joined audio (exit {proc.returncode}):\n{tail}")
    measured = json.loads(blocks[-1])
    level, peak = float(measured["input_i"]), float(measured["input_tp"])
    if math.isnan(level) or math.isnan(peak) or level == math.inf or peak == math.inf:
        raise RuntimeError(f"loudness measurement of the joined audio is invalid: input_i={level}, input_tp={peak}")
    if peak == -math.inf:
        if level == -math.inf:
            return 0.0
        raise RuntimeError(f"loudness measurement is inconsistent: input_i={level} with no true peak")
    if level == -math.inf:
        return min(0.0, settings.true_peak_dbtp - peak)
    return min(settings.loudness_lufs - level, settings.true_peak_dbtp - peak)


def _final_filter(settings: RenderSettings, pcm: str, channels: int, timeout: float) -> str:
    """The final pass's filter: the settings' own when given, else the measured fixed gain as one volume step."""
    if settings.audio_filter is not None:
        return settings.audio_filter
    return f"volume={_loudness_gain_db(pcm, channels, settings, timeout):.3f}dB"


def _mastering_command(settings: RenderSettings, pcm: str, codec: list[str], dest: str,
                       channels: int) -> list[str]:
    """The final-chain pass over the joined PCM, encoded by `codec` into `dest`."""
    if settings.audio_filter is None:
        raise RuntimeError("the final audio pass has no filter: measure the loudness gain first (_final_filter)")
    return [
        "ffmpeg", "-y", "-v", "error", "-nostdin",
        "-f", "f32le", "-ar", str(AUDIO_RATE), "-ac", str(channels),
        "-i", pcm,
        "-filter_complex", f"[0:a]{settings.audio_filter}[aout]",
        "-map", "[aout]", "-vn",
        *codec, "-ar", str(AUDIO_RATE),
        dest,
    ]


def _drawn(overlays: Overlays | None) -> bool:
    """Whether `overlays` put anything on the picture; None and an empty set leave the commands as they were."""
    return overlays is not None and bool(overlays.placements)


def frame_size(plan: Plan, settings: RenderSettings) -> tuple[int, int]:
    """The (width, height) of the picture render's frames, as ffmpeg computes them.

    One real frame of the first source is decoded by ffmpeg (autorotate and sample aspect ratio included)
    and run through the picture graph's own `scale`, then the written frame is probed. Nothing here
    evaluates a scale expression or assumes a size, so an overlay's width and the head-clip check
    measure the true frame.
    """
    if not plan.segments:
        raise RuntimeError("nothing to render: the plan removes the whole video")
    path = plan.sources[_source_order(plan)[0]].path
    timeout = RENDER_MIN_TIMEOUT_SEC
    with tempfile.TemporaryDirectory(prefix="talkinghead-size-") as work:
        frame = str(Path(work) / "frame.mkv")
        cmd = ["ffmpeg", "-y", "-v", "error", "-nostdin", "-i", path, "-map", "0:v:0", "-frames:v", "1"]
        # The format and codec only make the frame writable; neither changes its size.
        chain = [f"scale={settings.scale}"] if settings.scale else []
        proc = _tool(cmd + ["-vf", ",".join([*chain, "format=yuv420p"]), "-c:v", "ffv1", frame], timeout)
        if proc.returncode != 0 or not Path(frame).exists():
            raise RuntimeError(f"ffmpeg cannot render one frame of {path} through scale {settings.scale!r}: "
                               f"{(proc.stderr or '').strip()[-200:]}")
        probed = _tool(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height", "-of", "csv=p=0", frame], timeout)
    fields = probed.stdout.strip().splitlines()[0].rstrip(",").split(",") if probed.stdout.strip() else []
    if probed.returncode != 0 or len(fields) < 2:
        raise RuntimeError(f"cannot read the frame size of {path}: {(probed.stderr or '').strip()[-200:]}")
    return int(fields[0]), int(fields[1])


def _picture_command(plan: Plan, dest: str, settings: RenderSettings, *,
                     overlays: Overlays | None = None, cube: str | None = None) -> list[str]:
    """The picture pass alone: the plan's frame-snapped cut, colour handling and encoder, no audio stream.

    `overlays` are spliced after the cut's concat/scale/format chain, in the same graph and encode; their
    times are output times, which is what `t` is once the cut's `setpts` has run. The stills are extra
    inputs after the sources, so only the sources carry hardware-decode flags.
    """
    if not plan.segments:
        raise RuntimeError("nothing to render: the plan removes the whole video")
    if settings.strategy != "concat":
        raise NotImplementedError(f"render strategy {settings.strategy!r}")
    order = _source_order(plan)
    input_index = {sid: i for i, sid in enumerate(order)}
    segments = _snap(plan.segments, settings.fps) if settings.fps else plan.segments
    profile = plan_color(plan, order)
    seeked = _run_inputs(segments, order, settings.fps)
    inputs = seeked if seeked is not None else [(sid, 0.0) for sid in order]
    decode = {sid: decode_args(probe_color(plan.sources[sid].path).pix_fmt) for sid in order}
    cmd = ["ffmpeg", "-y", "-v", "error", "-nostdin"]
    for sid, seek in inputs:
        cmd = cmd + decode[sid] + (["-ss", f"{seek:.6f}"] if seek else []) + ["-i", plan.sources[sid].path]
    per_run = [(i, seek) for i, (_, seek) in enumerate(seeked)] if seeked is not None else None
    graph = _video_graph(segments, input_index, settings, profile, per_run,
                         out="[vbase]" if _drawn(overlays) else "[vout]")
    if _drawn(overlays):
        if not settings.fps:
            raise ValueError("overlays need a frame rate: render with a fps above 0")
        cmd += overlay_inputs(overlays, fps=settings.fps, duration=sum(s.length for s in segments))
        graph += ";" + ";".join(overlay_graph(
            overlays, base="[vbase]", first_input=len(inputs), width=frame_size(plan, settings)[0],
            color=profile, cube=cube))
    return cmd + [
        "-filter_complex", graph,
        "-map", "[vout]", "-an",
        *encode_args(profile, crf=settings.crf, preset=settings.preset),
        dest,
    ]


def build_commands(plan: Plan, out_path: str, settings: RenderSettings, audio_file: str, *,
                   overlays: Overlays | None = None, cube: str | None = None) -> list[list[str]]:
    """The picture pass and the stream-copy mux of `audio_file`'s first audio stream, in order.

    The audio is never encoded here: it is the master `render_audio_plan` wrote, so the video carries
    those packets unchanged. `overlays` (with `cube` for an HDR plan) go into the picture pass, the
    only video encode.
    """
    picture, _, staged = _parts(out_path)
    picture_cmd = _picture_command(plan, picture, settings, overlays=overlays, cube=cube)
    mux_cmd = [
        "ffmpeg", "-y", "-v", "error", "-nostdin",
        "-i", picture, "-i", audio_file,
        "-map", "0:v:0", "-map", "1:a:0", "-c", "copy",
        "-map_metadata", "-1", "-movflags", "+faststart",
        staged,
    ]
    return [picture_cmd, mux_cmd]


def _timeout(plan: Plan) -> float:
    """Seconds one ffmpeg pass may run for this plan's output length."""
    output = sum(s.length for s in plan.segments)
    return max(RENDER_MIN_TIMEOUT_SEC, RENDER_TIMEOUT_PER_OUTPUT_SEC * output)


def _run(cmd: list[str], timeout: float) -> None:
    proc = _tool(cmd, timeout)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-15:])
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}):\n{tail}")


def _run_decoding(cmd: list[str], timeout: float) -> list[str]:
    """Run a pass; one that fails with hardware decode reruns in software. Returns the command that ran.

    Only an ffmpeg failure reruns: a stall raises from `_tool` as before, since a second
    attempt would double a pass that is already over its time.
    """
    soft = without_hwaccel(cmd)
    if soft == cmd:
        _run(cmd, timeout)
        return cmd
    if _tool(cmd, timeout).returncode == 0:
        return cmd
    print("talkinghead: hardware decode failed; rerunning the pass with software decode", file=sys.stderr)
    _run(soft, timeout)
    return soft


def render_picture(plan: Plan, out_path: str, settings: RenderSettings, *, overlays: Overlays | None = None) -> str:
    """Render the picture alone (no audio stream) and return the command line.

    Any plan can be rendered this way, an excerpt included, without touching audio. The file is staged
    beside the output and replaced atomically, so `out_path` never holds a half-written picture.
    `overlays` are composited inside this one encode; an HDR plan's SDR-to-HDR cube lives in a temp
    dir that is gone when the call returns.
    """
    staged = f"{out_path}.staged.mp4"
    work: tempfile.TemporaryDirectory | None = None
    try:
        cube = None
        if _drawn(overlays) and plan.segments:
            profile = plan_color(plan, _source_order(plan))
            if profile.hdr:
                work = tempfile.TemporaryDirectory(prefix="talkinghead-overlays-")
                cube = str(write_sdr_to_hdr_cube(Path(work.name) / "sdr_to_hdr.cube", profile))
        cmd = _picture_command(plan, staged, settings, overlays=overlays, cube=cube)
        ran = _run_decoding(cmd, _timeout(plan))
        os.replace(staged, out_path)
    finally:
        Path(staged).unlink(missing_ok=True)
        if work:
            work.cleanup()
    return " ".join(str(c) for c in ran)


def render_plan(plan: Plan, out_path: str, settings: RenderSettings, *, audio_file: str | None = None,
                overlays: Overlays | None = None) -> str:
    """Render the picture and mux an audio master onto it by stream copy; return the command lines for the report.

    `audio_file` is the master to carry. Without it the master is rendered first by `render_audio_plan`
    (the one audio encoder), so a video's audio is always a standalone master whose packets the mux keeps.
    `overlays` are drawn in the picture pass, so the video is still encoded once.
    """
    if not plan.segments:
        raise RuntimeError("nothing to render: the plan removes the whole video")
    picture, audio_tmp, staged = _parts(out_path)
    made_master = audio_file is None
    cmds: list[str] = []
    try:
        if audio_file is None:
            cmds.append(render_audio_plan(plan, audio_tmp, settings))
            audio_file = audio_tmp
        cmds.append(render_picture(plan, picture, settings, overlays=overlays))
        mux = build_commands(plan, out_path, settings, audio_file)[1]
        _run(mux, _timeout(plan))
        cmds.append(" ".join(str(c) for c in mux))
        os.replace(staged, out_path)
    finally:
        for part in (picture, staged, *((audio_tmp,) if made_master else ())):
            Path(part).unlink(missing_ok=True)
    return " && ".join(cmds)


# The audio containers a master can be written to, and the codec arguments each one carries.
AUDIO_CODECS = {
    ".wav": ["-c:a", "pcm_s16le"],
    ".m4a": ["-c:a", "aac", "-b:a", "192k"],
}


def render_audio_plan(plan: Plan, out_path: str, settings: RenderSettings, *, original: bool = False) -> str:
    """Render a plan's audio to a 48 kHz `.wav` (PCM) or `.m4a` (the AAC 192k master) and return the command.

    The same overlap-add join and mastering chain for both; the `.m4a` is the exact audio a video render
    muxes, so it is what a lock hashes. The file is staged beside the output (keeping its suffix) and
    replaced atomically, so `out_path` never holds a half-written file. `original` reads the source audio
    instead of the isolated track (what the output pass transcribes under isolation).
    """
    suffix = Path(out_path).suffix.lower()
    if suffix not in AUDIO_CODECS:
        raise ValueError(f"cannot render audio to {out_path!r}: use a {' or '.join(AUDIO_CODECS)} output path")
    if not plan.segments:
        raise RuntimeError("nothing to render: the plan removes the whole recording")
    if settings.strategy != "concat":
        raise NotImplementedError(f"render strategy {settings.strategy!r}")
    order = _source_order(plan)
    audio = {sid: si.path for sid, si in plan.sources.items()} if original else _audio_sources(plan)
    segments = _snap(plan.segments, settings.fps) if settings.fps else plan.segments
    staged, pcm = f"{out_path}.staged{suffix}", f"{out_path}.joined.pcm"
    scratch = [f"{out_path}.src{i}.pcm" for i in range(len(order))]
    try:
        _write_joined_audio(plan, order, segments, settings, pcm, scratch, audio)
        timeout = _timeout(plan)
        channels = _channels(audio[order[0]], timeout)
        settings = replace(settings, audio_filter=_final_filter(settings, pcm, channels, timeout))
        cmd = _mastering_command(settings, pcm, AUDIO_CODECS[suffix], staged, channels)
        _run(cmd, timeout)
        os.replace(staged, out_path)
    finally:
        for part in (staged, pcm, *scratch):
            Path(part).unlink(missing_ok=True)
    return " ".join(str(c) for c in cmd)
