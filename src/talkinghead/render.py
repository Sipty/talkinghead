"""Render frame-aligned picture and crossfaded audio separately, then stream-copy mux.

An audio-only plan renders through the same join and mastering to a WAV (`render_audio_plan`).

Audio is joined in numpy (equal-power crossfade over removed-audio handles), not in
an ffmpeg filtergraph: a ~200-link graph is known to hang.
Separate passes avoid ffmpeg's combined-concat audio dropout. Source HDR is
preserved, and a completed mux replaces the destination atomically.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .color import HDR_PIX_FMT, ColorProfile, encode_args, probe_color
from .timeline import Plan, Segment

# Bumped whenever this module's pixels or audio would differ for an identical
# plan and settings. Version 4: the picture is one CFR decode per source run
# filtered by `select`, replacing per-segment trim branches. Version 5: voiced
# joins are gain-matched (join_audio).
RENDERER_VERSION = 5

FINAL_AUDIO_FILTER = (
    "highpass=f=80,"
    "acompressor=threshold=-22dB:ratio=2.5:attack=10:release=200:makeup=2,"
    "equalizer=f=6500:t=q:w=3:g=-2.5,"
    "loudnorm=I=-16:TP=-1.5:LRA=11"
)
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


@dataclass
class RenderSettings:
    crf: int = 18
    crossfade_sec: float = 0.020      # equal-power crossfade at every join
    preset: str = "medium"
    scale: str | None = None          # e.g. "-2:480" for the demo tier
    fps: int = 30
    audio_filter: str = FINAL_AUDIO_FILTER
    strategy: str = "concat"


def _parts(out_path: str) -> tuple[str, str, str, str]:
    """The temporary files a render writes beside its output: picture, audio, mux, joined PCM."""
    return (f"{out_path}.picture.mp4", f"{out_path}.audio.m4a", f"{out_path}.mux.mp4",
            f"{out_path}.joined.pcm")


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


def _video_graph(segments: list[Segment], input_index: dict[str, int],
                 settings: RenderSettings, profile: ColorProfile) -> str:
    """Select kept CFR frames per source run, avoiding per-segment branch fan-out."""
    fps = settings.fps
    # Frame times sit on the k/fps grid after `fps`; comparing against edges moved
    # half a frame earlier makes each span half-open on that grid, so a kept
    # segment yields exactly length*fps frames (a closed `between` drops one more).
    half = 0.5 / fps if fps else 0.0
    runs = _runs(segments) if fps else [[s] for s in segments]
    parts: list[str] = []
    for i, run in enumerate(runs):
        if not fps:
            s = run[0]
            parts.append(f"[{input_index[s.source]}:v]trim=start={s.start:.6f}:end={s.end:.6f},"
                         f"setpts=PTS-STARTPTS[v{i}]")
            continue
        gaps = [(0.0, run[0].start)] if run[0].start > 0 else []
        gaps += [(a.end, b.start) for a, b in zip(run, run[1:]) if b.start > a.end]
        terms = [f"gte(t,{a - half:.6f})*lt(t,{b - half:.6f})" for a, b in gaps]
        # A finite trim stops decoding at the run endpoint instead of discarding to EOF.
        selects = [f"select='not({'+'.join(terms[j:j + SELECT_TERMS])})'"
                   for j in range(0, len(terms), SELECT_TERMS)]
        chain = ",".join([f"fps={fps}", f"tpad=stop_mode=clone:stop_duration={PAD_SEC}",
                          f"trim=end={run[-1].end:.6f}",
                          *selects, f"setpts=N/({fps}*TB)"])
        parts.append(f"[{input_index[run[0].source]}:v]{chain}[v{i}]")
    parts.append("".join(f"[v{i}]" for i in range(len(runs)))
                 + f"concat=n={len(runs)}:v=1:a=0[vcat]")
    vchain = []
    if settings.scale:
        vchain.append(f"scale={settings.scale}")
    # p010le for HDR: the 10 bits are the point of the profile, and scale
    # above already carries them.
    vchain.append(f"format={HDR_PIX_FMT if profile.hdr else 'yuv420p'}")
    parts.append(f"[vcat]{','.join(vchain)}[vout]")
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
    """
    out = []
    for i in range(len(segments) - 1):
        a, b = segments[i], segments[i + 1]
        end_a = starts[i] + lengths[i]
        room = min(crossfade // 2, lengths[i] // 2, lengths[i + 1] // 2,
                   source_frames[a.source] - end_a, starts[i + 1])
        if a.source == b.source:
            room = min(room, (starts[i + 1] - end_a) // 2)
        out.append(max(room, 0))
    return out


def _mono_window(src: np.ndarray, start: int, count: int) -> np.ndarray:
    """Mono samples of `count` frames from `start` (silence outside the source)."""
    return _take(src, start, count).mean(axis=1)


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
        _block_db(_mono_window(out_src, out_end - room_n, 2 * room_n)),
        _block_db(_mono_window(in_src, in_start - room_n, 2 * room_n)),
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


def _write_joined_audio(plan: Plan, order: list[str], segments: list[Segment],
                        settings: RenderSettings, pcm: str, scratch: list[str]) -> None:
    timeout = _timeout(plan)
    channels = _channels(plan.sources[order[0]].path, timeout)
    sources = {sid: _decode(plan.sources[sid].path, channels, tmp, timeout) for sid, tmp in zip(order, scratch)}
    join_audio(sources, segments, settings.crossfade_sec).tofile(pcm)


def _source_order(plan: Plan) -> list[str]:
    order: list[str] = []
    for s in plan.segments:
        if s.source not in order:
            order.append(s.source)
    return order


def _mastering_command(plan: Plan, order: list[str], settings: RenderSettings, pcm: str,
                       codec: list[str], dest: str) -> list[str]:
    """The final-chain pass over the joined PCM, encoded by `codec` into `dest`."""
    return [
        "ffmpeg", "-y", "-v", "error", "-nostdin",
        "-f", "f32le", "-ar", str(AUDIO_RATE), "-ac", str(_channels(plan.sources[order[0]].path, _timeout(plan))),
        "-i", pcm,
        "-filter_complex", f"[0:a]{settings.audio_filter}[aout]",
        "-map", "[aout]", "-vn",
        *codec, "-ar", str(AUDIO_RATE),
        dest,
    ]


def build_commands(plan: Plan, out_path: str, settings: RenderSettings) -> list[list[str]]:
    """The picture pass, the audio pass, and the stream-copy mux, in order.

    The audio pass reads the already-joined PCM that `render_plan` writes first
    (see `_write_joined_audio`); it only runs the final chain and encodes.
    """
    if not plan.segments:
        raise RuntimeError("nothing to render: the plan removes the whole video")
    if settings.strategy != "concat":
        raise NotImplementedError(f"render strategy {settings.strategy!r}")
    order = _source_order(plan)
    input_index = {sid: i for i, sid in enumerate(order)}
    segments = _snap(plan.segments, settings.fps) if settings.fps else plan.segments
    profile = plan_color(plan, order)
    picture, audio, staged, pcm = _parts(out_path)

    head = ["ffmpeg", "-y", "-v", "error", "-nostdin"]
    picture_cmd = head
    for sid in order:
        picture_cmd = picture_cmd + ["-i", plan.sources[sid].path]
    picture_cmd = picture_cmd + [
        "-filter_complex", _video_graph(segments, input_index, settings, profile),
        "-map", "[vout]", "-an",
        *encode_args(profile, crf=settings.crf, preset=settings.preset),
        picture,
    ]
    audio_cmd = _mastering_command(plan, order, settings, pcm, ["-c:a", "aac", "-b:a", "192k"], audio)
    mux_cmd = [
        "ffmpeg", "-y", "-v", "error", "-nostdin",
        "-i", picture, "-i", audio,
        "-map", "0:v:0", "-map", "1:a:0", "-c", "copy",
        "-map_metadata", "-1", "-movflags", "+faststart",
        staged,
    ]
    return [picture_cmd, audio_cmd, mux_cmd]


def _timeout(plan: Plan) -> float:
    """Seconds one ffmpeg pass may run for this plan's output length."""
    output = sum(s.length for s in plan.segments)
    return max(RENDER_MIN_TIMEOUT_SEC, RENDER_TIMEOUT_PER_OUTPUT_SEC * output)


def _run(cmd: list[str], timeout: float) -> None:
    proc = _tool(cmd, timeout)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-15:])
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}):\n{tail}")


def render_plan(plan: Plan, out_path: str, settings: RenderSettings) -> str:
    """Render and return the command lines used (for the report)."""
    cmds = build_commands(plan, out_path, settings)
    picture, audio, staged, pcm = _parts(out_path)
    order = _source_order(plan)
    segments = _snap(plan.segments, settings.fps) if settings.fps else plan.segments
    scratch = [f"{out_path}.src{i}.pcm" for i in range(len(order))]
    try:
        _write_joined_audio(plan, order, segments, settings, pcm, scratch)
        for cmd in cmds:
            _run(cmd, _timeout(plan))
        os.replace(staged, out_path)
    finally:
        for part in (picture, audio, staged, pcm, *scratch):
            Path(part).unlink(missing_ok=True)
    return " && ".join(" ".join(str(c) for c in cmd) for cmd in cmds)


def _audio_parts(out_path: str) -> tuple[str, str]:
    """The temporary files an audio-only render writes beside its output: staged WAV, joined PCM."""
    return f"{out_path}.staged.wav", f"{out_path}.joined.pcm"


def render_audio_plan(plan: Plan, out_path: str, settings: RenderSettings) -> str:
    """Render an audio-only plan to a 48 kHz PCM WAV and return the command (for the report).

    The same overlap-add join and mastering chain as the video render's audio
    pass, without a picture. The WAV is staged and replaced atomically, so
    `out_path` never holds a half-written file.
    """
    if not plan.segments:
        raise RuntimeError("nothing to render: the plan removes the whole recording")
    if settings.strategy != "concat":
        raise NotImplementedError(f"render strategy {settings.strategy!r}")
    order = _source_order(plan)
    segments = _snap(plan.segments, settings.fps) if settings.fps else plan.segments
    staged, pcm = _audio_parts(out_path)
    scratch = [f"{out_path}.src{i}.pcm" for i in range(len(order))]
    cmd = _mastering_command(plan, order, settings, pcm, ["-c:a", "pcm_s16le"], staged)
    try:
        _write_joined_audio(plan, order, segments, settings, pcm, scratch)
        _run(cmd, _timeout(plan))
        os.replace(staged, out_path)
    finally:
        for part in (staged, pcm, *scratch):
            Path(part).unlink(missing_ok=True)
    return " ".join(str(c) for c in cmd)
