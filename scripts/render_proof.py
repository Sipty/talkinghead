#!/usr/bin/env python3
"""Milestone 0 render proof: which ffmpeg cut-render strategy is safe at 31 minutes?

Renders the same cut plans with three strategies and measures, at full length:
  1. duration vs plan math (container / video track / audio track),
  2. audio dropouts (long silent runs in the output whose source span is not silent),
  3. A/V sync on the VIDEO track (frame correlation output-vs-source) and on the
     audio track, at six probe points; lip-sync error = video offset - audio offset.

Strategies
  A  filter_complex: per-keep trim/atrim + setpts/asetpts, 40 ms afades, concat=n=N
  B  single pass select/aselect with CFR normalisation (fps=30 first)
  B2 same as B, but the not(between(...)+...) expression is split across chained
     select/aselect filters of at most EXPR_CHUNK terms each (ffmpeg 9.0.1's expression
     parser rejects >= 100 terms with "Invalid argument" / "Cannot allocate memory")
  C  encode every keep range to its own file (-ss/-to before -i), join with the
     concat demuxer and stream copy

Usage (from repo root, uv run python scripts/render_proof.py ...):
  render  --plan P1 --strategy A [--head 120]     render one plan/strategy
  measure --plan P1 --strategy A                  measure an existing render
  all     [--plans P0 P1 P2] [--strategies A B C] render + measure everything
  report                                          print markdown tables from *.measure.json

Outputs go to `--out-dir` (or `$TALKINGHEAD_RENDER_PROOF_OUT_DIR`) and the source video is `--source` (or `$TALKINGHEAD_RENDER_PROOF_SOURCE`). Both are required; nothing is written into the repo.
Sign convention for every offset: POSITIVE = output content appears LATER than the
plan says it should (output lags the source), in milliseconds.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SOURCE_ENV = "TALKINGHEAD_RENDER_PROOF_SOURCE"
OUT_DIR_ENV = "TALKINGHEAD_RENDER_PROOF_OUT_DIR"

# Proof encoding: speed over quality, sync does not depend on crf.
ENC_VIDEO = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p"]
ENC_AUDIO = ["-c:a", "aac", "-b:a", "128k", "-ar", "48000"]
FADE = 0.040  # seconds, per-segment afade in strategy A (mirrors src/talkinghead/render.py)
FPS = 30
EXPR_CHUNK = 40  # between() terms per select expression in strategy B2 (parser limit is 99)

# Analysis parameters
AUDIO_SR = 16000  # analysis sample rate
RMS_FRAME_S = 0.050
DROPOUT_DB = -60.0
DROPOUT_MIN_S = 1.5
DROPOUT_SLACK_S = 1.0  # tolerated output-vs-source drift when matching a silent run to a source pause
DROPOUT_SRC_DB = -55.0  # source counts as silent below this (5 dB margin: AAC re-encode pushes near-silence lower)
PROBE_FRACS = (0.05, 0.20, 0.40, 0.60, 0.80, 0.97)
PROBE_CLEARANCE_S = 3.0  # probe windows stay this far from any cut point
PROBE_WINDOW_S = 3.0
PROBE_MARGIN_S = 8.0  # extra source context on each side: lag search range is +-8 s

MAX_CONCURRENT_FFMPEG = 2
_FFMPEG_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_FFMPEG)


# ----------------------------------------------------------------------------
# Plans and keep-range math
# ----------------------------------------------------------------------------
Span = tuple[float, float]


def plan_cuts(name: str, duration: float) -> list[Span]:
    """Cut plans used by the proof. Each cut is a source-time span to remove."""
    if name == "P0":
        return []
    if name == "P1":  # 15 cuts: remove 1.0 s at t=120, 240, ..., 1800
        return [(float(t), float(t) + 1.0) for t in range(120, 1801, 120)]
    if name == "P2":  # 208 cuts: remove 0.6 s every 9 s from t=9 to t=1880 (boundaries on exact frame times)
        return [(float(t), float(t) + 0.6) for t in range(9, 1881, 9)]
    if name == "P2j":  # same, but boundaries jittered off the frame grid (realistic ASR-derived times)
        return [(t + 0.0137, t + 0.0137 + 0.6) for t in range(9, 1881, 9)]
    raise ValueError(f"unknown plan {name}")


def merge_cuts(cuts: list[Span]) -> list[Span]:
    merged: list[Span] = []
    for a, b in sorted(cuts):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def keep_ranges(cuts: list[Span], duration: float) -> list[Span]:
    """Complement of the cuts within [0, duration]; empty keeps are dropped."""
    keeps: list[Span] = []
    pos = 0.0
    for a, b in merge_cuts(cuts):
        a, b = max(a, 0.0), min(b, duration)
        if a > pos:
            keeps.append((pos, a))
        pos = max(pos, b)
    if pos < duration:
        keeps.append((pos, duration))
    return keeps


def removed_within(cuts: list[Span], limit: float) -> float:
    """Seconds of cut material that lie inside [0, limit] (the video track ends before the container)."""
    return sum(max(0.0, min(b, limit) - max(a, 0.0)) for a, b in merge_cuts(cuts))


def output_to_source(t_out: float, keeps: list[Span]) -> float:
    """Map an output timestamp to the source timestamp it should show."""
    pos = 0.0
    for s, e in keeps:
        d = e - s
        if t_out < pos + d:
            return s + (t_out - pos)
        pos += d
    return keeps[-1][1] + (t_out - pos)  # past the end: extrapolate


def output_cut_points(keeps: list[Span]) -> list[float]:
    """Output timestamps where a splice happens."""
    pts, pos = [], 0.0
    for s, e in keeps[:-1]:
        pos += e - s
        pts.append(pos)
    return pts


# ----------------------------------------------------------------------------
# ffmpeg helpers
# ----------------------------------------------------------------------------
def run(cmd: list[str], *, capture: bool = True, log: Path | None = None) -> subprocess.CompletedProcess:
    """Run an ffmpeg/ffprobe command inside the global concurrency slot."""
    with _FFMPEG_SLOTS:
        proc = subprocess.run(cmd, check=False, capture_output=capture, text=False)
    if log is not None:
        with log.open("ab") as fh:
            fh.write(b"$ " + " ".join(cmd).encode() + b"\n")
            fh.write(proc.stderr or b"")
            fh.write(b"\n")
    if proc.returncode != 0:
        tail = (proc.stderr or b"").decode(errors="replace").strip().splitlines()[-25:]
        raise RuntimeError(f"command failed (exit {proc.returncode}): {' '.join(cmd[:12])}...\n" + "\n".join(tail))
    return proc


def ffprobe_json(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,nb_streams:stream=index,codec_type,codec_name,r_frame_rate,avg_frame_rate,"
        "nb_frames,duration,time_base,start_time,sample_rate",
        "-of", "json", str(path),
    ]
    return json.loads(run(cmd).stdout)


def source_info(path: Path) -> dict:
    info = ffprobe_json(path)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next(s for s in info["streams"] if s["codec_type"] == "audio")
    return {
        "path": str(path),
        "container_duration": float(info["format"]["duration"]),
        "video_duration": float(v["duration"]),
        "video_nb_frames": int(v["nb_frames"]),
        "audio_duration": float(a["duration"]),
        "avg_frame_rate": v["avg_frame_rate"],
    }


def ffmpeg_version() -> str:
    return subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]


# ----------------------------------------------------------------------------
# Strategy commands
# ----------------------------------------------------------------------------
def filter_complex_A(keeps: list[Span], fade: float = FADE) -> str:
    """Mirror of the strategy-A filter graph in src/talkinghead/render.py, without the audio chain."""
    parts, v_labels, a_labels = [], [], []
    for i, (start, end) in enumerate(keeps):
        dur = end - start
        fd = min(fade, dur / 3)
        parts.append(f"[0:v]trim=start={start:.6f}:end={end:.6f},setpts=PTS-STARTPTS[v{i}]")
        parts.append(
            f"[0:a]atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS,"
            f"afade=t=in:curve=tri:st=0:d={fd:.3f},"
            f"afade=t=out:curve=tri:st={dur - fd:.3f}:d={fd:.3f}[a{i}]"
        )
        v_labels.append(f"[v{i}]")
        a_labels.append(f"[a{i}]")
    parts.append("".join(v + a for v, a in zip(v_labels, a_labels)) + f"concat=n={len(keeps)}:v=1:a=1[vout][aout]")
    return ";".join(parts)


def build_cmd_A(src: Path, keeps: list[Span], out: Path) -> list[str]:
    return [
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-i", str(src),
        "-filter_complex", filter_complex_A(keeps),
        "-map", "[vout]", "-map", "[aout]",
        *ENC_VIDEO, *ENC_AUDIO, str(out),
    ]


def select_expr(cuts: list[Span]) -> str:
    cuts = merge_cuts(cuts)
    if not cuts:
        return "1"
    return "not(" + "+".join(f"between(t\\,{a:.6f}\\,{b:.6f})" for a, b in cuts) + ")"


def build_cmd_B(src: Path, cuts: list[Span], out: Path, fps: int = FPS, chunk: int | None = None) -> list[str]:
    """Strategy B (chunk=None: one expression) or B2 (chunk=N: chained selects of N terms each).

    select does not rewrite timestamps, so every chained select still sees the original `t`;
    setpts/asetpts run once at the end of each chain.
    """
    cuts = merge_cuts(cuts)
    if chunk is None or not cuts:
        exprs = [select_expr(cuts)]
    else:
        exprs = [select_expr(cuts[i:i + chunk]) for i in range(0, len(cuts), chunk)]
    vsel = ",".join(f"select='{e}'" for e in exprs)
    asel = ",".join(f"aselect='{e}'" for e in exprs)
    return [
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-i", str(src),
        "-vf", f"fps={fps},{vsel},setpts=N/FRAME_RATE/TB",
        "-af", f"{asel},asetpts=N/SR/TB",
        *ENC_VIDEO, *ENC_AUDIO, str(out),
    ]


def build_cmd_C(src: Path, keeps: list[Span], out: Path, workdir: Path) -> tuple[list[list[str]], Path, list[str]]:
    """Return (per-segment encode commands, concat list path, final join command)."""
    workdir.mkdir(parents=True, exist_ok=True)
    seg_cmds, seg_paths = [], []
    for i, (s, e) in enumerate(keeps):
        seg = workdir / f"seg_{i:04d}.mp4"
        seg_paths.append(seg)
        seg_cmds.append([
            "ffmpeg", "-y", "-hide_banner", "-nostats",
            "-ss", f"{s:.6f}", "-to", f"{e:.6f}", "-i", str(src),
            *ENC_VIDEO, *ENC_AUDIO, str(seg),
        ])
    lst = workdir / "concat.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in seg_paths))
    join = ["ffmpeg", "-y", "-hide_banner", "-nostats", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)]
    return seg_cmds, lst, join


# ----------------------------------------------------------------------------
# Render driver
# ----------------------------------------------------------------------------
def tag(plan: str, strategy: str) -> str:
    return f"{plan}_{strategy}"


def render(plan: str, strategy: str, out_dir: Path, src: Path, head: float | None = None) -> dict:
    """Render one plan with one strategy; returns and writes a small JSON record."""
    out_dir.mkdir(parents=True, exist_ok=True)
    info = source_info(src)
    duration = info["container_duration"]
    cuts = plan_cuts(plan, duration)
    if head is not None:  # smoke test: pretend the source ends at `head`
        cuts = merge_cuts(cuts + [(head, duration)])
    keeps = keep_ranges(cuts, duration)
    out = out_dir / f"{tag(plan, strategy)}.mp4"
    log = out_dir / f"{tag(plan, strategy)}.ffmpeg.log"
    log.write_text("")
    cmds: list[list[str]] = []
    t0 = time.monotonic()
    if strategy == "A":
        cmd = build_cmd_A(src, keeps, out)
        cmds.append(cmd)
        run(cmd, log=log)
    elif strategy in ("B", "B2"):
        cmd = build_cmd_B(src, cuts, out, chunk=EXPR_CHUNK if strategy == "B2" else None)
        cmds.append(cmd)
        run(cmd, log=log)
    elif strategy == "C":
        seg_cmds, lst, join = build_cmd_C(src, keeps, out, out_dir / f"{tag(plan, strategy)}_segs")
        cmds.extend(seg_cmds)
        cmds.append(join)
        # Segments run sequentially inside this render's slot so wall time is comparable
        # with A/B (one ffmpeg process at a time per render).
        for c in seg_cmds:
            run(c, log=log)
        run(join, log=log)
    else:
        raise ValueError(strategy)
    wall = time.monotonic() - t0
    rec = {
        "plan": plan, "strategy": strategy, "output": str(out), "wall_s": wall,
        "head": head, "n_cuts": len(cuts), "n_keeps": len(keeps),
        "removed_s": sum(b - a for a, b in merge_cuts(cuts)),
        "cuts": cuts, "keeps": keeps, "source": info,
        "ffmpeg": ffmpeg_version(), "commands": cmds,
    }
    (out_dir / f"{tag(plan, strategy)}.render.json").write_text(json.dumps(rec, indent=1))
    print(f"[render] {tag(plan, strategy)}: {wall:.1f}s wall, {len(keeps)} keeps", flush=True)
    return rec


# ----------------------------------------------------------------------------
# Measurements
# ----------------------------------------------------------------------------
def decode_audio(path: Path, start: float | None = None, dur: float | None = None, sr: int = AUDIO_SR,
                 conform: bool = False) -> np.ndarray:
    """Decode (a span of) the audio track to mono float32 at `sr`.

    conform=True makes the sample stream follow the packet timestamps (aresample async: drop
    overlapping samples, pad gaps), i.e. what a timestamp-driven player renders. conform=False
    returns the raw sequential decode, i.e. what a sample-sequential player renders. The two
    differ when packets carry more samples than their timestamps advance (strategy C's join).
    """
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start is not None:
        cmd += ["-ss", f"{start:.6f}"]
    cmd += ["-i", str(path)]
    if dur is not None:
        cmd += ["-t", f"{dur:.6f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(sr)]
    if conform:
        cmd += ["-af", "aresample=async=1:min_hard_comp=0.001:first_pts=0"]
    cmd += ["-f", "f32le", "pipe:1"]
    return np.frombuffer(run(cmd).stdout, dtype=np.float32)


def decode_frames(path: Path, start: float, dur: float, w: int = 160, h: int = 90) -> np.ndarray:
    """Decode frames as (n, h, w) uint8 gray, timestamps passed through untouched."""
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-ss", f"{start:.6f}", "-i", str(path), "-t", f"{dur:.6f}",
        "-an", "-vf", f"scale={w}:{h},format=gray", "-fps_mode", "passthrough",
        "-f", "rawvideo", "pipe:1",
    ]
    raw = np.frombuffer(run(cmd).stdout, dtype=np.uint8)
    n = len(raw) // (w * h)
    return raw[: n * w * h].reshape(n, h, w)


def frame_pts(path: Path, start: float, dur: float) -> list[float]:
    """PTS (s) of the video frames with start <= pts < start + dur, i.e. what `-ss start -t dur` decodes."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts_time",
        "-of", "csv=p=0", "-read_intervals", f"{start:.6f}%{start + dur + 0.5:.6f}", str(path),
    ]
    pts = [float(x.strip().rstrip(",")) for x in run(cmd).stdout.decode().split() if x.strip(",")]
    return [t for t in pts if start - 1e-6 <= t < start + dur - 1e-6]


def measure_duration(out: Path, rec: dict) -> dict:
    info = ffprobe_json(out)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next(s for s in info["streams"] if s["codec_type"] == "audio")
    src = rec["source"]
    cuts = [tuple(c) for c in rec["cuts"]]
    removed = rec["removed_s"]
    expected = src["container_duration"] - removed
    # The source video track ends 5.3 s before the container; the best a renderer can do
    # for video is the kept part of the video track.
    expected_video = src["video_duration"] - removed_within(cuts, src["video_duration"])
    num, den = (int(x) for x in v["avg_frame_rate"].split("/"))
    avg_fps = num / den if den else float("nan")
    nb_frames = int(v.get("nb_frames", 0) or 0)
    container = float(info["format"]["duration"])
    vdur = float(v["duration"])
    adur = float(a["duration"])
    return {
        "expected_s": expected,
        "expected_video_s": expected_video,
        "container_s": container, "video_s": vdur, "audio_s": adur,
        "video_nb_frames": nb_frames,
        "video_frames_duration_s": nb_frames / avg_fps if avg_fps else float("nan"),
        "avg_frame_rate": v["avg_frame_rate"], "avg_fps": avg_fps, "r_frame_rate": v["r_frame_rate"],
        "video_start_time": float(v.get("start_time", 0) or 0), "audio_start_time": float(a.get("start_time", 0) or 0),
        "delta_ms": {
            "container": (container - expected) * 1000,
            "video": (vdur - expected) * 1000,
            "video_vs_video_expected": (vdur - expected_video) * 1000,
            "frames": (nb_frames / avg_fps - expected) * 1000 if avg_fps else float("nan"),
            "audio": (adur - expected) * 1000,
        },
    }


def rms_db_frames(x: np.ndarray, sr: int = AUDIO_SR, frame_s: float = RMS_FRAME_S) -> np.ndarray:
    n = int(sr * frame_s)
    m = len(x) // n
    if m == 0:
        return np.array([])
    fr = x[: m * n].reshape(m, n).astype(np.float64)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-9))


def silent_runs(db: np.ndarray, frame_s: float, thresh_db: float, min_s: float) -> list[Span]:
    """Runs (start_s, end_s) of consecutive frames below thresh_db lasting at least min_s."""
    below = db < thresh_db
    runs, i, n = [], 0, len(below)
    while i < n:
        if below[i]:
            j = i
            while j < n and below[j]:
                j += 1
            if (j - i) * frame_s >= min_s:
                runs.append((i * frame_s, j * frame_s))
            i = j
        else:
            i += 1
    return runs


def measure_dropouts(out: Path, rec: dict, src: Path,
                     audio_offsets: list[tuple[float, float]] | None = None) -> dict:
    """audio_offsets: (t_out, offset_s) pairs from measure_sync; the output->source mapping is
    corrected by the interpolated offset so a strategy whose audio has drifted by seconds is
    still compared against the right source span."""
    keeps = [tuple(k) for k in rec["keeps"]]
    if audio_offsets:
        xs, ys = zip(*sorted(audio_offsets))
        offset_at = lambda t: float(np.interp(t, xs, ys))  # noqa: E731
    else:
        offset_at = lambda t: 0.0  # noqa: E731
    raw_len = len(decode_audio(out)) / AUDIO_SR
    audio = decode_audio(out, conform=True)
    db = rms_db_frames(audio)
    candidates = silent_runs(db, RMS_FRAME_S, DROPOUT_DB, DROPOUT_MIN_S)
    dropouts, benign = [], []
    for a, b in candidates:
        # Content at output time t is the planned content of t - offset (offset > 0 = output lags).
        sa, sb = output_to_source(a - offset_at(a), keeps), output_to_source(b - offset_at(b), keeps)
        # Is there a matching pause in the source? Search the mapped span widened by
        # DROPOUT_SLACK_S on each side so an A/V drift of up to that much (strategy B) does not
        # turn a real pause into a false dropout. A run of silence at least 80% as long as the
        # candidate counts as the same pause.
        src_audio = decode_audio(src, max(0.0, sa - DROPOUT_SLACK_S), (b - a) + 2 * DROPOUT_SLACK_S)
        src_db = rms_db_frames(src_audio)
        src_runs = silent_runs(src_db, RMS_FRAME_S, DROPOUT_SRC_DB, 0.8 * (b - a))
        item = {"out_start_s": a, "out_end_s": b, "len_s": b - a, "src_start_s": sa, "src_end_s": sb,
                "src_fraction_below_thresh": float((src_db < DROPOUT_SRC_DB).mean()) if len(src_db) else float("nan"),
                "src_longest_silence_s": max((e - s0 for s0, e in silent_runs(src_db, RMS_FRAME_S, DROPOUT_SRC_DB, 0.0)), default=0.0),
                "src_max_db": float(src_db.max()) if len(src_db) else None,
                "audio_offset_applied_s": offset_at(a)}
        (benign if src_runs else dropouts).append(item)
    return {
        "decoded_s": len(audio) / AUDIO_SR,
        "raw_sequential_decoded_s": raw_len,
        # > 0 means the file carries more audio samples than its timestamps account for
        "audio_sequential_excess_ms": (raw_len - len(audio) / AUDIO_SR) * 1000,
        "overall_rms_db": float(20 * np.log10(max(float(np.sqrt((audio.astype(np.float64) ** 2).mean())), 1e-9))),
        "n_candidates": len(candidates), "dropouts": dropouts, "benign_silences": benign,
        "n_dropouts": len(dropouts),
    }


def output_keep_spans(keeps: list[Span]) -> list[Span]:
    """Keep ranges expressed in output time."""
    spans, pos = [], 0.0
    for s, e in keeps:
        spans.append((pos, pos + (e - s)))
        pos += e - s
    return spans


def choose_probes(out_duration: float, keeps: list[Span]) -> list[tuple[float, float]]:
    """Probe (start_time, clearance) pairs at PROBE_FRACS of the output.

    The 3 s output window must not cross a splice. It is kept PROBE_CLEARANCE_S away from the
    splices bounding its keep segment when the segment is long enough; otherwise it is centred in
    the segment and the achieved clearance (segment slack / 2) is reported instead.
    """
    spans = output_keep_spans(keeps)
    probes = []
    for f in PROBE_FRACS:
        t = f * out_duration
        o_s, o_e = next((sp for sp in spans if sp[0] <= t < sp[1]), spans[-1])
        # Bound the window by real splices only; the file start/end are not splices.
        lo = o_s + PROBE_CLEARANCE_S if o_s > 0 else 0.0
        hi = (o_e - PROBE_CLEARANCE_S if o_e < spans[-1][1] else out_duration - 0.5) - PROBE_WINDOW_S
        if lo <= hi:
            start = min(max(t - PROBE_WINDOW_S / 2, lo), hi)
        else:
            start = (o_s + o_e) / 2 - PROBE_WINDOW_S / 2
        clearance = min(start - o_s if o_s > 0 else math.inf, o_e - (start + PROBE_WINDOW_S) if o_e < spans[-1][1] else math.inf)
        probes.append((round(start, 3), round(clearance, 3)))
    return probes


def best_lag_frames(out_frames: np.ndarray, src_frames: np.ndarray, margin_frames: int) -> tuple[int, float, list[float]]:
    """Align output window inside the longer source window by full-frame mean absolute difference.

    Returns (lag_frames, score_ratio, mad_curve). lag_frames = index into src window where output[0]
    matches, minus margin_frames. lag>0 means output[0] shows source content LATER than nominal
    (output is AHEAD of plan); the caller flips the sign to the 'positive = output lags' convention.
    """
    n = len(out_frames)
    m = len(src_frames)
    if n == 0 or m < n:
        return 0, float("nan"), []
    o = out_frames.astype(np.int16).reshape(n, -1)
    sf = src_frames.astype(np.int16).reshape(m, -1)
    mads = [float(np.abs(o - sf[k:k + n]).mean()) for k in range(m - n + 1)]
    mads_arr = np.array(mads)
    k = int(mads_arr.argmin())
    # Peak sharpness: best MAD vs. median MAD (lower is sharper; ~1 means no alignment found)
    ratio = float(mads_arr[k] / np.median(mads_arr)) if np.median(mads_arr) > 0 else float("nan")
    return k - margin_frames, ratio, mads


def luma_xcorr_lag(out_luma: np.ndarray, src_luma: np.ndarray, margin_frames: int) -> tuple[int, float]:
    """Spec method: normalised cross-correlation of per-frame mean-luma sequences."""
    n = len(out_luma)
    m = len(src_luma)
    if n < 2 or m < n:
        return 0, float("nan")
    o = out_luma - out_luma.mean()
    best_k, best_c = 0, -np.inf
    for k in range(m - n + 1):
        s = src_luma[k:k + n]
        s = s - s.mean()
        denom = np.linalg.norm(o) * np.linalg.norm(s)
        c = float(o @ s / denom) if denom > 0 else 0.0
        if c > best_c:
            best_c, best_k = c, k
    return best_k - margin_frames, best_c


def audio_lag_ms(out_a: np.ndarray, src_a: np.ndarray, margin_samples: int, sr: int = AUDIO_SR) -> tuple[float, float]:
    """Cross-correlate a short output window against a longer source window (FFT).

    Returns (lag_ms, peak_ratio) with lag>0 meaning output[0] corresponds to source content
    LATER than nominal (output ahead of plan); the caller flips the sign.
    """
    n, m = len(out_a), len(src_a)
    if n == 0 or m < n:
        return float("nan"), float("nan")
    o = out_a.astype(np.float64) - out_a.mean()
    s = src_a.astype(np.float64) - src_a.mean()
    size = 1 << int(math.ceil(math.log2(n + m)))
    # correlation c[k] = sum_i o[i] * s[i+k]
    c = np.fft.irfft(np.conj(np.fft.rfft(o, size)) * np.fft.rfft(s, size), size)[: m - n + 1]
    # normalise by local energy of the source window to avoid biasing to loud parts
    s2 = np.concatenate([[0.0], np.cumsum(s ** 2)])
    energy = s2[n:m + 1] - s2[: m - n + 1]
    norm = np.sqrt(energy * float((o ** 2).sum())) + 1e-12
    cn = c / norm
    k = int(cn.argmax())
    peak = float(cn[k])
    others = np.delete(cn, np.arange(max(0, k - sr // 100), min(len(cn), k + sr // 100)))
    ratio = float(peak / (np.abs(others).max() + 1e-12)) if len(others) else float("nan")
    return (k - margin_samples) * 1000.0 / sr, ratio


def measure_sync(out: Path, rec: dict, out_duration: float, src: Path) -> dict:
    keeps = [tuple(k) for k in rec["keeps"]]
    probes = choose_probes(out_duration, keeps)
    results = []
    for t_out, clearance in probes:
        t_src = output_to_source(t_out, keeps)
        # Video: output window [t_out, +3 s]; source window [t_src-8, t_src+11] (clamped to the file)
        src_start = max(0.0, t_src - PROBE_MARGIN_S)
        margin_frames = int(round((t_src - src_start) * FPS))
        margin_samples = int(round((t_src - src_start) * AUDIO_SR))
        of = decode_frames(out, t_out, PROBE_WINDOW_S)
        sf = decode_frames(src, src_start, PROBE_WINDOW_S + PROBE_MARGIN_S + (t_src - src_start))
        lag_mad, mad_ratio, _ = best_lag_frames(of, sf, margin_frames)
        lag_luma, luma_corr = luma_xcorr_lag(of.mean(axis=(1, 2)), sf.mean(axis=(1, 2)), margin_frames)
        # Audio
        oa = decode_audio(out, t_out, PROBE_WINDOW_S)
        sa = decode_audio(src, src_start, PROBE_WINDOW_S + PROBE_MARGIN_S + (t_src - src_start))
        a_lag_ms, a_ratio = audio_lag_ms(oa, sa, margin_samples)
        # Convert to the 'positive = output lags' convention (flip sign).
        video_offset_frames = -lag_mad
        video_offset_frames_ms = video_offset_frames * 1000.0 / FPS
        audio_offset_ms = -a_lag_ms
        # Sub-frame video offset from real frame PTS: the content of the matched source frame
        # (pts p_s, nominal output time t_out + p_s - t_src) actually appears at output pts p_o.
        opts = frame_pts(out, t_out, PROBE_WINDOW_S)
        spts = frame_pts(src, src_start, PROBE_WINDOW_S + PROBE_MARGIN_S + (t_src - src_start))
        # ffmpeg -ss/-t and ffprobe -read_intervals may disagree by one frame at the window end;
        # only the first output frame and the matched source frame are used, so allow that.
        pts_ok = (abs(len(opts) - len(of)) <= 1 and abs(len(spts) - len(sf)) <= 1
                  and len(opts) > 0 and 0 <= margin_frames + lag_mad < len(spts))
        if pts_ok:
            video_offset_ms = (opts[0] - t_out - (spts[margin_frames + lag_mad] - t_src)) * 1000.0
        else:
            video_offset_ms = video_offset_frames_ms
        results.append({
            "t_out": t_out, "t_src": t_src, "splice_clearance_s": clearance,
            "n_out_frames": int(len(of)), "n_src_frames": int(len(sf)),
            "video_offset_frames": video_offset_frames, "video_offset_frames_ms": video_offset_frames_ms,
            "video_offset_ms": video_offset_ms, "video_offset_subframe_ok": pts_ok,
            "video_mad_peak_ratio": mad_ratio,
            "video_offset_frames_lumaxcorr": -lag_luma, "video_luma_corr": luma_corr,
            "audio_offset_ms": audio_offset_ms, "audio_peak_ratio": a_ratio,
            "lipsync_error_ms": video_offset_ms - audio_offset_ms,
        })
    return {"probes": results}


def measure(plan: str, strategy: str, out_dir: Path, src: Path) -> dict:
    rec = json.loads((out_dir / f"{tag(plan, strategy)}.render.json").read_text())
    out = Path(rec["output"])
    dur = measure_duration(out, rec)
    sync = measure_sync(out, rec, dur["container_s"], src)
    drops = measure_dropouts(out, rec, src, [(p["t_out"], p["audio_offset_ms"] / 1000) for p in sync["probes"]])
    m = {"plan": plan, "strategy": strategy, "output": str(out), "wall_s": rec["wall_s"],
         "n_cuts": rec["n_cuts"], "n_keeps": rec["n_keeps"], "removed_s": rec["removed_s"],
         "duration": dur, "dropouts": drops, "sync": sync}
    (out_dir / f"{tag(plan, strategy)}.measure.json").write_text(json.dumps(m, indent=1))
    d = dur["delta_ms"]
    print(f"[measure] {tag(plan, strategy)}: dC={d['container']:+.0f}ms dV={d['video']:+.0f}ms "
          f"(vs video-track {d['video_vs_video_expected']:+.0f}ms) dA={d['audio']:+.0f}ms "
          f"dropouts={drops['n_dropouts']} seq-excess={drops['audio_sequential_excess_ms']:+.0f}ms "
          f"fps={dur['avg_frame_rate']}", flush=True)
    for p in sync["probes"]:
        print(f"   t={p['t_out']:8.2f}s video {p['video_offset_ms']:+7.1f}ms (mad-ratio {p['video_mad_peak_ratio']:.2f}, "
              f"luma-xcorr {p['video_offset_frames_lumaxcorr']:+d}f r={p['video_luma_corr']:.2f}) "
              f"audio {p['audio_offset_ms']:+7.2f}ms (peak {p['audio_peak_ratio']:.1f}) "
              f"lipsync {p['lipsync_error_ms']:+7.1f}ms", flush=True)
    return m


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def fmt_ms(x: float) -> str:
    return f"{x:+.0f}" if abs(x) >= 10 else f"{x:+.1f}"


def report(out_dir: Path, plans=("P0", "P1", "P2", "P2j"), strategies=("A", "B", "B2", "C")) -> str:
    lines = []
    for plan in plans:
        rows = []
        for s in strategies:
            f = out_dir / f"{tag(plan, s)}.measure.json"
            if not f.exists():
                logf = out_dir / f"{tag(plan, s)}.ffmpeg.log"
                if logf.exists() and not (out_dir / f"{tag(plan, s)}.render.json").exists():
                    rows.append(f"| {s} | FAILED (ffmpeg error, see {logf.name}) | - | - | - | - | - | - | - |")
                continue
            m = json.loads(f.read_text())
            d = m["duration"]["delta_ms"]
            pr = m["sync"]["probes"]
            v = " / ".join(fmt_ms(p["video_offset_ms"]) for p in pr)
            a = " / ".join(fmt_ms(p["audio_offset_ms"]) for p in pr)
            lip = " / ".join(fmt_ms(p["lipsync_error_ms"]) for p in pr)
            rows.append(
                f"| {s} | {m['wall_s']/60:.1f} min | {fmt_ms(d['container'])} / {fmt_ms(d['video'])} ({fmt_ms(d['video_vs_video_expected'])}) / {fmt_ms(d['audio'])} "
                f"| {m['dropouts']['n_dropouts']} | {fmt_ms(m['dropouts'].get('audio_sequential_excess_ms', float('nan')))} "
                f"| {v} | {a} | {lip} | {m['duration']['avg_frame_rate']} |"
            )
        if rows:
            mfile = next(out_dir / f"{tag(plan, s)}.measure.json" for s in strategies if (out_dir / f"{tag(plan, s)}.measure.json").exists())
            probes = json.loads(mfile.read_text())["sync"]["probes"]
            lines.append(f"### {plan}\n")
            lines.append("Probe output times (s) [splice clearance s]: " + ", ".join(
                f"{p['t_out']:.0f} [{p['splice_clearance_s']:.1f}]" for p in probes) + "\n")
            lines.append("| strategy | wall | duration delta ms: container / video (vs video track) / audio | dropouts | audio seq. excess ms | video offset ms per probe | audio offset ms per probe | lipsync error ms per probe | avg_frame_rate |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            lines.extend(rows)
            lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def cmd_all(args) -> None:
    jobs = [(p, s) for p in args.plans for s in args.strategies]
    failures = {}

    def do(job):
        p, s = job
        try:
            if not args.measure_only:
                render(p, s, args.out_dir, args.source, head=args.head)
            measure(p, s, args.out_dir, args.source)
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:  # keep going, report at the end
            failures[tag(p, s)] = str(exc)
            print(f"[FAIL] {tag(p, s)}: {exc}", file=sys.stderr, flush=True)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FFMPEG) as ex:
        list(ex.map(do, jobs))
    print(report(args.out_dir, args.plans, args.strategies))
    if failures:
        print("FAILURES:\n" + json.dumps(failures, indent=1), file=sys.stderr)
        sys.exit(1)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=os.environ.get(OUT_DIR_ENV),
                    help=f"proof output directory (default: ${OUT_DIR_ENV})")
    ap.add_argument("--source", type=Path, default=os.environ.get(SOURCE_ENV),
                    help=f"source video (default: ${SOURCE_ENV})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render")
    r.add_argument("--plan", required=True, choices=["P0", "P1", "P2", "P2j"])
    r.add_argument("--strategy", required=True, choices=["A", "B", "B2", "C"])
    r.add_argument("--head", type=float, default=None, help="smoke test: treat source as ending at this many seconds")

    m = sub.add_parser("measure")
    m.add_argument("--plan", required=True, choices=["P0", "P1", "P2", "P2j"])
    m.add_argument("--strategy", required=True, choices=["A", "B", "B2", "C"])

    a = sub.add_parser("all")
    a.add_argument("--plans", nargs="+", default=["P0", "P1", "P2"])
    a.add_argument("--strategies", nargs="+", default=["A", "B", "C"])
    a.add_argument("--head", type=float, default=None)
    a.add_argument("--measure-only", action="store_true")

    sub.add_parser("report")

    args = ap.parse_args(argv)
    if args.out_dir is None:
        ap.error(f"--out-dir or ${OUT_DIR_ENV} is required")
    if args.cmd != "report" and args.source is None:
        ap.error(f"--source or ${SOURCE_ENV} is required")
    if args.cmd == "render":
        render(args.plan, args.strategy, args.out_dir, args.source, head=args.head)
    elif args.cmd == "measure":
        measure(args.plan, args.strategy, args.out_dir, args.source)
    elif args.cmd == "all":
        cmd_all(args)
    elif args.cmd == "report":
        print(report(args.out_dir))


if __name__ == "__main__":
    main()
