"""Voice isolation of the rendered audio through Apple's AUSoundIsolation unit (macOS only, opt-in).

Only what the render joins, masters and transcribes uses the isolated track. Transcription,
judging and edge placement keep reading the ORIGINAL audio, so turning isolation on never
re-judges an editorial decision. The steps:

1. Decode the master's audio to 48 kHz float32 WAV: the helper's input (`isolate.swift`).
2. Run the helper, compiled once into the cache. Its raw output is cached per source file and
   helper version. The unit delays its output, so the raw file starts with silence-like latency.
3. Measure that latency by cross-correlation at several loud points. Every point must agree,
   or the run stops: a single fixed delay is only trusted once it is seen to hold across the file.
4. Write the aligned track: trimmed by the measured lag, dropout words replaced by the original
   microphone audio (equal-power crossfades), then blended with the original at `isolate_mix`.

The plan records the track's path, sha256, mix, lag and repaired words, so `talkinghead render PLAN`
rebuilds byte-identical audio from the same source and helper even when the track file is gone.
"""

from __future__ import annotations

import bisect
import hashlib
import os
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .store import sha256_file, stable_key
from .timeline import Plan

# Bump when the track this module writes changes for the same inputs.
ISOLATE_VERSION = 1
# The rate the render decodes at; the track and the helper run at it.
RATE = 48000
# Latency is searched up to this far (the unit adds about 76 ms at 48 kHz).
MAX_LAG_SEC = 0.5
# Extra silence the helper appends so the trimmed track keeps the full length.
TAIL_SEC = 1.0
# Latency points: this many loud windows, spread across the recording.
LAG_POINTS = 6
LAG_WINDOW_SEC = 6.0
# Residual check at the same kind of points, after trimming.
RESIDUAL_MAX_SAMPLES = 4
# A point votes on the lag only when its normalised correlation reaches this: real speech through the unit gives
# 0.95+, and a window of silence or of a passage the unit smeared gives little, which must not decide the trim.
LAG_STRONG_CORR = 0.8
# The lag is measured from at least this many agreeing points.
LAG_MIN_STRONG = 3
# A kept word is a dropout when the isolated level sits this far below the recording's median
# isolated/original ratio (linear). Calibrated on the hotel-lobby and restaurant recordings.
DROPOUT_RATIO = 0.35
# A word longer than this is a bad stamp (it spans a run of speech), not a dropout: it is never repaired.
REPAIR_MAX_SEC = 1.5
# Equal-power crossfade at each repaired word's edges.
XFADE_SEC = 0.010
# Frames of the track written per chunk.
CHUNK_FRAMES = RATE * 10
# Bounds one helper run; a 60-minute recording takes about 13 minutes.
HELPER_TIMEOUT_SEC = 3600.0
# A word carries a ratio only when its original sits above the noise floor by this much. Quiet speech is what the
# unit drops, so the gate stays at the floor: a word above it is speech, not room tone.
ELIGIBLE_ABOVE_FLOOR_DB = 0.0

ISOLATE_SOURCE = Path(__file__).with_name("isolate.swift")


class IsolationError(RuntimeError):
    """Voice isolation cannot run here, or its measurements disagree: the run stops rather than ship unisolated audio."""


@dataclass
class LagPoint:
    at: float
    lag: int
    corr: float
    runner_up: float

    def to_dict(self) -> dict:
        return {"at": round(self.at, 3), "lag_samples": self.lag, "corr": round(self.corr, 3),
                "runner_up": round(self.runner_up, 3)}


# -- platform and helper --------------------------------------------------------------------------

def helper_key() -> str:
    """Identity of the helper: its source and this module's version."""
    digest = hashlib.sha256(ISOLATE_SOURCE.read_bytes() + str(ISOLATE_VERSION).encode()).hexdigest()
    return digest[:16]


def helper_binary(cache_root: Path) -> Path:
    """The compiled helper, built with xcrun swiftc once per helper version and reused from the cache."""
    if sys.platform != "darwin":
        raise IsolationError("voice isolation needs macOS: AUSoundIsolation is an Apple audio unit. "
                             "Drop --isolate-voice / isolate_voice on this machine.")
    if shutil.which("xcrun") is None:
        raise IsolationError("xcrun not found: install the Xcode command line tools (xcode-select --install)")
    binary = Path(cache_root) / "bin" / f"talkinghead-isolate-{helper_key()}"
    if not binary.is_file():
        binary.parent.mkdir(parents=True, exist_ok=True)
        tmp = binary.with_name(binary.name + f".{os.getpid()}.tmp")
        proc = subprocess.run(["xcrun", "swiftc", "-O", str(ISOLATE_SOURCE), "-o", str(tmp)],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise IsolationError(f"could not compile the isolation helper:\n{proc.stderr.strip()[-800:]}")
        os.replace(tmp, binary)
    return binary


def preflight(cache_root: Path) -> None:
    """Fail before any work when the unit is missing: the helper must be able to run here."""
    binary = helper_binary(cache_root)
    proc = subprocess.run([str(binary), "--probe"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise IsolationError(f"AUSoundIsolation is not available on this machine: {proc.stderr.strip()}")


# -- WAV in and out ---------------------------------------------------------------------------------

def _wav_data(path: Path) -> tuple[int, int, int]:
    """(byte offset, frames, channels) of a 32-bit float WAV's sample data."""
    with open(path, "rb") as f:
        head = f.read(12)
        if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise IsolationError(f"not a WAV file: {path}")
        channels = bits = fmt_tag = rate = None
        while True:
            head = f.read(8)
            if len(head) < 8:
                raise IsolationError(f"no data chunk in {path}")
            cid, size = struct.unpack("<4sI", head)
            if cid == b"fmt ":
                body = f.read(size)
                fmt_tag, channels, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
                if size % 2:
                    f.seek(1, 1)
            elif cid == b"data":
                if channels is None or bits != 32 or fmt_tag not in (3, 0xFFFE) or rate != RATE:
                    raise IsolationError(f"{path} must be 48 kHz 32-bit float WAV")
                return f.tell(), size // (4 * channels), channels
            else:
                f.seek(size + (size & 1), 1)


def _memmap(path: Path) -> np.ndarray:
    offset, frames, channels = _wav_data(path)
    return np.memmap(path, dtype="<f4", mode="r", offset=offset, shape=(frames, channels))


def _wav_header(channels: int, frames: int) -> bytes:
    data = frames * channels * 4
    return struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + data, b"WAVE", b"fmt ", 16, 3, channels, RATE,
                       RATE * channels * 4, channels * 4, 32, b"data", data)


def _channels(path: str) -> int:
    proc = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                           "stream=channels", "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    return max(1, int(proc.stdout.strip().splitlines()[0]))


def decode_input(source: str, channels: int, out: Path) -> Path:
    """The master's audio as 48 kHz float32 WAV, with the channel count the render uses."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-i", source, "-vn", "-map", "0:a:0",
                    "-ac", str(channels), "-ar", str(RATE), "-c:a", "pcm_f32le", str(tmp)],
                   check=True, capture_output=True, text=True)
    os.replace(tmp, out)
    return out


# -- helper run and raw cache -----------------------------------------------------------------------

def _tail_frames() -> int:
    return round(TAIL_SEC * RATE)


def raw_path(cache_root: Path, source_sha: str) -> Path:
    """The helper's raw output: per source file and helper version (the mix is applied after it)."""
    key = stable_key({"helper": helper_key(), "tail": _tail_frames()}, ISOLATE_VERSION)
    return Path(cache_root) / source_sha / f"isolated-raw-{key}.wav"


def raw_isolation(wav_in: Path, source_sha: str, cache_root: Path) -> Path:
    """The helper's output for the decoded input, computed at most once per source file and helper version."""
    out = raw_path(cache_root, source_sha)
    if out.is_file():
        return out
    binary = helper_binary(cache_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    # The helper picks the container from the extension, so the temporary keeps .wav.
    tmp = out.with_name(f"{out.stem}.{os.getpid()}.tmp.wav")
    try:
        proc = subprocess.run([str(binary), str(wav_in), str(tmp), str(_tail_frames())],
                              capture_output=True, text=True, timeout=HELPER_TIMEOUT_SEC)
        if proc.returncode != 0:
            raise IsolationError(f"the isolation unit failed (exit {proc.returncode}): "
                                 f"{proc.stderr.strip()[-600:]}")
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return out


# -- latency --------------------------------------------------------------------------------------

def _mono(block: np.ndarray) -> np.ndarray:
    return np.asarray(block, dtype=np.float64).mean(axis=1)


def _xcorr_peak(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[int, float, float]:
    """Normalised cross-correlation of `a` against `b[k:k+len(a)]` for k in [0, max_lag].

    Returns (best k, its correlation, the best correlation at least 20 samples away from it).
    `b` must hold len(a) + max_lag samples.
    """
    n = len(a)
    if len(b) < n + max_lag:
        raise IsolationError("correlation window too short")
    nfft = 1 << (n + len(b) - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(b, nfft) * np.conj(np.fft.rfft(a, nfft)), nfft)[: max_lag + 1]
    energy_a = float(np.sqrt(np.sum(a * a)))
    cum = np.concatenate([[0.0], np.cumsum(b * b)])
    energy_b = np.sqrt(np.maximum(cum[n:n + max_lag + 1] - cum[: max_lag + 1], 0.0))
    ncc = corr / np.maximum(energy_a * energy_b, 1e-30)
    k = int(np.argmax(ncc))
    away = np.ones(len(ncc), dtype=bool)
    away[max(0, k - 20):k + 21] = False
    runner = float(ncc[away].max()) if away.any() else 0.0
    return k, float(ncc[k]), runner


def _loud_points(orig: np.ndarray, count: int, window: int, span: int) -> list[int]:
    """One loud window in each of `count` equal stretches of the recording."""
    stretch = (len(orig) - span) // count
    if stretch < window:
        raise IsolationError(f"recording too short to measure isolation latency ({len(orig) / RATE:.1f} s)")
    points = []
    for i in range(count):
        lo = i * stretch
        candidates = np.linspace(lo, lo + stretch - window, 20).astype(int)
        levels = [float(np.mean(np.square(_mono(orig[p:p + window])))) for p in candidates]
        points.append(int(candidates[int(np.argmax(levels))]))
    return points


def measure_lag(orig: np.ndarray, raw: np.ndarray) -> tuple[int, list[LagPoint]]:
    """The unit's latency in samples: raw[t + lag] matches orig[t]. Every point must agree."""
    max_lag = round(MAX_LAG_SEC * RATE)
    window = round(LAG_WINDOW_SEC * RATE)
    points = []
    for p in _loud_points(orig, LAG_POINTS, window, window + max_lag):
        a = _mono(orig[p:p + window])
        b = _mono(raw[p:p + window + max_lag])
        k, corr, runner = _xcorr_peak(a, b, max_lag)
        points.append(LagPoint(p / RATE, k, corr, runner))
    strong = [pt for pt in points if pt.corr >= LAG_STRONG_CORR]
    if len(strong) < LAG_MIN_STRONG:
        best = max(pt.corr for pt in points)
        raise IsolationError(f"isolated audio does not follow the source: {len(strong)} of {len(points)} points "
                             f"correlate (best {best:.2f}, need {LAG_STRONG_CORR} at {LAG_MIN_STRONG} points)")
    lags = sorted({pt.lag for pt in strong})
    if len(lags) > 1:
        detail = ", ".join(f"{pt.at:.1f} s: {pt.lag}" for pt in strong)
        raise IsolationError(f"the unit's latency is not constant across the recording ({detail} samples); "
                             "the trim cannot be right everywhere, so the run stops")
    return lags[0], points


def residual_lags(orig: np.ndarray, track: np.ndarray, points: list[int]) -> list[dict]:
    """Per point, the lag left between the written track and the original (0 when aligned)."""
    window = round(LAG_WINDOW_SEC * RATE)
    r = RESIDUAL_MAX_SAMPLES
    rows = []
    for p in points:
        if p < r:
            continue
        a = _mono(orig[p:p + window])
        b = _mono(track[p - r:p + window + r])
        k, corr, _ = _xcorr_peak(a, b, 2 * r)
        rows.append({"at": round(p / RATE, 3), "residual_samples": k - r, "corr": round(corr, 3)})
    return rows


# -- dropouts -------------------------------------------------------------------------------------

def _kept_words(plan: Plan, words) -> list:
    """Words that survive whole in the main source: each lies inside one kept segment."""
    segments = sorted((s.start, s.end) for s in plan.segments if s.source == plan.main)
    starts = [s for s, _ in segments]
    kept = []
    for w in words:
        i = bisect.bisect_right(starts, w.start) - 1
        if i >= 0 and segments[i][1] >= w.end:
            kept.append(w)
    return kept


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def find_dropouts(orig: np.ndarray, iso: np.ndarray, plan: Plan, words, floor_db: float) -> tuple[list[dict], float]:
    """Kept words the unit swallowed, and the recording's median isolated/original level ratio.

    A word is measured only when the original sits above the noise floor; the median of those
    ratios is the level isolated speech keeps, and a word below DROPOUT_RATIO of it is a dropout.
    """
    rows = []
    for w in _kept_words(plan, words):
        a, b = round(w.start * RATE), round(w.end * RATE)
        if b - a < 2 or w.end - w.start > REPAIR_MAX_SEC:
            continue
        o = _mono(orig[a:b])
        i = _mono(iso[a:b])
        o_rms, i_rms = _rms(o), _rms(i)
        if o_rms <= 0 or 20 * np.log10(o_rms) <= floor_db + ELIGIBLE_ABOVE_FLOOR_DB:
            continue
        rows.append({"text": w.text, "start": float(w.start), "end": float(w.end),
                     "ratio": i_rms / o_rms})
    if not rows:
        return [], 1.0
    median = float(np.median([r["ratio"] for r in rows]))
    repaired = [r for r in rows if r["ratio"] < DROPOUT_RATIO * median]
    return [{**r, "ratio": round(r["ratio"], 4)} for r in repaired], median


def _regions(repaired: list[dict]) -> list[tuple[int, int]]:
    """Sample spans of the repaired words, merged where their crossfades would meet."""
    xf = round(XFADE_SEC * RATE)
    spans = sorted((round(r["start"] * RATE), round(r["end"] * RATE)) for r in repaired)
    merged: list[list[int]] = []
    for a, b in spans:
        if merged and a - merged[-1][1] < xf:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def _orig_weight(regions: list[tuple[int, int]], start: int, count: int) -> np.ndarray:
    """Per frame in [start, start+count): 1 inside a repaired span, 0 outside, linear over XFADE_SEC at each edge."""
    xf = round(XFADE_SEC * RATE)
    half = xf / 2.0
    w = np.zeros(count, dtype=np.float64)
    t = np.arange(start, start + count, dtype=np.float64)
    for a, b in regions:
        if b + half <= start or a - half >= start + count:
            continue
        ramp_in = np.clip((t - (a - half)) / xf, 0.0, 1.0)
        ramp_out = np.clip(((b + half) - t) / xf, 0.0, 1.0)
        w = np.maximum(w, np.minimum(ramp_in, ramp_out))
    return w


# -- the track ------------------------------------------------------------------------------------

def write_track(orig: np.ndarray, raw: np.ndarray, lag: int, *, mix: float, regions: list[tuple[int, int]],
                gain: float, out: Path) -> str:
    """Write the aligned, repaired, mixed track as 48 kHz float32 WAV; returns its sha256."""
    frames, channels = orig.shape
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(_wav_header(channels, frames))
            for start in range(0, frames, CHUNK_FRAMES):
                n = min(CHUNK_FRAMES, frames - start)
                o = np.asarray(orig[start:start + n], dtype=np.float64)
                iso = np.asarray(raw[lag + start:lag + start + n], dtype=np.float64)
                w = _orig_weight(regions, start, n)[:, None] if regions else None
                if w is not None:
                    # Equal power: the isolated share falls as the original's rises.
                    iso = iso * np.cos(np.pi / 2 * w) + o * gain * np.sin(np.pi / 2 * w)
                out_chunk = mix * iso + (1.0 - mix) * o if mix < 1.0 else iso
                f.write(np.ascontiguousarray(out_chunk, dtype="<f4").tobytes())
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return sha256_file(out)


def _build(source: str, source_sha: str, channels: int, plan: Plan, words, floor_db: float, *,
           mix: float, track: Path, cache_root: Path, scratch: Path, repair: list[dict] | None,
           gain: float | None) -> dict:
    """The track and its record: shared by a fresh build and by a rebuild from a recorded plan.

    `repair` and `gain` come from the recording when rebuilding; otherwise they are measured here.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    wav_in = decode_input(source, channels, scratch / "input.wav")
    try:
        raw = raw_isolation(wav_in, source_sha, cache_root)
        orig_map = _memmap(wav_in)
        raw_map = _memmap(raw)
        lag, points = measure_lag(orig_map, raw_map)
        if repair is None:
            repair, gain = find_dropouts(orig_map, raw_map[lag:lag + orig_map.shape[0]], plan, words, floor_db)
        digest = write_track(orig_map, raw_map, lag, mix=mix, regions=_regions(repair), gain=float(gain), out=track)
        frames = orig_map.shape[0]
        del orig_map, raw_map
    finally:
        wav_in.unlink(missing_ok=True)
    return {
        "source_sha256": source_sha, "helper": helper_key(), "version": ISOLATE_VERSION,
        "mix": mix, "rate": RATE, "channels": channels, "lag_samples": lag,
        "lag_points": [p.to_dict() for p in points], "gain": round(float(gain), 4),
        "repaired": repair,
        "track": {"path": str(track), "sha256": digest, "frames": frames},
    }


def ensure(plan: Plan, words, *, floor_db: float, mix: float, track: Path, cache_root: Path,
           scratch: Path) -> dict:
    """Build the isolated track for `plan` and return the record the plan stores.

    The source file is the plan's main source; `words` are the analysis words the repairs judge.
    """
    src = plan.sources[plan.main]
    channels = _channels(src.path)
    return _build(src.path, src.sha256, channels, plan, words, floor_db, mix=mix, track=track,
                  cache_root=cache_root, scratch=scratch, repair=None, gain=None)


def audio_path(plan: Plan, *, cache_root: Path, scratch: Path | None = None) -> str:
    """The track the render must read, verified against the plan's record.

    A missing or changed track is rebuilt from the source and the recorded parameters. A rebuild
    that does not reproduce the recorded sha256 stops the render: the audio would differ silently.
    """
    rec = plan.isolation
    track = Path(rec["track"]["path"])
    if track.is_file() and sha256_file(track) == rec["track"]["sha256"]:
        return str(track)
    src = plan.sources[plan.main]
    if sha256_file(src.path) != src.sha256:
        raise IsolationError(f"the source {src.path} changed since the plan was made: rerun the cook")
    rebuilt = _build(src.path, src.sha256, rec["channels"], plan, None, 0.0, mix=rec["mix"], track=track,
                     cache_root=cache_root, scratch=scratch or track.parent / "isolation-scratch",
                     repair=rec["repaired"], gain=rec["gain"])
    if rebuilt["track"]["sha256"] != rec["track"]["sha256"]:
        raise IsolationError(f"the rebuilt isolated track ({rebuilt['track']['sha256'][:12]}) does not match the "
                             f"plan's record ({rec['track']['sha256'][:12]}): the helper or the source changed")
    return str(track)


def timing_audio(plan: Plan, source_id: str) -> str:
    """The audio the render was cut from for `source_id`: the isolated track for the main source, when present."""
    if plan.isolation and source_id == plan.main:
        return plan.isolation["track"]["path"]
    return plan.sources[source_id].path


# -- verification facts -----------------------------------------------------------------------------

def verify_facts(plan: Plan, *, expected_words: int, rendered_words: int, scratch: Path) -> dict:
    """What the report and the checks read: the trim's residual on the written track, and the word counts."""
    rec = plan.isolation
    src = plan.sources[plan.main]
    wav_in = decode_input(src.path, rec["channels"], scratch / "verify-input.wav")
    try:
        orig = _memmap(wav_in)
        track = _memmap(Path(rec["track"]["path"]))
        window = round(LAG_WINDOW_SEC * RATE)
        stretch = (orig.shape[0] - window) // LAG_POINTS
        points = [i * stretch + stretch // 2 for i in range(LAG_POINTS)]
        rows = residual_lags(orig, track, points)
    finally:
        wav_in.unlink(missing_ok=True)
    worst = max((abs(r["residual_samples"]) for r in rows), default=0)
    return {"lag_samples": rec["lag_samples"], "residual": rows, "max_abs_residual": worst,
            "expected_words": expected_words, "rendered_words": rendered_words}


def report_facts(plan: Plan) -> dict:
    """The record in the form the report prints: repaired words on the output timeline."""
    rec = plan.isolation
    rows = []
    for r in rec["repaired"]:
        at = plan.source_to_output(r["start"] + 1e-6)
        rows.append({**r, "output_start": None if at is None else round(at, 3)})
    return {"lag_samples": rec["lag_samples"], "lag_ms": round(rec["lag_samples"] * 1000 / RATE, 3),
            "mix": rec["mix"], "gain": rec["gain"], "helper": rec["helper"], "repaired": rows,
            "points": rec["lag_points"]}
