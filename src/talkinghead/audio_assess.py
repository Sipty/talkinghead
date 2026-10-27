"""Audio assessment: the recording's own sound, measured, then read once by the configured judge.

Advisory only. It measures the original audio (before any isolation or mastering), so the judge sees the
room as recorded. What comes back is written to the report and nothing else: no render, config or plan reads it,
so it cannot enable isolation, denoising or compression.

Every measurement that cannot be established is null (JSON null) with the status naming why. Nothing is ever
encoded as NaN or Infinity.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess

import numpy as np

from . import prompts
from .audio import VOICED_ABOVE_FLOOR_DB, noise_floor
from .judge import Judge, vote

# Bump when a measurement or the judged question changes: stored assessments then re-run.
ASSESS_VERSION = 4
RATE = 16000
FRAME = 160                  # 10 ms at RATE: the level envelope
WINDOW = 4096                # 256 ms: ~3.9 Hz bins, enough to separate 50 Hz from 60 Hz
HOP = 2048
# Below 24-bit PCM precision, resampling residue is not usable mono room-sound evidence.
MONO_ANALYSIS_FLOOR = 2.0 ** -24
BANDS = (("below_200_hz", 0.0, 200.0), ("200_1000_hz", 200.0, 1000.0),
         ("1_4_khz", 1000.0, 4000.0), ("4_8_khz", 4000.0, 8000.0))
PROCESS_TIMEOUT_SEC = 1800.0
# Facts that are null when the source has no sound (or, for the loudness ones, when loudnorm cannot establish them).
# mono_sample_peak_dbfs is the peak of the mono downmix only: a downmix can sum channels above the original peak, so it
# never proves the original channels clipped (the true peak above does that).
LEVEL_KEYS = ("floor_dbfs", "quiet_frame_share", "quiet_level_spread_db", "speech_p90_dbfs", "mono_sample_peak_dbfs",
              "input_integrated_lufs", "input_true_peak_dbtp", "input_lra_lu", "input_threshold_lufs",
              "gain_to_target_db", "gain_ceiling_from_peak_db", "quiet_spectrum")


def guidelines_fingerprint(guidelines: str) -> str:
    return hashlib.sha256(guidelines.encode()).hexdigest()[:16]


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-15:])
        raise RuntimeError(f"ffmpeg failed measuring the source audio (exit {proc.returncode}):\n{tail}")
    return proc


def _finite(value, digits: int = 2) -> float | None:
    """A measured number, or None when it is not finite (loudnorm's -inf for a signal too short to gate)."""
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def _mono_samples(path: str, timeout: float) -> np.ndarray:
    """The source's audio as mono float32 in [-1, 1] at RATE. float32 keeps an hour of audio near 230 MB."""
    proc = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-vn", "-ac", "1", "-ar", str(RATE),
                           "-f", "f32le", "-"], capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="replace").strip().splitlines()[-15:]
        raise RuntimeError(f"could not decode the source audio (exit {proc.returncode}):\n" + "\n".join(tail))
    return np.frombuffer(proc.stdout, dtype=np.float32)


def _loudness(path: str, target_lufs: float, true_peak_dbtp: float, timeout: float) -> dict:
    """Integrated loudness, true peak and loudness range of the full-rate source, from loudnorm's analysis only."""
    proc = _run(["ffmpeg", "-v", "info", "-nostdin", "-i", path, "-vn", "-af",
                 f"loudnorm=I={target_lufs}:TP={true_peak_dbtp}:LRA=11:print_format=json", "-f", "null", "-"],
                timeout)
    blocks = re.findall(r"\{[^{}]*\}", proc.stderr or "")
    if not blocks:
        raise RuntimeError("loudnorm printed no measurement for the source audio")
    measured = json.loads(blocks[-1])
    return {"input_integrated_lufs": _finite(measured["input_i"]),
            "input_true_peak_dbtp": _finite(measured["input_tp"]),
            "input_lra_lu": _finite(measured["input_lra"]),
            "input_threshold_lufs": _finite(measured["input_thresh"])}


def _tone_db(power: np.ndarray, freqs: np.ndarray, hz: float) -> float | None:
    """A tone's peak over the median power on either side of it, in dB. None when the bins cannot tell."""
    peak_mask = (freqs >= hz - 4) & (freqs <= hz + 4)
    ref_mask = ((freqs >= hz - 20) & (freqs <= hz - 6)) | ((freqs >= hz + 6) & (freqs <= hz + 20))
    if not peak_mask.any() or not ref_mask.any():
        return None
    peak, ref = float(power[peak_mask].max()), float(np.median(power[ref_mask]))
    if peak <= 0 or ref <= 0:
        return None
    return _finite(10 * np.log10(peak / ref), 1)


def _quiet_spectrum(x: np.ndarray, quiet_frame: np.ndarray) -> dict | None:
    """Average spectrum of the windows whose every 10 ms frame is quiet: what the room sounds like with nobody talking.

    None when no such window exists, or when the windows carry no power (digital silence has no spectrum to share).
    """
    taper = np.hanning(WINDOW)
    acc = np.zeros(WINDOW // 2 + 1)
    count = 0
    for start in range(0, len(x) - WINDOW + 1, HOP):
        first, last = start // FRAME, (start + WINDOW - 1) // FRAME
        if not quiet_frame[first:last + 1].all():
            continue
        spec = np.fft.rfft(x[start:start + WINDOW].astype(np.float64) * taper)
        acc += np.abs(spec) ** 2
        count += 1
    if count == 0:
        return None
    power = acc / count
    total = float(power.sum())
    if not (math.isfinite(total) and total > 0):
        return None
    freqs = np.fft.rfftfreq(WINDOW, 1.0 / RATE)
    shares = {name: _finite(power[(freqs >= lo) & (freqs < hi)].sum() / total, 3) for name, lo, hi in BANDS}
    return {"windows": count, "band_share": shares,
            "hum_50hz_db": _tone_db(power, freqs, 50.0), "hum_60hz_db": _tone_db(power, freqs, 60.0)}


def measure(source: str, *, target_lufs: float, true_peak_dbtp: float, timeout: float = PROCESS_TIMEOUT_SEC) -> dict:
    """The measured facts the judge reads. Every number comes from the source audio as recorded.

    `status` distinguishes a measured mono signal, a silent source, and unavailable mono analysis.
    A cancelling downmix cannot establish that the original multichannel audio is silent.
    """
    x = _mono_samples(source, timeout)
    if len(x) < FRAME * 10:
        raise RuntimeError(f"the source audio is too short to assess ({len(x) / RATE:.2f} s)")
    base = {"duration_s": round(len(x) / RATE, 1), "target_lufs": target_lufs, "true_peak_ceiling_dbtp": true_peak_dbtp}
    loud = _loudness(source, target_lufs, true_peak_dbtp, timeout)
    level, peak = loud["input_integrated_lufs"], loud["input_true_peak_dbtp"]
    mono_peak = max(abs(float(x.min())), abs(float(x.max())))
    if mono_peak <= MONO_ANALYSIS_FLOOR:
        missing = {key: None for key in LEVEL_KEYS}
        if level is None and peak is None:
            return {**base, "status": "silence", **missing}
        return {
            **base, "status": "mono_analysis_unavailable", **missing, **loud,
            "gain_to_target_db": None if level is None else _finite(target_lufs - level),
            "gain_ceiling_from_peak_db": None if peak is None else _finite(true_peak_dbtp - peak),
        }
    n = len(x) // FRAME
    frames = x[:n * FRAME].reshape(n, FRAME)
    db = 20 * np.log10(np.sqrt(np.mean(frames * frames, axis=1)) + 1e-12)
    floor = float(noise_floor(db))
    quiet = db <= floor + VOICED_ABOVE_FLOOR_DB
    quiet_db = db[quiet].astype(np.float64)
    return {
        **base,
        "status": "measured",
        "floor_dbfs": _finite(floor, 1),
        "quiet_frame_share": _finite(quiet.mean(), 3),
        "quiet_level_spread_db": _finite(np.percentile(quiet_db, 90) - np.percentile(quiet_db, 10), 1),
        "speech_p90_dbfs": _finite(np.percentile(db, 90), 1),
        "mono_sample_peak_dbfs": _finite(20 * np.log10(mono_peak), 2),
        **loud,
        "gain_to_target_db": None if level is None else _finite(target_lufs - level),
        "gain_ceiling_from_peak_db": None if peak is None else _finite(true_peak_dbtp - peak),
        "quiet_spectrum": _quiet_spectrum(x, quiet),
    }


def assess(source: str, judge: Judge, cache, *, target_lufs: float, true_peak_dbtp: float,
           guidelines: str, preferences: str) -> dict:
    """Measure the source, then ask the configured judge once for its reading. No fallback judge is used.

    The prompt carries the owner's snapshotted preferences and a fingerprint of the editing guidelines (the judge's
    system prompt carries the guidelines themselves), so a sample cached for one set of instructions never answers
    for another.
    """
    facts = measure(source, target_lufs=target_lufs, true_peak_dbtp=true_peak_dbtp)
    prompt = prompts.with_preferences(
        prompts.audio_assessment_prompt(facts, guidelines_fingerprint(guidelines)), preferences)
    result = vote(judge, prompt, prompts.AUDIO_ASSESSMENT_SCHEMA, k=1, cache=cache)
    return {"judge": judge.name, "facts": facts, "assessment": result.responses[0]}
