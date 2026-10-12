"""Align a separately recorded audio track onto the video master.

A port of the standalone audiolign tool (its align, extract and render
modules); the numerics are unchanged, so an offset measured here matches one
measured there.

`offset_sec` is the time on the VIDEO timeline where t=0 of the AUDIO
recording lands (A_time = B_time + offset): positive means the external
recording joined late and its head is padded with silence, negative means it
joined early and its head is trimmed.

The video stream is COPIED untouched when the master carries no display
rotation: no generation loss, and HDR / Dolby Vision metadata survives. Only a
display matrix (iPhone portrait sensors) forces a re-encode, and then the
rotation is baked into the pixels (hardware HEVC main10 for 10-bit HDR with
the colour tags preserved, libx264 for SDR), producing physically upright 16:9
output with no rotation metadata, whatever the reader supports.

Two stages, both cross-correlating the two audio tracks:
1. Coarse: non-overlapping 30 ms RMS envelopes, full cross-correlation.
2. Fine: GCC-PHAT on 30 s windows spread across the timeline. Windows that are
   silent or ambiguous are skipped; the survivors are median-filtered and
   fitted with a line whose slope is the clock drift.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .color import hdr_codec, hdr_pix_fmt
from .transcribe import prepare_audio

# 16 kHz mono: plenty for speech envelopes and GCC-PHAT (sub-10 ms precision)
# and cheap enough to hold a 47-minute pair in memory. Matches prepare_audio.
ANALYSIS_SR = 16_000
# Envelope frame: 30 ms non-overlapping -> 480 samples at 16 kHz.
ENVELOPE_FRAME = 480
# Fine window length and how many are spread across the timeline.
FINE_WINDOW_S = 30.0
FINE_WINDOWS = 7
# Fine-window search margin around the coarse estimate (covers coarse error + drift).
WINDOW_MARGIN_S = 3.0
# Fine-window matches must beat the runner-up peak by at least this factor.
PEAK_RATIO_MIN = 3.0
# Windows that disagree with the median lag by more than this get dropped as outliers.
OUTLIER_MS = 50.0
# Re-encode bitrate for the external audio; 256k AAC is transparent for stereo voice.
AUDIO_BITRATE = "256k"
# Video re-encode bitrate when rotation must be baked in. iPhone masters run
# ~8.5 Mbps; 12M gives the hardware HEVC encoder headroom to avoid artifacts.
VIDEO_BITRATE = "12M"


class AlignError(RuntimeError):
    """Alignment or remux failed. Callers decide whether to continue."""


@dataclass(frozen=True)
class AlignResult:
    """Where the external recording sits on the video timeline, plus drift.

    `confidence` is the coarse match prominence (peak vs the best alternative
    1 s away); below ~3x the match is weak and sync should be verified by ear.
    """

    offset_sec: float
    drift_ppm: float
    confidence: float | None
    output: Path | None = None
    coarse_offset_sec: float = 0.0
    windows_used: int = 0
    windows_total: int = 0
    video_duration: float = 0.0
    audio_duration: float = 0.0
    # Stream-level video facts the remux needs (see _probe_video); not part of
    # the measured result, so it stays out of repr and equality.
    video_info: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def rotation(self) -> float:
        """Display rotation in degrees (-90/0/90/180) stored as metadata."""
        return float(self.video_info.get("rotation", 0.0) or 0.0)

    def to_json(self) -> dict:
        return {
            "offset_sec": round(self.offset_sec, 6),
            "drift_ppm": round(self.drift_ppm, 3),
            "confidence": None if self.confidence is None else round(self.confidence, 3),
            "output": None if self.output is None else str(self.output),
            "coarse_offset_sec": round(self.coarse_offset_sec, 6),
            "windows_used": self.windows_used,
            "windows_total": self.windows_total,
            "video_duration": round(self.video_duration, 3),
            "audio_duration": round(self.audio_duration, 3),
            "rotation": self.rotation,
        }

    def summary(self) -> str:
        """One human line: offset, drift, confidence."""
        confidence = "n/a" if self.confidence is None else f"{self.confidence:.2f}x above alternatives"
        return f"offset {self.offset_sec:+.3f} s, drift {self.drift_ppm:+.1f} ppm, confidence {confidence}"


# --------------------------------------------------------------- measurement


@dataclass(frozen=True)
class _Measurement:
    offset: float          # s on the video timeline where the audio t=0 lands
    drift: float           # s per s; the external clock's rate relative to the video's
    coarse_offset: float
    windows_used: int
    windows_total: int
    confidence: float | None


def _rms_envelope(x: np.ndarray) -> np.ndarray:
    """RMS per non-overlapping 30 ms frame; a partial tail frame is dropped."""
    n = (x.size // ENVELOPE_FRAME) * ENVELOPE_FRAME
    frames = x[:n].reshape(-1, ENVELOPE_FRAME)
    return np.sqrt(np.mean(frames * frames, axis=1))


def _cross_correlate_lag(a: np.ndarray, b: np.ndarray) -> tuple[int, float, float]:
    """Full linear cross-correlation of a vs b via FFT.

    Returns (lag, peak, prominence) where b[t] aligns with a[t + lag].
    Prominence = peak / second-best-peak outside a +-1 s neighbourhood: how
    much the match stands out from plausible alternatives. irfft(rfft(a) *
    conj(rfft(b))) puts lag L at index L for L >= 0 and at nfft + L for L < 0,
    so negative lags are recovered explicitly instead of truncated away.
    """
    a = a - a.mean()
    b = b - b.mean()
    nfft = 1 << (a.size + b.size - 2).bit_length()
    full = np.fft.irfft(np.fft.rfft(a, nfft) * np.conj(np.fft.rfft(b, nfft)), nfft)
    # Negative lags -(b.size-1)..-1, then non-negative lags 0..a.size-1.
    corr = np.concatenate([full[nfft - (b.size - 1):], full[: a.size]])
    k = int(np.argmax(corr))
    lag = k - (b.size - 1)
    peak = float(corr[k])
    others = corr.copy()
    blank = int(1.0 * ANALYSIS_SR / ENVELOPE_FRAME)
    others[max(0, k - blank): min(corr.size, k + blank + 1)] = -np.inf
    second = float(np.max(others))
    return lag, peak, peak / max(abs(second), 1e-12)


def _gcc_phat_lag(a_seg: np.ndarray, b_seg: np.ndarray) -> tuple[int, float, float]:
    """GCC-PHAT of b_seg against a_seg; returns (lag, peak, peak/runner-up).

    a_seg must already include the search margin. The phase transform whitens
    the cross-spectrum so broadband speech peaks sharply even when the two
    microphones have very different timbres. Index k < len(a_seg) maps to lag k.
    """
    n = a_seg.size + b_seg.size - 1
    nfft = 1 << (n - 1).bit_length()
    spec = np.fft.rfft(a_seg, nfft) * np.conj(np.fft.rfft(b_seg, nfft))
    spec /= np.abs(spec) + 1e-12
    corr = np.fft.irfft(spec, nfft)[: a_seg.size]
    k = int(np.argmax(corr))
    peak = float(corr[k])
    others = corr.copy()
    # Blank +-1 s around the winner so the runner-up is a distinct alternative.
    others[max(0, k - ANALYSIS_SR): k + ANALYSIS_SR + 1] = -np.inf
    second = float(np.max(others))
    return k, peak, peak / max(abs(second), 1e-12)


def _window_lag(a: np.ndarray, b: np.ndarray, b_start: int, window_samples: int,
                expected_lag_samples: float) -> tuple[float, float] | None:
    """Lag from one window of b matched into a near the expected lag.

    Returns (lag_samples, b_center_seconds), or None when the window is too
    quiet or the match too weak/ambiguous to trust.
    """
    b_seg = b[b_start: b_start + window_samples]
    if b_seg.size < window_samples // 2:
        return None
    # Skip silent windows: nothing to correlate, GCC-PHAT peaks are noise.
    global_rms = np.sqrt(np.mean(b * b) + 1e-12)
    if np.sqrt(np.mean(b_seg * b_seg)) < 0.15 * global_rms:
        return None
    exp = int(round(expected_lag_samples))
    # The match sits at a-index b_start + lag, so the search window shifts with b_start.
    lo = b_start + exp - int(WINDOW_MARGIN_S * ANALYSIS_SR)
    hi = b_start + exp + int(WINDOW_MARGIN_S * ANALYSIS_SR) + b_seg.size
    lo_c, hi_c = max(0, lo), min(a.size, hi)
    a_seg = a[lo_c:hi_c]
    if a_seg.size < b_seg.size:
        return None
    k, peak, ratio = _gcc_phat_lag(a_seg, b_seg)
    if peak < 0.01 or ratio < PEAK_RATIO_MIN:
        return None
    # b_seg[m] matches a_seg[m+k]; convert to an absolute a-time minus b-time lag.
    lag = (lo_c + k) - b_start
    return float(lag), (b_start + b_seg.size / 2) / ANALYSIS_SR


def _drop_outliers(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Drop windows whose lag disagrees with the median by more than OUTLIER_MS."""
    if len(points) < 3:
        return points
    lags = np.array([p[0] for p in points])
    med = np.median(lags)
    keep = np.abs(lags - med) < OUTLIER_MS / 1000 * ANALYSIS_SR
    return [p for p, k in zip(points, keep) if k]


def _measure_arrays(a: np.ndarray, b: np.ndarray) -> _Measurement:
    """Alignment of recording b (external audio) onto recording a (video master)."""
    # Stage 1: coarse envelope correlation over the full timelines.
    ea, eb = _rms_envelope(a), _rms_envelope(b)
    lag_frames, peak, prominence = _cross_correlate_lag(ea, eb)
    coarse = lag_frames * ENVELOPE_FRAME / ANALYSIS_SR

    # Stage 2: fine GCC-PHAT windows spread across b (the shorter recording).
    window_samples = int(FINE_WINDOW_S * ANALYSIS_SR)
    usable = max(0, b.size - window_samples)
    starts = [int(usable * frac) for frac in np.linspace(0.02, 0.98, FINE_WINDOWS)]
    points: list[tuple[float, float]] = []
    for start in starts:
        got = _window_lag(a, b, start, window_samples, coarse * ANALYSIS_SR)
        if got is not None:
            points.append(got)

    points = _drop_outliers(points)
    if len(points) >= 2:
        # lag(t) = lag0 + drift * t -> slope is the clock drift, intercept the offset.
        t = np.array([p[1] for p in points])
        lag = np.array([p[0] for p in points]) / ANALYSIS_SR
        if np.ptp(t) < 1.0:
            # Windows clustered in time: the slope would be rank-deficient noise.
            offset, drift = float(np.mean(lag)), 0.0
        else:
            slope, intercept = np.polyfit(t, lag, 1)
            offset, drift = float(intercept), float(slope)
    elif points:
        offset, drift = points[0][0] / ANALYSIS_SR, 0.0
    else:
        # All fine windows failed; the coarse estimate is all we have.
        offset, drift = coarse, 0.0

    # A non-positive correlation peak means the envelopes share no positive
    # overlap anywhere (silence, or inverted content): the offset above is then
    # meaningless, which is what a None confidence reports.
    confidence = None if peak <= 0 else prominence
    return _Measurement(offset, drift, coarse, len(points), FINE_WINDOWS, confidence)


# ------------------------------------------------------------------ probing


def _run(cmd: list[str], what: str) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise AlignError(f"{cmd[0]} not found; install ffmpeg (e.g. brew install ffmpeg)") from e
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-15:])
        raise AlignError(f"{what} failed (exit {proc.returncode}):\n{tail}")
    return proc.stdout


def _require(path: Path, what: str) -> None:
    if not path.is_file():
        raise AlignError(f"{what} not found: {path}")


def _probe_duration(path: Path) -> float:
    """Container duration in seconds via ffprobe."""
    out = _run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        f"ffprobe duration of {path}",
    ).strip()
    try:
        return float(out)
    except ValueError as e:
        raise AlignError(f"ffprobe returned no duration for {path}: {out!r}") from e


def _probe_video(path: Path) -> dict:
    """Video stream facts the remux needs: rotation, pixel format, colour tags."""
    out = _run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=pix_fmt,color_primaries,color_transfer,colorspace:stream_side_data=rotation",
         "-of", "json", str(path)],
        f"ffprobe video stream of {path}",
    )
    streams = (json.loads(out) or {}).get("streams") or [{}]
    stream = streams[0]
    rotation = 0.0
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            rotation = float(sd["rotation"])
    return {
        "rotation": rotation,
        "pix_fmt": stream.get("pix_fmt", ""),
        "color_primaries": stream.get("color_primaries", ""),
        "color_transfer": stream.get("color_transfer", ""),
        "colorspace": stream.get("colorspace", ""),
    }


def _read_wav(wav: str | Path) -> np.ndarray:
    """16-bit PCM wav to mono float64; channel 0 if it turns out multi-channel."""
    with wave.open(str(wav), "rb") as w:
        raw, channels = w.readframes(w.getnframes()), w.getnchannels()
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    if channels > 1:
        x = x.reshape(-1, channels)[:, 0]
    if x.size == 0:
        raise AlignError(f"no audio decoded from {wav}")
    return x


def _samples(path: Path, workdir: Path) -> np.ndarray:
    """Decode a file's first audio stream to mono 16 kHz float64 samples."""
    try:
        wav, _sha = prepare_audio(str(path), workdir)
    except subprocess.CalledProcessError as e:
        raise AlignError(f"could not extract audio from {path}: {(e.stderr or '').strip()}") from e
    return _read_wav(wav)


# ---------------------------------------------------------------- rendering


def _transpose_chain(rotation: float) -> str:
    """Filter that undoes the display rotation, baking it into the pixels.

    A display rotation of r means a player rotates the content r degrees
    counter-clockwise; we pre-apply the opposite. Verified against a real
    iPhone master (rotation -90 -> transpose=1 clockwise, upright 16:9).
    """
    r = rotation % 360
    if r == 270:
        return "transpose=1"
    if r == 90:
        return "transpose=2"
    if r == 180:
        return "hflip,vflip"
    raise AlignError(f"unsupported display rotation: {rotation}")


def _filter_complex(result: AlignResult, fix_drift: bool) -> str:
    """Filter graph placing video and external audio on the output timeline.

    offset > 0: the external recording started late -> adelay pads its head.
    offset < 0: it started early -> atrim drops its head.
    With fix_drift the audio is time-stretched by 1/(1+drift) so its clock
    matches the video's before the delay is applied.
    apad (+ -t video_duration) covers the tail when the audio runs out first.
    """
    rotate = result.rotation % 360 != 0
    parts = []
    if rotate:
        parts.append(f"[0:v:0]{_transpose_chain(result.rotation)}[v]")
    audio = ["[1:a]aresample=48000"]
    drift = result.drift_ppm / 1e6
    if result.offset_sec < 0:
        # Trimming the head lands the audio ~21 ms (one AAC frame at 48 kHz)
        # late; inherited from audiolign, identical output measured on both.
        # `atrim` keeps the original timestamps, so without `asetpts` the
        # track starts `offset` seconds into the file: a hole with no audio
        # under the head of the picture, and a track shorter than the video.
        audio.append(f"atrim=start={-result.offset_sec:.6f},asetpts=PTS-STARTPTS")
    if fix_drift and abs(drift) > 1e-9:
        # atempo stays inside its valid [0.5, 2] range; clock drift is ppm-scale.
        audio.append(f"atempo={1.0 / (1.0 + drift):.9f}")
    if result.offset_sec > 0:
        audio.append(f"adelay={int(round(result.offset_sec * 1000))}:all=1")
        # After atempo, adelay can shift timestamps instead of emitting silence. The track then
        # has a timestamp hole that a full decode (prepare_audio) closes, moving every word
        # `offset` seconds early against the picture. Fill any hole with real silent samples.
        audio.append("aresample=async=1:first_pts=0")
    audio.append("apad[a]")
    parts.append(",".join(audio))
    return ";".join(parts)


def _video_args(video_info: dict) -> list[str]:
    """Stream-copy when no rotation; hardware re-encode when baking it in."""
    pix = video_info.get("pix_fmt", "")
    args: list[str] = []
    if pix in ("yuv420p10le", "p010le"):
        # Hardware HEVC main10 keeps the HDR/HLG look of iPhone masters (libx265 off macOS).
        codec = hdr_codec()
        args += ["-c:v", codec, "-profile:v", "main10",
                 "-b:v", VIDEO_BITRATE, "-pix_fmt", hdr_pix_fmt(codec), "-tag:v", "hvc1"]
    else:
        args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
    # Propagate source colour tagging; decoding into raw frames loses it.
    for src_key, out_flag in (("color_primaries", "-color_primaries"),
                              ("color_transfer", "-color_trc"),
                              ("colorspace", "-colorspace")):
        value = video_info.get(src_key)
        if value:
            args += [out_flag, value]
    return args


def _remux_command(video: Path, audio: Path, out: Path, result: AlignResult, *,
                   fix_drift: bool) -> list[str]:
    rotate = result.rotation % 360 != 0
    cmd = ["ffmpeg", "-y", "-v", "error", "-nostdin"]
    if rotate:
        # -display_rotation 0 strips the input's rotation side data so it does
        # not ride through the filter graph into the output container; our
        # transpose filter is then the only rotation applied.
        cmd += ["-display_rotation", "0"]
    cmd += ["-i", str(video), "-i", str(audio), "-filter_complex", _filter_complex(result, fix_drift)]
    cmd += ["-map", "[v]" if rotate else "0:v:0", "-map", "[a]"]
    cmd += _video_args(result.video_info) if rotate else ["-c:v", "copy"]
    cmd += ["-c:a", "aac", "-b:a", AUDIO_BITRATE,
            "-t", f"{result.video_duration:.6f}",
            "-map_metadata", "0", "-movflags", "+faststart"]
    if str(out).endswith(".mov"):
        cmd += ["-f", "mov"]
    cmd.append(str(out))
    return cmd


# ------------------------------------------------------------------- public


def _measure(video: Path, audio: Path) -> AlignResult:
    """Probe both files and measure the offset; no rendering."""
    _require(video, "video")
    _require(audio, "audio")
    video_info = _probe_video(video)
    video_duration = _probe_duration(video)
    audio_duration = _probe_duration(audio)
    # Both decodes go to scratch wavs; the arrays are ~360 MB each at 47 minutes,
    # so they are released (locals go out of scope) before any render.
    with tempfile.TemporaryDirectory(prefix="talkinghead-align-") as tmp:
        a = _samples(video, Path(tmp) / "video")
        b = _samples(audio, Path(tmp) / "audio")
    m = _measure_arrays(a, b)
    return AlignResult(
        offset_sec=m.offset,
        drift_ppm=m.drift * 1e6,
        confidence=m.confidence,
        coarse_offset_sec=m.coarse_offset,
        windows_used=m.windows_used,
        windows_total=m.windows_total,
        video_duration=video_duration,
        audio_duration=audio_duration,
        video_info=video_info,
    )


def analyze(video: Path, audio: Path, *, fix_drift: bool = False) -> AlignResult:
    """Measure where `audio` lands on `video`'s timeline; render nothing.

    `fix_drift` is the render-time option (see `align`): the drift is always
    measured and reported, so it changes nothing here.
    """
    return _measure(Path(video), Path(audio))


def align(video: Path, audio: Path, out: Path, *, fix_drift: bool = False) -> AlignResult:
    """Measure, then remux the video's picture with the aligned `audio` into `out`."""
    video, audio, out = Path(video), Path(audio), Path(out)
    if out.resolve() in (video.resolve(), audio.resolve()):
        raise AlignError("output path must differ from both inputs")
    result = _measure(video, audio)
    _run(_remux_command(video, audio, out, result, fix_drift=fix_drift), f"ffmpeg remux to {out}")
    return replace(result, output=out)
