"""Mechanical mouth-noise pass: tongue smacks, breaths, and throat clears.

These are the events that "look like jumps in the audiograph" and get edited by
hand today. Detection is purely acoustic - no ASR judgment, no LLM - so cuts
land in MECHANICAL_REASONS with reason "noise".

The pass is VAD-gated: Silero VAD (ONNX, CPU) labels every 32 ms frame
speech/not-speech, and a candidate is an energy burst that sits INSIDE a
non-speech gap. Bursts are then LABELLED by the dB+ZCR classifier calibrated
against hand-located ground truth in docs/noise-detector-spec.md (16 kHz mono,
5 ms frames, dBFS envelope + ZCR).

Non-obvious decisions:
  - VAD is the gate, the envelope is the evidence. A speech model answers
    "is anyone talking here?" far better than a threshold does, but its frame
    grid is coarse (32 ms) and its edges are soft, so cut EDGES still come
    from the 5 ms envelope. When silero-vad is not installed the pass falls
    back to the envelope alone and says so on stderr.
  - The noise floor is the 25th percentile of the WHOLE file. A 1.5 s sliding
    floor tracked the bursts themselves and over-segmented; one adaptive
    per-file floor matched the ground truth.
  - Word spans are never trusted as gap evidence (spec safety rule 1). ASR
    spans are generous and sometimes swallow a throat clear whole, so the gap
    around a burst is measured on the envelope; words only bound how far a cut
    may grow toward the next onset.
  - A burst with less than MIN_QUIET of true quiet on either side is LEFT
    ALONE (spec safety rule 2): a clipped word is worse than a small smack.
"""

from __future__ import annotations

import math
import sys
import wave
from array import array
from collections.abc import Sequence
from pathlib import Path

from .transcribe import Word, extract_wav
from .types import CutSpan

FRAME_SEC = 0.005
# Implausible-floor guard: a talking-head recording always has non-speech
# (pauses, breaths), so the 25th-percentile floor must sit well below speech
# level. A floor near full scale means the envelope was computed on the wrong
# or a truncated/clipped file (past bug: a pass measured the floor on the
# golden master instead of its input, silently under-detecting everything) --
# fail loudly instead of producing garbage cuts.
FLOOR_MAX_DBFS = -6.0
FLOOR_PERCENTILE = 0.25
# A burst opens on frames this far over the floor and closes when energy falls
# back to within BURST_EXIT_DB of it (spec: "above floor+8, ending below +2").
BURST_ENTER_DB = 8.0
BURST_EXIT_DB = 2.0

SMACK_MAX_DUR = 0.2
SMACK_MIN_ZCR = 0.4
BREATH_MIN_DUR = 0.15
BREATH_MAX_DUR = 0.6
BREATH_MAX_ZCR = 0.22
BREATH_MAX_PEAK_DB = 16.0
# A throat clear is "like breath but stronger/longer" (spec), so it gets a
# wider window than BREATH - but a bounded one. Both bounds are deliberately
# tight: a low-ZCR burst that is longer than THROAT_MAX_DUR or louder than
# THROAT_MAX_PEAK_DB over the floor is far more likely a quietly spoken phrase
# than a throat clear, and clipping a word is the one outcome worth avoiding.
THROAT_MAX_DUR = 0.8
THROAT_MAX_PEAK_DB = 24.0

# Safety rule 1: the gap around a burst, measured on the envelope. Energy
# before the burst must have dropped below floor+GAP_BEFORE_DB, and after it
# must stay below floor+GAP_AFTER_DB up to the next word onset.
GAP_BEFORE_DB = 4.0
GAP_AFTER_DB = 8.0
# Safety rule 2: less quiet than this on either side means embedded in speech.
MIN_QUIET = 0.15

# Silero runs on fixed 512-sample chunks at 16 kHz (256 at 8 kHz) - both are
# 32 ms, so the mask frame rate is the same either way.
VAD_CHUNK_16K = 512
VAD_FRAME_SEC = VAD_CHUNK_16K / 16000
VAD_SAMPLE_RATES = (8000, 16000)
# Frames at or above this speech probability are speech; the rest are gap.
VAD_SPEECH_PROB = 0.5
# A burst may poke this far out of its gap and still count as inside it: gap
# edges are quantized to 32 ms, and a smack's quiet rising/falling edge often
# bleeds into the VAD frame that already contains the neighbouring word.
GAP_EDGE_SLACK = VAD_FRAME_SEC

_NO_VAD_NOTE = (
    "talkinghead: silero-vad unavailable; mouth-noise pass falling back to the "
    "envelope-only gap path (install with: uv sync --extra vad)"
)

DEFAULT_PAD = 0.02
# Anything shorter than this is not worth a splice (mirrors derive._MIN_SPAN).
_MIN_SPAN = 0.02
# dB value used for a digitally silent frame, so log10(0) never happens.
_SILENT_DB = -120.0
# Full-scale reference for 16-bit PCM, so dB reads as dBFS like the spec table.
_FULL_SCALE = 32768.0


def _read_wav(wav_path: str) -> tuple[array, int]:
    """Read a 16-bit PCM WAV into (samples, sample_rate); channel 0 if stereo."""
    with wave.open(wav_path, "rb") as w:
        if w.getsampwidth() != 2:
            raise RuntimeError(f"{wav_path}: expected 16-bit PCM, got {w.getsampwidth() * 8}-bit")
        channels = w.getnchannels()
        sr = w.getframerate()
        data = w.readframes(w.getnframes())
    samples = array("h")
    samples.frombytes(data)
    if channels > 1:
        samples = samples[::channels]
    return samples, sr


def _envelope(samples: Sequence[int], sample_rate: int, frame_sec: float) -> tuple[list[float], list[float]]:
    """Per-frame (dBFS, zero-crossing rate) over fixed-length frames."""
    frame = max(1, int(frame_sec * sample_rate))
    nframes = len(samples) // frame
    db: list[float] = []
    zcr: list[float] = []
    for i in range(nframes):
        chunk = samples[i * frame : (i + 1) * frame]
        acc = 0.0
        crossings = 0
        prev = chunk[0]
        for s in chunk:
            acc += float(s) * float(s)
            # Sign flips, counting zero as continuing the previous sign.
            if (s > 0) != (prev > 0) and s != 0 and prev != 0:
                crossings += 1
            prev = s if s != 0 else prev
        rms = (acc / len(chunk)) ** 0.5
        db.append(20 * math.log10(rms / _FULL_SCALE) if rms > 0 else _SILENT_DB)
        zcr.append(crossings / max(1, len(chunk) - 1))
    return db, zcr


def _percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..1) of a non-empty list."""
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(len(ordered) * q)))
    return ordered[idx]


def silero_speech_mask(samples: Sequence[int], sample_rate: int) -> list[bool] | None:
    """Per-VAD-frame speech flags from Silero VAD, or None when unavailable.

    Returns one bool per VAD_FRAME_SEC of audio. None means "no speech model
    here" - an uninstalled silero-vad/onnxruntime, an unsupported sample rate,
    or a model that failed to load - and the caller falls back to the
    envelope-only gap path. A VAD failure must degrade the pass, never break
    the pipeline, which is why model loading catches broadly.
    """
    if sample_rate not in VAD_SAMPLE_RATES:
        return None
    try:
        import torch
        from silero_vad import load_silero_vad
    except ImportError:
        return None
    try:
        # onnx=True keeps this on the tiny CPU ONNX graph shipped in the
        # package: no download, no torch inference, ~2s for a 5-minute file.
        model = load_silero_vad(onnx=True)
    except Exception:  # noqa: BLE001 - onnxruntime raises its own exception classes; any load failure degrades
        return None
    model.reset_states()

    # array('h') exposes the buffer protocol (the real path); plain sequences
    # from tests do not, so fall back to the element-wise constructor.
    try:
        audio = torch.frombuffer(samples, dtype=torch.int16).to(torch.float32)
    except TypeError:
        audio = torch.tensor(list(samples), dtype=torch.float32)
    audio = audio / _FULL_SCALE

    chunk = VAD_CHUNK_16K if sample_rate == 16000 else VAD_CHUNK_16K // 2
    return [
        float(model(audio[i : i + chunk], sample_rate)) >= VAD_SPEECH_PROB
        for i in range(0, len(audio) - chunk + 1, chunk)
    ]


def _mask_gaps(speech_mask: Sequence[bool], frame_sec: float) -> list[tuple[float, float]]:
    """Time spans of consecutive non-speech frames in a VAD mask."""
    gaps: list[tuple[float, float]] = []
    start: int | None = None
    for i, speaking in enumerate(speech_mask):
        if speaking:
            if start is not None:
                gaps.append((start * frame_sec, i * frame_sec))
                start = None
        elif start is None:
            start = i
    if start is not None:
        gaps.append((start * frame_sec, len(speech_mask) * frame_sec))
    return gaps


def _in_gap(gaps: list[tuple[float, float]], start: float, end: float, slack: float) -> bool:
    """True when [start, end] fits inside one gap, allowing slack at each edge."""
    return any(s - slack <= start and end <= e + slack for s, e in gaps)


def _bursts(db: list[float], floor: float) -> list[tuple[int, int]]:
    """Inclusive frame ranges of energy bursts over the noise floor.

    A burst is seeded by a frame above floor+BURST_ENTER_DB and then grown in
    both directions while frames stay above floor+BURST_EXIT_DB, so the quiet
    rising/falling edges of a smack belong to it rather than to the gap.
    """
    enter, leave = floor + BURST_ENTER_DB, floor + BURST_EXIT_DB
    out: list[tuple[int, int]] = []
    i, n, prev_end = 0, len(db), -1
    while i < n:
        if db[i] <= enter:
            i += 1
            continue
        start = i
        while start > prev_end + 1 and db[start - 1] > leave:
            start -= 1
        end = i
        while end + 1 < n and db[end + 1] > leave:
            end += 1
        out.append((start, end))
        prev_end = end
        i = end + 1
    return out


def _classify(duration: float, mean_zcr: float, peak_over_floor: float) -> str | None:
    """Noise kind for a burst, or None when it is not mouth noise.

    Thresholds come straight from the calibrated table in the spec. Real
    speech falls out here: it is either too long for a smack, too broadband
    for a breath, or (the common case) far louder than BREATH_MAX_PEAK_DB.
    """
    if duration <= SMACK_MAX_DUR and mean_zcr > SMACK_MIN_ZCR:
        return "smack"
    if BREATH_MIN_DUR <= duration <= BREATH_MAX_DUR and mean_zcr < BREATH_MAX_ZCR and peak_over_floor < BREATH_MAX_PEAK_DB:
        return "breath"
    if BREATH_MIN_DUR <= duration <= THROAT_MAX_DUR and mean_zcr < BREATH_MAX_ZCR and peak_over_floor < THROAT_MAX_PEAK_DB:
        return "throat"
    return None


def _quiet_before(db: list[float], start: int, threshold: float) -> int:
    """First frame of the quiet run ending just before `start`."""
    i = start
    while i > 0 and db[i - 1] < threshold:
        i -= 1
    return i


def _quiet_after(db: list[float], end: int, threshold: float) -> int:
    """Last frame of the quiet run beginning just after `end` (inclusive)."""
    i = end
    while i + 1 < len(db) and db[i + 1] < threshold:
        i += 1
    return i


def _word_bounds(words: list[Word], burst_start: float, burst_end: float) -> tuple[float, float]:
    """(earliest, latest) times a cut around this burst may reach.

    Only used to keep a cut off the NEXT word's onset and the PREVIOUS word's
    tail. A previous word whose span runs past the burst is ignored rather
    than believed: that is the throat-clear case, where ASR swallowed the
    noise into a word span (spec safety rule 1).
    """
    lower, upper = 0.0, math.inf
    for w in words:
        if w.end <= burst_start:
            lower = max(lower, w.end)
        elif w.start >= burst_end:
            upper = min(upper, w.start)
    return lower, upper


def detect_mouth_noises(
    audio: str | Path | Sequence[int],
    words: list[Word],
    *,
    sample_rate: int = 16000,
    frame_sec: float = FRAME_SEC,
    pad: float = DEFAULT_PAD,
    min_quiet: float = MIN_QUIET,
    speech_mask: Sequence[bool] | None = None,
    mask_frame_sec: float = VAD_FRAME_SEC,
    envelope: tuple[Sequence[float], Sequence[float]] | None = None,
) -> list[CutSpan]:
    """Cut spans for tongue smacks, breaths, and throat clears in speech gaps.

    envelope: optional precomputed (db, zcr) per frame_sec frame (same numbers
        as _envelope), shared across detectors by the pipeline.

    audio: path to a 16 kHz mono PCM WAV (see transcribe.extract_wav), or a
        raw 16-bit sample sequence, in which case sample_rate applies.
    words: ASR words, used only to bound cut edges - never as gap evidence.
    speech_mask: per-frame speech flags at mask_frame_sec resolution. Defaults
        to running Silero VAD over the audio; pass one to reuse a mask or to
        drive the gate from another VAD.

    Every returned span sits inside a VAD gap (when a mask is available) AND
    inside verified quiet: the envelope must fall below floor+GAP_BEFORE_DB
    before the burst and stay below floor+GAP_AFTER_DB after it, with at least
    min_quiet of quiet on each side.
    """
    if isinstance(audio, (str, Path)):
        samples, sample_rate = _read_wav(str(audio))
    else:
        samples = audio
    if envelope is not None:
        db, zcr = list(envelope[0]), list(envelope[1])
    else:
        db, zcr = _envelope(samples, sample_rate, frame_sec)
    if not db:
        return []
    total = len(db) * frame_sec
    floor = _percentile(db, FLOOR_PERCENTILE)
    if floor > FLOOR_MAX_DBFS:
        raise RuntimeError(
            f"implausible noise floor {floor:.1f} dBFS (limit {FLOOR_MAX_DBFS:.0f}) over "
            f"{len(db)} envelope frames: the audio looks like the wrong or a "
            "clipped/silent source; refusing to classify bursts against it"
        )
    if speech_mask is None:
        speech_mask = silero_speech_mask(samples, sample_rate)
        mask_frame_sec = VAD_FRAME_SEC
        if speech_mask is None:
            print(_NO_VAD_NOTE, file=sys.stderr)
    # None gaps = no speech model: every burst is a candidate and the envelope
    # safety rules below are the only gate.
    gaps = None if speech_mask is None else _mask_gaps(speech_mask, mask_frame_sec)

    cuts: list[CutSpan] = []
    for start, end in _bursts(db, floor):
        burst_start, burst_end = start * frame_sec, (end + 1) * frame_sec
        if gaps is not None and not _in_gap(gaps, burst_start, burst_end, GAP_EDGE_SLACK):
            continue  # VAD hears speech here; not a candidate
        frames = range(start, end + 1)
        kind = _classify(
            burst_end - burst_start,
            sum(zcr[i] for i in frames) / len(frames),
            max(db[i] for i in frames) - floor,
        )
        if kind is None:
            continue

        # Safety rules 1 and 2: real quiet on both sides, measured acoustically.
        # The VAD gap is a gate only - cut edges come from the envelope, whose
        # 5 ms grid is six times finer than the mask's.
        quiet_start = _quiet_before(db, start, floor + GAP_BEFORE_DB) * frame_sec
        quiet_end = (_quiet_after(db, end, floor + GAP_AFTER_DB) + 1) * frame_sec
        if burst_start - quiet_start < min_quiet or quiet_end - burst_end < min_quiet:
            continue

        lower, upper = _word_bounds(words, burst_start, burst_end)
        cut_start = max(burst_start - pad, quiet_start, lower)
        cut_end = min(burst_end + pad, quiet_end, upper, total)
        if cut_end - cut_start > _MIN_SPAN:
            cuts.append(CutSpan(cut_start, cut_end, "noise", f"mouth noise: {kind}"))
    return cuts


def detect_mouth_noises_in_video(video_path: str, words: list[Word], **kwargs) -> list[CutSpan]:
    """detect_mouth_noises for a video file: extract a temp WAV, then clean up."""
    wav = extract_wav(video_path)
    try:
        return detect_mouth_noises(wav, words, **kwargs)
    finally:
        Path(wav).unlink(missing_ok=True)
