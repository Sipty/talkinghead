"""Voiced-burst pass: hums, moans, and hesitation vowels that ASR cannot see.

Two systemic failures motivate this pass; in both the transcript itself hides
the event, so text-driven passes (filler, restart, silence) never see it:

  1. Absorption: whisper stamps a hum/moan INTO a word span, which comes out
     far longer than the word can plausibly take (a 1.72 s "this" spanning
     155.22-156.94). The absorbed vocalization rides along uncut,
     and detect_mouth_noises cannot catch it: a moan is voiced, so Silero
     labels the frame speech and there is no VAD gap to gate on. Parakeet
     (asr=parakeet) tokenizes these instead of absorbing them, so backend
     selection is the primary fix; this pass is the belt-and-braces for
     whisper-timed pipelines.

  2. Gap hums: whisper drops a hum into an inter-word gap it never
     transcribes, and detect_silences' RMS threshold reads the hum as
     quiet, so the pause the silence pass keeps (or the shorter-than-
     pause_above gap it never trims) still carries audible voiced energy
     (measured >= -31 dBFS on a real render).

Detection is purely acoustic - the same 5 ms dBFS+ZCR envelope as the
mouth-noise pass - with deliberately FP-averse gates: cutting a real word
is the one unacceptable outcome, so anything ambiguous is left for the
reviewer to toggle. There is NO VAD gate here, by design: Silero calls
hums/moans speech (they are voiced), so a speech-gap gate would blind the
pass to exactly its targets.

Threshold rationale lives with each constant below. The one invariant: the
word CORE is protected by construction - a cut is only proposed when the
span splits into a loud contiguous core and a clearly quieter run separated
by a real dip. A moan that abuts the word with no dip is unseparable and is
left alone.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from statistics import median

from .noise import (
    _MIN_SPAN,
    BURST_ENTER_DB,
    DEFAULT_PAD,
    FLOOR_MAX_DBFS,
    FLOOR_PERCENTILE,
    FRAME_SEC,
    _envelope,
    _percentile,
    _read_wav,
)
from .transcribe import Word, extract_wav
from .types import CutSpan

# --- absorption signature (finding 1) ---------------------------------
# A span is "abnormally long" when it exceeds SPAN_FACTOR x the median
# duration of the SPAN_NEIGHBORS words on each side - whisper inflates only
# the span it glued material into, neighbours stay word-sized.
SPAN_NEIGHBORS = 6
# Below this few neighbours the median is not evidence; skip the word.
SPAN_MIN_NEIGHBORS = 3
# 3x the neighbour median: the smallest absorption whisper produces (a glued
# filler roughly doubles the span), times a safety margin.
SPAN_FACTOR = 3.0
# Absolute floor: spans shorter than this are within normal word variance
# even for fast speech, and there is no room for a detectable burst inside.
ABSORB_MIN_DUR = 0.8

# --- burst gates (shared by both findings) -----------------------------
# Audibility: a burst must clear the noise floor by this much, or it is
# floor wobble, not a vocalization.
BURST_MIN_DB = 10.0
# Shorter than this is smack/breath territory (noise.py's job) and not worth
# a splice; longer than this inside one word span is far more likely quiet
# real speech whisper glued in, and clipping words is the one outcome to
# avoid, so it is left to the reviewer.
BURST_MIN_DUR = 0.15
BURST_MAX_DUR = 2.0
# Voicing: hums/moans are low-frequency (ZCR near 0.02 at 120 Hz); smacks
# sit at ~1.0 and fricatives above 0.3. 0.25 matches noise.py's breath
# ceiling - a "voiced burst" that clicks like a smack is not this pass's
# finding.
VOICED_MAX_ZCR = 0.25
# The absorbed material must sit this far under the span's peak (the word
# core). A vocalization nearly as loud as the speech it is glued to cannot
# be told apart from the word itself, so it is skipped.
CORE_CONTRAST_DB = 5.0
# The core and the burst must be separated by at least this much envelope
# dip (frames back under floor+BURST_ENTER_DB); less means one contiguous
# sound, which cannot be cut without clipping the word.
SPLIT_MIN_DUR = 0.03

# --- gap hums (finding 2) ---------------------------------------------
# Inter-word gaps shorter than this are inside normal co-articulation; the
# hum has no room to be audible.
GAP_HUM_MIN_DUR = 0.25
# Without a detect_silences witness, a gap burst may not exceed floor+this:
# real speech cores sit tens of dB over the floor (42 dB in the calibration
# fixtures), a hum under the silence threshold stays around +10..+20.
GAP_HUM_MAX_DB = 20.0
# Slack for "the hum sits inside a detect_silences range": the two grid their
# frames differently (5 ms vs 20 ms), so edges wobble a little.
SILENCE_SLACK = 0.05


def _runs(db: list[float], start: int, end: int, threshold: float) -> list[tuple[int, int]]:
    """Inclusive frame ranges within [start, end] where db stays above threshold."""
    out: list[tuple[int, int]] = []
    i = start
    while i <= end:
        if db[i] <= threshold:
            i += 1
            continue
        j = i
        while j < end and db[j + 1] > threshold:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def _core_range(db: list[float], f0: int, f1: int) -> tuple[int, int]:
    """Contiguous frame range of the word core: frames within CORE_CONTRAST_DB
    of the span's peak, grown outward from the peak frame."""
    peak = f0 + max(range(f1 - f0 + 1), key=lambda i: db[f0 + i])
    target = db[peak] - CORE_CONTRAST_DB
    cs, ce = peak, peak
    while cs > f0 and db[cs - 1] >= target:
        cs -= 1
    while ce < f1 and db[ce + 1] >= target:
        ce += 1
    return cs, ce


def _absorbed_spans(words: list[Word]) -> list[tuple[Word, float]]:
    """Words whose span is abnormally long for their neighbourhood.

    Returns (word, neighbour median duration) pairs; the median is the
    expected word length the span is judged against.
    """
    out: list[tuple[Word, float]] = []
    for i, w in enumerate(words):
        dur = w.end - w.start
        if dur <= ABSORB_MIN_DUR:
            continue
        lo, hi = max(0, i - SPAN_NEIGHBORS), min(len(words), i + 1 + SPAN_NEIGHBORS)
        neighbors = [words[j].end - words[j].start for j in range(lo, hi) if j != i]
        if len(neighbors) < SPAN_MIN_NEIGHBORS:
            continue
        med = median(neighbors)
        if med > 0 and dur > SPAN_FACTOR * med:
            out.append((w, med))
    return out


def _burst_frames(
    db: list[float], floor: float, frame_sec: float, f0: int, f1: int, core: tuple[int, int]
) -> list[tuple[int, int]]:
    """Burst frame ranges inside [f0, f1]: runs over floor+BURST_ENTER_DB that
    do not touch the word core and are separated from it by a real dip.

    The dip requirement (>= SPLIT_MIN_DUR of frames back under
    floor+BURST_ENTER_DB) is what protects the word: a run that abuts the
    core is one contiguous sound and cannot be cut without clipping it.
    """
    out: list[tuple[int, int]] = []
    for rs, re in _runs(db, f0, f1, floor + BURST_ENTER_DB):
        if not (re < core[0] or rs > core[1]):
            continue  # overlaps the word core: part of the word itself
        gap_frames = rs - core[1] - 1 if rs > core[1] else core[0] - re - 1
        if gap_frames * frame_sec < SPLIT_MIN_DUR:
            continue
        out.append((rs, re))
    return out


def detect_voiced_bursts(
    audio: str | Path | Sequence[int],
    words: list[Word],
    *,
    sample_rate: int = 16000,
    frame_sec: float = FRAME_SEC,
    pad: float = DEFAULT_PAD,
    silences: Sequence[tuple[float, float]] | None = None,
    envelope: tuple[Sequence[float], Sequence[float]] | None = None,
) -> list[CutSpan]:
    """Cut spans for hums/moans/hesitation vowels whisper cannot see.

    envelope: optional precomputed (db, zcr) per frame_sec frame (same numbers
        as _envelope); lets a caller share one envelope across detectors.

    audio: path to a 16 kHz mono PCM WAV (see transcribe.extract_wav), or a
        raw 16-bit sample sequence, in which case sample_rate applies.
    words: ASR word timings. Their SPANS are evidence here (unlike
        detect_mouth_noises, which never trusts them) because the absorption
        signature IS a span being too long; the audio inside decides what is
        word and what is burst.
    silences: detect_silences() ranges for the same audio. Gap-hum findings
        require the hum to sit inside one of them (detect_silences calls it
        quiet); without a witness the fallback gate is peak <=
        floor+GAP_HUM_MAX_DB.

    Returns CutSpans with reason "noise" (mechanical: measured off the
    envelope, not judged), sorted by start time.
    """
    if envelope is not None:
        db, zcr = list(envelope[0]), list(envelope[1])
    else:
        if isinstance(audio, (str, Path)):
            samples, sample_rate = _read_wav(str(audio))
        else:
            samples = audio
        db, zcr = _envelope(samples, sample_rate, frame_sec)
    if not db or not words:
        return []
    floor = _percentile(db, FLOOR_PERCENTILE)
    if floor > FLOOR_MAX_DBFS:
        raise RuntimeError(
            f"implausible noise floor {floor:.1f} dBFS (limit {FLOOR_MAX_DBFS:.0f}) over "
            f"{len(db)} envelope frames: the audio looks like the wrong or a "
            "clipped/silent source; refusing to classify bursts against it"
        )

    cuts: list[CutSpan] = []
    n = len(db) - 1
    # Finding 1: bursts absorbed into abnormally long word spans.
    for w, _med in _absorbed_spans(words):
        f0, f1 = max(0, int(w.start / frame_sec)), min(n, math.ceil(w.end / frame_sec))
        if f1 < f0:
            continue
        core = _core_range(db, f0, f1)
        span_peak = max(db[f0 : f1 + 1])
        for rs, re in _burst_frames(db, floor, frame_sec, f0, f1, core):
            t0 = max(rs * frame_sec, w.start)
            t1 = min((re + 1) * frame_sec, w.end)
            peak = max(db[rs : re + 1])
            dur = t1 - t0
            if not (BURST_MIN_DUR <= dur <= BURST_MAX_DUR):
                continue
            if peak < floor + BURST_MIN_DB or peak > span_peak - CORE_CONTRAST_DB:
                continue
            if sum(zcr[rs : re + 1]) / (re - rs + 1) > VOICED_MAX_ZCR:
                continue
            cut_start = max(t0 - pad, w.start)
            cut_end = min(t1 + pad, w.end)
            if cut_end - cut_start > _MIN_SPAN:
                cuts.append(
                    CutSpan(cut_start, cut_end, "noise", f"voiced burst: absorbed in {w.text!r}")
                )

    # Finding 2: hums inside inter-word gaps that detect_silences calls quiet.
    for a, b in zip(words, words[1:]):
        g0, g1 = a.end, b.start
        if g1 - g0 < GAP_HUM_MIN_DUR:
            continue
        gf0, gf1 = max(0, int(g0 / frame_sec)), min(n, math.ceil(g1 / frame_sec))
        if gf1 < gf0:
            continue
        for rs, re in _runs(db, gf0, gf1, floor + BURST_ENTER_DB):
            t0 = max(rs * frame_sec, g0)
            t1 = min((re + 1) * frame_sec, g1)
            peak = max(db[rs : re + 1])
            dur = t1 - t0
            if not (BURST_MIN_DUR <= dur <= BURST_MAX_DUR):
                continue
            if peak < floor + BURST_MIN_DB:
                continue
            if silences is None:
                if peak > floor + GAP_HUM_MAX_DB:
                    continue
            elif not any(s - SILENCE_SLACK <= t0 and t1 <= e + SILENCE_SLACK for s, e in silences):
                continue  # detect_silences calls this speech, not quiet
            if sum(zcr[rs : re + 1]) / (re - rs + 1) > VOICED_MAX_ZCR:
                continue
            cut_start = max(t0 - pad, g0)
            cut_end = min(t1 + pad, g1)
            if cut_end - cut_start > _MIN_SPAN:
                cuts.append(CutSpan(cut_start, cut_end, "noise", "voiced burst: hum between words"))

    cuts.sort(key=lambda c: c.start)
    return cuts


def detect_voiced_bursts_in_video(video_path: str, words: list[Word], **kwargs) -> list[CutSpan]:
    """detect_voiced_bursts for a video file: extract a temp WAV, then clean up."""
    wav = extract_wav(video_path)
    try:
        return detect_voiced_bursts(wav, words, **kwargs)
    finally:
        Path(wav).unlink(missing_ok=True)