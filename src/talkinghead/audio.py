"""Audio analysis primitives on 16 kHz mono PCM.

The dBFS/ZCR envelope reproduces `noise._envelope` exactly (the
mouth-noise and burst thresholds were calibrated against it) but in numpy, so
a 31-minute file takes seconds instead of minutes.

`refine_word_bounds` answers the question the owner put plainly: "when do I
actually finish a word?" ASR stamps are unreliable at the edges in BOTH
directions on this speaker (measured on the real source: a "Hello." stamped
15.12-20.80 whose voice runs 15.46-15.82; "for" stamped to end at 219.76 whose
voice runs to 220.42). So word bounds are not nudged from the stamps; they are
re-derived from the audio: the envelope is segmented into voiced bursts
separated by real quiet, each word is attached to the burst it overlaps, and
the outer words of a burst take the burst's acoustic edges. Inside a
continuous burst the stamps are kept (nothing can be cut there anyway).

`unclassified_voice` is the waveform's arbitration of what the words left
behind: voiced, speech-level spans no refined word covers. They are audio no
transcript accounts for, and the only way a cut can avoid eating them is to
measure the audio instead of reading the transcript.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .noise import FLOOR_MAX_DBFS, FLOOR_PERCENTILE, VAD_FRAME_SEC
from .transcribe import Word

FRAME_SEC = 0.005
_FULL_SCALE = 32768.0
_SILENT_DB = -120.0

# A frame is voiced (speech-level) when it sits this far above the noise floor;
# the same offset the edge sweep uses for "quiet".
VOICED_ABOVE_FLOOR_DB = 6.0
# Quiet shorter than this never splits a burst: plosive closures and unvoiced
# consonants dip to the floor for 50-120 ms inside words. Any pause worth
# cutting is far longer (pause_above is 0.35 s).
BURST_SPLIT_QUIET_SEC = 0.15
# Voiced runs shorter than this are clicks, not words.
BURST_MIN_SEC = 0.05
# A word is attached to a burst within this distance of its stamp span.
ATTACH_REACH_SEC = 0.35


@dataclass
class Envelope:
    db: np.ndarray
    zcr: np.ndarray
    frame_sec: float
    sample_rate: int

    @property
    def duration(self) -> float:
        return len(self.db) * self.frame_sec

    def frame_at(self, t: float) -> int:
        return int(t / self.frame_sec)


def envelope(samples: np.ndarray, sample_rate: int, frame_sec: float = FRAME_SEC) -> Envelope:
    """Per-frame (dBFS, zero-crossing rate); identical numbers to noise._envelope."""
    x = np.asarray(samples, dtype=np.float64)
    frame = max(1, int(frame_sec * sample_rate))
    n = len(x) // frame
    if n == 0:
        return Envelope(np.array([]), np.array([]), frame_sec, sample_rate)
    frames = x[: n * frame].reshape(n, frame)
    rms = np.sqrt((frames * frames).mean(axis=1))
    with np.errstate(divide="ignore"):
        db = np.where(rms > 0, 20.0 * np.log10(rms / _FULL_SCALE), _SILENT_DB)
    signs = np.sign(frames)
    idx = np.where(signs != 0, np.arange(frame)[None, :], 0)
    idx = np.maximum.accumulate(idx, axis=1)
    filled = np.take_along_axis(signs, idx, axis=1)
    a, b = filled[:, :-1], filled[:, 1:]
    crossings = ((a != b) & (a != 0) & (b != 0)).sum(axis=1)
    zcr = crossings / max(1, frame - 1)
    return Envelope(db, zcr, frame_sec, sample_rate)


def noise_floor(db: np.ndarray, percentile: float = FLOOR_PERCENTILE) -> float:
    """Nearest-rank percentile floor with the calibrated plausibility guard."""
    ordered = np.sort(np.asarray(db))
    idx = min(len(ordered) - 1, max(0, int(len(ordered) * percentile)))
    floor = float(ordered[idx])
    if floor > FLOOR_MAX_DBFS:
        raise RuntimeError(
            f"implausible noise floor {floor:.1f} dBFS (limit {FLOOR_MAX_DBFS:.0f}) over "
            f"{len(ordered)} envelope frames: wrong, clipped, or silent source"
        )
    return floor


def voiced_bursts(env: Envelope, floor: float) -> list[tuple[float, float]]:
    """Spans where the voice is on: frames above floor+VOICED_ABOVE_FLOOR_DB,
    bridged across quiet shorter than BURST_SPLIT_QUIET_SEC, clicks dropped."""
    voiced = env.db > floor + VOICED_ABOVE_FLOOR_DB
    fs = env.frame_sec
    runs: list[list[float]] = []
    i, n = 0, len(voiced)
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and voiced[j + 1]:
            j += 1
        start, end = i * fs, (j + 1) * fs
        if runs and start - runs[-1][1] < BURST_SPLIT_QUIET_SEC:
            runs[-1][1] = end
        else:
            runs.append([start, end])
        i = j + 1
    return [(s, e) for s, e in runs if e - s >= BURST_MIN_SEC]


@dataclass(frozen=True)
class RefinedWord:
    text: str
    start: float
    end: float
    asr_start: float
    asr_end: float
    # 0.0: no voiced burst near the stamp (ghost word, stamps kept). 0.5: inside
    # a shared burst, stamps kept (nothing can be cut there). 0.75/1.0: one/both
    # edges are the burst's acoustic edges.
    confidence: float
    # True when the voice actually stops after this word (it ends its burst, or
    # no burst was found): the only places a sentence may be split by a pause.
    burst_last: bool = True
    burst_first: bool = True


def _attach(w: Word, bursts: list[tuple[float, float]]) -> int | None:
    """Index of the burst this word belongs to: the first burst overlapping the
    stamp span by at least BURST_MIN_SEC, else the nearest burst within reach."""
    best_near, best_d = None, ATTACH_REACH_SEC
    for k, (s, e) in enumerate(bursts):
        if e < w.start - ATTACH_REACH_SEC:
            continue
        if s > w.end + ATTACH_REACH_SEC:
            break
        overlap = min(e, w.end) - max(s, w.start)
        if overlap >= BURST_MIN_SEC:
            return k
        d = max(s - w.end, w.start - e, 0.0)
        if d < best_d:
            best_near, best_d = k, d
    return best_near


def refine_word_bounds(
    words: list[Word], env: Envelope, floor: float | None = None
) -> list[RefinedWord]:
    """Snap word bounds to the voiced bursts of the audio (see module doc)."""
    if not words:
        return []
    if floor is None:
        floor = noise_floor(env.db)
    bursts = voiced_bursts(env, floor)
    attached = [_attach(w, bursts) for w in words]
    # Words are in time order; a burst's words are a contiguous group.
    out: list[RefinedWord] = []
    for i, w in enumerate(words):
        k = attached[i]
        if k is None:
            out.append(RefinedWord(w.text, w.start, w.end, w.start, w.end, 0.0, True, True))
            continue
        bs, be = bursts[k]
        first = i == 0 or attached[i - 1] != k
        last = i == len(words) - 1 or attached[i + 1] != k
        start = bs if first else min(max(w.start, bs), be)
        end = be if last else min(max(w.end, bs), be)
        if end - start < BURST_MIN_SEC:
            # The stamp lies (partly) outside its burst: keep whatever acoustic
            # edge we have and give the word a minimal span inside the burst.
            if first:
                end = min(be, max(end, start + BURST_MIN_SEC))
            elif last:
                start = max(bs, min(start, end - BURST_MIN_SEC))
            else:
                start, end = min(max(w.start, bs), be - BURST_MIN_SEC), min(be, max(w.end, w.start + BURST_MIN_SEC))
        out.append(RefinedWord(w.text, start, end, w.start, w.end, 0.5 + 0.25 * first + 0.25 * last, last, first))
    # Monotonic: a word never starts before the previous one ends, and never
    # collapses below BURST_MIN_SEC.
    fixed: list[RefinedWord] = []
    for r in out:
        if fixed and r.start < fixed[-1].end:
            start = fixed[-1].end
            r = RefinedWord(r.text, start, max(r.end, start + BURST_MIN_SEC), r.asr_start, r.asr_end,
                            r.confidence, r.burst_last, r.burst_first)
        fixed.append(r)
    return fixed


# -------------------------------------------------------- waveform arbitration

# Speech level is the 90th-percentile frame RMS, the reference detect_silences
# thresholds against. A dropped word was still spoken, so it sits within a few dB
# of that; the spread between spoken fragments and everything else on the
# reference recording was 12.1 dB, so the gate sits there.
SPEECH_PERCENTILE = 0.90
SPEECH_LEVEL_BELOW_DB = 12.0
# Shorter than this is a click or a syllable fragment, not a word worth guarding.
MIN_UNCLASSIFIED_VOICE_SEC = 0.10
# A word's acoustic edge can sit a frame or two inside the burst it was attached
# to, so coverage is tested with slack before calling a burst unclaimed.
COVER_SLACK_SEC = 0.02


def speech_level(db: np.ndarray, percentile: float = SPEECH_PERCENTILE) -> float:
    """The file's own speech reference: nearest-rank percentile of frame dBFS."""
    ordered = np.sort(np.asarray(db))
    if len(ordered) == 0:
        raise RuntimeError("no envelope frames: cannot measure a speech level")
    return float(ordered[min(len(ordered) - 1, max(0, int(len(ordered) * percentile)))])


def _covered(start: float, end: float, spans, slack: float) -> bool:
    return any(min(end, e + slack) - max(start, s - slack) > 0.0 for s, e in spans)


def _masked_out(start: float, end: float, mask: Sequence[bool], frame_sec: float) -> bool:
    """True when a speech model labels the whole span non-speech."""
    i0 = max(0, int(start / frame_sec))
    i1 = min(len(mask), max(i0 + 1, int(end / frame_sec) + 1))
    return not any(mask[i] for i in range(i0, i1))


def filter_masked(
    spans: Sequence[tuple[float, float]], mask: Sequence[bool], frame_sec: float = VAD_FRAME_SEC,
) -> list[tuple[float, float]]:
    """Drop the spans a speech model labels non-speech throughout.

    The only authority a model gets over the guard: protection is the default
    and confirmation is the exception, because a clipped word cannot be undone.
    """
    return [(s, e) for s, e in spans if not _masked_out(s, e, mask, frame_sec)]


def unclassified_voice(
    env: Envelope,
    floor: float,
    words: list[RefinedWord],
    exclude: Sequence[tuple[float, float]] = (),
    below_speech_db: float = SPEECH_LEVEL_BELOW_DB,
) -> list[tuple[float, float]]:
    """Voiced, speech-level spans no refined word and no already-claimed event
    (moan, mouth noise) covers.

    Only the waveform can see these, which is why it arbitrates them: on the
    reference recording a 0.16 s fragment inside a pause trim had no transcript
    entry at all, and the words alone would have let the trim delete it.

    UNCLASSIFIED, not "spoken words": the gates are level and voicing (RMS,
    zero crossings), which a hum, a sigh or loud room tone also pass. Callers
    may narrow the result with `filter_masked` when a speech model is available;
    nothing here claims to tell a word from a noise.
    """
    if len(env.db) == 0:
        return []
    threshold = speech_level(env.db) - below_speech_db
    claimed = [(w.start, w.end) for w in words]
    spans: list[tuple[float, float]] = []
    for start, end in voiced_bursts(env, floor):
        if end - start < MIN_UNCLASSIFIED_VOICE_SEC:
            continue
        if _covered(start, end, claimed, COVER_SLACK_SEC):
            continue
        if _covered(start, end, exclude, COVER_SLACK_SEC):
            continue
        i0, i1 = max(0, env.frame_at(start)), min(len(env.db), env.frame_at(end) + 1)
        if i1 <= i0 or float(np.max(env.db[i0:i1])) < threshold:
            continue
        spans.append((start, end))
    return spans
