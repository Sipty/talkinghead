"""Restarts inside one word: "impl-implement", where the speaker began the word, broke off and said it whole.

Scribe stamps the whole token once, so where the broken-off part ends is found in the audio. The restart repeats
the word's first sound, so the first frames of the token are compared with frames at every candidate offset, and
the offset that matches best is the restart. The cut ends in the quietest frame just before that restart. Energy
alone is not enough: a stop consonant's closure is silent and sits inside the broken-off part.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass

import numpy as np

from . import voice

# Log-mel frames compared at the token's start: 10 frames at a 10 ms hop, about the first 100 ms of the word.
ONSET_FRAMES = 10
# A restart is never sooner than the broken-off part's shortest sound, nor later than this share of the token.
TAU_MIN_SEC = 0.08
TAU_MAX_SEC = 0.70
TAU_MAX_FRAC = 0.7
# The cut ends in the quietest frame in this window before the restart.
SNAP_BEFORE_SEC = 0.04
# A restart is claimed only when the onset matches this well AND the cut point is a real closure. Set on real audio
# (the cook's five stutters against 120 ordinary words): at 0.90 and 8 dB, four of five stutters pass and 3 of 120
# ordinary words would pass. Lower values catch the fifth stutter but let more ordinary words through.
MIN_SIMILARITY = 0.90
MIN_DIP_DB = 8.0
# The energy dip is measured against the loudest frame this far before the restart.
DIP_LOOKBACK_SEC = 0.20
FRAME_SEC = voice.FRAME_HOP / voice.SAMPLE_RATE


@dataclass(frozen=True)
class Restart:
    """A restart found in one token: `tau` is where the restart begins and `cut_end` where the cut stops, both
    seconds from the token's start. `similarity` is the onset match at `tau`."""

    tau: float
    cut_end: float
    similarity: float
    dip_db: float

    @property
    def found(self) -> bool:
        return self.similarity >= MIN_SIMILARITY and self.dip_db >= MIN_DIP_DB


def load_span(path: str, start: float, end: float) -> np.ndarray:
    """Mono 16 kHz float32 samples of [start, end) seconds of the source's audio."""
    dur = max(0.0, end - start)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", path, "-t", f"{dur:.3f}", "-vn",
         "-ac", "1", "-ar", str(voice.SAMPLE_RATE), "-f", "f32le", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def _centred(patch: np.ndarray) -> np.ndarray:
    flat = patch.reshape(-1)
    return flat - flat.mean()


def locate(samples: np.ndarray) -> Restart | None:
    """The restart in a token's audio (starting at the token's first sample), or None when the token is too short
    to hold one."""
    dur = samples.size / voice.SAMPLE_RATE
    mel = voice.log_mel(samples)
    if mel.shape[0] < ONSET_FRAMES:
        return None
    k_lo = math.ceil(TAU_MIN_SEC / FRAME_SEC - 1e-9)
    k_hi = min(int(min(TAU_MAX_SEC, TAU_MAX_FRAC * dur) / FRAME_SEC + 1e-9), mel.shape[0] - ONSET_FRAMES)
    if k_hi < k_lo:
        return None
    onset = _centred(mel[:ONSET_FRAMES])
    best_k, best_sim = k_lo, -math.inf
    for k in range(k_lo, k_hi + 1):
        sim = voice.similarity(onset, _centred(mel[k:k + ONSET_FRAMES]))
        if sim > best_sim:
            best_k, best_sim = k, sim
    cut = _quiet_before(samples, best_k)
    return Restart(tau=best_k * FRAME_SEC, cut_end=cut * FRAME_SEC, similarity=best_sim,
                   dip_db=_dip_db(samples, best_k, cut))


def _frame_rms(samples: np.ndarray, j: int) -> float:
    seg = samples[j * voice.FRAME_HOP:(j + 1) * voice.FRAME_HOP]
    return float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) if seg.size else 0.0


def _dip_db(samples: np.ndarray, k: int, cut: int) -> float:
    """How far the cut point sits below the loudest frame in the lookback before the restart, in dB."""
    lo = max(0, k - int(round(DIP_LOOKBACK_SEC / FRAME_SEC)))
    loudest = max(_frame_rms(samples, j) for j in range(lo, k + 1))
    return 20 * math.log10((loudest + 1e-9) / (_frame_rms(samples, cut) + 1e-9))


def _quiet_before(samples: np.ndarray, k: int) -> int:
    """The frame index (10 ms hops) of the lowest-energy frame starting in the window before frame `k`.

    Ties go to the later frame, the one nearest the restart.
    """
    span = int(round(SNAP_BEFORE_SEC / FRAME_SEC))
    best_j, best_e = k, math.inf
    for j in range(max(0, k - span), k + 1):
        energy = _frame_rms(samples, j)
        if energy <= best_e:
            best_j, best_e = j, energy
    return best_j
