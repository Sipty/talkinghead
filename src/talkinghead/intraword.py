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
# Scribe's stamp for the token can start and end tens of milliseconds off the audio. The onset is therefore searched
# within this margin either side of the stamp, so the measurement does not depend on where the stamp put the start.
ALIGN_SEC = 0.08
# A restart is never sooner than the broken-off part's shortest sound, nor later than this share of the token.
TAU_MIN_SEC = 0.08
TAU_MAX_SEC = 0.70
TAU_MAX_FRAC = 0.7
# The cut ends in the quietest frame in this window before the restart.
SNAP_BEFORE_SEC = 0.04
# A restart is claimed only when the onset matches this well AND the cut point is a real closure. Measured on the
# cook's five stutters and 731 ordinary words of the same recording, with the onset aligned and the closure chosen
# as described in locate. At 0.90 and 8 dB, four stutters pass with their numbers constant across stamp offsets
# (impl 0.938/22.3, a-away 0.917/17.3, in-invaluable 0.926/11.2, ben 0.939/11.3). About 76 ordinary words pass too:
# a placement needs Scribe's X-Xy token and the judge's confirmation first. The fifth, dis-discuss (0.88 to 0.90
# across offsets), is not stable at any threshold here, and local whisper hears "thing we need to test is" after its
# cut, so it stays unplaced. A dip of 6 dB would pass the synthetic no-closure guards as well.
MIN_SIMILARITY = 0.90
MIN_DIP_DB = 8.0
# The energy dip is measured against the loudest frame this far before the restart.
DIP_LOOKBACK_SEC = 0.20
FRAME_SEC = voice.FRAME_HOP / voice.SAMPLE_RATE


@dataclass(frozen=True)
class Restart:
    """A restart found in one token: `start` is where the aligned token starts, `tau` where the restart begins and
    `cut_end` where the cut stops, all seconds from the stamped token start. `similarity` is the onset match at `tau`."""

    start: float
    tau: float
    cut_end: float
    similarity: float
    dip_db: float

    @property
    def found(self) -> bool:
        return self.similarity >= MIN_SIMILARITY and self.dip_db >= MIN_DIP_DB


def load_span(path: str, start: float, end: float, lead: float = 0.0) -> np.ndarray:
    """Mono 16 kHz float32 samples of [start - lead, end) seconds of the source's audio."""
    start = max(0.0, start - lead)
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


def locate(samples: np.ndarray, lead: float = 0.0) -> Restart | None:
    """The restart in a token's audio, or None when the token is too short to hold one.

    `samples` start `lead` seconds before the stamped token start (see load_span). Every candidate (onset, restart)
    pair within ALIGN_SEC of the stamp is scored. The restart reported is the best onset match among the candidates
    that stop on a closure (their cut point dips by MIN_DIP_DB), so a match on a voiced run with no closure cannot
    hide the real restart, and an onset moved by tens of milliseconds gives the same answer. Times are seconds from
    the stamped start.
    """
    dur = samples.size / voice.SAMPLE_RATE - lead
    mel = voice.log_mel(samples)
    lead_frames = round(lead / FRAME_SEC)
    if mel.shape[0] < ONSET_FRAMES + lead_frames:
        return None
    k_lo = math.ceil(TAU_MIN_SEC / FRAME_SEC - 1e-9)
    k_hi_rel = int(min(TAU_MAX_SEC, TAU_MAX_FRAC * dur) / FRAME_SEC + 1e-9)
    shift = round(ALIGN_SEC / FRAME_SEC) if lead else 0
    best = None  # (has_closure, similarity, onset frame, restart frame, cut frame, dip)
    for o in range(max(0, lead_frames - shift), lead_frames + shift + 1):
        if o + k_lo + ONSET_FRAMES > mel.shape[0]:
            break
        onset = _centred(mel[o:o + ONSET_FRAMES])
        k_hi = min(o + k_hi_rel, mel.shape[0] - ONSET_FRAMES)
        for k in range(o + k_lo, k_hi + 1):
            sim = voice.similarity(onset, _centred(mel[k:k + ONSET_FRAMES]))
            cut = _quiet_before(samples, k)
            dip = _dip_db(samples, k, cut)
            key = (dip >= MIN_DIP_DB, sim)
            if best is None or key > (best[0], best[1]):
                best = (dip >= MIN_DIP_DB, sim, o, k, cut, dip)
    if best is None:
        return None
    _, best_sim, o, k, cut, dip = best
    return Restart(start=(o - lead_frames) * FRAME_SEC, tau=(k - lead_frames) * FRAME_SEC,
                   cut_end=(cut - lead_frames) * FRAME_SEC, similarity=best_sim, dip_db=dip)


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
