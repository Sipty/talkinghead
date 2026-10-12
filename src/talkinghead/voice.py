"""Voice signatures for speaker reconciliation. numpy only.

A signature describes one speaker's voice over the frames of their words: the mean and spread of
log-mel energy per band. The mean is centred on the chunk's own mean, so the microphone and drift
between requests cancel out and only the difference between speakers in the same chunk remains.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
FRAME_HOP = 160  # 10 ms
FRAME_WIN = 400  # 25 ms
FFT_SIZE = 512
MEL_BANDS = 40
# Below this many usable frames a label has no signature and is matched on text evidence only.
SIGNATURE_MIN_FRAMES = 25
# Frames per block when transforming a whole recording, to bound memory.
_BLOCK_FRAMES = 60000


def _mel_filterbank() -> np.ndarray:
    freqs = np.linspace(0, SAMPLE_RATE / 2, FFT_SIZE // 2 + 1)
    def to_mel(hz: np.ndarray | float) -> np.ndarray | float:
        return 2595 * np.log10(1 + hz / 700)

    def from_mel(mel: np.ndarray) -> np.ndarray:
        return 700 * (10 ** (mel / 2595) - 1)
    edges = from_mel(np.linspace(to_mel(0), to_mel(SAMPLE_RATE / 2), MEL_BANDS + 2))
    bank = np.zeros((MEL_BANDS, freqs.size), np.float32)
    for i in range(MEL_BANDS):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        bank[i] = np.maximum(0, np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)))
    return bank


def log_mel(samples: np.ndarray) -> np.ndarray:
    """(frames, MEL_BANDS) log-mel energies of 16 kHz mono samples in [-1, 1].

    Frame i is centred at frame_centre(i).
    """
    x = np.asarray(samples, dtype=np.float32)
    if x.size < FRAME_WIN:
        return np.empty((0, MEL_BANDS), np.float32)
    bank = _mel_filterbank()
    window = np.hanning(FRAME_WIN).astype(np.float32)
    view = np.lib.stride_tricks.sliding_window_view(x, FRAME_WIN)[::FRAME_HOP]
    out = np.empty((view.shape[0], MEL_BANDS), np.float32)
    for s in range(0, view.shape[0], _BLOCK_FRAMES):
        e = min(view.shape[0], s + _BLOCK_FRAMES)
        spec = np.abs(np.fft.rfft(view[s:e] * window, n=FFT_SIZE, axis=1)) ** 2
        out[s:e] = np.log(spec @ bank.T + 1e-10)
    return out


def frames_in(start: float, end: float, n_frames: int) -> np.ndarray:
    """Indices of the frames whose centre lies in [start, end] seconds."""
    centre_offset = FRAME_WIN / 2
    first = int(np.ceil((start * SAMPLE_RATE - centre_offset) / FRAME_HOP))
    last = int(np.floor((end * SAMPLE_RATE - centre_offset) / FRAME_HOP))
    first, last = max(first, 0), min(last, n_frames - 1)
    return np.arange(first, last + 1) if last >= first else np.empty(0, np.int64)


def chunk_signatures(mel: np.ndarray, spans: list[tuple[float, float, str]]) -> dict[str, tuple[np.ndarray, int]]:
    """Voice signature and frame count per label, from one chunk's (start, end, label) word spans.

    Frames where another label speaks are left out, and so are quiet frames (below the chunk's
    25th percentile of word-frame energy). The mean is centred on the chunk's own word-frame mean.
    """
    by_label: dict[str, list[np.ndarray]] = {}
    for start, end, label in spans:
        by_label.setdefault(label, []).append(frames_in(start, end, mel.shape[0]))
    frames = {label: np.unique(np.concatenate(parts)) for label, parts in by_label.items() if parts}
    if not frames:
        return {}
    everything = np.unique(np.concatenate(list(frames.values())))
    if everything.size == 0:
        return {}
    energy = mel[everything].mean(1)
    loud = everything[energy >= np.percentile(energy, 25)]
    centre = mel[loud].mean(0)
    out: dict[str, tuple[np.ndarray, int]] = {}
    for label, mine in frames.items():
        others = [f for other, f in frames.items() if other != label]
        exclusive = np.setdiff1d(mine, np.unique(np.concatenate(others))) if others else mine
        keep = exclusive[mel[exclusive].mean(1) >= np.percentile(energy, 25)] if exclusive.size else exclusive
        if keep.size < SIGNATURE_MIN_FRAMES:
            continue
        out[label] = (np.concatenate([mel[keep].mean(0) - centre, mel[keep].std(0)]), int(keep.size))
    return out


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two signatures."""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def merge(total: np.ndarray | None, weight: float, vec: np.ndarray, count: int) -> tuple[np.ndarray, float]:
    """Frame-weighted running mean of signatures."""
    if total is None:
        return vec.copy(), float(count)
    return (total * weight + vec * count) / (weight + count), weight + count
