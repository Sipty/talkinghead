"""Acoustic join grader: does a splice in a rendered cut survive scrutiny?

`check_join(RENDERED_VIDEO, join_t)` inspects the audio at one join on the
RENDERED timeline and grades it:

  - step_db:      RMS level jump across the join (a harsh splice shows as a
                  >6dB step between two voiced regions with no pause)
  - resume_gap:   seconds after the join before sustained speech starts
                  (a natural sentence beat is >150ms; an instant resume with a
                  big level step is harsh; instant resume with a continuous
                  level is a clean hard cut)
  - clipped:      voiced level within 30ms of the join = phoneme cut mid-onset

Scope, stated honestly: this catches ACOUSTIC artifacts only. A cut that
truncates a word whose final phoneme was already decaying leaves no acoustic
signature and passes here; catching linguistic truncation requires
transcribing the render and reading it as prose (the eval-loop QA pass).
Run both.

`source_edge_facts` is the second, source-side view of the same seams: it
looks at the SOURCE audio kept next to each cut edge on BOTH sides, so a
word tail or onset chopped while still voiced is caught even when the render
afterwards sounds clean at the seam (the grader above reads only what follows
it). It also measures the quiet the seam keeps, for the pacing facts.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .audio import Envelope
from .timeline import Plan

SR = 48000  # analysis sample rate
HOP_S = 0.005  # envelope resolution
# Speech/silence is judged relative to the slice's own room tone (p25), NOT an
# absolute dB: talking-head audio varies hugely in level between recordings and
# even between sections, and a fixed floor reads breath noise as speech (or
# normal speech as silence) depending on the recording.
ROOM_PCTL = 25  # percentile of slice envelope treated as room tone
SPEECH_ABOVE_ROOM_DB = 12.0  # level above room tone that counts as speech
SUSTAINED_HOPS = 3  # speech onset must hold this many hops (15ms) to count
CLIP_ABOVE_ROOM_DB = 15.0  # level above room within 30ms = phoneme cut mid-onset
STEP_THRESHOLD_DB = 6.0  # voiced-to-voiced step above this is harsh
CLIP_RISE_MS = 30  # speech onset window checked for clipped phonemes

# Source-side edge check: the kept audio right next to a cut edge is voiced
# (a chopped word) when it is well above the floor AND close to the loudest
# frame within +-EDGE_PEAK_WINDOW_SEC. Both conditions are needed: a quiet
# edge beside loud speech is a clean cut, and a tail that has already decayed
# well below the word's peak is a natural end, not a chop.
EDGE_SAMPLE_SEC = 0.010
EDGE_PEAK_WINDOW_SEC = 0.070
EDGE_VOICED_ABOVE_FLOOR_DB = 15.0
EDGE_NEAR_PEAK_DB = 6.0
# A seam is matched to the cut whose start lies within this of the segment end.
EDGE_CUT_MATCH_SEC = 0.002


def _decode_pcm(video: str, t0: float, t1: float) -> np.ndarray:
    """Decode mono f32 PCM for [t0, t1] of the video's audio track."""
    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as f:
        out = f.name
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", video, "-ss", str(t0),
         "-t", str(t1 - t0), "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", out],
        check=True,
    )
    pcm = np.fromfile(out, dtype=np.float32)
    Path(out).unlink(missing_ok=True)
    return pcm


def _envelope_db(pcm: np.ndarray) -> tuple[np.ndarray, float]:
    """RMS envelope in dB at HOP_S resolution."""
    hop = int(HOP_S * SR)
    n = len(pcm) // hop
    if n == 0:
        return np.array([]), 0.0
    env = np.sqrt((pcm[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    return 20 * np.log10(env + 1e-9), n * HOP_S


def _speech_resume(db: np.ndarray, i: int, room: float) -> float | None:
    """Seconds after join index i until sustained speech starts; None if none."""
    thr = room + SPEECH_ABOVE_ROOM_DB
    run = 0
    for j in range(i, len(db)):
        if db[j] > thr:
            run += 1
            if run >= SUSTAINED_HOPS:
                return (j - SUSTAINED_HOPS + 1 - i) * HOP_S
        else:
            run = 0
    return None


def check_join(video: str, join_t: float, window: float = 2.5) -> dict:
    """Analyze one join in the RENDERED video at absolute time join_t."""
    clip_start = max(0.0, join_t - window / 2)
    pcm = _decode_pcm(video, clip_start, join_t + window / 2)
    db, dur = _envelope_db(pcm)
    if len(db) == 0:
        return {"t": join_t, "error": "no audio decoded"}
    i = int((join_t - clip_start) / HOP_S)  # join position in envelope index

    # Room tone from the slice itself; the pre-join half is usually longer
    # speech, so p25 lands near the quiet floor even in loud sections.
    room = float(np.percentile(db, ROOM_PCTL))

    pre = db[max(0, i - 40) : max(1, i - 4)]  # 200-20 ms before join
    post = db[i + 4 : i + 40]  # 20-200 ms after join
    step = float(post.mean() - pre.mean()) if len(pre) and len(post) else 0.0

    resume_gap = _speech_resume(db, i, room)  # None = speech never (re)starts

    # clipped phoneme: voiced level within CLIP_RISE_MS of the join
    clip_window = db[i : i + max(1, int(CLIP_RISE_MS / 1000 / HOP_S))]
    clipped = bool(len(clip_window) and clip_window.max() > room + CLIP_ABOVE_ROOM_DB)

    # grade
    problems = []
    if clipped:
        problems.append(
            f"CLIPPED: voiced audio within {CLIP_RISE_MS}ms of join (phoneme cut mid-onset)"
        )
    if resume_gap is not None and resume_gap < 0.05:
        # speech starts immediately: fine only if the level is continuous
        if step > STEP_THRESHOLD_DB:
            problems.append(f"HARSH: {step:+.1f}dB voiced-to-voiced step with no pause")
    elif resume_gap is not None and resume_gap < 0.15 and step < -3.0:
        # quick drop then instant resume = ducked-word stutter; a >150ms pause
        # is a natural sentence beat and is fine
        problems.append(f"ABRUPT: {resume_gap*1000:.0f}ms gap after a {step:+.1f}dB drop")
    # resume_gap None = voiced continues throughout: clean hard cut if step small
    grade = "OK" if not problems else "; ".join(problems)
    return {
        "t": round(join_t, 3),
        "step_db": round(step, 1),
        "resume_gap_s": round(resume_gap, 3) if resume_gap is not None else None,
        "clipped": clipped,
        "grade": grade,
    }


def edge_state(env: Envelope, floor: float, t: float, side: str) -> dict:
    """Level of the EDGE_SAMPLE_SEC of kept source audio beside a cut edge at
    source time t (`side` "left": the audio just before t; "right": just after)
    against the local peak, and whether that reads as a chopped voiced word."""
    fs, n = env.frame_sec, len(env.db)
    i = int(round(t / fs))
    k = max(1, round(EDGE_SAMPLE_SEC / fs))
    lo, hi = (i - k, i) if side == "left" else (i, i + k)
    lo, hi = max(0, lo), min(n, hi)
    if hi <= lo:
        return {"t": round(t, 3), "db": None, "peak": None, "voiced": False}
    w = int(np.ceil(EDGE_PEAK_WINDOW_SEC / fs))
    level = float(env.db[lo:hi].max())
    peak = float(env.db[max(0, i - w) : min(n, i + w)].max())
    voiced = level >= floor + EDGE_VOICED_ABOVE_FLOOR_DB and level >= peak - EDGE_NEAR_PEAK_DB
    return {"t": round(t, 3), "db": round(level, 1), "peak": round(peak, 1), "voiced": bool(voiced)}


def _quiet_kept(env: Envelope, threshold_db: float, t: float, side: str, limit_sec: float) -> float:
    """Seconds of contiguous audio below `threshold_db` kept on one side of an
    edge, walking away from the edge into the kept segment (at most limit_sec)."""
    fs, n = env.frame_sec, len(env.db)
    i = int(round(t / fs))
    steps = int(limit_sec / fs)
    count = 0
    while count < steps:
        j = i - 1 - count if side == "left" else i + count
        if j < 0 or j >= n or env.db[j] >= threshold_db:
            break
        count += 1
    return count * fs


def source_edge_facts(plan: Plan, env: Envelope, floor: float, quiet_above_floor_db: float = 6.0) -> list[dict]:
    """One record per join of the main source, judged on the SOURCE audio.

    Both edges of each seam are checked (see EDGE_* above); a voiced edge is
    flagged "left", "right" or "both". `quiet` is the speechless audio the seam
    keeps in total (left + right), which the pacing facts group by cut reason.
    `t` is the join on the output timeline, as in `check_join`.
    """
    threshold = floor + quiet_above_floor_db
    out: list[dict] = []
    offset = 0.0
    for prev, seg in zip(plan.segments, plan.segments[1:]):
        offset += prev.length
        if prev.source != plan.main or seg.source != plan.main:
            continue
        # A seam the source does not run forward across (a reordered plan) has no cut at its
        # start: matching one by time would pin an unrelated cut to it.
        forward = seg.start > prev.end
        cut = next((c for c in plan.cuts if abs(c.start - prev.end) <= EDGE_CUT_MATCH_SEC), None) if forward else None
        left = edge_state(env, floor, prev.end, "left")
        right = edge_state(env, floor, seg.start, "right")
        flag = "both" if left["voiced"] and right["voiced"] else "left" if left["voiced"] else \
            "right" if right["voiced"] else None
        ql = _quiet_kept(env, threshold, prev.end, "left", prev.length)
        qr = _quiet_kept(env, threshold, seg.start, "right", seg.length)
        out.append({
            "t": round(offset, 3), "reason": cut.reason if cut else ("unknown" if forward else "reorder"),
            "left": left, "right": right, "flag": flag,
            "quiet_left": round(ql, 3), "quiet_right": round(qr, 3), "quiet": round(ql + qr, 3),
        })
    return out
