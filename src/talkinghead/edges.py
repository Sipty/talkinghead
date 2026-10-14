"""Edge placement: one pass applied to every cut list before it becomes a plan.

Order matters and each step is a pure function:
1. merge            overlapping/touching cuts fuse
2. protect_words    no kept word is ever clipped (word-removing cuts snap to
                    whole words; silence/noise cuts retreat from any word); a judged
                    cut (JUDGED_REASONS) is never split around a kept-looking event
3. steer_out_of_moans   a join never lands inside a detected moan: the cut
                    swallows the moan when it sits free in the gap, else
                    retreats so the moan stays whole in the kept audio
4. sweep_to_quiet   edges move outward over weak excursions until the
                    envelope sits at floor level, never into a kept word
5. drop cuts shorter than min_cut (a join costs more than they save)
6. snap_to_valleys  word-removing cuts only: each edge moves to the quietest
                    frame-grid position near it (word stamps are 80 ms
                    quantised and biased, so a stamp edge often sits on speech)
7. tighten_repairs  retake/restart cuts inside a sentence: real quiet beside
                    the kept words is cut away so the correction follows
                    directly
8. keep_sentence_tail   word-removing cuts after a completed sentence: at
                    least `splice_quiet_sentence` of real source audio stays
                    after the last kept word
9. keep_splice_quiet    other word-removing cuts after a clause end: real
                    source quiet next to a kept word is handed back
10. keep_splice_lead_in  kept speech within splice_lead_in of a cut end: the
                    end moves back through quiet so that much source precedes it
11. flow_guard      cuts with less than flow_min_kept of speechless audio
                    between them fuse (a rapid-cut pileup reads as a glitch)
12. snap_to_frames  edges onto the video frame grid, shrinking, so segments
                    are whole frames and audio/video cannot drift
13. hold_judged     every word a judged cut removes (coverage test) stays removed: a word the
                    edge steps left uncovered is added as a cut of its own extent
14. flow_guard      again: slivers the hold left between cuts fuse
15. settle_filler_edges  short filler cuts still on voiced audio: the voiced run is
                    walked into the cut, else the filler is dropped
16. spare_other_speaker  two-speaker recordings only: a mechanical cut shrinks away from
                    words spoken by someone other than its own target (runs first, and last)
17. merge

"Kept word" means a refined word not removed by the cut being placed. `pad`
is the minimum distance a silence/noise edge keeps from any kept word.

`protected_spans` are whole event intervals (laughter, unknown events) that
no cut of any reason may touch: they count as kept audio in every step, keep
the same pad as a kept word, and block flow_guard fusions. Frame snapping only
shrinks cuts, so a protected interval survives it.

A word-removing cut is a decision about words, and its edges land mid-speech
where the transcript's stamps are least reliable, so it follows a different
rule. It keeps only `splice_pad` from a kept word's refined (acoustic where the
word ends a voiced burst) edge, then the edge is placed by the waveform: the
lowest-energy grid position within `splice_valley_window`, never inside kept
audio and never inward of a removed word's refined edge (that would keep a
fragment of the removed word).

A retake inside a sentence then joins quickly: quiet beside the kept words
beyond `splice_pad` is cut. After a completed sentence the cut starts no
earlier than `splice_quiet_sentence` after the last kept word. After a clause
end any other word-removing cut is topped up from the removed side to
`splice_quiet_clause`, split about 60/40 after the left word / before the right
word, and only from source frames that are actually quiet: silence is never
synthesised and voiced audio of a removed word is never kept to reach the
target.

Whatever the cut, kept speech that starts within `splice_lead_in` of the cut end is given that much
real source in front of it: the end moves back through quiet audio only, and never over a removed word.

A judged cut (`JUDGED_REASONS`: cold-open, outro, tangent, lead-in, retake, content) is a decision about
the words in its span, so the whole span goes: no split around a laugh, a voice or any other event inside
it, and no other cut may shrink it. Only its two edges take the refinement above. Every word the judge's span
covers (`judged_removes`) is therefore removed, which `verify` checks (judged_removals).
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field, replace

import numpy as np

from .audio import Envelope, RefinedWord
from .derive import pause_boundary
from .joins import EDGE_CUT_MATCH_SEC, EDGE_VOICED_ABOVE_FLOOR_DB, edge_state
from .residual import _split_restart
from .timeline import MIN_CUT, Cut, Plan, Segment, fuse_into, merge_cuts
from .types import JUDGED_REASONS, MECHANICAL_REASONS

# Reasons whose spans are meant to remove words (edges snap to word bounds).
WORD_CUT_REASONS = frozenset({"retake", "restart", "filler", "content", "cold-open", "outro", "join",
                              "tangent", "lead-in", "polish"})
# Reasons whose whole job is removing NON-speech: for these, voiced audio the
# transcript does not cover may not be landed inside or swallowed by the cut.
# An editorial cut (above) is a decision about content, so the waveform only
# keeps its EDGES clear and what it removes is reported instead.
VOICE_SAFE_REASONS = frozenset({"silence", "noise", "gap"})
# Silence placed from word stamps with fixed pads (polish.gap_cuts): the envelope steps would pull these back into
# the words they separate, so finalize only trims them around the cuts that exist after the other steps.
PLACED_REASONS = frozenset({"gap"})


def is_intra_restart(c: Cut) -> bool:
    """An output-pass cut that removes the broken-off start of one word. It lies inside a word the transcript keeps,
    so it is placed on the audio (see intraword), not guarded against the word's edges."""
    return c.provenance.get("intra") is True


def _is_placed(c: Cut) -> bool:
    return c.reason in PLACED_REASONS or is_intra_restart(c)
# A cut this close to a placed gap takes the ASR-stamped word extent (see asr_guarded), so the pair cannot clip a word.
ASR_GUARD_REACH = 0.5
# A word is removed by a word-cut when at least this fraction of it is covered.
REMOVE_COVERAGE = 0.5
# Kept-quiet split of a splice target: this share follows the left kept word,
# the rest precedes the right one.
SPLICE_QUIET_LEFT_SHARE = 0.6
# Cuts that abandon words said wrongly and restate them: inside a sentence
# they join straight to the correction.
REPAIR_REASONS = frozenset({"retake", "restart", "polish"})
# Valley candidates within this many dB of the quietest count as tied, and the
# one nearest the original edge wins: envelope noise must not move an edge.
VALLEY_TIE_DB = 0.25
# Join cost (lower is better), summed over a cut's two edges. Each edge adds the
# dip its crossfade blends away; the join adds the level step between the kept
# audio before the start and after the end, and voiced audio the kept side holds
# beside an edge (a chopped word, or an onset the crossfade would clip).
LEVEL_STEP_WEIGHT = 0.5
VOICED_EDGE_WEIGHT = 0.5
# refine_joins moves a join only when its cost falls by more than this.
JOIN_MOVE_MARGIN = 1.0
# refine_joins searches each edge only this far from where the author put it.
JOIN_MOVE_WINDOW = 0.04
# pause_boundary only needs the topic threshold for pauses; a kept word's
# punctuation alone decides the splice target.
_NO_TOPIC = type("_NoTopic", (), {"pause_topic_above": math.inf})()


@dataclass
class EdgeContext:
    duration: float
    env: Envelope
    floor: float
    words: list[RefinedWord]
    moans: list[tuple[float, float]] = field(default_factory=list)
    # Complete laughter / unknown audio-event intervals: kept audio for every
    # cut reason, including their quiet onset and decay.
    protected_spans: list[tuple[float, float]] = field(default_factory=list)
    # Voiced speech-level spans no word covers (audio.unclassified_voice):
    # audio no transcript accounts for, which only the waveform can protect.
    voices: list[tuple[float, float]] = field(default_factory=list)
    voice_guard: bool = True
    # Minimum distance a silence/noise edge keeps from any kept word.
    pad: float = 0.05
    # Word-removing cuts: distance kept from a kept word, valley search reach,
    # and the crossfade the valley score is measured over (the join overlaps
    # by this much).
    splice_pad: float = 0.03
    splice_valley_window: float = 0.12
    # Quiet handed back after a clause end, and the tail of real source audio a
    # cut leaves after a completed sentence's last kept word.
    splice_quiet_sentence: float = 0.08
    splice_quiet_clause: float = 0.18
    # Kept speech that starts this soon after a cut end gets this much real source
    # in front of it, handed back through quiet audio only (keep_splice_lead_in).
    splice_lead_in: float = 0.06
    splice_crossfade_sec: float = 0.020
    quiet_above_floor_db: float = 6.0
    flow_min_kept: float = 2.0
    # A cut shorter than this buys nothing and costs a join; it is dropped
    # whatever its reason (leaving a 60 ms filler in is the accepted error).
    min_cut: float = 0.08
    # Video frame rate: cut edges snap to the frame grid (start later, end
    # earlier) so every kept segment is a whole number of frames and the
    # render cannot drift between audio and video. None = no snapping.
    fps: float | None = 30.0
    # Filler cuts at most this long are dropped when an edge still sits on voiced
    # audio after the voiced run is walked in. 0 (the default) keeps every filler
    # cut. See Config.
    filler_keep_voiced_max: float = 0.0
    # Set by finalize_cuts: filler cuts dropped for a voiced edge.
    dropped_fillers: int = 0
    # Set by finalize_cuts: mechanical cuts shrunk or dropped to spare another speaker's words.
    spared_other_speaker: int = 0
    # Set by take_removed_whole: fragments settled per outcome, and the removed words given back whole.
    fragments: dict = field(default_factory=dict)
    given_back: list = field(default_factory=list)
    # Set by take_removed_whole: one record per fragment decision (word, side, outcome).
    decisions: list = field(default_factory=list)
    # Set by finalize_cuts: ids of the words the stage decisions remove, read by spare_other_speaker.
    decided_words: set = field(default_factory=set)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _pad(reason: str, ctx: EdgeContext) -> float:
    """Distance an edge of this kind of cut keeps from kept audio."""
    return ctx.splice_pad if reason in WORD_CUT_REASONS else ctx.pad


def removed_words(cut: Cut, words: list[RefinedWord]) -> list[RefinedWord]:
    """Words this cut removes: none for silence/noise cuts, majority-covered ones otherwise.

    Coverage is measured on the refined bounds only: a word is removed by the audio the cut covers, never by ASR
    padding alone (a neighbour whose ASR stamp runs into the cut is still a kept word).
    """
    if cut.reason not in WORD_CUT_REASONS:
        return []
    out = []
    for w in words:
        if w.end <= w.start:
            # Degenerate token refined to a point (whisper emits these): it
            # carries no audio, so any word-cut spanning its timestamp removes
            # it whole. Left "kept" it pins a phantom hole mid-cut.
            if cut.start <= w.start <= cut.end:
                out.append(w)
            continue
        length = w.end - w.start
        if _overlap(cut.start, cut.end, w.start, w.end) / length >= REMOVE_COVERAGE:
            out.append(w)
    return out


def _coalesce(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sorted spans with overlapping ones fused (touching ones stay apart)."""
    out: list[list[float]] = []
    for s, e in sorted(spans):
        if out and s < out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def word_extent(w) -> tuple[float, float]:
    """A word's audio extent: its refined bounds widened to its ASR stamps, which either can shorten.

    The ASR stamps and the refined bounds disagree by up to half a second on a short word, so a cut guarded by one
    alone can take part of the other. The gap stage and the edge guards both use this extent.
    """
    return min(w.start, getattr(w, "asr_start", w.start)), max(w.end, getattr(w, "asr_end", w.end))


def _is_output_cut(c: Cut) -> bool:
    """A cut the output pass confirmed on the render's transcript (a filler or a repeat the source did not show)."""
    return c.reason == "polish" and c.provenance.get("pass") == "output"


def asr_guarded(c: Cut) -> bool:
    """Whether this cut's guards use the ASR-stamped word extent: output-pass cuts, and cuts marked by mark_asr_guard."""
    return _is_output_cut(c) or bool(c.provenance.get("asr_guard"))


def mark_asr_guard(cuts: list[Cut], placed: list[Cut]) -> list[Cut]:
    """Mark the cuts that sit beside a placed gap, so they are guarded like the gap is.

    The gap stage cuts against a word's ASR-stamped extent, while the other cuts use the refined bounds. Where the
    two sit side by side, they fuse into one gap-labelled cut, and that cut can clip a word the stamps place
    earlier than the audio does. Marking them keeps the pair on one rule.
    """
    if not placed:
        return cuts
    out: list[Cut] = []
    for c in cuts:
        near = any(g.start - ASR_GUARD_REACH <= c.end and c.start <= g.end + ASR_GUARD_REACH for g in placed)
        out.append(Cut(c.start, c.end, c.reason, c.text, {**c.provenance, "asr_guard": True}) if near else c)
    return out


def _guard_extent(cut: Cut, w: RefinedWord) -> tuple[float, float]:
    """The audio a kept word claims against `cut`.

    A mechanical ASR-guarded cut (a silence beside a placed gap) always claims the word's ASR-widened extent: that
    guard exists so a cut never takes part of stamped speech. An output-confirmed cut claims it only where the word's
    refined audio meets the cut, because its removal was judged on the render: a stamp that runs into the cut from a
    word the cut does not touch claims nothing there.
    """
    if not asr_guarded(cut):
        return (w.start, w.end)
    if _is_output_cut(cut) and _overlap(cut.start, cut.end, w.start, w.end) <= 0:
        return (w.start, w.end)
    return word_extent(w)


def _kept_spans(cut: Cut, ctx: EdgeContext, *, voices: bool = True) -> list[tuple[float, float]]:
    """The audio this cut keeps: the words it does not remove, the protected
    event intervals (whatever the cut's reason), plus (unless `voices` is off)
    the voiced audio no word covers.

    `voices` comes from the waveform, so a dropped word is covered too: the
    reference recording had a 0.16 s utterance inside a pause trim that no
    transcript entry named, and only the audio proved it was there. An output-pass
    cut is a judgment about audio that no source word names, so its own voices
    never count as kept.
    """
    removed = {id(w) for w in removed_words(cut, ctx.words)}
    spans = [_guard_extent(cut, w) for w in ctx.words if id(w) not in removed]
    spans += [(s, e) for s, e in ctx.protected_spans if e > s]
    if voices and ctx.voice_guard and not _is_output_cut(cut):
        spans += ctx.voices
    return _coalesce(spans)


def _kept_bounds(cut: Cut, ctx: EdgeContext) -> tuple[float, float]:
    """(left, right): the furthest an edge may reach without touching kept audio."""
    left, right = 0.0, ctx.duration
    # An output-pass cut removes audio no kept word claims, so it may reach the neighbours' stamped extents.
    pad = 0.0 if _is_output_cut(cut) else _pad(cut.reason, ctx)
    for s, e in _kept_spans(cut, ctx):
        if e <= cut.start + 1e-9 or (s < cut.end and e > cut.start and s < cut.start):
            left = max(left, e + pad)
        if s >= cut.end - 1e-9 or (s < cut.end and e > cut.start and e > cut.end):
            right = min(right, s - pad)
    return left, right


def _crosstalk(w: RefinedWord, words: list[RefinedWord]) -> bool:
    """Whether a word of another speaker overlaps `w` in time."""
    return any(o.speaker is not None and o.speaker != w.speaker and o.start < w.end and o.end > w.start
               for o in words)


def protect_words(cuts: list[Cut], ctx: EdgeContext) -> list[Cut]:
    out: list[Cut] = []
    for c in cuts:
        removed = removed_words(c, ctx.words)
        judged = c.reason in JUDGED_REASONS
        start, end = c.start, c.end
        if removed:
            start = min(start, min(w.start for w in removed))
            end = max(end, max(w.end for w in removed))
        # Every edge stays clear of kept audio; only a cut whose job is removing
        # non-speech may split around a voice the transcript never named (split
        # an editorial cut there and a fragment of the passage the judge chose
        # to remove ships instead). A judged cut is never split at all: a protected
        # event or voice inside its span is part of what the judge removed (hold_judged
        # restores any word a guard pushed out).
        guards = _kept_spans(c, ctx)
        pad = _pad(c.reason, ctx)
        splits = [] if judged else _kept_spans(c, ctx, voices=c.reason in VOICE_SAFE_REASONS)
        for s, e in guards:
            if s < end and e > start:
                if s <= start:                # span straddles the start: move start past it
                    start = max(start, e + pad)
                elif e >= end:                # span straddles the end: move end before it
                    end = min(end, s - pad)
        for s, e in splits:
            if s < end and e > start and s > start and e < end:
                if s - pad - start > MIN_CUT:
                    out.append(Cut(start, s - pad, c.reason, c.text, dict(c.provenance)))
                start = e + pad
        # Cross-talk: a kept word of another speaker can push an edge into a removed word of this cut.
        # That word is then left whole (the cut stops at its near side), never clipped mid-word.
        # A judged cut keeps its removed words instead (hold_judged).
        for w in ([] if judged else removed):
            if w.speaker is None or not _crosstalk(w, ctx.words):
                continue
            if w.start < end < w.end and start <= w.start:
                end = w.start
            elif w.start < start < w.end and end >= w.end:
                start = w.end
        # Keep pad from adjacent kept audio even when not overlapping.
        for s, e in guards:
            if 0 <= start - e < pad:
                start = e + pad
            if 0 <= s - end < pad:
                end = s - pad
        if end - start > MIN_CUT:
            out.append(Cut(start, end, c.reason, c.text, dict(c.provenance)))
    return out


def steer_out_of_moans(cuts: list[Cut], ctx: EdgeContext) -> list[Cut]:
    out: list[Cut] = []
    for c in cuts:
        start, end = c.start, c.end
        left, right = _kept_bounds(c, ctx)
        pad = _pad(c.reason, ctx)
        for ms, me in ctx.moans:
            if ms < end < me:                      # join at the end lands inside the moan
                if me + pad <= right:
                    end = me + pad                 # swallow it: it sits free in the gap
                else:
                    end = max(start, ms - pad)     # leave it whole in the kept audio
            if ms < start < me:                    # join at the start lands inside the moan
                if ms - pad >= left:
                    start = ms - pad
                else:
                    start = min(end, me + pad)
        if end - start > MIN_CUT:
            out.append(Cut(start, end, c.reason, c.text, dict(c.provenance)))
    return out


def sweep_to_quiet(cuts: list[Cut], ctx: EdgeContext) -> list[Cut]:
    """Extend edges outward over every excursion above floor+quiet_db between
    the cut and the nearest kept word, so the join sits in genuine quiet."""
    db = ctx.env.db
    if len(db) == 0:
        return list(cuts)
    fs = ctx.env.frame_sec
    n = len(db) - 1
    thr = ctx.floor + ctx.quiet_above_floor_db
    out: list[Cut] = []
    for c in cuts:
        left, right = _kept_bounds(c, ctx)
        start, end = c.start, c.end
        lf = max(0, min(n, int(left / fs)))
        for i in range(lf, min(n, int(start / fs)) + 1):
            if db[i] > thr:
                start = min(start, max(left, i * fs))
                break
        rf = max(0, min(n, math.ceil(right / fs) - 1))
        for i in range(rf, min(n, math.ceil(end / fs) - 1) - 1, -1):
            if db[i] > thr:
                end = max(end, min(right, (i + 1) * fs))
                break
        out.append(Cut(start, end, c.reason, c.text, dict(c.provenance)))
    return out


def _inward_limits(cut: Cut, removed: list[RefinedWord], ctx: EdgeContext) -> tuple[float, float]:
    """(hi_start, lo_end): how far each edge may move INTO the cut before a
    removed word stops being majority-covered (or the cut stops being a cut).

    A retake/restart edge never passes the refined edge of the removed word
    beside it: the correction must not be preceded by a fragment of the
    abandoned attempt (a doubled "in in", "ta-takes")."""
    hi_start, lo_end = cut.end - ctx.min_cut, cut.start + ctx.min_cut
    if removed and cut.reason in REPAIR_REASONS:
        hi_start = min(hi_start, max(cut.start, removed[0].start))
        lo_end = max(lo_end, min(cut.end, removed[-1].end))
    elif removed:
        first, last = removed[0], removed[-1]
        hi_start = min(hi_start, first.end - REMOVE_COVERAGE * (first.end - first.start) - 1e-6)
        lo_end = max(lo_end, last.start + REMOVE_COVERAGE * (last.end - last.start) + 1e-6)
    return hi_start, lo_end


def _dip_score(db, fs: float, t: float, half: float) -> float:
    """Mean envelope level (dBFS) in the +-half neighbourhood of t: what the
    crossfade at a join centred on t would blend away."""
    lo = max(0, int(round((t - half) / fs)))
    hi = min(len(db), max(lo + 1, int(round((t + half) / fs))))
    return float(db[lo:hi].mean())


def _grid_points(ctx: EdgeContext, orig: float, lo: float, hi: float, window: float | None = None) -> list[float]:
    """Frame-grid positions within `window` (default the splice valley window) of `orig` and inside [lo, hi], skipping moans."""
    grid = ctx.fps or 1.0 / ctx.env.frame_sec
    reach = ctx.splice_valley_window if window is None else window
    a = max(lo, orig - reach)
    b = min(hi, orig + reach)
    out: list[float] = []
    for k in range(math.ceil(a * grid - 1e-6), math.floor(b * grid + 1e-6) + 1):
        t = k / grid
        if not any(ms < t < me for ms, me in ctx.moans):
            out.append(t)
    return out


def _span_peak(db, fs: float, a: float, b: float) -> float:
    """Loudest envelope frame (dBFS) over [a, b), at least one frame."""
    n = len(db)
    lo = min(max(0, round(a / fs)), n - 1)
    hi = min(n, max(lo + 1, round(b / fs)))
    return float(db[lo:hi].max())


def _edge_terms(ctx: EdgeContext, t: float, side: str) -> tuple[float, float | None, float | None]:
    """(dip, level, peak) of a cut edge at t. side "start": kept audio before t; "end": kept audio after t.

    dip is what the crossfade blends away. level is the loudest 20-200 ms of kept
    audio beside the join: the speech level, where a mean would also count the
    pause the window holds. peak is the loudest 30 ms beside the edge: a chopped
    word on the left, a voiced onset on the right. Level and peak are None where
    no kept audio lies on that side (a head or tail trim is not a join).
    """
    db, fs = ctx.env.db, ctx.env.frame_sec
    dip = _dip_score(db, fs, t, ctx.splice_crossfade_sec / 2)
    if side == "start":
        if t <= 0.02:
            return dip, None, None
        return dip, _span_peak(db, fs, t - 0.2, t - 0.02), _span_peak(db, fs, t - 0.03, t)
    if t >= ctx.duration - 0.02:
        return dip, None, None
    return dip, _span_peak(db, fs, t + 0.02, t + 0.2), _span_peak(db, fs, t, t + 0.03)


def _join_cost(ctx: EdgeContext, start: tuple[float, float | None, float | None],
               end: tuple[float, float | None, float | None]) -> float:
    """Level-aware cost of a join whose removed span starts and ends at these edge terms (lower is better)."""
    voiced = ctx.floor + EDGE_VOICED_ABOVE_FLOOR_DB
    ds, ls, ps = start
    de, le, pe = end
    step = abs(ls - le) if ls is not None and le is not None else 0.0
    onset = sum(max(0.0, p - voiced) for p in (ps, pe) if p is not None)
    return ds + de + LEVEL_STEP_WEIGHT * step + VOICED_EDGE_WEIGHT * onset


def _pairs(ctx: EdgeContext, c: Cut, starts: list[float], ends: list[float]) -> list[tuple[float, float, float, float]]:
    """(cost, distance from the original edges, start, end) for every admissible pair."""
    s_terms = {t: _edge_terms(ctx, t, "start") for t in starts}
    e_terms = {t: _edge_terms(ctx, t, "end") for t in ends}
    out: list[tuple[float, float, float, float]] = []
    for s in starts:
        for e in ends:
            if e - s >= ctx.min_cut:
                out.append((_join_cost(ctx, s_terms[s], e_terms[e]), abs(s - c.start) + abs(e - c.end), s, e))
    return out


def _ranked(pairs: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    """Cheapest first, except a near-tie (VALLEY_TIE_DB) goes to the pair nearest the original edges."""
    if not pairs:
        return []
    best = min(p[0] for p in pairs)
    tied = sorted((p for p in pairs if p[0] <= best + VALLEY_TIE_DB), key=lambda p: p[1])
    rest = sorted((p for p in pairs if p[0] > best + VALLEY_TIE_DB), key=lambda p: (p[0], p[1]))
    return tied + rest


def _near(c: Cut, ctx: EdgeContext) -> list[RefinedWord]:
    """The words any edge placement of this cut could remove or keep."""
    return [w for w in ctx.words if w.end >= c.start - 1.0 and w.start <= c.end + 1.0]


def _first_valid(c: Cut, removed: list[RefinedWord], ctx: EdgeContext,
                 ranked: list[tuple[float, float, float, float]]) -> tuple[float, Cut] | None:
    """The first pair in `ranked` that removes exactly the words `c` removes, with its cost."""
    near, want = _near(c, ctx), [id(w) for w in removed]
    for cost, _, s, e in ranked:
        moved = Cut(s, e, c.reason, c.text, dict(c.provenance))
        if [id(w) for w in removed_words(moved, near)] == want:
            return cost, moved
    return None


def snap_to_valleys(cuts: list[Cut], ctx: EdgeContext) -> list[Cut]:
    """Word-removing cuts: move both edges to the cheapest join nearby, jointly.

    The pair is scored by _join_cost, so the level step across the join and
    voiced audio beside an edge count, not only the dip at each edge. Outward
    each edge stops splice_pad short of kept audio (a kept word's voiced run is
    never entered); inward it stops where a removed word would drop below
    REMOVE_COVERAGE. The set of removed words is unchanged, else the cut is left
    as it was.
    """
    if len(ctx.env.db) == 0:
        return list(cuts)
    out: list[Cut] = []
    for c in cuts:
        if c.reason not in WORD_CUT_REASONS:
            out.append(c)
            continue
        removed = removed_words(c, ctx.words)
        left, right = _kept_bounds(c, ctx)
        hi_start, lo_end = _inward_limits(c, removed, ctx)
        starts = _grid_points(ctx, c.start, left, hi_start) or [c.start]
        ends = _grid_points(ctx, c.end, lo_end, right) or [c.end]
        found = _first_valid(c, removed, ctx, _ranked(_pairs(ctx, c, starts, ends)))
        out.append(found[1] if found else c)
    return out


def _quiet_run(db, i: int, step: int, thr: float, limit: int) -> int:
    """Consecutive envelope frames at or below `thr` from frame i, walking by `step`."""
    n = 0
    while 0 <= i < len(db) and n < limit and db[i] <= thr:
        n += 1
        i += step
    return n


def keep_splice_quiet(cuts: list[Cut], ctx: EdgeContext) -> list[Cut]:
    """Word-removing cuts after a clause end: hand back real source quiet next
    to the kept words.

    The target total is `splice_quiet_clause`, split SPLICE_QUIET_LEFT_SHARE
    after the kept word before the cut / the rest before the kept word after
    it. Quiet the kept side already has counts toward its share; only frames at
    or below floor + quiet_above_floor_db adjoining the edge are ever given
    back, so no silence is invented and no voiced audio of a removed word is
    kept. Retakes join tightly (tighten_repairs) and sentence ends have their
    own tail rule (keep_sentence_tail), so neither is padded here.
    """
    db, fs = ctx.env.db, ctx.env.frame_sec
    if len(db) == 0:
        return list(cuts)
    thr = ctx.floor + ctx.quiet_above_floor_db
    grid = ctx.fps or 1.0 / fs
    out: list[Cut] = []
    for c in cuts:
        padded = c.reason in WORD_CUT_REASONS and c.reason not in REPAIR_REASONS
        out.append(_with_splice_quiet(c, ctx, db, fs, thr, grid) if padded else c)
    return out


def _boundary(word: RefinedWord) -> str:
    """What the kept word's punctuation ends: 'sentence', 'clause' or 'mid'."""
    return pause_boundary(word.text, 0.0, _NO_TOPIC)


def _kept_neighbours(c: Cut, removed: list[RefinedWord], ctx: EdgeContext) -> tuple[list[RefinedWord], list[RefinedWord]]:
    """The kept words before and after the cut, in order."""
    gone = {id(w) for w in removed}
    kept = [w for w in ctx.words if id(w) not in gone and w.end > w.start]
    before = [w for w in kept if (w.start + w.end) / 2 <= c.start]
    after = [w for w in kept if (w.start + w.end) / 2 >= c.end]
    return before, after


def tighten_repairs(cuts: list[Cut], ctx: EdgeContext) -> list[Cut]:
    """Retake/restart cuts inside a sentence: join straight to the correction.

    Real quiet next to the kept words is cut away, back to `splice_pad` from
    the kept word's refined edge, so no sentence or clause pause appears
    between the retained words and the correction. Only frames at or below
    floor + quiet_above_floor_db are ever cut and never kept audio, so the
    surviving words keep their complete phonemes. A retake after a completed
    sentence is left to keep_sentence_tail.
    """
    db, fs = ctx.env.db, ctx.env.frame_sec
    if len(db) == 0:
        return list(cuts)
    thr = ctx.floor + ctx.quiet_above_floor_db
    grid = ctx.fps or 1.0 / fs
    return [_tightened(c, ctx, db, fs, thr, grid) if c.reason in REPAIR_REASONS else c for c in cuts]


def _tightened(c: Cut, ctx: EdgeContext, db, fs: float, thr: float, grid: float) -> Cut:
    removed = removed_words(c, ctx.words)
    before, after = _kept_neighbours(c, removed, ctx)
    if not before or not after or _boundary(before[-1]) == "sentence":
        return c
    left, right = _kept_bounds(c, ctx)
    lo = max(before[-1].end + ctx.splice_pad, left)
    hi = min(after[0].start - ctx.splice_pad, right)
    i_start, i_end = int(round(c.start / fs)), int(round(c.end / fs))
    reach_l = c.start - _quiet_run(db, i_start - 1, -1, thr, int(max(0.0, c.start - lo) / fs) + 2) * fs
    reach_r = c.end + _quiet_run(db, i_end, 1, thr, int(max(0.0, hi - c.end) / fs) + 2) * fs
    start = min(c.start, math.ceil(max(lo, reach_l) * grid - 1e-6) / grid)
    end = max(c.end, math.floor(min(hi, reach_r) * grid + 1e-6) / grid)
    moved = Cut(start, end, c.reason, c.text, dict(c.provenance))
    same = [id(w) for w in removed_words(moved, ctx.words)] == [id(w) for w in removed]
    return moved if same else c


def keep_sentence_tail(cuts: list[Cut], ctx: EdgeContext) -> list[Cut]:
    """Word-removing cuts after a completed sentence: keep the sentence's tail.

    The cut starts at least `splice_quiet_sentence` after the last kept word,
    rounded up to the frame grid, so at least that much real source audio
    follows it. The tail never reaches into the first removed word (that would
    leave a fragment of it), so continuous speech with less room keeps what is
    there. Natural pauses longer than the tail are left as they are.
    """
    grid = ctx.fps or 1.0 / ctx.env.frame_sec
    out: list[Cut] = []
    for c in cuts:
        out.append(_with_tail(c, ctx, grid) if c.reason in WORD_CUT_REASONS else c)
    return out


def _with_tail(c: Cut, ctx: EdgeContext, grid: float) -> Cut:
    removed = removed_words(c, ctx.words)
    before, _ = _kept_neighbours(c, removed, ctx)
    if not before or _boundary(before[-1]) != "sentence":
        return c
    want = math.ceil((before[-1].end + ctx.splice_quiet_sentence) * grid - 1e-9) / grid
    room = removed[0].start if removed else c.end - ctx.min_cut
    start = max(c.start, min(want, math.floor(room * grid + 1e-6) / grid))
    moved = Cut(start, c.end, c.reason, c.text, dict(c.provenance))
    same = [id(w) for w in removed_words(moved, ctx.words)] == [id(w) for w in removed]
    return moved if same and moved.length >= ctx.min_cut else c


def _with_splice_quiet(c: Cut, ctx: EdgeContext, db, fs: float, thr: float, grid: float) -> Cut:
    removed = removed_words(c, ctx.words)
    before, after = _kept_neighbours(c, removed, ctx)
    target = ctx.splice_quiet_clause
    if not before or _boundary(before[-1]) != "clause" or target <= 0:
        return c

    hi_start, lo_end = _inward_limits(c, removed, ctx)
    cap = math.ceil(target / fs) + 1
    i_start, i_end = int(round(c.start / fs)), int(round(c.end / fs))
    # Quiet already on the kept side of each edge, and the quiet the cut could give back.
    have_l = _quiet_run(db, i_start - 1, -1, thr, cap) * fs
    have_r = _quiet_run(db, i_end, 1, thr, cap) * fs
    room_l = min(_quiet_run(db, i_start, 1, thr, cap) * fs, max(0.0, hi_start - c.start))
    room_r = min(_quiet_run(db, i_end - 1, -1, thr, cap) * fs, max(0.0, c.end - lo_end)) if after else 0.0
    avail_l, avail_r = have_l + room_l, have_r + room_r
    if not after:
        have_r = avail_r = 0.0

    want_l = min(SPLICE_QUIET_LEFT_SHARE * target, avail_l)
    want_r = min(target - want_l, avail_r)
    want_l = min(target - want_r, avail_l)     # a short side hands its share to the other
    slack = 0.5 / grid + fs                    # a shortfall this small is frame rounding, not missing quiet
    give_l = min(want_l - have_l, room_l) if want_l - have_l > slack else 0.0
    give_r = min(want_r - have_r, room_r) if want_r - have_r > slack else 0.0
    spare = c.length - ctx.min_cut
    if give_l + give_r > spare:                # never let the cut collapse
        give_r = max(0.0, min(give_r, spare - give_l))
        give_l = max(0.0, min(give_l, spare))

    start, end = c.start, c.end
    if give_l:
        start = min(max(c.start, round((c.start + give_l) * grid) / grid),
                    math.floor((c.start + room_l) * grid + 1e-6) / grid)
    if give_r:
        end = max(min(c.end, round((c.end - give_r) * grid) / grid),
                  math.ceil((c.end - room_r) * grid - 1e-6) / grid)
    moved = Cut(start, end, c.reason, c.text, dict(c.provenance))
    same = [id(w) for w in removed_words(moved, ctx.words)] == [id(w) for w in removed]
    return moved if same and moved.length >= ctx.min_cut else c


def _with_lead_in(c: Cut, ctx: EdgeContext) -> Cut:
    """`c` with its end moved earlier, so kept speech after it starts at least
    `splice_lead_in` past the cut end.

    Only quiet frames (at or below floor + quiet_above_floor_db) are handed back.
    The end never moves back past the cut start or a removed word, and stops at
    the first voiced frame when the quiet runs out first. It lands on the frame
    grid at or before the target, so the later snap keeps the lead-in.
    """
    lead = ctx.splice_lead_in
    db, fs = ctx.env.db, ctx.env.frame_sec
    if lead <= 0 or len(db) == 0:
        return c
    removed = removed_words(c, ctx.words)
    _, after = _kept_neighbours(c, removed, ctx)
    if not after or not 0 <= after[0].start - c.end < lead:
        return c
    lo = max([c.start] + [w.end for w in removed])
    if lo >= c.end:
        return c
    thr = ctx.floor + ctx.quiet_above_floor_db
    grid = ctx.fps or 1.0 / fs
    j = min(len(db), math.ceil(c.end / fs - 1e-9)) - 1
    while j >= 0 and j * fs >= lo and db[j] <= thr:
        j -= 1
    quiet_from = math.ceil(max(lo, (j + 1) * fs) * grid - 1e-6) / grid
    target = math.floor((after[0].start - lead) * grid + 1e-6) / grid
    end = min(c.end, max(target, quiet_from))
    if end >= c.end or end - c.start < ctx.min_cut:
        return c
    moved = Cut(c.start, end, c.reason, c.text, dict(c.provenance))
    same = [id(w) for w in removed_words(moved, ctx.words)] == [id(w) for w in removed]
    return moved if same else c


def keep_splice_lead_in(cuts: list[Cut], ctx: EdgeContext) -> list[Cut]:
    """Every cut's kept speech gets its lead-in; see _with_lead_in."""
    return [_with_lead_in(c, ctx) for c in cuts]


def _speech_between(a: float, b: float, ctx: EdgeContext) -> bool:
    """Is there kept speech between these two cuts? Words, plus the voiced audio
    no word covers: fusing across that would delete the span the guard keeps.
    A protected interval blocks the fusion whenever any part of it lies between."""
    def hit(s: float, e: float) -> bool:
        return a < (s + e) / 2 < b

    if any(hit(w.start, w.end) for w in ctx.words):
        return True
    if any(s < b and e > a for s, e in ctx.protected_spans):
        return True
    return ctx.voice_guard and any(hit(s, e) for s, e in ctx.voices)


def flow_guard(cuts: list[Cut], ctx: EdgeContext) -> list[Cut]:
    out: list[Cut] = []
    for c in sorted(cuts, key=lambda c: (c.start, c.end)):
        if out and c.start - out[-1].end < ctx.flow_min_kept and not _speech_between(out[-1].end, c.start, ctx):
            fuse_into(out[-1], c, by_rank=False)   # a judged reason wins over a mechanical one (timeline.fuse_into)
        else:
            out.append(Cut(c.start, c.end, c.reason, c.text, dict(c.provenance)))
    return out


def judged_removes(start: float, end: float, w) -> bool:
    """Whether a judged cut over [start, end) removes the word `w` (any object with start and end).

    The test is coverage, the same one `removed_words` uses, and verify's judged-leak check uses it too. A midpoint
    test lets a cut that only grazes a word's middle take the whole word: a broken-off fragment ("impl-") ends just
    past the middle of the token it sits in ("imple-implement"), and the midpoint rule then removed the kept restart.
    """
    if w.end <= w.start:
        return start <= w.start <= end
    return _overlap(start, end, w.start, w.end) / (w.end - w.start) >= REMOVE_COVERAGE


def held_words(raw: list[Cut], words: list[RefinedWord]) -> list[Cut]:
    """Each word a judged cut removes, as a cut over the word's own extent, taken from the judges' raw decisions.

    A word is removed by a judged cut when the cut covers it (`judged_removes`): the same test the
    verify check uses. Degenerate (zero-length) words carry no audio and are not held.
    """
    judged = [c for c in raw if c.reason in JUDGED_REASONS]
    out: list[Cut] = []
    for w in words:
        if w.end <= w.start:
            continue
        for c in judged:
            if judged_removes(c.start, c.end, w):
                out.append(Cut(w.start, w.end, c.reason, c.text, dict(c.provenance)))
                break
    return out


def hold_judged(cuts: list[Cut], held: list[Cut], duration: float, fps: float | None = None) -> list[Cut]:
    """Judged decisions win over the edge pass: every word a judged cut removes stays removed.

    A word counts as removed while a cut covers it (`judged_removes`: the same test as `held_words` and the
    verify check). The edge steps move a judged cut's edges through quiet and kept-word pads, and a guard
    or cross-talk rule can pull an edge in past a removed word's coverage. Each word that is left
    uncovered is put back as a cut of its own extent, on the frame grid when there is one (outward, so
    the cut still reaches the word). Kept audio beside it is trimmed by that much. A word still covered
    keeps the edge the valley search chose, so mid-word valley edges are unchanged.
    """
    spans = [(c.start, c.end) for c in merge_cuts(cuts, duration)]
    starts = [a for a, _ in spans]
    missing: list[Cut] = []
    for h in held:
        mid = (h.start + h.end) / 2
        i = bisect_right(starts, mid) - 1
        if i < 0 or not spans[i][0] <= mid < spans[i][1]:
            a, b = h.start, h.end
            if fps:
                a, b = math.floor(a * fps + 1e-6) / fps, math.ceil(b * fps - 1e-6) / fps
            missing.append(Cut(a, b, h.reason, h.text, dict(h.provenance)))
    return merge_cuts(cuts + missing, duration) if missing else cuts


def snap_to_frames(cuts: list[Cut], fps: float) -> list[Cut]:
    """Shrink each cut onto the frame grid: start up, end down."""
    out: list[Cut] = []
    for c in cuts:
        start = math.ceil(c.start * fps - 1e-6) / fps
        end = math.floor(c.end * fps + 1e-6) / fps
        if end - start > MIN_CUT:
            out.append(Cut(start, end, c.reason, c.text, dict(c.provenance)))
    return out


def _grid_outward(t: float, grid: float | None, bound: float, *, later: bool) -> float:
    """`t` moved onto the frame grid, later (end edge) or earlier (start edge), never past `bound`."""
    if not grid:
        return t
    g = (math.ceil(t * grid - 1e-6) if later else math.floor(t * grid + 1e-6)) / grid
    if later and g > bound:
        g = math.floor(bound * grid + 1e-6) / grid
    if not later and g < bound:
        g = math.ceil(bound * grid - 1e-6) / grid
    return g


def _claim_voiced_run(c: Cut, ctx: EdgeContext) -> Cut:
    """Walk each voiced edge of a cut outward over the voiced run beside it, stopping at quiet or the kept-word bound."""
    db, fs = ctx.env.db, ctx.env.frame_sec
    thr = ctx.floor + ctx.quiet_above_floor_db
    left, right = _kept_bounds(c, ctx)
    start, end = c.start, c.end
    if edge_state(ctx.env, ctx.floor, start, "left")["voiced"]:
        j = math.ceil(start / fs - 1e-9) - 1
        while j >= 0 and db[j] > thr and j * fs >= left:
            j -= 1
        start = min(start, _grid_outward(max(left, (j + 1) * fs), ctx.fps, left, later=False))
    if edge_state(ctx.env, ctx.floor, end, "right")["voiced"]:
        j = int(end / fs)
        while j < len(db) and db[j] > thr and j * fs <= right:
            j += 1
        end = max(end, _grid_outward(min(right, j * fs), ctx.fps, right, later=True))
    return Cut(start, end, c.reason, c.text, dict(c.provenance))


def _is_filler_cut(c: Cut) -> bool:
    """A filler cut: a filler the transcript named, or a filler the output pass confirmed on the render's transcript.

    The output pass finds fillers the source transcript missed, so its voiced edges are settled exactly like a
    source filler's: a judged source-side polish cut keeps the edge pass's own placement.
    """
    if c.reason == "filler":
        return True
    return c.reason == "polish" and c.provenance.get("pass") == "output" and c.provenance.get("kind") == "filler"


def settle_output_cuts(cuts: list[Cut], ctx: EdgeContext) -> list[Cut]:
    """Output-pass cuts keep their confirmed span whole, then settle outward onto quiet audio.

    The output pass confirmed the span on the render's transcript, where the sound is audible even when no source word
    names it, so the edge chain may not shrink it. The span goes back in full, clamped to the kept neighbours (the cut
    never reaches into a kept word), and each edge then walks outward over the voiced audio beside it until quiet.
    """
    out: list[Cut] = []
    for c in cuts:
        if not _is_output_cut(c):
            out.append(c)
            continue
        s0, s1 = c.provenance.get("span", (c.start, c.end))
        left, right = _kept_bounds(Cut(s0, s1, c.reason, c.text, dict(c.provenance)), ctx)
        start = _grid_outward(max(left, min(c.start, s0)), ctx.fps, left, later=False)
        end = _grid_outward(min(right, max(c.end, s1)), ctx.fps, right, later=True)
        if end - start < ctx.min_cut:
            out.append(c)   # the neighbours leave no room for the span: keep what the edge chain placed
            continue
        settled = Cut(start, end, c.reason, c.text, dict(c.provenance))
        out.append(_claim_voiced_run(settled, ctx) if len(ctx.env.db) else settled)
    return out


def settle_filler_edges(cuts: list[Cut], ctx: EdgeContext) -> list[Cut]:
    """Filler cuts whose edge still sits on voiced audio: take the voiced run into the cut.

    The valley pass can settle a filler's edge on a dip inside the filler's own
    voiced burst, leaving a syllable of it in the kept audio. The run beside each
    voiced edge is walked into the cut, out to quiet or to the kept-word bound.
    Only when `filler_keep_voiced_max` is above 0 is a short filler still on
    voiced audio after that dropped instead: the kept word beside it owns that
    voice. The default keeps every filler cut. Other word-cut reasons remove
    words on purpose.
    """
    if len(ctx.env.db) == 0:
        return list(cuts)
    out: list[Cut] = []
    for c in cuts:
        if not _is_filler_cut(c):
            out.append(c)
            continue
        settled = _claim_voiced_run(c, ctx)
        still_voiced = (edge_state(ctx.env, ctx.floor, settled.start, "left")["voiced"]
                        or edge_state(ctx.env, ctx.floor, settled.end, "right")["voiced"])
        if still_voiced and c.length <= ctx.filler_keep_voiced_max:
            ctx.dropped_fillers += 1
            continue
        out.append(settled)
    return out


def _cut_speaker(c: Cut, words: list[RefinedWord]) -> str | None:
    """Speaker of the labelled word a cut removes that it overlaps most; None when it removes none."""
    best, speaker = 0.0, None
    for w in removed_words(c, words):
        if w.speaker is None:
            continue
        overlap = _overlap(c.start, c.end, w.start, w.end)
        if speaker is None or overlap > best:
            best, speaker = overlap, w.speaker
    return speaker


def _subtract(start: float, end: float, holes: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """[start, end) minus sorted, disjoint `holes`: the pieces left."""
    pieces: list[tuple[float, float]] = []
    cur = start
    for s, e in holes:
        if e <= cur:
            continue
        if s >= end:
            break
        if s > cur:
            pieces.append((cur, s))
        cur = max(cur, e)
    if cur < end:
        pieces.append((cur, end))
    return pieces


# A mechanical cut's target speaker, carried in its provenance so the guard's second run judges it as placed.
SPEAKER_KEY = "speaker"


def spare_other_speaker(cuts: list[Cut], ctx: EdgeContext) -> list[Cut]:
    """A mechanical cut never removes or clips words said by someone other than its own target.

    Another speaker's words are holes in the cut, padded as a kept word is; what remains is kept
    when it is still at least min_cut long. Cuts that touch no such word keep their values, so a
    single-speaker recording is untouched.
    """
    out: list[Cut] = []
    for c in cuts:
        if c.reason not in MECHANICAL_REASONS:
            out.append(c)
            continue
        target = c.provenance[SPEAKER_KEY] if SPEAKER_KEY in c.provenance else _cut_speaker(c, ctx.words)
        prov = dict(c.provenance)
        if target is not None:
            prov[SPEAKER_KEY] = target
        pad = _pad(c.reason, ctx)
        holes = _coalesce([
            (w.start - pad, w.end + pad) for w in ctx.words if id(w) not in ctx.decided_words
            if w.speaker is not None and w.speaker != target and w.end > w.start
            and w.end + pad > c.start and w.start - pad < c.end
        ])
        if not holes:
            out.append(Cut(c.start, c.end, c.reason, c.text, prov))
            continue
        ctx.spared_other_speaker += 1
        for a, b in _subtract(c.start, c.end, holes):
            if ctx.fps:
                a = math.ceil(a * ctx.fps - 1e-6) / ctx.fps
                b = math.floor(b * ctx.fps + 1e-6) / ctx.fps
            if b - a >= ctx.min_cut:
                out.append(Cut(a, b, c.reason, c.text, dict(prov)))
    return out


def _fit_placed(placed: list[Cut], others: list[Cut], ctx: EdgeContext) -> list[Cut]:
    """Placed cuts trimmed around the other cuts, then snapped inward to whole frames."""
    holes = [(c.start, c.end) for c in others]
    out = [Cut(a, b, c.reason, c.text, c.provenance) for c in placed for a, b in _subtract(c.start, c.end, holes)]
    out = [c for c in out if c.length >= ctx.min_cut]
    if ctx.fps:
        out = snap_to_frames(out, ctx.fps)
    return [c for c in out if c.length >= ctx.min_cut]


def _is_token_fragment(c: Cut, words: list[RefinedWord]) -> bool:
    """An output-pass restart that is a broken-off part of one restart-shaped source token.

    The render shows the fragment ("impl-") but the source has the whole token ("imple-implement"), so the cut
    covers only part of that token's audio. Like a placed restart it keeps its span: the token's other half is speech.
    """
    if c.provenance.get("pass") != "output" or c.provenance.get("kind") != "restart":
        return False
    return any(w.start - 1e-6 <= c.start and c.end <= w.end + 1e-6 and _split_restart(w.text) is not None
               for w in words)


def _window(spans: list[tuple[float, float]], starts: list[float], longest: float, lo: float,
           hi: float) -> list[tuple[float, float]]:
    """The sorted spans overlapping [lo, hi): `starts` are the spans' starts, `longest` the longest span."""
    out: list[tuple[float, float]] = []
    i = bisect_left(starts, lo - longest)
    while i < len(spans) and spans[i][0] < hi:
        if spans[i][1] > lo:
            out.append(spans[i])
        i += 1
    return out


def _decided_removed(cuts: list[Cut], words: list[RefinedWord]) -> set[int]:
    """Ids of the words the stage decisions remove: covered by a word-removing cut (judged_removes)."""
    ordered = sorted(words, key=lambda w: w.start)
    starts = [w.start for w in ordered]
    longest = max((w.end - w.start for w in ordered), default=0.0)
    decided: set[int] = set()
    for c in cuts:
        if c.reason not in WORD_CUT_REASONS | JUDGED_REASONS:
            continue
        for w in ordered[bisect_left(starts, c.start - longest):bisect_right(starts, c.end)]:
            if judged_removes(c.start, c.end, w):
                decided.add(id(w))
    return decided


def _guarded(lo: float, hi: float, guard: list[float]) -> bool:
    """Whether a guarded point (a decided word's midpoint; `guard` sorted) lies in [lo, hi): a cut covering it stops
    covering it if its edge moves across lo or to hi."""
    i = bisect_left(guard, lo)
    return i < len(guard) and guard[i] < hi


def _midpoints(words: list[RefinedWord]) -> list[float]:
    return sorted((w.start + w.end) / 2 for w in words)


def _after(b: float, pad: float, guard: list[float]) -> float:
    """Where audio after a kept word ending at b may start: b plus pad, or b itself when the pad would uncover a
    guarded point (a decided word's midpoint stays removed)."""
    return b if _guarded(b, b + pad, guard) else b + pad


def _before(a: float, pad: float, guard: list[float]) -> float:
    """Where audio before a kept word starting at a must end: a minus pad, or a itself when the pad would uncover a
    guarded point."""
    return a if _guarded(a - pad, a, guard) else a - pad


def _split_around(pieces: list[tuple[float, float]], a: float, b: float, pad: float,
                  guard: list[float]) -> list[tuple[float, float]]:
    """`pieces` with the kept word [a, b) taken out: each edge that lands inside it moves to its far side plus pad
    (see _after and _before), and a piece that covers it is split in two."""
    out: list[tuple[float, float]] = []
    for s, e in pieces:
        if e <= s or e <= a or s >= b:
            if e > s:
                out.append((s, e))
        elif s < a and e > b:
            out += [(s, _before(a, pad, guard)), (_after(b, pad, guard), e)]
        elif s >= a and e <= b:
            continue
        elif s < a:
            out.append((s, _before(a, pad, guard)))
        else:
            out.append((_after(b, pad, guard), e))
    return out


def _uncovers(old: Cut, new: Cut, guard: list[float]) -> bool:
    """Whether moving `old` to `new` uncovers a guarded point (a decided word's midpoint) that `old` covered."""
    lost = [(old.start, min(old.end, new.start)), (max(old.start, new.end), old.end)]
    return any(_guarded(lo, hi, guard) for lo, hi in lost if hi > lo)


def _shrink(c: Cut, fps: float) -> Cut | None:
    """`c` shrunk onto the frame grid: a start rounds up and an end rounds down, so the cut never reaches the kept audio
    beside it. None when no frame is left."""
    start = math.ceil(c.start * fps - 1e-6) / fps
    end = math.floor(c.end * fps + 1e-6) / fps
    if end - start <= MIN_CUT:
        return None
    return Cut(start, end, c.reason, c.text, dict(c.provenance))


def _merged_spans(cuts: list[Cut]) -> list[tuple[float, float]]:
    """The sorted, merged spans the non-restart cuts cover."""
    out: list[tuple[float, float]] = []
    for s, e in sorted((c.start, c.end) for c in cuts if not is_intra_restart(c)):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _covers(spans: list[tuple[float, float]], t: float) -> bool:
    """Whether a merged, sorted span list covers instant t."""
    i = bisect_right([s for s, _ in spans], t) - 1
    return i >= 0 and spans[i][0] <= t < spans[i][1]


def _carve(cuts: list[Cut], a: float, b: float, fps: float) -> list[Cut]:
    """`cuts` with [a, b) taken out on the frame grid: a piece left of it ends at the frame at or before a,
    a piece right of it starts at the frame at or after b, so neither reaches the span."""
    lo = math.floor(a * fps + 1e-6) / fps
    hi = math.ceil(b * fps - 1e-6) / fps
    out: list[Cut] = []
    for c in cuts:
        if c.end <= a or c.start >= b:
            out.append(c)
            continue
        if c.start < lo:
            out.append(Cut(c.start, lo, c.reason, c.text, dict(c.provenance)))
        if c.end > hi:
            out.append(Cut(hi, c.end, c.reason, c.text, dict(c.provenance)))
    return out


def _gaps(lo: float, hi: float, spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The parts of [lo, hi) that the sorted, merged spans do not cover."""
    out: list[tuple[float, float]] = []
    cur = lo
    for s, e in spans:
        if e <= cur:
            continue
        if s >= hi:
            break
        if s > cur:
            out.append((cur, s))
        cur = max(cur, e)
    if cur < hi:
        out.append((cur, hi))
    return out


def _unsettled(w: RefinedWord, wanted: list[tuple[float, float]], spans: list[tuple[float, float]],
                is_decided: bool) -> bool:
    """Whether a word needs the settle step: a judged midpoint left uncovered (any word a judged span covers, decided or
    not), or, for a decided word, a kept piece of at least FRAGMENT_MIN_SEC inside the word's refined bounds while part
    of the word stays covered (verify's fragment rule, counted per side)."""
    mid = (w.start + w.end) / 2
    if _covers(wanted, mid) and not _covers(spans, mid):
        return True
    if not is_decided:
        return False
    gaps = _gaps(w.start, w.end, spans)
    kept = sum(q - p for p, q in gaps)
    return kept < 0.9 * (w.end - w.start) and any(q - p >= FRAGMENT_MIN_SEC for p, q in gaps)


def settle_grid(cuts: list[Cut], ctx: EdgeContext, decided: set[int],
                judged: list[Cut]) -> tuple[list[Cut], list[RefinedWord]]:
    """The last edge step: every cut edge on the frame grid, the grid the render snaps to, so the render moves nothing.

    Each cut shrinks onto the grid (see _shrink), which never reaches kept audio. Then a decided word is checked: a
    judged midpoint (`judged`, as the edge pass received it) must stay covered, and a removed word must not keep a
    fragment of FRAGMENT_MIN_SEC or more. Either is fixed by a grid cut over the word's extent, unless a kept word lies
    in that extent; then the word is given back whole. A carve that uncovers another such word gives it back the same
    way. Words the take-removed step gave back keep that give-back. Intra-token restarts keep their span."""
    fps = ctx.fps
    wanted = _merged_spans(judged)
    taken = {(round(a, 6), round(b, 6)) for a, b in ctx.given_back}
    kept = sorted(ext for ext in (word_extent(w) for w in ctx.words if id(w) not in decided) if ext[1] > ext[0])
    kstarts = [a for a, _ in kept]
    kmax = max((b - a for a, b in kept), default=0.0)
    intra = [c for c in cuts if is_intra_restart(c)]
    cover = [g for c in cuts if not is_intra_restart(c) for g in [_shrink(c, fps)] if g is not None]
    given: list[RefinedWord] = []
    added: set[int] = set()
    while True:
        result = merge_cuts(cover + intra, ctx.duration)
        for w in given:
            a, b = word_extent(w)
            result = _carve(result, a, b, fps)
        spans = _merged_spans(result)
        given_ids = {id(g) for g in given}
        missing = [w for w in ctx.words if w.end > w.start and id(w) not in given_ids
                   and (round(w.start, 6), round(w.end, 6)) not in taken
                   and _unsettled(w, wanted, spans, id(w) in decided)]
        if not missing:
            return merge_cuts(result, ctx.duration), given
        for w in missing:
            mid = (w.start + w.end) / 2
            a, b = word_extent(w)
            a, b = math.floor(a * fps + 1e-6) / fps, math.ceil(b * fps - 1e-6) / fps
            if id(w) in added or _window(kept, kstarts, kmax, a, b):
                decided.discard(id(w))
                given.append(w)
                continue
            src = next((c for c in judged + cuts if c.start <= mid < c.end), None)
            reason, text, prov = (src.reason, src.text, dict(src.provenance)) if src else ("polish", w.text, {})
            cover.append(Cut(a, b, reason, text, prov))
            added.add(id(w))


def keep_words_whole(cuts: list[Cut], ctx: EdgeContext, decided: set[int]) -> list[Cut]:
    """Last edge step: no cut edge lies inside a kept word.

    A kept word is one the stage decisions did not remove; its extent is word_extent (the ASR stamp widened by the
    refined bounds), the extent verify's clipped_words measures. A cut that takes part of one gives that part back: an
    edge moves to the word's boundary on the cut's side plus the cut's pad, and a cut that covers the word is split
    around it. An edge never moves past a decided word's midpoint, so decided words stay removed: a pad that would
    uncover one stops at the kept boundary. Mid-word valley edges on removed words are left alone (hold_judged), and
    intra-token restarts keep their span: they cut inside a kept token on purpose.
    """
    kept = sorted(ext for ext in (word_extent(w) for w in ctx.words if id(w) not in decided) if ext[1] > ext[0])
    if not kept:
        return cuts
    guard = _midpoints([w for w in ctx.words if id(w) in decided])
    kstarts = [a for a, _ in kept]
    kmax = max(b - a for a, b in kept)
    out: list[Cut] = []
    for c in cuts:
        if is_intra_restart(c) or _is_token_fragment(c, ctx.words):
            out.append(c)
            continue
        s, e = c.start, c.end
        near = _window(kept, kstarts, kmax, s, e)
        pad = _pad(c.reason, ctx)
        pieces = [(s, e)]
        for _ in range(4):
            before = pieces
            for a, b in near:
                pieces = _split_around(pieces, a, b, pad, guard)
            if pieces == before:
                break
        for x, y in pieces:
            if y - x > ctx.min_cut:
                out.append(Cut(x, y, c.reason, c.text, dict(c.provenance)))
    return out


# A removed word whose audio is still partly in the output (a fragment) is settled by moving the cut edge that leaves
# it: to the quietest point of a real silence beside it, or to the valley between it and the touching kept word. A
# word with no such boundary is given back whole (its cut edge is pulled to the word's far edge).
FRAGMENT_MIN_SEC = 0.04
FRAGMENT_SHARE = 0.10
# The clipped_words tolerance: a boundary may take at most this share of the kept word beside it.
KEPT_CLIP_SHARE = 0.10
# A real silence this long between a removed and a kept word: its quietest point takes the edge.
BOUNDARY_GAP_SEC = 0.06
# Touching words: the boundary is searched this far past each word's ASR stamp.
BOUNDARY_MARGIN = 0.03
# A touching boundary needs this much level contrast (dB) in its window: flat audio has no valley.
VALLEY_CONTRAST_DB = 6.0


def _uncovered(a: float, b: float, spans: list[tuple[float, float]], starts: list[float],
               longest: float) -> list[tuple[float, float]]:
    """The parts of [a, b) that no span covers; `spans` sorted by start, `starts` their starts."""
    out: list[tuple[float, float]] = []
    cur = a
    i = bisect_left(starts, a - longest)
    while i < len(spans) and spans[i][0] < b:
        s, e = spans[i]
        if e > cur:
            if s > cur:
                out.append((cur, min(s, b)))
            cur = e
        i += 1
    if cur < b:
        out.append((cur, b))
    return [(p, q) for p, q in out if q > p]


def uncovered_pieces(a: float, b: float, spans: list[tuple[float, float]], starts: list[float],
                     longest: float) -> list[tuple[float, float]]:
    """The parts of [a, b) that no cut covers (see _uncovered)."""
    return _uncovered(a, b, spans, starts, longest)


def fragment_pieces(a: float, b: float, spans: list[tuple[float, float]], starts: list[float],
                    longest: float) -> list[tuple[float, float]]:
    """The uncovered pieces of [a, b) that are fragments: each at least FRAGMENT_MIN_SEC and FRAGMENT_SHARE of the word.

    One side at a time. A splice margin (splice_pad on each side, under FRAGMENT_MIN_SEC) is not a scrap."""
    return [(p, q) for p, q in _uncovered(a, b, spans, starts, longest)
            if q - p >= FRAGMENT_MIN_SEC and q - p > FRAGMENT_SHARE * (b - a)]


def _quietest(ctx: EdgeContext, lo: float, hi: float, *, late: bool) -> tuple[float, float] | None:
    """(time, contrast dB) of the envelope minimum over [lo, hi]; among equal minima the latest (or earliest) frame."""
    db = ctx.env.db
    if len(db) == 0 or hi < lo:
        return None
    fs = ctx.env.frame_sec
    i0 = max(0, min(len(db) - 1, int(lo / fs)))
    i1 = max(i0, min(len(db) - 1, int(hi / fs)))
    seg = db[i0:i1 + 1]
    low = float(seg.min())
    ties = np.flatnonzero(seg <= low)
    k = int(ties[-1] if late else ties[0])
    return (i0 + k) * fs, float(seg.max()) - low


def _grid(t: float, fps: float | None, *, up: bool) -> float:
    """`t` on the frame grid: up for a start edge, down for an end edge (the snap_to_frames convention)."""
    if not fps:
        return t
    return (math.ceil(t * fps - 1e-6) if up else math.floor(t * fps + 1e-6)) / fps


OUTWARD_SEC = 0.5
# Provenance flag on a cut the fragment step placed: refine_joins leaves its edges where the step put them.
SETTLED_KEY = "settled_by_edge_pass"


def _quiet_outside(lo: float, hi: float, ctx: EdgeContext, *, late: bool) -> float | None:
    """A quiet point in [lo, hi] for a cut edge beside a removed word with no kept word on that side, else None.

    The quietest frame (the latest among ties when `late`, nearest the word) must lie within quiet_above_floor_db
    of the noise floor.
    """
    lo, hi = max(0.0, lo), min(ctx.duration, hi)
    found = _quietest(ctx, lo, hi, late=late)
    if found is None:
        return None
    t = found[0]
    if float(ctx.env.db[min(len(ctx.env.db) - 1, int(t / ctx.env.frame_sec))]) > ctx.floor + ctx.quiet_above_floor_db:
        return None
    return _grid(t, ctx.fps, up=not late)


def _overlap_edge(hit: list, p: float, q: float, *, tail: bool, speaker) -> tuple[str, float | None]:
    """A kept word overlaps the uncovered fragment [p, q) of a removed word.

    The edge moves to the removed word's own edge (p for a head cut, q for a tail cut) when the kept word's refined
    audio is clear of the removed word and the overlap is within KEPT_CLIP_SHARE of the kept word's extent.
    Another speaker's word, or no clean boundary, gives the word back.
    """
    if any(k.speaker != speaker for k in hit):
        return "fail", None
    for k in hit:
        a, b = word_extent(k)
        if tail and (k.start < q - 1e-3 or q - a > KEPT_CLIP_SHARE * (b - a)):
            return "fail", None
        if not tail and (k.end > p + 1e-3 or b - p > KEPT_CLIP_SHARE * (b - a)):
            return "fail", None
    return "overlap", (q if tail else p)


def _tail_edge(p: float, r: RefinedWord, kept: list, kstarts: list[float],
               ctx: EdgeContext, pad: float) -> tuple[str, float | None]:
    """(kind, t) for a cut whose edge sits at p inside removed word r, given the kept word that follows r.

    kind "quiet" when no kept word lies on this side, "overlap" or "gap"/"touch" for a boundary; kind "fail" when
    the boundary is not clear or would clip too much (t is None).
    """
    q = word_extent(r)[1]
    hit = [k for a, b, k in kept if a < q and b > p]
    if hit:
        return _overlap_edge(hit, p, q, tail=True, speaker=r.speaker)
    j = bisect_left(kstarts, q - 1e-9)
    if j >= len(kept):
        t = _quiet_outside(q, q + OUTWARD_SEC, ctx, late=False)
        return "quiet", (t if t is not None else min(ctx.duration, q + pad))
    ka, kb, k = kept[j]
    if ka - q >= BOUNDARY_GAP_SEC:
        kind = "gap"
        found = _quietest(ctx, q, ka - ctx.env.frame_sec, late=False)
    else:
        kind = "touch"
        found = _quietest(ctx, max(p, r.asr_end - BOUNDARY_MARGIN), k.asr_start + BOUNDARY_MARGIN, late=True)
        if found is None or found[1] < VALLEY_CONTRAST_DB or max(0.0, found[0] - ka) / (kb - ka) > KEPT_CLIP_SHARE:
            return "fail", None
    if found is None:
        return "fail", None
    t = _grid(found[0], ctx.fps, up=False)
    return (kind, t) if t > p else ("fail", None)


def _head_edge(q: float, r: RefinedWord, kept: list, kends: list[float],
               ctx: EdgeContext, pad: float) -> tuple[str, float | None]:
    """(kind, t) for a cut whose edge sits at q inside removed word r, given the kept word that precedes r."""
    p = word_extent(r)[0]
    hit = [k for a, b, k in kept if a < q and b > p]
    if hit:
        return _overlap_edge(hit, p, q, tail=False, speaker=r.speaker)
    j = bisect_right(kends, p + 1e-9) - 1
    if j < 0:
        t = _quiet_outside(p - OUTWARD_SEC, p, ctx, late=True)
        return "quiet", (t if t is not None else max(0.0, p - pad))
    ka, kb, k = kept[j]
    if p - kb >= BOUNDARY_GAP_SEC:
        kind = "gap"
        found = _quietest(ctx, kb + ctx.env.frame_sec, p, late=True)
    else:
        kind = "touch"
        found = _quietest(ctx, k.asr_end - BOUNDARY_MARGIN, min(q, r.asr_start + BOUNDARY_MARGIN), late=False)
        if found is None or found[1] < VALLEY_CONTRAST_DB or max(0.0, kb - found[0]) / (kb - ka) > KEPT_CLIP_SHARE:
            return "fail", None
    if found is None:
        return "fail", None
    t = _grid(found[0], ctx.fps, up=True)
    return (kind, t) if t < q else ("fail", None)


def _give_back(c: Cut, r: RefinedWord, tail: bool, ctx: EdgeContext) -> Cut | None:
    """The cut with its edge pulled off removed word r, so r stays whole; None when nothing of the cut is left."""
    a, b = word_extent(r)
    start, end = (c.start, min(c.end, a)) if tail else (max(c.start, b), c.end)
    if end - start <= ctx.min_cut:
        return None
    return Cut(start, end, c.reason, c.text, {**c.provenance, SETTLED_KEY: True})


def take_removed_whole(cuts: list[Cut], ctx: EdgeContext, decided: set[int]) -> list[Cut]:
    """Settles the fragments of removed words (see the constants above). Sets ctx.fragments and ctx.given_back.

    A fragment is a removed word with at least FRAGMENT_SHARE and FRAGMENT_MIN_SEC still in the output, beside a kept
    word it touches. Its cut edge moves to a boundary found in the audio. A boundary that is not clear, that would clip
    the kept word beyond KEPT_CLIP_SHARE, or that leaves a fragment, gives the word back. A word with no kept word on
    that side moves its edge to a quiet point outside it. Mid-word valleys are not this step's.
    """
    stats = {"gap": 0, "touch": 0, "given_back": 0, "uncut": 0, "quiet": 0, "overlap": 0}
    kept = sorted(((a, b, w) for w in ctx.words if id(w) not in decided for a, b in [word_extent(w)] if b > a),
                  key=lambda t: t[0])
    kstarts = [a for a, _, _ in kept]
    kept_by_end = sorted(kept, key=lambda t: t[1])
    kends = [b for _, b, _ in kept_by_end]
    removed = sorted((w for w in ctx.words if id(w) in decided and word_extent(w)[1] > word_extent(w)[0]),
                     key=lambda w: word_extent(w)[0])
    guard = _midpoints([w for w in ctx.words if id(w) in decided])
    cuts_l: list[Cut | None] = list(cuts)
    given: list[tuple[float, float]] = []

    def cover() -> tuple[list[tuple[float, float]], list[float], float]:
        spans = sorted((c.start, c.end) for c in cuts_l if c is not None and not is_intra_restart(c)
                       and not _is_token_fragment(c, ctx.words))
        return spans, [s for s, _ in spans], max((e - s for s, e in spans), default=0.0)

    spans, starts, longest = cover()
    for r in removed:
        a, b = word_extent(r)
        for _ in range(4):
            frags = fragment_pieces(a, b, spans, starts, longest)
            if not frags:
                break
            p, q = frags[0]
            settleable = [i for i, c in enumerate(cuts_l) if c is not None and not is_intra_restart(c)
                          and not _is_token_fragment(c, ctx.words) and c.reason not in PLACED_REASONS]
            i_tail = next((i for i in settleable if abs(cuts_l[i].end - p) < 1e-6 and cuts_l[i].start < p), None)
            i_head = next((i for i in settleable if abs(cuts_l[i].start - q) < 1e-6 and cuts_l[i].end > q), None)
            if i_tail is None and i_head is None:
                stats["uncut"] += 1
                ctx.decisions.append({"word": r.text, "start": round(a, 3), "end": round(b, 3), "side": "none",
                                      "frag": [round(p, 3), round(q, 3)], "outcome": "uncut"})
                break
            tail = i_tail is not None
            i = i_tail if tail else i_head
            c = cuts_l[i]
            pad = _pad(c.reason, ctx)
            kind, t = _tail_edge(p, r, kept, kstarts, ctx, pad) if tail else _head_edge(q, r, kept, kends, ctx, pad)
            if t is not None:
                moved = (Cut(c.start, t, c.reason, c.text, {**c.provenance, SETTLED_KEY: True}) if tail
                         else Cut(t, c.end, c.reason, c.text, {**c.provenance, SETTLED_KEY: True}))
                cuts_l[i] = moved
                spans, starts, longest = cover()
                if not fragment_pieces(a, b, spans, starts, longest) and not _uncovers(c, moved, guard):
                    stats[kind] += 1
                    ctx.decisions.append({"word": r.text, "start": round(a, 3), "end": round(b, 3), "side": "tail" if tail else "head",
                                          "frag": [round(p, 3), round(q, 3)], "outcome": kind, "edge": round(t, 3)})
                    break
            back = _give_back(c, r, tail, ctx)
            others = [m for m in guard if abs(m - (r.start + r.end) / 2) > 1e-9]
            lost = _guarded(c.start, c.end, others) if back is None else _uncovers(c, back, others)
            if lost:
                # Giving the word back would uncover a decided neighbour's midpoint: the fragment stays (verify warns).
                cuts_l[i] = c
                spans, starts, longest = cover()
                stats["uncut"] += 1
                break
            cuts_l[i] = back
            given.append((round(r.start, 6), round(r.end, 6)))
            stats["given_back"] += 1
            ctx.decisions.append({"word": r.text, "start": round(a, 3), "end": round(b, 3), "side": "tail" if tail else "head",
                                  "frag": [round(p, 3), round(q, 3)], "outcome": "given", "reason_kind": kind})
            spans, starts, longest = cover()
            break
    # A decided word that keeps (nearly) all its audio is a fallback when no other speaker's word overlaps it: the
    # cut was lost (keep_words_whole, or a give-back already counted). Recorded, so the count matches verify's.
    spans, starts, longest = cover()
    for r in removed:
        a, b = word_extent(r)
        key = (round(r.start, 6), round(r.end, 6))
        if key in given or b <= a:
            continue
        if sum(q2 - p2 for p2, q2 in _uncovered(a, b, spans, starts, longest)) < 0.9 * (b - a):
            continue
        given.append(key)
        stats["given_back"] += 1
        ctx.decisions.append({"word": r.text, "start": round(a, 3), "end": round(b, 3), "side": "none",
                              "frag": [round(a, 3), round(b, 3)], "outcome": "given"})
    ctx.fragments = stats
    ctx.given_back = given
    return [c for c in cuts_l if c is not None]


def _clip_conflicts(decided: set[int], ctx: EdgeContext) -> list[RefinedWord]:
    """Removes from `decided` each word whose midpoint lies inside a kept word's extent, and returns those words.
    Such a word cannot be cut without clipping that kept word, so it stays whole: the owner-approved give-back.
    Repeats, since a word kept now can hold another decided midpoint."""
    out: list[RefinedWord] = []
    while True:
        kept = sorted(ext for ext in (word_extent(w) for w in ctx.words if id(w) not in decided) if ext[1] > ext[0])
        starts = [a for a, _ in kept]
        longest = max((b - a for a, b in kept), default=0.0)
        found = [w for w in ctx.words if id(w) in decided and w.end > w.start
                 and _window(kept, starts, longest, (w.start + w.end) / 2, (w.start + w.end) / 2 + 1e-9)]
        if not found:
            return out
        for w in found:
            decided.discard(id(w))
        out += found


def finalize_cuts(cuts: list[Cut], ctx: EdgeContext) -> list[Cut]:
    ctx.given_back = []
    ctx.decisions = []
    decided = _decided_removed(cuts, ctx.words)
    judged_in = [c for c in cuts if c.reason in JUDGED_REASONS and not is_intra_restart(c)]
    conflicts = _clip_conflicts(decided, ctx)
    ctx.decided_words = decided
    ctx.dropped_fillers = 0
    ctx.spared_other_speaker = 0
    placed = [c for c in cuts if _is_placed(c) or _is_token_fragment(c, ctx.words)]
    cuts = [c for c in cuts if not _is_placed(c)]
    cuts = mark_asr_guard(cuts, [c for c in placed if c.reason in PLACED_REASONS])
    held = held_words(cuts, ctx.words)
    cuts = merge_cuts(cuts, ctx.duration)
    cuts = spare_other_speaker(cuts, ctx)
    cuts = protect_words(cuts, ctx)
    cuts = steer_out_of_moans(cuts, ctx)
    cuts = sweep_to_quiet(cuts, ctx)
    cuts = [c for c in cuts if c.length >= ctx.min_cut]   # a sliver must not seed a fusion
    cuts = snap_to_valleys(cuts, ctx)
    cuts = tighten_repairs(cuts, ctx)
    cuts = keep_sentence_tail(cuts, ctx)
    cuts = keep_splice_quiet(cuts, ctx)
    cuts = keep_splice_lead_in(cuts, ctx)
    cuts = flow_guard(cuts, ctx)
    if ctx.fps:
        cuts = snap_to_frames(cuts, ctx.fps)
    cuts = hold_judged(cuts, held, ctx.duration, ctx.fps)
    # The held words can leave a kept sliver between two cuts (a few frames of a removed word's tail):
    # fuse again, as flow_guard would have had the words been removed in the first place.
    cuts = flow_guard(cuts, ctx)
    cuts = settle_output_cuts(cuts, ctx)
    cuts = settle_filler_edges(cuts, ctx)
    cuts = spare_other_speaker(cuts, ctx)
    cuts = merge_cuts(cuts, ctx.duration)
    out = merge_cuts(cuts + _fit_placed(placed, cuts, ctx), ctx.duration)
    out = keep_words_whole(out, ctx, decided)
    out = merge_cuts(take_removed_whole(out, ctx, decided), ctx.duration)
    given_grid: list[RefinedWord] = []
    if ctx.fps:
        # Last edge step: the render snaps every edge to the frame grid, so the cut edges must already be on it.
        out, given_grid = settle_grid(out, ctx, decided, judged_in)
    ctx.given_back = [(round(w.start, 6), round(w.end, 6)) for w in conflicts + given_grid] + ctx.given_back
    return merge_cuts(out, ctx.duration)


REFINED_JOINS = "refined_joins"


def _is_refinable(plan: Plan, prev: Segment, seg: Segment) -> bool:
    """A join between two forward-running segments of one recording."""
    return plan.sources[prev.source].sha256 == plan.sources[seg.source].sha256 and seg.start > prev.end


def _join_key(source: str, prev_end: float, next_start: float) -> tuple[str, float, float]:
    """A join's identity: its source and edge pair in ms, as serialised."""
    return (source, round(prev_end, 3), round(next_start, 3))


def _leaves_fragment(c: Cut, removed: list[RefinedWord]) -> bool:
    """Whether cut `c` leaves a fragment (see fragment_pieces) of a removed word in the output."""
    for r in removed:
        a, b = word_extent(r)
        if fragment_pieces(a, b, [(c.start, c.end)], [c.start], c.end - c.start):
            return True
    return False


def refine_joins(plan: Plan, ctx: EdgeContext, margin: float = JOIN_MOVE_MARGIN,
                 window: float = JOIN_MOVE_WINDOW) -> tuple[Plan, list[dict]]:
    """Re-place each same-recording join of an existing plan (hand-authored EDLs).

    A join is the gap between two consecutive segments of one recording. Its
    two edges are searched within `window` of their current position (kept
    small so pauses keep their length) under the bounds and cost of
    snap_to_valleys, so no kept word or voiced audio is cut and no removed word
    is kept. A join moves only when its cost falls by more than `margin`: an
    edge the author placed well stays where it is. Segment order, sources and
    count are unchanged. Returns the plan and one record per join. Joins already
    recorded in `provenance["refined_joins"]` are skipped, so refining a refined
    plan moves nothing; a join edited since has a new edge pair and is refined.
    The incoming edge then gets the splice lead-in (keep_splice_lead_in), whether or not the join moved.
    A backward join (a reordered plan, see stages.restructure.reorder) has no cut to move and is skipped.
    """
    segments = list(plan.segments)
    cuts = list(plan.cuts)
    done = {_join_key(e["source"], e["prev_end"], e["next_start"])
            for e in plan.provenance.get(REFINED_JOINS, [])}
    records: list[dict] = []
    for i in range(1, len(segments)):
        prev, seg = segments[i - 1], segments[i]
        rec: dict = {"join": i, "old": [round(prev.end, 4), round(seg.start, 4)], "moved": False}
        records.append(rec)
        if not _is_refinable(plan, prev, seg):
            rec["skipped"] = "not a forward join within one recording"
            continue
        if _join_key(seg.source, prev.end, seg.start) in done:
            rec["skipped"] = "already refined"
            continue
        k = next((k for k, c in enumerate(cuts)
                  if abs(c.start - prev.end) <= EDGE_CUT_MATCH_SEC and abs(c.end - seg.start) <= EDGE_CUT_MATCH_SEC),
                 None)
        source_cut = cuts[k] if k is not None else Cut(prev.end, seg.start, "content")
        if k is not None and source_cut.provenance.get(SETTLED_KEY):
            rec["skipped"] = "settled by the fragment step"
            continue
        if k is not None and _is_output_cut(source_cut):
            # Already settled whole onto quiet audio inside its neighbours (settle_output_cuts); a join move would shrink it.
            rec["skipped"] = "output-pass cut already settled"
            continue
        c = Cut(prev.end, seg.start, source_cut.reason, source_cut.text, dict(source_cut.provenance))
        removed = removed_words(c, _near(c, ctx))
        left, right = _kept_bounds(c, ctx)
        left = max(left, prev.start + ctx.min_cut)
        right = min(right, seg.end - ctx.min_cut)
        hi_start, lo_end = _inward_limits(c, removed, ctx)
        here = _join_cost(ctx, _edge_terms(ctx, c.start, "start"), _edge_terms(ctx, c.end, "end"))
        pairs = _pairs(ctx, c, _grid_points(ctx, c.start, left, hi_start, window),
                       _grid_points(ctx, c.end, lo_end, right, window))
        found = _first_valid(c, removed, ctx, sorted(pairs, key=lambda p: (p[0], p[1])))
        rec["cost_before"] = round(here, 3)
        rec["cost_after"] = None if found is None else round(found[0], 3)
        target = found[1] if found is not None and found[0] < here - margin else c
        if target is not c and _leaves_fragment(target, removed):
            target = c
        moved = _with_lead_in(target, ctx)
        if ctx.fps:
            # Joins stay on the frame grid (the render snaps to it): a shrunk edge that would uncover a removed word
            # keeps the join's original, grid-placed cut.
            grid = _shrink(moved, ctx.fps)
            moved = grid if grid is not None and not _uncovers(moved, grid, _midpoints(removed)) else c
        if (moved.start, moved.end) != (c.start, c.end):
            segments[i - 1] = Segment(prev.source, prev.start, moved.start)
            segments[i] = Segment(seg.source, moved.end, seg.end)
            if k is not None:
                cuts[k] = moved
            rec.update(moved=True, new=[round(moved.start, 4), round(moved.end, 4)])
    for rec in records:
        rec["t"] = round(sum(s.length for s in segments[: rec["join"]]), 4)
    refined = [{"source": seg.source, "prev_end": round(prev.end, 3), "next_start": round(seg.start, 3)}
               for prev, seg in zip(segments, segments[1:]) if _is_refinable(plan, prev, seg)]
    provenance = {**plan.provenance, REFINED_JOINS: refined}
    if ctx.fps and segments:
        # The last segment ends at its source's end, rarely on the frame grid: floor it (at most one frame of tail goes)
        # so the render's snap moves nothing.
        last = segments[-1]
        if last.end >= plan.sources[last.source].duration - 1e-9:
            floored = math.floor(last.end * ctx.fps + 1e-6) / ctx.fps
            if floored > last.start:
                segments[-1] = Segment(last.source, last.start, floored)
    return replace(plan, segments=segments, cuts=cuts, provenance=provenance), records
