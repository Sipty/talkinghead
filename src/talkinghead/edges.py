"""Edge placement: one pass applied to every cut list before it becomes a plan.

Order matters and each step is a pure function:
1. merge            overlapping/touching cuts fuse
2. protect_words    no kept word is ever clipped (word-removing cuts snap to
                    whole words; silence/noise cuts retreat from any word)
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
13. settle_filler_edges  short filler cuts still on voiced audio: the voiced run is
                    walked into the cut, else the filler is dropped
14. merge

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
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from .audio import Envelope, RefinedWord
from .derive import pause_boundary
from .joins import EDGE_CUT_MATCH_SEC, EDGE_VOICED_ABOVE_FLOOR_DB, edge_state
from .timeline import MIN_CUT, Cut, Plan, Segment, merge_cuts

# Reasons whose spans are meant to remove words (edges snap to word bounds).
WORD_CUT_REASONS = frozenset({"retake", "restart", "filler", "content", "cold-open", "outro", "join"})
# Reasons whose whole job is removing NON-speech: for these, voiced audio the
# transcript does not cover may not be landed inside or swallowed by the cut.
# An editorial cut (above) is a decision about content, so the waveform only
# keeps its EDGES clear and what it removes is reported instead.
VOICE_SAFE_REASONS = frozenset({"silence", "noise"})
# A word is removed by a word-cut when at least this fraction of it is covered.
REMOVE_COVERAGE = 0.5
# Kept-quiet split of a splice target: this share follows the left kept word,
# the rest precedes the right one.
SPLICE_QUIET_LEFT_SHARE = 0.6
# Cuts that abandon words said wrongly and restate them: inside a sentence
# they join straight to the correction.
REPAIR_REASONS = frozenset({"retake", "restart"})
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


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _pad(reason: str, ctx: EdgeContext) -> float:
    """Distance an edge of this kind of cut keeps from kept audio."""
    return ctx.splice_pad if reason in WORD_CUT_REASONS else ctx.pad


def removed_words(cut: Cut, words: list[RefinedWord]) -> list[RefinedWord]:
    """Words this cut removes: none for silence/noise cuts, majority-covered ones otherwise."""
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


def _kept_spans(cut: Cut, ctx: EdgeContext, *, voices: bool = True) -> list[tuple[float, float]]:
    """The audio this cut keeps: the words it does not remove, the protected
    event intervals (whatever the cut's reason), plus (unless `voices` is off)
    the voiced audio no word covers.

    `voices` comes from the waveform, so a dropped word is covered too: the
    reference recording had a 0.16 s utterance inside a pause trim that no
    transcript entry named, and only the audio proved it was there.
    """
    removed = {id(w) for w in removed_words(cut, ctx.words)}
    spans = [(w.start, w.end) for w in ctx.words if id(w) not in removed]
    spans += [(s, e) for s, e in ctx.protected_spans if e > s]
    if voices and ctx.voice_guard:
        spans += ctx.voices
    return _coalesce(spans)


def _kept_bounds(cut: Cut, ctx: EdgeContext) -> tuple[float, float]:
    """(left, right): the furthest an edge may reach without touching kept audio."""
    left, right = 0.0, ctx.duration
    pad = _pad(cut.reason, ctx)
    for s, e in _kept_spans(cut, ctx):
        if e <= cut.start + 1e-9 or (s < cut.end and e > cut.start and s < cut.start):
            left = max(left, e + pad)
        if s >= cut.end - 1e-9 or (s < cut.end and e > cut.start and e > cut.end):
            right = min(right, s - pad)
    return left, right


def protect_words(cuts: list[Cut], ctx: EdgeContext) -> list[Cut]:
    out: list[Cut] = []
    for c in cuts:
        removed = removed_words(c, ctx.words)
        start, end = c.start, c.end
        if removed:
            start = min(start, min(w.start for w in removed))
            end = max(end, max(w.end for w in removed))
        # Every edge stays clear of kept audio; only a cut whose job is removing
        # non-speech may split around a voice the transcript never named (split
        # an editorial cut there and a fragment of the passage the judge chose
        # to remove ships instead).
        guards = _kept_spans(c, ctx)
        pad = _pad(c.reason, ctx)
        splits = _kept_spans(c, ctx, voices=c.reason in VOICE_SAFE_REASONS)
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
            last = out[-1]
            last.end = max(last.end, c.end)
            if c.text and c.text not in last.text:
                last.text = f"{last.text} {c.text}".strip()
        else:
            out.append(Cut(c.start, c.end, c.reason, c.text, dict(c.provenance)))
    return out


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
        if c.reason != "filler":
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


def finalize_cuts(cuts: list[Cut], ctx: EdgeContext) -> list[Cut]:
    ctx.dropped_fillers = 0
    cuts = merge_cuts(cuts, ctx.duration)
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
    cuts = settle_filler_edges(cuts, ctx)
    return merge_cuts(cuts, ctx.duration)


REFINED_JOINS = "refined_joins"


def _is_refinable(plan: Plan, prev: Segment, seg: Segment) -> bool:
    """A join between two forward-running segments of one recording."""
    return plan.sources[prev.source].sha256 == plan.sources[seg.source].sha256 and seg.start > prev.end


def _join_key(source: str, prev_end: float, next_start: float) -> tuple[str, float, float]:
    """A join's identity: its source and edge pair in ms, as serialised."""
    return (source, round(prev_end, 3), round(next_start, 3))


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
        moved = _with_lead_in(target, ctx)
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
    return replace(plan, segments=segments, cuts=cuts, provenance=provenance), records
