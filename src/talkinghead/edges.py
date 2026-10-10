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
10. flow_guard      cuts with less than flow_min_kept of speechless audio
                    between them fuse (a rapid-cut pileup reads as a glitch)
11. snap_to_frames  edges onto the video frame grid, shrinking, so segments
                    are whole frames and audio/video cannot drift
12. merge

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
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .audio import Envelope, RefinedWord
from .derive import pause_boundary
from .joins import edge_state
from .timeline import MIN_CUT, Cut, merge_cuts

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
    # Filler cuts at most this long are dropped when an edge sits on voiced
    # audio; a longer "um" costs more to hear than the tight splice it leaves.
    filler_keep_voiced_max: float = 0.35
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


def _valley(ctx: EdgeContext, orig: float, lo: float, hi: float) -> float:
    """The quietest frame-grid position within the valley window of `orig` and
    inside [lo, hi]; ties go to the one nearest `orig`. `orig` when nothing fits."""
    grid = ctx.fps or 1.0 / ctx.env.frame_sec
    a = max(lo, orig - ctx.splice_valley_window)
    b = min(hi, orig + ctx.splice_valley_window)
    half = ctx.splice_crossfade_sec / 2
    scored: list[tuple[float, float, float]] = []   # (score, distance, t)
    for k in range(math.ceil(a * grid - 1e-6), math.floor(b * grid + 1e-6) + 1):
        t = k / grid
        if any(ms < t < me for ms, me in ctx.moans):
            continue
        scored.append((_dip_score(ctx.env.db, ctx.env.frame_sec, t, half), abs(t - orig), t))
    if not scored:
        return orig
    best = min(s for s, _, _ in scored)
    return min((d, t) for s, d, t in scored if s <= best + VALLEY_TIE_DB)[1]


def snap_to_valleys(cuts: list[Cut], ctx: EdgeContext) -> list[Cut]:
    """Word-removing cuts: move each edge to the quietest grid position nearby.

    Outward it stops splice_pad short of kept audio (so a kept word's voiced
    run is never entered); inward it stops where a removed word would drop
    below REMOVE_COVERAGE. The set of removed words is unchanged, else the cut
    is left as it was.
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
        start = _valley(ctx, c.start, left, hi_start)
        end = _valley(ctx, c.end, max(lo_end, start + ctx.min_cut), right)
        moved = Cut(start, end, c.reason, c.text, dict(c.provenance))
        same = [id(w) for w in removed_words(moved, ctx.words)] == [id(w) for w in removed]
        out.append(moved if same and end - start >= ctx.min_cut else c)
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


def drop_voiced_fillers(cuts: list[Cut], ctx: EdgeContext) -> list[Cut]:
    """Drop a short filler cut whose edge still sits on voiced audio.

    A tight splice around a micro-filler in continuous speech sounds worse than
    the filler; a long one (over `filler_keep_voiced_max`, or fused with a
    pause trim) is still worth removing. Same voiced-edge test as the join
    verifier (joins.edge_state), so the two cannot disagree. Other word-cut
    reasons remove words on purpose and keep their cuts.
    """
    if len(ctx.env.db) == 0:
        return list(cuts)
    out: list[Cut] = []
    for c in cuts:
        if (c.reason == "filler" and c.length <= ctx.filler_keep_voiced_max
                and (edge_state(ctx.env, ctx.floor, c.start, "left")["voiced"]
                     or edge_state(ctx.env, ctx.floor, c.end, "right")["voiced"])):
            ctx.dropped_fillers += 1
            continue
        out.append(c)
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
    cuts = flow_guard(cuts, ctx)
    if ctx.fps:
        cuts = snap_to_frames(cuts, ctx.fps)
    cuts = drop_voiced_fillers(cuts, ctx)
    return merge_cuts(cuts, ctx.duration)
