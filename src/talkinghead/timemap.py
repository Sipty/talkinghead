"""Time maps over the segments the renderer actually cuts.

Animations and head-clip checks need to find a frame before `final.mp4` exists. The plan's own
`source_to_output` is unsnapped, so it can disagree with the picture by up to half a frame at every
cut. These pure functions work on the frame-snapped segments (the same cut the picture and the audio
master use), so their answers agree with the render to the frame.
"""
from __future__ import annotations

from talkinghead.render import _snap
from talkinghead.timeline import Plan, Segment

# Seconds below which two instants are the same one (guards float noise in cumulative sums).
_EPS = 1e-9
# Output pieces and source coverage are compared and reported at microsecond precision.
_DECIMALS = 6


def snapped_segments(plan: Plan, fps: float) -> list[Segment]:
    """The cut the picture and the audio master use: edges on the frame grid, or the plan's own at fps 0."""
    return _snap(plan.segments, fps) if fps else list(plan.segments)


def _starts(segments: list[Segment]) -> list[float]:
    """Output time at which each segment starts."""
    out, at = [], 0.0
    for s in segments:
        out.append(at)
        at += s.length
    return out


def _total(segments: list[Segment]) -> float:
    return sum(s.length for s in segments)


def locate(plan: Plan, t: float, *, fps: float) -> tuple[str, float] | None:
    """(source id, source seconds) playing at output time `t`; at a join the segment that starts there.

    None when `t` is before the output or at or past its end.
    """
    segments = snapped_segments(plan, fps)
    if t < 0 or t >= _total(segments) - _EPS:
        return None
    best = None
    for seg, at in zip(segments, _starts(segments)):
        if at <= t + _EPS:
            best = (seg, at)
        else:
            break
    if best is None:
        return None
    seg, at = best
    return seg.source, seg.start + max(0.0, t - at)


def map_span(plan: Plan, start: float, end: float, *, fps: float, source: str | None = None) -> dict:
    """Where source span `[start, end)` of `source` (default the main source) plays in the output.

    `kept` is one piece covering the whole span, `trimmed` one piece with part of the span cut away,
    `split` several pieces (an inner cut or a reorder), `cut` none. Pieces are in output order.
    """
    source = source or plan.main
    if source not in plan.sources:
        raise ValueError(f"unknown source {source!r}; the plan has: {', '.join(plan.sources)}")
    if end <= start:
        raise ValueError(f"span end {end} is not after start {start}; pass start < end")
    segments = snapped_segments(plan, fps)
    # (out_a, out_b, src_a, src_b), merged when consecutive in both output and source.
    pieces: list[list[float]] = []
    for seg, at in zip(segments, _starts(segments)):
        if seg.source != source:
            continue
        a, b = max(seg.start, start), min(seg.end, end)
        if b - a <= _EPS:
            continue
        piece = [at + (a - seg.start), at + (b - seg.start), a, b]
        if pieces and abs(pieces[-1][1] - piece[0]) <= _EPS and abs(pieces[-1][3] - piece[2]) <= _EPS:
            pieces[-1][1], pieces[-1][3] = piece[1], piece[3]
        else:
            pieces.append(piece)
    rounded = [[round(p[0], _DECIMALS), round(p[1], _DECIMALS)] for p in pieces]
    if not pieces:
        status = "cut"
    elif len(pieces) > 1:
        status = "split"
    elif (pieces[0][3] - pieces[0][2]) >= (end - start) - 1e-6:
        status = "kept"
    else:
        status = "trimmed"
    return {
        "status": status,
        "pieces": rounded,
        "out_start": rounded[0][0] if rounded else None,
        "out_end": rounded[-1][1] if rounded else None,
    }


def window_segments(plan: Plan, start: float, end: float, *, fps: float) -> list[Segment]:
    """The source spans that play during output `[start, end)`, in plan order.

    With `fps` the window is rounded to whole frames first, so the spans' lengths sum to the rounded
    window; an `end` past the output end is clamped.
    """
    segments = snapped_segments(plan, fps)
    total = _total(segments)
    if fps:
        start, end = round(start * fps) / fps, round(end * fps) / fps
    end = min(end, total)
    if end - start <= _EPS or start >= total - _EPS:
        raise ValueError(
            f"window {start:.6f}-{end:.6f} s is empty or past the output end ({total:.6f} s); "
            "pass a window inside the output"
        )
    out: list[Segment] = []
    for seg, at in zip(segments, _starts(segments)):
        a, b = max(at, start), min(at + seg.length, end)
        if b - a <= _EPS:
            continue
        out.append(Segment(seg.source, seg.start + (a - at), seg.start + (b - at)))
    return out
