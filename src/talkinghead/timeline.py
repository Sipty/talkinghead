"""Cuts on the source timeline, the EDL derived from them, and time maps.

Everything here is pure: no I/O, no ffmpeg, no models. A `Plan` is the
pipeline's deliverable and the only thing the renderer reads.

Invariants:
- Cut and Segment times are seconds on their SOURCE's timeline, half-open.
- `Plan.segments` is the EDL: what the viewer sees, in order. For a plan in
  source order (every plan without restructure's reorder) it is exactly the
  complement of `merge_cuts(plan.cuts)`. A reordered plan (restructure moved
  topic blocks to the front) keeps the same cuts, and its segments are the
  kept audio in output order: each source instant appears once, so as a set
  they still equal the complement, but the list is not monotone in source
  time. Other sources (a future intro, a sizzle reel) are appended by later
  stages.
- Serialisation is byte-stable: sorted keys, millisecond precision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .types import JUDGED_REASONS

SCHEMA_VERSION = 1
# Spans shorter than this are noise, not a cut (mirrors the renderer's floor).
MIN_CUT = 0.02
# Cuts closer than this fuse: two joins this close read as one to the ear and
# would leave an unrenderable sliver between them.
FUSE_GAP = 0.0
# When cuts fuse, the most informative reason survives: a structural decision
# over a word-level judgment over a mechanical trim. Between equal ranks a judged
# reason beats a mechanical one (a restart fused with a tangent is still a tangent).
REASON_RANK = {"cold-open": 3, "outro": 3, "retake": 2, "content": 2, "restart": 2, "join": 2,
               "tangent": 2, "lead-in": 2, "polish": 2, "filler": 1, "noise": 1, "silence": 0, "gap": 0}


@dataclass
class SourceInfo:
    id: str
    path: str
    sha256: str
    duration: float


@dataclass
class Cut:
    """A span of the main source to remove, and why."""

    start: float
    end: float
    reason: str
    text: str = ""
    provenance: dict = field(default_factory=dict)

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Segment:
    """One kept span of one source, in output order."""

    source: str
    start: float
    end: float

    @property
    def length(self) -> float:
        return self.end - self.start


def _union_span(last: Cut, c: Cut) -> list[float] | None:
    """The union of the two cuts' confirmed spans, when either carries one (an output-pass cut's `span`).

    A fused cut keeps the extent of every cut it absorbed: the edge pass re-settles a confirmed span from its
    recorded `span`, so a span that dropped a fused cut would cut back the words that cut removed.
    """
    a, b = last.provenance.get("span"), c.provenance.get("span")
    if a is None and b is None:
        return None
    a = a or (last.start, last.end)
    b = b or (c.start, c.end)
    return [min(a[0], b[0]), max(a[1], b[1])]


def fuse_into(last: Cut, c: Cut, *, by_rank: bool) -> None:
    """Fold `c` into `last`, which starts no later than `c`: the span grows to cover both, the winning
    reason survives with its provenance (its confirmed span widened to cover `c`'s too), and the texts are joined.

    `by_rank` (merge_cuts): the higher REASON_RANK wins, and between equal ranks a judged reason wins.
    Otherwise (flow_guard, which fuses across quiet): the first cut's reason stands unless `c` is judged
    and `last` is not, so a judged decision is never lost to a mechanical cut that comes first, which
    is what let the speaker guard shrink a tangent. Mechanical fusions keep their historical label.
    """
    span = _union_span(last, c)
    last.end = max(last.end, c.end)
    if by_rank:
        wins = (REASON_RANK.get(c.reason, 1), c.reason in JUDGED_REASONS) > \
            (REASON_RANK.get(last.reason, 1), last.reason in JUDGED_REASONS)
    else:
        wins = c.reason in JUDGED_REASONS and last.reason not in JUDGED_REASONS
    if wins:
        last.reason = c.reason
        last.provenance = {**last.provenance, **c.provenance}
    if span is not None:
        last.provenance = {**last.provenance, "span": span}
    if c.text and c.text not in last.text:
        last.text = f"{last.text} {c.text}".strip()


def merge_cuts(cuts: list[Cut], duration: float, fuse_gap: float = FUSE_GAP) -> list[Cut]:
    """Clip to [0, duration], drop tiny spans, fuse overlaps and touches.

    On a fuse the winning reason (`fuse_rank`) survives and texts are joined so
    the report still shows what was removed.
    """
    clipped: list[Cut] = []
    for c in sorted(cuts, key=lambda c: (c.start, c.end)):
        start, end = max(0.0, c.start), min(duration, c.end)
        if end - start > MIN_CUT:
            clipped.append(Cut(start, end, c.reason, c.text, dict(c.provenance)))
    merged: list[Cut] = []
    for c in clipped:
        if merged and c.start <= merged[-1].end + fuse_gap:
            fuse_into(merged[-1], c, by_rank=True)
        else:
            merged.append(c)
    return merged


def cuts_to_segments(cuts: list[Cut], source: SourceInfo) -> list[Segment]:
    """The complement of the merged cuts within the source."""
    segments: list[Segment] = []
    cursor = 0.0
    for c in merge_cuts(cuts, source.duration):
        if c.start - cursor > MIN_CUT:
            segments.append(Segment(source.id, cursor, c.start))
        cursor = max(cursor, c.end)
    if source.duration - cursor > MIN_CUT:
        segments.append(Segment(source.id, cursor, source.duration))
    return segments


def in_source_order(segments: list[Segment]) -> bool:
    """True when consecutive segments of one source run forward in source time.

    A reordered plan (a block moved ahead of earlier material) jumps backwards
    at some seam; anything that reads the picture or the transcript in one
    forward pass must refuse it.
    """
    return all(b.source != a.source or b.start >= a.end for a, b in zip(segments, segments[1:]))


@dataclass
class Plan:
    sources: dict[str, SourceInfo]
    main: str
    cuts: list[Cut]
    segments: list[Segment]
    anchors: dict[str, float] = field(default_factory=dict)
    decisions: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    # The isolated audio the render reads for the main source (isolate.py's record); empty when isolation is off.
    isolation: dict = field(default_factory=dict)

    @classmethod
    def from_cuts(cls, source: SourceInfo, cuts: list[Cut]) -> "Plan":
        cuts = merge_cuts(cuts, source.duration)
        return cls(
            sources={source.id: source},
            main=source.id,
            cuts=cuts,
            segments=cuts_to_segments(cuts, source),
        )

    # -- time maps -----------------------------------------------------------

    @property
    def output_duration(self) -> float:
        return sum(s.length for s in self.segments)

    def source_to_output(self, t: float, source: str | None = None) -> float | None:
        """Output time of a source instant; None when it was cut."""
        source = source or self.main
        offset = 0.0
        for s in self.segments:
            if s.source == source and s.start <= t < s.end:
                return offset + (t - s.start)
            offset += s.length
        return None

    def output_to_source(self, t: float) -> float | None:
        """Source instant playing at output time t (None past the end).

        At a join, the source time of the segment that STARTS there is
        returned, which is the join's cut start on the source."""
        offset = 0.0
        for s in self.segments:
            if offset <= t < offset + s.length:
                return s.start + (t - offset)
            offset += s.length
        return None

    def joins(self) -> list[float]:
        """Output times where two segments meet (head and tail trims are not joins)."""
        out: list[float] = []
        offset = 0.0
        for i, s in enumerate(self.segments):
            if i > 0:
                out.append(offset)
            offset += s.length
        return out

    # -- serialisation -------------------------------------------------------

    def to_json(self) -> str:
        def ms(x: float) -> float:
            return round(x, 3)

        payload = {
            "schema_version": self.schema_version,
            "main": self.main,
            "sources": {
                k: {"id": v.id, "path": v.path, "sha256": v.sha256, "duration": ms(v.duration)}
                for k, v in self.sources.items()
            },
            "cuts": [
                {"start": ms(c.start), "end": ms(c.end), "reason": c.reason, "text": c.text,
                 "provenance": c.provenance}
                for c in self.cuts
            ],
            "segments": [
                {"source": s.source, "start": ms(s.start), "end": ms(s.end)} for s in self.segments
            ],
            "anchors": {k: ms(v) for k, v in self.anchors.items()},
            "decisions": self.decisions,
            "provenance": self.provenance,
        }
        if self.isolation:
            # Absent when off, so a plan without isolation serialises exactly as before.
            payload["isolation"] = self.isolation
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "Plan":
        d = json.loads(text)
        if d.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported plan schema {d.get('schema_version')!r}")
        return cls(
            sources={k: SourceInfo(**v) for k, v in d["sources"].items()},
            main=d["main"],
            cuts=[Cut(**c) for c in d["cuts"]],
            segments=[Segment(**s) for s in d["segments"]],
            anchors=dict(d.get("anchors", {})),
            decisions=dict(d.get("decisions", {})),
            provenance=dict(d.get("provenance", {})),
            schema_version=d["schema_version"],
            isolation=dict(d.get("isolation", {})),
        )
