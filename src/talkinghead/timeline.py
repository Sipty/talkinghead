"""Cuts on the source timeline, the EDL derived from them, and time maps.

Everything here is pure: no I/O, no ffmpeg, no models. A `Plan` is the
pipeline's deliverable and the only thing the renderer reads.

Invariants:
- Cut and Segment times are seconds on their SOURCE's timeline, half-open.
- `Plan.segments` is the EDL: what the viewer sees, in order. For the main
  source it is exactly the complement of `merge_cuts(plan.cuts)`; other
  sources (a future intro, a sizzle reel) are appended by later stages.
- Serialisation is byte-stable: sorted keys, millisecond precision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

SCHEMA_VERSION = 1
# Spans shorter than this are noise, not a cut (mirrors the renderer's floor).
MIN_CUT = 0.02
# Cuts closer than this fuse: two joins this close read as one to the ear and
# would leave an unrenderable sliver between them.
FUSE_GAP = 0.0
# When cuts fuse, the most informative reason survives: a structural decision
# over a word-level judgment over a mechanical trim.
REASON_RANK = {"cold-open": 3, "outro": 3, "retake": 2, "content": 2, "restart": 2, "join": 2,
               "filler": 1, "noise": 1, "silence": 0}


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


def merge_cuts(cuts: list[Cut], duration: float, fuse_gap: float = FUSE_GAP) -> list[Cut]:
    """Clip to [0, duration], drop tiny spans, fuse overlaps and touches.

    On a fuse the highest-ranked reason (REASON_RANK) survives and texts are
    joined so the report still shows what was removed.
    """
    clipped: list[Cut] = []
    for c in sorted(cuts, key=lambda c: (c.start, c.end)):
        start, end = max(0.0, c.start), min(duration, c.end)
        if end - start > MIN_CUT:
            clipped.append(Cut(start, end, c.reason, c.text, dict(c.provenance)))
    merged: list[Cut] = []
    for c in clipped:
        if merged and c.start <= merged[-1].end + fuse_gap:
            last = merged[-1]
            last.end = max(last.end, c.end)
            if REASON_RANK.get(c.reason, 1) > REASON_RANK.get(last.reason, 1):
                last.reason = c.reason
                last.provenance = {**last.provenance, **c.provenance}
            if c.text and c.text not in last.text:
                last.text = f"{last.text} {c.text}".strip()
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
        )
