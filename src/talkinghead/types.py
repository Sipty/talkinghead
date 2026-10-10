"""Shared data model for cut plans.

This module is the contract between the pipeline stages:
    transcribe.py -> derive.py (+ noise.py) -> render.py
Do not change field names without updating every stage and the Hermes skill.
New optional fields must round-trip through to_json/from_json.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

SEMANTIC_REASONS: set[str] = {"redundant", "false-start", "self-talk", "digression"}

# Every reason an LLM pass may propose; review.py groups these for bulk toggles.
# no-value marks a whole passage with nothing in it for the viewer (audience
# meta-talk, filler that neither advances the topic nor builds curiosity),
# as opposed to ramble, which only covers value-DELAYING speech.
LLM_CUT_REASONS: set[str] = SEMANTIC_REASONS | {"hook-delay", "ramble", "no-value"}
# Waveform-mechanical reasons: cuts whose span should contain no words. When a
# fused/merged span removes real speech it must be relabeled with the LLM
# judgment that covers it, never left with one of these.
# noise (tongue smacks, breaths, throat clears) belongs here and NOT in
# LLM_CUT_REASONS: it is measured off the audio envelope, not judged.
MECHANICAL_REASONS: set[str] = {"silence", "filler", "restart", "noise"}


@dataclass
class CutSpan:
    """A time range (seconds, float) proposed for removal, and why."""

    start: float
    end: float
    # one of: silence | filler | restart | noise | redundant | false-start
    # | self-talk | digression | hook-delay | ramble | no-value
    reason: str
    # Transcript excerpt inside the span (empty for pure silence), for review.
    text: str = ""

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "reason": self.reason, "text": self.text}

    @classmethod
    def from_dict(cls, d: dict) -> CutSpan:
        start, end = float(d["start"]), float(d["end"])
        # json.loads accepts bare NaN/Infinity; a non-finite span would round-trip
        # into non-standard JSON and hand ffmpeg "nan" start times.
        if not (math.isfinite(start) and math.isfinite(end)):
            raise ValueError(f"cut span must have finite start/end, got {d}")
        return cls(start=start, end=end, reason=str(d["reason"]), text=str(d.get("text", "")))


@dataclass
class CutPlan:
    """Full reviewable plan: the source video and every proposed cut, in order."""

    video: str  # path as given on the command line
    duration: float  # source duration in seconds
    cuts: list[CutSpan] = field(default_factory=list)
    # LLM-proposed cuts rejected during validation (bad ids/reasons), kept for the review UI.
    dropped: list[dict] = field(default_factory=list)
    # Hook-pass judgment (verdict, mistakes, value line, rewrites), present
    # only on plans produced by the hookdoctor flow.
    hook: dict | None = None

    @property
    def cut_total(self) -> float:
        return sum(c.end - c.start for c in self.cuts)

    def to_json(self) -> str:
        payload = {
            "video": self.video,
            "duration": self.duration,
            "cuts": [c.to_dict() for c in self.cuts],
        }
        if self.dropped:
            payload["dropped"] = self.dropped
        if self.hook is not None:
            payload["hook"] = self.hook
        return json.dumps(payload, indent=2)

    @classmethod
    def from_json(cls, s: str) -> CutPlan:
        d = json.loads(s)
        return cls(
            video=d["video"],
            duration=float(d["duration"]),
            cuts=[CutSpan.from_dict(c) for c in d.get("cuts", [])],
            dropped=d.get("dropped", []),
            hook=d.get("hook"),
        )
