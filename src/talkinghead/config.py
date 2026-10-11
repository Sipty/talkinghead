"""Run configuration. Defaults are the values chosen by ear on the 240 s
reference sample. Overrides come from a TOML file (`--config` or
TALKINGHEAD_CONFIG), the environment (TALKINGHEAD_JUDGE, TALKINGHEAD_TRANSCRIBER)
and the command line; the effective values are written into the plan's provenance."""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path

# The judge and the transcriber have no built-in default: the environment names them.
ENV_JUDGE = "TALKINGHEAD_JUDGE"
ENV_TRANSCRIBER = "TALKINGHEAD_TRANSCRIBER"
ENV_CONFIG = "TALKINGHEAD_CONFIG"

# Corner name -> ffmpeg overlay expressions for (x, y), with `{m}` the margin.
OVERLAY_CORNERS: dict[str, tuple[str, str]] = {
    "top-right": ("W-w-{m}", "{m}"),
    "top-left": ("{m}", "{m}"),
    "bottom-right": ("W-w-{m}", "H-h-{m}"),
    "bottom-left": ("{m}", "H-h-{m}"),
}


@dataclass
class Config:
    # Pauses: interior pauses longer than pause_above are trimmed; leading and
    # trailing silence goes outright. The kept length grows with the raw pause,
    # keep = pause_keep + pause_keep_ratio * (raw - pause_above), capped by the
    # boundary the pause sits on, so a topic change keeps a longer beat than a
    # mid-sentence hesitation. 0.30 s at the trigger is the amount chosen by ear;
    # the caps sit near the speaker's own p75-p90 natural pauses.
    pause_above: float = 0.35
    pause_keep: float = 0.30
    pause_keep_ratio: float = 0.25
    pause_cap_mid: float = 0.40
    pause_cap_clause: float = 0.45
    pause_cap_sentence: float = 0.55
    pause_cap_topic: float = 0.80
    # A raw pause at least this long after a sentence end counts as a topic change.
    pause_topic_above: float = 1.5
    # Waveform arbitration: a silence/noise cut may not land inside, or swallow,
    # voiced speech-level audio no transcript word covers.
    voice_guard: bool = True
    # Re-transcribe each such span with the other backend, longest first, capped.
    # Evidence for the report; the guard that keeps the audio is acoustic.
    speech_probe: bool = True
    speech_probe_max: int = 24
    # Fillers (uh, um, hmm...) shorter than this are kept as natural speech.
    # 0 removes every filler the transcript names; 0.15 restores the old floor.
    min_filler_dur: float = 0.0
    # Word-removing cuts (retake, filler, content, join, ...): each edge snaps
    # to the quietest envelope frame within splice_valley_window of its word
    # boundary (never inward of a removed word) and stays splice_pad clear of
    # kept voiced audio. A retake inside a sentence then joins straight to the
    # correction. After a completed sentence the cut leaves at least
    # splice_quiet_sentence of real source audio behind the last kept word;
    # after a clause end the other cuts keep real source quiet up to
    # splice_quiet_clause.
    splice_pad: float = 0.03
    splice_valley_window: float = 0.12
    splice_quiet_sentence: float = 0.08
    splice_quiet_clause: float = 0.18
    # Filler cuts at most this long are dropped (the filler stays) when an edge
    # still sits on voiced audio after valley snapping. 0 keeps every filler
    # cut; 0.35 restores the old behaviour of sparing short voiced fillers.
    filler_keep_voiced_max: float = 0.0
    # Joins are equal-power crossfades this long; each side overlaps the cut
    # by half of it, so the rendered audio length still equals the plan's.
    splice_crossfade_sec: float = 0.020
    # Cut edges are swept outward until the envelope sits within this of the floor.
    quiet_above_floor_db: float = 6.0
    # Cuts closer than this fuse (two joins this close read as a glitch).
    flow_min_kept: float = 2.0
    # Cuts shorter than this are dropped: a join costs more than they save.
    min_cut_sec: float = 0.08
    # A gap this long between words ends a sentence even without punctuation.
    sentence_split_pause: float = 0.5
    # Structural decisions
    cold_open_window_sec: float = 300.0
    outro_window_sec: float = 120.0
    # Retakes: a single abandoned attempt longer than this is not a stumble.
    retake_max_sec: float = 30.0
    # Content cuts
    content_enabled: bool = True
    content_max_cut_sec: float = 30.0
    content_max_fraction: float = 0.10
    # Voting
    votes: int = 5
    agree: int = 4
    # Per-sample wall clock for the shell judge; a hung model answer must not
    # kill a long cook, but sol occasionally needs well over 30 min on big
    # whole-transcript prompts before it streams a schema reply.
    judge_timeout: float = 3600.0
    # Pass B (marker adjudication) votes on a small window, so fewer samples
    # buy the same confidence than the whole-transcript sweep needs.
    marker_votes: int = 3
    marker_agree: int = 2
    # Adapters, required at run time: no default. "claude" routes through Claude
    # Code; "shell:<command>" bridges any CLI (prompt on stdin, JSON on stdout).
    # Resolved from --judge, TALKINGHEAD_JUDGE, then the config file.
    judge: str | None = None
    # Applied only when given: a shell judge's --model token is rewritten to it
    # (judge.with_model). None leaves a shell command as written and Claude Code on its own model.
    model: str | None = None
    # A second shell judge for the rest of the run once the primary reports a
    # usage limit. Off by default: a limited judge stops the run rather than
    # degrading the vote. Used verbatim, so it keeps pointing at its own model.
    fallback_command: str | None = None
    # Extra judges sharing the vote: every k-th sample goes to panel[i % n].
    # Shell commands used verbatim (they name their own --model, so samples
    # stay attributable in the cache). For retake-heavy recordings, widen the
    # panel so one model's blind spot cannot own the vote.
    panel: list[str] | None = None
    effort: str | None = None  # Claude Code only; the shell bridge ignores it
    transcriber: str | None = None  # "scribe" | "parakeet" | "whisper"; no default
    guidelines: str = "editorial/guidelines.md"
    # Render: output frame rate; cut edges snap to this grid.
    fps: int = 30
    crf: int = 18
    preset: str = "medium"
    demo_scale: str = "-2:480"
    demo_crf: int = 28
    demo_preset: str = "veryfast"
    loudness_lufs: float = -16.0
    # Content overlays include quote cards and researched visuals; roadmap
    # cards remain controlled independently by `plan_enabled`.
    visuals_content_enabled: bool = True
    visuals_enabled: bool = True
    visuals_research: bool = True
    # Which corner the overlays take. Top-right suits a framing where
    # the RODE boom owns bottom-right and his hands gesture into frame-left,
    # but the framing changes between recordings: a cook whose head sits in the
    # top-right corner passes a different one.
    visuals_corner: str = "top-right"
    # Cards are the mascot with a speech bubble, so they need more room than a
    # plain plate. The canvas is fixed and the bubble is right-anchored inside
    # it, so a short title still renders compact -- this is the ceiling, not
    # the width every card takes.
    visuals_scale_card: float = 0.46
    visuals_scale_shot: float = 0.42
    # Square PNG or JPEG drawn as the mascot on quote cards and the plan card.
    # None renders cards without one.
    visuals_avatar: str | None = None
    visuals_margin: int = 40
    visuals_fade: float = 0.4
    # 1.5 s was not long enough to notice a card, never mind read it. A card
    # also gets at least its reading time (cards.reading_seconds).
    visuals_min_hold: float = 3.0
    visuals_min_gap: float = 2.0
    # Below this match score an anchor gets a fresh asset instead of the
    # closest old one: reuse must be earned, never a fudge.
    visuals_match_threshold: float = 0.70
    # The plan card: a fully readable roadmap. It returns at each chapter and
    # highlights only the current row.
    plan_enabled: bool = True
    plan_chapters: int = 6
    # Retained for compatibility with older config files; the full plan is now
    # always readable and the first chapter is highlighted at the opening.
    plan_revealed_at_open: int = 1
    plan_hold: float = 9.0
    # The opening card belongs at the START, over the hook -- not in the last
    # free gap before chapter one, which on this cut was 3:35 deep. It also
    # holds longer, because it is the first thing the viewer has to scan.
    plan_open_at: float = 2.0
    plan_open_hold: float = 12.0
    # How far ahead of a chapter's own visual the reveal is allowed to sit. The
    # corner is contested at most chapter marks, so a reveal takes the last
    # free gap inside this window rather than fighting for the same instant.
    plan_lead_sec: float = 26.0
    assets_dir: str = "editorial/assets"
    # Input guard
    av_mismatch_tolerance: float = 1.0
    # Alignment of an external recording onto the video master (`--audio`).
    # Measured on the coarse cross-correlation's prominence: how far the
    # winning peak stands above the best alternative more than 1 s away. Below
    # ~3x the match is weak (see align.AlignResult), and a wrong offset unsyncs
    # the entire cut silently, so the run stops; `--allow-bad-align` cuts
    # regardless. A `None` confidence -- the two tracks share no signal -- is
    # refused whatever this is set to.
    align_min_confidence: float = 3.0

    def validate(self) -> "Config":
        if self.agree > self.votes:
            raise ValueError(f"agree ({self.agree}) cannot exceed votes ({self.votes})")
        if self.marker_agree > self.marker_votes:
            raise ValueError(
                f"marker_agree ({self.marker_agree}) cannot exceed marker_votes ({self.marker_votes})")
        if self.pause_keep >= self.pause_above:
            raise ValueError("pause_keep must be shorter than pause_above")
        if not 0.0 <= self.pause_keep_ratio <= 1.0:
            raise ValueError("pause_keep_ratio must be within [0, 1]")
        caps = (self.pause_cap_mid, self.pause_cap_clause, self.pause_cap_sentence, self.pause_cap_topic)
        if any(c < self.pause_keep for c in caps):
            raise ValueError("every pause_cap_* must be at least pause_keep")
        lengths = (self.splice_crossfade_sec, self.splice_pad, self.splice_valley_window,
                   self.splice_quiet_sentence, self.splice_quiet_clause)
        if any(v < 0 for v in lengths):
            raise ValueError("splice_* lengths cannot be negative")
        if self.speech_probe_max < 0:
            raise ValueError("speech_probe_max cannot be negative")
        if self.visuals_corner not in OVERLAY_CORNERS:
            raise ValueError(
                f"visuals_corner {self.visuals_corner!r} is not one of "
                f"{', '.join(sorted(OVERLAY_CORNERS))}")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


def config_path_from_env() -> Path | None:
    """The config file named by TALKINGHEAD_CONFIG, or None when unset or empty."""
    value = os.environ.get(ENV_CONFIG)
    return Path(value) if value else None


def load_config(path: str | Path | None = None, **overrides) -> Config:
    """Defaults < config file < environment < keyword overrides (the command-line flags).

    An empty environment variable counts as unset.
    """
    data: dict = {}
    if path is not None:
        data = tomllib.loads(Path(path).read_text())
    for key, var in (("judge", ENV_JUDGE), ("transcriber", ENV_TRANSCRIBER)):
        if os.environ.get(var):
            data[key] = os.environ[var]
    data.update({k: v for k, v in overrides.items() if v is not None})
    known = {f.name for f in fields(Config)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown config key(s): {', '.join(unknown)}")
    return Config(**data).validate()
