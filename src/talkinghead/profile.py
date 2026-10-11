"""The editing profile: approved preferences, plus the feedback they were learned from.

A profile is a directory named by TALKINGHEAD_PROFILE (or `--profile`):

    preferences.md   approved, durable editing preferences; every judge prompt carries them
    feedback.jsonl   every recorded correction, with an optional proposed lesson

Feedback has a kind. `video` fixes one cut only, `preference` is taste worth
remembering, and `defect` is a tool bug (fixed in code with a test, never
promoted as a preference). A proposed lesson stays pending until someone
accepts it; nothing is learned silently, so a one-off fix cannot quietly
become a rule.

Each cook snapshots the preferences into its run directory on its first run
and replays that snapshot afterwards. Editing the profile therefore shapes new
cooks without changing what an existing cook asked its judge, which keeps its
cached decisions valid.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

from .store import RunDir, atomic_write_text
from .timeline import Plan

ENV_PROFILE = "TALKINGHEAD_PROFILE"
PREFERENCES = "preferences.md"
FEEDBACK = "feedback.jsonl"
# Per-cook copy of the preferences that cook was judged with.
SNAPSHOT = "preferences.md"
KINDS = ("video", "preference", "defect")


class ProfileError(ValueError):
    """A profile operation that cannot proceed as asked."""


def profile_dir(given: str | Path | None = None) -> Path | None:
    """`--profile`, else TALKINGHEAD_PROFILE, else no profile (an empty variable counts as unset)."""
    if given:
        return Path(given).expanduser()
    value = os.environ.get(ENV_PROFILE)
    return Path(value).expanduser() if value else None


def read_preferences(profile: Path | None) -> str:
    if profile is None:
        return ""
    path = profile / PREFERENCES
    return path.read_text().strip() if path.is_file() else ""


def snapshot_preferences(run: RunDir, current: str, *, refresh: bool = False) -> tuple[str, str]:
    """The preferences this cook is judged with, and where they came from.

    The first run of a cook writes the snapshot. A cook that already holds
    judged work but no snapshot predates profiles: it is frozen with no
    preferences, so its earlier prompts, memos and answers replay unchanged.
    `refresh` re-takes the snapshot from the current profile on purpose.
    """
    path = run.root / SNAPSHOT
    if path.is_file() and not refresh:
        return path.read_text().strip(), "snapshot"
    prior_work = any(run.memo_dir.glob("*.json")) or any(run.llm_dir.glob("*.json"))
    if prior_work and not path.is_file() and not refresh:
        text, origin = "", "frozen: cook predates the profile"
    else:
        text, origin = current.strip(), ("refreshed" if refresh else "profile")
    atomic_write_text(path, text + "\n" if text else "")
    return text, origin


def _ledger(profile: Path) -> Path:
    return profile / FEEDBACK


def read_feedback(profile: Path) -> list[dict]:
    path = _ledger(profile)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_feedback(profile: Path, entries: list[dict]) -> None:
    atomic_write_text(_ledger(profile), "".join(json.dumps(e, sort_keys=True) + "\n" for e in entries))


def _locate(run_dir: Path, out_time: float | None) -> dict:
    """Where the note points: the cook, and the source instant behind an output time.

    Output times shift with every re-cut, so the source time is what stays
    meaningful after the next edit.
    """
    where: dict = {"run_dir": str(run_dir.resolve())}
    plan_file = next((run_dir / p for p in ("plan.json", "demo-plan.json") if (run_dir / p).is_file()), None)
    if plan_file is not None:
        plan = Plan.from_json(plan_file.read_text())
        source = plan.sources[plan.main]
        where.update(source=source.path, video_sha256=source.sha256)
        if out_time is not None:
            src_t = plan.output_to_source(out_time)
            if src_t is None:
                raise ProfileError(f"output time {out_time:.2f}s is past the end of the render "
                                   f"({plan.output_duration:.2f}s)")
            where["source_time"] = round(src_t, 3)
    if out_time is not None:
        where["output_time"] = out_time
    return where


def record_feedback(profile: Path, run_dir: str | Path, *, kind: str, note: str,
                    out_time: float | None = None, lesson: str | None = None,
                    extra: dict | None = None) -> dict:
    if kind not in KINDS:
        raise ProfileError(f"unknown feedback kind {kind!r}: use one of {', '.join(KINDS)}")
    if kind == "defect" and lesson:
        raise ProfileError("a defect is fixed in code with a test, not learned as a preference: drop --lesson")
    profile.mkdir(parents=True, exist_ok=True)
    entries = read_feedback(profile)
    entry = {
        "id": max((e["id"] for e in entries), default=0) + 1,
        "date": date.today().isoformat(), "kind": kind, "note": note,
        **_locate(Path(run_dir), out_time),
        **({"lesson": lesson.strip(), "status": "pending"} if lesson else {"status": "noted"}),
        **(extra or {}),
    }
    _write_feedback(profile, entries + [entry])
    return entry


def pending_lessons(profile: Path) -> list[dict]:
    return [e for e in read_feedback(profile) if e.get("status") == "pending"]


def _resolve(profile: Path, lesson_id: int) -> tuple[list[dict], dict]:
    entries = read_feedback(profile)
    entry = next((e for e in entries if e["id"] == lesson_id), None)
    if entry is None or "lesson" not in entry:
        raise ProfileError(f"no lesson #{lesson_id} in {_ledger(profile)}")
    if entry["status"] != "pending":
        raise ProfileError(f"lesson #{lesson_id} is already {entry['status']}")
    return entries, entry


def accept_lesson(profile: Path, lesson_id: int, *, text: str | None = None) -> dict:
    """Promote a pending lesson into preferences.md, optionally reworded first."""
    entries, entry = _resolve(profile, lesson_id)
    rule = " ".join((text or entry["lesson"]).split())
    path = profile / PREFERENCES
    current = path.read_text().rstrip() if path.is_file() else ""
    atomic_write_text(path, (current + "\n" if current else "") + f"- {rule}\n")
    entry.update(status="accepted", accepted=date.today().isoformat(), rule=rule)
    _write_feedback(profile, entries)
    return entry


def reject_lesson(profile: Path, lesson_id: int) -> dict:
    entries, entry = _resolve(profile, lesson_id)
    entry.update(status="rejected", rejected=date.today().isoformat())
    _write_feedback(profile, entries)
    return entry
