"""Editing profile: preferences reach every judge prompt, each cook keeps the
snapshot it was judged with, and lessons only become rules once accepted."""

from pathlib import Path

import pytest

from talkinghead.cli import main
from talkinghead.profile import (
    ProfileError,
    accept_lesson,
    pending_lessons,
    read_preferences,
    record_feedback,
    reject_lesson,
    snapshot_preferences,
)
from talkinghead.prompts import with_preferences
from talkinghead.store import RunDir
from talkinghead.timeline import Cut, Plan, SourceInfo


def _cook(tmp_path: Path) -> RunDir:
    run = RunDir(tmp_path / "v.talkinghead").ensure()
    plan = Plan.from_cuts(SourceInfo("main", str(tmp_path / "v.mp4"), "abc", 30.0), [Cut(5.0, 10.0, "silence")])
    run.plan.write_text(plan.to_json())
    return run


def test_empty_preferences_leave_the_prompt_byte_identical() -> None:
    # Byte-identical prompts are what let cooks without preferences replay cached answers.
    assert with_preferences("TASK: X", "") == "TASK: X"
    assert with_preferences("TASK: X", "  \n") == "TASK: X"
    led = with_preferences("TASK: X", "- keep laughs")
    assert led.startswith("--- OWNER EDITING PREFERENCES") and led.endswith("- keep laughs\n\nTASK: X")


def test_a_new_cook_snapshots_the_profile_and_later_runs_replay_it(tmp_path: Path) -> None:
    run = _cook(tmp_path)
    assert snapshot_preferences(run, "- rule one") == ("- rule one", "profile")
    # The profile changed afterwards: this cook still uses what it was first judged with.
    assert snapshot_preferences(run, "- rule two") == ("- rule one", "snapshot")
    assert snapshot_preferences(run, "- rule two", refresh=True) == ("- rule two", "refreshed")
    assert snapshot_preferences(run, "- rule three") == ("- rule two", "snapshot")


def test_a_cook_that_predates_the_profile_is_frozen_without_preferences(tmp_path: Path) -> None:
    run = _cook(tmp_path)
    (run.memo_dir / "retakes.json").write_text("{}")
    text, origin = snapshot_preferences(run, "- new rule")
    assert text == "" and origin.startswith("frozen")
    # The freeze is persistent, not re-decided on every run.
    assert snapshot_preferences(run, "- new rule") == ("", "snapshot")


def test_feedback_maps_the_output_time_to_the_source(tmp_path: Path) -> None:
    run = _cook(tmp_path)
    entry = record_feedback(tmp_path / "profile", run.root, kind="video", note="cut the aside", out_time=7.0)
    # Output 7.0 s is source 12.0 s (5 s were cut before it): source time survives a re-cut.
    assert entry["source_time"] == 12.0 and entry["status"] == "noted"
    with pytest.raises(ProfileError):
        record_feedback(tmp_path / "profile", run.root, kind="video", note="x", out_time=99.0)


def test_lessons_become_preferences_only_when_accepted(tmp_path: Path) -> None:
    run, profile = _cook(tmp_path), tmp_path / "profile"
    first = record_feedback(profile, run.root, kind="preference", note="n1", lesson="Commit to the pitch.")
    second = record_feedback(profile, run.root, kind="preference", note="n2", lesson="Never use emojis.")
    assert read_preferences(profile) == ""
    assert [e["id"] for e in pending_lessons(profile)] == [first["id"], second["id"]]

    accept_lesson(profile, first["id"], text="Commit to the pitch:  pain, cost, solution.")
    reject_lesson(profile, second["id"])
    assert read_preferences(profile) == "- Commit to the pitch: pain, cost, solution."
    assert pending_lessons(profile) == []
    with pytest.raises(ProfileError):
        accept_lesson(profile, second["id"])


def test_a_defect_cannot_be_promoted_as_a_preference(tmp_path: Path) -> None:
    run = _cook(tmp_path)
    with pytest.raises(ProfileError):
        record_feedback(tmp_path / "profile", run.root, kind="defect", note="clipped", lesson="cut better")


def test_feedback_without_a_profile_names_the_fix(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    run = _cook(tmp_path)
    assert main(["feedback", str(run.root), "too long"]) == 1
    assert "TALKINGHEAD_PROFILE" in capsys.readouterr().err
