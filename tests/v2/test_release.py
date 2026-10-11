"""scripts/release.py: the version CI tags on a push, run against real throwaway git repos."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "release.py"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.name", "t")
    _git(tmp_path, "config", "user.email", "t@example.invalid")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "first release")
    _git(tmp_path, "tag", "-a", "v1.4.2", "-m", "v1.4.2")
    return tmp_path


def _commit(repo: Path, message: str) -> None:
    _git(repo, "commit", "-q", "--allow-empty", "-m", message)


def _plan(repo: Path, notes: Path | None = None) -> subprocess.CompletedProcess:
    args = [sys.executable, str(SCRIPT)] + (["--notes-file", str(notes)] if notes else [])
    return subprocess.run(args, cwd=repo, capture_output=True, text=True)


def test_a_push_without_a_trailer_is_a_patch_release(repo: Path) -> None:
    _commit(repo, "Fix a clipped word at joins")
    assert _plan(repo).stdout.strip() == "1.4.3"


def test_the_largest_trailer_in_the_push_wins_and_resets_lower_parts(repo: Path) -> None:
    _commit(repo, "Fix a typo")
    _commit(repo, "Add Linux support\n\nRuns on Ubuntu too.\n\nRelease: minor")
    _commit(repo, "Fix another typo")
    assert _plan(repo).stdout.strip() == "1.5.0"
    _commit(repo, "New config format\n\nrelease: MAJOR")
    assert _plan(repo).stdout.strip() == "2.0.0"


def test_an_already_tagged_head_plans_nothing(repo: Path) -> None:
    assert _plan(repo).stdout == ""


def test_a_misspelt_trailer_fails_instead_of_guessing(repo: Path) -> None:
    _commit(repo, "Something\n\nRelease: minr")
    result = _plan(repo)
    assert result.returncode == 1 and "minr" in result.stderr and result.stdout == ""


def test_notes_list_each_commit_with_its_body_and_without_trailers(repo: Path, tmp_path: Path) -> None:
    _commit(repo, "Add Linux support\n\nRuns on Ubuntu and WSL2.\n\nRelease: minor")
    _commit(repo, "Fix the smoke test")
    notes = tmp_path / "notes.md"
    _plan(repo, notes)
    text = notes.read_text()
    assert text.index("- Add Linux support") < text.index("- Fix the smoke test")
    assert "Runs on Ubuntu and WSL2." in text and "Release:" not in text


def test_the_highest_tag_counts_not_the_newest(repo: Path) -> None:
    _git(repo, "tag", "-a", "v1.10.0", "-m", "v1.10.0")
    _commit(repo, "Fix")
    assert _plan(repo).stdout.strip() == "1.10.1"
