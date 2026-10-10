"""Content-addressed cache and run directory: the reason a rerun reuses the
transcript and every LLM verdict instead of re-sampling them."""

import json
import os
from pathlib import Path

import pytest

from talkinghead.store import Cache, RunDir, RunDirConflict, atomic_write_text, sha256_file, stable_key


def test_sha256_file_is_content_addressed(tmp_path: Path) -> None:
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"same bytes")
    b.write_bytes(b"same bytes")
    assert sha256_file(a) == sha256_file(b)
    b.write_bytes(b"other")
    assert sha256_file(a) != sha256_file(b)


def test_stable_key_is_order_independent_and_versioned() -> None:
    assert stable_key({"b": 1, "a": [1, 2]}, version=3) == stable_key({"a": [1, 2], "b": 1}, version=3)
    assert stable_key({"a": 1}, version=1) != stable_key({"a": 1}, version=2)
    assert stable_key({"a": 1.0}, version=1) == stable_key({"a": 1.0}, version=1)


def test_atomic_write_never_leaves_a_partial_file(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    atomic_write_text(target, "{}")
    assert target.read_text() == "{}"
    assert list(tmp_path.iterdir()) == [target]


def test_cache_get_or_compute_computes_once_per_key(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "cache")
    calls = []

    def compute() -> dict:
        calls.append(1)
        return {"words": [1, 2, 3]}

    first = cache.get_or_compute("abc123", "transcript", {"backend": "parakeet"}, 1, compute)
    second = cache.get_or_compute("abc123", "transcript", {"backend": "parakeet"}, 1, compute)
    assert first == second == {"words": [1, 2, 3]}
    assert calls == [1]
    # A different parameter set or version is a different entry.
    cache.get_or_compute("abc123", "transcript", {"backend": "whisper"}, 1, compute)
    cache.get_or_compute("abc123", "transcript", {"backend": "parakeet"}, 2, compute)
    assert calls == [1, 1, 1]
    # Entries live under the content hash so a wiped video folder still hits.
    assert (tmp_path / "cache" / "abc123").is_dir()


def test_cache_entries_are_plain_json_files(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "cache")
    cache.get_or_compute("h", "envelope", {}, 1, lambda: {"x": 1})
    files = list((tmp_path / "cache" / "h").glob("*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text())["value"] == {"x": 1}


def test_rundir_layout_and_json_helpers(tmp_path: Path) -> None:
    run = RunDir(tmp_path / "video.talkinghead")
    run.ensure()
    assert run.plan.name == "plan.json"
    assert run.report.name == "report.md"
    assert run.decisions.name == "decisions.json"
    assert run.llm_dir.is_dir() and run.verify_dir.is_dir()
    run.write_json(run.decisions, {"cold_open": 104.4})
    assert run.read_json(run.decisions) == {"cold_open": 104.4}
    assert run.read_json(run.plan, default=None) is None


def test_rundir_binds_to_one_recording_and_refuses_another(tmp_path: Path) -> None:
    run = RunDir(tmp_path / "cook")
    run.ensure()
    run.bind("/videos/take.mov", "a" * 64)
    assert json.loads(run.manifest.read_text())["source"].endswith("take.mov")
    # The same recording resumes in the directory it created.
    run.bind("/videos/take.mov", "a" * 64)
    with pytest.raises(RunDirConflict) as exc:
        run.bind("/videos/other.mov", "b" * 64)
    assert "--out-dir" in str(exc.value)
    # The refused cook leaves the directory's own manifest alone.
    assert json.loads(run.manifest.read_text())["source_sha256"] == "a" * 64


def test_rundir_stage_memo_skips_when_inputs_unchanged(tmp_path: Path) -> None:
    run = RunDir(tmp_path / "v.talkinghead")
    run.ensure()
    calls = []

    def stage() -> dict:
        calls.append(1)
        return {"cuts": 3}

    assert run.memo("pauses", {"transcript": "sha1", "config": {"pause": 0.35}}, 1, stage) == {"cuts": 3}
    assert run.memo("pauses", {"transcript": "sha1", "config": {"pause": 0.35}}, 1, stage) == {"cuts": 3}
    assert calls == [1]
    run.memo("pauses", {"transcript": "sha2", "config": {"pause": 0.35}}, 1, stage)
    assert calls == [1, 1]
    run.memo("pauses", {"transcript": "sha2", "config": {"pause": 0.35}}, 1, stage, fresh=True)
    assert calls == [1, 1, 1]


def test_a_failed_atomic_write_leaves_no_temp_file(tmp_path: Path, monkeypatch) -> None:
    def broken(src, dst):
        raise OSError("replace failed")
    monkeypatch.setattr(os, "replace", broken)
    with pytest.raises(OSError):
        atomic_write_text(tmp_path / "out.json", "{}")
    assert list(tmp_path.iterdir()) == []
