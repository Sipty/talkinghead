"""The edit lock: plan digest, audio packet MD5, write/read/check round trip, mismatch messages, transcript."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from talkinghead.lock import (
    LOCK_VERSION,
    LockMismatch,
    audio_stream_md5,
    check_lock,
    locked_transcript,
    plan_digest,
    read_lock,
    write_lock,
)
from talkinghead.timeline import Cut, Plan, Segment, SourceInfo
from talkinghead.transcribe import Transcript, Word


def _plan(**kw) -> Plan:
    source = SourceInfo(id="main", path="/x/a.mp4", sha256="a" * 64, duration=10.0)
    base = dict(
        sources={"main": source}, main="main",
        cuts=[Cut(start=2.0, end=3.0, reason="retake", text="um")],
        segments=[Segment("main", 0.0, 2.0), Segment("main", 3.0, 10.0)],
        decisions={"k": 1}, provenance={"talkinghead": "1.0"},
        isolation={"track": {"sha256": "b" * 64}},
    )
    base.update(kw)
    return Plan(**base)


def _m4a(path: Path, freq: int = 440, seconds: float = 1.0) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency={freq}:sample_rate=48000:duration={seconds}",
         "-c:a", "aac", "-b:a", "192k", str(path)], check=True)


def test_a_digest_tracks_edges_sources_and_isolation_but_not_commentary():
    base = plan_digest(_plan())
    assert len(base) == 64
    moved = _plan(segments=[Segment("main", 0.0, 2.001), Segment("main", 3.0, 10.0)])
    assert plan_digest(moved) != base
    other_src = _plan(sources={"main": SourceInfo("main", "/x/a.mp4", "c" * 64, 10.0)})
    assert plan_digest(other_src) != base
    assert plan_digest(_plan(isolation={"track": {"sha256": "d" * 64}})) != base
    assert plan_digest(_plan(isolation={})) != base
    same = _plan(
        cuts=[Cut(start=2.0, end=3.0, reason="content", text="different")],
        decisions={"other": 2}, provenance={"talkinghead": "9.9"}, anchors={"a": 1.0},
    )
    assert plan_digest(same) == base


def test_a_stream_md5_survives_a_copy_mux_and_changes_on_reencode(tmp_path):
    a = tmp_path / "a.m4a"
    _m4a(a)
    mp4 = tmp_path / "v.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1:r=10", "-i", str(a),
         "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-c:a", "copy", "-shortest", str(mp4)], check=True)
    re = tmp_path / "re.m4a"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(a), "-c:a", "aac", "-b:a", "192k", str(re)], check=True)
    assert audio_stream_md5(a) == audio_stream_md5(mp4)
    assert audio_stream_md5(re) != audio_stream_md5(a)


def test_a_stream_md5_names_a_file_without_audio(tmp_path):
    silent = tmp_path / "v.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1:r=10", str(silent)], check=True)
    with pytest.raises(RuntimeError, match="v.mp4"):
        audio_stream_md5(silent)


def test_a_lock_round_trips_with_paths_relative_to_the_lock_directory(tmp_path):
    audio = tmp_path / "audio-master.m4a"
    _m4a(audio)
    lock_path = tmp_path / "edit-lock.json"
    plan = _plan()
    written = write_lock(lock_path, plan, audio, fps=30.0)
    assert written["version"] == LOCK_VERSION
    assert written["audio"]["path"] == "audio-master.m4a"
    assert written["audio"]["codec"] == "aac" and written["audio"]["sample_rate"] == 48000
    assert written["transcript"] is None
    assert written["output_duration"] == 9.0
    lock = read_lock(lock_path)
    assert lock == written
    assert check_lock(lock, plan, fps=30.0, lock_path=lock_path) == audio.resolve()
    outside = tmp_path.parent / f"{tmp_path.name}-outside.m4a"
    _m4a(outside)
    assert Path(write_lock(lock_path, plan, outside, fps=30.0)["audio"]["path"]).is_absolute()


def test_a_read_lock_refuses_missing_garbled_and_other_version_files(tmp_path):
    with pytest.raises(LockMismatch, match="edit-lock.json"):
        read_lock(tmp_path / "edit-lock.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    with pytest.raises(LockMismatch):
        read_lock(bad)
    bad.write_text(json.dumps({"version": 99}))
    with pytest.raises(LockMismatch, match="version"):
        read_lock(bad)


def test_a_check_lock_mismatch_names_plan_audio_and_fps(tmp_path):
    audio = tmp_path / "m.m4a"
    _m4a(audio)
    lock_path = tmp_path / "edit-lock.json"
    plan = _plan()
    lock = write_lock(lock_path, plan, audio, fps=30.0)
    edited = replace(plan, segments=[Segment("main", 0.0, 2.5), Segment("main", 3.0, 10.0)])
    with pytest.raises(LockMismatch, match="plan") as e:
        check_lock(lock, edited, fps=30.0, lock_path=lock_path)
    assert plan_digest(edited)[:12] in str(e.value) and lock["plan_sha"][:12] in str(e.value)
    assert str(lock_path) in str(e.value)
    with pytest.raises(LockMismatch, match="fps"):
        check_lock(lock, plan, fps=25.0, lock_path=lock_path)
    _m4a(audio, freq=880)
    with pytest.raises(LockMismatch, match="audio"):
        check_lock(lock, plan, fps=30.0, lock_path=lock_path)


def test_a_locked_transcript_returns_words_and_refuses_edits_or_absence(tmp_path):
    audio = tmp_path / "m.m4a"
    _m4a(audio)
    tr = Transcript("fake", "1", 1.0, [Word("hello", 0.0, 0.4, 0.9), Word("world", 0.5, 0.9, 0.8)])
    tpath = tmp_path / "locked-transcript.json"
    tpath.write_text(json.dumps(tr.to_dict(), sort_keys=True))
    lock_path = tmp_path / "edit-lock.json"
    lock = write_lock(lock_path, _plan(), audio, fps=30.0, transcript=tpath)
    got = locked_transcript(lock, lock_path=lock_path)
    assert [w.text for w in got.words] == ["hello", "world"]
    tpath.write_text(json.dumps({**tr.to_dict(), "version": "2"}))
    with pytest.raises(LockMismatch, match="transcript"):
        locked_transcript(lock, lock_path=lock_path)
    bare = write_lock(lock_path, _plan(), audio, fps=30.0)
    with pytest.raises(LockMismatch, match="transcript"):
        locked_transcript(bare, lock_path=lock_path)
