"""Defect loop: a heard tool bug becomes a fixture, a ledger entry and a test
stub in one step, so the fix is proven and guarded."""

import json
import subprocess
import wave
from pathlib import Path

from talkinghead.feedback import record_defect
from talkinghead.store import RunDir
from talkinghead.timeline import Cut, Plan, SourceInfo


def _wav(path: Path, seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency=300:sample_rate=16000:duration={seconds}",
                    "-c:a", "pcm_s16le", str(path)], check=True)


def test_record_defect_cuts_a_source_fixture_and_appends_the_ledger(tmp_path: Path) -> None:
    run = RunDir(tmp_path / "v.talkinghead").ensure()
    plan = Plan.from_cuts(SourceInfo("main", str(tmp_path / "v.mp4"), "abc", 30.0), [Cut(5.0, 10.0, "silence")])
    run.plan.write_text(plan.to_json())
    _wav(run.root / "audio" / "audio-16k.wav", 30.0)
    fixtures = tmp_path / "fixtures"
    tests_dir = tmp_path / "tests_feedback"

    entry = record_defect(run.root, out_time=7.0, note="clipped 'the'", fixtures_dir=fixtures,
                            tests_dir=tests_dir, half_window=3.0)
    # Output 7.0 s is source 12.0 s (5 s were cut before it).
    assert entry["source_time"] == 12.0 and entry["output_time"] == 7.0
    wav = Path(entry["fixture"])
    assert wav.is_file() and wav.parent == fixtures
    with wave.open(str(wav)) as w:
        assert abs(w.getnframes() / w.getframerate() - 6.0) < 0.05
    ledger = [json.loads(line) for line in (fixtures / "ledger.jsonl").read_text().splitlines()]
    assert ledger[-1]["note"] == "clipped 'the'" and ledger[-1]["nearby_cuts"] == [{"start": 5.0, "end": 10.0, "reason": "silence"}]
    stub = next(tests_dir.glob("test_feedback_*.py"))
    text = stub.read_text()
    assert "clipped 'the'" in text and wav.name in text


def test_record_defect_clips_the_window_at_the_source_edges(tmp_path: Path) -> None:
    run = RunDir(tmp_path / "v.talkinghead").ensure()
    plan = Plan.from_cuts(SourceInfo("main", str(tmp_path / "v.mp4"), "abc", 8.0), [])
    run.plan.write_text(plan.to_json())
    _wav(run.root / "audio" / "audio-16k.wav", 8.0)
    entry = record_defect(run.root, out_time=0.5, note="x", fixtures_dir=tmp_path / "f", tests_dir=tmp_path / "t", half_window=5.0)
    assert entry["fixture_start"] == 0.0 and entry["fixture_end"] == 5.5
