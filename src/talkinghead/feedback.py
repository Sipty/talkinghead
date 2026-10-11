"""`talkinghead report`: turn what a reviewer heard into a fixture and a test.

A reviewer watches the output and says "at 7:03 the word 'the' is clipped".
That instant is mapped back through the plan to the source, a short wav around
it is cut into the fixtures directory, the ledger records the note with the
cuts near it, and a test stub is scaffolded so the fix is proven by a test
against the very audio that failed, and guarded forever after.
"""

from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path

from .store import RunDir, atomic_write_text
from .timeline import Plan


def _next_id(ledger: Path) -> int:
    if not ledger.is_file():
        return 1
    return sum(1 for _ in ledger.read_text().splitlines()) + 1


def record_feedback(
    run_dir: str | Path, *, out_time: float, note: str, fixtures_dir: str | Path,
    tests_dir: str | Path, half_window: float = 6.0,
) -> dict:
    run = RunDir(Path(run_dir))
    plan_file = next((p for p in ("plan.json", "demo-plan.json") if (run.root / p).is_file()), None)
    if plan_file is None:
        raise FileNotFoundError(f"no plan.json in {run.root}")
    plan = Plan.from_json((run.root / plan_file).read_text())
    src_t = plan.output_to_source(out_time)
    if src_t is None:
        raise ValueError(f"output time {out_time:.2f}s is past the end of the render ({plan.output_duration:.2f}s)")
    duration = plan.sources[plan.main].duration
    start, end = max(0.0, src_t - half_window), min(duration, src_t + half_window)

    fixtures = Path(fixtures_dir)
    fixtures.mkdir(parents=True, exist_ok=True)
    ledger = fixtures / "ledger.jsonl"
    fid = _next_id(ledger)
    wav_out = fixtures / f"feedback-{fid:03d}-{src_t:.1f}.wav"
    source_wav = run.root / "audio" / "audio-16k.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(source_wav),
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-f", "wav", "-fflags", "+bitexact", "-flags:a", "+bitexact",
         str(wav_out)],
        check=True, capture_output=True, text=True,
    )
    nearby = [
        {"start": c.start, "end": c.end, "reason": c.reason}
        for c in plan.cuts if c.end >= start and c.start <= end
    ]
    entry = {
        "id": fid, "date": date.today().isoformat(), "note": note,
        "output_time": out_time, "source_time": round(src_t, 3),
        "fixture": str(wav_out), "fixture_start": round(start, 3), "fixture_end": round(end, 3),
        "source": plan.sources[plan.main].path, "video_sha256": plan.sources[plan.main].sha256,
        "nearby_cuts": nearby,
    }
    with ledger.open("a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")

    tests = Path(tests_dir)
    tests.mkdir(parents=True, exist_ok=True)
    stub = tests / f"test_feedback_{fid:03d}.py"
    atomic_write_text(stub, f'''"""Feedback #{fid} ({entry["date"]}): {note}

Heard at output {out_time:.2f}s = source {src_t:.2f}s of {entry["source"]}.
Fixture: {wav_out.name} (source {start:.2f}-{end:.2f}s, event at {src_t - start:.2f}s in the clip).
Nearby cuts at report time: {json.dumps(nearby)}

Replace the skip with the assertion that would have caught this, then fix the code
or the guidelines until it passes.
"""

import pytest

FIXTURE = {json.dumps(wav_out.name)}
EVENT_T = {src_t - start:.3f}


def test_feedback_{fid:03d}() -> None:
    pytest.skip("feedback #{fid}: write the assertion for: {note.replace(chr(34), chr(39))}")
''')
    entry["test_stub"] = str(stub)
    return entry
