"""Verification for the overlays in the delivered picture.

One machine fact and one contact sheet. The fact is that a frame can be grabbed at every overlay's midpoint. The
contact sheet is those frames plus controls in clean gaps, which catches a control frame that has drifted inside an
overlay window. The picture table (`av_sync`, `color`, `resolution`, `audio_locked`) already runs on the same file.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..verify import CheckResult
from .render import probe_duration
from .schedule import Placement


def grab(video: str | Path, t: float, out: Path, *, width: int = 960) -> Path | None:
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video),
         "-frames:v", "1", "-vf", f"scale={width}:-2", str(out)],
        capture_output=True, text=True)
    return out if proc.returncode == 0 and out.is_file() else None


def control_times(placements: list[Placement], duration: float, want: int = 2) -> list[float]:
    """Times with no overlay up, so a control frame proves the corner is clean."""
    busy = [(p.start, p.end) for p in placements]
    out: list[float] = []
    for frac in (0.25, 0.5, 0.75, 0.1, 0.9):
        t = duration * frac
        if all(not (a - 1.0 <= t <= b + 1.0) for a, b in busy):
            out.append(round(t, 2))
        if len(out) >= want:
            break
    return out


def verify_visuals(out: str | Path, placements: list[Placement],
                   work_dir: str | Path) -> tuple[list[CheckResult], dict]:
    """The contact sheet and the one check only the overlays can fail: a frame grabs at every overlay's midpoint.

    Duration, audio and frame size are the picture table's job on the same file."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out_d = probe_duration(out)
    checks: list[CheckResult] = []
    frames: list[dict] = []
    for p in placements:
        name = f"chk_{p.asset}_{p.mid:.2f}.jpg"
        got = grab(out, p.mid, work / name)
        frames.append({"asset": p.asset, "t": round(p.mid, 2),
                       "window": [round(p.start, 2), round(p.end, 2)],
                       "frame": name if got else None})
    controls = []
    for t in control_times(placements, out_d):
        name = f"chk_control_{t:.2f}.jpg"
        got = grab(out, t, work / name)
        controls.append({"t": t, "frame": name if got else None})
    missing = [f["asset"] for f in frames if f["frame"] is None]
    checks.append(CheckResult(
        "visuals_frames", "warn", not missing,
        f"{len(frames) - len(missing)}/{len(frames)} overlay frames grabbed, "
        f"{len([c for c in controls if c['frame']])} controls"
        + (f"; failed: {', '.join(missing)}" if missing else "")))

    lines = ["# visuals contact sheet", "",
             f"- source: `{Path(out).name}` ({out_d:.2f}s)",
             f"- overlays: {len(placements)}", ""]
    for f in frames:
        lines.append(f"## {f['asset']} @ {f['t']}s (window {f['window'][0]}-{f['window'][1]}s)")
        lines.append(f"![{f['asset']}]({f['frame']})" if f["frame"] else "_frame grab failed_")
        lines.append("")
    for c in controls:
        lines.append(f"## control @ {c['t']}s (no overlay expected)")
        lines.append(f"![control]({c['frame']})" if c["frame"] else "_frame grab failed_")
        lines.append("")
    (work / "contact.md").write_text("\n".join(lines))
    return checks, {"frames": frames, "controls": controls, "contact": "contact.md"}
