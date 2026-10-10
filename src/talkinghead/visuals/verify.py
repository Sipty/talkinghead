"""Verification for the visuals pass.

Two machine facts and one contact sheet. The facts are the pass's whole
promise: the audio stream is untouched and the duration did not move. The
contact sheet is a frame at every overlay's midpoint plus controls in clean
gaps, which catches a control frame that has drifted inside an overlay window.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..verify import CheckResult
from .render import probe_duration
from .schedule import Placement

FRAME_TOL = 0.05


def _audio_facts(path: str | Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_name,sample_rate,channels", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    streams = json.loads(out).get("streams", [])
    return streams[0] if streams else {}


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


def verify_visuals(base: str | Path, out: str | Path, placements: list[Placement],
                   work_dir: str | Path) -> tuple[list[CheckResult], dict]:
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    base_d, out_d = probe_duration(base), probe_duration(out)
    delta = out_d - base_d
    checks = [CheckResult(
        "visuals_duration", "fatal", abs(delta) <= FRAME_TOL,
        f"{out_d:.3f}s with visuals vs {base_d:.3f}s cut (delta {delta:+.3f}s, tol {FRAME_TOL:.2f}s)")]
    a_base, a_out = _audio_facts(base), _audio_facts(out)
    same = all(a_base.get(k) == a_out.get(k) for k in ("codec_name", "sample_rate", "channels"))
    checks.append(CheckResult(
        "visuals_audio_copied", "fatal", same,
        f"{a_out.get('codec_name')} {a_out.get('sample_rate')}Hz "
        f"{a_out.get('channels')}ch vs cut {a_base.get('codec_name')} "
        f"{a_base.get('sample_rate')}Hz {a_base.get('channels')}ch"))

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
