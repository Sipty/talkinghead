"""Head-clip check: refuse overlays that sit over the speaker's head.

Runs on the BASE cut (``final.mp4``) BEFORE anything is baked, so the head
is fully visible and an intersection between the overlay rectangle and the
person mask means the bake would cover it. Detection is Apple Vision person
segmentation through the compiled ``headcoverage.swift`` helper (first-party
frameworks only, no new pip dependencies); person segmentation covers hair,
which face-box detectors miss on profile shots and lean-ins.

Decision rule: a placement FAILS only when at least ``min_clipped`` samples
in its window exceed the coverage threshold. One frame above the threshold
is segmentation noise; a real lean-in stays over the threshold for seconds
of samples.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from ..config import OVERLAY_CORNERS
from ..verify import CheckResult
from .render import probe_duration, probe_width
from .schedule import Placement

# Fraction of the overlay rect that may hold person pixels before a sample
# counts as clipped. A hair tuft crossing the card edge is a few percent; the
# background and the corner region above the speaker's head sit far below this.
COVERAGE_THRESHOLD = 0.02
# Consecutive-ish samples (one per window step) that must be clipped before
# a placement is judged clipped; kills mask flicker on paintings and props.
MIN_CLIPPED = 2
# Seconds between samples inside a placement window.
SAMPLE_STEP = 0.5
# Sample margin off the window edges: the fade-in/out means the first and
# last instants carry little visible card.
EDGE_MARGIN = 0.2
# Frames for the detector are grabbed at this width; rects are scaled to it.
GRAB_WIDTH = 960

HELPER_CACHE = Path.home() / ".cache" / "talkinghead" / "bin"


def _even(x: float) -> int:
    """Mirror render._even: overlay widths land on even numbers for yuv420p."""
    return max(2, int(round(x / 2.0)) * 2)


def _image_dims(path: str | Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip().rstrip(",")
    w, h = (int(v) for v in out.split(","))
    return w, h


def overlay_rect(p: Placement, width: int, height: int, margin: int, corner: str) -> tuple[int, int, int, int]:
    """The on-screen rectangle the renderer will paint for this placement.

    Mirrors build_overlay_command exactly: even width, height from the
    asset's own aspect, anchored per corner with the margin. Coordinates are
    (x, y, w, h) in base pixels.
    """
    if corner not in OVERLAY_CORNERS:
        raise ValueError(f"unknown overlay corner {corner!r}")
    aw, ah = _image_dims(p.path)
    w = _even(p.scale * width)
    h = max(2, round(w * ah / aw))
    x_expr, y_expr = OVERLAY_CORNERS[corner]
    x = width - w - margin if "W" in x_expr else margin
    y = height - h - margin if "H" in y_expr else margin
    return x, y, w, h


def sample_times(p: Placement) -> list[float]:
    """Sample instants inside a placement window, off the fade edges."""
    out: list[float] = []
    t = p.start + EDGE_MARGIN
    while t < p.end - EDGE_MARGIN:
        out.append(round(t, 3))
        t += SAMPLE_STEP
    return out or [p.mid]


def _helper_path() -> Path:
    """Compile the Swift detector once per source revision; cache by sha."""
    src = Path(__file__).with_name("headcoverage.swift")
    sha = hashlib.sha256(src.read_bytes()).hexdigest()[:8]
    dest = HELPER_CACHE / f"headcoverage-{sha}"
    if not dest.exists():
        if shutil.which("swiftc") is None:
            raise RuntimeError("swiftc not found; install Xcode command line tools (xcode-select --install)")
        HELPER_CACHE.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        proc = subprocess.run(["swiftc", "-O", str(src), "-o", str(tmp)],
                              capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(f"headcoverage compile failed: {proc.stderr[-500:]}")
        tmp.rename(dest)
    return dest


def _coverages(items: list[dict]) -> dict[str, float]:
    """Person-pixel coverage per frame via the compiled helper."""
    req = json.dumps({"items": [{"path": it["frame"], "rect": list(it["rect"])} for it in items]})
    proc = subprocess.run([str(_helper_path())], input=req, capture_output=True,
                          text=True, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(f"headcoverage failed (exit {proc.returncode}): {proc.stderr[-500:]}")
    out = json.loads(proc.stdout)["results"]
    failed = [r for r in out if r.get("error")]
    if failed:
        raise RuntimeError(f"headcoverage errors: {failed[0]['error'][:300]}")
    return {r["path"]: r["coverage"] for r in out}


def _grab(video: str | Path, t: float, out: Path) -> Path | None:
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video),
         "-frames:v", "1", "-vf", f"scale={GRAB_WIDTH}:-2", str(out)],
        capture_output=True, text=True)
    return out if proc.returncode == 0 and out.is_file() else None


def summarize(coverages: list[float], *, threshold: float = COVERAGE_THRESHOLD,
              min_clipped: int = MIN_CLIPPED) -> tuple[bool, float, int]:
    """Pure decision over one placement's sample coverages.

    Returns (clipped, worst coverage, number of clipped samples).
    """
    clipped = sum(1 for c in coverages if c > threshold)
    worst = max(coverages, default=0.0)
    return clipped >= min_clipped, worst, clipped


def check_head_clip(
    base: str | Path, placements: list[Placement], *, margin: int, corner: str,
    work_dir: str | Path, threshold: float = COVERAGE_THRESHOLD,
    min_clipped: int = MIN_CLIPPED, duration: float | None = None,
) -> tuple[CheckResult, dict]:
    """Fatal when any overlay would cover the speaker's head in the bake.

    Sampled on ``base`` (the bare cut) so the check gates the bake; clipping
    frames are kept in ``work_dir`` for eyeballing the failure.
    """
    facts: dict = {"threshold": threshold, "min_clipped": min_clipped,
                   "margin": margin, "corner": corner, "placements": []}
    if not placements:
        return CheckResult("visuals_head_clip", "fatal", True, "no overlays placed"), facts
    try:
        _helper_path()
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        # Degrade loudly but do not block the cook on a broken toolchain.
        return CheckResult("visuals_head_clip", "warn", True,
                           f"skipped: head detector unavailable ({str(e)[:200]})"), facts

    width = probe_width(base)
    height = _image_dims(base)[1]
    work = Path(work_dir) / "headclip"
    work.mkdir(parents=True, exist_ok=True)
    duration = duration if duration is not None else probe_duration(base)

    items: list[dict] = []
    index: dict[str, tuple[Placement, list[float]]] = {}
    for p in placements:
        x, y, w, h = overlay_rect(p, width, height, margin, corner)
        times = [t for t in sample_times(p) if t <= duration - 0.1]
        # Rect in grabbed-frame coordinates.
        s = GRAB_WIDTH / width
        rect = [round(x * s), round(y * s), round(w * s), round(h * s)]
        frames = []
        for t in times:
            name = f"chip_{p.asset}_{t:.2f}.jpg"
            got = _grab(base, t, work / name)
            if got:
                frames.append(got)
                items.append({"frame": str(got), "rect": rect,
                              "placement": p.asset, "t": t})
        index[p.asset] = (p, times)

    try:
        cov = _coverages(items)
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        return CheckResult("visuals_head_clip", "warn", True,
                           f"skipped: head detector failed ({str(e)[:200]})"), facts

    by_placement: dict[str, list[tuple[float, float]]] = {}
    for it in items:
        by_placement.setdefault(it["placement"], []).append((it["t"], cov[it["frame"]]))

    worst_all = 0.0
    clipped_placements: list[str] = []
    for asset, samples in by_placement.items():
        samples.sort()
        coverages = [c for _, c in samples]
        clipped, worst, n_clipped = summarize(coverages, threshold=threshold, min_clipped=min_clipped)
        worst_all = max(worst_all, worst)
        p, times = index[asset]
        entry = {"asset": asset, "window": [round(p.start, 2), round(p.end, 2)],
                 "samples": len(samples), "max_coverage": round(worst, 4),
                 "clipped_samples": n_clipped,
                 "clipped_times": [round(t, 2) for t, c in samples if c > threshold]}
        facts["placements"].append(entry)
        if clipped:
            clipped_placements.append(
                f"{asset} @ {p.start:.1f}-{p.end:.1f}s: max coverage {worst * 100:.1f}% "
                f"in {n_clipped} samples at "
                f"{', '.join(f'{t:.1f}s' for t in entry['clipped_times'][:6])}")

    if clipped_placements:
        detail = ("overlay covers the speaker: " + "; ".join(clipped_placements)
                  + f" (threshold {threshold * 100:.0f}%, margin {margin}px, corner {corner}). "
                  "Fix per cook in the cook-local TOML (visuals_corner / visuals_scale_card / "
                  "visuals_scale_shot) BEFORE the first render; late CLI overrides re-key stage "
                  "memos. Clipping frames are in verify/visuals/headclip/.")
        return CheckResult("visuals_head_clip", "fatal", False, detail), facts
    return CheckResult(
        "visuals_head_clip", "fatal", True,
        f"{len(placements)} placements, {len(items)} samples: max person coverage "
        f"{worst_all * 100:.1f}% (threshold {threshold * 100:.0f}%)"), facts
