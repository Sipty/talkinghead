"""The visuals pass: one ffmpeg run over the finished cut.

One picture pass over the cut, leaving the cut pipeline unchanged. The base
video's audio stream is COPIED, so loudness is untouched and the cut checks
keep their meaning. Overlays are gated by `enable=between(t,...)` and
faded on alpha, anchored to the top-right corner, the one region of this
framing the speaker does not occupy.

Colour: the bake must not cost the cut its colour. An SDR base keeps the
libx264/yuv420p encode it always had; an HLG/PQ base is re-encoded as 10-bit
with its own tags, and the SDR overlay assets are converted into that system
with the cube from `color.py` -- a relabel would put an sRGB card into an HLG
frame as if it were already HLG, and a straight SDR encode would drop the
whole picture to 8-bit. With no overlays the pass is a stream copy, so there
is nothing to convert.

Duration and memory both took several attempts:

- `-loop 1` with no bound never ends. A 20 s probe reached 165 MB and was
  still encoding, because ffmpeg keeps going while any input yields frames.
- `-shortest` is the wrong cure. It trims to the shortest stream, which is
  the base's own audio, and lost 2 frames on the same probe.
- Bounding each still with `-t` fixes the duration, but looping it from t=0
  to its window END is ruinous for memory: a 40-overlay pass over a 22-minute
  cut generated 801,135 frames of static images, because an overlay at 22:28
  looped 40,635 frames to get there. That pass wanted ~2.4 GB and the OS
  killed it five times.

So each still is looped for only its OWN window and then shifted into place
with `setpts`, which is ~70x fewer frames. `eof_action=pass` lets the main
video through before a still's stream begins.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from ..color import HDR_BITRATE, HDR_CODEC, ColorProfile, encode_args, probe_color, write_sdr_to_hdr_cube
from ..config import OVERLAY_CORNERS
from .schedule import Placement


def probe_duration(path: str | Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def probe_width(path: str | Path) -> int:
    """The base's real width.

    Never assume 1920. The demo tier renders at -2:480, so a 0.34 overlay
    scaled against a 1920 default came out 652 px wide on an 854 px frame --
    76%% of the picture, covering the speaker instead of sitting in the corner.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip().rstrip(",")
    return int(out)


def _even(x: float) -> int:
    """Overlay widths land on an even number: yuv420p chroma needs it."""
    return max(2, int(round(x / 2.0)) * 2)


def _overlay_chain(i: int, p: Placement, width: int, fade: float,
                   color: ColorProfile, cube: str | None) -> str:
    """One still, scaled, (colour-converted,) faded and dropped at `p.start`."""
    w = _even(p.scale * width)
    hold = p.end - p.start
    fin = min(fade, max(0.01, hold / 2))
    steps = [f"scale={w}:-1", "format=rgba"]
    if color.hdr:
        # The cube maps sRGB to the base's HDR signal, primaries included; the
        # scale after it only does the matrix/range step into BT.2020, with NO
        # out_primaries -- swscale would then convert the gamut a second time
        # (measured: pure red lands 60 p010 levels high) and no transfer, since
        # the LUT already produced encoded HLG/PQ values.
        steps += [
            f"lut3d=file={cube}:interp=tetrahedral",
            f"scale=out_color_matrix={color.matrix}:out_range={color.range}",
            "format=yuva420p10le",
        ]
    steps += [
        f"fade=t=in:st=0:d={fin:.3f}:alpha=1",
        f"fade=t=out:st={hold - fin:.3f}:d={fin:.3f}:alpha=1",
        f"setpts=PTS+{p.start:.6f}/TB",
    ]
    return f"[{i + 1}:v]{','.join(steps)}[ov{i}]"


def build_overlay_command(
    base: str, placements: list[Placement], out: str, *,
    margin: int = 40, fade: float = 0.4, width: int = 1920,
    base_duration: float | None = None, fps: int = 30, corner: str = "top-right",
    color: ColorProfile | None = None, cube: str | None = None,
    hdr_codec: str = HDR_CODEC, hdr_bitrate: str = HDR_BITRATE,
) -> list[str]:
    """`color` is the base's profile; None means SDR (`probe_color` in the caller).

    An HDR base without `cube` raises: composing SDR assets into HDR frames
    without the conversion would be the relabel this pass exists to avoid.
    """
    if corner not in OVERLAY_CORNERS:
        raise ValueError(f"unknown overlay corner {corner!r}")
    x_expr, y_expr = (e.format(m=margin) for e in OVERLAY_CORNERS[corner])
    profile = color or ColorProfile()
    if not placements:
        # Nothing to draw: a stream copy, so the deliverable still exists and
        # is byte-identical to the cut (and so is its colour).
        return ["ffmpeg", "-y", "-v", "error", "-nostdin", "-i", base, "-c", "copy",
                "-movflags", "+faststart", out]
    if profile.hdr and not cube:
        raise ValueError("HDR base: overlays need the SDR->HDR cube from write_sdr_to_hdr_cube")
    cmd = ["ffmpeg", "-y", "-v", "error", "-nostdin", "-i", base]
    for p in placements:
        # Only this still's own window, not from t=0 to it. Six decimals
        # because -t rounded UP to 20.067 on a 20.066667 s base once put an
        # extra frame on the end.
        span = p.end - p.start + 0.5
        if base_duration is not None:
            span = min(span, base_duration)
        cmd += ["-loop", "1", "-framerate", str(fps), "-t", f"{span:.6f}", "-i", p.path]
    parts: list[str] = []
    first = "[0:v]"
    if profile.hdr:
        parts.append("[0:v]format=p010le[base]")
        first = "[base]"
    for i, p in enumerate(placements):
        # Fades are relative to the still's own clip, then setpts drops the
        # clip onto the output timeline at `start`.
        parts.append(_overlay_chain(i, p, width, fade, profile, cube))
    last = first
    # The overlay filter otherwise negotiates an 8-bit intermediate, even for HDR.
    overlay_format = "format=yuv420p10:" if profile.hdr else ""
    for i, p in enumerate(placements):
        nxt = f"[t{i}]" if i < len(placements) - 1 else "[vout]"
        # eof_action=pass: before a still's stream starts and after it ends,
        # the main video passes straight through.
        parts.append(
            f"{last}[ov{i}]overlay=x={x_expr}:y={y_expr}:"
            f"{overlay_format}enable='between(t,{p.start:.3f},{p.end:.3f})':eof_action=pass{nxt}"
        )
        last = nxt
    cmd += [
        "-filter_complex", ";".join(parts),
        "-map", "[vout]", "-map", "0:a",
        *encode_args(profile, crf=18, preset="medium", codec=hdr_codec, bitrate=hdr_bitrate),
        "-c:a", "copy", "-movflags", "+faststart", out,
    ]
    return cmd


def render_visuals(base: str, placements: list[Placement], out: str, *,
                   margin: int = 40, fade: float = 0.4, width: int | None = None,
                   corner: str = "top-right",
                   hdr_codec: str = HDR_CODEC, hdr_bitrate: str = HDR_BITRATE) -> str:
    """Overlay `placements` onto `base`. Width, duration and colour are PROBED."""
    color = probe_color(base)
    work: tempfile.TemporaryDirectory | None = None
    if color.hdr and placements:
        # The cube is derived from the base's profile alone, and lives only as
        # long as this bake: nothing on disk to go stale between cooks.
        work = tempfile.TemporaryDirectory(prefix="talkinghead-visuals-")
    try:
        cube = str(write_sdr_to_hdr_cube(Path(work.name) / "sdr_to_hdr.cube", color)) if work else None
        cmd = build_overlay_command(base, placements, out, margin=margin, fade=fade,
                                    width=width or probe_width(base),
                                    base_duration=probe_duration(base), corner=corner,
                                    color=color, cube=cube,
                                    hdr_codec=hdr_codec, hdr_bitrate=hdr_bitrate)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or "").strip().splitlines()[-15:])
            raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}):\n{tail}")
        if not Path(out).is_file():
            raise RuntimeError(f"visuals pass produced no file: {out}")
        return " ".join(str(c) for c in cmd)
    finally:
        if work:
            work.cleanup()
