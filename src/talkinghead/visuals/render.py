"""The overlay graph: how placed stills are composited over a picture.

`render.render_plan` splices these builders into the cut's one picture pass, so overlays are drawn inside the same
encode as the cut and cost no second generation. Overlays are gated by `enable=between(t,...)`, faded on alpha and
anchored to a configurable corner of the frame.

Colour: an SDR picture keeps the libx264/yuv420p encode it always had; an HLG/PQ picture stays 10-bit with its own
tags, and the SDR overlay assets are converted into that system with the cube from `color.py` -- a relabel would put
an sRGB card into an HLG frame as if it were already HLG.

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
from dataclasses import asdict, dataclass
from pathlib import Path

from ..color import ColorProfile
from ..config import OVERLAY_CORNERS
from ..store import sha256_file
from .schedule import Placement


def probe_duration(path: str | Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def _even(x: float) -> int:
    """Overlay widths land on an even number: yuv420p chroma needs it."""
    return max(2, int(round(x / 2.0)) * 2)


def _overlay_chain(i: int, input_index: int, p: Placement, width: int, fade: float,
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
    return f"[{input_index}:v]{','.join(steps)}[ov{i}]"


@dataclass(frozen=True)
class Overlays:
    """Everything that decides what the overlays look like, apart from the picture they land on.

    One value so a picture pass and a render memo both read the same
    placements, margin, fade and corner instead of passing four loose arguments.
    """

    placements: tuple[Placement, ...]
    margin: int = 40
    fade: float = 0.4
    corner: str = "top-right"

    def __post_init__(self) -> None:
        if self.corner not in OVERLAY_CORNERS:
            raise ValueError(f"unknown overlay corner {self.corner!r}")

    def identity(self) -> dict:
        """What a render memo keys on: the asset BYTES, not where they live.

        Hashing the file means an asset rewritten at the same path invalidates
        the memo, and moving the asset directory does not.
        """
        placements = []
        for p in self.placements:
            entry = asdict(p)
            del entry["path"]
            entry["sha256"] = sha256_file(p.path)
            placements.append(entry)
        return {"placements": placements, "margin": self.margin, "fade": self.fade, "corner": self.corner}


def overlay_inputs(overlays: Overlays, *, fps: int, duration: float | None) -> list[str]:
    """The bounded still inputs, one per placement, in placement order."""
    args: list[str] = []
    for p in overlays.placements:
        # Only this still's own window, not from t=0 to it. Six decimals
        # because -t rounded UP to 20.067 on a 20.066667 s base once put an
        # extra frame on the end.
        span = p.end - p.start + 0.5
        if duration is not None:
            span = min(span, duration)
        args += ["-loop", "1", "-framerate", str(fps), "-t", f"{span:.6f}", "-i", p.path]
    return args


def overlay_graph(overlays: Overlays, *, base: str, first_input: int, width: int,
                  color: ColorProfile, cube: str | None, out: str = "[vout]") -> list[str]:
    """The filter parts that composite every placement over the stream labelled `base`.

    Still `i` is read from input `first_input + i`, so any picture pass can
    splice the overlays into its own graph. HDR without `cube` raises:
    composing SDR assets into HDR frames without the conversion would be the
    relabel this pass exists to avoid.
    """
    placements = overlays.placements
    if not placements:
        raise ValueError("no overlay placements: there is no graph to build")
    if color.hdr and not cube:
        raise ValueError("HDR base: overlays need the SDR->HDR cube from write_sdr_to_hdr_cube")
    x_expr, y_expr = (e.format(m=overlays.margin) for e in OVERLAY_CORNERS[overlays.corner])
    parts: list[str] = []
    first = base
    if color.hdr:
        parts.append(f"{base}format=p010le[base]")
        first = "[base]"
    for i, p in enumerate(placements):
        # Fades are relative to the still's own clip, then setpts drops the
        # clip onto the output timeline at `start`.
        parts.append(_overlay_chain(i, first_input + i, p, width, overlays.fade, color, cube))
    last = first
    # The overlay filter otherwise negotiates an 8-bit intermediate, even for HDR.
    overlay_format = "format=yuv420p10:" if color.hdr else ""
    for i, p in enumerate(placements):
        nxt = f"[t{i}]" if i < len(placements) - 1 else out
        # eof_action=pass: before a still's stream starts and after it ends,
        # the main video passes straight through.
        parts.append(
            f"{last}[ov{i}]overlay=x={x_expr}:y={y_expr}:"
            f"{overlay_format}enable='between(t,{p.start:.3f},{p.end:.3f})':eof_action=pass{nxt}"
        )
        last = nxt
    return parts

