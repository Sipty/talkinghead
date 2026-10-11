"""Preserve source HDR encoding and convert SDR graphics into its color space.

Video is never implicitly tone-mapped. Graphics use BT.2408's 203-nit white;
edited HDR video drops stale Dolby Vision metadata and stays ten-bit.
"""

from __future__ import annotations

import functools
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Transfers that carry HDR. Everything else -- including a 10-bit stream with
# no tags at all -- is treated as SDR: guessing HDR from bit depth alone would
# mean inventing primaries and a transfer the source never claimed.
HDR_TRANSFERS = ("arib-std-b67", "smpte2084")
# The encoder the approved 10-bit HLG export used (M-series, videotoolbox). It
# is bitrate-driven -- no crf -- and it is the encoder whose output was accepted
# by eye, so it is preferred wherever ffmpeg has it. Elsewhere libx265 writes the
# same 10-bit stream in software (slower; same bitrate contract).
HDR_CODEC = "hevc_videotoolbox"
SOFTWARE_HDR_CODEC = "libx265"
HDR_BITRATE = "20M"
HDR_PROFILE = "main10"
# Each encoder takes its own 10-bit layout: videotoolbox MSB-aligned p010, libx265 yuv420p10le.
HDR_PIX_FMT = "p010le"
SOFTWARE_HDR_PIX_FMT = "yuv420p10le"

# BT.2408: SDR graphics white sits at 203 nits in an HDR program.
GRAPHICS_WHITE_NITS = 203.0
HLG_PEAK_NITS = 1000.0  # the reference display the HLG OOTF is defined against
PQ_PEAK_NITS = 10000.0

# BT.709 -> BT.2020 (D65) on linear RGB, and its inverse. The inverse is the
# matrix the approved HLG-to-SDR chain used, kept so a test can prove the pair
# round-trips: a typo in either one is invisible in a still and obvious here.
BT709_TO_BT2020 = (
    (0.627403895934699, 0.329283038377883, 0.043313065687418),
    (0.069097289358232, 0.919540395075459, 0.011362315566309),
    (0.016391438875150, 0.088013307877226, 0.895595253247624),
)
BT2020_TO_BT709 = (
    (1.660491002108435, -0.587641138788551, -0.072849863319884),
    (-0.124550474521591, 1.132899897125960, -0.008349422604369),
    (-0.018150763354905, -0.100578898008007, 1.118729661362912),
)

_HLG_A, _HLG_B, _HLG_C = 0.17883277, 0.28466892, 0.55991073
_HLG_OOTF_GAMMA = 1.2
_PQ_M1, _PQ_M2 = 0.1593017578125, 78.84375
_PQ_C1, _PQ_C2, _PQ_C3 = 0.8359375, 18.8515625, 18.6875

_LUMA_2020 = (0.2627, 0.6780, 0.0593)


@dataclass(frozen=True)
class ColorProfile:
    """A video stream's colour facts, defaulting to the SDR pipeline's."""

    primaries: str = "bt709"
    transfer: str = "bt709"
    matrix: str = "bt709"
    range: str = "tv"
    pix_fmt: str = "yuv420p"
    hdr: bool = False

    @property
    def tags(self) -> list[str]:
        return ["-color_primaries", self.primaries, "-color_trc", self.transfer,
                "-colorspace", self.matrix, "-color_range", self.range]


def _stream(path: str | Path) -> dict:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,pix_fmt,color_primaries,color_transfer,color_space,color_range",
         "-of", "json", str(path)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-5:])
        raise RuntimeError(f"cannot probe {path}: {tail}")
    streams = json.loads(proc.stdout or "{}").get("streams") or []
    if not streams:
        raise RuntimeError(f"cannot probe {path}: no video stream")
    return streams[0]


def probe_color(path: str | Path) -> ColorProfile:
    """The first video stream's colour. Untagged values fall back to SDR/bt709."""
    stream = _stream(path)
    transfer = str(stream.get("color_transfer") or "unknown").lower()
    hdr = transfer in HDR_TRANSFERS
    pix_fmt = str(stream.get("pix_fmt") or "yuv420p")

    def known(key: str, default: str) -> str:
        value = str(stream.get(key) or "unknown").lower()
        return default if value == "unknown" else value

    # HLG and PQ are defined in BT.2020 (BT.2100), so a stream that claims the
    # transfer but not the primaries is still BT.2020: that is the only
    # container those transfers are specified in, not a guess about content.
    return ColorProfile(
        primaries=known("color_primaries", "bt2020" if hdr else "bt709"),
        transfer=transfer,
        matrix=known("color_space", "bt2020nc" if hdr else "bt709"),
        range=known("color_range", "tv"),
        pix_fmt=pix_fmt,
        hdr=hdr,
    )


@functools.lru_cache(maxsize=1)
def _ffmpeg_encoders() -> frozenset[str]:
    """Encoder names this ffmpeg build has (`ffmpeg -encoders`), read once per process."""
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    names: set[str] = set()
    for line in out.splitlines():
        parts = line.split()
        # Table rows are "<flags> <name> <description>" with six-character flags; the legend
        # (" V..... = Video") and the header and rule lines are not encoders.
        if len(parts) >= 3 and len(parts[0]) == 6 and parts[1] != "=":
            names.add(parts[1])
    return frozenset(names)


def hdr_codec() -> str:
    """The 10-bit HEVC encoder for HDR output: VideoToolbox when this ffmpeg has it, else libx265."""
    encoders = _ffmpeg_encoders()
    if HDR_CODEC in encoders:
        return HDR_CODEC
    if SOFTWARE_HDR_CODEC in encoders:
        return SOFTWARE_HDR_CODEC
    raise RuntimeError("no 10-bit HEVC encoder in this ffmpeg (need hevc_videotoolbox or libx265): "
                       "install an ffmpeg build with libx265, e.g. apt install ffmpeg")


def hdr_pix_fmt(codec: str) -> str:
    """The pixel format `codec` takes for 10-bit HDR output."""
    return HDR_PIX_FMT if codec == HDR_CODEC else SOFTWARE_HDR_PIX_FMT


def encode_args(profile: ColorProfile, *, crf: int = 18, preset: str = "medium",
                codec: str | None = None, bitrate: str = HDR_BITRATE) -> list[str]:
    """Video encode flags that produce `profile`, libx264 for SDR as before.

    HDR strips source metadata (`-map_metadata -1`): the output is a new grade
    of a re-timed picture, so the source's Dolby Vision record -- if any -- must
    not survive it. `codec` defaults to the best HDR encoder this ffmpeg has.
    """
    if not profile.hdr:
        return ["-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p"]
    codec = codec or hdr_codec()
    return ["-c:v", codec, "-profile:v", HDR_PROFILE, "-pix_fmt", hdr_pix_fmt(codec),
            "-b:v", bitrate, "-tag:v", "hvc1", *profile.tags, "-map_metadata", "-1"]


# ---------------------------------------------------------------------------
# SDR -> HDR transfer (the only colour transform this repo applies)
# ---------------------------------------------------------------------------
def _bt709_to_linear(v: float) -> float:
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def linear_to_hlg(e: float) -> float:
    """BT.2100 OETF: scene linear -> HLG signal."""
    e = min(1.0, max(0.0, e))
    if e <= 1 / 12:
        return math.sqrt(3 * e)
    return _HLG_A * math.log(12 * e - _HLG_B) + _HLG_C


def hlg_to_linear(v: float) -> float:
    """BT.2100 inverse OETF: HLG signal -> scene linear."""
    if v <= 0.5:
        return v * v / 3
    return (math.exp((v - _HLG_C) / _HLG_A) + _HLG_B) / 12


def linear_to_pq(relative: float) -> float:
    """Encode display-linear luminance, relative to 10 000 nits, as ST 2084 PQ."""
    m = min(1.0, max(0.0, relative)) ** _PQ_M1
    return ((_PQ_C1 + _PQ_C2 * m) / (1 + _PQ_C3 * m)) ** _PQ_M2


def sdr_to_hdr(rgb: tuple[float, float, float], transfer: str) -> tuple[float, float, float]:
    """One sRGB/BT.709 pixel (0..1 signals) as an HDR signal for `transfer`.

    SDR white maps to BT.2408 graphics white, so a white card matches the
    video's own white instead of clipping to peak.
    """
    linear_709 = [_bt709_to_linear(c) for c in rgb]
    display = [sum(BT709_TO_BT2020[i][j] * linear_709[j] for j in range(3)) for i in range(3)]
    if transfer == "arib-std-b67":
        peak = HLG_PEAK_NITS
        scaled = [max(0.0, c) * (GRAPHICS_WHITE_NITS / peak) for c in display]
        luma = sum(_LUMA_2020[i] * scaled[i] for i in range(3))
        if luma <= 0.0:
            return (0.0, 0.0, 0.0)
        # Invert the BT.2100 OOTF (display = luma(scene)^(gamma-1) * scene):
        # scene luma is display luma^(1/gamma), so scene = display / luma^0.2/gamma.
        scene = [c / luma ** (0.2 / _HLG_OOTF_GAMMA) for c in scaled]
        return linear_to_hlg(scene[0]), linear_to_hlg(scene[1]), linear_to_hlg(scene[2])
    if transfer == "smpte2084":
        return tuple(linear_to_pq(c * (GRAPHICS_WHITE_NITS / PQ_PEAK_NITS)) for c in display)
    raise ValueError(f"no SDR->HDR transform for transfer {transfer!r}")


def write_sdr_to_hdr_cube(path: str | Path, profile: ColorProfile, size: int = 33) -> Path:
    """Write the SDR -> `profile` 3D LUT that `lut3d` applies to overlay assets.

    33 grid points is far past what flat card art needs, and the cube is a pure
    function of the profile, so this is deterministic: same profile, same file.
    Cube rows vary red fastest; ffmpeg rearranges them into its internal layout.
    """
    if not profile.hdr:
        raise ValueError("SDR overlays need no conversion for an SDR base")
    if profile.transfer not in HDR_TRANSFERS:
        raise ValueError(f"no SDR->HDR transform for transfer {profile.transfer!r}")
    step = 1.0 / (size - 1)
    lines = [f"LUT_3D_SIZE {size}"]
    for n in range(size ** 3):
        r = (n % size) * step
        g = ((n // size) % size) * step
        b = (n // (size * size)) * step
        out = sdr_to_hdr((r, g, b), profile.transfer)
        lines.append(" ".join(f"{c:.6f}" for c in out))
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
