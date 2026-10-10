"""Colour: what the source claims, and the one transform the pipeline applies.

The LUT tests are here rather than in the render tests because the transform is
the part that cannot be checked by eye: a swapped cube axis or a 709/2020
matrix mix-up still looks like a plausible card.
"""

import math
import struct
import subprocess
from pathlib import Path

import pytest

from talkinghead.color import (
    BT709_TO_BT2020,
    BT2020_TO_BT709,
    ColorProfile,
    encode_args,
    hlg_to_linear,
    linear_to_hlg,
    linear_to_pq,
    probe_color,
    sdr_to_hdr,
    write_sdr_to_hdr_cube,
)

HLG = ColorProfile(primaries="bt2020", transfer="arib-std-b67", matrix="bt2020nc",
                   range="tv", pix_fmt="yuv420p10le", hdr=True)


def _encoder(name: str) -> bool:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                         capture_output=True, text=True).stdout
    return name in out


def _hlg_clip(path: Path, seconds: float = 2.0) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", f"testsrc=size=320x240:rate=30:duration={seconds}",
         "-vf", "format=yuv420p10le,"
                "scale=out_color_matrix=bt2020nc:out_transfer=arib-std-b67:"
                "out_primaries=bt2020:out_range=tv",
         "-c:v", "hevc_videotoolbox", "-profile:v", "main10", "-pix_fmt", "p010le",
         "-b:v", "4M", "-tag:v", "hvc1", *HLG.tags, str(path)],
        check=True)


def _sdr_clip(path: Path, seconds: float = 2.0) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", f"testsrc=size=320x240:rate=30:duration={seconds}",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)],
        check=True)


# -- the SDR -> HDR transform ------------------------------------------------
def test_white_lands_on_bt2408_graphics_white_not_peak():
    """203 nits is HLG signal 0.75 and PQ signal 0.5807. Mapping white to 1.0
    instead would make every card peak-bright next to the video's own white."""
    hlg = sdr_to_hdr((1.0, 1.0, 1.0), "arib-std-b67")
    assert all(abs(c - 0.7499) < 0.002 for c in hlg), hlg
    assert abs(round(hlg_to_linear(hlg[0]) ** 1.2 * 1000) - 203) <= 1
    pq = sdr_to_hdr((1.0, 1.0, 1.0), "smpte2084")
    assert all(abs(c - 0.5807) < 0.002 for c in pq), pq


def test_black_stays_black_and_the_curve_stays_monotonic():
    assert sdr_to_hdr((0.0, 0.0, 0.0), "arib-std-b67") == (0.0, 0.0, 0.0)
    greys = [sdr_to_hdr((g, g, g), "arib-std-b67")[0] for g in (0.1, 0.3, 0.5, 0.7, 1.0)]
    assert greys == sorted(greys) and greys[0] > 0.0


def test_the_709_and_2020_matrices_are_actual_inverses():
    """A transposed matrix reads as slightly-off colour and nothing else."""
    for i in range(3):
        for j in range(3):
            got = sum(BT2020_TO_BT709[i][k] * BT709_TO_BT2020[k][j] for k in range(3))
            assert abs(got - (1.0 if i == j else 0.0)) < 1e-9


def test_hlg_oetf_round_trips_and_swapping_a_and_c_would_break_it():
    for signal in (0.05, 0.3, 0.5, 0.75, 1.0):
        # 1e-7 is float noise on the exp/log form, not a slack tolerance: the
        # classic a/c swap is off by ~0.4 at 0.75.
        assert math.isclose(linear_to_hlg(hlg_to_linear(signal)), signal, abs_tol=1e-7)
    assert math.isclose(linear_to_hlg(1 / 12), 0.5, abs_tol=1e-9)  # the OETF knee
    # PQ for 1000 nits on a 10 000 nit peak is the published 0.7518.
    assert math.isclose(linear_to_pq(0.1), 0.7518, abs_tol=1e-4)


def test_a_colour_outside_the_sdr_gamut_is_not_clipped_into_a_hue_shift():
    hlg = sdr_to_hdr((1.0, 0.0, 0.0), "arib-std-b67")
    assert min(hlg) > 0.0            # 709 red is inside 2020, it must not go negative
    assert hlg[0] > hlg[1] > hlg[2]


# -- probing -----------------------------------------------------------------
def test_probe_reads_hdr_tags_and_bit_depth_from_a_real_stream(tmp_path: Path):
    if not _encoder("hevc_videotoolbox"):
        pytest.skip("no hevc_videotoolbox in this ffmpeg")
    src = tmp_path / "hlg.mp4"
    _hlg_clip(src)
    profile = probe_color(src)
    assert profile.hdr
    assert (profile.primaries, profile.transfer, profile.matrix) == ("bt2020", "arib-std-b67", "bt2020nc")
    assert profile.pix_fmt == "yuv420p10le"


def test_an_untagged_eight_bit_stream_is_sdr(tmp_path: Path):
    src = tmp_path / "sdr.mp4"
    _sdr_clip(src)
    profile = probe_color(src)
    assert not profile.hdr
    assert encode_args(profile) == ["-c:v", "libx264", "-crf", "18", "-preset", "medium",
                                    "-pix_fmt", "yuv420p"]


def test_a_missing_file_is_an_error_not_an_sdr_guess(tmp_path: Path):
    with pytest.raises(RuntimeError, match="cannot probe"):
        probe_color(tmp_path / "nope.mp4")


def test_hdr_encode_carries_the_tags_and_strips_source_metadata():
    args = encode_args(HLG, bitrate="20M")
    assert "-map_metadata" in args and args[args.index("-map_metadata") + 1] == "-1"
    assert args[args.index("-pix_fmt") + 1] == "p010le"
    assert args[args.index("-profile:v") + 1] == "main10"
    for tag in ("bt2020", "arib-std-b67", "bt2020nc", "tv"):
        assert tag in args


# -- the cube ----------------------------------------------------------------
def test_the_cube_is_deterministic_and_endpoints_are_right(tmp_path: Path):
    first = write_sdr_to_hdr_cube(tmp_path / "a.cube", HLG, size=17)
    second = write_sdr_to_hdr_cube(tmp_path / "b.cube", HLG, size=17)
    assert first.read_bytes() == second.read_bytes()
    lines = first.read_text().splitlines()
    assert lines[0] == "LUT_3D_SIZE 17"
    assert lines[1] == "0.000000 0.000000 0.000000"
    assert all(abs(float(c) - 0.7499) < 0.002 for c in lines[-1].split())


def test_the_cube_refuses_a_profile_it_cannot_convert(tmp_path: Path):
    with pytest.raises(ValueError, match="no conversion"):
        write_sdr_to_hdr_cube(tmp_path / "x.cube", ColorProfile())


def test_ffmpeg_applies_the_cube_with_the_order_it_was_written_in(tmp_path: Path) -> None:
    """ffmpeg's .cube parser indexes blue-fastest; the generator follows it.

    Asserted through the real chain: a red plate and a blue plate must land on
    the luma their own colour predicts, in 10-bit limited-range BT.2020. A
    transposed cube sends red where blue belongs and is obvious here.
    """
    cube = write_sdr_to_hdr_cube(tmp_path / "sdr_to_hlg.cube", HLG)
    for name, rgb in (("red", (1.0, 0.0, 0.0)), ("blue", (0.0, 0.0, 1.0)), ("white", (1.0, 1.0, 1.0))):
        plate = tmp_path / f"{name}.png"
        # format=rgb24 first: the colour source is YUV-backed and a red plate
        # arrived at the cube as (254, 0, 0), which is off the cube's grid.
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                        "-i", f"color=c=0x{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}:s=64x64",
                        "-vf", "format=rgb24", "-frames:v", "1", str(plate)], check=True)
        raw = subprocess.check_output(
            ["ffmpeg", "-v", "error", "-i", str(plate),
             "-vf", f"format=rgba,lut3d=file={cube}:interp=tetrahedral,"
                    "scale=out_color_matrix=bt2020nc:out_range=tv,format=yuv420p10le",
             "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-"])
        got = struct.unpack_from("<H", raw, (32 * 64 + 32) * 2)[0]   # Y of the middle row
        signal = sdr_to_hdr(rgb, "arib-std-b67")
        luma = 0.2627 * signal[0] + 0.6780 * signal[1] + 0.0593 * signal[2]
        expected = 64 + 876 * luma
        assert abs(got - expected) <= 4, f"{name}: got {got}, expected {expected:.1f}"
