import hashlib
import json
import struct
import subprocess

import pytest

from talkinghead.color import ColorProfile, decode_args, hdr_codec, hdr_pix_fmt, without_hwaccel, write_sdr_to_hdr_cube
from talkinghead.visuals.render import (
    Overlays,
    _even,
    build_overlay_command,
    overlay_graph,
    overlay_inputs,
    render_visuals,
)
from talkinghead.visuals.schedule import Placement


def cmd_for(*placements, **kw):
    opts = {"margin": 40, "fade": 0.4, "width": 1920}
    opts.update(kw)
    return build_overlay_command("base.mp4", list(placements), "out.mp4", **opts)


def graph_for(*placements, base="[0:v]", first_input=1, width=1920, color=None, cube=None, out="[vout]", **kw):
    """The overlay chain from the builder directly: `kw` are the Overlays fields."""
    return ";".join(overlay_graph(Overlays(tuple(placements), **kw), base=base, first_input=first_input,
                                  width=width, color=color or ColorProfile(), cube=cube, out=out))


def inputs_for(*placements, fps=30, duration=None):
    return overlay_inputs(Overlays(tuple(placements)), fps=fps, duration=duration)


def test_audio_is_copied_never_re_encoded():
    c = cmd_for(Placement("a", "/a.png", 10.0, 20.0, 0.28))
    assert "-c:a" in c and c[c.index("-c:a") + 1] == "copy"


def test_shortest_is_not_used_so_duration_is_preserved():
    """-shortest trimmed the output to the shorter of the base's own video and
    audio streams (2 frames on a 20 s probe). overlay already ends with the
    main input, so the flag only broke the duration invariant."""
    assert "-shortest" not in cmd_for(Placement("a", "/a.png", 1.0, 5.0, 0.28))


def test_overlay_is_anchored_top_right():
    assert "x=W-w-40:y=40" in graph_for(Placement("a", "/a.png", 10.0, 20.0, 0.28))


def test_the_overlay_corner_is_configurable_per_cook():
    """The framing is not the same in every recording: on one cook his head sat
    in the top-right corner, so the overlays moved to the left."""
    def at(corner):
        return graph_for(Placement("a", "/a.png", 10.0, 20.0, 0.28), corner=corner)

    assert "x=40:y=40" in at("top-left")
    assert "x=W-w-40:y=H-h-40" in at("bottom-right")
    assert "x=40:y=H-h-40" in at("bottom-left")
    with pytest.raises(ValueError):
        Overlays((Placement("a", "/a.png", 10.0, 20.0, 0.28),), corner="middle")


def test_display_window_is_gated_by_enable():
    assert "between(t,10.000,20.000)" in graph_for(Placement("a", "/a.png", 10.0, 20.0, 0.28))


def test_scale_follows_the_placement_and_lands_even():
    assert "scale=538:-1" in graph_for(Placement("a", "/a.png", 1.0, 9.0, 0.28))   # 0.28 * 1920 = 537.6 -> 538
    assert "scale=652:-1" in graph_for(Placement("b", "/b.png", 1.0, 9.0, 0.34))   # 0.34 * 1920 = 652.8 -> 652


def test_fade_is_applied_on_alpha_at_both_edges():
    """Relative to the still's own clip now, because setpts shifts it into
    place afterwards."""
    fc = graph_for(Placement("a", "/a.png", 5.0, 9.0, 0.34))
    assert "fade=t=in:st=0:d=0.400:alpha=1" in fc
    assert "fade=t=out:st=3.600:d=0.400:alpha=1" in fc
    assert "setpts=PTS+5.000000/TB" in fc


def test_short_window_shrinks_the_fade_rather_than_overlapping():
    assert "d=0.250" in graph_for(Placement("a", "/a.png", 5.0, 5.5, 0.28))


def test_overlays_chain_in_order_to_one_output():
    fc = graph_for(Placement("a", "/a.png", 1.0, 5.0, 0.28),
                   Placement("b", "/b.png", 9.0, 12.0, 0.28))
    assert "[0:v][ov0]overlay" in fc and "[t0][ov1]overlay" in fc
    assert fc.count("[vout]") == 1


def test_every_placement_becomes_an_input():
    c = inputs_for(Placement("a", "/a.png", 1.0, 5.0, 0.28),
                   Placement("b", "/b.png", 9.0, 12.0, 0.28))
    assert c.count("-loop") == 2
    assert "/a.png" in c and "/b.png" in c
    assert inputs_for() == []


def test_each_still_is_bounded_to_its_own_window():
    """An unbounded -loop 1 never ends: a 20 s probe reached 165 MB and was
    still encoding. Every image input carries its own -t instead."""
    c = inputs_for(Placement("a", "/a.png", 1.0, 5.0, 0.28),
                   Placement("b", "/b.png", 9.0, 12.0, 0.28))
    assert c.count("-t") == 2
    assert "4.500000" in c and "3.500000" in c      # hold + 0.5, not end + 1


def test_a_late_overlay_does_not_loop_from_the_start_of_the_video():
    """Looping each still from t=0 to its window end generated 801,135 frames
    of static images across a 40-overlay pass, wanted ~2.4 GB, and the OS
    killed it five times."""
    late = Placement("late", "/l.png", 1300.0, 1309.0, 0.46)
    c = inputs_for(late)
    span = float(c[c.index("-t") + 1])
    assert span < 12.0, f"still spans {span}s; it must cover its window, not the video"
    assert "setpts=PTS+1300.000000/TB" in graph_for(late)


def test_the_main_video_passes_through_before_a_still_begins():
    assert "eof_action=pass" in graph_for(Placement("a", "/a.png", 100.0, 109.0, 0.46))


def test_still_bound_is_capped_at_the_base_duration():
    """Uncapped, a still longer than the base EXTENDS the output (+0.43 s on a
    probe). Shifting makes this rare, but the cap stays as a guard."""
    assert "20.066667" in inputs_for(Placement("a", "/a.png", 0.0, 30.0, 0.28), duration=20.066667)


def test_still_bound_keeps_six_decimals():
    """-t rounded up to 20.067 on a 20.066667 s base added one frame."""
    assert "20.067" not in inputs_for(Placement("a", "/a.png", 0.0, 30.0, 0.28), duration=20.066667)


def test_the_builders_never_use_shortest():
    """-shortest is not in any input the builders emit either."""
    p = Placement("a", "/a.png", 1.0, 5.0, 0.28)
    assert "-shortest" not in inputs_for(p)


def test_no_placements_is_a_stream_copy():
    c = build_overlay_command("base.mp4", [], "out.mp4", margin=40, fade=0.4, width=1920)
    assert "-filter_complex" not in c
    assert c[c.index("-c") + 1] == "copy"


def test_even_helper():
    assert _even(537.6) == 538 and _even(652.8) == 652 and _even(1.0) == 2


def test_scale_follows_the_actual_frame_width_not_a_default():
    """The demo tier renders 854x480. Scaling 0.34 against a 1920 default gave
    a 652 px overlay on an 854 px frame, covering the speaker."""
    fc = graph_for(Placement("a", "/a.png", 1.0, 9.0, 0.34), width=854)
    assert "scale=290:-1" in fc          # 0.34 * 854 = 290.4 -> 290
    assert "scale=652:-1" not in fc


def test_overlay_graph_reads_stills_from_first_input_and_ends_in_the_requested_label():
    fc = graph_for(Placement("a", "/a.png", 1.0, 5.0, 0.28), Placement("b", "/b.png", 9.0, 12.0, 0.28),
                   base="[vbase]", first_input=3, out="[pic]")
    assert "[3:v]" in fc and "[4:v]" in fc and "[1:v]" not in fc and "[0:v]" not in fc
    assert "[vbase][ov0]overlay" in fc and "[t0][ov1]overlay" in fc
    assert fc.count("[pic]") == 1 and fc.endswith("[pic]") and "[vout]" not in fc


def test_the_graph_builder_refuses_nothing_to_draw():
    with pytest.raises(ValueError, match="placements"):
        graph_for()


def test_the_same_bake_command_comes_out_of_the_builders(monkeypatch):
    """The expected lists were captured from the implementation before the builders existed."""
    monkeypatch.setattr("talkinghead.visuals.render.decode_args", lambda pix_fmt: ["-hwaccel", "videotoolbox"])
    head = ["ffmpeg", "-y", "-v", "error", "-nostdin", "-hwaccel", "videotoolbox", "-i", "base.mp4"]
    a, b = Placement("a", "/a.png", 1.0, 5.0, 0.28), Placement("b", "/b.png", 9.0, 12.0, 0.34)
    chain_a = ("scale=538:-1,format=rgba,fade=t=in:st=0:d=0.400:alpha=1,"
               "fade=t=out:st=3.600:d=0.400:alpha=1,setpts=PTS+1.000000/TB")
    chain_b = ("scale=652:-1,format=rgba,fade=t=in:st=0:d=0.400:alpha=1,"
               "fade=t=out:st=2.600:d=0.400:alpha=1,setpts=PTS+9.000000/TB")
    still_a = ["-loop", "1", "-framerate", "30", "-t", "4.500000", "-i", "/a.png"]
    still_b = ["-loop", "1", "-framerate", "30", "-t", "3.500000", "-i", "/b.png"]
    sdr_tail = ["-map", "[vout]", "-map", "0:a", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", "out.mp4"]

    one = build_overlay_command("base.mp4", [a], "out.mp4", margin=40, fade=0.4, width=1920,
                                base_duration=20.066667)
    assert one == [*head, *still_a, "-filter_complex",
                   f"[1:v]{chain_a}[ov0];[0:v][ov0]overlay=x=W-w-40:y=40:"
                   "enable='between(t,1.000,5.000)':eof_action=pass[vout]", *sdr_tail]

    two = build_overlay_command("base.mp4", [a, b], "out.mp4", margin=40, fade=0.4, width=1920,
                                corner="top-left")
    assert two == [*head, *still_a, *still_b, "-filter_complex",
                   f"[1:v]{chain_a}[ov0];[2:v]{chain_b}[ov1];"
                   "[0:v][ov0]overlay=x=40:y=40:enable='between(t,1.000,5.000)':eof_action=pass[t0];"
                   "[t0][ov1]overlay=x=40:y=40:enable='between(t,9.000,12.000)':eof_action=pass[vout]",
                   *sdr_tail]

    hdr_chain = ("scale={w}:-1,format=rgba,lut3d=file=/c.cube:interp=tetrahedral,"
                 "scale=out_color_matrix=bt2020nc:out_range=tv,format=yuva420p10le,"
                 "fade=t=in:st=0:d=0.400:alpha=1,fade=t=out:st={o}:d=0.400:alpha=1,setpts=PTS+{s}/TB")
    hlg = build_overlay_command("base.mp4", [a, b], "out.mp4", margin=40, fade=0.4, width=1920,
                                color=HLG, cube="/c.cube", hdr_codec="hevc_videotoolbox")
    assert hlg == [
        *head, *still_a, *still_b, "-filter_complex",
        f"[0:v]format=p010le[base];[1:v]{hdr_chain.format(w=538, o='3.600', s='1.000000')}[ov0];"
        f"[2:v]{hdr_chain.format(w=652, o='2.600', s='9.000000')}[ov1];"
        "[base][ov0]overlay=x=W-w-40:y=40:format=yuv420p10:enable='between(t,1.000,5.000)':eof_action=pass[t0];"
        "[t0][ov1]overlay=x=W-w-40:y=40:format=yuv420p10:enable='between(t,9.000,12.000)':eof_action=pass[vout]",
        "-map", "[vout]", "-map", "0:a", "-c:v", "hevc_videotoolbox", "-profile:v", "main10",
        "-pix_fmt", "p010le", "-b:v", "20M", "-tag:v", "hvc1", "-color_primaries", "bt2020",
        "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc", "-color_range", "tv",
        "-map_metadata", "-1", "-c:a", "copy", "-movflags", "+faststart", "out.mp4"]

    assert build_overlay_command("base.mp4", [], "out.mp4", margin=40, fade=0.4, width=1920) == [
        "ffmpeg", "-y", "-v", "error", "-nostdin", "-i", "base.mp4", "-c", "copy", "-movflags", "+faststart",
        "out.mp4"]


def test_identity_follows_the_asset_bytes_not_the_path(tmp_path):
    first, moved = tmp_path / "one", tmp_path / "two"
    first.mkdir()
    moved.mkdir()
    (first / "a.png").write_bytes(b"card-one")
    before = Overlays((Placement("a", str(first / "a.png"), 1.0, 5.0, 0.28),)).identity()
    assert "path" not in before["placements"][0] and before["placements"][0]["sha256"]
    (first / "a.png").write_bytes(b"card-two")
    rewritten = Overlays((Placement("a", str(first / "a.png"), 1.0, 5.0, 0.28),)).identity()
    assert rewritten != before
    (moved / "a.png").write_bytes(b"card-two")
    assert Overlays((Placement("a", str(moved / "a.png"), 1.0, 5.0, 0.28),)).identity() == rewritten


def test_hdr_graph_without_a_cube_is_refused():
    with pytest.raises(ValueError, match="cube"):
        graph_for(Placement("a", "/a.png", 1.0, 5.0, 0.28), color=HLG)


# -- colour: the bake must not cost the cut its colour -----------------------
HLG = ColorProfile(primaries="bt2020", transfer="arib-std-b67", matrix="bt2020nc",
                   range="tv", pix_fmt="yuv420p10le", hdr=True)


def _bake_clip(path, seconds=2.0, hdr=True):
    """A base with audio: the bake copies that audio stream, so it must exist."""
    enc = (["-vf", "format=yuv420p10le,scale=out_color_matrix=bt2020nc:"
            "out_transfer=arib-std-b67:out_primaries=bt2020:out_range=tv",
            "-c:v", hdr_codec(), "-profile:v", "main10", "-pix_fmt", hdr_pix_fmt(hdr_codec()),
            "-b:v", "4M", "-tag:v", "hvc1", *HLG.tags] if hdr else
           ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"])
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=30:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}",
         *enc, "-c:a", "aac", "-b:a", "128k", str(path)], check=True)


def _card(path, color="white"):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", f"color=c={color}:s=100x60", "-vf", "format=rgb24",
                    "-frames:v", "1", str(path)], check=True)


def _video(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=pix_fmt,profile,color_primaries,color_transfer,color_space,nb_frames,duration",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"][0]


def _audio_hash(path):
    payload = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-c:a", "copy", "-f", "adts", "-"])
    return hashlib.sha256(payload).hexdigest()


def _luma(path, t, x, y, width=320):
    """The 10-bit limited-range luma sample at (x, y) of the frame at t."""
    raw = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
         "-pix_fmt", "yuv420p10le", "-f", "rawvideo", "-"])
    return struct.unpack_from("<H", raw, (y * width + x) * 2)[0]


def test_an_hdr_base_keeps_its_ten_bit_hlg_through_the_bake(tmp_path):
    """The overlay pass used to re-encode any base to 8-bit yuv420p, silently
    dropping a 10-bit HLG cut to SDR. Colour and audio both survive now."""
    base, card, out = tmp_path / "base.mp4", tmp_path / "card.png", tmp_path / "out.mp4"
    _bake_clip(base)
    _card(card)
    render_visuals(str(base), [Placement("card", str(card), 0.5, 1.5, 0.3)], str(out))

    video = _video(out)
    assert video["pix_fmt"] == "yuv420p10le"
    assert video["profile"] == "Main 10"
    assert (video["color_primaries"], video["color_transfer"], video["color_space"]) == \
        ("bt2020", "arib-std-b67", "bt2020nc")
    assert int(video["nb_frames"]) == int(_video(base)["nb_frames"])
    assert float(video["duration"]) == pytest.approx(float(_video(base)["duration"]), abs=0.05)
    assert _audio_hash(out) == _audio_hash(base)


def test_an_sdr_card_lands_on_hlg_graphics_white_not_peak(tmp_path):
    """A white card must read at the video's own white (HLG 0.75, 10-bit luma
    ~721). Tagging the sRGB values as HLG would put it at peak, and encoding the
    base to SDR would put the whole picture somewhere else entirely.

    The red card is the other half of the check: its luma is where the gamut
    conversion shows up. Composing with swscale's own `out_primaries` on top of
    the cube put pure red 60 levels high, which a white card cannot see.
    """
    base, out = tmp_path / "base.mp4", tmp_path / "out.mp4"
    white, red = tmp_path / "white.png", tmp_path / "red.png"
    _bake_clip(base, 4.0)
    _card(white)
    _card(red, "red")
    before_outside = _luma(base, 1.7, 20, 200)
    # Windows long enough to have an opaque instant: fade is 0.4 s per edge.
    render_visuals(str(base),
                   [Placement("white", str(white), 0.2, 1.6, 0.3),
                    Placement("red", str(red), 1.8, 3.2, 0.3)], str(out))

    # 0.3 * 320 = 96 wide, top-right with margin 40 -> x 184..280, y 40..98.
    assert abs(_luma(out, 0.9, 230, 70) - 721) <= 25     # HLG 0.75, not 940
    assert abs(_luma(out, 2.5, 230, 70) - 392) <= 25     # BT.2020 luma of that red
    assert abs(_luma(out, 1.7, 20, 200) - before_outside) <= 8      # untouched picture
    assert _luma(out, 3.5, 230, 70) != _luma(out, 2.5, 230, 70)     # windows are gated


def test_an_sdr_base_still_bakes_eight_bit_and_copies_audio(tmp_path):
    base, card, out = tmp_path / "base.mp4", tmp_path / "card.png", tmp_path / "out.mp4"
    _bake_clip(base, hdr=False)
    _card(card)
    render_visuals(str(base), [Placement("card", str(card), 0.5, 1.5, 0.3)], str(out))
    assert _video(out)["pix_fmt"] == "yuv420p"
    assert _audio_hash(out) == _audio_hash(base)


def test_without_a_conversion_an_hdr_base_refuses_sdr_overlays():
    with pytest.raises(ValueError, match="cube"):
        build_overlay_command("base.mp4", [Placement("a", "/a.png", 1.0, 5.0, 0.28)],
                              "out.mp4", color=HLG)


def test_no_placements_on_an_hdr_base_is_a_stream_copy(tmp_path):
    """Nothing to draw means nothing to convert: the deliverable stays the cut."""
    base, out = tmp_path / "base.mp4", tmp_path / "out.mp4"
    _bake_clip(base)
    cmd = render_visuals(str(base), [], str(out))
    assert "-c copy" in cmd and "lut3d" not in cmd
    assert _audio_hash(out) == _audio_hash(base)
    assert _video(out) == _video(base)


def test_hdr_compositor_preserves_source_precision_outside_graphics(tmp_path):
    """A 10-bit encoder cannot restore detail lost in an 8-bit overlay filter."""
    width, height = 1024, 180
    base, card = tmp_path / "ramp.mkv", tmp_path / "card.png"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
         f"nullsrc=s={width}x{height}:r=30:d=0.1,format=yuv420p10le,"
         "geq=lum='64+876*X/(W-1)':cb=512:cr=512",
         "-c:v", "ffv1", *HLG.tags, str(base)],
        check=True,
    )
    _card(card)
    cube = write_sdr_to_hdr_cube(tmp_path / "graphics.cube", HLG)
    cmd = build_overlay_command(
        str(base), [Placement("card", str(card), 0.0, 0.1, 0.1)],
        str(tmp_path / "unused.mp4"), width=width, color=HLG, cube=str(cube),
        margin=0, fade=0, base_duration=0.1,
    )
    graph_end = cmd.index("-filter_complex") + 2
    raw = subprocess.check_output(
        cmd[:graph_end] + ["-map", "[vout]", "-frames:v", "1",
                          "-pix_fmt", "yuv420p10le", "-f", "rawvideo", "-"],
    )
    source = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", str(base), "-frames:v", "1",
         "-pix_fmt", "yuv420p10le", "-f", "rawvideo", "-"],
    )
    row_start = (height - 1) * width * 2
    got = struct.unpack_from(f"<{width}H", raw, row_start)
    expected = struct.unpack_from(f"<{width}H", source, row_start)
    assert max(abs(a - b) for a, b in zip(got, expected)) <= 1


# -- hardware decode ---------------------------------------------------------
def test_only_the_base_is_decoded_on_the_hardware(monkeypatch):
    monkeypatch.setattr("talkinghead.visuals.render.decode_args", lambda pix_fmt: ["-hwaccel", "videotoolbox"])
    c = cmd_for(Placement("a", "/a.png", 1.0, 5.0, 0.28), Placement("b", "/b.png", 6.0, 9.0, 0.28))
    assert c.count("-hwaccel") == 1
    assert c.index("-hwaccel") < c.index("base.mp4") < c.index("/a.png")


def test_a_stream_copy_decodes_nothing():
    c = build_overlay_command("base.mp4", [], "out.mp4", margin=40, fade=0.4, width=1920)
    assert "-hwaccel" not in c


@pytest.mark.skipif(not decode_args("p010le"), reason="this ffmpeg has no VideoToolbox decode")
def test_hardware_decode_gives_the_bake_the_software_frames(tmp_path):
    """The composited HLG frames are the same whichever decoder read the base, and the
    debug log proves VideoToolbox did the decoding rather than quietly falling back."""
    base, card = tmp_path / "base.mp4", tmp_path / "card.png"
    _bake_clip(base)
    _card(card)
    cube = write_sdr_to_hdr_cube(tmp_path / "graphics.cube", HLG)
    cmd = build_overlay_command(str(base), [Placement("card", str(card), 0.5, 1.5, 0.3)],
                                str(tmp_path / "unused.mp4"), width=320, color=HLG, cube=str(cube),
                                base_duration=2.0)
    assert cmd[cmd.index("-hwaccel") + 1] == "videotoolbox"

    def frames(c):
        graph_end = c.index("-filter_complex") + 2
        head = c[:graph_end]
        head[head.index("-v") + 1] = "debug"
        proc = subprocess.run(head + ["-map", "[vout]", "-pix_fmt", "yuv420p10le", "-f", "rawvideo", "-"],
                              capture_output=True, check=True)
        return proc.stdout, proc.stderr.decode(errors="replace")

    hard, log = frames(cmd)
    soft, _ = frames(without_hwaccel(cmd))
    assert "videotoolbox_vld" in log
    assert hard and hard == soft


def test_a_bake_the_hardware_cannot_decode_reruns_in_software(tmp_path, monkeypatch, capsys):
    """A real ffmpeg failure from an unknown hwaccel: the bake still lands, and the command
    the report records is the software one that ran."""
    monkeypatch.setattr("talkinghead.visuals.render.decode_args", lambda pix_fmt: ["-hwaccel", "no-such-hwaccel"])
    base, card, out = tmp_path / "base.mp4", tmp_path / "card.png", tmp_path / "out.mp4"
    _bake_clip(base, hdr=False)
    _card(card)
    cmd = render_visuals(str(base), [Placement("card", str(card), 0.5, 1.5, 0.3)], str(out))
    assert "-hwaccel" not in cmd
    assert _audio_hash(out) == _audio_hash(base)
    assert "software decode" in capsys.readouterr().err
