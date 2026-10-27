import pytest

from talkinghead.color import ColorProfile
from talkinghead.visuals.render import Overlays, _even, overlay_graph, overlay_inputs
from talkinghead.visuals.schedule import Placement


def graph_for(*placements, base="[0:v]", first_input=1, width=1920, color=None, cube=None, out="[vout]", **kw):
    """The overlay chain from the builder directly: `kw` are the Overlays fields."""
    return ";".join(overlay_graph(Overlays(tuple(placements), **kw), base=base, first_input=first_input,
                                  width=width, color=color or ColorProfile(), cube=cube, out=out))


def inputs_for(*placements, fps=30, duration=None):
    return overlay_inputs(Overlays(tuple(placements)), fps=fps, duration=duration)


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


HLG = ColorProfile(primaries="bt2020", transfer="arib-std-b67", matrix="bt2020nc",
                   range="tv", pix_fmt="yuv420p10le", hdr=True)
