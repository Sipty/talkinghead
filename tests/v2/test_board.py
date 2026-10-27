"""Pure consumer-facing behavior for board specifications and timeline geometry."""

import pytest

from talkinghead.board.render import (
    appear_alpha,
    ease,
    expand_rect,
    map_view_to_output,
    render_board,
    source_at_output,
    view_rect_at,
)
from talkinghead.board.spec import BoardView, load_spec, save_spec, validate
from talkinghead.timeline import Segment


def sample_spec():
    return {
        "version": 1,
        "tablet": "/tmp/tablet.mp4",
        "offset": 55,
        "canvas": [236, 20, 1906, 1570],
        "masks": [[1920, 40, 415, 88]],
        "panel": [0.515, 0.155, 0.465, 0.69],
        "transition": 0.8,
        "views": [],
        "notes": "test",
    }


@pytest.mark.parametrize(("change", "message"), [
    ({"version": 2}, "version must be 1"),
    ({"canvas": [0, 0, 10, 0]}, "canvas width and height must be positive"),
    ({"panel": [0.8, 0.2, 0.3, 0.4]}, "panel must"),
    ({"views": [{"start": 1, "end": 2, "rect": [0.8, 0.2, 0.3, 0.4]}]}, r"views\[0\].rect"),
    ({"views": [{"start": 3, "end": 4, "rect": [0, 0, 1, 1]},
                {"start": 2, "end": 3, "rect": [0, 0, 1, 1]}]}, "sorted and non-overlapping"),
    ({"views": [{"start": 1, "end": 3, "rect": [0, 0, 1, 1]},
                {"start": 2, "end": 4, "rect": [0, 0, 1, 1]}]}, "sorted and non-overlapping"),
])
def test_validate_rejects_invalid_data(change, message):
    data = sample_spec() | change
    with pytest.raises(ValueError, match=message):
        validate(data)


def test_validate_accepts_schema_and_normalizes_nested_values():
    parsed = validate(sample_spec() | {"views": [{"start": 1.5, "end": 3.0,
                                                     "rect": [0.1, 0.2, 0.3, 0.4],
                                                     "label": "Code"}]})
    assert parsed.version == 1
    assert parsed.views == (BoardView(1.5, 3.0, (0.1, 0.2, 0.3, 0.4), "Code"),)
    assert parsed.to_json()["canvas"] == [236, 20, 1906, 1570]



def test_appear_alpha_fades_from_zero_to_one():
    assert appear_alpha(9, 10) == 0
    assert appear_alpha(10, 10) == 0
    assert appear_alpha(10.2, 10) == pytest.approx(0.5)
    assert appear_alpha(10.4, 10) == 1
    assert appear_alpha(20, None) == 1


@pytest.mark.parametrize(("change", "message"), [
    ({"appear": -0.1}, "appear must be non-negative"),
    ({"ink": []}, "unknown board spec field: ink"),
])
def test_validate_rejects_invalid_appear_and_unknown_ink(change, message):
    with pytest.raises(ValueError, match=message):
        validate(sample_spec() | change)


def test_old_spec_defaults_appear_and_round_trips():
    old = validate(sample_spec())
    assert old.appear is None
    parsed = validate(sample_spec() | {"appear": 12.5})
    assert parsed.to_json()["appear"] == 12.5


def test_save_spec_round_trips_appear(tmp_path):
    spec = validate(sample_spec() | {"appear": 10})
    path = save_spec(spec, tmp_path / "board.json")
    loaded = load_spec(path)
    assert loaded.appear == 10

def test_source_time_views_map_across_removed_region_and_fully_cut_views_drop():
    segments = [Segment("main", 0, 10), Segment("main", 20, 30), Segment("main", 40, 50)]
    spanning = BoardView(5, 25, (0.2, 0.2, 0.4, 0.4), "span")
    removed = BoardView(11, 19, (0.1, 0.1, 0.2, 0.2), "removed")
    assert map_view_to_output(spanning, segments) == [
        BoardView(5, 10, spanning.rect, "span"), BoardView(10, 15, spanning.rect, "span")]
    assert map_view_to_output(removed, segments) == []
    assert source_at_output(10, segments) == 20
    assert source_at_output(15, segments) == 25


def test_view_rect_expands_to_panel_aspect_and_clamps_at_edges():
    # Canvas aspect 2, panel aspect 1, so the crop grows vertically.
    expanded = expand_rect((0.8, 0.8, 0.2, 0.2), panel_aspect=1, canvas_aspect=2)
    assert expanded[0] == pytest.approx(0.8)
    assert expanded[1] == pytest.approx(0.6)
    assert expanded[2:] == pytest.approx((0.2, 0.4))
    # A narrow region expands horizontally, then clamps to the left edge.
    expanded = expand_rect((0.0, 0.4, 0.1, 0.2), panel_aspect=2, canvas_aspect=1)
    assert expanded == pytest.approx((0.0, 0.4, 0.4, 0.2))


def test_easing_is_monotonic_and_hits_exact_endpoints():
    values = [ease(i / 100) for i in range(101)]
    assert values[0] == 0
    assert values[-1] == 1
    assert all(a <= b for a, b in zip(values, values[1:]))


def test_full_canvas_before_first_view_and_after_last():
    views = (BoardView(10, 20, (0.2, 0.3, 0.4, 0.5), "A"),)
    assert view_rect_at(5, views, 0.8) == (0, 0, 1, 1)
    assert view_rect_at(25, views, 0.8) == (0, 0, 1, 1)
    immediate = (BoardView(0, 10, (0.2, 0.3, 0.4, 0.5), "A"),)
    assert view_rect_at(0, immediate, 0.8) == (0, 0, 1, 1)
    assert view_rect_at(0.8, immediate, 0.8) == immediate[0].rect
    assert view_rect_at(15, views, 0) == views[0].rect


def test_a_visuals_cook_without_final_mp4_tells_the_user_to_rerun_with_no_visuals(tmp_path):
    (tmp_path / "final-visuals.mp4").write_bytes(b"x")
    spec = validate(sample_spec())
    with pytest.raises(FileNotFoundError, match="--no-visuals"):
        render_board(tmp_path, spec)


def test_a_run_dir_without_any_cut_keeps_the_plain_message(tmp_path):
    spec = validate(sample_spec())
    with pytest.raises(FileNotFoundError, match="verified base cut not found") as info:
        render_board(tmp_path, spec)
    assert "--no-visuals" not in str(info.value)


def _board_run_dir(tmp_path, *, lock_plan_sha=None, lock_md5=None, write_lock=True):
    """A run dir with a one-segment plan, a real `final.mp4` and (optionally) its lock."""
    import json
    import subprocess

    from talkinghead.lock import audio_stream_md5, plan_digest
    from talkinghead.timeline import Plan, SourceInfo

    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=30:duration=1",
                    "-f", "lavfi", "-i", "sine=d=1", "-c:v", "libx264", "-c:a", "aac",
                    str(tmp_path / "final.mp4")], check=True)
    plan = Plan(sources={"main": SourceInfo("main", "/v/in.mp4", "sha", 1.0)}, main="main", cuts=[],
                segments=[Segment("main", 0.0, 1.0)])
    (tmp_path / "plan.json").write_text(plan.to_json())
    tablet = tmp_path / "tablet.mp4"
    tablet.write_bytes(b"x")
    if write_lock:
        (tmp_path / "edit-lock.json").write_text(json.dumps({
            "version": 1, "plan_sha": lock_plan_sha or plan_digest(plan),
            "audio": {"stream_md5": lock_md5 or audio_stream_md5(tmp_path / "final.mp4")}}))
    return validate(sample_spec() | {"tablet": str(tablet)})


@pytest.mark.parametrize("fault", ["audio", "plan", "no_lock"])
def test_a_board_refuses_a_stale_base_cut_and_names_the_fix_stale_base(tmp_path, fault):
    spec = _board_run_dir(tmp_path, lock_md5="0" * 32 if fault == "audio" else None,
                          lock_plan_sha="f" * 64 if fault == "plan" else None, write_lock=fault != "no_lock")
    error = FileNotFoundError if fault == "no_lock" else ValueError
    with pytest.raises(error, match="--no-visuals"):
        render_board(tmp_path, spec)
