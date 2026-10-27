"""Time maps agree with the frame-snapped cut the renderer makes."""
from __future__ import annotations

import pytest

from talkinghead.render import _snap
from talkinghead.timeline import Plan, Segment, SourceInfo
from talkinghead.timemap import locate, map_span, snapped_segments, window_segments

FPS = 30.0
SRC = SourceInfo(id="main", path="/v/in.mp4", sha256="abc", duration=100.0)


def _plan(*spans: tuple[float, float]) -> Plan:
    return Plan(sources={"main": SRC}, main="main", cuts=[], segments=[Segment("main", a, b) for a, b in spans])


def _three() -> Plan:
    # Output: 0-2 plays source 1-3, 2-4 plays 5-7, 4-6 plays 10-12.
    return _plan((1.0, 3.0), (5.0, 7.0), (10.0, 12.0))


def test_a_snapped_segments_match_the_renderer_off_grid_and_pass_through_at_fps_zero() -> None:
    plan = _plan((1.013, 3.021), (5.0, 7.0))
    assert snapped_segments(plan, FPS) == _snap(plan.segments, FPS)
    assert snapped_segments(plan, FPS) != plan.segments
    assert snapped_segments(plan, 0) == plan.segments


def test_a_locate_gives_source_and_time_for_interiors_joins_and_none_past_the_end() -> None:
    plan = _three()
    assert locate(plan, 0.5, fps=FPS) == ("main", pytest.approx(1.5))
    assert locate(plan, 3.0, fps=FPS) == ("main", pytest.approx(6.0))
    assert locate(plan, 5.5, fps=FPS) == ("main", pytest.approx(11.5))
    assert locate(plan, 0.0, fps=FPS) == ("main", pytest.approx(1.0))
    assert locate(plan, 2.0, fps=FPS) == ("main", pytest.approx(5.0))
    assert locate(plan, 4.0, fps=FPS) == ("main", pytest.approx(10.0))
    assert locate(plan, 6.0, fps=FPS) is None
    assert locate(plan, 9.0, fps=FPS) is None
    assert locate(plan, -0.1, fps=FPS) is None


def test_a_locate_uses_the_snapped_cut_not_the_plan_segments() -> None:
    plan = _plan((1.013, 3.0), (5.0, 7.0))
    src = locate(plan, 0.0, fps=FPS)
    assert src is not None and src[1] == pytest.approx(round(1.013 * FPS) / FPS)


def test_a_map_span_reports_kept_trimmed_split_and_cut() -> None:
    plan = _three()
    kept = map_span(plan, 1.5, 2.5, fps=FPS)
    assert kept == {"status": "kept", "pieces": [[0.5, 1.5]], "out_start": 0.5, "out_end": 1.5}
    trimmed = map_span(plan, 2.5, 3.5, fps=FPS)
    assert trimmed == {"status": "trimmed", "pieces": [[1.5, 2.0]], "out_start": 1.5, "out_end": 2.0}
    split = map_span(plan, 2.0, 6.0, fps=FPS)
    assert split["status"] == "split"
    assert split["pieces"] == [[1.0, 2.0], [2.0, 3.0]]
    assert split["out_start"] == 1.0 and split["out_end"] == 3.0
    cut = map_span(plan, 3.5, 4.5, fps=FPS)
    assert cut == {"status": "cut", "pieces": [], "out_start": None, "out_end": None}


def test_a_map_span_merges_pieces_that_touch_in_output_and_source() -> None:
    plan = _plan((1.0, 3.0), (3.0, 5.0))
    assert map_span(plan, 2.0, 4.0, fps=FPS)["pieces"] == [[1.0, 3.0]]
    assert map_span(plan, 2.0, 4.0, fps=FPS)["status"] == "kept"


def test_a_map_span_rejects_empty_spans_and_unknown_sources() -> None:
    plan = _three()
    with pytest.raises(ValueError):
        map_span(plan, 2.0, 2.0, fps=FPS)
    with pytest.raises(ValueError):
        map_span(plan, 1.0, 2.0, fps=FPS, source="nope")


def test_a_map_span_on_a_reordered_plan_returns_where_the_span_really_plays() -> None:
    plan = _plan((10.0, 12.0), (1.0, 3.0))
    early = map_span(plan, 1.5, 2.5, fps=FPS)
    assert early["pieces"] == [[2.5, 3.5]] and early["status"] == "kept"
    late = map_span(plan, 10.5, 11.5, fps=FPS)
    assert late["pieces"] == [[0.5, 1.5]]
    both = map_span(plan, 2.5, 11.0, fps=FPS)
    assert both["status"] == "split"
    assert both["pieces"] == [[0.0, 1.0], [3.5, 4.0]]


def test_a_window_segments_across_a_join_sum_to_the_frame_rounded_window() -> None:
    plan = _three()
    segs = window_segments(plan, 1.5, 2.5, fps=FPS)
    assert [(s.source, round(s.start, 6), round(s.end, 6)) for s in segs] == [("main", 2.5, 3.0), ("main", 5.0, 5.5)]
    assert sum(s.length for s in segs) == pytest.approx(1.0)
    off = window_segments(plan, 1.51, 2.49, fps=FPS)
    assert sum(s.length for s in off) == pytest.approx(round(2.49 * FPS) / FPS - round(1.51 * FPS) / FPS)


def test_a_window_segments_clamp_the_end_and_reject_empty_or_past_the_end_windows() -> None:
    plan = _three()
    segs = window_segments(plan, 5.0, 99.0, fps=FPS)
    assert sum(s.length for s in segs) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        window_segments(plan, 2.0, 2.0, fps=FPS)
    with pytest.raises(ValueError):
        window_segments(plan, 2.0, 2.01, fps=FPS)
    with pytest.raises(ValueError):
        window_segments(plan, 6.0, 7.0, fps=FPS)
