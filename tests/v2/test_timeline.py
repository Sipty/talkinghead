"""Timeline: cuts on the source timeline, the EDL derived from them, and the
time maps between source and output. Pure functions, no I/O."""

import json

import pytest

from talkinghead.timeline import Cut, Plan, Segment, SourceInfo, cuts_to_segments, merge_cuts

SRC = SourceInfo(id="main", path="/v/in.mp4", sha256="abc", duration=100.0)


def test_merge_cuts_fuses_overlaps_and_touches_and_clips_to_source() -> None:
    cuts = [
        Cut(10.0, 12.0, "silence"),
        Cut(11.5, 13.0, "filler"),      # overlaps -> fused
        Cut(13.0, 14.0, "silence"),     # touches -> fused
        Cut(-1.0, 0.5, "silence"),      # clipped to 0
        Cut(99.0, 120.0, "outro"),      # clipped to duration
        Cut(50.0, 50.01, "silence"),    # too short -> dropped
    ]
    merged = merge_cuts(cuts, duration=100.0)
    assert [(c.start, c.end) for c in merged] == [(0.0, 0.5), (10.0, 14.0), (99.0, 100.0)]


def test_merge_cuts_keeps_the_most_informative_reason_and_joins_texts() -> None:
    merged = merge_cuts([Cut(1.0, 2.0, "restart", "take one"), Cut(1.5, 3.0, "silence", "")], 10.0)
    assert merged[0].reason == "restart"
    assert merged[0].text == "take one"
    # A head-silence trim fused into the cold-open cut reports as the decision it is.
    merged = merge_cuts([Cut(0.0, 0.5, "silence"), Cut(0.0, 1.6, "cold-open")], 10.0)
    assert merged == [Cut(0.0, 1.6, "cold-open")]


def test_cuts_to_segments_is_the_complement() -> None:
    segs = cuts_to_segments([Cut(0.0, 1.0, "cold-open"), Cut(10.0, 12.0, "silence")], SRC)
    assert segs == [Segment("main", 1.0, 10.0), Segment("main", 12.0, 100.0)]


def test_plan_output_duration_is_source_minus_removed() -> None:
    plan = Plan.from_cuts(SRC, [Cut(0.0, 1.0, "cold-open"), Cut(10.0, 12.0, "silence")])
    assert plan.output_duration == pytest.approx(97.0)


def test_source_to_output_mapping_shifts_by_removed_time_and_hides_cut_interiors() -> None:
    plan = Plan.from_cuts(SRC, [Cut(0.0, 1.0, "cold-open"), Cut(10.0, 12.0, "silence")])
    assert plan.source_to_output(5.0) == pytest.approx(4.0)
    assert plan.source_to_output(11.0) is None
    assert plan.source_to_output(20.0) == pytest.approx(17.0)
    assert plan.output_to_source(17.0) == pytest.approx(20.0)
    assert plan.output_to_source(9.0) == pytest.approx(12.0)  # at a join, the frame playing is the next segment


def test_joins_are_output_times_of_interior_cuts_only() -> None:
    plan = Plan.from_cuts(SRC, [Cut(0.0, 1.0, "cold-open"), Cut(10.0, 12.0, "silence"), Cut(90.0, 100.0, "outro")])
    assert plan.joins() == [pytest.approx(9.0)]


def test_json_round_trip_is_exact_and_byte_stable() -> None:
    plan = Plan.from_cuts(SRC, [Cut(10.0, 12.3456789, "silence", "x", provenance={"stage": "pauses"})])
    plan.anchors["hook"] = 104.4
    plan.decisions["cold_open"] = {"chosen": "S0012", "votes": 5}
    text = plan.to_json()
    again = Plan.from_json(text)
    assert again == plan or again.to_json() == text
    assert Plan.from_json(text).to_json() == text
    data = json.loads(text)
    assert data["schema_version"] == 1
    assert data["cuts"][0]["end"] == 12.346  # millisecond precision on disk


def test_segments_from_another_source_extend_the_output() -> None:
    plan = Plan.from_cuts(SRC, [Cut(0.0, 1.0, "cold-open")])
    intro = SourceInfo(id="intro", path="/v/intro.mp4", sha256="def", duration=5.0)
    plan.sources[intro.id] = intro
    plan.segments.insert(0, Segment("intro", 0.0, 5.0))
    assert plan.output_duration == pytest.approx(104.0)
    # main-source mapping accounts for the prepended segment
    assert plan.source_to_output(5.0) == pytest.approx(9.0)
