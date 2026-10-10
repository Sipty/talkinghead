from talkinghead.visuals.schedule import Placement
from talkinghead.visuals.verify import control_times


def p(start, end):
    return Placement("a", "/a.png", start, end, 0.28)


def test_controls_avoid_overlay_windows():
    places = [p(20.0, 30.0), p(45.0, 60.0)]
    ts = control_times(places, 100.0)
    for t in ts:
        assert all(not (a - 1.0 <= t <= b + 1.0) for a, b in [(20.0, 30.0), (45.0, 60.0)])


def test_control_at_a_quarter_is_skipped_when_busy():
    """v5 had to move its control frame because 900 s fell inside an overlay."""
    ts = control_times([p(24.0, 26.0)], 100.0)
    assert 25.0 not in ts
    assert len(ts) == 2


def test_two_controls_by_default():
    assert len(control_times([], 100.0)) == 2


def test_no_control_found_returns_empty_rather_than_a_bad_time():
    ts = control_times([p(0.0, 100.0)], 100.0)
    assert ts == []
