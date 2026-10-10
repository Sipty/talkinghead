"""Placing the opening chapter card in an early, free slot."""

from talkinghead.visuals.schedule import free_slot_after


def test_opening_slot_is_early_not_late():
    """Reusing the before-a-chapter search put the opening card 3:35 into the
    video, because that was the latest free gap before chapter one."""
    got = free_slot_after(2.0, hold=12.0, taken=[], min_gap=2.0, limit=226.0)
    assert got == 2.0


def test_opening_slot_steps_past_an_early_overlay():
    got = free_slot_after(2.0, hold=12.0, taken=[(1.0, 10.0)], min_gap=2.0, limit=226.0)
    assert got == 12.0


def test_opening_slot_gives_up_rather_than_overrun_the_limit():
    assert free_slot_after(2.0, hold=12.0, taken=[(0.0, 20.0)], min_gap=2.0, limit=25.0) is None


def test_a_gap_between_two_overlays_is_usable():
    got = free_slot_after(2.0, hold=6.0, taken=[(0.0, 10.0), (20.0, 30.0)], min_gap=2.0, limit=60.0)
    assert got == 12.0
