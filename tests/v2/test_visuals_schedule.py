from talkinghead.visuals.schedule import schedule


class R:
    def __init__(self, asset, out_t, hold, scale=0.28, score=0.0, sequence=None, order=0,
                 until=None):
        self.anchor = type("A", (), {"asset": asset, "hold": hold, "scale": scale,
                                     "score": score, "sequence": sequence, "order": order,
                                     "until": until})()
        self.out_t = out_t
        self.path = f"/{asset}.png"


def test_holds_are_trimmed_to_the_next_overlay():
    placed, dropped = schedule([R("a", 10.0, 60.0), R("b", 20.0, 5.0)],
                               out_duration=100.0, min_hold=1.5, min_gap=2.0)
    assert placed[0].end == 18.0
    assert dropped == []


def test_colliding_anchors_of_equal_merit_keep_the_longer_hold():
    """`a` and `b` are 1 s apart, so only one fits. With equal match scores the
    tie goes to the longer requested hold: the judge asked for more screen time
    on that topic. (Same-topic candidates are merged upstream by
    collapse_moments, so anything reaching here is genuinely two topics.)"""
    placed, dropped = schedule([R("a", 10.0, 30.0), R("b", 11.0, 5.0)],
                               out_duration=100.0, min_hold=1.5, min_gap=2.0)
    assert [p.asset for p in placed] == ["a"]
    assert len(dropped) == 1 and "within" in dropped[0]


def test_last_overlay_clamps_to_duration():
    placed, _ = schedule([R("a", 95.0, 30.0)], out_duration=100.0, min_hold=1.5, min_gap=2.0)
    assert placed[0].end == 100.0


def test_placements_never_overlap():
    placed, _ = schedule([R(str(i), i * 3.0, 10.0) for i in range(6)],
                         out_duration=100.0, min_hold=1.0, min_gap=2.0)
    for x, y in zip(placed, placed[1:]):
        assert y.start >= x.end


def test_output_is_in_time_order_regardless_of_input_order():
    placed, _ = schedule([R("b", 40.0, 5.0), R("a", 10.0, 5.0)],
                         out_duration=100.0, min_hold=1.5, min_gap=2.0)
    assert [p.asset for p in placed] == ["a", "b"]


def test_a_collision_is_settled_on_match_not_arrival_order():
    """The demo regression: s-zbench (0.75) lost its room to a generated Opus
    card (0.50) placed 2.7 s later, purely because it came second."""
    from talkinghead.visuals.schedule import thin_collisions
    kept, dropped = thin_collisions(
        [R("s-zbench", 30.0, 6.0, score=0.75), R("c-opus", 32.7, 6.0, score=0.50)],
        min_hold=1.5, min_gap=2.0)
    assert [r.anchor.asset for r in kept] == ["s-zbench"]
    assert "c-opus" in dropped[0]


def test_the_better_asset_survives_whichever_side_it_is_on():
    from talkinghead.visuals.schedule import thin_collisions
    kept, _ = thin_collisions(
        [R("c-weak", 30.0, 6.0, score=0.10), R("s-strong", 32.0, 6.0, score=0.90)],
        min_hold=1.5, min_gap=2.0)
    assert [r.anchor.asset for r in kept] == ["s-strong"]


def test_anchors_far_enough_apart_are_both_kept():
    from talkinghead.visuals.schedule import thin_collisions
    kept, dropped = thin_collisions(
        [R("a", 10.0, 5.0, score=0.5), R("b", 40.0, 5.0, score=0.9)],
        min_hold=1.5, min_gap=2.0)
    assert len(kept) == 2 and dropped == []


def test_a_run_of_three_collisions_thins_to_the_best():
    from talkinghead.visuals.schedule import thin_collisions
    kept, _ = thin_collisions(
        [R("a", 10.0, 5.0, score=0.2), R("b", 11.0, 5.0, score=0.8), R("c", 12.0, 5.0, score=0.4)],
        min_hold=1.5, min_gap=2.0)
    assert [r.anchor.asset for r in kept] == ["b"]


# ------------------------------------------------------- build-up sequences

def test_build_up_steps_chain_with_no_gap():
    """v11 showed only three of five pipeline stages: the steps are 1.5-3 s
    apart, so min_gap tore them apart and min_hold dropped them. A build-up is
    one visual that changes, so each step runs straight into the next."""
    steps = [R(f"p{i}", 158.0 + 1.6 * i, 8.0, score=1.0, sequence="pipeline", order=i)
             for i in range(1, 6)]
    placed, dropped = schedule(steps, out_duration=1357.0, min_hold=3.0, min_gap=2.0)
    assert [p.asset for p in placed] == ["p1", "p2", "p3", "p4", "p5"]
    assert dropped == []
    for x, y in zip(placed, placed[1:]):
        assert x.end == y.start          # exactly abutting, no gap


def test_the_last_step_of_a_build_up_gets_a_full_hold():
    steps = [R("p1", 100.0, 8.0, score=1.0, sequence="pipeline", order=1),
             R("p2", 102.0, 12.0, score=1.0, sequence="pipeline", order=2)]
    placed, _ = schedule(steps, out_duration=1000.0, min_hold=3.0, min_gap=2.0)
    assert placed[0].end == 102.0
    assert placed[-1].end - placed[-1].start == 12.0


def test_build_up_steps_never_thin_each_other():
    from talkinghead.visuals.schedule import thin_collisions
    steps = [R("p1", 100.0, 8.0, score=1.0, sequence="pipeline", order=1),
             R("p2", 101.0, 8.0, score=1.0, sequence="pipeline", order=2)]
    kept, dropped = thin_collisions(steps, min_hold=3.0, min_gap=2.0)
    assert len(kept) == 2 and dropped == []


def test_a_sequence_still_yields_to_an_unrelated_neighbour_on_merit():
    items = [R("p5", 100.0, 8.0, score=1.0, sequence="pipeline", order=5),
             R("c-weak", 101.0, 8.0, score=0.1)]
    placed, _ = schedule(items, out_duration=1000.0, min_hold=3.0, min_gap=2.0)
    assert [p.asset for p in placed] == ["p5"]


# ------------------------------------------------------- reserved windows

def test_a_content_overlay_yields_to_a_reserved_chapter_card():
    """Chapter cards own their corner; the content pass schedules around them."""
    placed, _ = schedule([R("q", 100.0, 5.0)], out_duration=200.0, min_hold=1.5, min_gap=2.0,
                         reserved=[(99.0, 108.0)])
    assert len(placed) == 1
    assert placed[0].start >= 108.0 + 2.0


def test_a_reserved_window_far_away_does_not_move_an_overlay():
    placed, _ = schedule([R("q", 50.0, 5.0)], out_duration=200.0, min_hold=1.5, min_gap=2.0,
                         reserved=[(99.0, 108.0)])
    assert placed[0].start == 50.0


def test_no_room_beside_a_reserved_wall_drops_the_overlay():
    placed, dropped = schedule([R("q", 190.0, 5.0)], out_duration=200.0, min_hold=1.5,
                               min_gap=2.0, reserved=[(180.0, 200.0)])
    assert placed == [] and dropped


def test_a_build_up_yields_the_corner_to_a_chapter_card():
    """A library sequence must not punch through a reserved chapter card: the
    card owns its corner even against a build-up."""
    steps = [R("p1", 100.0, 8.0, score=1.0, sequence="pipeline", order=1),
             R("p2", 104.0, 8.0, score=1.0, sequence="pipeline", order=2)]
    placed, dropped = schedule(steps, out_duration=200.0, min_hold=3.0, min_gap=2.0,
                               reserved=[(99.0, 106.0)])
    assert placed == []
    assert len(dropped) == 2


def test_a_fitting_final_build_up_step_is_not_dropped_at_a_tight_end():
    """The final step abuts its predecessor; treating that predecessor as a
    blocker pushed the step past the cut and dropped a fitting overlay."""
    steps = [R("p1", 100.0, 8.0, score=1.0, sequence="pipeline", order=1),
             R("p2", 104.0, 8.0, score=1.0, sequence="pipeline", order=2)]
    placed, dropped = schedule(steps, out_duration=110.0, min_hold=3.0, min_gap=2.0)
    assert [p.asset for p in placed] == ["p1", "p2"]
    assert placed[-1].end == 110.0
    assert dropped == []


def test_an_overlay_is_confined_to_its_until_bound():
    placed, _ = schedule([R("q", 10.0, 30.0, until=18.0)], out_duration=100.0,
                         min_hold=3.0, min_gap=2.0)
    assert len(placed) == 1 and placed[0].end == 18.0


def test_an_overlay_that_cannot_fit_before_until_is_dropped():
    placed, dropped = schedule([R("q", 17.0, 30.0, until=18.0)], out_duration=100.0,
                               min_hold=3.0, min_gap=2.0)
    assert placed == [] and dropped
