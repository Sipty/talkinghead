"""Collapsing near-duplicate candidates for the same moment.

Sampling the judge three times names the same beat three ways. On the 4-minute
demo that produced 18 candidates for 7 real moments, and 11 were reported
'dropped: no room' while fighting each other for one slot. Worse, the real v5
screenshot for GLM (match 0.75) lost its slot to a generated card for the same
beat, which inverts the earned-reuse rule.
"""

from talkinghead.visuals.library import tokens
from talkinghead.visuals.stage import collapse_moments, pick_winner


class FakeAsset:
    def __init__(self, id, sequence=None, order=0):
        self.id, self.sequence, self.order = id, sequence, order


def cand(word, title, depicts, score=0.0, hold=5.0, asset=None):
    return {"word": word, "title": title, "depicts": depicts, "score": score, "hold": hold,
            "title_key": " ".join(title.lower().split()), "depicts_tokens": tokens(depicts),
            "asset": asset}


def test_identical_titles_nearby_collapse():
    items = [cand(100, "Loop engineering", "term card for loop engineering buzzword"),
             cand(104, "Loop Engineering", "text plate reading loop engineering")]
    assert collapse_moments(items) == [[0, 1]]


def test_similar_depicts_collapse_even_without_a_shared_title():
    items = [cand(300, "", "GLM 5.3 Flash model card and benchmark chart"),
             cand(303, "GLM 5.3 Flash", "GLM 5.3 Flash model card and coding benchmark chart")]
    assert collapse_moments(items) == [[0, 1]]


def test_two_different_models_in_the_same_breath_stay_separate():
    """GLM and Opus are named seconds apart and must not merge."""
    items = [cand(300, "GLM 5.3 Flash", "GLM 5.3 Flash model card and benchmark chart"),
             cand(310, "Claude Opus 4.8", "Claude Opus 4.8 model card and coding benchmark comparison")]
    assert collapse_moments(items) == [[0], [1]]


def test_same_title_far_apart_stays_separate():
    items = [cand(100, "15 years", "credential card"), cand(900, "15 years", "credential card")]
    assert collapse_moments(items) == [[0], [1]]


def test_three_way_collapse():
    items = [cand(500, "Vibecode", "definition plate vibecoding means applying strong engineering"),
             cand(505, "Vibecode", "definition card redefining vibecode as applying strong engineering"),
             cand(508, "Vibecode", "definition card vibecoding defined as applying strong engineering")]
    assert collapse_moments(items) == [[0, 1, 2]]


def test_the_highest_match_wins_so_reuse_is_earned():
    """The exact demo regression: a reusable v5 asset must beat a fresh card."""
    items = [cand(300, "GLM 5.3 Flash", "GLM 5.3 Flash card", score=0.62),
             cand(303, "", "GLM 5.3 Flash model card and benchmark chart", score=0.75)]
    assert pick_winner([0, 1], items) == 1


def test_longer_hold_breaks_a_match_tie():
    items = [cand(300, "A", "a thing", score=0.5, hold=4.0),
             cand(302, "A", "a thing", score=0.5, hold=9.0)]
    assert pick_winner([0, 1], items) == 1


def test_single_candidate_is_its_own_group():
    assert collapse_moments([cand(1, "A", "x")]) == [[0]]


def test_no_candidates_is_no_groups():
    assert collapse_moments([]) == []


# ------------------------------------------------- assets and build-up steps

def test_the_same_asset_twice_is_always_one_moment():
    """s-cache collided with itself -- seeded at match 1.00 and judge-matched
    at 0.80 -- and one copy was reported dropped for "no room"."""
    a = FakeAsset("s-cache")
    items = [cand(100, "", "Anthropic prompt caching docs", score=1.0, asset=a),
             cand(104, "Caching", "the five minute TTL mistake", score=0.8, asset=a)]
    assert collapse_moments(items) == [[0, 1]]


def test_different_steps_of_a_build_up_stay_separate():
    p1 = FakeAsset("p1-scout", sequence="pipeline", order=1)
    p2 = FakeAsset("p2-planner", sequence="pipeline", order=2)
    items = [cand(100, "", "agent pipeline diagram scout stage", score=1.0, asset=p1),
             cand(103, "", "agent pipeline diagram scout to planner", score=1.0, asset=p2)]
    assert collapse_moments(items) == [[0], [1]]
