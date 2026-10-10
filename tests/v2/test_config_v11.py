import pytest

from talkinghead.config import load_config


def test_marker_voting_defaults():
    c = load_config()
    assert (c.marker_votes, c.marker_agree) == (3, 2)


def test_visuals_defaults():
    c = load_config()
    assert c.visuals_enabled is True
    assert c.visuals_research is True
    assert c.visuals_match_threshold == 0.70
    # Cards carry the mascot and a speech bubble, so they need more room than
    # the plain plate they replaced.
    assert (c.visuals_scale_card, c.visuals_scale_shot) == (0.46, 0.42)
    assert (c.visuals_margin, c.visuals_fade) == (40, 0.4)


def test_marker_agree_cannot_exceed_marker_votes():
    with pytest.raises(ValueError):
        load_config(marker_votes=2, marker_agree=3)


def test_visuals_can_be_disabled_by_override():
    assert load_config(visuals_enabled=False).visuals_enabled is False
