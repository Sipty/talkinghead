from talkinghead.markers import find_markers


class W:
    def __init__(self, text):
        self.text = text


def words(s):
    return [W(t) for t in s.split()]


def test_strong_marker_found():
    c = find_markers(words("passes over to a reviewer, sorry, uh so a scout"))
    assert [(x.marker, x.tier) for x in c] == [("sorry", "strong")]


def test_weak_marker_tiered_weak():
    c = find_markers(words("by vibecode, I mean, you know, apply strong practices"))
    assert [(x.marker, x.tier) for x in c] == [("i mean", "weak")]


def test_multiword_marker():
    c = find_markers(words("no wait let me say that again the planner runs"))
    assert [x.marker for x in c] == ["no wait", "let me say that again"]


def test_marker_word_index_points_at_marker_start():
    c = find_markers(words("a b sorry c"))
    assert c[0].word == 2


def test_case_and_punctuation_insensitive():
    assert [x.marker for x in find_markers(words("Sorry, guys"))] == ["sorry"]
