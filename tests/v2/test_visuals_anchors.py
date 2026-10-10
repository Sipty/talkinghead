from talkinghead.visuals.anchors import (
    Anchor,
    find_phrase,
    find_quote,
    quote_token,
    resolve_output,
)


class W:
    def __init__(self, text, start, end):
        self.text, self.start, self.end = text, start, end


def stream(s, step=1.0):
    return [W(t, i * step, i * step + 0.5) for i, t in enumerate(s.split())]


def a(text, hold=5.0):
    return Anchor(text, "asset", hold, 0.28)


def test_anchor_resolves_to_its_own_word_start():
    """Anchors are output-timeline phrases: the matched word's start IS out_t."""
    r = resolve_output(a("three four"), stream("one two three four five"))
    assert r.out_t == 2.0
    assert r.word == 2


def test_output_anchor_never_shifts_for_cut_material():
    """The words are the final cut, so a later anchor keeps its own start; no
    Plan maps it back to a source time and no removed seconds move it."""
    r = resolve_output(a("five"), stream("one two three four five"))
    assert r.out_t == 4.0


def test_punctuation_and_case_insensitive():
    words = stream("GLM 53 Flash, which is competing")
    r = resolve_output(a("glm 53 flash which is"), words)
    assert r.out_t == 0.0


def test_absent_phrase_is_none():
    assert resolve_output(a("nowhere here"), stream("one two")) is None


def test_find_phrase_returns_first_occurrence():
    assert find_phrase(stream("a b a b"), "a b") == 0


def test_quote_token_keeps_currency_and_signs():
    assert quote_token("$1.50") == "$1.50"
    assert quote_token("-10") == "-10"
    assert quote_token("point.") == "point"
    assert quote_token("\u201cQuote\u201d") == "quote"


def test_find_quote_rejects_a_punctuation_stripped_match():
    words = stream("I pay $1.50 a day")
    assert find_quote(words, "150") is None
    assert find_quote(words, "$1.50") == (2, 3)


def test_find_quote_finds_a_sentence_and_ignores_its_period():
    words = stream("that is the whole point.")
    assert find_quote(words, "that is the whole point") == (0, 5)


def test_quote_matches_a_spoken_number_with_a_sentence_period():
    words = stream("I pay 150.")
    assert quote_token("150.") == "150"
    assert find_quote(words, "150") == (2, 3)


def test_quote_matches_a_spoken_money_with_a_sentence_period():
    words = stream("it cost $1.50.")
    assert quote_token("$1.50.") == "$1.50"
    assert find_quote(words, "$1.50") == (2, 3)
