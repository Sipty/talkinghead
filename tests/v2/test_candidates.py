"""Code-generated candidates: the only things a judge may vote on."""

import pytest

from talkinghead.candidates import (
    Candidate,
    cold_open_candidates,
    machinery_candidates,
    outro_candidates,
    sentence_table,
)
from talkinghead.transcribe import Word

WORDS = [
    Word("Hold", 0.00, 0.30, 1.0), Word("on.", 0.30, 0.60, 1.0),
    Word("Are", 2.00, 2.20, 1.0), Word("you", 2.20, 2.40, 1.0), Word("tired?", 2.40, 2.90, 1.0),
    Word("Um", 3.60, 3.90, 1.0),
    Word("Are", 4.50, 4.70, 1.0), Word("you", 4.70, 4.90, 1.0), Word("tired", 4.90, 5.20, 1.0), Word("yet?", 5.20, 5.60, 1.0),
    Word("Second", 6.00, 6.40, 1.0), Word("point.", 6.40, 6.80, 1.0),
    Word("Goodbye.", 8.00, 8.50, 1.0),
]
DURATION = 12.0
SILENCES = [(0.6, 2.0), (2.9, 3.6), (3.9, 4.5), (5.6, 6.0), (6.8, 8.0), (8.5, 12.0)]


def test_sentence_table_has_stable_ids_and_times() -> None:
    sents = sentence_table(WORDS)
    assert [s.sid for s in sents] == ["S0000", "S0001", "S0002", "S0003", "S0004"]
    assert sents[1].text == "Are you tired?" and sents[1].start == 2.0 and sents[1].end == 2.9


def test_sentence_table_also_splits_at_long_pauses_when_the_asr_forgot_the_punctuation() -> None:
    # Parakeet glued a 77 s rehearsal into one "sentence" on the real source. A
    # pause of >= split_pause between words is a sentence boundary too.
    words = [Word("Have", 0.0, 0.3, 1.0), Word("you", 0.3, 0.5, 1.0), Word("heard", 0.5, 0.9, 1.0),
             Word("hello", 2.0, 2.4, 1.0), Word("are", 2.5, 2.7, 1.0), Word("you", 2.7, 2.9, 1.0),
             Word("tired", 4.0, 4.4, 1.0)]
    sents = sentence_table(words, split_pause=0.5)
    assert [(s.sid, s.text) for s in sents] == [("S0000", "Have you heard"), ("S0001", "hello are you"), ("S0002", "tired")]
    assert [s.text for s in sentence_table(words)] == ["Have you heard hello are you tired"]


def test_machinery_candidates_split_by_kind_with_stable_ids() -> None:
    cands = machinery_candidates(DURATION, WORDS, SILENCES)
    kinds = {c.kind for c in cands}
    assert {"silence", "filler"} <= kinds
    filler = [c for c in cands if c.kind == "filler"]
    assert len(filler) == 1 and filler[0].text == "um"
    assert 3.4 <= filler[0].start <= 3.6 and 3.9 <= filler[0].end <= 4.1
    ids = [c.id for c in cands]
    assert len(set(ids)) == len(ids)
    assert [c.start for c in cands] == sorted(c.start for c in cands)
    assert all(c.id.startswith({"silence": "P", "filler": "F", "restart": "R"}[c.kind]) for c in cands)
    # Deterministic: the same inputs give the same list.
    assert machinery_candidates(DURATION, WORDS, SILENCES) == cands


def test_cold_open_candidates_are_every_sentence_in_the_head_window() -> None:
    # The judge sees the whole transcript anyway; offering every sentence that
    # starts inside the window means the true start is always on the list.
    sents = sentence_table(WORDS)
    assert [s.sid for s in cold_open_candidates(sents, window_sec=7.0)] == ["S0000", "S0001", "S0002", "S0003"]
    assert [s.sid for s in cold_open_candidates(sents, window_sec=3.0)] == ["S0000", "S0001"]


def test_outro_candidates_are_the_last_sentences_in_the_tail_window() -> None:
    sents = sentence_table(WORDS)
    cands = outro_candidates(sents, DURATION, window_sec=6.0)
    assert [s.sid for s in cands] == ["S0003", "S0004"]


def test_candidate_is_a_plain_record() -> None:
    c = Candidate("F0000", 1.0, 1.5, "filler", "um")
    assert c.length == pytest.approx(0.5)


def test_sentence_split_ignores_stamp_gaps_inside_a_continuous_burst() -> None:
    # "Are you tired | of" had a 0.72 s STAMP gap while the voice never stopped;
    # a split is allowed only after a word that acoustically ends its burst.
    words = [Word("Are", 0.0, 0.2, 1.0), Word("you", 0.2, 0.4, 1.0), Word("tired", 0.4, 0.8, 1.0),
             Word("of", 1.6, 1.8, 1.0), Word("Anthropic.", 1.8, 2.4, 1.0), Word("Well", 3.2, 3.5, 1.0)]
    burst_last = [False, False, False, False, True, True]
    sents = sentence_table(words, split_pause=0.5, burst_last=burst_last)
    assert [s.text for s in sents] == ["Are you tired of Anthropic.", "Well"]
