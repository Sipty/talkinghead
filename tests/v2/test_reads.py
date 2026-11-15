"""The two proposal reads work on the kept cut as the viewer hears it: the projection must follow the plan's
order and clocks, and a judge row that names lines the table never offered must not become a proposal."""

from talkinghead.judge import FakeJudge
from talkinghead.reads import cold_read, kept_lines, repeats
from talkinghead.timeline import Plan, Segment, SourceInfo

# Source: "One two." 0-1 | "Cut me." 2-3 | "Three four." 4-5 | "Five six." 6-7
WORDS = [
    {"text": "One", "start": 0.0, "end": 0.5}, {"text": "two.", "start": 0.5, "end": 1.0},
    {"text": "Cut", "start": 2.0, "end": 2.5}, {"text": "me.", "start": 2.5, "end": 3.0},
    {"text": "Three", "start": 4.0, "end": 4.5}, {"text": "four.", "start": 4.5, "end": 5.0},
    {"text": "Five", "start": 6.0, "end": 6.5}, {"text": "six.", "start": 6.5, "end": 7.0},
]


def _plan(segments: list[tuple[float, float]]) -> Plan:
    return Plan(sources={"main": SourceInfo("main", "/v/in.mp4", "sha", 8.0)}, main="main", cuts=[],
                segments=[Segment("main", a, b) for a, b in segments])


def test_kept_lines_follow_plan_order_and_carry_both_clocks() -> None:
    # The last sentence is moved to the front and "Cut me." is removed.
    lines = kept_lines(_plan([(6.0, 7.0), (0.0, 1.0), (4.0, 5.0)]), WORDS)
    assert [ln.text for ln in lines] == ["Five six.", "One two.", "Three four."]
    assert [(ln.out_start, ln.out_end) for ln in lines] == [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    assert [(ln.src_start, ln.src_end) for ln in lines] == [(6.0, 7.0), (0.0, 1.0), (4.0, 5.0)]


def test_a_sentence_joined_across_a_cut_is_one_line() -> None:
    words = [{"text": "I", "start": 0.0, "end": 0.2}, {"text": "uh", "start": 1.0, "end": 1.2},
             {"text": "agree.", "start": 2.0, "end": 2.5}]
    lines = kept_lines(_plan([(0.0, 0.5), (1.9, 2.6)]), words)
    assert [ln.text for ln in lines] == ["I agree."]
    assert lines[0].out_end == 0.5 + (2.5 - 1.9)


def test_repeats_score_each_passage_by_the_samples_that_named_it() -> None:
    lines = kept_lines(_plan([(0.0, 8.0)]), WORDS)  # L001..L004
    named = {"repeats": [{"from": "L003", "to": "L004", "earlier": "L001", "why": "same point"}]}
    partly = {"repeats": [{"from": "L004", "to": "L004", "earlier": "L001", "why": "again"}]}
    rows = repeats(lines, FakeJudge([named, partly, {"repeats": []}]), votes=3, cache=None)
    assert [(r["from"], r["to"], r["votes"], r["of"], r["earlier"]) for r in rows] == [("L003", "L004", 2, 3, "L001")]
    assert (rows[0]["src_start"], rows[0]["src_end"]) == (4.0, 7.0)


def test_repeats_drop_rows_that_cannot_be_a_repeat() -> None:
    lines = kept_lines(_plan([(0.0, 8.0)]), WORDS)
    bad = {"repeats": [
        {"from": "L009", "to": "L009", "earlier": "L001", "why": "no such line"},
        {"from": "L001", "to": "L001", "earlier": "L003", "why": "the 'earlier' line comes later"},
        {"from": "L003", "to": "L002", "earlier": "L001", "why": "runs backwards"},
    ]}
    assert repeats(lines, FakeJudge([bad]), votes=1, cache=None) == []


def test_cold_read_maps_marks_to_output_and_source_time() -> None:
    lines = kept_lines(_plan([(4.0, 7.0), (0.0, 1.0)]), WORDS)  # "Three four." "Five six." "One two."
    answer = {"marks": [{"from": "L002", "to": "L003", "reason": "drags", "why": "nothing new"},
                        {"from": "L404", "to": "L404", "reason": "drags", "why": "no such line"}],
              "verdict": "delivers late"}
    read = cold_read(lines, FakeJudge([answer]), "A title", votes=1, cache=None)
    assert read["verdicts"] == ["delivers late"]
    assert [(m["from"], m["to"], m["out_start"], m["out_end"], m["reason"]) for m in read["marks"]] == [
        ("L002", "L003", 2.0, 4.0, "drags")]
