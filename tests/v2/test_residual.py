"""The output pass's scan and mapping, and one judged round on fake transcripts."""

from types import SimpleNamespace

import pytest

from talkinghead.config import Config
from talkinghead.judge import FakeJudge
from talkinghead.residual import finding_source, scan, to_source
from talkinghead.stages import polish
from talkinghead.stages.polish import fragment_window, merge_keeps
from talkinghead.timeline import Cut, Plan, Segment, SourceInfo
from talkinghead.transcribe import Word
from talkinghead.verify import _residual_disfluencies, residual_findings


def _words(*spec: tuple[str, float]) -> list[Word]:
    """Each (text, start) becomes a 0.3 s word."""
    return [Word(text, start, start + 0.3, 1.0) for text, start in spec]


def _plan(*cuts: tuple[float, float], duration: float = 10.0) -> Plan:
    return Plan.from_cuts(SourceInfo("main", "/v/in.mp4", "sha", duration), [Cut(a, b, "silence") for a, b in cuts])


# ------------------------------------------------------------- scan

def test_fillers_are_found_in_any_case_and_spelling() -> None:
    found = scan(_words(("Uh,", 0), ("I", 1), ("UMM", 2), ("hmm.", 3), ("mm", 4), ("know", 5)))
    assert [(f.kind, f.heard) for f in found] == [("filler", "Uh,"), ("filler", "UMM"), ("filler", "hmm."),
                                                  ("filler", "mm")]


def test_a_filler_at_the_end_takes_its_own_end_and_keeps_nothing_after() -> None:
    (f,) = scan(_words(("we", 0), ("uh", 1)))
    assert (f.start, f.end, f.follows) == (1.0, 1.3, False)


def test_a_repeated_word_is_found_with_its_kept_copy_as_the_end() -> None:
    (f,) = scan(_words(("it's,", 0), ("it's", 1), ("ready", 2)))
    assert (f.kind, f.first, f.last, f.start, f.end, f.follows) == ("repeat", 0, 0, 0.0, 1.0, True)
    assert f.key == "repeat it's it's"


def test_repeats_of_one_to_three_words_are_found_and_chained_copies_do_not_overlap() -> None:
    found = scan(_words(("I,", 0), ("I,", 1), ("I,", 2), ("ends", 3), ("up,", 4), ("ends", 5), ("up", 6)))
    assert [(f.kind, f.first, f.last) for f in found] == [("repeat", 0, 0), ("repeat", 1, 1), ("repeat", 3, 4)]
    assert [f.heard for f in found] == ["I,", "I,", "ends up,"]


def test_a_repeat_never_crosses_a_speaker_change() -> None:
    words = _words(("yeah,", 0), ("yeah", 1))
    words[0] = Word("yeah,", 0.0, 0.3, 1.0, "S0")
    words[1] = Word("yeah", 1.0, 1.3, 1.0, "S1")
    assert scan(words) == []


def test_a_fragment_restarted_is_a_restart_with_or_without_a_hyphen() -> None:
    hyphen = scan(_words(("adver-", 0), ("advertising", 1)))
    prefix = scan(_words(("the", 0), ("theory", 1)))
    assert [(f.kind, f.heard) for f in hyphen] == [("restart", "adver-")]
    assert [(f.kind, f.heard) for f in prefix] == [("restart", "the")]
    assert scan(_words(("I", 0), ("imagine", 1))) == []  # a one-letter stem is a word, not a fragment


def test_plain_words_give_no_findings() -> None:
    assert scan(_words(("Hello.", 0), ("Second.", 1), ("Third.", 2))) == []


# ------------------------------------------------------------- mapping

def test_an_output_span_maps_through_the_cuts_to_its_source_time() -> None:
    plan = _plan((2.0, 3.0))  # output 0-2 is source 0-2; output 2-9 is source 3-10
    assert to_source(plan, 2.5, 2.8) == (3.5, 3.8)
    assert to_source(plan, 0.5, 1.5) == (0.5, 1.5)


def test_a_span_that_crosses_a_forward_join_covers_the_cut_between_its_pieces() -> None:
    # Output 0-2 is source 0-2; the cut 2-3 is already gone, so output 2 onwards is source 3. The span [1.8, 2.3)
    # covers source 1.8 to 3.3: the kept piece, the removed gap, and the head of the next piece.
    plan = _plan((2.0, 3.0))
    assert to_source(plan, 1.8, 2.3) == (1.8, 3.3)


def test_a_span_that_crosses_a_backward_join_has_no_single_source_span() -> None:
    # A moved block: the second segment starts before the first one ends in the source.
    plan = Plan(sources={"main": SourceInfo("main", "/v/in.mp4", "sha", 10.0)}, main="main", cuts=[],
                segments=[Segment("main", 0.0, 1.2), Segment("main", 0.5, 5.0)])
    assert to_source(plan, 1.0, 1.25) is None
    assert to_source(plan, 0.2, 1.0) == (0.2, 1.0)


def test_a_span_outside_the_output_maps_to_nothing() -> None:
    assert to_source(_plan((2.0, 3.0), duration=5.0), 4.9, 6.0) is None


# ------------------------------------------------------------- the judged round

def _ctx(judge, **config) -> SimpleNamespace:
    return SimpleNamespace(config=Config(**config), judge=judge, llm_cache=None, topic=None,
                           conversation=[], preferences="")


def _verdicts(*pairs: tuple[str, str]) -> dict:
    return {"verdicts": [{"id": i, "verdict": v, "why": "test"} for i, v in pairs]}


EMPHASIS = _words(("I,", 1.0), ("I,", 1.3), ("we", 1.6), ("have", 1.9), ("very,", 3.0), ("very", 3.3), ("loud", 3.6))


def test_a_round_cuts_what_the_judge_confirms_and_keeps_emphasis_on_purpose() -> None:
    # Two findings: "I, I," (a stutter) and "very, very" (emphasis). Two of three samples cut the first only.
    judge = FakeJudge([_verdicts(("c0", "cut"), ("c1", "keep"))] * 2 + [_verdicts(("c0", "keep"), ("c1", "keep"))])
    value = polish.output_round(_ctx(judge, polish_votes=3, polish_agree=2), _plan(duration=5.0), EMPHASIS, 1)
    assert judge.calls == 3  # one batched prompt per sample, however many findings
    (cut,) = value["cuts"]
    assert (cut["reason"], cut["start"], cut["end"]) == ("polish", 1.0, 1.25)
    assert cut["provenance"]["pass"] == "output" and cut["provenance"]["round"] == 1
    assert [k["key"] for k in value["keep"]] == ["repeat very very"]
    assert [r["decision"] for r in value["findings"]] == ["cut", "keep"]


def test_a_confirmed_finding_across_a_backward_join_is_reported_and_not_cut() -> None:
    judge = FakeJudge([_verdicts(("c0", "cut"), ("c1", "keep"))] * 3)
    plan = Plan(sources={"main": SourceInfo("main", "/v/in.mp4", "sha", 10.0)}, main="main", cuts=[],
                segments=[Segment("main", 0.0, 1.2), Segment("main", 0.5, 5.0)])
    value = polish.output_round(_ctx(judge, polish_votes=3, polish_agree=2), plan, EMPHASIS, 1)
    assert value["cuts"] == []
    assert value["findings"][0]["decision"] == "skipped: crosses a join"
    assert value["notes"]["skipped"] == 1


def test_a_confirmed_finding_across_a_forward_join_is_cut_through_the_join() -> None:
    judge = FakeJudge([_verdicts(("c0", "cut"), ("c1", "keep"))] * 3)
    plan = _plan((1.2, 1.25), duration=5.0)  # a cut between the two "I," copies, as a gap cut would leave
    value = polish.output_round(_ctx(judge, polish_votes=3, polish_agree=2), plan, EMPHASIS, 1)
    (cut,) = value["cuts"]
    assert (cut["start"], cut["end"]) == pytest.approx((1.0, 1.3))


def test_a_round_with_no_findings_makes_no_judge_call() -> None:
    judge = FakeJudge([])
    value = polish.output_round(_ctx(judge), _plan(duration=5.0), _words(("Hello.", 0.0), ("Goodbye.", 1.0)), 1)
    assert judge.calls == 0 and value["cuts"] == [] and value["keep"] == []


def test_the_output_policy_changes_when_the_rules_change(monkeypatch: pytest.MonkeyPatch) -> None:
    before = polish.output_policy()
    assert before == polish.output_policy()  # the same rules give the same fingerprint
    monkeypatch.setattr(polish, "OUTPUT_RULES", polish.OUTPUT_RULES + "\n- an added rule.")
    assert polish.output_policy() != before


def test_only_a_filler_the_output_pass_or_the_source_named_settles_its_voiced_edges() -> None:
    from talkinghead.edges import _is_filler_cut

    assert _is_filler_cut(Cut(0.0, 1.0, "filler"))
    assert _is_filler_cut(Cut(0.0, 1.0, "polish", "", {"pass": "output", "kind": "filler"}))
    assert not _is_filler_cut(Cut(0.0, 1.0, "polish", "", {"kind": "filler"}))  # a source-side judged cut
    assert not _is_filler_cut(Cut(0.0, 1.0, "polish", "", {"pass": "output", "kind": "repeat"}))


# ------------------------------------------------------------- reordered plans and occurrences

REORDERED = Plan(sources={"main": SourceInfo("main", "/v/in.mp4", "sha", 30.0)}, main="main", cuts=[],
                 segments=[Segment("main", 10.0, 20.0), Segment("main", 0.0, 10.0), Segment("main", 20.0, 30.0)])


def test_a_forward_join_that_skips_an_opener_kept_elsewhere_has_no_single_source_span() -> None:
    # The opener (source 10-20) plays first, at output 0-10. Output 19.8-20.3 is the end of source 0-10 and the
    # start of source 20-30: one span over the join would also cut the opener, which the plan keeps.
    assert to_source(REORDERED, 19.8, 20.3) is None
    assert to_source(REORDERED, 10.5, 11.0) == (0.5, 1.0)  # a span inside one segment still maps


def test_a_confirmed_repeat_across_that_join_is_skipped_and_the_opener_is_not_cut() -> None:
    judge = FakeJudge([_verdicts(("c0", "cut"))] * 3)
    words = _words(("we,", 19.8), ("we", 20.1), ("know", 20.4))  # a repeat whose kept copy starts at 20.1
    value = polish.output_round(_ctx(judge, polish_votes=3, polish_agree=2), REORDERED, words, 1)
    assert judge.calls == 3 and value["cuts"] == []
    assert value["findings"][0]["decision"] == "skipped: crosses a join"


def _finding(start: float, source: float | None) -> dict:
    return {"key": "repeat very very", "heard": "very, very", "kind": "repeat", "start": start, "source": source}


def test_a_judged_keep_suppresses_only_its_own_occurrence_not_every_one_with_the_same_text() -> None:
    # Two identical "very, very" repeats: the judge kept the one at source 5.0 and never saw the one at 12.0.
    facts = {"disfluency_findings": [_finding(3.0, 5.0), _finding(8.0, 12.0)],
             "disfluency_keep": [{"key": "repeat very very", "source": 5.0}]}
    assert [x["source"] for x in residual_findings(facts)] == [12.0]
    check = _residual_disfluencies(None, facts)
    assert check.severity == "warn" and not check.ok and "very, very" in check.detail


def test_a_finding_that_cannot_be_placed_in_the_source_is_never_suppressed_by_a_keep() -> None:
    facts = {"disfluency_findings": [_finding(3.0, None)],
             "disfluency_keep": [{"key": "repeat very very", "source": None}]}
    assert len(residual_findings(facts)) == 1


# ------------------------------------------------------------- abandoned fragments

# Scribe marks cut-off speech with a dash. Shape 1: "there's also--" abandoned, then a new thought.
ABANDONED = _words(("and", 0.0), ("that", 0.3), ("means", 0.6), ("there's", 0.9), ("also--", 1.2),
                   ("it's", 1.5), ("gonna", 1.8), ("blow", 2.1), ("up", 2.4))
# Shape 2: a false start, "to--", before "for a harness".
FALSE_START = _words(("things", 0.0), ("are", 0.3), ("too", 0.6), ("volatile", 0.9), ("to--", 1.2),
                     ("for", 1.5), ("a", 1.8), ("harness", 2.1))
GAP = _plan((0.5, 0.6), duration=5.0)  # a 0.1 s cut before the fragments: output 0.5 onwards is source 0.6 onwards


def _pick(from_index: int | None, verdict: str = "cut") -> dict:
    item = {"id": "c0", "verdict": verdict, "why": "test"}
    if from_index is not None:
        item["from_index"] = from_index
    return {"verdicts": [item]}


def test_every_dashed_token_is_a_fragment_and_a_restart_stays_a_restart() -> None:
    assert [(f.kind, f.heard) for f in scan(ABANDONED)] == [("fragment", "also--")]
    assert [(f.kind, f.heard) for f in scan(FALSE_START)] == [("fragment", "to--")]
    assert [(f.kind, f.heard) for f in scan(_words(("adver-", 0), ("advertising", 1)))] == [("restart", "adver-")]
    assert scan(_words(("Hello.", 0), ("Goodbye.", 1))) == []


def test_a_judged_fragment_cut_maps_its_chosen_span_to_the_source() -> None:
    # The judge picks index 3 of the window (#0 and ... #3 there's #4 also--): cut from "there's" to the kept "it's".
    judge = FakeJudge([_pick(3)] * 3)
    value = polish.output_round(_ctx(judge, polish_votes=3, polish_agree=2), GAP, ABANDONED, 1)
    (cut,) = value["cuts"]
    assert (cut["start"], cut["end"]) == pytest.approx((1.0, 1.55))  # output 0.9-1.45 is source 1.0-1.55
    assert cut["provenance"]["kind"] == "fragment" and cut["text"] == "there's also--"
    assert value["findings"][0]["decision"] == "cut"


def test_a_judged_false_start_that_picks_only_the_dashed_word_cuts_only_that_word() -> None:
    judge = FakeJudge([_pick(4)] * 3)  # index 4 is "to--" itself; output 1.2-1.45 is source 1.3-1.55
    value = polish.output_round(_ctx(judge, polish_votes=3, polish_agree=2), GAP, FALSE_START, 1)
    (cut,) = value["cuts"]
    assert (cut["start"], cut["end"]) == pytest.approx((1.3, 1.55))
    assert cut["text"] == "to--"


def test_a_judged_keep_leaves_the_fragment_in_and_the_final_check_accepts_that_occurrence() -> None:
    judge = FakeJudge([_pick(None, verdict="keep")] * 3)
    value = polish.output_round(_ctx(judge, polish_votes=3, polish_agree=2), GAP, ABANDONED, 1)
    assert value["cuts"] == []
    assert [(k["key"], k["source"]) for k in value["keep"]] == [("fragment also", pytest.approx(1.3))]
    facts = {"disfluency_findings": [{**f.to_dict(), "source": finding_source(GAP, f)} for f in scan(ABANDONED)],
             "disfluency_keep": value["keep"]}
    assert residual_findings(facts) == []


def test_a_fragment_window_stops_at_the_sentence_start() -> None:
    words = _words(("Done.", 0.0), ("so", 0.3), ("ran", 0.6), ("off", 0.9), ("on--", 1.2), ("next", 1.5))
    (frag,) = scan(words)
    assert fragment_window(words, frag) == [1, 2, 3, 4]  # "Done." closes the sentence before "so"
    judge = FakeJudge([_pick(0)] * 3)  # index 0 is "so": the abandoned words begin after "Done."
    value = polish.output_round(_ctx(judge, polish_votes=3, polish_agree=2), GAP, words, 1)
    (cut,) = value["cuts"]
    assert (cut["start"], cut["end"]) == pytest.approx((0.3, 1.55))


def test_the_judged_keeps_list_shows_each_occurrence_once_with_the_latest_verdict() -> None:
    keep_round = polish.output_round(_ctx(FakeJudge([_pick(None, "keep")] * 3), polish_votes=3, polish_agree=2),
                                     GAP, ABANDONED, 1)
    again_keep = polish.output_round(_ctx(FakeJudge([_pick(None, "keep")] * 3), polish_votes=3, polish_agree=2),
                                     GAP, ABANDONED, 2)
    once = merge_keeps(keep_round["keep"], again_keep["findings"], again_keep["keep"])
    assert len(once) == 1
    later_cut = polish.output_round(_ctx(FakeJudge([_pick(3)] * 3), polish_votes=3, polish_agree=2),
                                    GAP, ABANDONED, 3)
    assert merge_keeps(once, later_cut["findings"], later_cut["keep"]) == []  # cut later: no longer a keep

