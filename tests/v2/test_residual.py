"""The output pass's scan and mapping, and one judged round on fake transcripts."""

from types import SimpleNamespace

import pytest

from talkinghead.config import Config
from talkinghead.judge import FakeJudge
from talkinghead.residual import scan, to_source
from talkinghead.stages import polish
from talkinghead.timeline import Cut, Plan, Segment, SourceInfo
from talkinghead.transcribe import Word


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


def test_the_output_policy_is_a_fingerprint_that_changes_with_the_rules() -> None:
    assert polish.output_policy() == polish.output_policy()
    assert "NEVER remove a laugh" in polish.OUTPUT_RULES and "- filler:" in polish.OUTPUT_RULES


def test_only_a_filler_the_output_pass_or_the_source_named_settles_its_voiced_edges() -> None:
    from talkinghead.edges import _is_filler_cut

    assert _is_filler_cut(Cut(0.0, 1.0, "filler"))
    assert _is_filler_cut(Cut(0.0, 1.0, "polish", "", {"pass": "output", "kind": "filler"}))
    assert not _is_filler_cut(Cut(0.0, 1.0, "polish", "", {"kind": "filler"}))  # a source-side judged cut
    assert not _is_filler_cut(Cut(0.0, 1.0, "polish", "", {"pass": "output", "kind": "repeat"}))
