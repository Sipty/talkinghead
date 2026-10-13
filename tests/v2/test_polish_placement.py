"""Gap placement from word extents, agreed vote extents, clipped words, and hand cuts across a re-key."""

import json
from pathlib import Path

from talkinghead.audio import RefinedWord
from talkinghead.judge import FakeJudge
from talkinghead.spans import cluster_spans
from talkinghead.stages import polish
from talkinghead.timeline import Cut, Plan, SourceInfo
from talkinghead.verify import clipped_words

from .test_polish import GAP_WORDS, W, _ctx


def test_a_gap_cut_keeps_clear_of_a_word_whose_asr_stamps_run_past_its_refined_bounds() -> None:
    # Restaurant noise shortens the refined bounds of "OpenLlama" to 0.5-0.7 s; the ASR stamps still say 0.5-1.6 s.
    words = [W("a.", 0.0, 0.3, "S0"), RefinedWord("OpenLlama", 0.5, 0.7, 0.5, 1.6, 1.0, speaker="S0"),
             W("b.", 2.4, 2.6, "S0")]
    cuts = polish.gap_cuts(_ctx(words), [], [])
    assert len(cuts) == 1
    cut = cuts[0]
    assert all(not (cut.start < w.asr_end and cut.end > w.asr_start) for w in words)
    assert cut.start >= 1.6 + polish.GAP_PAD_AFTER - 1e-9
    assert cut.end <= 2.4 - polish.GAP_PAD_BEFORE + 1e-9


def test_agreed_extents_remove_only_words_enough_samples_name() -> None:
    # Samples name [0,1], [1,3] and [2,4]. Narrow intersects the two that overlap at 2, finds no common core and
    # falls back to the union (0-4). Agreed removes the words two samples name: 1-3.
    responses = [[(0, 1)], [(1, 3)], [(2, 4)]]
    assert [(s.a, s.b) for s in cluster_spans(responses, agree=2, mode="agreed")] == [(1, 3)]
    assert [(s.a, s.b) for s in cluster_spans(responses, agree=2, mode="narrow")] == [(0, 4)]


def test_a_cut_that_takes_part_of_a_word_is_reported_and_a_whole_or_sliver_word_is_not() -> None:
    source = SourceInfo("main", "/v/in.mp4", "sha", 10.0)
    plan = Plan.from_cuts(source, [Cut(2.1, 2.3, "gap"), Cut(4.0, 4.2, "polish"), Cut(6.19, 6.3, "gap")])
    words = [W("OpenLlama", 2.0, 2.4), W("gone", 4.0, 4.2), W("kept", 6.0, 6.2)]
    flagged = clipped_words(plan, words)
    assert [x["text"] for x in flagged] == ["OpenLlama"]
    assert flagged[0]["cut"] == 0.5 and flagged[0]["reasons"] == ["gap"]


def test_hand_cuts_of_a_stored_memo_are_read_back_and_judged_cuts_are_not(tmp_path: Path) -> None:
    hand = {"start": 1.0, "end": 1.5, "reason": "polish", "text": "uh", "provenance": {"hand": "uh before a restart"}}
    judged = {"start": 2.0, "end": 2.2, "reason": "polish", "text": "I", "provenance": {"kind": "repeat"}}
    memo = tmp_path / "polish.json"
    memo.write_text(json.dumps({"stage": "polish", "key": "old", "version": 2,
                                "value": {"cuts": [hand, judged], "notes": {}}}))
    assert [c.text for c in polish.hand_cuts(memo)] == ["uh"]
    assert polish.hand_cuts(tmp_path / "missing.json") == []


def test_a_run_carries_the_hand_cuts_it_is_given_into_its_value() -> None:
    hand = Cut(0.35, 0.50, "polish", "two.", {"hand": "two. tighter"})
    out = polish.run(_ctx(GAP_WORDS, judge=FakeJudge([{"cuts": []}] * 3)), [], [], hand=[hand])
    assert [c["provenance"] for c in out["cuts"] if c["reason"] == "polish"] == [{"hand": "two. tighter"}]
    assert out["notes"]["hand"] == 1
