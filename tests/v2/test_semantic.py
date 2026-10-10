"""Tests for the shared LLM helpers and the flow guard (semantic.py), and CutPlan.dropped."""

from __future__ import annotations

import sys

import pytest

from talkinghead.semantic import (
    FLOW_MIN_FRAGMENT,
    FLOW_MIN_KEPT,
    Sentence,
    chunk_sentences,
    flow_guard,
    parse_llm_json,
    run_llm,
)
from talkinghead.transcribe import Word
from talkinghead.types import SEMANTIC_REASONS, CutPlan, CutSpan

# --- run_llm --------------------------------------------------------------


def test_run_llm_echo() -> None:
    echo = f'{sys.executable} -c "import sys; sys.stdout.write(sys.stdin.read())"'
    assert run_llm(echo, "ping") == "ping"


def test_run_llm_failure_raises_with_stderr() -> None:
    cmd = f"{sys.executable} -c \"import sys; sys.stderr.write('boom'); sys.exit(3)\""
    with pytest.raises(RuntimeError, match="llm command failed"):
        run_llm(cmd, "in")


def test_run_llm_empty_output_raises() -> None:
    with pytest.raises(RuntimeError):
        run_llm("true", "in")


# --- parse_llm_json -------------------------------------------------------


def test_parse_llm_json_bare() -> None:
    assert parse_llm_json('{"cuts": []}') == {"cuts": []}


def test_parse_llm_json_fenced() -> None:
    raw = '```json\n{"cuts": [{"ids": ["S0000"], "reason": "redundant"}]}\n```'
    assert parse_llm_json(raw)["cuts"][0]["reason"] == "redundant"


def test_parse_llm_json_garbage_raises() -> None:
    with pytest.raises(ValueError):
        parse_llm_json("not json at all")


# --- flow_guard ------------------------------------------------------------


def test_flow_guard_thresholds() -> None:
    assert (FLOW_MIN_KEPT, FLOW_MIN_FRAGMENT) == (2.0, 1.5)


def test_flow_guard_fuses_across_reasons() -> None:
    # The 1:57 failure: a filler cut 0.4s after a silence cut left a jump cut.
    cuts = [CutSpan(117.0, 118.0, "silence"), CutSpan(118.4, 119.0, "filler", "uh")]
    out = flow_guard(cuts)
    assert len(out) == 1
    assert (out[0].start, out[0].end) == (117.0, 119.0)
    # Union span, first reason wins, absorbed text appended.
    assert out[0].reason == "silence"
    assert out[0].text == "uh"


def test_flow_guard_keeps_cuts_with_enough_kept_speech() -> None:
    cuts = [CutSpan(10.0, 11.0, "silence"), CutSpan(14.0, 15.0, "filler", "um")]
    assert flow_guard(cuts) == cuts


def test_flow_guard_fuses_a_run_of_three() -> None:
    # 2:26/2:30 failure: three cuts a second apart become one span.
    cuts = [
        CutSpan(146.0, 147.0, "ramble", "a"),
        CutSpan(148.0, 149.0, "silence"),
        CutSpan(149.5, 150.0, "filler", "b"),
    ]
    out = flow_guard(cuts)
    assert len(out) == 1
    assert (out[0].start, out[0].end) == (146.0, 150.0)
    assert out[0].reason == "ramble"
    assert out[0].text == "a b"


def test_flow_guard_absorbs_short_fragment_pass() -> None:
    # Second pass in isolation: min_kept off, so only the 1.2s island goes.
    cuts = [CutSpan(0.0, 1.0, "silence"), CutSpan(2.2, 3.0, "filler", "uh")]
    out = flow_guard(cuts, min_kept=0.0, min_fragment=1.5)
    assert len(out) == 1
    assert (out[0].start, out[0].end) == (0.0, 3.0)


def test_flow_guard_second_pass_leaves_long_fragment() -> None:
    cuts = [CutSpan(0.0, 1.0, "silence"), CutSpan(3.0, 4.0, "filler", "uh")]
    assert flow_guard(cuts, min_kept=0.0, min_fragment=1.5) == cuts


def test_flow_guard_fuses_overlapping_and_contained_spans() -> None:
    cuts = [CutSpan(5.0, 9.0, "ramble", "long"), CutSpan(6.0, 7.0, "filler", "uh")]
    out = flow_guard(cuts)
    assert len(out) == 1
    assert (out[0].start, out[0].end) == (5.0, 9.0)


def test_flow_guard_empty_and_single() -> None:
    assert flow_guard([]) == []
    one = [CutSpan(1.0, 2.0, "silence")]
    assert flow_guard(one) == one


def test_flow_guard_is_idempotent() -> None:
    cuts = [
        CutSpan(1.0, 2.0, "silence"),
        CutSpan(2.5, 3.0, "filler", "uh"),
        CutSpan(20.0, 21.0, "ramble", "x"),
    ]
    once = flow_guard(cuts)
    assert flow_guard(once) == once


def test_flow_guard_output_sorted_from_unsorted_input() -> None:
    cuts = [CutSpan(20.0, 21.0, "ramble", "x"), CutSpan(1.0, 2.0, "silence")]
    assert [c.start for c in flow_guard(cuts)] == [1.0, 20.0]


def test_flow_guard_does_not_mutate_input() -> None:
    cuts = [CutSpan(1.0, 2.0, "silence"), CutSpan(2.2, 3.0, "filler", "uh")]
    flow_guard(cuts)
    assert (cuts[0].start, cuts[0].end) == (1.0, 2.0)
    assert cuts[0].text == ""


# --- chunk_sentences -------------------------------------------------------


def test_chunk_sentences_overlap() -> None:
    sents = [Sentence(f"S{i:04d}", float(i), float(i) + 1.0, "x" * 50) for i in range(300)]
    chunks = chunk_sentences(sents, max_chars=2000)
    assert len(chunks) > 1
    for a, b in zip(chunks, chunks[1:]):
        sids_a = {s.sid for s in a}
        assert b[0].sid in sids_a, "chunk must start inside the previous chunk"
        assert b[0].sid < a[-1].sid


def test_chunk_sentences_small_input_single_chunk() -> None:
    sents = [Sentence("S0000", 0.0, 1.0, "hi.")]
    assert chunk_sentences(sents) == [sents]


# --- CutPlan.dropped -------------------------------------------------------


def test_cut_plan_dropped_round_trip() -> None:
    plan = CutPlan(video="v.mp4", duration=10.0, dropped=[{"reason": "r", "note": "", "detail": "x"}])
    restored = CutPlan.from_json(plan.to_json())
    assert restored.dropped == plan.dropped
    assert '"dropped"' in plan.to_json()


def test_cut_plan_dropped_absent_key_stays_absent() -> None:
    plan = CutPlan(video="v.mp4", duration=10.0)
    assert '"dropped"' not in plan.to_json()
    assert CutPlan.from_json(plan.to_json()).dropped == []


def test_semantic_reasons_constant() -> None:
    assert SEMANTIC_REASONS == {"redundant", "false-start", "self-talk", "digression"}


def test_gap_has_speech_uses_strict_bounds() -> None:
    from talkinghead.semantic import _gap_has_speech
    cuts = [CutSpan(0.0, 2.5, "filler", ""), CutSpan(4.0, 5.0, "filler", "")]
    # Word midpoint exactly on the gap edge must NOT block fusion (old code
    # used inclusive bounds); a word strictly inside the gap must.
    on_edge = [Word("hello", 2.0, 3.0, 1.0)]  # midpoint 2.5 == gap_start
    assert _gap_has_speech(cuts, 0, on_edge) is False
    inside = [Word("hello", 3.0, 4.0, 1.0)]  # midpoint 3.5 strictly inside
    assert _gap_has_speech(cuts, 0, inside) is True
    # Zero-width gap between exactly adjacent cuts is never blocked.
    adjacent = [CutSpan(0.0, 2.5, "filler", ""), CutSpan(2.5, 5.0, "filler", "")]
    assert _gap_has_speech(adjacent, 0, inside) is False
