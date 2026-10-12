"""Restructure: windows, block ownership, votes, cuts, reorder, reorder-safe checks, and the disabled path.

Judges are fakes. The end-to-end cases run the full pipeline on a short synthetic
clip, so a reordered plan has to render and verify, and a hand-edited memo has to
be honoured on replay."""

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from talkinghead.audio import Envelope, RefinedWord
from talkinghead.board.render import render_board
from talkinghead.board.spec import BoardSpec
from talkinghead.candidates import sentence_table
from talkinghead.config import Config
from talkinghead.context import Context
from talkinghead.edges import EdgeContext
from talkinghead.joins import source_edge_facts
from talkinghead.judge import FakeJudge, prompt_sha
from talkinghead.pipeline import run_pipeline
from talkinghead.prompts import cold_open_prompt, content_prompt, retake_prompt, with_preferences
from talkinghead.prompts import sentence_table as prompt_table
from talkinghead.render import _handles, join_audio
from talkinghead.semantic import Sentence
from talkinghead.stages import restructure as rs
from talkinghead.store import Cache
from talkinghead.timeline import Cut, Plan, Segment, SourceInfo, in_source_order
from talkinghead.transcribe import Transcript, Word
from talkinghead.verify import residual_dead_air


def sents(n: int, length: float = 10.0, gap: float = 0.0) -> list[Sentence]:
    step = length + gap
    return [Sentence(f"S{i:04d}", i * step, i * step + length, f"sentence {i}.") for i in range(n)]


def judged(first: int, **kw) -> dict:
    base = {"first": first, "title": f"block {first}", "summary": "", "payoff": "a take", "payoff_sid": None,
            "interest": 5, "relevance": 3, "depends_on_earlier": False, "depends_note": "", "lead_in_sids": []}
    return {**base, **kw}


def block(bid: str, first: int = 0, last: int = 0, start: float = 0.0, end: float = 1.0, **kw) -> rs.Block:
    return rs.Block(id=bid, first=first, last=last, start=start, end=end, **kw)


def item(bid: str, decision: str = "keep", front: int | None = None, reason: str = "") -> dict:
    return {"id": bid, "decision": decision, "reason": reason, "front_rank": front}


def edge_ctx(duration: float, db: np.ndarray | None = None, fps: float | None = 30.0) -> EdgeContext:
    fs = 0.005
    n = int(round(duration / fs))
    level = np.full(n, -50.0) if db is None else db
    env = Envelope(level, np.zeros(n), fs, 16000)
    return EdgeContext(duration=duration, env=env, floor=-50.0, words=[], fps=fps)


# ------------------------------------------------------------- windows

def test_windows_partition_the_sentences_with_context_either_side() -> None:
    s = sents(40)  # 0..400 s, 10 s sentences
    wins = rs.windows(s, window_sec=200, context_sec=30)  # core 140 s
    cores = [w.core for w in wins]
    assert cores[0][0] == 0 and cores[-1][1] == len(s)
    assert all(a[1] == b[0] for a, b in zip(cores, cores[1:]))
    assert len(wins) > 1
    for w in wins:
        assert s[w.core[1] - 1].end - s[w.core[0]].start <= 140 + 10
        assert w.context[0] <= w.core[0] and w.context[1] >= w.core[1]
        if w.core[0] > 0:
            assert s[w.context[0]].end >= s[w.core[0]].start - 30
    assert wins[0].context[0] == 0 and wins[-1].context[1] == len(s)


def test_window_answers_start_blocks_only_inside_their_core() -> None:
    s = sents(6)
    index = {x.sid: i for i, x in enumerate(s)}
    w = rs.Window(core=(2, 4), context=(0, 6))
    resp = {"blocks": [judged(0) | {"start": "S0000"}, judged(2) | {"start": "S0002"},
                       judged(2) | {"start": "S0002"}, judged(3) | {"start": "S0003"},
                       judged(9) | {"start": "S0009"}]}
    resp["blocks"] = [{k: v for k, v in r.items() if k != "first"} for r in resp["blocks"]]
    got = rs._window_blocks(resp, s, index, w)
    assert [g["first"] for g in got] == [2, 3]


def test_blocks_partition_the_kept_sentences_and_lead_in_stays_inside_its_block() -> None:
    s = sents(6)
    index = {x.sid: i for i, x in enumerate(s)}
    lead = judged(2, lead_in_sids=["S0002", "S0005"])  # S0005 belongs to the next block: not a lead-in here
    blocks = rs.merge_windows(s, [lead, judged(4)], index)
    assert [(b.first, b.last, b.judged) for b in blocks] == [(0, 1, False), (2, 3, True), (4, 5, True)]
    assert blocks[0].start == s[0].start and blocks[-1].end == s[-1].end
    assert blocks[1].end == s[3].end
    assert [x[0] for x in blocks[1].lead_in] == ["S0002"]
    assert [b.id for b in blocks] == ["B01", "B02", "B03"]


# ------------------------------------------------------------- votes

def test_a_block_is_dropped_only_by_a_strict_majority() -> None:
    blocks = [block("B01", 0, 0, 0.0, 1.0, judged=True), block("B02", 1, 1, 1.0, 2.0, judged=True)]
    responses = [
        {"blocks": [item("B01", "drop", reason="warm-up"), item("B02", "drop", reason="tangent")]},
        {"blocks": [item("B01", "drop", reason="small talk"), item("B02", "keep", reason="joke lands")]},
        {"blocks": [item("B01", "keep", reason="has a point"), item("B02", "keep", reason="joke lands")]},
    ]
    rs._decide(blocks, responses, front=0)
    assert blocks[0].decision == "drop"
    assert blocks[1].decision == "keep" and blocks[1].reason == "joke lands"


def test_a_sample_that_does_not_answer_a_block_counts_as_keep() -> None:
    blocks = [block("B01", judged=True)]
    rs._decide(blocks, [{"blocks": [item("B01", "drop")]}, {"blocks": []}], front=0)
    assert blocks[0].decision == "keep"


def test_openers_are_borda_ranked_by_majority_and_a_dependency_vetoes() -> None:
    blocks = [block("A", 0, 0, 0.0, 1.0, judged=True),
              block("B", 1, 1, 1.0, 2.0, judged=True, depends_on_earlier=True),
              block("C", 2, 2, 2.0, 3.0, judged=True),
              block("E", 3, 3, 3.0, 4.0, judged=True)]
    responses = [
        {"blocks": [item("B", front=1), item("A", front=2), item("E", front=3)]},
        {"blocks": [item("A", front=1), item("C", front=2), item("E", front=2)]},
        {"blocks": [item("E", front=1), item("A", front=1)]},
    ]
    rs._decide(blocks, responses, front=2)
    ranked = {b.id: b.front_rank for b in blocks}
    # Points are front + 1 - rank for ranks within the cap: A = 1 + 2 + 2 = 5 and E = 0 + 1 + 2 = 3, each ranked
    # by all three samples. B is vetoed by its dependency. C is ranked by one sample only: no majority.
    assert ranked["B"] is None and ranked["C"] is None
    assert {ranked["A"], ranked["E"]} == {1, 2}


def test_openers_are_capped_and_the_highest_points_go_first() -> None:
    blocks = [block(x, i, i, float(i), i + 1.0, judged=True) for i, x in enumerate("ABCD")]
    responses = [
        {"blocks": [item("C", front=1), item("D", front=2)]},
        {"blocks": [item("C", front=1), item("D", front=2)]},
        {"blocks": [item("D", front=1), item("C", front=2)]},
    ]
    rs._decide(blocks, responses, front=1)
    fronts = {b.id: b.front_rank for b in blocks if b.front_rank is not None}
    assert fronts == {"C": 1}


# ------------------------------------------------------------- cuts

def test_dropped_blocks_are_tangent_cuts_and_openers_lose_their_lead_in() -> None:
    blocks = [
        block("B01", start=0.0, end=1.0, decision="drop", title="warm-up"),
        block("B02", start=1.0, end=3.0, decision="keep", front_rank=1, lead_in=[["S0004", 1.0, 1.4]]),
        block("B03", start=3.0, end=5.0, decision="keep"),
    ]
    cuts = rs.block_cuts(blocks)
    assert [(c.reason, c.start, c.end) for c in cuts] == [("tangent", 0.0, 1.0), ("lead-in", 1.0, 1.4)]


# ------------------------------------------------------------- reorder

def _plan(segments: list[Segment], duration: float = 10.0, cuts: list[Cut] | None = None) -> Plan:
    src = SourceInfo("main", "/v/in.mp4", "sha", duration)
    return Plan(sources={"main": src}, main="main", cuts=cuts or [], segments=segments)


def test_reorder_moves_the_opener_first_and_keeps_the_rest_in_source_order() -> None:
    plan = _plan([Segment("main", 0.0, 4.0), Segment("main", 4.5, 10.0)], cuts=[Cut(4.0, 4.5, "filler")])
    blocks = [block("B01", start=0.0, end=3.0, decision="keep"),
              block("B02", start=3.0, end=6.0, decision="keep", front_rank=1),
              block("B03", start=6.0, end=10.0, decision="keep")]
    out = rs.reorder(plan, blocks, edge_ctx(10.0))
    assert [(s.start, s.end) for s in out.segments] == [(3.0, 4.0), (4.5, 6.0), (0.0, 3.0), (6.0, 10.0)]
    assert out.output_duration == pytest.approx(9.5)
    assert not in_source_order(out.segments)
    layout = {b["id"]: b for b in out.provenance["blocks"]}
    assert layout["B02"]["output"] == [0.0, 2.5]
    assert layout["B01"]["output"] == [2.5, 5.5]
    assert layout["B03"]["output"] == [5.5, 9.5]


def test_reorder_without_an_opener_leaves_the_segments_as_they_were() -> None:
    segs = [Segment("main", 0.0, 4.0), Segment("main", 4.5, 10.0)]
    plan = _plan(segs, cuts=[Cut(4.0, 4.5, "filler")])
    blocks = [block("B01", start=0.0, end=3.0), block("B02", start=3.0, end=10.0)]
    out = rs.reorder(plan, blocks, edge_ctx(10.0))
    assert out.segments == segs
    assert in_source_order(out.segments)


def test_the_split_lands_at_the_quietest_point_between_two_sentences() -> None:
    db = np.full(int(10.0 / 0.005), -50.0)
    db[int(3.0 / 0.005):int(3.1 / 0.005)] = -90.0  # a dip inside the gap 2.8..3.2
    plan = _plan([Segment("main", 0.0, 10.0)], cuts=[])
    blocks = [block("B01", start=0.0, end=2.8, decision="keep"),
              block("B02", start=3.2, end=10.0, decision="keep", front_rank=1)]
    out = rs.reorder(plan, blocks, edge_ctx(10.0, db=db, fps=None))
    # The dip spans 3.00..3.10 s; the quietest frame nearest the gap's middle inside it wins.
    assert 3.0 <= out.segments[0].start <= 3.1
    assert out.segments[0].end == pytest.approx(10.0)
    assert out.segments[1] == Segment("main", 0.0, out.segments[0].start)


def test_lead_in_cut_inside_an_opener_keeps_both_pieces_in_order() -> None:
    plan = _plan([Segment("main", 0.0, 4.0), Segment("main", 4.5, 10.0)], cuts=[Cut(4.0, 4.5, "lead-in")])
    blocks = [block("B01", start=0.0, end=3.0), block("B02", start=3.0, end=10.0, decision="keep", front_rank=1)]
    out = rs.reorder(plan, blocks, edge_ctx(10.0))
    assert [(s.start, s.end) for s in out.segments] == [(3.0, 4.0), (4.5, 10.0), (0.0, 3.0)]
    assert out.provenance["blocks"][1]["output"] == [0.0, 6.5]


# ------------------------------------------------------------- reorder-safe render, verify, joins, board

def test_a_backward_join_gets_a_real_crossfade() -> None:
    rate = 48000
    t = np.arange(3 * rate) / rate
    src = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)[:, None]
    # The second keep starts at a phase the first does not end on (1.77 s is not a whole number of 440 Hz cycles).
    segs = [Segment("main", 1.0, 2.0), Segment("main", 0.23, 0.73)]  # the second jumps back in the source
    lengths = [round(2.0 * rate) - round(1.0 * rate), round(0.73 * rate) - round(0.23 * rate)]
    starts = [round(s.start * rate) for s in segs]
    half = _handles(segs, lengths, starts, {"main": len(src)}, round(0.02 * rate))
    assert half[0] > 0

    def worst_step(crossfade: float) -> float:
        out = join_audio({"main": src}, segs, crossfade)[:, 0]
        j = round(1.0 * rate)
        return float(np.max(np.abs(np.diff(out[j - 480:j + 480]))))

    hard, faded = worst_step(0.0), worst_step(0.02)
    assert hard > 0.1
    assert faded < 0.05
    assert faded < hard / 3


def test_dead_air_is_joined_across_a_forward_seam_but_not_a_backward_one() -> None:
    # Forward: silence at the end of the first keep and at the start of the second meet in the output.
    forward = _plan([Segment("main", 0.0, 2.0), Segment("main", 2.5, 3.0)], duration=3.0)
    runs = residual_dead_air(forward, [(1.6, 2.0), (2.5, 2.9)], min_gap=0.3)
    assert [(round(a, 2), round(b, 2)) for a, b in runs] == [(1.6, 2.4)]
    # Backward: the same two silences meet at the seam, but the source jumped back, so they stay two runs.
    backward = _plan([Segment("main", 0.0, 2.0), Segment("main", 0.2, 1.0)], duration=3.0)
    runs = residual_dead_air(backward, [(1.6, 2.0), (0.2, 0.6)], min_gap=0.3)
    assert [(round(a, 2), round(b, 2)) for a, b in runs] == [(0.2, 0.6), (1.6, 2.0), (2.0, 2.4)]


def test_source_edge_facts_do_not_match_a_cut_to_a_backward_join() -> None:
    env = Envelope(np.full(1000, -50.0), np.zeros(1000), 0.005, 16000)
    backward = _plan([Segment("main", 0.0, 2.0), Segment("main", 0.2, 1.0)], duration=5.0,
                     cuts=[Cut(2.0, 2.5, "content")])
    facts = source_edge_facts(backward, env, -50.0)
    assert facts[0]["reason"] == "reorder"
    forward = _plan([Segment("main", 0.0, 2.0), Segment("main", 2.5, 3.0)], duration=5.0,
                    cuts=[Cut(2.0, 2.5, "content")])
    assert source_edge_facts(forward, env, -50.0)[0]["reason"] == "content"


def test_board_render_refuses_a_reordered_cut(tmp_path: Path) -> None:
    (tmp_path / "final.mp4").write_bytes(b"x")
    tablet = tmp_path / "tablet.mp4"
    tablet.write_bytes(b"x")
    (tmp_path / "plan.json").write_text(json.dumps({"segments": [
        {"source": "main", "start": 2.0, "end": 3.0}, {"source": "main", "start": 0.0, "end": 1.0}]}))
    spec = BoardSpec(version=1, tablet=str(tablet), offset=0.0, canvas=(0, 0, 10, 10), masks=(),
                     panel=(0.0, 0.0, 0.5, 0.5), transition=0.0)
    with pytest.raises(ValueError, match="source order"):
        render_board(tmp_path, spec)


def test_in_source_order_accepts_forward_and_contiguous_seams() -> None:
    assert in_source_order([Segment("main", 0, 1), Segment("main", 1, 2), Segment("main", 3, 4)])
    assert not in_source_order([Segment("main", 3, 4), Segment("main", 0, 1)])


# ------------------------------------------------------------- config and the disabled path

def test_disabled_restructure_leaves_no_trace_in_the_config_dict() -> None:
    assert not any(k.startswith("restructure") for k in Config().to_dict())
    on = Config(restructure=True).to_dict()
    assert on["restructure"] is True and on["restructure_front"] == 3


def test_restructure_config_is_validated() -> None:
    with pytest.raises(ValueError, match="restructure_window_sec"):
        Config(restructure=True, restructure_window_sec=200, restructure_context_sec=120).validate()
    with pytest.raises(ValueError):
        Config(restructure_votes=0).validate()


# Digests of a speakerless cook with restructure off, captured at the base commit before any restructure
# code existed. A change here means existing prompts, memo keys or plans would stop replaying.
BASE_DIGESTS = {
    "cold_open_prompt": "ba262e3b38eb2895cb744166",
    "config_full": "7f75077128cf4bbecd3b69626c641b322acf8f3330b8286425ad7f34562169ab",
    "content_prompt": "58e067ef91e916e3ca06680b",
    "memo_content": "85861a8145656ff10240b8bf",
    "memo_pauses": "5fd662422a0aa7b81242f84f",
    "memo_retakes": "666065350fa02f44600575af",
    "memo_structure": "e948824683c9030cb80d6d82",
    "plan_json": "c8352e6d19ec5c72b599402dbaf7768d33b9e07679f30498edde3ce9dac954b1",
    "retake_prompt": "d62a725f56ad4dbf3e5265a6",
    "sentences": "90070d97a16d3a75cec664be2a6a4b004ebd95574b2b6154272f062a87b8ed19",
}


def test_a_speakerless_cook_with_restructure_off_keeps_its_prompts_keys_and_plan() -> None:
    import hashlib

    from talkinghead.pipeline import STAGE_CONFIG_KEYS
    from talkinghead.store import stable_key

    words = [Word("Hold", 0.00, 0.30, 1.0), Word("on.", 0.30, 0.60, 1.0),
             Word("Are", 2.00, 2.20, 1.0), Word("you", 2.20, 2.40, 1.0), Word("tired?", 2.40, 2.90, 1.0),
             Word("Um", 3.60, 3.90, 1.0), Word("Second", 6.00, 6.40, 1.0), Word("here.", 6.70, 7.00, 1.0),
             Word("Goodbye.", 8.00, 8.50, 1.0)]
    cfg = Config()
    table = sentence_table(words, split_pause=cfg.sentence_split_pause, burst_last=None)

    def h(x) -> str:
        return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()

    got = {
        "sentences": h([(s.sid, s.start, s.end, s.text) for s in table]),
        "cold_open_prompt": prompt_sha(with_preferences(cold_open_prompt(table, table[:2], "topic"), ""), "shell:x"),
        "content_prompt": prompt_sha(with_preferences(content_prompt(table, "topic"), ""), "shell:x"),
        "retake_prompt": prompt_sha(with_preferences(retake_prompt(prompt_table(table), "topic"), ""), "shell:x"),
        "config_full": h(cfg.to_dict()),
    }
    for stage in ("structure", "retakes", "content", "pauses"):
        keys = STAGE_CONFIG_KEYS[stage]
        got[f"memo_{stage}"] = stable_key({"audio": "a", "config": {k: getattr(cfg, k) for k in keys}, "judge": "j"}, 2)
    plan = Plan.from_cuts(SourceInfo("main", "/v/in.mp4", "sha", 9.0),
                          [Cut(0.0, 2.0, "cold-open"), Cut(4.0, 4.5, "filler"), Cut(8.5, 9.0, "outro")])
    got["plan_json"] = h(plan.to_json())
    assert {k: got[k] for k in BASE_DIGESTS} == BASE_DIGESTS


# ------------------------------------------------------------- stage run with fakes

class KeyedJudge:
    """Stateless fake: answers each restructure prompt from its own text, so it is safe in threads."""

    name = "keyed"

    def __init__(self, decisions: dict[str, tuple[str, int | None]]) -> None:
        self.decisions = decisions

    def decide(self, prompt: str, schema: dict) -> dict:
        if "TASK: RESTRUCTURE WINDOW" in prompt:
            core = re.search(r"--- CORE \(start blocks here\) ---\n(S\d+) \|", prompt)
            sid = core.group(1)
            return {"blocks": [{"start": sid, "title": f"block at {sid}", "summary": "s", "payoff": "a take",
                                "payoff_sid": sid, "interest": 6, "relevance": 4,
                                "depends_on_earlier": False, "depends_note": "", "lead_in": []}]}
        if "TASK: RESTRUCTURE KEEP" in prompt:
            out = []
            for bid in re.findall(r"^(B\d+) \|", prompt, re.M):
                decision, front = self.decisions.get(bid, ("keep", None))
                out.append({"id": bid, "decision": decision, "reason": f"{decision} {bid}", "front_rank": front})
            return {"blocks": out}
        raise AssertionError("unexpected prompt")


SIX = [Sentence("S0000", 0.0, 0.6, "Hold on."), Sentence("S0001", 2.0, 2.9, "Are you tired?"),
       Sentence("S0002", 3.6, 3.9, "Um"), Sentence("S0003", 4.5, 5.6, "Are you tired yet?"),
       Sentence("S0004", 6.0, 7.0, "Second point here."), Sentence("S0005", 8.0, 8.5, "Goodbye.")]
SIX_WORDS = [Word("Hold", 0.00, 0.30, 1.0), Word("on.", 0.30, 0.60, 1.0),
             Word("Are", 2.00, 2.20, 1.0), Word("you", 2.20, 2.40, 1.0), Word("tired?", 2.40, 2.90, 1.0),
             Word("Um", 3.60, 3.90, 1.0),
             Word("Are", 4.50, 4.70, 1.0), Word("you", 4.70, 4.90, 1.0), Word("tired", 4.90, 5.20, 1.0),
             Word("yet?", 5.20, 5.60, 1.0),
             Word("Second", 6.00, 6.40, 1.0), Word("point", 6.40, 6.70, 1.0), Word("here.", 6.70, 7.00, 1.0),
             Word("Goodbye.", 8.00, 8.50, 1.0)]


def _ctx(judge, cfg: Config) -> Context:
    duration = 12.0
    db = np.full(int(duration / 0.005), -50.0)
    for w in SIX_WORDS:
        db[int(w.start / 0.005):int(w.end / 0.005)] = -20.0
    env = Envelope(db, np.zeros_like(db), 0.005, 16000)
    refined = [RefinedWord(w.text, w.start, w.end, w.start, w.end, 1.0) for w in SIX_WORDS]
    return Context(config=cfg, source=SourceInfo("main", "/v/in.mp4", "sha", duration), words=refined, env=env,
                   floor=-50.0, silences=[], moans=[], judge=judge, llm_cache=None, guidelines="G")


def test_blocks_are_judged_per_window_then_voted_once_and_cut_as_decided() -> None:
    cfg = Config(restructure=True, restructure_window_sec=5.0, restructure_context_sec=1.0, restructure_votes=3,
                 restructure_front=1)
    judge = KeyedJudge({"B02": ("drop", None), "B03": ("keep", 1)})
    value = rs.run(_ctx(judge, cfg), [])
    assert value["windows"] == 3
    blocks = rs.blocks_from(value)
    assert [(b.first, b.last) for b in blocks] == [(0, 1), (2, 3), (4, 5)]
    assert [b.decision for b in blocks] == ["keep", "drop", "keep"]
    assert [b.front_rank for b in blocks] == [None, None, 1]
    assert [(c.reason, round(c.start, 2), round(c.end, 2)) for c in rs.block_cuts(blocks)] == [("tangent", 3.6, 5.6)]


def test_parallel_windows_give_the_same_blocks_as_serial(monkeypatch) -> None:
    decisions = {"B02": ("drop", None), "B03": ("keep", 1)}
    cfg = Config(restructure=True, restructure_window_sec=5.0, restructure_context_sec=1.0, restructure_votes=3,
                 restructure_front=1)
    serial = rs.run(_ctx(KeyedJudge(decisions), cfg), [])
    monkeypatch.setattr(rs, "concurrent_safe", lambda judge: True)
    parallel = rs.run(_ctx(KeyedJudge(decisions), Config(**{**cfg.to_dict(), "restructure_parallel": 4})), [])
    assert parallel == serial


def test_a_stateful_judge_is_never_called_from_threads() -> None:
    assert not rs.concurrent_safe(FakeJudge([]))


def test_policy_fingerprint_is_stable_within_a_build() -> None:
    assert rs.policy() == rs.policy()


# ------------------------------------------------------------- end to end

TONES = [(0.5, 1.0), (1.6, 2.2), (3.0, 3.5), (4.0, 4.6)]
SRC_WORDS = [Word("Hello.", 0.5, 1.0, 1.0), Word("Second.", 1.6, 2.2, 1.0),
             Word("Third.", 3.0, 3.5, 1.0), Word("Goodbye.", 4.0, 4.6, 1.0)]
DURATION = 6.0


class OrderedTranscriber:
    """Source wav -> the canned words; render wav -> the words mapped through the plan, in output time order."""

    name, version = "fake", "1"

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    def transcribe(self, wav_path: str) -> Transcript:
        if Path(wav_path).parent.name == "audio":
            return Transcript(self.name, self.version, DURATION, list(SRC_WORDS))
        plan = Plan.from_json((self.run_dir / "plan.json").read_text())
        out = []
        for w in SRC_WORDS:
            s = plan.source_to_output(w.start + 0.01)
            if s is not None:
                out.append(Word(w.text, s, s + (w.end - w.start), 1.0))
        out.sort(key=lambda w: w.start)
        return Transcript(self.name, self.version, plan.output_duration, out)


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    expr = "+".join(f"between(t\\,{a}\\,{b})" for a, b in TONES)
    p = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=30:duration={DURATION}",
         "-f", "lavfi", "-i", f"aevalsrc='0.3*sin(440*2*PI*t)*({expr})+0.0005*random(0)':s=16000:d={DURATION}",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-b:a", "192k", str(p)],
        check=True,
    )
    return p


def _answers() -> list[dict]:
    # Structure (cold open S0001, outro S0003), retakes and content: the cook's usual answers.
    return ([{"true_start": "S0001", "why": "topic"}] * 5 + [{"last_sentence": "S0003", "why": "landing"}] * 5
            + [{"retakes": []}] * 5 + [{"cuts": []}] * 5)


def _window() -> dict:
    # One window over the kept sentences S0001..S0003: two blocks, the second has a payoff.
    return {"blocks": [
        {"start": "S0001", "title": "Second", "summary": "a point", "payoff": "a take", "payoff_sid": "S0001",
         "interest": 6, "relevance": 4, "depends_on_earlier": False, "depends_note": "", "lead_in": []},
        {"start": "S0002", "title": "Third and goodbye", "summary": "the close", "payoff": "a punchline",
         "payoff_sid": "S0002", "interest": 8, "relevance": 5, "depends_on_earlier": False, "depends_note": "",
         "lead_in": []},
    ]}


def _keep(front_for_b02: int | None = 1) -> dict:
    return {"blocks": [{"id": "B01", "decision": "keep", "reason": "a point", "front_rank": None},
                       {"id": "B02", "decision": "keep", "reason": "the close", "front_rank": front_for_b02}]}


def _cfg() -> Config:
    return Config(crf=30, preset="ultrafast", votes=5, agree=4, visuals_enabled=False, restructure=True,
                  restructure_votes=3, restructure_front=1)


def test_a_reordered_cut_renders_verifies_and_replays_from_its_memo(clip: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    cache = Cache(tmp_path / "cache")
    judge = FakeJudge(_answers() + [_window()] + [_keep()] * 3)
    first = run_pipeline(str(clip), out, _cfg(), transcriber=OrderedTranscriber(out), judge=judge,
                         cache=cache, guidelines="G", topic="t")
    assert first.ok, [c.line() for c in first.checks if not c.ok]
    plan = Plan.from_json(first.plan_path.read_text())
    assert not in_source_order(plan.segments)
    layout = {b["id"]: b for b in plan.provenance["blocks"]}
    assert layout["B02"]["output"][0] == 0.0 and layout["B02"]["front_rank"] == 1
    assert layout["B01"]["output"][0] > 0.0
    assert plan.output_duration == pytest.approx(first.facts["duration"], abs=0.3)
    assert "## Blocks" in first.report.read_text()
    assert judge.calls == 20 + 1 + 3

    # Replay: no judge calls, and a hand-edited decision is honoured.
    memo_path = out / "stages" / "restructure.json"
    memo = json.loads(memo_path.read_text())
    memo["value"]["blocks"][0]["decision"] = "drop"
    memo_path.write_text(json.dumps(memo, indent=2))
    strict = FakeJudge([])
    second = run_pipeline(str(clip), out, _cfg(), transcriber=OrderedTranscriber(out), judge=strict,
                          cache=cache, guidelines="G", topic="t")
    assert strict.calls == 0
    replayed = Plan.from_json(second.plan_path.read_text())
    # The dropped block (1.6-2.2 s) sits against the cold-open cut (0-1.6 s), so the two fuse and
    # the stronger reason survives: check the span is removed, not the label.
    assert any(c.start <= 1.7 and c.end >= 2.1 for c in replayed.cuts)
    assert {b["id"]: b["decision"] for b in replayed.provenance["blocks"]} == {"B01": "drop", "B02": "keep"}
