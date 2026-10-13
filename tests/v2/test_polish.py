"""Polish: gap caps by boundary, protected events, no cut across a word, judge agreement, memo replay.

Judges are fakes. The end-to-end case runs the full pipeline on a short synthetic clip, so the polish cut has
to render and verify, and a hand-edited polish memo has to be honoured on replay."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from talkinghead.audio import FRAME_SEC, Envelope, RefinedWord
from talkinghead.config import Config
from talkinghead.context import Context
from talkinghead.edges import EdgeContext, finalize_cuts
from talkinghead.judge import FakeJudge
from talkinghead.pipeline import run_pipeline
from talkinghead.stages import polish
from talkinghead.stages.restructure import Block, Window
from talkinghead.store import Cache
from talkinghead.timeline import Cut, Plan, SourceInfo
from talkinghead.transcribe import Transcript, Word


def W(text: str, start: float, end: float, speaker: str | None = None) -> RefinedWord:
    return RefinedWord(text, start, end, start, end, 1.0, speaker=speaker)


def _ctx(words: list[RefinedWord], *, cfg: Config | None = None, protected=(), judge=None) -> Context:
    env = Envelope(np.zeros(1), np.zeros(1), FRAME_SEC, 16000)
    return Context(
        config=cfg or Config(polish=True), source=SourceInfo("main", "/v/in.mp4", "sha", 10.0),
        words=words, env=env, floor=-50.0, silences=[], moans=[], judge=judge or FakeJudge([]),
        llm_cache=None, guidelines="G", topic="testing", protected_spans=list(protected),
    )


# "two." ends a sentence, then a 1.20 s gap (sentence end); "three"->"four" is a 0.30 s gap inside a
# sentence; "four"->"five." is a 1.20 s gap at a speaker change.
GAP_WORDS = [
    W("one", 0.00, 0.30, "S0"), W("two.", 0.30, 0.60, "S0"),
    W("three", 1.80, 2.10, "S0"), W("four", 2.40, 2.70, "S0"),
    W("five.", 3.90, 4.20, "S1"),
]


def _gap_cut(cuts: list[Cut], boundary: str) -> Cut:
    found = [c for c in cuts if c.reason == "gap" and c.provenance["boundary"] == boundary]
    assert len(found) == 1, cuts
    return found[0]


# ------------------------------------------------------------- gap caps

def test_each_gap_is_capped_by_the_boundary_it_sits_on() -> None:
    cuts = polish.gap_cuts(_ctx(GAP_WORDS), [], [])
    sentence = _gap_cut(cuts, "sentence")
    assert (sentence.start, sentence.end) == pytest.approx((0.75, 1.65))     # 0.30 kept: 0.15 each side
    assert _gap_cut(cuts, "within").length == pytest.approx(0.15)            # 0.30 gap, cap 0.15
    assert _gap_cut(cuts, "speaker").length == pytest.approx(1.00)           # 1.20 gap, cap 0.20


def test_a_block_boundary_takes_the_longest_cap() -> None:
    blocks = [Block("B01", 0, 1, 0.0, 0.60), Block("B02", 2, 2, 1.80, 2.10)]
    block = _gap_cut(polish.gap_cuts(_ctx(GAP_WORDS), [], blocks), "block")
    assert block.length == pytest.approx(0.75)                              # 1.20 gap, cap 0.45
    assert (block.start, block.end) == pytest.approx((0.825, 1.575))


def test_a_gap_already_within_its_cap_is_left_alone() -> None:
    words = [W("a", 0.0, 0.3, "S0"), W("b.", 0.3, 0.6, "S0"), W("c", 0.7, 1.0, "S0")]   # 0.10 gap, cap 0.30
    assert polish.gap_cuts(_ctx(words), [], []) == []


# ------------------------------------------------------------- what a gap cut may not touch

def test_protected_events_inside_a_gap_are_kept_whole() -> None:
    cuts = polish.gap_cuts(_ctx(GAP_WORDS, protected=[(1.0, 1.2)]), [], [])
    sentence_gap = sorted((c.start, c.end) for c in cuts if c.reason == "gap" and c.start < 1.65 and c.end > 0.75)
    assert [(round(a, 6), round(b, 6)) for a, b in sentence_gap] == [(0.75, 1.0), (1.2, 1.65)]
    assert not [c for c in cuts if c.start < 1.2 and c.end > 1.0]


def test_no_gap_cut_overlaps_a_word() -> None:
    cuts = polish.gap_cuts(_ctx(GAP_WORDS), [], [])
    assert cuts
    for c in cuts:
        assert all(not (c.start < w.end and c.end > w.start) for w in GAP_WORDS)


def test_a_word_removing_cut_over_the_gap_leaves_the_gap_alone() -> None:
    # A retake (or any word-removing cut) over the gap makes the two sides not one kept stretch.
    cuts = polish.gap_cuts(_ctx(GAP_WORDS), [Cut(0.65, 1.75, "retake")], [])
    assert not [c for c in cuts if c.start >= 0.6 and c.end <= 1.8]


def test_kept_audio_left_in_the_gap_is_the_cap_when_a_mechanical_cut_already_took_some() -> None:
    # The pauses stage already took 0.9-1.5 s out: kept audio in the gap is 0.3 + 0.3 s. Capping it at 0.30 s
    # cuts 0.30 s of kept audio, so the cut spans the silence and 0.15 s stays on each side.
    cuts = polish.gap_cuts(_ctx(GAP_WORDS), [Cut(0.9, 1.5, "silence")], [])
    sentence = _gap_cut(cuts, "sentence")
    assert (sentence.start, sentence.end) == pytest.approx((0.75, 1.65))
    assert (sentence.start - 0.60) + (1.80 - sentence.end) == pytest.approx(0.30)


def test_touching_words_have_no_gap_to_cut() -> None:
    words = [W("a", 0.0, 0.5, "S0"), W("b.", 0.5, 1.0, "S0")]
    assert polish.gap_cuts(_ctx(words), [], []) == []


# ------------------------------------------------------------- disfluency judge

DIS_WORDS = [
    W("I,", 0.00, 0.30, "S0"), W("I,", 0.35, 0.60, "S0"), W("we,", 0.65, 0.90, "S0"),
    W("we", 1.00, 1.20, "S0"), W("have", 1.25, 1.50, "S0"), W("a", 1.55, 1.60, "S0"), W("daily.", 1.65, 2.00, "S0"),
]


def _span(a: int, b: int, kind: str = "repeat") -> dict:
    return {"from_word": f"w{a:04d}", "to_word": f"w{b:04d}", "kind": kind, "why": "test"}


def _judge(ctx: Context, answers: list[dict]) -> dict:
    sents, groups = polish.kept_sentences(ctx, polish._Spans([]))
    ctx.judge = FakeJudge(answers)
    return polish._judge_window(ctx, sents, groups, Window((0, len(sents)), (0, len(sents))))


def test_a_span_is_cut_when_the_samples_agree_and_a_single_vote_is_dropped() -> None:
    answers = [{"cuts": [_span(0, 1)]}, {"cuts": [_span(0, 1), _span(5, 5, "filler")]}, {"cuts": [_span(0, 1)]}]
    out = _judge(_ctx(DIS_WORDS), answers)
    assert [(s["a"], s["b"], s["votes"], s["kind"]) for s in out["spans"]] == [(0, 1, 3, "repeat")]


def test_agreeing_samples_cut_only_the_span_they_share() -> None:
    # Narrow mode: the cut is the intersection of the agreeing spans, so an unsure edge keeps its word.
    answers = [{"cuts": [_span(0, 2)]}, {"cuts": [_span(0, 1)]}, {"cuts": [_span(0, 1, "false-start")]}]
    out = _judge(_ctx(DIS_WORDS), answers)
    assert [(s["a"], s["b"], s["votes"]) for s in out["spans"]] == [(0, 1, 3)]
    assert out["spans"][0]["kind"] == "repeat"   # two of the three samples said repeat


def test_unknown_ids_and_overlong_spans_are_dropped_and_counted() -> None:
    cfg = Config(polish=True, polish_max_sec=0.5)
    bad = {"from_word": "w0099", "to_word": "w0099", "kind": "repeat", "why": "x"}
    out = _judge(_ctx(DIS_WORDS, cfg=cfg), [{"cuts": [_span(0, 1), bad]}] * 3)
    assert out["spans"] == []                       # (0, 1) runs 0.6 s, over the 0.5 s limit
    assert out["stats"] == {"bad_ids": 3, "overlong": 3}


def test_disfluency_cuts_carry_their_reason_words_and_votes() -> None:
    results = [{"spans": [{"a": 0, "b": 1, "votes": 2, "kind": "repeat", "why": "stutter"}]}]
    cuts = polish.disfluency_cuts(_ctx(DIS_WORDS), results)
    assert len(cuts) == 1
    assert (cuts[0].start, cuts[0].end, cuts[0].reason) == (0.0, 0.60, "polish")
    assert cuts[0].provenance == {"words": [0, 1], "votes": 2, "kind": "repeat", "why": "stutter"}


def test_the_disfluency_prompt_names_every_kind_and_the_worked_examples() -> None:
    prompt = polish.polish_prompt("", "w0000 I,", "", "a talk")
    for kind in polish.KINDS:
        assert kind in prompt
    assert "I, I, we, we have a daily" in prompt and "Uh, I could have just" in prompt
    assert "Video topic (from the owner): a talk" in prompt


def test_policy_fingerprint_is_stable_within_a_build() -> None:
    assert polish.policy() == polish.policy()


# ------------------------------------------------------------- config

def test_polish_knobs_stay_out_of_every_other_stages_key() -> None:
    # Turning polish on or off, or tuning it, re-keys only the polish stages: a cook's structure, retake,
    # content, pause and visuals memos keep their keys (and their hand edits).
    from talkinghead.pipeline import STAGE_CONFIG_KEYS, _stage_config

    class Ctx:
        def __init__(self, config: Config) -> None:
            self.config = config

    base = Config(polish=False)
    for other in (Config(polish=True), Config(polish=True, polish_votes=5, polish_output_rounds=0),
                  Config(polish=True, polish_gap_block=0.1, polish_output_rounds=4)):
        for stage in ("structure", "retakes", "content", "pauses", "restructure", "visuals"):
            assert _stage_config(Ctx(other), stage) == _stage_config(Ctx(base), stage), stage
    assert _stage_config(Ctx(Config(polish=True, polish_votes=5)), "polish") != \
        _stage_config(Ctx(Config(polish=True)), "polish")
    assert "polish_output" in STAGE_CONFIG_KEYS and "polish_votes" in STAGE_CONFIG_KEYS["polish_output"]


def test_the_config_dict_keeps_polish_when_it_is_on_and_leaves_no_trace_when_off() -> None:
    assert Config().to_dict()["polish"] is True and Config().to_dict()["polish_output_rounds"] == 2
    off = Config(polish=False).to_dict()
    assert not [k for k in off if k == "polish" or k.startswith("polish_")]
    assert Config(polish=True).to_dict()["polish_gap_block"] == 0.45


def test_polish_agreement_cannot_exceed_its_votes() -> None:
    with pytest.raises(ValueError, match="polish_agree"):
        Config(polish=True, polish_votes=2, polish_agree=3).validate()


# ------------------------------------------------------------- end to end

TONES = [(0.5, 1.0), (1.6, 2.2), (3.0, 3.5), (4.0, 4.6)]
SRC_WORDS = [W("Hello.", 0.5, 1.0), W("Second.", 1.6, 2.2), W("Third.", 3.0, 3.5), W("Goodbye.", 4.0, 4.6)]
DURATION = 6.0


def _word(w: RefinedWord) -> Word:
    return Word(w.text, w.start, w.end, 1.0)


class OrderedTranscriber:
    """Source wav -> the canned words; render wav -> the words mapped through the plan, in output time order."""

    name, version = "fake", "1"

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    def transcribe(self, wav_path: str) -> Transcript:
        if Path(wav_path).parent.name == "audio":
            return Transcript(self.name, self.version, DURATION, [_word(w) for w in SRC_WORDS])
        plan = Plan.from_json((self.run_dir / "plan.json").read_text())
        out = []
        for w in SRC_WORDS:
            s = plan.source_to_output(w.start + 0.01)
            if s is not None:
                out.append(_word(W(w.text, s, s + (w.end - w.start))))
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
    return {"blocks": [
        {"start": "S0001", "title": "Second", "summary": "a point", "payoff": "a take", "payoff_sid": "S0001",
         "interest": 6, "relevance": 4, "depends_on_earlier": False, "depends_note": "", "lead_in": []},
        {"start": "S0002", "title": "Third and goodbye", "summary": "the close", "payoff": "a punchline",
         "payoff_sid": "S0002", "interest": 8, "relevance": 5, "depends_on_earlier": False, "depends_note": "",
         "lead_in": []},
    ]}


def _keep() -> dict:
    return {"blocks": [{"id": "B01", "decision": "keep", "reason": "a point", "front_rank": None},
                       {"id": "B02", "decision": "keep", "reason": "the close", "front_rank": None}]}


def _cfg() -> Config:
    return Config(crf=30, preset="ultrafast", votes=5, agree=4, visuals_enabled=False, restructure=True,
                  restructure_votes=3, restructure_front=0, polish=True)


def _covers_third(plan: Plan) -> bool:
    """Whether a cut removes the midpoint of "Third." (3.0-3.5 s)."""
    return any(c.start <= 3.25 < c.end for c in plan.cuts)


def test_a_polish_cut_renders_verifies_and_replays_from_its_memo(clip: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    cache = Cache(tmp_path / "cache")
    # The disfluency window: all three samples name "Third." (w0002) as a false start.
    span = {"cuts": [{"from_word": "w0002", "to_word": "w0002", "kind": "false-start", "why": "test"}]}
    judge = FakeJudge(_answers() + [_window()] + [_keep()] * 3 + [span] * 3)
    first = run_pipeline(str(clip), out, _cfg(), transcriber=OrderedTranscriber(out), judge=judge,
                         cache=cache, guidelines="G", topic="t")
    assert first.ok, [c.line() for c in first.checks if not c.ok]
    assert judge.calls == 20 + 1 + 3 + 3
    assert _covers_third(Plan.from_json(first.plan_path.read_text()))
    assert "## Polish" in first.report.read_text()
    stored = json.loads((out / "stages" / "polish.json").read_text())["value"]["cuts"]
    assert [c["reason"] for c in stored if c["reason"] == "polish"] == ["polish"]

    # Replay: no judge calls, and a hand-edited memo is honoured (the disfluency cut is removed by hand).
    memo_path = out / "stages" / "polish.json"
    memo = json.loads(memo_path.read_text())
    memo["value"]["cuts"] = [c for c in memo["value"]["cuts"] if c["reason"] != "polish"]
    memo_path.write_text(json.dumps(memo, indent=2))
    strict = FakeJudge([])
    second = run_pipeline(str(clip), out, _cfg(), transcriber=OrderedTranscriber(out), judge=strict,
                          cache=cache, guidelines="G", topic="t")
    assert strict.calls == 0
    assert not _covers_third(Plan.from_json(second.plan_path.read_text()))


def test_no_cut_takes_part_of_a_word_whose_asr_stamp_runs_ahead_of_its_refined_bounds() -> None:
    # Shape of the voiced-gap fixture: the next word's refined burst starts at 7.745 but its ASR stamp at 7.30. A
    # silence cut from the pause stage (7.334-7.766) reaches into that stamp; with the gap cut before it, the pair
    # fuses into one gap cut that clips the word. Guards must use the stamped extent, so nothing lands in 7.30-8.00.
    db = np.full(int(10.0 / FRAME_SEC), -50.0)
    env = Envelope(db, np.zeros_like(db), FRAME_SEC, 16000)
    words = [RefinedWord(" tasks", 4.9, 5.3, 4.9, 5.3, 1.0),
             RefinedWord(" And", 7.745, 7.875, 7.30, 8.00, 1.0)]
    ctx = EdgeContext(duration=10.0, env=env, floor=-50.0, words=words, fps=None)
    out = finalize_cuts([Cut(5.315, 7.15, "gap", None, {"boundary": "sentence"}),
                         Cut(7.334, 7.766, "silence")], ctx)
    assert out, "the gap must still be cut"
    assert all(c.end <= 7.30 or c.start >= 8.00 for c in out), out

