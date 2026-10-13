"""The output pass end to end: a stutter in the render is judged, cut from source, re-rendered and gone."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

import talkinghead.pipeline as pipeline
from talkinghead.audio import FRAME_SEC, Envelope, RefinedWord
from talkinghead.config import Config
from talkinghead.edges import EdgeContext, finalize_cuts
from talkinghead.judge import FakeJudge
from talkinghead.pipeline import run_pipeline
from talkinghead.store import Cache
from talkinghead.timeline import Cut, Plan
from talkinghead.transcribe import Transcript, Word

# Five tones: "Second." is said twice in a row, 0.1 s apart (a stutter the source transcript can show). Each word
# has its own pitch, so the source-timing check cannot line a tone up with an identical tone 1.1 s away.
TONES = [(0.5, 1.0), (1.6, 2.2), (2.3, 2.9), (3.2, 3.7), (4.2, 4.8)]
PITCHES = [330, 440, 550, 660, 770]
SRC_WORDS = [Word("Hello.", 0.5, 1.0, 1.0), Word("Second.", 1.6, 2.2, 1.0), Word("Second.", 2.3, 2.9, 1.0),
             Word("Third.", 3.2, 3.7, 1.0), Word("Goodbye.", 4.2, 4.8, 1.0)]
DURATION = 6.0


class PlanAwareTranscriber:
    """Source audio -> the canned words; the render -> the same words mapped through the plan on disk.

    Words the plan cut are absent from the render, as a real ASR would hear them.
    """

    name, version = "fake", "1"

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.calls = 0

    def transcribe(self, wav_path: str) -> Transcript:
        self.calls += 1
        if Path(wav_path).parent.name == "audio":
            return Transcript(self.name, self.version, DURATION, list(SRC_WORDS))
        plan = Plan.from_json((self.run_dir / "plan.json").read_text())
        out = []
        for w in SRC_WORDS:
            s = plan.source_to_output(w.start + 0.01)
            if s is not None:
                out.append(Word(w.text, s, s + (w.end - w.start), 1.0))
        return Transcript(self.name, self.version, plan.output_duration, out)


class NoTranscriber:
    """Plan-only rebuilds must not transcribe anything: any call is a failure."""

    name, version = "fake", "1"

    def transcribe(self, wav_path: str) -> Transcript:
        raise AssertionError("plan-only transcribed " + wav_path)


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    expr = "+".join(f"sin(2*PI*{f}*t)*between(t\\,{a}\\,{b})" for (a, b), f in zip(TONES, PITCHES))
    p = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=30:duration={DURATION}",
         "-f", "lavfi", "-i", f"aevalsrc='0.3*({expr})+0.0005*random(0)':s=16000:d={DURATION}",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-b:a", "192k", str(p)],
        check=True,
    )
    return p


def _sources() -> list[dict]:
    """Structure (cold open S0000, outro S0004), retakes and content: nothing to remove."""
    return ([{"true_start": "S0000", "why": "topic"}] * 5 + [{"last_sentence": "S0004", "why": "landing"}] * 5
            + [{"retakes": []}] * 5 + [{"cuts": []}] * 5)


SOURCE_POLISH = [{"cuts": []}] * 3  # the source disfluency judge: one window, three samples, nothing to cut


def _verdict(decision: str) -> dict:
    return {"verdicts": [{"id": "c0", "verdict": decision, "why": "test"}]}


def _cfg(**overrides) -> Config:
    return Config(crf=30, preset="ultrafast", votes=5, agree=4, visuals_enabled=False, polish=True, **overrides)


def _covers(plan: Plan, t: float) -> bool:
    return any(c.start <= t < c.end for c in plan.cuts)


def _output_cuts(plan: Plan) -> list:
    return [c for c in plan.cuts if c.reason == "polish" and c.provenance.get("pass") == "output"]


def test_a_stutter_in_the_render_is_cut_from_source_and_gone_in_the_next_round(clip: Path, tmp_path: Path) -> None:
    out, cache = tmp_path / "out", Cache(tmp_path / "cache")
    judge = FakeJudge(_sources() + SOURCE_POLISH + [_verdict("cut")] * 3)
    result = run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out), judge=judge,
                          cache=cache, guidelines="G", topic="t")
    assert result.ok, [c.line() for c in result.checks if not c.ok]
    # Structure 20, source polish 3, output round 3. Round 2 finds nothing and asks nothing.
    assert judge.calls == 26
    plan = Plan.from_json(result.plan_path.read_text())
    (cut,) = _output_cuts(plan)
    assert cut.provenance["kind"] == "repeat" and cut.provenance["round"] == 1
    assert _covers(plan, 1.9) and not _covers(plan, 2.6)  # the first "Second." goes, the second stays
    assert result.facts["disfluency_findings"] == []  # the final render shows no filler, repeat or restart
    assert sum(w.text == "Second." for w in result.facts["words"]) == 1
    output = result.notes["polish"]["output"]
    assert [r["findings"] for r in output["rounds"]] == [1, 0]
    assert output["keep"] == [] and output["residual"] == []
    report = result.report.read_text()
    assert "### Output pass" in report and "round 1: 1 findings, 1 cut" in report


def test_a_finding_the_judge_keeps_is_left_in_and_listed_as_judged_keep(clip: Path, tmp_path: Path) -> None:
    out, cache = tmp_path / "out", Cache(tmp_path / "cache")
    judge = FakeJudge(_sources() + SOURCE_POLISH + [_verdict("keep")] * 3)
    result = run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out), judge=judge,
                          cache=cache, guidelines="G", topic="t")
    assert result.ok and judge.calls == 26
    plan = Plan.from_json(result.plan_path.read_text())
    assert _output_cuts(plan) == []
    # The repeat is still there, but it is a judged keep, so the final check does not flag it.
    output = result.notes["polish"]["output"]
    assert [(k["kind"], k["text"]) for k in output["keep"]] == [("repeat", "Second.")]
    assert [r["findings"] for r in output["rounds"]] == [1]
    check = next(c for c in result.checks if c.name == "residual_disfluencies")
    assert check.ok


def test_with_no_output_rounds_the_leftover_stutter_is_a_warning_not_a_failure(clip: Path, tmp_path: Path) -> None:
    out, cache = tmp_path / "out", Cache(tmp_path / "cache")
    judge = FakeJudge(_sources() + SOURCE_POLISH)
    result = run_pipeline(str(clip), out, _cfg(polish_output_rounds=0), transcriber=PlanAwareTranscriber(out),
                          judge=judge, cache=cache, guidelines="G", topic="t")
    assert result.ok and judge.calls == 23
    check = next(c for c in result.checks if c.name == "residual_disfluencies")
    assert check.severity == "warn" and not check.ok and "Second." in check.detail
    assert result.notes["polish"]["output"]["skipped"] == "polish_output_rounds is 0"
    assert [x["heard"] for x in result.notes["polish"]["output"]["residual"]] == ["Second."]
    assert "WARN" in check.line() and "### Output pass" in result.report.read_text()


def test_without_output_asr_the_pass_is_skipped_and_the_report_says_so(clip: Path, tmp_path: Path) -> None:
    out, cache = tmp_path / "out", Cache(tmp_path / "cache")
    judge = FakeJudge(_sources() + SOURCE_POLISH)
    result = run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out), judge=judge,
                          cache=cache, guidelines="G", topic="t", verify_output_asr=False)
    assert result.ok and judge.calls == 23
    check = next(c for c in result.checks if c.name == "residual_disfluencies")
    assert check.severity == "info" and check.ok
    assert "output ASR is off" in result.notes["polish"]["output"]["skipped"]
    assert "output ASR is off" in result.report.read_text()


def test_plan_only_replays_the_stored_output_round_without_transcribing(clip: Path, tmp_path: Path) -> None:
    out, cache = tmp_path / "out", Cache(tmp_path / "cache")
    first = run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out),
                         judge=FakeJudge(_sources() + SOURCE_POLISH + [_verdict("cut")] * 3),
                         cache=cache, guidelines="G", topic="t")
    strict = FakeJudge([])
    again = run_pipeline(str(clip), out, _cfg(), transcriber=NoTranscriber(), judge=strict, cache=cache,
                         guidelines="G", topic="t", plan_only=True)
    assert strict.calls == 0
    assert again.plan_path.read_bytes() == first.plan_path.read_bytes()
    assert "plan-only" not in again.notes["polish"]["output"].get("skipped", "")


def test_the_output_round_memo_is_written_per_round(clip: Path, tmp_path: Path) -> None:
    out, cache = tmp_path / "out", Cache(tmp_path / "cache")
    run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out),
                 judge=FakeJudge(_sources() + SOURCE_POLISH + [_verdict("cut")] * 3),
                 cache=cache, guidelines="G", topic="t")
    round1 = json.loads((out / "stages" / "polish-output-1.json").read_text())["value"]
    round2 = json.loads((out / "stages" / "polish-output-2.json").read_text())["value"]
    assert len(round1["cuts"]) == 1 and round2["cuts"] == []


def test_an_output_cut_is_removed_whole_between_its_neighbours_not_shrunk_by_the_edge_chain() -> None:
    # Shape of smack-958: the filler "uh" is a quiet gap between "itself," (kept, ends 4.34) and "just" (kept,
    # starts 4.66). Round 1 confirms 4.30-4.78 on the render. The span must come back whole, bounded by the
    # neighbours, so the filler is gone and the round-2 residual is clean.
    db = np.full(int(10.0 / FRAME_SEC), -50.0)
    env = Envelope(db, np.zeros_like(db), FRAME_SEC, 16000)
    words = [RefinedWord(" itself,", 3.88, 4.34, 3.88, 4.34, 1.0),
             RefinedWord(" just", 4.66, 5.34, 4.66, 5.34, 1.0)]
    ctx = EdgeContext(duration=10.0, env=env, floor=-50.0, words=words, fps=None)
    confirmed = Cut(4.30, 4.78, "polish", "uh", {"pass": "output", "kind": "filler", "span": [4.30, 4.78]})
    out = finalize_cuts([confirmed], ctx)
    assert len(out) == 1 and out[0].reason == "polish"
    assert out[0].start == pytest.approx(4.34, abs=1e-6)
    assert out[0].end == pytest.approx(4.66, abs=1e-6)


def test_an_unchanged_resume_replays_the_judged_rounds_without_a_judge_call(clip: Path, tmp_path: Path) -> None:
    out, cache = tmp_path / "out", Cache(tmp_path / "cache")
    first = run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out),
                         judge=FakeJudge(_sources() + SOURCE_POLISH + [_verdict("cut")] * 3),
                         cache=cache, guidelines="G", topic="t")
    strict = FakeJudge([])
    again = run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out), judge=strict,
                         cache=cache, guidelines="G", topic="t")
    assert strict.calls == 0 and again.ok
    # Nothing judged was superseded: every output round replayed from its stored memo.
    assert "polish-output" not in again.notes.get("memos", {}).get("recomputed", "")
    assert _output_cuts(Plan.from_json(again.plan_path.read_text())) == _output_cuts(
        Plan.from_json(first.plan_path.read_text()))


class EditedRenderTranscriber(PlanAwareTranscriber):
    """The same render, heard differently: the output ASR now reads "Third." as "Thirty."."""

    version = "2"  # a changed transcriber is a new version: the transcript cache must not return the old words

    def transcribe(self, wav_path: str) -> Transcript:
        t = super().transcribe(wav_path)
        if Path(wav_path).parent.name == "audio":
            return t
        words = [Word(w.text.replace("Third.", "Thirty."), w.start, w.end, w.probability) for w in t.words]
        return Transcript(t.backend, t.version, t.duration, words)


def test_a_changed_rendered_transcript_rejudges_the_output_round(clip: Path, tmp_path: Path) -> None:
    out, cache = tmp_path / "out", Cache(tmp_path / "cache")
    run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out),
                 judge=FakeJudge(_sources() + SOURCE_POLISH + [_verdict("cut")] * 3),
                 cache=cache, guidelines="G", topic="t")
    again = run_pipeline(str(clip), out, _cfg(), transcriber=EditedRenderTranscriber(out),
                         judge=FakeJudge([_verdict("cut")] * 3), cache=cache, guidelines="G", topic="t")
    # Round 1 saw a different render transcript, so its stored memo is superseded and the round runs again.
    assert "polish-output-1" in again.notes["memos"]["recomputed"]


def test_a_changed_render_chain_rejudges_the_output_round(clip: Path, tmp_path: Path, monkeypatch) -> None:
    out, cache = tmp_path / "out", Cache(tmp_path / "cache")
    run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out),
                 judge=FakeJudge(_sources() + SOURCE_POLISH + [_verdict("cut")] * 3),
                 cache=cache, guidelines="G", topic="t")
    # A changed ffmpeg chain renders the same plan to different pixels and audio: the round must not replay.
    monkeypatch.setattr(pipeline, "RENDERER_VERSION", pipeline.RENDERER_VERSION + 1)
    again = run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out),
                         judge=FakeJudge([_verdict("cut")] * 3), cache=cache, guidelines="G", topic="t")
    assert "polish-output-1" in again.notes["memos"]["recomputed"]



def test_fused_output_cuts_keep_the_union_of_their_confirmed_spans() -> None:
    # Shape of the 683 s judged span: two confirmed output cuts overlap ("I--" and "I just--"). Fused, the cut
    # must keep both spans, or the edge chain re-settles it from the first span and leaves "just" and "uh" in.
    db = np.full(int(10.0 / FRAME_SEC), -50.0)
    env = Envelope(db, np.zeros_like(db), FRAME_SEC, 16000)
    words = [RefinedWord(" I", 3.60, 3.69, 3.60, 3.69, 1.0),
             RefinedWord(" just", 3.75, 4.20, 3.75, 4.20, 1.0),
             RefinedWord(" uh", 4.30, 4.60, 4.30, 4.60, 1.0),
             RefinedWord(" it,", 4.70, 5.10, 4.70, 5.10, 1.0),
             RefinedWord(" now", 5.30, 5.80, 5.30, 5.80, 1.0)]
    ctx = EdgeContext(duration=10.0, env=env, floor=-50.0, words=words, fps=None)
    first = Cut(3.60, 3.69, "polish", "I--", {"pass": "output", "span": [3.60, 3.69]})
    second = Cut(3.64, 5.12, "polish", "I just--", {"pass": "output", "span": [3.64, 5.12]})
    out = finalize_cuts([first, second], ctx)
    assert len(out) == 1
    assert out[0].start <= 3.60 + 1e-6 and out[0].end >= 5.10 - 1e-6  # "it," is removed, not kept
    assert out[0].provenance["span"] == pytest.approx([3.60, 5.12])
