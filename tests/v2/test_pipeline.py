"""End to end with fakes: one command, one plan, one render, one report; a
rerun replays every verdict and produces a byte-identical plan."""

import json
import subprocess
from pathlib import Path

import pytest

from talkinghead.align import AlignResult
from talkinghead.config import Config
from talkinghead.context import NeedsDecision
from talkinghead.judge import FakeJudge
from talkinghead.pipeline import PipelineError, make_clip, run_pipeline
from talkinghead.store import Cache, RunDirConflict
from talkinghead.timeline import Plan
from talkinghead.transcribe import Transcript, Word

# Four tone "words" over near-silence; sentence-final punctuation on each.
TONES = [(0.5, 1.0), (1.6, 2.2), (3.0, 3.5), (4.0, 4.6)]
SRC_WORDS = [Word("Hello.", 0.5, 1.0, 1.0), Word("Second.", 1.6, 2.2, 1.0),
             Word("Third.", 3.0, 3.5, 1.0), Word("Goodbye.", 4.0, 4.6, 1.0)]
DURATION = 6.0


def _clip(path: Path) -> None:
    expr = "+".join(f"between(t\\,{a}\\,{b})" for a, b in TONES)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=30:duration={DURATION}",
         "-f", "lavfi", "-i", f"aevalsrc='0.3*sin(440*2*PI*t)*({expr})+0.0005*random(0)':s=16000:d={DURATION}",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-b:a", "192k", str(path)],
        check=True,
    )


class PlanAwareTranscriber:
    """Source wav -> the canned words; render wav -> the same words mapped
    through the plan on disk (what a real ASR would hear on the render)."""

    name, version = "fake", "1"

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.calls = 0

    def transcribe(self, wav_path: str) -> Transcript:
        self.calls += 1
        if Path(wav_path).parent.name == "audio":
            return Transcript(self.name, self.version, DURATION, list(SRC_WORDS))
        plan_file = next(p for p in ("demo-plan.json", "plan.json") if (self.run_dir / p).is_file())
        plan = Plan.from_json((self.run_dir / plan_file).read_text())
        out = []
        for w in SRC_WORDS:
            s = plan.source_to_output(w.start + 0.01)
            if s is not None:
                out.append(Word(w.text, s, s + (w.end - w.start), 1.0))
        return Transcript(self.name, self.version, plan.output_duration, out)


def _answers(cold_open="S0001", outro="S0003"):
    return ([{"true_start": cold_open, "why": "topic"}] * 5
            + ([{"last_sentence": outro, "why": "landing"}] * 5 if outro else [])
            + [{"retakes": []}] * 5 + [{"cuts": []}] * 5)


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    p = tmp_path / "clip.mp4"
    _clip(p)
    return p


def _cfg() -> Config:
    # Visuals off: these tests pin the CUT pipeline's contract (exact judge
    # call counts, byte-identical plans). The overlay pass has its own test
    # below, and its own judge calls would make these counts meaningless.
    return Config(crf=30, preset="ultrafast", votes=5, agree=4, visuals_enabled=False)


def test_run_produces_plan_render_and_report(clip: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    judge = FakeJudge(_answers())
    result = run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out), judge=judge,
                          cache=Cache(tmp_path / "cache"), guidelines="G", topic="t")
    assert result.final.is_file() and result.plan_path.is_file() and result.report.is_file()
    plan = Plan.from_json(result.plan_path.read_text())
    reasons = {c.reason for c in plan.cuts}
    assert {"cold-open", "outro", "silence"} <= reasons
    assert plan.decisions["cold_open"]["chosen"] == "S0001"
    assert plan.output_duration == pytest.approx(result.facts["duration"], abs=0.3)
    report = result.report.read_text()
    assert "## Decisions" in report and "## Verification" in report and "## Cuts" in report
    assert judge.calls == 20
    assert result.ok, [c.line() for c in result.checks if not c.ok]


def test_rerun_replays_every_verdict_and_reproduces_the_plan_byte_for_byte(clip: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    cache = Cache(tmp_path / "cache")
    first = run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out), judge=FakeJudge(_answers()),
                         cache=cache, guidelines="G")
    plan_bytes = first.plan_path.read_bytes()
    strict = FakeJudge([])  # any call would raise StopIteration
    second = run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out), judge=strict,
                          cache=cache, guidelines="G")
    assert strict.calls == 0
    assert second.plan_path.read_bytes() == plan_bytes


class PromptRecorder(FakeJudge):
    """FakeJudge that keeps every prompt it was asked."""

    def __init__(self, responses) -> None:
        super().__init__(responses)
        self.prompts: list[str] = []

    def decide(self, prompt: str, schema: dict) -> dict:
        self.prompts.append(prompt)
        return super().decide(prompt, schema)


def test_preferences_reach_every_judge_and_the_cook_keeps_its_snapshot(clip: Path, tmp_path: Path) -> None:
    out, cache = tmp_path / "out", Cache(tmp_path / "cache")
    judge = PromptRecorder(_answers())
    first = run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out), judge=judge,
                         cache=cache, guidelines="G", preferences="- keep laughs")
    assert judge.calls == 20 and all("- keep laughs" in p for p in judge.prompts)
    assert (out / "preferences.md").read_text().strip() == "- keep laughs"

    # The profile moved on: this cook replays what it was judged with.
    strict = FakeJudge([])
    again = run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out), judge=strict,
                         cache=cache, guidelines="G", preferences="- cut tangents")
    assert strict.calls == 0 and again.plan_path.read_bytes() == first.plan_path.read_bytes()

    # Opting in asks the judges again under the new preferences.
    fresh = PromptRecorder(_answers())
    run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out), judge=fresh,
                 cache=cache, guidelines="G", preferences="- cut tangents", refresh_preferences=True)
    assert fresh.calls == 20 and all("- cut tangents" in p and "keep laughs" not in p for p in fresh.prompts)


def test_a_cook_directory_belongs_to_one_recording(clip: Path, tmp_path: Path) -> None:
    """Two recordings sharing one run directory would mix their renders and
    hand the second one the first one's persisted decisions."""
    out = tmp_path / "out"
    cache = Cache(tmp_path / "cache")
    first = run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out),
                         judge=FakeJudge(_answers()), cache=cache, guidelines="G")
    assert first.final.is_file()

    other = tmp_path / "other.mp4"
    other.write_bytes(clip.read_bytes() + b"\x00")

    with pytest.raises(RunDirConflict) as exc:
        run_pipeline(str(other), out, _cfg(), transcriber=PlanAwareTranscriber(out),
                     judge=FakeJudge([]), cache=cache, guidelines="G")
    assert "--out-dir" in str(exc.value)
    assert json.loads((out / "run.json").read_text())["source"] == str(clip.resolve())

    # The recording that owns the directory still resumes in it.
    again = run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out),
                         judge=FakeJudge([]), cache=cache, guidelines="G")
    assert again.plan_path.read_bytes() == first.plan_path.read_bytes()

    # A directory of its own takes the other recording.
    second = run_pipeline(str(other), tmp_path / "other-out", _cfg(), transcriber=PlanAwareTranscriber(tmp_path / "other-out"),
                          judge=FakeJudge(_answers()), cache=cache, guidelines="G")
    assert second.final.is_file()


def test_split_cold_open_vote_stops_and_asks_then_the_answer_is_persisted(clip: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    cache = Cache(tmp_path / "cache")
    split = FakeJudge([{"true_start": "S0001", "why": ""}] * 3 + [{"true_start": "S0002", "why": ""}] * 2)
    with pytest.raises(NeedsDecision) as exc:
        run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out), judge=split,
                     cache=cache, guidelines="G")
    assert exc.value.key == "cold_open"
    assert "cold open" in (out / "report.md").read_text().lower()
    # The user answers once; the run completes and remembers.
    rest = FakeJudge([{"last_sentence": "S0003", "why": ""}] * 5 + [{"retakes": []}] * 5 + [{"cuts": []}] * 5)
    result = run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out), judge=rest,
                          cache=cache, guidelines="G", decisions={"cold_open": 1.6})
    assert result.ok
    assert json.loads((out / "decisions.json").read_text())["cold_open"] == 1.6
    plan = Plan.from_json(result.plan_path.read_text())
    assert plan.decisions["cold_open"]["source"] == "user"


def test_demo_mode_clips_the_head_and_renders_the_draft_tier(clip: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out),
                          judge=FakeJudge(_answers(outro=None)), cache=Cache(tmp_path / "cache"), guidelines="G",
                          demo_minutes=DURATION / 60.0)
    assert result.final.name == "demo-final.mp4"
    assert (out / "demo-clip.mp4").is_file()
    # No outro question on a truncated clip: the run keeps to the clip end.
    plan = Plan.from_json(result.plan_path.read_text())
    assert plan.decisions["outro"]["source"] == "user" and plan.decisions["outro"]["end"] == pytest.approx(DURATION)
    assert not [c for c in plan.cuts if c.reason == "outro"]


def test_a_crash_after_the_decisions_resumes_without_asking_the_judge_again(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    cache = Cache(tmp_path / "cache")
    judge = FakeJudge(_answers())

    def boom(*a, **k):
        raise RuntimeError("ffmpeg died")

    monkeypatch.setattr("talkinghead.pipeline.render_plan", boom)
    with pytest.raises(RuntimeError, match="ffmpeg died"):
        run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out), judge=judge,
                     cache=cache, guidelines="G")
    assert judge.calls == 20
    monkeypatch.undo()
    strict = FakeJudge([])
    result = run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out), judge=strict,
                          cache=cache, guidelines="G")
    assert strict.calls == 0 and result.ok


def test_rerun_skips_the_render_when_the_plan_and_settings_are_unchanged(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    cache = Cache(tmp_path / "cache")
    run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out), judge=FakeJudge(_answers()),
                 cache=cache, guidelines="G")
    calls = []
    import talkinghead.pipeline as pl
    real = pl.render_plan
    monkeypatch.setattr(pl, "render_plan", lambda *a, **k: calls.append(1) or real(*a, **k))
    run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out), judge=FakeJudge([]),
                 cache=cache, guidelines="G")
    assert calls == []
    # A changed render tier re-renders.
    cfg = _cfg()
    cfg.crf = 31
    run_pipeline(str(clip), out, cfg, transcriber=PlanAwareTranscriber(out), judge=FakeJudge([]),
                 cache=cache, guidelines="G")
    assert calls == [1]


# ------------------------------------------------------- alignment (step 0)

def _fake_align(calls: list, *, confidence: float | None = 9.0, offset: float = 1.25, drift: float = 12.5,
                seen: dict | None = None):
    """A stand-in aligner: the "master" is the source clip's own picture and
    audio, so the run after it is a real run; only the measurement is canned."""

    def run(video, audio, out, *, fix_drift=False):
        calls.append(str(out))
        if seen is not None:
            seen["fix_drift"] = fix_drift
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(Path(video).read_bytes())
        return AlignResult(offset_sec=offset, drift_ppm=drift, confidence=confidence,
                           output=out, video_duration=DURATION, audio_duration=DURATION,
                           windows_used=6, windows_total=7)

    return run


def test_a_run_with_an_external_recording_aligns_once_and_reuses_the_master(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    calls: list[str] = []
    seen: dict = {}
    monkeypatch.setattr("talkinghead.pipeline.align", _fake_align(calls, seen=seen))
    other = tmp_path / "other-recording.mov"
    other.write_bytes(clip.read_bytes() + b"\x00")  # a different file: a different measurement

    first = run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out),
                         judge=FakeJudge(_answers()), cache=Cache(tmp_path / "cache"),
                         guidelines="G", audio=str(clip))
    master, measured = out / "master.mov", json.loads((out / "align.json").read_text())
    assert calls == [str(master)] and seen["fix_drift"] is False   # measured exactly once, no stretch
    assert master.is_file() and measured["output"] == str(master)
    assert measured["offset_sec"] == 1.25 and measured["drift_ppm"] == 12.5
    plan = Plan.from_json(first.plan_path.read_text())
    assert Path(plan.sources[plan.main].path).resolve() == master.resolve()  # every stage cut the master
    assert first.notes["align"]["offset_sec"] == 1.25 and first.notes["align"]["confidence"] == 9.0
    assert "- align:" in first.report.read_text()

    # A second run on the same pair reuses the finished master and its
    # measurement: the aligner is not called again and the plan is identical.
    second = run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out),
                          judge=FakeJudge([]), cache=Cache(tmp_path / "cache"),
                          guidelines="G", audio=str(clip))
    assert calls == [str(master)]
    assert second.plan_path.read_bytes() == first.plan_path.read_bytes()

    # ...but another recording, or a different stretch, must never reuse a
    # master aligned from the old one.
    run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out),
                 judge=FakeJudge(_answers()), cache=Cache(tmp_path / "cache"),
                 guidelines="G", audio=str(other))
    assert len(calls) == 2
    run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out),
                 judge=FakeJudge(_answers()), cache=Cache(tmp_path / "cache"),
                 guidelines="G", audio=str(other), fix_drift=True)
    assert len(calls) == 3 and seen["fix_drift"] is True


@pytest.mark.parametrize("confidence", [1.5, None])
def test_an_untrustworthy_alignment_stops_the_run_unless_allowed(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, confidence: float | None
) -> None:
    out = tmp_path / "out"
    monkeypatch.setattr("talkinghead.pipeline.align", _fake_align([], confidence=confidence))
    transcriber = PlanAwareTranscriber(out)
    cache = Cache(tmp_path / "cache")

    with pytest.raises(PipelineError) as exc:
        run_pipeline(str(clip), out, _cfg(), transcriber=transcriber, judge=FakeJudge([]),
                     cache=cache, guidelines="G", audio=str(clip))
    message = str(exc.value)
    assert "untrustworthy" in message and "--allow-bad-align" in message
    assert ("below the 3.0x" if confidence else "share no signal") in message
    assert transcriber.calls == 0          # nothing is analysed against a bad offset
    assert (out / "align.json").is_file()  # the measurement stays on disk to inspect

    # The override cuts anyway, and the report says what it was cut against.
    result = run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out),
                          judge=FakeJudge(_answers()), cache=cache, guidelines="G",
                          audio=str(clip), allow_bad_align=True)
    assert result.plan_path.is_file()
    expected = "n/a" if confidence is None else confidence
    assert result.notes["align"]["confidence"] == expected



def test_visuals_pass_writes_a_deliverable_and_leaves_the_cut_alone(clip: Path, tmp_path: Path) -> None:
    """The whole promise of the second pass: `final.mp4` is untouched and
    `final-visuals.mp4` has the same duration and the same audio stream."""
    from talkinghead.judge import validate
    from talkinghead.visuals.render import probe_duration

    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "manifest.toml").write_text("")

    class VisualsJudge:
        name = "vis"

        def __init__(self) -> None:
            self.seq = list(_answers())
            self.calls = 0

        def decide(self, prompt: str, schema: dict) -> dict:
            self.calls += 1
            if "TASK: EDITORIAL PLAN" in prompt:
                return validate({
                    "central_argument": "the second point is the whole thing",
                    "chapters": [
                        {"title": "The second point", "anchor_text": "Second.", "purpose": "b"},
                        {"title": "The third point", "anchor_text": "Third.", "purpose": "b"}],
                    "visuals": [{
                        "chapter": 1, "anchor_text": "Second.", "type": "quote_card",
                        "quote": "Second.", "visual_brief": "", "editorial_reason": "payoff",
                        "hold_sec": 3.0}]}, schema)
            if "TASK: RESEARCH" in prompt:
                return validate({"sources": []}, schema)
            if "TASK: MARKER" in prompt:
                return validate({"verdict": "keep", "why": "n/a"}, schema)
            return validate(self.seq.pop(0), schema)

    out = tmp_path / "out"
    cfg = Config(crf=30, preset="ultrafast", votes=5, agree=4,
                 visuals_enabled=True, visuals_research=False, visuals_min_hold=1.0,
                 plan_enabled=True, plan_hold=1.0, plan_open_at=0.0, plan_open_hold=1.0,
                 visuals_min_gap=0.4)
    result = run_pipeline(str(clip), out, cfg, transcriber=PlanAwareTranscriber(out),
                          judge=VisualsJudge(), cache=Cache(tmp_path / "cache"),
                          guidelines="G", topic="t", assets_dir=assets)
    assert result.final_visuals is not None and result.final_visuals.is_file()
    assert result.final.is_file()
    base_d, vis_d = probe_duration(result.final), probe_duration(result.final_visuals)
    assert abs(vis_d - base_d) <= 0.05, f"visuals moved the duration: {vis_d} vs {base_d}"
    report = result.report.read_text()
    assert "## Asset decisions" in report
    assert "quote_card" in report
    assert "## Plan card" in report
    assert "visuals_duration" in report and "visuals_audio_copied" in report
    assert result.ok, [c.line() for c in result.checks if not c.ok]


def test_visuals_can_be_skipped(clip: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out),
                          judge=FakeJudge(_answers()), cache=Cache(tmp_path / "cache"),
                          guidelines="G")
    assert result.final_visuals is None
    assert "## Asset decisions" not in result.report.read_text()


def test_render_memo_ignores_settings_the_renderer_never_reads(clip: Path, tmp_path: Path) -> None:
    """Analysis and overlay settings must not replace an unchanged finished cut."""
    out = tmp_path / "out"
    cache = Cache(tmp_path / "cache")
    first = run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out),
                         judge=FakeJudge(_answers()), cache=cache, guidelines="G")
    rendered_at = first.final.stat().st_mtime_ns
    original = first.final.read_bytes()
    cfg = _cfg()
    cfg.visuals_min_hold = 9.0
    cfg.visuals_research = False
    cfg.speech_probe_max = 1
    second = run_pipeline(str(clip), out, cfg, transcriber=PlanAwareTranscriber(out),
                          judge=FakeJudge([]), cache=cache, guidelines="G")
    assert second.final.stat().st_mtime_ns == rendered_at
    assert second.final.read_bytes() == original
    cfg.crf = 40
    third = run_pipeline(str(clip), out, cfg, transcriber=PlanAwareTranscriber(out),
                         judge=FakeJudge([]), cache=cache, guidelines="G")
    assert third.final.stat().st_mtime_ns != rendered_at


# ------------------------------------------------------- audio-only runs

def _recording(path: Path) -> None:
    """The clip's tones as a bare WAV: nothing but audio to cook."""
    expr = "+".join(f"between(t\\,{a}\\,{b})" for a, b in TONES)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", f"aevalsrc='0.3*sin(440*2*PI*t)*({expr})+0.0005*random(0)':s=16000:d={DURATION}",
         "-c:a", "pcm_s16le", str(path)],
        check=True,
    )


def _streams(path: Path) -> list[dict]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,sample_rate",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"]


class SavedSource:
    """Replays one saved source response: valid for the source, never for an edited render."""

    name, version = "saved", "1"
    replays_fixed_source = True

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, wav_path: str) -> Transcript:
        self.calls += 1
        return Transcript(self.name, self.version, DURATION, list(SRC_WORDS))


def test_an_audio_only_run_ships_a_verified_wav_with_no_picture_checks(tmp_path: Path) -> None:
    wav = tmp_path / "rec.wav"
    _recording(wav)
    out = tmp_path / "out"
    result = run_pipeline(str(wav), out, _cfg(), transcriber=PlanAwareTranscriber(out),
                          judge=FakeJudge(_answers()), cache=Cache(tmp_path / "cache"),
                          guidelines="G", audio_only=True)
    assert result.final == out / "final.wav" and result.final_visuals is None
    (stream,) = _streams(result.final)
    assert (stream["codec_type"], stream["codec_name"], stream["sample_rate"]) == ("audio", "pcm_s16le", "48000")
    names = {c.name for c in result.checks}
    assert {"audio_stream", "duration", "loudness", "render_silence", "source_timing"} <= names
    assert not names & {"av_sync", "color", "resolution"}
    assert result.ok, [c.line() for c in result.checks if not c.ok]
    plan = Plan.from_json(result.plan_path.read_text())
    assert plan.output_duration == pytest.approx(result.facts["duration"], abs=0.3)
    assert not list(out.glob("final.wav.*"))


def test_an_audio_only_run_refuses_what_needs_a_picture(tmp_path: Path) -> None:
    wav = tmp_path / "rec.wav"
    _recording(wav)
    with pytest.raises(PipelineError, match="audio-only"):
        run_pipeline(str(wav), tmp_path / "out", _cfg(), transcriber=PlanAwareTranscriber(tmp_path / "out"),
                     judge=FakeJudge([]), cache=Cache(tmp_path / "cache"), guidelines="G",
                     audio_only=True, demo_minutes=1.0)


def test_a_saved_source_response_cannot_stand_in_for_output_asr(tmp_path: Path) -> None:
    wav = tmp_path / "rec.wav"
    _recording(wav)
    saved = SavedSource()
    with pytest.raises(PipelineError, match="--no-output-asr"):
        run_pipeline(str(wav), tmp_path / "out", _cfg(), transcriber=saved, judge=FakeJudge([]),
                     cache=Cache(tmp_path / "cache"), guidelines="G", audio_only=True)
    assert saved.calls == 0


def test_without_output_asr_the_report_says_the_transcript_is_only_projected(tmp_path: Path) -> None:
    wav = tmp_path / "rec.wav"
    _recording(wav)
    out = tmp_path / "out"
    saved = SavedSource()
    result = run_pipeline(str(wav), out, _cfg(), transcriber=saved, judge=FakeJudge(_answers()),
                          cache=Cache(tmp_path / "cache"), guidelines="G", audio_only=True,
                          verify_output_asr=False)
    assert saved.calls == 1, "only the source is transcribed"
    assert result.ok, [c.line() for c in result.checks if not c.ok]
    checks = {c.name: c for c in result.checks}
    assert "NOT transcribed" in checks["output_asr"].detail
    assert "not measured" in checks["cold_open"].detail
    assert result.facts["output_asr"] == "source-projection"
    plan = Plan.from_json(result.plan_path.read_text())
    kept = [w.text for w in SRC_WORDS if plan.source_to_output(w.start + 0.01) is not None]
    assert [w.text for w in result.facts["words"]] == kept
    assert "NOT an independent measurement" in result.report.read_text()


def test_a_changed_retake_policy_recomputes_the_retake_memo_not_replays_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav = tmp_path / "rec.wav"
    _recording(wav)
    out = tmp_path / "out"
    cache = Cache(tmp_path / "cache")
    run_pipeline(str(wav), out, _cfg(), transcriber=PlanAwareTranscriber(out), judge=FakeJudge(_answers()),
                 cache=cache, guidelines="G", audio_only=True)
    monkeypatch.setattr("talkinghead.pipeline._retake_policy", lambda: "a different retake policy")
    again = run_pipeline(str(wav), out, _cfg(), transcriber=PlanAwareTranscriber(out), judge=FakeJudge([]),
                         cache=cache, guidelines="G", audio_only=True)
    assert again.notes["memos"]["recomputed"] == "retakes"


def test_a_failed_demo_clip_cut_raises_a_runtime_error_and_leaves_no_file(tmp_path: Path) -> None:
    out = tmp_path / "clip.mp4"
    with pytest.raises(RuntimeError, match="ffmpeg failed cutting the demo clip"):
        make_clip(str(tmp_path / "missing.mov"), 1.0, out)
    assert not out.exists()
