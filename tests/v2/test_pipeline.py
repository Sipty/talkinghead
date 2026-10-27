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
    # Visuals and polish off: these tests pin the CUT pipeline's contract (exact judge
    # call counts, byte-identical plans). The overlay pass and the polish passes have their own
    # tests, and their own judge calls would make these counts meaningless.
    return Config(crf=30, preset="ultrafast", votes=5, agree=4, visuals_enabled=False, polish=False)


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



def _visuals_assets(tmp_path: Path) -> Path:
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "manifest.toml").write_text("")
    return assets


def _visuals_judge():
    """A judge that answers the editorial pass with one quote card on "Second." and the cut's rounds in order."""
    from talkinghead.judge import AUDIO_READING, validate

    class VisualsJudge:
        name = "vis"

        def __init__(self) -> None:
            self.seq = list(_answers())
            self.calls = 0

        def decide(self, prompt: str, schema: dict) -> dict:
            if schema.get("title") == "AudioAssessment":
                return validate(AUDIO_READING, schema)
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

    return VisualsJudge()


def _visuals_cfg() -> Config:
    return Config(crf=30, preset="ultrafast", votes=5, agree=4,
                  visuals_enabled=True, visuals_research=False, visuals_min_hold=1.0,
                  plan_enabled=True, plan_hold=1.0, plan_open_at=0.0, plan_open_hold=1.0,
                  visuals_min_gap=0.4, polish=False)


def test_visuals_pass_ships_one_picture_encode_with_the_overlays_inside_it(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`final-visuals.mp4` is the only picture written: one video encode, no `final.mp4`, and the picture table
    (audio lock, sync, resolution) plus the contact sheet pass on it."""
    import talkinghead.pipeline as pl

    order: list[str] = []
    _stub_head_clip(monkeypatch, order)
    calls: list[list[str]] = []
    real_run = subprocess.run

    def spy(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd and cmd[0] == "ffmpeg":
            calls.append([str(c) for c in cmd])
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy)
    rendered: list[dict] = []
    real_render = pl.render_plan
    monkeypatch.setattr(pl, "render_plan", lambda *a, **k: rendered.append(k) or real_render(*a, **k))
    assets = _visuals_assets(tmp_path)
    out = tmp_path / "out"
    result = run_pipeline(str(clip), out, _visuals_cfg(), transcriber=PlanAwareTranscriber(out),
                          judge=_visuals_judge(), cache=Cache(tmp_path / "cache"),
                          guidelines="G", topic="t", assets_dir=assets)
    assert result.final == out / "final-visuals.mp4" and result.final_visuals == result.final
    assert result.final.is_file() and not (out / "final.mp4").exists()
    assert len(rendered) == 1 and rendered[0]["overlays"] is not None and rendered[0]["overlays"].placements
    # One-frame size probes carry -c:v too; a picture encode is the ffmpeg call that writes more than one frame.
    encodes = [c for c in calls if "-c:v" in c and "-frames:v" not in c]
    assert len(encodes) == 1 and "overlay=" in encodes[0][encodes[0].index("-filter_complex") + 1]
    report = result.report.read_text()
    assert "## Asset decisions" in report
    assert "quote_card" in report
    assert "## Plan card" in report
    names = {c.name for c in result.checks}
    assert {"audio_locked", "av_sync", "resolution", "visuals_frames"} <= names
    assert {n for n in names if n.startswith("visuals_")} == {"visuals_frames", "visuals_head_clip"}
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


# ------------------------------------------------- audio assessment overlap

def _stub_assessment(monkeypatch: pytest.MonkeyPatch, fn) -> None:
    monkeypatch.setattr("talkinghead.pipeline.audio_assess.assess", fn)


def _live_extra_threads() -> list:
    import threading

    return [t for t in threading.enumerate() if t is not threading.main_thread() and not t.daemon]


def test_a_stateless_judges_assessment_alongside_the_editorial_stages_overlaps_them(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    import talkinghead.pipeline as pl

    structure_started = threading.Event()
    seen: dict = {}

    def assess(*a, **k):
        seen["thread"] = threading.current_thread()
        # Inline this would wait for a stage that cannot start until it returns.
        seen["overlapped"] = structure_started.wait(10)
        return {"skipped": "stub reading"}

    real = pl.structure.run

    def structure_run(ctx):
        structure_started.set()
        return real(ctx)

    _stub_assessment(monkeypatch, assess)
    monkeypatch.setattr(pl.structure, "run", structure_run)
    monkeypatch.setattr(pl.restructure, "concurrent_safe", lambda judge: True)
    out = tmp_path / "out"
    result = run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out), judge=FakeJudge(_answers()),
                          cache=Cache(tmp_path / "cache"), guidelines="G", topic="t")
    assert seen["overlapped"] is True and seen["thread"] is not threading.main_thread()
    assert result.ok and result.notes["audio_assessment"] == {"skipped": "stub reading"}
    assert _live_extra_threads() == []


def test_a_stateful_judges_assessment_inline_runs_before_the_first_render(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    import talkinghead.pipeline as pl

    order: list = []

    def assess(*a, **k):
        order.append(("assess", threading.current_thread() is threading.main_thread()))
        return {"skipped": "stub reading"}

    real_render = pl.render_plan
    _stub_assessment(monkeypatch, assess)
    monkeypatch.setattr(pl, "render_plan", lambda *a, **k: order.append(("render", True)) or real_render(*a, **k))
    out = tmp_path / "out"
    run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out), judge=FakeJudge(_answers()),
                 cache=Cache(tmp_path / "cache"), guidelines="G", topic="t")
    assert order[0] == ("assess", True) and ("render", True) in order[1:]


def test_the_assessment_joined_before_run_pipeline_raises_when_the_editorial_stages_fail(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    import talkinghead.pipeline as pl

    release, finished = threading.Event(), threading.Event()

    def assess(*a, **k):
        release.wait(10)
        finished.set()
        return {"skipped": "stub reading"}

    def boom(ctx):
        # The assessment is still blocked when the stage fails; it is released a moment later.
        threading.Timer(0.3, release.set).start()
        raise RuntimeError("stage died")

    _stub_assessment(monkeypatch, assess)
    monkeypatch.setattr(pl.structure, "run", boom)
    monkeypatch.setattr(pl.restructure, "concurrent_safe", lambda judge: True)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="stage died"):
        run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out), judge=FakeJudge(_answers()),
                     cache=Cache(tmp_path / "cache"), guidelines="G", topic="t")
    assert finished.is_set()
    assert _live_extra_threads() == []


def test_an_exception_raised_inside_the_assessment_surfaces_from_run_pipeline_before_rendering(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import talkinghead.pipeline as pl

    class AssessmentDied(Exception):
        pass

    def assess(*a, **k):
        raise AssessmentDied("assessment died")

    rendered: list[bool] = []
    real_render = pl.render_plan
    _stub_assessment(monkeypatch, assess)
    monkeypatch.setattr(pl, "render_plan", lambda *a, **k: rendered.append(True) or real_render(*a, **k))
    monkeypatch.setattr(pl.restructure, "concurrent_safe", lambda judge: True)
    out = tmp_path / "out"
    with pytest.raises(AssessmentDied, match="assessment died"):
        run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out), judge=FakeJudge(_answers()),
                     cache=Cache(tmp_path / "cache"), guidelines="G", topic="t")
    assert rendered == []
    assert _live_extra_threads() == []


def test_a_plan_only_rebuild_replays_the_stored_assessment_without_calling_assess(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import talkinghead.pipeline as pl

    out, cache = tmp_path / "out", Cache(tmp_path / "cache")
    _stub_assessment(monkeypatch, lambda *a, **k: {"skipped": "stub reading"})
    monkeypatch.setattr(pl.restructure, "concurrent_safe", lambda judge: True)
    run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out), judge=FakeJudge(_answers()),
                 cache=cache, guidelines="G", topic="t")

    def refuse(*a, **k):
        raise AssertionError("plan-only must not assess")

    _stub_assessment(monkeypatch, refuse)
    again = run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out), judge=FakeJudge([]),
                         cache=cache, guidelines="G", topic="t", plan_only=True)
    assert again.notes["audio_assessment"] == {"skipped": "stub reading"}
    assert _live_extra_threads() == []


# ------------------------------------------------- audio master first, picture once

def _counted(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    """Count the pipeline's renders: the picture (`render_plan`) and every audio render (`render_audio_plan`)."""
    import talkinghead.pipeline as pl

    calls: dict[str, list] = {"picture": [], "audio": []}
    real_picture, real_audio = pl.render_plan, pl.render_audio_plan
    monkeypatch.setattr(pl, "render_plan", lambda *a, **k: calls["picture"].append(k) or real_picture(*a, **k))
    monkeypatch.setattr(pl, "render_audio_plan", lambda *a, **k: calls["audio"].append(a[1]) or real_audio(*a, **k))
    return calls


def _cook(clip: Path, tmp_path: Path, cfg: Config | None = None, judge=None):
    out = tmp_path / "out"
    return run_pipeline(str(clip), out, cfg or _cfg(), transcriber=PlanAwareTranscriber(out),
                        judge=judge or FakeJudge(_answers()), cache=Cache(tmp_path / "cache"), guidelines="G")


def test_the_video_carries_the_audio_master_bytes_and_the_report_lists_audio_locked(
    clip: Path, tmp_path: Path
) -> None:
    from talkinghead.lock import audio_stream_md5

    result = _cook(clip, tmp_path)
    master = tmp_path / "out" / "audio-master.m4a"
    assert result.audio_master == master and master.is_file()
    assert audio_stream_md5(result.final) == audio_stream_md5(master)
    checks = {c.name: c for c in result.checks}
    assert checks["audio_locked"].ok and "audio_stream" in checks and "av_sync" in checks
    assert "audio_locked: OK" in result.report.read_text()
    memo = json.loads((tmp_path / "out" / "audio-render.json").read_text())
    assert set(memo) == {"key", "cmd", "sha256"}
    facts = json.loads((tmp_path / "out" / "verify" / "facts.json").read_text())
    assert facts["picture"]["audio_lock"]["locked"] == facts["picture"]["audio_lock"]["rendered"]
    assert not {"mode", "color", "frame", "video_duration"} & set(facts)


def test_a_fatal_audio_check_stops_the_cook_before_any_picture_is_rendered(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import talkinghead.pipeline as pl
    from talkinghead.verify import CheckResult

    real = pl.run_checks
    monkeypatch.setattr(pl, "run_checks", lambda plan, facts: [*real(plan, facts),
                                                              CheckResult("loudness", "fatal", False, "forced")])
    calls = _counted(monkeypatch)
    with pytest.raises(PipelineError, match="loudness"):
        _cook(clip, tmp_path)
    out = tmp_path / "out"
    assert calls["picture"] == [] and calls["audio"]
    assert not (out / "final.mp4").exists() and (out / "audio-master.m4a").is_file()
    assert "FAILED" in (out / "report.md").read_text()


def test_changing_crf_renders_only_the_picture(clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _cook(clip, tmp_path)
    calls = _counted(monkeypatch)
    cfg = _cfg()
    cfg.crf = 31
    strict = FakeJudge([])
    result = _cook(clip, tmp_path, cfg, strict)
    assert len(calls["picture"]) == 1 and calls["audio"] == [] and strict.calls == 0
    assert result.ok, [c.line() for c in result.checks if not c.ok]


def test_an_overwritten_audio_master_is_rendered_again_not_replayed(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from talkinghead.lock import audio_stream_md5

    _cook(clip, tmp_path)
    master = tmp_path / "out" / "audio-master.m4a"
    master.write_bytes(b"not an audio master")
    calls = _counted(monkeypatch)
    result = _cook(clip, tmp_path, judge=FakeJudge([]))
    assert calls["audio"] == [str(master)]
    assert result.ok, [c.line() for c in result.checks if not c.ok]
    assert audio_stream_md5(result.final) == audio_stream_md5(master)


def test_a_full_cook_locks_the_plan_and_the_master_it_rendered(clip: Path, tmp_path: Path) -> None:
    from talkinghead.lock import audio_stream_md5, plan_digest
    from talkinghead.store import sha256_file

    result = _cook(clip, tmp_path)
    out = tmp_path / "out"
    lock = json.loads((out / "edit-lock.json").read_text())
    assert result.lock == out / "edit-lock.json"
    assert lock["plan_sha"] == plan_digest(Plan.from_json((out / "plan.json").read_text()))
    assert lock["audio"]["sha256"] == sha256_file(out / "audio-master.m4a")
    assert lock["audio"]["stream_md5"] == audio_stream_md5(result.final)
    transcript = out / "verify" / "locked-transcript.json"
    assert lock["transcript"]["sha256"] == sha256_file(transcript)
    assert json.loads(transcript.read_text())["words"]
    assert result.notes["lock"]["plan_sha"] == lock["plan_sha"]
    assert "## Edit lock" in result.report.read_text()


def test_stop_at_lock_leaves_the_master_and_the_lock_and_the_next_run_renders_only_the_picture(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    cache = Cache(tmp_path / "cache")
    calls = _counted(monkeypatch)
    stopped = run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out),
                           judge=FakeJudge(_answers()), cache=cache, guidelines="G", stop_at_lock=True)
    assert calls["picture"] == [] and calls["audio"]
    assert (out / "audio-master.m4a").is_file() and (out / "verify" / "locked-transcript.json").is_file()
    assert stopped.lock == out / "edit-lock.json" and stopped.lock.is_file()
    assert not (out / "final.mp4").exists() and not stopped.final.exists()
    report = stopped.report.read_text()
    assert "**AUDIO LOCKED**" in report and "## Edit lock" in report and stopped.ok
    lock_bytes = stopped.lock.read_bytes()

    calls["audio"].clear()
    done = run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out), judge=FakeJudge([]),
                        cache=cache, guidelines="G")
    assert len(calls["picture"]) == 1 and calls["audio"] == []
    assert done.final.is_file() and done.ok, [c.line() for c in done.checks if not c.ok]
    assert done.lock.read_bytes() == lock_bytes
    assert "**OK**" in done.report.read_text()


def test_an_edit_after_the_lock_is_refused_only_while_the_lock_is_held(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from talkinghead.lock import plan_digest

    out = tmp_path / "out"
    cache = Cache(tmp_path / "cache")
    calls = _counted(monkeypatch)

    def run(**extra):
        return run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out),
                            judge=FakeJudge(_answers()), cache=cache, guidelines="G", **extra)

    run(stop_at_lock=True)  # held for the owner's listen
    lock_bytes = (out / "edit-lock.json").read_bytes()
    old_sha = json.loads(lock_bytes)["plan_sha"]
    assert (out / "edit-lock.hold").is_file()

    with pytest.raises(PipelineError, match="--stop-at-lock") as err:
        run(decisions={"cold_open": 3.0})
    assert str(out / "edit-lock.json") in str(err.value) and "edit-lock.hold" in str(err.value)
    assert calls["picture"] == [] and not (out / "final.mp4").exists()
    assert (out / "edit-lock.json").read_bytes() == lock_bytes and (out / "edit-lock.hold").is_file()
    assert "FAILED" in (out / "report.md").read_text()

    relocked = run(decisions={"cold_open": 3.0}, stop_at_lock=True)
    new = json.loads((out / "edit-lock.json").read_text())
    assert new["plan_sha"] != old_sha and new["plan_sha"] == plan_digest(Plan.from_json(relocked.plan_path.read_text()))
    assert relocked.notes["lock"]["relocked_from"] == old_sha
    assert json.loads((out / "edit-lock.hold").read_text())["plan_sha"] == new["plan_sha"]
    assert calls["picture"] == [] and not (out / "final.mp4").exists()


def test_an_edit_after_the_lock_relocks_when_it_is_unheld_and_renders_the_picture(clip: Path, tmp_path: Path) -> None:
    from talkinghead.lock import plan_digest

    out = tmp_path / "out"
    cache = Cache(tmp_path / "cache")
    _cook(clip, tmp_path)
    old_sha = json.loads((out / "edit-lock.json").read_text())["plan_sha"]
    assert not (out / "edit-lock.hold").exists()
    result = run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out),
                          judge=FakeJudge(_answers()), cache=cache, guidelines="G", decisions={"cold_open": 3.0})
    new = json.loads((out / "edit-lock.json").read_text())
    assert new["plan_sha"] != old_sha and new["plan_sha"] == plan_digest(Plan.from_json(result.plan_path.read_text()))
    assert result.notes["lock"]["relocked_from"] == old_sha
    assert result.final.is_file()


def test_the_hold_released_after_the_full_run_renders_the_held_picture(clip: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    cache = Cache(tmp_path / "cache")
    run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out), judge=FakeJudge(_answers()),
                 cache=cache, guidelines="G", stop_at_lock=True)
    assert (out / "edit-lock.hold").is_file()
    done = run_pipeline(str(clip), out, _cfg(), transcriber=PlanAwareTranscriber(out), judge=FakeJudge([]),
                        cache=cache, guidelines="G")
    assert done.final.is_file() and not (out / "edit-lock.hold").exists()


def _stub_head_clip(monkeypatch: pytest.MonkeyPatch, order: list[str], *, fatal: bool = False) -> list[dict]:
    """Replace the pipeline's head-clip gate with a recorded verdict, so the tests need no detector."""
    import talkinghead.pipeline as pl
    from talkinghead.verify import CheckResult

    seen: list[dict] = []

    def fake(plan, placements, **kwargs):
        order.append("head_clip")
        seen.append({"placements": placements, **kwargs})
        if fatal:
            return (CheckResult("visuals_head_clip", "fatal", False, "overlay covers the speaker: stub"),
                    {"frames": "source", "placements": []})
        return CheckResult("visuals_head_clip", "fatal", True, "stub: clear"), {"frames": "source", "placements": []}

    monkeypatch.setattr(pl, "check_head_clip_plan", fake)
    return seen


def _record_order(monkeypatch: pytest.MonkeyPatch, order: list[str], words: list | None = None) -> None:
    """Record when `plan_visuals` (and the words it is handed) and `render_plan` are called."""
    import talkinghead.pipeline as pl

    real_plan, real_render = pl.plan_visuals, pl.render_plan

    def plan(ctx, plan_, **kwargs):
        order.append("plan_visuals")
        if words is not None:
            words.extend(kwargs["words"])
        return real_plan(ctx, plan_, **kwargs)

    monkeypatch.setattr(pl, "plan_visuals", plan)
    monkeypatch.setattr(pl, "render_plan", lambda *a, **k: order.append("render_plan") or real_render(*a, **k))


def test_visuals_before_the_picture_are_planned_on_the_locked_words(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    words: list = []
    _record_order(monkeypatch, order, words)
    seen = _stub_head_clip(monkeypatch, order)
    out = tmp_path / "out"
    result = run_pipeline(str(clip), out, _visuals_cfg(), transcriber=PlanAwareTranscriber(out),
                          judge=_visuals_judge(), cache=Cache(tmp_path / "cache"), guidelines="G", topic="t",
                          assets_dir=_visuals_assets(tmp_path))
    assert order == ["plan_visuals", "head_clip", "render_plan"]  # one picture encode, after the gate
    locked = Transcript.from_dict(json.loads((out / "verify" / "locked-transcript.json").read_text()))
    assert words and words == locked.words
    assert seen[0]["placements"] and seen[0]["settings"].fps and seen[0]["work_dir"] == out / "verify" / "visuals"
    assert result.final_visuals is not None and result.final_visuals.is_file()
    assert result.notes["visuals"]["head_clip"]["frames"] == "source"


def test_stop_at_lock_plans_visuals_and_checks_head_clip_and_the_next_run_replays_them(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    _record_order(monkeypatch, order)
    _stub_head_clip(monkeypatch, order)
    out = tmp_path / "out"
    cache = Cache(tmp_path / "cache")
    assets = _visuals_assets(tmp_path)
    stopped = run_pipeline(str(clip), out, _visuals_cfg(), transcriber=PlanAwareTranscriber(out),
                           judge=_visuals_judge(), cache=cache, guidelines="G", topic="t", assets_dir=assets,
                           stop_at_lock=True)
    assert order == ["plan_visuals", "head_clip"]
    assert (out / "stages" / "visuals.json").is_file() and not (out / "final.mp4").exists()
    assert stopped.final_visuals is None and "head_clip" in stopped.notes["visuals"]
    report = stopped.report.read_text()
    assert "**AUDIO LOCKED**" in report and "visuals_head_clip" in report and "## Asset decisions" in report

    order.clear()
    replay = _visuals_judge()
    done = run_pipeline(str(clip), out, _visuals_cfg(), transcriber=PlanAwareTranscriber(out), judge=replay,
                        cache=cache, guidelines="G", topic="t", assets_dir=assets)
    # The visuals memo replays (no judge call, no `plan_visuals`); head-clip is re-run, then the picture renders once.
    assert order == ["head_clip", "render_plan"] and replay.calls == 0
    assert done.final.is_file() and done.final_visuals is not None and done.ok


def test_a_fatal_head_clip_stops_the_cook_before_any_picture(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    _record_order(monkeypatch, order)
    _stub_head_clip(monkeypatch, order, fatal=True)
    out = tmp_path / "out"
    with pytest.raises(PipelineError, match="visuals_head_clip"):
        run_pipeline(str(clip), out, _visuals_cfg(), transcriber=PlanAwareTranscriber(out),
                     judge=_visuals_judge(), cache=Cache(tmp_path / "cache"), guidelines="G", topic="t",
                     assets_dir=_visuals_assets(tmp_path))
    assert order == ["plan_visuals", "head_clip"]
    assert not (out / "final.mp4").exists() and not (out / "final-visuals.mp4").exists()
    report = (out / "report.md").read_text()
    assert "FAILED" in report and "visuals_head_clip" in report


def test_a_changed_overlay_setting_re_renders_the_picture_only_and_an_unchanged_one_replays_it_overlay_settings_memo(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import talkinghead.pipeline as pl

    order: list[str] = []
    _record_order(monkeypatch, order)
    _stub_head_clip(monkeypatch, order)
    audio_renders: list[str] = []
    real_audio = pl.render_audio_plan
    monkeypatch.setattr(pl, "render_audio_plan", lambda *a, **k: audio_renders.append("audio") or real_audio(*a, **k))
    out = tmp_path / "out"
    cache = Cache(tmp_path / "cache")
    assets = _visuals_assets(tmp_path)

    def cook(cfg: Config, judge):
        return run_pipeline(str(clip), out, cfg, transcriber=PlanAwareTranscriber(out), judge=judge, cache=cache,
                            guidelines="G", topic="t", assets_dir=assets)

    first = cook(_visuals_cfg(), _visuals_judge())
    assert first.final == out / "final-visuals.mp4" and first.final_visuals == first.final
    assert order.count("render_plan") == 1
    rendered_at = first.final.stat().st_mtime_ns

    order.clear()
    audio_renders.clear()
    same = cook(_visuals_cfg(), _visuals_judge())
    assert "render_plan" not in order and same.final.stat().st_mtime_ns == rendered_at

    order.clear()
    audio_renders.clear()
    moved_cfg = _visuals_cfg()
    moved_cfg.visuals_corner = "top-left"
    judge = _visuals_judge()
    moved = cook(moved_cfg, judge)
    assert order.count("render_plan") == 1 and audio_renders == [] and judge.calls == 0
    assert moved.final.stat().st_mtime_ns != rendered_at
