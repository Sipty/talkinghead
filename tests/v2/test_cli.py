"""CLI surface: one command to run; adapters chosen by flag; the standalone
align step measures real files and remuxes only when asked."""

import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from talkinghead import __version__
from talkinghead.align import ANALYSIS_SR, AlignResult
from talkinghead.cli import build_judge, build_parser, build_transcriber, main, parse_time
from talkinghead.config import Config, load_config
from talkinghead.judge import ClaudeCodeJudge, FallbackJudge, ShellJudge, judge_model, prompt_sha
from talkinghead.timeline import Cut, Plan, SourceInfo
from talkinghead.transcribe import ParakeetTranscriber, ScribeTranscriber, WhisperTranscriber

SR = ANALYSIS_SR


def test_run_parser_defaults() -> None:
    args = build_parser().parse_args(["run", "v.mp4"])
    assert args.cmd == "run" and args.video == "v.mp4"
    assert args.demo is None and args.cold_open is None and args.outro is None
    assert args.judge is None and args.fresh is False
    assert args.audio is None and args.fix_drift is False and args.allow_bad_align is False


def test_run_parser_takes_the_external_recording() -> None:
    args = build_parser().parse_args(
        ["run", "v.mp4", "--audio", "rec.mov", "--fix-drift", "--allow-bad-align"])
    assert args.audio == "rec.mov" and args.fix_drift and args.allow_bad_align


def test_align_parser_defaults_and_flags() -> None:
    args = build_parser().parse_args(["align", "v.mov", "rec.mov"])
    assert args.cmd == "align" and (args.video, args.audio) == ("v.mov", "rec.mov")
    assert args.out is None and not args.json and not args.yes and not args.fix_drift
    args = build_parser().parse_args(
        ["align", "v.mov", "rec.mov", "-o", "m.mov", "--json", "-y", "--fix-drift"])
    assert args.out == "m.mov" and args.json and args.yes and args.fix_drift


def test_judge_factory() -> None:
    # The shell bridge's fallback is a second command, not a model: Claude Code
    # handles its own quota, so the claude spec never gets one.
    j = build_judge("claude", model="claude-fable-5-1", guidelines="G", effort=None,
                    fallback_command="omp -p --model openrouter/z-ai/glm-5.3-flash")
    assert isinstance(j, ClaudeCodeJudge) and j.model == "claude-fable-5-1"
    assert "G" in (j.system_prompt or "")
    s = build_judge("shell:omp -p --model slow", model=None, guidelines="G", effort=None)
    assert isinstance(s, ShellJudge) and s.cmd == "omp -p --model slow"
    with pytest.raises(ValueError):
        build_judge("magic", model="x", guidelines="G", effort=None)


def test_a_shell_judge_honours_its_model_and_the_fallback_command() -> None:
    primary = "omp -p --no-session --mode text --model openrouter/deepseek/deepseek-v4.1-flash"
    fallback = "omp -p --no-session --mode text --model openrouter/z-ai/glm-5.3-flash"
    judge = build_judge(f"shell:{primary}", model="openrouter/deepseek/deepseek-v4.1-flash",
                        guidelines="G", effort=None, fallback_command=fallback)
    assert isinstance(judge, FallbackJudge)
    assert isinstance(judge.primary, ShellJudge) and judge.primary.cmd == primary
    assert isinstance(judge.backup, ShellJudge) and judge.backup.model == "openrouter/z-ai/glm-5.3-flash"
    assert judge.name != judge.backup.name  # the cache key is the name: they must not replay each other
    # A shell judge takes its model from argv, so the override is written there.
    override = build_judge("shell:omp -p --model slow", model="fast", guidelines="G", effort=None)
    assert isinstance(override, ShellJudge) and override.cmd == "omp -p --model fast"
    # Without a fallback command the primary answers alone.
    assert isinstance(build_judge("shell:omp -p", model="m", guidelines="G", effort=None), ShellJudge)


def test_the_judge_and_the_transcriber_have_no_built_in_default() -> None:
    cfg = Config()
    assert cfg.judge is None and cfg.transcriber is None and cfg.model is None


def test_a_run_without_a_judge_or_a_transcriber_stops_at_once_and_names_the_variable(
        tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "in.mov"
    video.write_bytes(b"")
    monkeypatch.setattr("talkinghead.cli.run_pipeline", lambda *a, **k: pytest.fail("cooked"))
    assert main(["run", str(video), "--transcriber", "whisper"]) == 1
    assert "TALKINGHEAD_JUDGE" in capsys.readouterr().err
    monkeypatch.setenv("TALKINGHEAD_JUDGE", "shell:true")
    assert main(["run", str(video)]) == 1
    assert "TALKINGHEAD_TRANSCRIBER" in capsys.readouterr().err


def test_the_repo_root_talkinghead_toml_is_no_longer_loaded(tmp_path: Path, monkeypatch) -> None:
    from talkinghead import cli

    assert not (Path(cli.__file__).resolve().parents[2] / "talkinghead.toml").exists()
    assert cli._config_file(None) is None


OWNER_JUDGE_SPEC = "shell:omp -p --no-session --no-tools --mode text --model anthropic/claude-opus-5-5"


def test_the_owner_environment_keeps_the_judge_and_scribe_cache_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    # Values produced by the code before the environment contract, with the previous talkinghead.toml.
    # A change here re-keys every cached judge sample and transcript.
    monkeypatch.setenv("TALKINGHEAD_JUDGE", OWNER_JUDGE_SPEC)
    monkeypatch.setenv("TALKINGHEAD_TRANSCRIBER", "scribe")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    cfg = load_config()
    judge = build_judge(cfg.judge, model=judge_model(cfg.judge, cfg.model), guidelines="G", effort=cfg.effort,
                        timeout=cfg.judge_timeout, fallback_command=cfg.fallback_command)
    assert judge.name == "shell:omp:anthropic/claude-opus-5-5"
    assert prompt_sha("PROMPT", judge.name) == "c7f31151999031a3ebf6910e"
    transcriber = build_transcriber(cfg.transcriber)
    assert (transcriber.name, transcriber.version) == (
        "scribe", "scribe_v2;word;events;diarize=false;verbatim;parse=1")


def test_transcriber_factory_never_falls_back_silently(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("talkinghead.cli.find_fluidaudio", lambda: None)
    monkeypatch.setattr("sys.platform", "darwin")
    with pytest.raises(RuntimeError, match="fluidaudiocli"):
        build_transcriber("parakeet")
    monkeypatch.setattr("talkinghead.cli.find_fluidaudio", lambda: "/bin/fluidaudiocli")
    assert isinstance(build_transcriber("parakeet"), ParakeetTranscriber)
    assert isinstance(build_transcriber("whisper"), WhisperTranscriber)


def test_parakeet_is_refused_off_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("talkinghead.cli.find_fluidaudio", lambda: "/bin/fluidaudiocli")
    with pytest.raises(RuntimeError, match="macOS only"):
        build_transcriber("parakeet")


def test_main_reports_missing_video(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    rc = main(["run", str(tmp_path / "nope.mp4"), "--judge", "shell:true"])
    assert rc == 1
    assert "nope.mp4" in capsys.readouterr().err


def test_an_unreachable_judge_stops_before_any_work_with_exit_4(
        tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "in.mov"
    video.write_bytes(b"")
    fake = tmp_path / "omp"
    fake.write_text("#!/bin/sh\necho 'No API key found for openai-codex' >&2\nexit 1\n")
    fake.chmod(0o755)
    monkeypatch.setattr("talkinghead.cli.run_pipeline", lambda *a, **k: pytest.fail("ran without a judge"))
    rc = main(["run", str(video), "--out-dir", str(tmp_path / "run"), "--transcriber", "whisper", "--no-visuals",
               "--judge", f"shell:{fake} -p --mode text --model openai-codex/gpt-6.1-sol"])
    assert rc == 4
    err = capsys.readouterr().err
    assert "JUDGE UNAVAILABLE" in err and "No API key" in err and "omp login" in err


def test_a_judge_command_that_is_not_installed_stops_with_exit_4_and_names_it(
        tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "in.mov"
    video.write_bytes(b"")
    monkeypatch.setattr("talkinghead.cli.run_pipeline", lambda *a, **k: pytest.fail("ran without a judge"))
    rc = main(["run", str(video), "--out-dir", str(tmp_path / "run"), "--transcriber", "whisper", "--no-visuals",
               "--judge", "shell:no-such-judge-for-talkinghead"])
    assert rc == 4
    assert "judge command not found: no-such-judge-for-talkinghead" in capsys.readouterr().err


def test_feedback_parser_and_time_parsing() -> None:
    args = build_parser().parse_args(["feedback", "out", "clipped word", "--at", "7:03.5", "--kind", "defect"])
    assert args.cmd == "feedback" and args.note == "clipped word" and args.kind == "defect"
    assert parse_time(args.at) == 423.5 and parse_time("12.25") == 12.25 and parse_time("1:02:03") == 3723.0


# ------------------------------------------------------------------- align


def speech_signal(duration_s: float, seed: int = 0) -> np.ndarray:
    """Broadband noise gated by random pauses: non-periodic, so a cross
    correlation of it has one unambiguous peak."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * SR)
    env = np.zeros(n)
    pos = 0
    while pos < n:
        on = int(rng.uniform(1.0, 4.0) * SR)
        env[pos: pos + on] = 1.0
        pos += on + int(rng.uniform(0.3, 1.5) * SR)
    return env * rng.standard_normal(n) * 0.3


def _mux(path: Path, video_src: str, samples: np.ndarray, duration: float) -> None:
    pcm = (np.clip(samples[: int(duration * SR)], -1, 1) * 32767).astype(np.int16).tobytes()
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", video_src,
         "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-t", f"{duration}", str(path)],
        input=pcm, check=True,
    )


def _pair(tmp_path: Path, video_len: float, audio_len: float, offset: float) -> tuple[Path, Path]:
    """(master, external): the master heard the speech from its t=0, the
    external recording joined `offset` seconds later."""
    speech = speech_signal(max(video_len, offset + audio_len) + 2.0, seed=5)
    video = tmp_path / "master.mp4"
    _mux(video, f"testsrc=duration={video_len}:size=320x240:rate=30", speech, video_len)
    audio = tmp_path / "external.mp4"
    _mux(audio, f"color=black:duration={audio_len}:size=320x240:rate=30",
         speech[int(offset * SR):], audio_len)
    return video, audio


def test_align_json_reports_the_measurement_and_writes_the_master(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    video, audio = _pair(tmp_path, video_len=24.0, audio_len=18.0, offset=4.0)
    out = tmp_path / "aligned.mov"
    rc = main(["align", str(video), str(audio), "-o", str(out), "--yes", "--json"])
    assert rc == 0 and out.is_file()
    printed = capsys.readouterr()
    payload = json.loads(printed.out)  # JSON only on stdout; the report and progress go to stderr
    assert "match confidence:" in printed.err
    assert payload["offset_sec"] == pytest.approx(4.0, abs=0.05)
    assert payload["confidence"] is not None and payload["confidence"] > 3.0
    assert abs(payload["drift_ppm"]) < 30.0
    assert payload["output"] == str(out)
    assert payload["video_duration"] == pytest.approx(24.0, abs=0.2)
    assert payload["audio_duration"] == pytest.approx(18.0, abs=0.2)
    assert payload["windows_used"] <= payload["windows_total"]


def test_align_reports_a_weak_measurement_and_remuxes_only_on_confirmation(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    video, audio = tmp_path / "v.mov", tmp_path / "a.mov"
    video.write_bytes(b"")  # only existence matters: the measurement is stubbed
    audio.write_bytes(b"")
    measured = AlignResult(offset_sec=1.25, drift_ppm=64.0, confidence=1.5,
                           video_duration=600.0, audio_duration=900.0, windows_used=0, windows_total=7)
    seen: dict = {}
    monkeypatch.setattr("talkinghead.cli.analyze", lambda *a, **k: (seen.update(k), measured)[1])
    monkeypatch.setattr("talkinghead.cli.align", lambda *a, **k: pytest.fail("remuxed without --yes"))
    monkeypatch.setattr("sys.stdin", io.StringIO("n\n"))
    out = tmp_path / "aligned.mov"

    rc = main(["align", str(video), str(audio), "-o", str(out), "--fix-drift"])
    assert rc == 1 and not out.exists() and seen["fix_drift"] is True
    printed = capsys.readouterr()
    report = printed.out
    assert "+1.250 s" in report and "+64.0 ppm" in report and "1.50x above alternatives" in report
    assert "below 3.0x" in report and "--fix-drift" in report
    assert "Remux now?" in printed.err and "Aborted" in printed.err and f"output: {out}" in printed.err


# --- Scribe, saved responses, audio-only ---

def _tone_wav(path: Path) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "sine=frequency=440:sample_rate=16000:duration=3", "-c:a", "pcm_s16le", str(path)], check=True)


def test_the_scribe_factory_is_live_with_a_key_and_names_what_is_missing(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    assert isinstance(build_transcriber("scribe"), ScribeTranscriber)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        build_transcriber("scribe")
    with pytest.raises(ValueError, match="both"):
        build_transcriber("scribe", scribe_response=tmp_path / "r.json")
    with pytest.raises(ValueError, match="--transcriber scribe"):
        build_transcriber("whisper", scribe_response=tmp_path / "r.json", scribe_audio=tmp_path / "a.wav")


@pytest.mark.parametrize("flags, message", [
    (["--scribe-response", "r.json"], "go together"),
    (["--scribe-response", "r.json", "--scribe-audio", "a.wav"], "--no-output-asr"),
    (["--demo", "1"], "drop --audio/--demo"),
])
def test_unsound_audio_only_and_saved_source_runs_stop_before_the_judge_or_any_cook(
        tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch,
        flags: list[str], message: str) -> None:
    wav = tmp_path / "in.wav"
    _tone_wav(wav)
    monkeypatch.setattr("talkinghead.cli.preflight", lambda judge: pytest.fail("reached the judge"))
    monkeypatch.setattr("talkinghead.cli.run_pipeline", lambda *a, **k: pytest.fail("cooked"))
    rc = main(["run", str(wav), "--audio-only", "--judge", "shell:true", *flags])
    assert rc == 1
    assert message in capsys.readouterr().err


def test_a_corrupt_saved_response_fails_the_run_before_it_cooks(
        tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    wav = tmp_path / "in.wav"
    _tone_wav(wav)
    response = tmp_path / "response.json"
    response.write_text("{ not json")
    monkeypatch.setattr("talkinghead.cli.preflight", lambda judge: None)
    monkeypatch.setattr("talkinghead.cli.run_pipeline", lambda *a, **k: pytest.fail("cooked"))
    rc = main(["run", str(wav), "--audio-only", "--judge", "shell:true", "--no-output-asr",
               "--scribe-response", str(response), "--scribe-audio", str(wav)])
    assert rc == 1 and capsys.readouterr().err.startswith("talkinghead:")


def test_render_rebuilds_an_audio_plan_only_when_told_it_is_audio(
        tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    wav = tmp_path / "in.wav"
    _tone_wav(wav)
    plan = Plan.from_cuts(SourceInfo("main", str(wav), "x", 3.0), [Cut(1.0, 1.5, "silence")])
    plan.provenance = {"audio_only": True}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.to_json())
    out = tmp_path / "final.wav"
    assert main(["render", str(plan_path), "--out", str(out)]) == 1
    assert "--audio-only" in capsys.readouterr().err and not out.exists()
    assert main(["render", str(plan_path), "--out", str(out), "--audio-only"]) == 0
    probed = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,sample_rate:format=duration",
         "-of", "json", str(out)], capture_output=True, text=True, check=True).stdout)
    assert probed["streams"][0]["codec_name"] == "pcm_s16le"
    assert float(probed["format"]["duration"]) == pytest.approx(plan.output_duration, abs=0.01)


@pytest.mark.parametrize("tool", [
    "rsvg-convert",
    pytest.param("swiftc", marks=pytest.mark.skipif(sys.platform != "darwin", reason="swiftc is needed on macOS only")),
])
def test_visuals_need_their_tools_and_fail_before_any_work(
        tool: str, tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    real_which = shutil.which
    monkeypatch.setattr("talkinghead.cli.shutil.which",
                        lambda name, *a, **k: None if name == tool else real_which(name, *a, **k))
    monkeypatch.setenv("TALKINGHEAD_JUDGE", "shell:true")
    video = tmp_path / "in.mov"
    video.write_bytes(b"")
    monkeypatch.setattr("talkinghead.cli.preflight", lambda judge: pytest.fail("judged before the tool check"))
    monkeypatch.setattr("talkinghead.cli.run_pipeline", lambda *a, **k: pytest.fail("cooked"))
    assert main(["run", str(video), "--transcriber", "whisper"]) == 1
    err = capsys.readouterr().err
    assert tool in err and "--no-visuals" in err


def test_bundled_guidelines_missing_from_an_install_say_to_run_from_a_clone(
        tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "in.mov"
    video.write_bytes(b"")
    monkeypatch.setenv("TALKINGHEAD_JUDGE", "shell:true")
    monkeypatch.setattr("talkinghead.cli._repo_file", lambda rel: tmp_path / "not-installed" / rel)
    monkeypatch.setattr("talkinghead.cli.run_pipeline", lambda *a, **k: pytest.fail("cooked"))
    assert main(["run", str(video), "--transcriber", "whisper", "--no-visuals"]) == 1
    assert "run talkinghead from a clone" in capsys.readouterr().err


def test_ctrl_c_exits_130_and_says_a_rerun_resumes(
        tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "in.mov"
    video.write_bytes(b"")
    monkeypatch.setenv("TALKINGHEAD_JUDGE", "shell:true")
    monkeypatch.setattr("talkinghead.cli.preflight", lambda judge: None)

    def interrupted(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr("talkinghead.cli.run_pipeline", interrupted)
    rc = main(["run", str(video), "--out-dir", str(tmp_path / "run"), "--transcriber", "whisper", "--no-visuals"])
    assert rc == 130 and "resume" in capsys.readouterr().err


def test_a_failed_external_tool_is_a_one_line_error_not_a_traceback(
        tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "in.mov"
    video.write_bytes(b"")
    monkeypatch.setenv("TALKINGHEAD_JUDGE", "shell:true")
    monkeypatch.setattr("talkinghead.cli.preflight", lambda judge: None)

    def failing(*a, **k):
        raise subprocess.CalledProcessError(2, ["ffmpeg", "-i", "in.mov"])

    monkeypatch.setattr("talkinghead.cli.run_pipeline", failing)
    rc = main(["run", str(video), "--out-dir", str(tmp_path / "run"), "--transcriber", "whisper", "--no-visuals"])
    assert rc == 1 and "tool failed (exit 2): ffmpeg -i in.mov" in capsys.readouterr().err


def test_version_flag_and_help_mark_the_experimental_commands(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as done:
        main(["--version"])
    assert done.value.code == 0 and f"talkinghead {__version__}" in capsys.readouterr().out
    text = build_parser().format_help()
    assert "board-page" in text and "(experimental)" in text


def test_the_cli_does_not_import_the_apple_only_whisper_package() -> None:
    code = ("import sys, talkinghead.cli as cli; cli.build_parser().format_help(); "
            "assert 'mlx_whisper' not in sys.modules and 'mlx' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_render_refines_joins_only_when_asked() -> None:
    parser = build_parser()
    assert parser.parse_args(["render", "plan.json", "--out", "out.mp4"]).refine_joins is False
    assert parser.parse_args(["render", "plan.json", "--out", "out.mp4", "--refine-joins"]).refine_joins is True
