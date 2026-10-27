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


def test_the_cli_flags_decide_a_cooks_isolation_over_the_config_file(tmp_path: Path) -> None:
    config = tmp_path / "talkinghead.toml"
    config.write_text("isolate_voice = true\n")

    def cook_isolates(*flags: str) -> bool:
        args = build_parser().parse_args(["run", "v.mp4", *flags])
        return load_config(config, isolate_voice=args.isolate_voice).isolate_voice

    assert cook_isolates() is True                 # the config file enables it and no flag names it
    assert cook_isolates("--no-isolate-voice") is False
    assert cook_isolates("--isolate-voice") is True


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


def _video_plan(tmp_path: Path, cuts: list[Cut], name: str = "plan.json") -> tuple[Plan, Path]:
    """A 6 s noise-and-testsrc recording and a hand-edited plan over it."""
    src = tmp_path / "in.mp4"
    if not src.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=6",
             "-f", "lavfi", "-i", "anoisesrc=d=6:r=48000:seed=3", "-c:v", "libx264", "-preset", "ultrafast",
             "-pix_fmt", "yuv420p", "-c:a", "aac", str(src)], check=True)
    plan = Plan.from_cuts(SourceInfo("main", str(src), "x", 6.0), cuts)
    path = tmp_path / name
    path.write_text(plan.to_json())
    return plan, path


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "talkinghead.toml"
    path.write_text("fps = 30\n")
    return path


def _master(plan_path: Path, out: Path, cfg: Path) -> None:
    assert main(["render", str(plan_path), "--audio-only", "--out", str(out), "--config", str(cfg)]) == 0


def test_a_video_plan_audio_only_render_is_the_pictures_audio_master(tmp_path: Path) -> None:
    from talkinghead.cli import render_settings_for
    from talkinghead.lock import audio_stream_md5
    from talkinghead.render import RenderSettings, render_audio_plan

    plan, plan_path = _video_plan(tmp_path, [Cut(1.0, 1.4, "silence")])
    cfg = _config(tmp_path)
    out = tmp_path / "audio-master.m4a"
    _master(plan_path, out, cfg)
    settings = render_settings_for(plan, load_config(cfg))
    assert settings.fps == 30
    expected = tmp_path / "expected.m4a"
    render_audio_plan(plan, str(expected), RenderSettings(
        fps=30, crossfade_sec=settings.crossfade_sec, loudness_lufs=settings.loudness_lufs))
    assert audio_stream_md5(out) == audio_stream_md5(expected)
    audio_plan = Plan.from_cuts(plan.sources["main"], [])
    audio_plan.provenance = {"audio_only": True}
    assert render_settings_for(audio_plan, load_config(cfg)).fps == 0


def test_a_picture_render_requires_a_lock_and_names_the_way_to_make_one(
        tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _, plan_path = _video_plan(tmp_path, [Cut(1.0, 1.4, "silence")])
    out = tmp_path / "final.mp4"
    assert main(["render", str(plan_path), "--out", str(out), "--config", str(_config(tmp_path))]) == 1
    err = capsys.readouterr().err
    assert "--lock" in err and "talkinghead lock" in err and "--audio-only" in err
    assert not out.exists()


def test_lock_then_render_muxes_the_locked_master_unchanged(tmp_path: Path) -> None:
    from talkinghead.lock import audio_stream_md5

    _, plan_path = _video_plan(tmp_path, [Cut(1.0, 1.4, "silence")])
    cfg = _config(tmp_path)
    master = tmp_path / "audio-master.m4a"
    _master(plan_path, master, cfg)
    assert main(["lock", str(plan_path), "--audio", str(master), "--config", str(cfg)]) == 0
    lock = tmp_path / "edit-lock.json"
    assert lock.is_file() and not Path(f"{master}.check.m4a").exists()
    out = tmp_path / "final.mp4"
    assert main(["render", str(plan_path), "--out", str(out), "--lock", str(lock), "--config", str(cfg)]) == 0
    assert audio_stream_md5(out) == audio_stream_md5(master)


def test_lock_refuses_a_master_from_another_plan_and_an_audio_only_plan(
        tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _, plan_path = _video_plan(tmp_path, [Cut(1.0, 1.4, "silence")])
    _, other_path = _video_plan(tmp_path, [Cut(2.0, 3.0, "silence")], name="other.json")
    cfg = _config(tmp_path)
    master = tmp_path / "other-master.m4a"
    _master(other_path, master, cfg)
    assert main(["lock", str(plan_path), "--audio", str(master), "--config", str(cfg)]) == 1
    assert "not this plan's audio master" in capsys.readouterr().err
    assert not (tmp_path / "edit-lock.json").exists() and not Path(f"{master}.check.m4a").exists()

    wav = tmp_path / "in.wav"
    _tone_wav(wav)
    audio_plan = Plan.from_cuts(SourceInfo("main", str(wav), "x", 3.0), [Cut(1.0, 1.5, "silence")])
    audio_plan.provenance = {"audio_only": True}
    audio_plan_path = tmp_path / "audio-plan.json"
    audio_plan_path.write_text(audio_plan.to_json())
    assert main(["lock", str(audio_plan_path), "--audio", str(master), "--config", str(cfg)]) == 1
    assert "audio-only" in capsys.readouterr().err and not (tmp_path / "edit-lock.json").exists()


def test_a_render_after_the_plan_was_edited_after_the_lock_is_refused(
        tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _, plan_path = _video_plan(tmp_path, [Cut(1.0, 1.4, "silence")])
    cfg = _config(tmp_path)
    master = tmp_path / "audio-master.m4a"
    _master(plan_path, master, cfg)
    assert main(["lock", str(plan_path), "--audio", str(master), "--config", str(cfg)]) == 0
    _video_plan(tmp_path, [Cut(2.0, 3.0, "silence")])
    out = tmp_path / "final.mp4"
    assert main(["render", str(plan_path), "--out", str(out), "--lock", str(tmp_path / "edit-lock.json"),
                 "--config", str(cfg)]) == 1
    assert "plan" in capsys.readouterr().err and not out.exists()


def test_a_locked_render_refuses_refine_joins(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _, plan_path = _video_plan(tmp_path, [Cut(1.0, 1.4, "silence")])
    out = tmp_path / "final.mp4"
    assert main(["render", str(plan_path), "--out", str(out), "--lock", str(tmp_path / "edit-lock.json"),
                 "--refine-joins", "--config", str(_config(tmp_path))]) == 1
    assert "--refine-joins" in capsys.readouterr().err and not out.exists()


def test_run_stop_at_lock_parses_and_is_refused_with_audio_only_or_plan_only(
        tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    assert build_parser().parse_args(["run", "v.mp4"]).stop_at_lock is False
    assert build_parser().parse_args(["run", "v.mp4", "--stop-at-lock"]).stop_at_lock is True
    wav = tmp_path / "in.wav"
    _tone_wav(wav)
    monkeypatch.setattr("talkinghead.cli.preflight", lambda judge: pytest.fail("reached the judge"))
    monkeypatch.setattr("talkinghead.cli.run_pipeline", lambda *a, **k: pytest.fail("cooked"))
    for flag in ("--audio-only", "--plan-only"):
        assert main(["run", str(wav), flag, "--stop-at-lock", "--judge", "shell:true"]) == 1
        assert "--stop-at-lock" in capsys.readouterr().err


def _lock_for(tmp_path: Path, cuts: list[Cut]) -> tuple[Path, Path, Path]:
    """A video plan, its approved audio master and the lock over both: (plan path, config, lock path)."""
    _, plan_path = _video_plan(tmp_path, cuts)
    cfg = _config(tmp_path)
    master = tmp_path / "audio-master.m4a"
    _master(plan_path, master, cfg)
    assert main(["lock", str(plan_path), "--audio", str(master), "--config", str(cfg)]) == 0
    return plan_path, cfg, tmp_path / "edit-lock.json"


def test_timings_maps_every_anchor_status_onto_the_locked_plan(
        tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    from talkinghead.lock import read_lock
    from talkinghead.timemap import map_span

    plan_path, cfg, lock = _lock_for(tmp_path, [Cut(1.0, 1.4, "silence")])
    anchors = tmp_path / "anchors.json"
    spans = {"kept": (0.2, 0.6), "trimmed": (0.8, 1.2), "split": (0.5, 2.0), "cut": (1.1, 1.3)}
    anchors.write_text(json.dumps([{"id": k, "start": a, "end": b} for k, (a, b) in spans.items()]))
    capsys.readouterr()
    assert main(["timings", str(plan_path), "--lock", str(lock), "--anchors", str(anchors),
                 "--config", str(cfg)]) == 0
    out = tmp_path / "timings.json"
    assert capsys.readouterr().out.splitlines()[0] == str(out)
    data = json.loads(out.read_text())
    plan = Plan.from_json(plan_path.read_text())
    locked = read_lock(lock)
    assert (data["plan_sha"], data["fps"], data["output_duration"]) == (
        locked["plan_sha"], locked["fps"], locked["output_duration"])
    rows = {r["id"]: r for r in data["anchors"]}
    assert {k: r["status"] for k, r in rows.items()} == {k: k for k in spans}
    for k, (a, b) in spans.items():
        expected = map_span(plan, a, b, fps=30)
        assert (rows[k]["pieces"], rows[k]["out_start"], rows[k]["out_end"]) == (
            expected["pieces"], expected["out_start"], expected["out_end"])
        assert (rows[k]["source"], rows[k]["start"], rows[k]["end"]) == ("main", a, b)
    assert len(rows["split"]["pieces"]) == 2 and rows["cut"]["pieces"] == []


def test_timings_prints_only_the_anchors_that_are_not_kept(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    plan_path, cfg, lock = _lock_for(tmp_path, [Cut(1.0, 1.4, "silence")])
    anchors = tmp_path / "anchors.json"
    anchors.write_text(json.dumps([{"id": "a", "start": 0.2, "end": 0.6}, {"id": "b", "start": 1.1, "end": 1.3}]))
    out = tmp_path / "sub" / "t.json"
    capsys.readouterr()
    assert main(["timings", str(plan_path), "--lock", str(lock), "--anchors", str(anchors), "--out", str(out),
                 "--config", str(cfg)]) == 0
    assert capsys.readouterr().out.splitlines() == [str(out), "b: cut"]


def test_timings_refuses_an_edited_plan_and_a_malformed_anchor_and_writes_nothing(
        tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    plan_path, cfg, lock = _lock_for(tmp_path, [Cut(1.0, 1.4, "silence")])
    anchors = tmp_path / "anchors.json"
    good = [{"id": "a", "start": 0.2, "end": 0.6}]
    anchors.write_text(json.dumps(good))
    argv = ["timings", str(plan_path), "--lock", str(lock), "--anchors", str(anchors), "--config", str(cfg)]
    bad_anchors = [
        ([{"id": "a", "start": 0.2, "end": 0.6}, {"id": "flat", "start": 2.0, "end": 2.0}], "anchor 1"),
        ([{"id": "back", "start": 3.0, "end": 2.0}], "back"),
        ([{"id": "x", "start": "1", "end": 2}], "anchor 0"),
        ([{"id": "nan", "start": float("nan"), "end": 1.0}], "nan"),
        ([{"id": "inf", "start": 0, "end": float("inf")}], "inf"),
        ([{"id": "z", "start": 0.0, "end": 1.0, "source": "nope"}], "z"),
    ]
    for entries, needle in bad_anchors:
        anchors.write_text(json.dumps(entries))
        assert main(argv) == 1
        assert needle in capsys.readouterr().err
        assert not (tmp_path / "timings.json").exists()
    anchors.write_text(json.dumps(good))
    _video_plan(tmp_path, [Cut(2.0, 3.0, "silence")])
    assert main(argv) == 1
    assert "plan" in capsys.readouterr().err and not (tmp_path / "timings.json").exists()


def _frame(path: Path, index: int) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{index})", "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).astype(np.float64)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def _probe(path: Path, entries: str, select: str) -> list[str]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", select, "-show_entries", entries,
         "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout
    return out.split()


def test_a_window_matches_the_render_it_was_cut_from(tmp_path: Path) -> None:
    plan_path, cfg, lock = _lock_for(tmp_path, [Cut(1.0, 1.4, "silence")])
    final = tmp_path / "final.mp4"
    assert main(["render", str(plan_path), "--out", str(final), "--lock", str(lock), "--config", str(cfg)]) == 0
    passage = tmp_path / "passage.mp4"
    start, end = 0.5, 1.5
    assert main(["render", str(plan_path), "--lock", str(lock), "--window", f"{start}:{end}",
                 "--out", str(passage), "--config", str(cfg)]) == 0
    assert _probe(passage, "stream=index", "a") == []
    assert _probe(passage, "stream=nb_read_frames", "v") == [str(round((end - start) * 30))]
    first = round(start * 30)
    for k in (3, 15, 27):
        got = _frame(passage, k)
        match = _psnr(got, _frame(final, first + k))
        assert match > _psnr(got, _frame(final, first + k - 1))
        assert match > _psnr(got, _frame(final, first + k + 1))


def test_window_refuses_a_missing_lock_audio_only_and_refine_joins_and_writes_nothing(
        tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    plan_path, cfg, lock = _lock_for(tmp_path, [Cut(1.0, 1.4, "silence")])
    out = tmp_path / "passage.mp4"
    base = ["render", str(plan_path), "--window", "0.5:1.5", "--out", str(out), "--config", str(cfg)]
    for extra, needle in (([], "--lock"), (["--lock", str(lock), "--audio-only"], "--audio-only"),
                          (["--lock", str(lock), "--refine-joins"], "--refine-joins")):
        assert main(base + extra) == 1
        assert needle in capsys.readouterr().err
        assert not out.exists() and not Path(f"{out}.staged.mp4").exists()
    assert main(["render", str(plan_path), "--window", "9:10", "--lock", str(lock), "--out", str(out),
                 "--config", str(cfg)]) == 1
    assert not out.exists()
