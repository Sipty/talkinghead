"""talkinghead: one command from a talking-head recording to the finished cut.

    talkinghead run VIDEO [--audio RECORDING] [--demo MINUTES] [--topic ...] [--cold-open S] [--outro S]
                    [--audio-only] [--judge SPEC] [--transcriber scribe|parakeet|whisper]
                    [--scribe-response JSON --scribe-audio WAV] [--no-output-asr]
                    [--profile DIR] [--refresh-preferences]
    talkinghead align VIDEO AUDIO [-o OUT] [--json] [--fix-drift] [--yes]
    talkinghead render PLAN.json --out OUT.mp4 | OUT.wav --audio-only
    talkinghead inspect OUT_DIR
    talkinghead board RUN_DIR [--spec JSON] [--out MP4]                 (experimental)
    talkinghead board-page RUN_DIR [--spec JSON] [--port N] [--no-open] (experimental)
    talkinghead feedback RUN_DIR "note" [--at MM:SS] [--kind video|preference|defect] [--lesson TEXT]
    talkinghead lessons [list | accept ID [--text TEXT] | reject ID]
    talkinghead --version

The judge and the transcriber come from TALKINGHEAD_JUDGE and TALKINGHEAD_TRANSCRIBER
(or --judge, --transcriber, or the config file); the editing profile comes from
TALKINGHEAD_PROFILE (or --profile): see the README's Configuration section.

Exit codes: 0 done, 1 error, 3 a structural decision is needed (the message
names the flag; the answer is persisted for every later run), 4 the judge is
unavailable (command missing, signed out, or out of quota): the run stopped
instead of switching models, and a rerun resumes from the cached progress.
130 interrupted: rerun the same command to resume.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from . import __version__, isolate
from .align import AlignResult, align, analyze
from .analysis import analyze as analyze_recording
from .color import hdr_codec, probe_color
from .config import Config, config_path_from_env, load_config
from .context import NeedsDecision
from .edges import refine_joins
from .judge import (
    ClaudeCodeJudge,
    FallbackJudge,
    Judge,
    JudgeUnavailable,
    PanelJudge,
    ShellJudge,
    judge_model,
    preflight,
)
from .pipeline import PipelineError, edge_context, run_pipeline, settings_fps
from .profile import profile_dir, read_preferences
from .prompts import system_prompt
from .render import RenderSettings, render_audio_plan, render_plan
from .stages.polish import POLISH_SCHEMA
from .store import DEFAULT_CACHE_DIR, Cache
from .timeline import Plan
from .transcribe import (
    ParakeetTranscriber,
    SavedScribeTranscriber,
    ScribeTranscriber,
    Transcriber,
    WhisperTranscriber,
    find_fluidaudio,
    require_scribe_key,
)

EXIT_NEEDS_DECISION = 3
EXIT_JUDGE_UNAVAILABLE = 4
EXIT_INTERRUPTED = 130

NO_JUDGE = ("no judge configured: set TALKINGHEAD_JUDGE to 'claude' or 'shell:<command>' (or pass --judge); "
            "see the README's Configuration section")
NO_TRANSCRIBER = ("no transcriber configured: set TALKINGHEAD_TRANSCRIBER to scribe, parakeet or whisper "
                  "(or pass --transcriber); see the README's Configuration section")


def build_judge(spec: str, *, model: str | None, guidelines: str, effort: str | None,
                fallback_command: str | None = None,
                timeout: float | None = None,
                panel: list[str] | None = None) -> Judge:
    """The judge named by `spec`, plus the configured fallback for a shell judge.

    A shell judge takes its model from its own command line, so `model` is
    written into that command (`ShellJudge.model` is read back from it); with
    None the command runs as written. For `claude`, `model` is passed as
    --model, and None leaves Claude Code's own default. The fallback command is
    used verbatim: it names its own model, so the two judges keep different
    names and never replay each other's cached answers. Claude Code carries its
    own quota handling, so it gets no fallback. `panel` adds extra judges that
    share the vote round-robin (PanelJudge); a panel run never replays a
    single-judge run's cached answers because the cache key hashes the judge's name.
    """
    if spec == "claude":
        if panel:
            raise ValueError("panel requires a shell judge; 'claude' has no shell command to add")
        return ClaudeCodeJudge(model=model, system_prompt=system_prompt(guidelines), effort=effort)
    if spec.startswith("shell:"):
        # Every shell judge, fallback and panel members included, gets the guidelines.
        system = system_prompt(guidelines)
        primary = ShellJudge(spec[len("shell:"):], model=model,
                             timeout=timeout if timeout is not None else 1800.0, system_prompt=system)
        if fallback_command:
            judge = FallbackJudge(primary, ShellJudge(fallback_command, timeout=primary.timeout,
                                                      system_prompt=system))
        else:
            judge = primary
        if panel:
            members: list[Judge] = [judge]
            for cmd in panel:
                members.append(ShellJudge(cmd, timeout=primary.timeout, system_prompt=system))
            return PanelJudge(members)
        return judge
    raise ValueError(f"unknown judge {spec!r}: use 'claude' or 'shell:<command>'")


def build_transcriber(spec: str, *, scribe_response: str | Path | None = None,
                      scribe_audio: str | Path | None = None, diarize: bool = False,
                      diarize_speakers: int | None = None) -> Transcriber:
    saved = scribe_response is not None or scribe_audio is not None
    if saved and spec != "scribe":
        raise ValueError("--scribe-response/--scribe-audio replay a Scribe response: use --transcriber scribe")
    if (diarize or diarize_speakers is not None) and spec != "scribe":
        raise ValueError("--diarize labels Scribe speakers: use --transcriber scribe")
    if diarize and saved:
        raise ValueError("a saved Scribe response is not diarized again: drop --diarize")
    if spec == "scribe":
        if saved:
            if scribe_response is None or scribe_audio is None:
                raise ValueError("a saved Scribe response needs both the response JSON and its source audio")
            return SavedScribeTranscriber(response=Path(scribe_response), source_audio=Path(scribe_audio))
        require_scribe_key()
        return ScribeTranscriber(diarize=diarize, diarize_speakers=diarize_speakers)
    if spec == "parakeet":
        if sys.platform != "darwin":
            raise RuntimeError("Parakeet (FluidAudio) runs on macOS only: use --transcriber whisper "
                               "(whisper absorbs hesitations) or scribe")
        if not find_fluidaudio():
            raise RuntimeError(
                "fluidaudiocli not found: Parakeet is required for machinery timing. Put it on PATH or set "
                "FLUIDAUDIO_BIN (see README), or pass --transcriber whisper explicitly (whisper absorbs hesitations)."
            )
        return ParakeetTranscriber()
    if spec == "whisper":
        return WhisperTranscriber()
    raise ValueError(f"unknown transcriber {spec!r}: use scribe, parakeet or whisper")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="talkinghead", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"talkinghead {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run the whole chain on one video")
    run.add_argument("video")
    run.add_argument("--out-dir", default=None, help="run directory (default: <video stem>.talkinghead next to the video)")
    run.add_argument("--audio", default=None, metavar="RECORDING",
                     help="external recording (e.g. OBS mic) to align onto the video before cutting")
    run.add_argument("--fix-drift", action="store_true",
                     help="with --audio: time-stretch it to undo the measured clock drift")
    run.add_argument("--allow-bad-align", action="store_true",
                     help="cut against a low-confidence alignment anyway")
    run.add_argument("--topic", default=None, help="one line on what the video is about (helps the judge)")
    run.add_argument("--demo", type=float, default=None, metavar="MINUTES", help="first N minutes at draft quality")
    run.add_argument("--cold-open", type=float, default=None, metavar="SECONDS", help="answer a cold-open question")
    run.add_argument("--outro", type=float, default=None, metavar="SECONDS", help="answer an outro question")
    run.add_argument("--judge", default=None, metavar="SPEC",
                     help="'claude' (Claude Code) or 'shell:<command>' (prompt on stdin, JSON on stdout); "
                          "default: TALKINGHEAD_JUDGE")
    run.add_argument("--model", default=None,
                     help="judge model: rewrites a shell command's --model, or sets claude's --model "
                          "(default: the command's own model, or Claude Code's)")
    run.add_argument("--effort", default=None, help="effort level for the claude judge")
    run.add_argument("--transcriber", default=None, choices=["scribe", "parakeet", "whisper"],
                     help="default: TALKINGHEAD_TRANSCRIBER")
    run.add_argument("--diarize", action="store_true",
                     help="Scribe: label each word with its speaker (its own transcript cache)")
    run.add_argument("--audio-only", action="store_true",
                     help="VIDEO is an audio file: cut it and write final.wav (no picture, no overlays)")
    run.add_argument("--scribe-response", default=None, metavar="JSON",
                     help="replay this saved Scribe response as the source transcript (needs --scribe-audio "
                          "and --no-output-asr); no API call")
    run.add_argument("--scribe-audio", default=None, metavar="WAV",
                     help="the exact audio the saved Scribe response was made from")
    run.add_argument("--no-output-asr", action="store_true",
                     help="do not transcribe the finished output; word/pacing facts are then the source "
                          "words projected through the plan, not an independent measurement, and the polish "
                          "output pass (which needs the rendered transcript) is skipped")
    run.add_argument("--plan-only", action="store_true",
                     help="stop after plan.json: no render, verification or visuals. Every stage memo must hit, "
                          "so no judge or transcriber call is made (a miss fails); the polish judge is the one "
                          "stage allowed to call out, since its memo is built on this rebuild")
    run.add_argument("--config", default=None,
                     help="TOML file with overrides (default: $TALKINGHEAD_CONFIG)")
    run.add_argument("--guidelines", default=None, help="editorial guidelines markdown")
    run.add_argument("--fresh", action="store_true", help="ignore stage memos (LLM answers stay cached)")
    run.add_argument("--allow-av-mismatch", action="store_true")
    run.add_argument("--no-content", action="store_true", help="skip content cuts for this run")
    run.add_argument("--isolate-voice", action="store_true",
                     help="render the audio through Apple's voice isolation (macOS only); transcription and "
                          "judging stay on the original audio")
    run.add_argument("--isolate-mix", type=float, default=None, metavar="F",
                     help="with --isolate-voice: wet fraction of the isolated audio, 0 to 1 (default 1)")
    run.add_argument("--no-visuals", action="store_true", help="skip the visuals overlay pass")
    run.add_argument("--no-research", action="store_true",
                     help="skip fetching reference screenshots; cards only")
    run.add_argument("--assets-dir", default=None, help="asset library (default: editorial/assets)")
    run.add_argument("--visuals-corner", default=None,
                     choices=["top-right", "top-left", "bottom-right", "bottom-left"],
                     help="corner the overlays take (default: top-right); a framing choice, per cook")
    run.add_argument("--visuals-scale-card", type=float, default=None,
                     help="card overlays as a fraction of the frame width (default: 0.46)")
    run.add_argument("--visuals-scale-shot", type=float, default=None,
                     help="photo/screenshot overlays as a fraction of the frame width (default: 0.42)")
    run.add_argument("--profile", default=None, metavar="DIR",
                     help="editing profile directory (default: $TALKINGHEAD_PROFILE)")
    run.add_argument("--refresh-preferences", action="store_true",
                     help="re-take this cook's preferences snapshot from the current profile")

    al = sub.add_parser("align", help="align an external recording onto a video and remux the master")
    al.add_argument("video", help="video master (its picture is copied; only a display rotation is re-encoded)")
    al.add_argument("audio", help="external recording (e.g. OBS mic); replaces the video's audio track")
    al.add_argument("-o", "--out", default=None, help="output master (default: <video stem>.aligned.mov)")
    al.add_argument("--fix-drift", action="store_true",
                    help="time-stretch the audio to undo the measured clock drift")
    al.add_argument("--yes", "-y", action="store_true", help="remux without confirmation")
    al.add_argument("--json", action="store_true",
                    help="print the result as JSON instead of the report (the only thing on stdout)")

    board = sub.add_parser("board", help="render a synchronised tablet board panel (experimental)")
    board.add_argument("run_dir")
    board.add_argument("--spec", default=None, help="board.json (default: <RUN_DIR>/board/board.json)")
    board.add_argument("--out", default=None)

    board_page = sub.add_parser("board-page", help="edit a board video specification in a browser (experimental)")
    board_page.add_argument("run_dir")
    board_page.add_argument("--spec", default=None, help="board.json (default: <RUN_DIR>/board/board.json)")
    board_page.add_argument("--port", type=int, default=8765)
    board_page.add_argument("--no-open", action="store_true")

    rend = sub.add_parser("render", help="re-render a plan exactly")
    rend.add_argument("plan")
    rend.add_argument("--out", required=True)
    rend.add_argument("--config", default=None,
                      help="TOML file with overrides (default: $TALKINGHEAD_CONFIG)")
    rend.add_argument("--audio-only", action="store_true",
                      help="the plan is an audio-only cook: write its WAV (no picture, no ASR, no judge)")
    rend.add_argument("--refine-joins", action="store_true",
                      help="refine the join placement against the recording's cached analysis first; "
                           "writes <out>.plan.json and <out>.joins.json beside the output")

    insp = sub.add_parser("inspect", help="print a run's report")
    insp.add_argument("out_dir")

    fb = sub.add_parser("feedback", help="record a correction from reviewing a cook, optionally with a lesson")
    fb.add_argument("run_dir")
    fb.add_argument("note", help="what was wrong or what you changed")
    fb.add_argument("--at", default=None, metavar="TIME",
                    help="output timestamp the note refers to, seconds or MM:SS (required for a defect)")
    fb.add_argument("--kind", default="video", choices=["video", "preference", "defect"],
                    help="video: this cut only; preference: taste worth keeping; defect: a tool bug, "
                         "recorded as a fixture and a test stub under the clone's tests/ (default: video)")
    fb.add_argument("--lesson", default=None, metavar="TEXT",
                    help="a durable rule proposed from this note; it waits for `talkinghead lessons accept`")
    fb.add_argument("--profile", default=None, metavar="DIR",
                    help="editing profile directory (default: $TALKINGHEAD_PROFILE)")
    fb.add_argument("--fixtures-dir", default=None, help="defect fixtures (default: <repo>/tests/fixtures/feedback)")
    fb.add_argument("--tests-dir", default=None, help="defect test stubs (default: <repo>/tests/feedback)")

    les = sub.add_parser("lessons", help="review proposed lessons and promote them to preferences")
    les.add_argument("action", nargs="?", default="list", choices=["list", "accept", "reject"])
    les.add_argument("id", nargs="?", type=int, help="lesson id (for accept and reject)")
    les.add_argument("--text", default=None, help="with accept: the rule as it should be saved, reworded")
    les.add_argument("--profile", default=None, metavar="DIR",
                     help="editing profile directory (default: $TALKINGHEAD_PROFILE)")
    return p


def parse_time(text: str) -> float:
    if ":" in text:
        parts = [float(x) for x in text.split(":")]
        return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
    return float(text)


def _default_out_dir(video: Path) -> Path:
    return video.with_name(video.stem + ".talkinghead")


def _repo_file(rel: str) -> Path:
    """A file that ships with the source checkout (editorial rules, asset library, test fixtures).

    Only a clone (`uv sync` in the checkout) carries these; a wheel install does not.
    """
    return Path(__file__).resolve().parents[2] / rel


def _config_file(given: str | None) -> Path | None:
    path = Path(given) if given else config_path_from_env()
    if path is not None and not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    return path


def _check_hdr_encoder(video: Path) -> None:
    """An HDR source needs a 10-bit HEVC encoder: refuse before any work, not after the cook."""
    try:
        hdr = probe_color(video).hdr
    except RuntimeError:
        return  # not a readable video: the pipeline reports that with its own message
    if hdr:
        hdr_codec()


def _check_tools(*, visuals: bool = False) -> None:
    """Every external tool the command needs, checked before any work starts."""
    missing: list[str] = []
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        missing.append("ffmpeg/ffprobe (install: brew install ffmpeg)")
    if visuals:
        if shutil.which("rsvg-convert") is None:
            missing.append("rsvg-convert (install: brew install librsvg, or pass --no-visuals)")
        if sys.platform == "darwin" and shutil.which("swiftc") is None:
            missing.append("swiftc (install: xcode-select --install, or pass --no-visuals)")
    if missing:
        raise RuntimeError("missing tool(s): " + "; ".join(missing))


# Above this the external clock drifts far enough to be audible over a long
# recording; the measurement is always reported, the fix is opt-in.
_DRIFT_WARN_PPM = 30.0


def _align_report(result: AlignResult, video: Path, audio: Path) -> str:
    """What the measurement means, before anything is re-encoded."""
    confidence = "n/a" if result.confidence is None else f"{result.confidence:.2f}x above alternatives"
    threshold = Config().align_min_confidence
    lines = [
        f"video:            {video}",
        f"audio:            {audio}",
        f"offset:           {result.offset_sec:+.3f} s on the video timeline",
        f"coarse estimate:  {result.coarse_offset_sec:+.3f} s",
        f"drift:            {result.drift_ppm:+.1f} ppm",
        f"fine windows:     {result.windows_used}/{result.windows_total}",
        f"match confidence: {confidence}",
        f"durations:        video {result.video_duration:.3f} s, audio {result.audio_duration:.3f} s",
    ]
    if result.confidence is None:
        lines.append("WARNING: no positive overlap - the two tracks share no signal, so the offset is meaningless.")
    elif result.confidence < threshold:
        lines.append(f"WARNING: match confidence is below {threshold:.1f}x: verify sync on a known event "
                     "before publishing (a run with --audio refuses this unless --allow-bad-align).")
    if abs(result.drift_ppm) > _DRIFT_WARN_PPM:
        lines.append(f"WARNING: clock drift {result.drift_ppm:+.1f} ppm exceeds {_DRIFT_WARN_PPM:.0f} ppm; "
                     "--fix-drift time-stretches the audio to undo it.")
    if result.windows_used < 3:
        lines.append("WARNING: fewer than 3 fine windows matched; check that the two recordings share content.")
    return "\n".join(lines)


def _confirmed() -> bool:
    """Ask before re-encoding. The prompt goes to stderr so that `--json`
    leaves stdout as exactly one parseable object; EOF (no terminal) is a no."""
    print("Remux now? [y/N] ", end="", file=sys.stderr, flush=True)
    return sys.stdin.readline().strip().lower() in ("y", "yes")


def _align_cmd(args: argparse.Namespace) -> int:
    _check_tools()
    video, audio = Path(args.video), Path(args.audio)
    for path, what in ((video, "video"), (audio, "audio")):
        if not path.is_file():
            raise FileNotFoundError(f"{what} not found: {path}")
    out = Path(args.out) if args.out else video.with_name(video.stem + ".aligned.mov")
    # Measure, show, ask, then remux: the report belongs in front of a
    # confirmation, not behind a minutes-long re-encode. `align` measures a
    # second time, which costs one decode of each file. With --json the report
    # goes to stderr, so stdout carries the JSON object and nothing else.
    measured = analyze(video, audio, fix_drift=args.fix_drift)
    print(_align_report(measured, video, audio), file=sys.stderr if args.json else sys.stdout)
    print(f"output: {out}", file=sys.stderr)
    if not args.yes and not _confirmed():
        print(f"Aborted: {out} not written. Re-run with --yes to remux.", file=sys.stderr)
        return 1
    result = align(video, audio, out, fix_drift=args.fix_drift)
    # The JSON comes last so it can name the file the measurement was written
    # to, and it is the only thing this command ever puts on stdout.
    if args.json:
        print(json.dumps(result.to_json(), indent=2))
    print(f"done: {out}", file=sys.stderr)
    return 0


NO_PROFILE = ("no editing profile: set TALKINGHEAD_PROFILE to a directory (it is created on first use) "
              "or pass --profile DIR")


def _feedback_cmd(args: argparse.Namespace) -> int:
    from .feedback import record_defect
    from .profile import profile_dir, record_feedback

    profile = profile_dir(args.profile)
    out_time = parse_time(args.at) if args.at is not None else None
    extra: dict = {}
    if args.kind == "defect":
        if out_time is None:
            raise ValueError("a defect needs --at TIME: the fixture is cut around that instant")
        if args.lesson:
            raise ValueError("a defect is fixed in code with a test, not learned as a preference: drop --lesson")
        defect = record_defect(
            args.run_dir, out_time=out_time, note=args.note,
            fixtures_dir=args.fixtures_dir or _repo_file("tests/fixtures/feedback"),
            tests_dir=args.tests_dir or _repo_file("tests/feedback"),
        )
        print(f"fixture: {defect['fixture']}\nledger entry #{defect['id']}\ntest stub: {defect['test_stub']}")
        # The profile, when there is one, keeps the full review history.
        if profile is None:
            return 0
        extra = {"fixture": defect["fixture"], "test_stub": defect["test_stub"]}
    elif profile is None:
        raise ValueError(NO_PROFILE)
    entry = record_feedback(profile, args.run_dir, kind=args.kind, note=args.note, out_time=out_time,
                            lesson=args.lesson, extra=extra)
    where = f" (source {entry['source_time']:.2f}s)" if "source_time" in entry else ""
    print(f"feedback #{entry['id']} [{entry['kind']}]{where}: {entry['note']}")
    if entry["status"] == "pending":
        print(f"proposed lesson #{entry['id']}: {entry['lesson']}\n"
              f"accept it with: talkinghead lessons accept {entry['id']}")
    return 0


def _lessons_cmd(args: argparse.Namespace) -> int:
    from .profile import accept_lesson, pending_lessons, profile_dir, reject_lesson

    profile = profile_dir(args.profile)
    if profile is None:
        raise ValueError(NO_PROFILE)
    if args.action == "list":
        pending = pending_lessons(profile)
        if not pending:
            print("no pending lessons")
        for e in pending:
            print(f"#{e['id']} [{e['kind']}] {e['lesson']}\n    from: {e['note']}")
        return 0
    if args.id is None:
        raise ValueError(f"lessons {args.action} needs a lesson id: see `talkinghead lessons list`")
    if args.action == "accept":
        entry = accept_lesson(profile, args.id, text=args.text)
        print(f"accepted #{entry['id']}: {entry['rule']}\nsaved to {profile / 'preferences.md'}; "
              "new cooks use it (existing cooks: run again with --refresh-preferences)")
        return 0
    entry = reject_lesson(profile, args.id)
    print(f"rejected #{entry['id']}")
    return 0


class _CachedTranscriber:
    """The recording's cached transcript only: a render never spends an ASR call on a miss."""

    def __init__(self, real) -> None:
        self.name, self.version = real.name, real.version

    def transcribe(self, wav_path: str):
        raise RuntimeError("no cached transcript for this recording: run `talkinghead run` on it first")


class _MemoOnlyJudge:
    """Plan-only rebuilds: a judge call means a stage memo missed, which is an error, not a silent recompute.

    The exception is a schema named in `allow`: the polish judge, a stage whose memo the first plan-only rebuild
    of a cook may have to build. Every other stage must replay.
    """

    def __init__(self, real, allow: tuple = ()) -> None:
        self.name = real.name
        self._real = real
        self._allow = allow

    def decide(self, prompt, schema, *args, **kwargs):
        if any(schema is s for s in self._allow):
            return self._real.decide(prompt, schema, *args, **kwargs)
        raise RuntimeError("plan-only: a stage memo missed and would call the judge; run `talkinghead run` first")


def _refine_for_render(plan: Plan, cfg: Config, out: Path, *, audio_only: bool) -> Plan:
    """Refine the plan's joins against its recording's cached analysis, and write the refined plan
    and the join move records beside OUT (`<stem>.plan.json`, `<stem>.joins.json`)."""
    if cfg.transcriber is None:
        raise ValueError(NO_TRANSCRIBER)
    source = plan.sources[plan.main]
    run_dir = Path(DEFAULT_CACHE_DIR) / "refine" / source.sha256[:16]
    real = build_transcriber(cfg.transcriber, diarize=cfg.diarize, diarize_speakers=cfg.diarize_speakers)
    ctx = analyze_recording(source.path, run_dir, cfg, _CachedTranscriber(real), None, cache=Cache(),
                            guidelines="", audio_only=audio_only)
    refined, records = refine_joins(plan, edge_context(ctx, cfg, fps=None if audio_only else float(cfg.fps)))
    stem = out.parent / out.stem
    Path(f"{stem}.plan.json").write_text(refined.to_json())
    Path(f"{stem}.joins.json").write_text(json.dumps(records, indent=1) + "\n")
    print(f"refined {sum(1 for r in records if r['moved'])} of {len(records)} joins: {stem}.plan.json",
          file=sys.stderr)
    return refined



def _command(args: argparse.Namespace) -> int:
    if args.cmd == "align":
        return _align_cmd(args)
    if args.cmd == "inspect":
        report = Path(args.out_dir) / "report.md"
        print(report.read_text() if report.is_file() else f"no report in {args.out_dir}")
        return 0
    if args.cmd == "feedback":
        return _feedback_cmd(args)
    if args.cmd == "lessons":
        return _lessons_cmd(args)
    if args.cmd == "board":
        from .board.render import render_board
        from .board.spec import load_spec

        run_dir = Path(args.run_dir)
        spec_path = Path(args.spec) if args.spec else run_dir / "board" / "board.json"
        result = render_board(run_dir, load_spec(spec_path), Path(args.out) if args.out else None)
        print(f"rendered {result}")
        return 0
    if args.cmd == "board-page":
        from .board.page import serve_page

        run_dir = Path(args.run_dir)
        spec_path = Path(args.spec) if args.spec else run_dir / "board" / "board.json"
        serve_page(run_dir, spec_path, port=args.port, open_browser=not args.no_open)
        return 0
    if args.cmd == "render":
        _check_tools()
        cfg = load_config(_config_file(args.config))
        plan = Plan.from_json(Path(args.plan).read_text())
        if args.refine_joins:
            plan = _refine_for_render(plan, cfg, Path(args.out), audio_only=args.audio_only)
        if args.audio_only:
            cmd = render_audio_plan(plan, args.out, RenderSettings(fps=0, crossfade_sec=cfg.splice_crossfade_sec))
        elif plan.provenance.get("audio_only"):
            raise ValueError(f"{args.plan} is an audio-only plan: pass --audio-only to render its WAV")
        else:
            settings = RenderSettings(fps=settings_fps(cfg), crf=cfg.crf, preset=cfg.preset,
                                      crossfade_sec=cfg.splice_crossfade_sec)
            cmd = render_plan(plan, args.out, settings)
        print(f"rendered {args.out}")
        print(cmd, file=sys.stderr)
        return 0
    return _run_cmd(args)


def _run_cmd(args: argparse.Namespace) -> int:
    video = Path(args.video)
    if not video.is_file():
        raise FileNotFoundError(f"{'audio' if args.audio_only else 'video'} not found: {video}")
    if args.audio_only and (args.audio or args.demo is not None):
        raise ValueError("--audio-only has no picture to align an external recording onto or to clip "
                         "for a demo: drop --audio/--demo")
    if bool(args.scribe_response) != bool(args.scribe_audio):
        raise ValueError("--scribe-response and --scribe-audio go together: the saved response is only "
                         "valid for the exact audio that produced it")
    if args.scribe_response and not args.no_output_asr:
        raise ValueError("a saved Scribe response is the SOURCE transcript; it cannot transcribe the edited "
                         "output. Pass --no-output-asr (the report labels the transcript as the source "
                         "projection, not an independent measurement)")
    audio = None
    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.is_file():
            raise FileNotFoundError(f"audio recording not found: {audio_path}")
        audio = str(audio_path.resolve())
    cfg = load_config(
        _config_file(args.config), judge=args.judge, model=args.model, effort=args.effort,
        transcriber=args.transcriber,
        guidelines=args.guidelines, content_enabled=(False if args.no_content else None),
        visuals_enabled=(False if args.no_visuals else None),
        visuals_research=(False if args.no_research else None),
        visuals_corner=args.visuals_corner,
        visuals_scale_card=args.visuals_scale_card,
        visuals_scale_shot=args.visuals_scale_shot,
        diarize=(True if args.diarize else None),
        isolate_voice=(True if args.isolate_voice else None),
        isolate_mix=args.isolate_mix,
    )
    if cfg.judge is None:
        raise ValueError(NO_JUDGE)
    if cfg.isolate_voice:
        # Refuse here, before any judge or transcriber spend, when the unit cannot run on this machine.
        isolate.preflight(Cache().root)
    # A saved Scribe response is always replayed through the Scribe transcriber.
    transcriber_spec = "scribe" if args.scribe_response and args.transcriber is None else cfg.transcriber
    if transcriber_spec is None:
        raise ValueError(NO_TRANSCRIBER)
    visuals = cfg.visuals_enabled and not args.audio_only
    _check_tools(visuals=visuals)
    if not args.audio_only:
        _check_hdr_encoder(video)
    cfg = replace(cfg, model=judge_model(cfg.judge, cfg.model))
    guidelines_path = Path(cfg.guidelines)
    if not guidelines_path.is_absolute():
        guidelines_path = _repo_file(cfg.guidelines)
    if not guidelines_path.is_file():
        raise FileNotFoundError(f"guidelines not found: {guidelines_path}. Bundled files come from the source "
                                "checkout: run talkinghead from a clone (`uv sync` in the checkout), or pass "
                                "--guidelines with a path of your own.")
    guidelines = guidelines_path.read_text()
    assets = Path(args.assets_dir or cfg.assets_dir)
    if args.assets_dir is None and not assets.is_absolute():
        assets = _repo_file(cfg.assets_dir)
    if visuals and not assets.is_dir():
        raise FileNotFoundError(f"asset library not found: {assets}. Bundled assets come from the source "
                                "checkout: run talkinghead from a clone (`uv sync` in the checkout), or pass "
                                "--assets-dir.")
    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir(video.resolve())
    decisions = {k: v for k, v in (("cold_open", args.cold_open), ("outro", args.outro)) if v is not None}
    judge = build_judge(cfg.judge, model=cfg.model, guidelines=guidelines, effort=cfg.effort,
                        timeout=cfg.judge_timeout,
                        fallback_command=cfg.fallback_command, panel=cfg.panel)
    # Before transcription: a missing judge must surface in seconds, not
    # after the first hour of work.
    transcriber = build_transcriber(
        transcriber_spec, scribe_response=args.scribe_response, scribe_audio=args.scribe_audio,
        diarize=cfg.diarize, diarize_speakers=cfg.diarize_speakers)
    if args.plan_only:  # no judge preflight, and a memo miss fails instead of calling out
        judge, transcriber = _MemoOnlyJudge(judge, allow=(POLISH_SCHEMA,)), _CachedTranscriber(transcriber)
    else:
        preflight(judge)
    result = run_pipeline(
        str(video.resolve()), out_dir, cfg,
        transcriber=transcriber,
        judge=judge,
        cache=Cache(), guidelines=guidelines, topic=args.topic, demo_minutes=args.demo,
        decisions=decisions or None, audio=audio, fix_drift=args.fix_drift,
        allow_bad_align=args.allow_bad_align, allow_av_mismatch=args.allow_av_mismatch, fresh=args.fresh,
        assets_dir=str(assets),
        audio_only=args.audio_only, verify_output_asr=not args.no_output_asr,
        preferences=read_preferences(profile_dir(args.profile)),
        refresh_preferences=args.refresh_preferences, plan_only=args.plan_only,
    )
    if args.plan_only:
        print(f"plan: {result.plan_path}")
        return 0
    print(f"final: {result.final}\nreport: {result.report}")
    if result.final_visuals:
        print(f"visuals: {result.final_visuals}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _command(args)
    except NeedsDecision as e:
        print(f"talkinghead: {e}", file=sys.stderr)
        return EXIT_NEEDS_DECISION
    except JudgeUnavailable as e:
        print(f"talkinghead: JUDGE UNAVAILABLE: {e}\n"
              "The run stopped rather than answer through another model. Fix the judge named above "
              "(or wait out a usage limit) and rerun the same command; finished stages and judge "
              "samples replay from cache.", file=sys.stderr)
        return EXIT_JUDGE_UNAVAILABLE
    except subprocess.CalledProcessError as e:
        cmd = e.cmd if isinstance(e.cmd, str) else " ".join(map(str, e.cmd))
        print(f"talkinghead: a tool failed (exit {e.returncode}): {cmd[:300]}\n"
              "Rerun the same command; finished stages replay from cache.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("talkinghead: interrupted; rerun the same command to resume from the cache", file=sys.stderr)
        return EXIT_INTERRUPTED
    except (PipelineError, FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"talkinghead: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
