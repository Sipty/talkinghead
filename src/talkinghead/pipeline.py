"""The one-shot run: align -> analyse -> decide -> place edges -> plan -> render
-> verify -> report. Every step is memoised on its inputs, so a rerun (after a
crash, a rate limit, or a stop-and-ask) continues where it left off and a
rerun on the same input reproduces the same plan."""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from . import prompts
from .align import AlignResult, align
from .analysis import analyze
from .config import Config
from .context import Context, NeedsDecision
from .edges import EdgeContext, finalize_cuts, refine_joins
from .judge import Judge
from .profile import snapshot_preferences
from .render import RENDERER_VERSION, RenderSettings, render_audio_plan, render_plan
from .report import build_report
from .stages import content, pauses, polish, restructure, retakes, structure
from .store import Cache, RunDir, sha256_file, stable_key
from .timeline import Cut, Plan
from .transcribe import Transcriber
from .types import JUDGED_REASONS
from .verify import CheckResult, collect_facts, residual_findings, run_checks, visuals_resolution
from .visuals.cards import avatar_fingerprint
from .visuals.headclip import check_head_clip
from .visuals.render import render_visuals
from .visuals.stage import plan_visuals
from .visuals.verify import verify_visuals

# Bump when a stage's cuts change for identical inputs: it keys the stage memos.
# Per-stage overrides re-key only the changed stage, so hand-edited memos of
# the other stages survive an upgrade.
STAGE_VERSION = 2
STAGE_VERSION_OVERRIDES = {"content": 3}
VISUALS_STAGE_VERSION = 6
ALIGN_STAGE_VERSION = 2


class PipelineError(RuntimeError):
    """The run stopped: a fatal verification check or a hard failure."""


@dataclass
class RunResult:
    plan_path: Path
    final: Path
    report: Path
    checks: list[CheckResult] = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)
    final_visuals: Path | None = None

    @property
    def ok(self) -> bool:
        return all(c.ok or c.severity != "fatal" for c in self.checks)


def settings_fps(config: Config) -> float:
    return float(config.fps)


def edge_context(ctx: Context, config: Config, *, fps: float | None) -> EdgeContext:
    """The edge pass's view of one analysis: the audio, words and spans every cut and join is placed against."""
    return EdgeContext(
        duration=ctx.duration, env=ctx.env, floor=ctx.floor, words=ctx.words, moans=ctx.moans,
        voices=ctx.voices, voice_guard=config.voice_guard,
        pad=config.pause_keep / 2, splice_pad=config.splice_pad,
        splice_valley_window=config.splice_valley_window, splice_quiet_sentence=config.splice_quiet_sentence,
        splice_quiet_clause=config.splice_quiet_clause, splice_crossfade_sec=config.splice_crossfade_sec,
        splice_lead_in=config.splice_lead_in,
        quiet_above_floor_db=config.quiet_above_floor_db, filler_keep_voiced_max=config.filler_keep_voiced_max,
        flow_min_kept=config.flow_min_kept, min_cut=config.min_cut_sec,
        fps=fps, protected_spans=ctx.protected_spans,
    )


def _plan_memo_json(plan: Plan) -> str:
    """The plan as the visuals memo sees it: without the talkinghead release version.

    The version changes on every commit (git describe), and a new release must not
    re-plan overlays for a cut whose plan is otherwise identical.
    """
    payload = json.loads(plan.to_json())
    payload.get("provenance", {}).pop("talkinghead", None)
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_memo_key(plan: Plan, settings: RenderSettings, *, audio_only: bool = False) -> str:
    """The render memo's key: what actually determines the pixels and the audio.

    The EDL, the encoder settings, and `RENDERER_VERSION` -- the version of the
    ffmpeg chain in render.py, which neither the plan nor the settings can
    express (a changed colour path or pass structure produces different bytes
    from the same plan, so an old render must not replay). Keying on
    `plan.to_json()` instead put provenance -- and so the whole config -- in the
    key, and changing an unrelated setting like visuals_min_hold re-encoded a
    finished 22-minute render that could have been reused byte for byte.
    """
    return stable_key({
        "segments": [(sg.source, round(sg.start, 3), round(sg.end, 3)) for sg in plan.segments],
        "sources": {sid: si.sha256 for sid, si in sorted(plan.sources.items())},
        "settings": asdict(settings),
        "renderer": RENDERER_VERSION,
        # Only an audio-only render adds the key, so existing video memos still replay.
        **({"audio_only": True} if audio_only else {}),
    }, 2)


def make_clip(video: str, seconds: float, out: Path) -> Path:
    """First `seconds` of the video by stream copy (demo input)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.is_file():
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", video, "-t", f"{seconds:.3f}",
                 "-map", "0:v:0", "-map", "0:a:0", "-c", "copy", str(out)],
                check=True, capture_output=True, text=True,
            )
        except FileNotFoundError as e:
            raise RuntimeError("ffmpeg not found; install ffmpeg (e.g. brew install ffmpeg)") from e
        except subprocess.CalledProcessError as e:
            out.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg failed cutting the demo clip from {video}: {e.stderr.strip()}") from e
    return out


def _cuts_to_json(cuts: list[Cut]) -> list[dict]:
    return [asdict(c) for c in cuts]


def _cuts_from_json(items: list[dict]) -> list[Cut]:
    return [Cut(**c) for c in items]


def _align_result(data: dict) -> AlignResult:
    """A memoised measurement back as the measured result.

    `align.json` is `AlignResult.to_json()`, which carries a `rotation` the
    remux needs and the dataclass does not take (it belongs to `video_info`,
    and the render is already done by the time this is read).
    """
    names = {f.name for f in fields(AlignResult)}
    kwargs = {k: v for k, v in data.items() if k in names}
    if kwargs.get("output"):
        kwargs["output"] = Path(kwargs["output"])
    return AlignResult(**kwargs)


def _align_refusal(result: AlignResult, config: Config, measured_at: Path) -> PipelineError:
    if result.confidence is None:
        # Silence against silence still yields an offset; it means nothing.
        measured = "the two tracks share no signal, so there is no offset to trust"
    else:
        measured = (f"match confidence {result.confidence:.2f}x above alternatives, below the "
                    f"{config.align_min_confidence:.1f}x this config requires (offset "
                    f"{result.offset_sec:+.3f} s, drift {result.drift_ppm:+.1f} ppm)")
    return PipelineError(
        f"alignment is untrustworthy: {measured}.\n"
        f"Verify sync on a known event, then pass --allow-bad-align to cut against it anyway. "
        f"measurement: {measured_at}"
    )


def _align_source(run: RunDir, video: str, audio: str, config: Config, *, fix_drift: bool,
                  allow_bad_align: bool, fresh: bool, video_sha: str, audio_sha: str) -> tuple[str, dict]:
    """Step 0: put the external recording onto the video and cut against that.

    The measurement and the remux are memoised together on the pair of input
    files, so a rerun reuses the finished master instead of re-measuring it,
    and a different `--audio` can never reuse a master aligned to another
    recording. Every later stage sees one source path: the master.
    """
    master = run.root / "master.mov"
    measured_at = run.root / "align.json"
    payload = run.memo(
        "align",
        {"video": video_sha, "audio": audio_sha, "fix_drift": fix_drift},
        ALIGN_STAGE_VERSION,
        lambda: align(video, audio, master, fix_drift=fix_drift).to_json(),
        # The memo can outlive the master it describes (a cleaned run dir keeps
        # `stages/`); without the artifact there is nothing to reuse.
        fresh=fresh or not master.is_file(),
    )
    run.write_json(measured_at, payload)
    result = _align_result(payload)
    if not allow_bad_align and (
        result.confidence is None or result.confidence < config.align_min_confidence
    ):
        raise _align_refusal(result, config, measured_at)
    notes = {
        "summary": result.summary(),
        "offset_sec": round(result.offset_sec, 3),
        "drift_ppm": round(result.drift_ppm, 1),
        "confidence": "n/a" if result.confidence is None else round(result.confidence, 2),
        "windows_used": f"{result.windows_used}/{result.windows_total}",
    }
    return str(master), notes


# Stage memos key on the config the stage actually READS, not on the whole dict.
# The full dict made every new setting re-key every memo, which recomputes
# judged stages and discards hand-edited cuts; worse, a waveform-only setting
# has no business changing what a judge is asked. `None` (a stage not listed
# here, e.g. visuals) keeps the full dict, so its keying is unchanged.
# tests/v2/test_stages.py checks each list covers every `config.<field>` its
# stage module reads, so one cannot quietly narrow out a field it depends on
# (sentence_split_pause is listed for the judged stages because they read it
# indirectly, through Context.sentences()).
_JUDGE_KEYS = ("votes", "agree")
STAGE_CONFIG_KEYS: dict[str, tuple[str, ...] | None] = {
    "structure": _JUDGE_KEYS + ("cold_open_window_sec", "outro_window_sec", "sentence_split_pause"),
    "retakes": _JUDGE_KEYS + ("marker_votes", "marker_agree", "retake_max_sec", "sentence_split_pause"),
    "content": _JUDGE_KEYS + ("content_enabled", "content_max_cut_sec", "content_max_fraction",
                              "sentence_split_pause"),
    "pauses": ("pause_above", "pause_keep", "pause_keep_ratio", "pause_cap_mid", "pause_cap_clause",
               "pause_cap_sentence", "pause_cap_topic", "pause_topic_above", "min_filler_dur"),
    "restructure": ("restructure_brief", "restructure_window_sec", "restructure_context_sec",
                    "restructure_votes", "restructure_front", "sentence_split_pause"),
    "polish": ("polish_gap_block", "polish_gap_speaker", "polish_gap_sentence", "polish_gap_within",
               "polish_window_sec", "polish_context_sec", "polish_votes", "polish_agree", "polish_max_sec",
               "min_cut_sec", "sentence_split_pause"),
    "polish_output": ("polish_votes", "polish_agree", "min_cut_sec"),
}


def _stage_config(ctx: Context, stage: str) -> dict:
    keys = STAGE_CONFIG_KEYS.get(stage)
    cfg = ctx.config
    if keys is None:
        # The polish knobs (and the output pass's round count) change no other stage's output, so they stay out of
        # every other stage's key: turning polish on or tuning it re-keys only the polish stages and their plan.
        full = {k: v for k, v in cfg.to_dict().items() if k != "polish" and not k.startswith("polish_")}
        return {"config": full}
    return {"config": {k: getattr(cfg, k) for k in keys}}


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _retake_policy() -> str:
    """Fingerprint of everything the retake and marker judge tasks are asked.

    The stage memo is keyed on it, so a changed prompt, window or schema recomputes the
    retake decisions instead of replaying ones made under the old policy.
    """
    return _digest([inspect.getsource(fn) for fn in (
        prompts.retake_prompt, prompts.marker_prompt, prompts.marker_window, prompts.annotated_sentence_table,
    )] + [prompts.RETAKE_SCHEMA, prompts.MARKER_SCHEMA])


def _stored_round(run: RunDir, stage: str, inputs: dict, version: int):
    """A stored output round for this plan and render, replayed with the rendered transcript it was judged on.

    A round's key includes the rendered transcript's digest, which a plan-only rebuild never renders. So the round
    records its own identity: the digest of its inputs without the transcript, and the transcript digest. A plan-only
    replay accepts it only when the inputs (plan, render settings, raw cuts, policy) match now and the key the stored
    transcript digest produces is the stored key, so the transcript it replays is the one it was judged on.
    """
    path = run.memo_dir / f"{stage}.json"
    if not path.is_file():
        return None
    stored = json.loads(path.read_text())
    identity = (stored.get("value") or {}).get("identity")
    if not identity or identity.get("inputs") != _digest(inputs):
        return None
    full = {**inputs, "transcript": identity.get("transcript")}
    return stored["value"] if stored.get("key") == stable_key(full, version) else None


def _words_sha(words) -> str:
    """Digest of the words' text and bounds.

    Speakers join the digest only when a word has one, so undiarized cooks keep their memo keys.
    """
    rows = [(w.text, round(w.start, 3), round(w.end, 3)) for w in words]
    if any(w.speaker is not None for w in words):
        rows = [(*row, w.speaker) for row, w in zip(rows, words)]
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]


def _stage_inputs(ctx: Context, guidelines: str, extra: dict | None = None) -> dict:
    words_sha = _words_sha(ctx.words)
    return {
        "audio": ctx.provenance.get("audio_sha256"),
        "analysis_version": ctx.provenance.get("analysis_version"),
        "transcript": ctx.provenance.get("transcript"),
        "words": words_sha,
        "guidelines": hashlib.sha256(guidelines.encode()).hexdigest()[:16],
        "judge": ctx.judge.name,
        "decisions": ctx.decisions,
        "topic": ctx.topic,
        # Absent when empty, so cooks without preferences keep their memo keys.
        **({"preferences": _digest(ctx.preferences)} if ctx.preferences else {}),
        **({"protected": _digest([(round(a, 3), round(b, 3)) for a, b in ctx.protected_spans])}
           if ctx.protected_spans else {}),
        # Speaker labels and names reshape the judged prompts, so they key the memos; absent for one speaker.
        **({"speakers": _digest({"labels": [w.speaker for w in ctx.words], "names": ctx.config.speaker_names})}
           if any(w.speaker is not None for w in ctx.words) else {}),
        **(extra or {}),
    }


def run_pipeline(
    video: str,
    out_dir: str | Path,
    config: Config,
    *,
    transcriber: Transcriber,
    judge: Judge,
    cache: Cache,
    guidelines: str,
    topic: str | None = None,
    demo_minutes: float | None = None,
    decisions: dict | None = None,
    audio: str | None = None,
    fix_drift: bool = False,
    allow_bad_align: bool = False,
    allow_av_mismatch: bool = False,
    fresh: bool = False,
    assets_dir: str | Path | None = None,
    audio_only: bool = False,
    verify_output_asr: bool = True,
    preferences: str = "",
    refresh_preferences: bool = False,
    plan_only: bool = False,
) -> RunResult:
    if audio_only and (audio is not None or demo_minutes is not None):
        raise PipelineError("an audio-only run has no picture to align an external recording onto or to clip "
                            "for a demo: drop --audio/--demo")
    if verify_output_asr and getattr(transcriber, "replays_fixed_source", False):
        raise PipelineError(
            f"transcriber {transcriber.name!r} replays one saved source response, so it cannot transcribe "
            "the edited output: pass --no-output-asr (the report then labels the transcript as the source "
            "projection) or use a live transcriber")
    run = RunDir(Path(out_dir)).ensure()
    source_sha = sha256_file(video)
    run.bind(video, source_sha)
    notes: dict = {}
    preferences, origin = snapshot_preferences(run, preferences, refresh=refresh_preferences)
    if preferences:
        notes["profile"] = {"preferences": len([ln for ln in preferences.splitlines() if ln.strip()]),
                            "origin": origin}
    if audio is not None:
        video, notes["align"] = _align_source(run, video, audio, config, fix_drift=fix_drift,
                                              allow_bad_align=allow_bad_align, fresh=fresh,
                                              video_sha=source_sha, audio_sha=sha256_file(audio))
    prefix = ""
    if demo_minutes is not None:
        prefix = "demo-"
        video = str(make_clip(video, demo_minutes * 60.0, run.root / "demo-clip.mp4"))
        # A demo clip is truncated by construction: there is no outro to find,
        # and asking the judge for one only splits the vote. Keep to the end.
        stored = run.read_json(run.decisions, default={})
        if "outro" not in stored and not (decisions or {}).get("outro"):
            decisions = {**(decisions or {}), "outro": demo_minutes * 60.0}
    if decisions:
        stored = run.read_json(run.decisions, default={})
        stored.update(decisions)
        run.write_json(run.decisions, stored)

    ctx = analyze(video, run.root, config, transcriber, judge, cache=cache, guidelines=guidelines,
                  topic=topic, preferences=preferences, allow_av_mismatch=allow_av_mismatch,
                  audio_only=audio_only)
    plan_path = run.root / f"{prefix}plan.json"
    final = run.root / f"{prefix}final.{'wav' if audio_only else 'mp4'}"
    report_path = run.report
    base_inputs = _stage_inputs(ctx, guidelines)

    def memo_cuts(name: str, fn, extra: dict | None = None) -> tuple[list[Cut], dict]:
        payload = run.memo(
            f"{prefix}{name}", {**base_inputs, **_stage_config(ctx, name), **(extra or {})},
            STAGE_VERSION_OVERRIDES.get(name, STAGE_VERSION),
            lambda: (lambda c, d: {"cuts": _cuts_to_json(c), "notes": d})(*fn()), fresh=fresh,
        )
        return _cuts_from_json(payload["cuts"]), payload["notes"]

    try:
        structural, decided = memo_cuts("structure", lambda: structure.run(ctx))
    except NeedsDecision as e:
        plan = Plan.from_cuts(ctx.source, [])
        plan.provenance = ctx.provenance
        report_path.write_text(build_report(plan=plan, notes=notes, checks=[], status="DECISION NEEDED",
                                            render_cmd=None, question=str(e)))
        raise
    notes["structure"] = {k: {"chosen": v.get("chosen", v.get("value")), "source": v["source"]} for k, v in decided.items()}
    retake_cuts, n = memo_cuts("retakes", lambda: retakes.run(ctx), {"retake_policy": _retake_policy()})
    notes.update(n)
    content_cuts, n = memo_cuts("content", lambda: content.run(ctx, existing=structural + retake_cuts))
    notes.update(n)
    pause_cuts, n = memo_cuts("pauses", lambda: pauses.run(ctx, ctx.noise_cuts),
                              {"noise_cuts": _digest([(c.start, c.end, c.reason, c.text) for c in ctx.noise_cuts])})
    notes.update(n)

    edge_ctx = edge_context(ctx, config, fps=None if audio_only else settings_fps(config))
    raw = structural + retake_cuts + content_cuts + pause_cuts
    blocks: list[restructure.Block] = []
    if config.restructure:
        # Judged on the transcript the earlier cuts leave; the memo holds the blocks and their
        # decisions, so a hand-edited decision is honoured on replay.
        value = run.memo(
            f"{prefix}restructure",
            {**base_inputs, **_stage_config(ctx, "restructure"),
             "raw": _digest([(c.start, c.end, c.reason) for c in raw]), "policy": restructure.policy()},
            STAGE_VERSION_OVERRIDES.get("restructure", STAGE_VERSION),
            lambda: restructure.run(ctx, raw), fresh=fresh,
        )
        blocks = restructure.blocks_from(value)
        notes["restructure"] = restructure.summary(value)
        # The sentences the restructure judge saw (before its own cuts): the seams are placed in them.
        kept_sents = restructure.kept_sentences(ctx, raw)
        raw = raw + restructure.block_cuts(blocks)
    if config.polish:
        # After restructure, on the kept material: the gaps and disfluencies it leaves. The memo holds every cut
        # with its reason, words and votes, so a hand edit is honoured on replay, and a re-key carries it over.
        hand = polish.hand_cuts(run.memo_dir / f"{prefix}polish.json")
        value = run.memo(
            f"{prefix}polish",
            {**base_inputs, **_stage_config(ctx, "polish"),
             "kept": _digest([(c.start, c.end, c.reason) for c in raw]),
             "blocks": _digest([(b.first, b.last, b.start, b.end, b.decision) for b in blocks]),
             "policy": polish.policy()},
            STAGE_VERSION_OVERRIDES.get("polish", STAGE_VERSION),
            lambda: polish.run(ctx, raw, blocks, hand=hand), fresh=fresh,
        )
        notes["polish"] = value["notes"]
        raw = raw + polish.cuts_from(value)
    # The edge pass turns the raw cuts into the plan. The output pass adds rounds of cuts: each round renders the
    # plan so far, scans its transcript, and cuts what the judge confirms; the next round builds on that plan.
    def assemble(cur: list[Cut]) -> tuple[Plan, list[Cut], list]:
        cuts = finalize_cuts(cur, edge_ctx)
        plan = Plan.from_cuts(ctx.source, cuts)
        plan, joins = refine_joins(plan, edge_ctx)
        if config.restructure:
            plan = restructure.reorder(plan, blocks, edge_ctx, kept_sents)
        plan.decisions = decided
        restructured = plan.provenance.get("blocks")
        plan.provenance = {**ctx.provenance, "raw_cuts": len(cur), "final_cuts": len(cuts), "demo": demo_minutes,
                           **({"audio_only": True} if audio_only else {}),
                           **({"blocks": restructured} if restructured is not None else {})}
        # Source time of the cold open. Under restructure's reordering the output opens on an opener, so this is
        # a source instant only; it does not name where the hook sits in the cut.
        if decided.get("cold_open", {}).get("start", 0.0) > 0:
            plan.anchors["hook_start"] = float(decided["cold_open"]["start"])
        return plan, cuts, joins

    if audio_only:
        # No picture: no frame grid, no encoder settings; only the join and mastering inputs apply.
        settings = RenderSettings(fps=0, crossfade_sec=config.splice_crossfade_sec)
    else:
        settings = RenderSettings(crf=config.crf, preset=config.preset, fps=config.fps,
                                  crossfade_sec=config.splice_crossfade_sec)
    if demo_minutes is not None:
        settings = RenderSettings(crf=config.demo_crf, preset=config.demo_preset, scale=config.demo_scale,
                                  fps=config.fps, crossfade_sec=config.splice_crossfade_sec)

    output: dict = {}
    if config.polish:
        if not verify_output_asr:
            output["skipped"] = ("output ASR is off (--no-output-asr): the render was not transcribed, so the output "
                                 "pass did not run")
        elif config.polish_output_rounds == 0:
            output["skipped"] = "polish_output_rounds is 0"
    rounds_max = config.polish_output_rounds if config.polish and verify_output_asr else 0
    cur = raw
    applied: list[Cut] = []
    keep: list[dict] = []
    rounds: list[dict] = []
    round_no = 0
    facts: dict = {}
    render_cmd: str | None = None
    while True:
        # The judged word-removing decisions as the edge pass received them: verify checks the plan keeps none.
        judged = [(c.start, c.end, c.reason) for c in cur if c.reason in JUDGED_REASONS]
        plan, cuts, joins = assemble(cur)
        plan_path.write_text(plan.to_json())
        render_key = render_memo_key(plan, settings, audio_only=audio_only)
        round_inputs = {**base_inputs, **_stage_config(ctx, "polish_output"),
                        "raw": _digest([(c.start, c.end, c.reason) for c in cur]),
                        "blocks": _digest([(b.first, b.last, b.start, b.end, b.decision) for b in blocks]),
                        "round": round_no + 1, "policy": polish.output_policy(), "render": render_key}
        if plan_only:
            # Offline rebuild: nothing is rendered, so a round can only replay what a full run stored for this plan.
            if round_no >= rounds_max:
                break
            value = _stored_round(run, f"{prefix}polish-output-{round_no + 1}", round_inputs,
                                  STAGE_VERSION_OVERRIDES.get("polish_output", STAGE_VERSION))
            if value is None:
                output.setdefault("skipped", "plan-only: no stored output round for this plan; a full run "
                                             "transcribes the render and stores it")
                break
        else:
            memo_path = run.root / f"{prefix}render.json"
            memo = run.read_json(memo_path, default={})
            if final.is_file() and memo.get("key") == render_key and not fresh:
                render_cmd = memo.get("cmd", "")
            else:
                render_cmd = (render_audio_plan if audio_only else render_plan)(plan, str(final), settings)
                run.write_json(memo_path, {"key": render_key, "cmd": render_cmd})
            facts = collect_facts(plan, str(final), transcriber=transcriber if verify_output_asr else None,
                                  cache=cache, workdir=run.verify_dir,
                                  source_duration=ctx.duration, source_silences=ctx.silences,
                                  lufs_target=config.loudness_lufs, source_path=plan.sources[plan.main].path,
                                  voices=ctx.voices, speech_probes=ctx.speech_probes,
                                  config=config, env=ctx.env, floor=ctx.floor, source_words=ctx.words,
                                  scale=settings.scale, audio_only=audio_only, output_asr=verify_output_asr,
                                  judged=judged)
            if round_no >= rounds_max:
                break
            transcript = _digest(facts["render_transcript"])
            value = run.memo(
                f"{prefix}polish-output-{round_no + 1}", {**round_inputs, "transcript": transcript},
                STAGE_VERSION_OVERRIDES.get("polish_output", STAGE_VERSION),
                lambda: {**polish.output_round(ctx, plan, facts["words"], round_no + 1),
                         "identity": {"inputs": _digest(round_inputs), "transcript": transcript}},
                fresh=fresh,
            )
        rounds.append({**value["notes"], "rows": value["findings"]})
        keep = polish.merge_keeps(keep, value["findings"], value["keep"])
        if not value["cuts"]:
            break
        applied += polish.output_cuts_from(value)
        cur = raw + applied
        round_no += 1
    if config.polish:
        output["rounds"] = rounds
        output["keep"] = keep
        notes["polish"]["output"] = output
    # The edge pass's result of the final round.
    notes["edges"] = {"raw": len(cur), "final": len(cuts), "removed_sec": round(sum(c.length for c in cuts), 2),
                      "voice_spans": len(ctx.voices), "speech_probes": len(ctx.speech_probes),
                      "voice_masked_non_speech": ctx.provenance.get("unclassified_voice", {}).get("masked_non_speech"),
                      "dropped_fillers": edge_ctx.dropped_fillers, "spared_other_speaker": edge_ctx.spared_other_speaker,
                      "refined_joins": sum(j["moved"] for j in joins)}
    if run.superseded:
        # A re-key replaced memos that may hold hand edits: name them and where
        # the previous content was kept, rather than only reporting the new cut.
        notes["memos"] = {"recomputed": ",".join(sorted(set(run.superseded))),
                          "previous_kept_at": "stages/<stage>.superseded.json"}
    if plan_only:
        # Offline rebuild from the memos alone: nothing is rendered or verified. The CLI hands in a judge
        # and transcriber that refuse to be called, so a memo miss fails rather than recomputing.
        return RunResult(plan_path, final, report_path, notes=notes)
    if config.polish:
        # The judged-keep findings of every round, all judged on renders the final check may still see.
        facts["disfluency_keep"] = keep
        output["residual"] = residual_findings(facts)

    checks = run_checks(plan, facts)
    run.write_json(run.verify_dir / f"{prefix}facts.json",
                   {k: v for k, v in facts.items() if k not in ("words",)})
    fatal = [c for c in checks if not c.ok and c.severity == "fatal"]

    # Visuals are a SECOND pass over the finished cut, never part of it: the
    # audio stream is copied, so every check above keeps its meaning and a bad
    # overlay cannot cost the cut. `final.mp4` stays shippable on its own.
    final_visuals: Path | None = None
    if audio_only and config.visuals_enabled:
        notes["visuals"] = {"skipped": "audio-only run: there is no picture to overlay"}
    if config.visuals_enabled and not fatal and not audio_only:
        vdir = run.root / f"{prefix}visuals"
        assets = Path(assets_dir) if assets_dir else Path(config.assets_dir)
        avatar = avatar_fingerprint(config.visuals_avatar)
        payload = run.memo(
            f"{prefix}visuals", {**base_inputs, **_stage_config(ctx, "visuals"), "plan": _plan_memo_json(plan),
                                 "render_transcript": facts["render_transcript"],
                                 "assets": str(assets), "research": config.visuals_research,
                                 **({"avatar": avatar} if avatar else {})},
            VISUALS_STAGE_VERSION,
            lambda: (lambda pl, nt: {"placements": [asdict(x) for x in pl], "notes": nt})(
                *plan_visuals(ctx, plan, words=facts["words"], assets_dir=assets,
                              work_dir=vdir)),
            fresh=fresh,
        )
        from .visuals.schedule import Placement

        placements = [Placement(**x) for x in payload["placements"]]
        notes["visuals"] = payload["notes"]
        final_visuals = run.root / f"{prefix}final-visuals.mp4"
        # Head-clip gate runs BEFORE the bake, on the bare cut: the bake
        # would otherwise ship a card over the speaker's head, and the ~3min
        # re-encode is wasted. final.mp4 stays shippable on its own.
        hcheck, hfacts = check_head_clip(str(final), placements,
                                         margin=config.visuals_margin,
                                         corner=config.visuals_corner,
                                         work_dir=run.verify_dir / f"{prefix}visuals")
        checks.append(hcheck)
        notes["visuals"]["head_clip"] = hfacts
        if hcheck.ok:
            # Width is probed from the render inside render_visuals: `facts` has no
            # width, and a 1920 default put a 0.34 overlay across 76% of a
            # demo-tier 854 px frame.
            vcmd = render_visuals(str(final), placements, str(final_visuals),
                                  margin=config.visuals_margin, fade=config.visuals_fade,
                                  corner=config.visuals_corner)
            vchecks, vfacts = verify_visuals(final, final_visuals, placements,
                                             run.verify_dir / f"{prefix}visuals")
            checks += vchecks
            checks.append(visuals_resolution(str(final), str(final_visuals)))
            notes["visuals"]["render_cmd"] = vcmd
            notes["visuals"]["verify"] = vfacts
        else:
            final_visuals = None
        fatal = [c for c in checks if not c.ok and c.severity == "fatal"]

    if audio_only or not verify_output_asr:
        notes["verification"] = {
            "mode": "audio" if audio_only else "video",
            "output_asr": ("independent" if verify_output_asr else
                           "none: the transcript is the source words projected through the plan, "
                           "an expectation and NOT an independent measurement"),
        }
    status = "FAILED" if fatal else "OK"
    report_path.write_text(build_report(plan=plan, notes=notes, checks=checks, status=status, render_cmd=render_cmd,
                                        pacing=facts.get("pacing")))
    result = RunResult(plan_path, final, report_path, checks, facts, notes, final_visuals)
    if fatal:
        raise PipelineError("verification failed: " + "; ".join(c.line() for c in fatal) + f"\nreport: {report_path}")
    return result
