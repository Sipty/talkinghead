"""Verification of the render: measure facts, then evaluate a declarative
check table. A fatal failure stops the run with the report written; a warning
is recorded. There are no autonomous fix loops: if the plan was wrong, the
report says where and the fix lands in code or guidelines, not in a retry.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .audio import Envelope
from .color import probe_color
from .config import Config
from .derive import pause_boundary, pause_cap
from .edges import is_intra_restart
from .joins import check_join, source_edge_facts
from .render import AUDIO_RATE
from .residual import finding_source, same_occurrence
from .residual import scan as scan_disfluencies
from .store import Cache
from .timeline import Plan, Segment
from .transcribe import Transcriber, Transcript, Word, av_durations, prepare_audio, transcribe_cached

FIRST_WORD_MAX = 0.5
AV_SYNC_TOL = 0.2
LUFS_TOL = 1.0
GAP_MIN = 0.5
# Silence tolerated beyond the cap of the boundary a pause sits on: silence
# detection and word stamps do not agree to the millisecond.
DEAD_AIR_SLACK = 0.10
# Quiet pauses consume the gap budget; digital mute must be below one LSB
# of the prepared 16-bit PCM so nonzero room tone never counts as missing audio.
QUIET_DB = -50.0
QUIET_SILENCE_MIN = 0.5
DIGITAL_DB = -96.0
DIGITAL_SILENCE_MIN = 1.0
# Source<->render timing: interior speech of the render is cross-correlated
# against the source it was cut from. 60 ms and a normalized correlation of 0.65
# are the thresholds the reference narrative cut shipped with; a wrong mapping
# moves the audio, and nothing else in the table would notice.
SOURCE_TIMING_TOL = 0.06
SOURCE_CORR_MIN = 0.65
# Shortest window that still gives an unambiguous peak, and therefore the
# shortest keep worth sampling (half its own length, centred).
SOURCE_TIMING_MIN_WINDOW = 0.8
SOURCE_TIMING_MIN_SEGMENT = 2 * SOURCE_TIMING_MIN_WINDOW
# Preferred sample length; a shorter keep takes half of itself instead.
SOURCE_TIMING_WINDOW = 3.0
# Sampled keeps, spread evenly by index across the plan: bounded cost, and a
# wrong mapping anywhere in the EDL has a chance of being inside one of them.
SOURCE_TIMING_MAX_SAMPLES = 24
SOURCE_TIMING_PAD = 0.30
SOURCE_TIMING_SR = 2000
# A window with less energy than this has nothing to align: correlating digital
# silence answers with index 0, which would read as a fabricated -pad offset.
# The render's silence tiers own "the audio is not there", not this check.
SOURCE_TIMING_MIN_RMS_DB = -50.0
# Smaller overlap than this between a cut and an unclassified voiced span is
# edge rounding, not a swallowed word.
UNCLASSIFIED_REPORT_MIN = 0.05


@dataclass
class CheckResult:
    name: str
    severity: str  # "fatal" | "warn" | "info"
    ok: bool
    detail: str

    def line(self) -> str:
        status = "OK" if self.ok else ("FAIL" if self.severity == "fatal" else self.severity.upper())
        return f"- {self.name}: {status}: {self.detail}"


def _duration(plan: Plan, f: dict) -> CheckResult:
    expected = plan.output_duration
    tol = max(0.5, 0.03 * len(plan.joins()))
    delta = f["duration"] - expected
    return CheckResult("duration", "fatal", abs(delta) <= tol,
                       f"{f['duration']:.2f}s rendered vs {expected:.2f}s planned (delta {delta:+.2f}s, tol {tol:.2f}s)")


def _av_sync(plan: Plan, f: dict) -> CheckResult:
    d = f["video_duration"] - f["audio_duration"]
    return CheckResult("av_sync", "fatal", abs(d) <= AV_SYNC_TOL,
                       f"video {f['video_duration']:.2f}s vs audio {f['audio_duration']:.2f}s (delta {d:+.2f}s)")


def _loudness(plan: Plan, f: dict) -> CheckResult:
    target = f.get("lufs_target", -16.0)
    return CheckResult("loudness", "fatal", abs(f["lufs"] - target) <= LUFS_TOL,
                       f"{f['lufs']:.1f} LUFS (target {target:.0f} +-{LUFS_TOL:.0f})")


def _cold_open(plan: Plan, f: dict) -> CheckResult:
    if f.get("output_asr") == "source-projection":
        return CheckResult("cold_open", "info", True,
                           "not measured: output ASR disabled; the source projection is not an independent measurement")
    words: list[Word] = f["words"]
    if not words:
        return CheckResult("cold_open", "fatal", False, "render transcript has no words")
    first = words[0].start
    return CheckResult("cold_open", "fatal", first <= FIRST_WORD_MAX,
                       f"first word {words[0].text!r} at {first:.2f}s (<= {FIRST_WORD_MAX}s required)")




def judged_leaks(plan: Plan, words: Sequence, judged: Sequence[tuple[float, float, str]]) -> list[dict]:
    """Transcript words the plan keeps although a judged cut removed them.

    `judged` are the judges' word-removing decisions as (start, end, reason) on the main source. A word
    is removed by a decision when its midpoint lies in the decision's span; the plan keeps it when its
    midpoint lies in a kept segment of the main source. Empty when the plan honours every decision.
    """
    kept = sorted((s.start, s.end) for s in plan.segments if s.source == plan.main)
    starts = [a for a, _ in kept]
    out: list[dict] = []
    for w in words:
        mid = (w.start + w.end) / 2
        reason = next((r for a, b, r in judged if a <= mid < b), None)
        if reason is None:
            continue
        i = bisect_right(starts, mid) - 1
        if i >= 0 and kept[i][0] <= mid < kept[i][1]:
            out.append({"text": w.text, "start": round(w.start, 3), "end": round(w.end, 3), "reason": reason,
                        "speaker": getattr(w, "speaker", None)})
    return out


# A transcript word is clipped when cuts take more than CLIP_MIN and less than CLIP_MAX of its ASR audio: a cut
# edge lies inside the word. A word wholly removed or wholly kept is not clipped.
CLIP_MIN = 0.10
CLIP_MAX = 0.90


def clipped_words(plan: Plan, words: Sequence) -> list[dict]:
    """Transcript words (any speaker, ASR stamps) that a cut takes part of, with the cut reasons covering them."""
    kept = sorted((s.start, s.end) for s in plan.segments if s.source == plan.main)
    out: list[dict] = []
    for w in words:
        s, e = getattr(w, "asr_start", w.start), getattr(w, "asr_end", w.end)
        if e <= s:
            continue
        kept_share = sum(max(0.0, min(e, b) - max(s, a)) for a, b in kept) / (e - s)
        share = 1.0 - min(1.0, kept_share)
        if CLIP_MIN < share < CLIP_MAX:
            reasons = sorted({c.reason for c in plan.cuts if c.start < e and c.end > s})
            out.append({"text": w.text, "start": round(s, 3), "end": round(e, 3), "cut": round(share, 3),
                        "reasons": reasons, "speaker": getattr(w, "speaker", None)})
    return out


def _dead_air_runs(
    plan: Plan, silences: list[tuple[float, float]], min_gap: float,
) -> list[tuple[float, float, float]]:
    """(start, end, raw) output silence runs: source VAD silence mapped onto
    the output, joined across edit seams. `raw` is the longest whole source
    silence that contributes to the run, i.e. the pause before it was trimmed."""
    pieces: list[tuple[float, float, float, int]] = []
    offset = 0.0
    group = 0  # bumped at every seam the source does not run forward across
    prev: Segment | None = None
    for seg in plan.segments:
        if seg.source != plan.main:
            offset += seg.length
            continue
        if prev is not None and not (seg.source == prev.source and seg.start > prev.end):
            group += 1
        prev = seg
        for a, b in silences:
            lo, hi = max(a, seg.start), min(b, seg.end)
            if hi > lo:
                pieces.append((offset + lo - seg.start, offset + hi - seg.start, b - a, group))
        offset += seg.length
    pieces.sort()
    merged: list[list] = []
    for a, b, raw, g in pieces:
        if merged and merged[-1][3] == g and a <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], b)
            merged[-1][2] = max(merged[-1][2], raw)
        else:
            merged.append([a, b, raw, g])
    return [(a, b, raw) for a, b, raw, _ in merged if b - a >= min_gap]


def residual_dead_air(plan: Plan, silences: list[tuple[float, float]], min_gap: float = GAP_MIN) -> list[tuple[float, float]]:
    """Map source VAD silence onto the output, joining quiet across edit seams."""
    return [(a, b) for a, b, _ in _dead_air_runs(plan, silences, min_gap)]


def quiet_silence_min(config: Config) -> float:
    """Shortest quiet run in the render that counts as dead air: a pause the
    policy keeps on purpose (up to the topic cap, plus slack) never does."""
    return max(QUIET_SILENCE_MIN, config.pause_cap_topic + DEAD_AIR_SLACK)


def dead_air_excess(
    runs: Sequence[tuple[float, float]], words: Sequence[Word], config: Config,
    raw: Sequence[float] | None = None,
) -> list[dict]:
    """Each output silence run with the part of it that no pause policy asked for.

    `runs` and `words` are on the output timeline (words: the render's own
    transcript). The run is classified by the last word that STARTS before it,
    as the pause policy does, and only the length beyond that boundary's cap
    plus DEAD_AIR_SLACK is `excess`: dead air the cut failed to remove.
    `raw` is each run's untrimmed source pause: a topic pause trimmed to its
    cap is a topic pause however short the kept run. Default: the run itself.
    """
    starts = [w.start for w in words]
    out: list[dict] = []
    for k, (a, b) in enumerate(runs):
        i = bisect_left(starts, a) - 1
        boundary = pause_boundary(words[i].text if i >= 0 else "", raw[k] if raw else b - a, config)
        cap = pause_cap(boundary, config)
        out.append({"start": round(a, 3), "end": round(b, 3), "boundary": boundary, "cap": cap,
                    "excess": round(max(0.0, (b - a) - (cap + DEAD_AIR_SLACK)), 3)})
    return out


def _judged_removals(plan: Plan, f: dict) -> CheckResult:
    """Every word a judged cut removes stays removed: a judge's span is a decision the edge pass may not undo."""
    leaks = f.get("judged_leaks")
    if leaks is None:
        return CheckResult("judged_removals", "info", True, "not measured: no judged decisions were given")
    if not leaks:
        return CheckResult("judged_removals", "fatal", True, "every word a judged cut removes stays removed")
    head = "; ".join(f"{x['text']!r} at {x['start']:.2f}s ({x['reason']})" for x in leaks[:5])
    return CheckResult("judged_removals", "fatal", False,
                       f"{len(leaks)} transcript words the plan keeps inside judged cuts: {head}"
                       + (" ..." if len(leaks) > 5 else ""))


def _clipped_words(plan: Plan, f: dict) -> CheckResult:
    """No placed gap cut takes part of a transcript word: silence between words never clips the word it separates.

    Other cuts that take part of a word are reported as a warning (_clipped_speech): ASR stamps overlap for repeated
    words and run long at word ends, so a judged removal can legitimately meet one.
    """
    clipped = f.get("clipped_words")
    if clipped is None:
        return CheckResult("clipped_words", "fatal", True, "not measured: no source transcript was given")
    placed = [x for x in clipped if "gap" in x["reasons"]]
    if not placed:
        return CheckResult("clipped_words", "fatal", True, "no placed gap cut takes part of a transcript word")
    head = "; ".join(f"{x['text']!r} {x['start']:.2f}-{x['end']:.2f}s, {x['cut']:.0%} cut by {'/'.join(x['reasons'])}"
                     for x in placed[:5])
    return CheckResult("clipped_words", "fatal", False,
                       f"{len(placed)} transcript words partly cut by placed gaps: {head}"
                       + (" ..." if len(placed) > 5 else ""))


def _clipped_speech(plan: Plan, f: dict) -> CheckResult:
    """Words partly cut by word-removing cuts (silence and noise excluded): a warning with the count."""
    clipped = f.get("clipped_words")
    if clipped is None:
        return CheckResult("clipped_speech", "warn", True, "not measured: no source transcript was given")
    other = [x for x in clipped if "gap" not in x["reasons"] and set(x["reasons"]) - {"silence", "noise"}]
    if not other:
        return CheckResult("clipped_speech", "warn", True, "no word-removing cut takes part of a transcript word")
    head = "; ".join(f"{x['text']!r} {x['start']:.2f}s, {x['cut']:.0%} cut by {'/'.join(x['reasons'])}"
                     for x in other[:5])
    return CheckResult("clipped_speech", "warn", False,
                       f"{len(other)} transcript words partly cut by word-removing cuts: {head}"
                       + (" ..." if len(other) > 5 else ""))



def _residual_gaps(plan: Plan, f: dict) -> CheckResult:
    runs = f["dead_air"]
    over = [r for r in runs if r["excess"] > 0]
    total = sum(r["excess"] for r in over)
    budget = f["gap_budget"]
    worst = ", ".join(f"{r['start']:.1f}-{r['end']:.1f} (+{r['excess']:.2f}s {r['boundary']})" for r in over[:8])
    return CheckResult("residual_gaps", "fatal", total <= budget,
                       f"{total:.2f}s of dead air beyond the pause caps in {len(over)} of {len(runs)} "
                       f"silence runs >= {GAP_MIN}s (budget {budget:.2f}s)"
                       + (f": {worst}" if worst else ""))


def _joins(plan: Plan, f: dict) -> CheckResult:
    grades = f["join_grades"]
    flagged = [g for g in grades if g.get("grade") != "OK"]
    detail = f"{len(grades) - len(flagged)}/{len(grades)} joins clean"
    if flagged:
        detail += "; flagged: " + "; ".join(f"{g['t']:.2f}s {g['grade']}" for g in flagged[:12])
    return CheckResult("joins", "warn", not flagged, detail)


def _join_edges(plan: Plan, f: dict) -> CheckResult:
    """A chopped voiced word at either edge of a cut, judged on the source.

    `_joins` grades the render from the seam forward; a word tail cut while
    still voiced sits BEFORE the seam and passes it. Warn only: the envelope
    cannot tell a chopped word from a loud consonant at an intended edge."""
    edges = f.get("join_edges")
    if edges is None:
        return CheckResult("join_edges", "warn", True, "not measured (no source envelope given)")
    flagged = [e for e in edges if e["flag"]]
    detail = f"{len(edges) - len(flagged)}/{len(edges)} cut edges land on quiet source audio"
    if flagged:
        detail += "; voiced edge: " + "; ".join(
            f"{e['t']:.2f}s {e['flag']} ({e['reason']})" for e in flagged[:12])
    return CheckResult("join_edges", "warn", not flagged, detail)


def _render_silence(plan: Plan, f: dict) -> CheckResult:
    """Silence measured on the rendered audio itself, not predicted from the
    plan: a botched render can mute or drop spans the plan says are speech
    (observed: an ffmpeg concat bug shipped 284 s of digital silence while
    loudness and duration checks passed). Any digital-silence run is fatal;
    quieter dead air shares the residual_gaps budget."""
    sil = f["render_silences"]
    quiet, digital = sil["quiet"], sil["digital"]
    total = sum(b - a for a, b in quiet)
    budget = f["gap_budget"]
    ok = total <= budget and not digital
    detail = (f"{total:.2f}s of silence <= {QUIET_DB:.0f} dB in {len(quiet)} runs >= "
              f"{f.get('quiet_silence_min', QUIET_SILENCE_MIN):.2f}s "
              f"(budget {budget:.2f}s)")
    if digital:
        runs = ", ".join(f"{a:.1f}-{b:.1f}" for a, b in digital[:8])
        detail += f"; DIGITAL silence <= {DIGITAL_DB:.0f} dB in {len(digital)} runs: {runs}"
    return CheckResult("render_silence", "fatal", ok, detail)


def _source_timing(plan: Plan, f: dict) -> CheckResult:
    """The render's own audio must sit where the plan says it does.

    Fatal on the offset: a mis-mapped segment ships the wrong speech while
    duration, loudness and silence all stay green. Evidence that could not be
    gathered at all is reported as a warning of its own, never as a pass.
    """
    checks = f.get("source_timing")
    if checks is None:
        return CheckResult("source_timing", "fatal", True, "not measured (no source path given)")
    eligible = sum(1 for s in plan.segments if s.length >= SOURCE_TIMING_MIN_SEGMENT)
    measured = [c for c in checks if "unmeasured" not in c]
    skipped = [c for c in checks if "unmeasured" in c]
    if not measured:
        why = f"; {len(skipped)} sampled keep(s) carried no alignable audio" if skipped else ""
        return CheckResult("source_timing", "warn", False,
                           f"not measured: {eligible} of {len(plan.segments)} kept segments are >= "
                           f"{SOURCE_TIMING_MIN_SEGMENT:.1f}s and none could be correlated{why}")
    bad = [c for c in measured if abs(c["source_offset_error_seconds"]) > SOURCE_TIMING_TOL]
    worst = max(abs(c["source_offset_error_seconds"]) for c in measured)
    detail = (f"{len(measured) - len(bad)}/{len(measured)} sampled keeps within "
              f"{SOURCE_TIMING_TOL * 1000:.0f} ms of the source (worst {worst * 1000:.0f} ms; "
              f"{len(measured)}/{eligible} eligible keeps measured"
              + (f", {len(skipped)} without alignable audio" if skipped else "") + ")")
    if bad:
        detail += "; off: " + ", ".join(
            f"{c['output_seconds']:.1f}s {c['source_offset_error_seconds'] * 1000:+.0f}ms" for c in bad[:6])
    return CheckResult("source_timing", "fatal", not bad, detail)


def _source_similarity(plan: Plan, f: dict) -> CheckResult:
    """Correlation of the same samples: a warning, not a verdict.

    A low offset error with a low correlation is usually the broadcast chain
    (compressor, loudnorm) rather than a wrong cut, so this never stops a run.
    Unmeasured samples are left to `source_timing`, which has the plan to say
    why they could not be gathered.
    """
    measured = [c for c in (f.get("source_timing") or []) if "unmeasured" not in c]
    if not measured:
        return CheckResult("source_similarity", "warn", True, "not measured")
    bad = [c for c in measured if c["correlation"] < SOURCE_CORR_MIN]
    worst = min(c["correlation"] for c in measured)
    detail = (f"{len(measured) - len(bad)}/{len(measured)} sampled keeps correlate >= {SOURCE_CORR_MIN} "
              f"with the source (worst {worst:.2f})")
    if bad:
        detail += "; weak: " + ", ".join(f"{c['output_seconds']:.1f}s {c['correlation']:.2f}" for c in bad[:6])
    return CheckResult("source_similarity", "warn", not bad, detail)


def _unclassified_voice(plan: Plan, f: dict) -> CheckResult:
    """Voiced audio no transcript word accounts for, and what the plan does with it.

    It is UNCLASSIFIED: level and voicing gates cannot tell a word from a hum,
    so this reports the measurement and the focused re-transcription's guess,
    and leaves the judgment to a person. The edge guard only rewrites mechanical
    trims, so a judged cut that removes such a span lands here.
    """
    data = f.get("unclassified_voice") or {}
    spans, removed = data.get("spans") or [], data.get("removed") or []
    if not spans:
        return CheckResult("unclassified_voice", "warn", True,
                           "no voiced audio outside the transcript")
    detail = f"{len(spans)} unclassified voiced span(s) no word covers, {len(removed)} removed by the plan"
    if removed:
        total = sum(r["removed_sec"] for r in removed)
        detail += f" ({total:.2f}s)"
        detail += "; " + "; ".join(
            f"{r['start']:.1f}s {r['removed_sec']:.2f}s by {r['reason']}"
            + (f" (re-transcription heard {r['probe']!r})" if r.get("probe") else "")
            for r in removed[:6])
    return CheckResult("unclassified_voice", "warn", not removed, detail)


def _bit_depth(pix_fmt: str) -> int:
    """Bits per sample from an ffmpeg pixel format name; 8 when it names none.

    Matches the depth position only, so `nv12` (a 8-bit name that ends in 12) is
    not read as 12-bit while `p010le`, `yuv420p10le` and `x2rgb10le` are 10.
    """
    m = re.search(r"p0?(9|10|12|14|16)(?:le|be)?$|(9|10|12|14|16)(?:le|be)$", pix_fmt or "")
    return int(m.group(1) or m.group(2)) if m else 8


def _color(plan: Plan, f: dict) -> CheckResult:
    """The render must not silently convert the source's colour.

    An HDR source rendered to untagged SDR reads as flat and was rejected by
    eye while every duration and loudness check stayed green. Carrying the right
    tags on an 8-bit stream is the same conversion wearing the tags, so depth is
    part of the comparison.
    """
    c = f.get("color")
    if not c:
        return CheckResult("color", "fatal", True, "not measured (no source path)")
    src, ren = c["source"], c["render"]
    ren_bits = _bit_depth(ren["pix_fmt"])
    src_desc = f"{src['transfer']}/{src['primaries']}/{src['matrix']}/{src['pix_fmt']}"
    ren_desc = f"{ren['transfer']}/{ren['primaries']}/{ren['matrix']}/{ren['pix_fmt']}"
    ok = src["hdr"] == ren["hdr"] and (not src["hdr"] or (
        ren["transfer"] == src["transfer"] and ren["primaries"] == src["primaries"]
        and ren["matrix"] == src["matrix"] and ren["range"] == src["range"] and ren_bits >= 10))
    if ok:
        detail = f"render {ren_desc} matches the source's {src_desc}"
    elif src["hdr"]:
        why = ("the render carries the source's tags on a "
               f"{ren_bits}-bit stream") if ren["hdr"] and ren_bits < 10 else "the picture was converted"
        detail = f"source is HDR ({src_desc}) but the render is {ren_desc}: {why}"
    else:
        detail = f"source is SDR ({src_desc}) but the render is {ren_desc}"
    return CheckResult("color", "fatal", ok, detail)


def _audio_stream(plan: Plan, f: dict) -> CheckResult:
    """An audio-only render must be the 48 kHz 16-bit PCM WAV the renderer promises."""
    s = f.get("audio_stream")
    if not s:
        return CheckResult("audio_stream", "fatal", False, "no audio stream facts measured")
    ok = s["codec"] == "pcm_s16le" and s["sample_rate"] == AUDIO_RATE and s["duration"] > 0
    return CheckResult("audio_stream", "fatal", ok,
                       f"{s['codec']} {s['sample_rate']} Hz, {s['channels']} ch, {s['duration']:.2f}s "
                       f"(pcm_s16le {AUDIO_RATE} Hz required)")


def _resolution(plan: Plan, f: dict) -> CheckResult:
    """The full render keeps the source's native frame; only the demo tier scales."""
    fr = f.get("frame")
    if not fr:
        return CheckResult("resolution", "fatal", True, "not measured (no source path)")
    (sw, sh), (rw, rh) = fr["source"], fr["render"]
    if fr.get("scale"):
        return CheckResult("resolution", "info", True,
                           f"{rw}x{rh}: demo tier scales to {fr['scale']} (source {sw}x{sh})")
    return CheckResult("resolution", "fatal", (sw, sh) == (rw, rh),
                       f"render {rw}x{rh} vs source {sw}x{sh} as displayed")


def visuals_resolution(base: str, out: str) -> CheckResult:
    """The overlay pass is what ships: it must keep the cut's frame size."""
    (bw, bh), (ow, oh) = display_size(base), display_size(out)
    return CheckResult("visuals_resolution", "fatal", (bw, bh) == (ow, oh),
                       f"overlay render {ow}x{oh} vs cut {bw}x{bh}")


def _word_cuts(plan: Plan) -> Plan:
    """The plan without intra-token restarts: they cut inside a word on purpose, so they are not clipped words."""
    return replace(plan, cuts=[c for c in plan.cuts if not is_intra_restart(c)])


def residual_findings(f: dict) -> list[dict] | None:
    """The render's disfluency findings the output pass did not judge keep; None when the render was not transcribed.

    A keep suppresses only its own occurrence (same text, same main-source instant): a second "very, very" the judge
    never saw, or one that cannot be placed in the source, stays listed.
    """
    found = f.get("disfluency_findings")
    if found is None:
        return None
    keep = f.get("disfluency_keep") or []
    return [x for x in found if not any(same_occurrence(x, k) for k in keep)]


def _residual_disfluencies(plan: Plan, f: dict) -> CheckResult:
    """Fillers, repeated words and restarts the finished render still contains that the output pass did not cut.

    A warning, not a failure: deliberate emphasis looks the same to the scan, and the judge keeps it.
    """
    found = residual_findings(f)
    if found is None:
        return CheckResult("residual_disfluencies", "info", True,
                           "not measured: the render was not transcribed (output ASR disabled)")
    if not found:
        return CheckResult("residual_disfluencies", "warn", True,
                           "no filler, repeated word or restart left in the render's transcript")
    head = "; ".join(f"{x['heard']!r} at {x['start']:.2f}s ({x['kind']})" for x in found[:5])
    return CheckResult("residual_disfluencies", "warn", False,
                       f"{len(found)} disfluencies left in the render, not judged keep: {head}"
                       + (" ..." if len(found) > 5 else ""))


CHECKS: list[Callable[[Plan, dict], CheckResult]] = [
    _duration, _av_sync, _loudness, _cold_open, _judged_removals, _clipped_words, _clipped_speech, _residual_gaps,
    _residual_disfluencies, _render_silence,
    _joins, _join_edges, _source_timing, _source_similarity, _unclassified_voice, _color, _resolution,
]

AUDIO_CHECKS: list[Callable[[Plan, dict], CheckResult]] = [
    _audio_stream, _duration, _loudness, _cold_open, _judged_removals, _clipped_words, _clipped_speech,
    _residual_gaps, _residual_disfluencies,
    _render_silence, _joins, _join_edges, _source_timing, _source_similarity, _unclassified_voice,
]


def _output_asr(plan: Plan, f: dict) -> CheckResult:
    """Added only when the render was not transcribed: the transcript facts are then expectations."""
    return CheckResult("output_asr", "info", True,
                       "the render was NOT transcribed: word, pause and cold-open facts come from the source "
                       "words projected through the plan (an expectation, not an independent measurement)")


def run_checks(plan: Plan, facts: dict) -> list[CheckResult]:
    """The video table, or the audio table (no picture: no AV sync, colour or resolution)."""
    table = AUDIO_CHECKS if facts.get("mode") == "audio" else CHECKS
    results = [check(plan, facts) for check in table]
    if facts.get("output_asr") == "source-projection":
        results.append(_output_asr(plan, facts))
    return results


# ------------------------------------------------------------- measuring


def measure_lufs(path: str) -> float:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-filter_complex", "ebur128=peak=none", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    matches = re.findall(r"I:\s*(-?[\d.]+|-inf)\s*LUFS", proc.stderr)
    if not matches:
        raise RuntimeError(f"ebur128 produced no integrated loudness for {path}")
    return float(matches[-1])


def render_silences(wav_path: str, quiet_min: float = QUIET_SILENCE_MIN) -> dict[str, list[tuple[float, float]]]:
    """Silence runs measured on the rendered audio with ffmpeg silencedetect,
    at two tiers: quiet (dead air) and digital (muting). An unterminated
    trailing run (silence to EOF) is closed at the wav duration."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", wav_path],
        capture_output=True, text=True, check=True,
    ).stdout
    duration = float(json.loads(probe)["format"]["duration"])

    def runs(db: float, min_len: float) -> list[tuple[float, float]]:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", wav_path,
             "-af", f"aformat=sample_fmts=flt,silencedetect=n={db}dB:d={min_len}", "-f", "null", "-"],
            capture_output=True, text=True,
        )
        out: list[tuple[float, float]] = []
        start: float | None = None
        for line in proc.stderr.splitlines():
            m = re.search(r"silence_start:\s*(-?[\d.]+)", line)
            if m:
                start = float(m.group(1))
                continue
            m = re.search(r"silence_end:\s*(-?[\d.]+)", line)
            if m and start is not None:
                out.append((start, float(m.group(1))))
                start = None
        if start is not None:
            out.append((start, duration))
        return out

    return {
        "quiet": runs(QUIET_DB, quiet_min),
        "digital": runs(DIGITAL_DB, DIGITAL_SILENCE_MIN),
    }


def source_timing_indices(
    plan: Plan, *, min_segment: float = SOURCE_TIMING_MIN_SEGMENT, max_samples: int = SOURCE_TIMING_MAX_SAMPLES,
) -> list[int]:
    """Indices of the kept segments to sample: every one long enough for a
    centred window, thinned evenly when there are more than `max_samples`.

    Even coverage rather than longest-first: a wrong mapping anywhere in the EDL
    should have a chance of falling inside a sample.
    """
    eligible = [i for i, s in enumerate(plan.segments) if s.length >= min_segment]
    if not eligible or max_samples <= 0:
        return []
    if len(eligible) <= max_samples:
        return eligible
    stride = len(eligible) / max_samples
    return [eligible[min(len(eligible) - 1, int(i * stride))] for i in range(max_samples)]


def measure_source_timing(
    plan: Plan, render_path: str, *, min_segment: float = SOURCE_TIMING_MIN_SEGMENT,
    window: float = SOURCE_TIMING_WINDOW, max_samples: int = SOURCE_TIMING_MAX_SAMPLES,
    source_pad: float = SOURCE_TIMING_PAD, sr: int = SOURCE_TIMING_SR,
) -> list[dict]:
    """Cross-correlate interior speech of the render against the source it was
    cut from: how far the rendered audio sits from where the plan put it, and
    how much it still looks like the original.

    Every segment source is measured, and a short keep pairs half of its own
    length instead of the preferred window, so a plan of short keeps still
    yields evidence. Mono f32 at `sr` is enough for a peak alignment. The source
    window is padded by `source_pad` on each side, so a mis-mapping up to that
    is measurable rather than clipped by the window. A window that will not
    decode is left out (and the check says so) instead of silently counted as
    clean.
    """
    checks: list[dict] = []
    sampled = set(source_timing_indices(plan, min_segment=min_segment, max_samples=max_samples))
    offset = 0.0
    for i, segment in enumerate(plan.segments):
        if i in sampled:
            sample = min(window, segment.length / 2)
            t = segment.length / 2 - sample / 2
            rendered = _decode_pcm(render_path, offset + t, sample, sr)
            source = plan.sources[segment.source]
            original = _decode_pcm(source.path, segment.start + t - source_pad, sample + 2 * source_pad, sr)
            entry: dict = {"output_seconds": round(offset + t, 3), "source": segment.source,
                           "window_sec": round(sample, 3)}
            level = _rms_db(rendered)
            if level < SOURCE_TIMING_MIN_RMS_DB:
                entry["unmeasured"] = "digital silence" if level == -np.inf else f"{level:.0f} dBFS window"
            elif len(original) <= len(rendered):
                entry["unmeasured"] = "source window could not be decoded"
            else:
                rendered = rendered - rendered.mean()
                original = original - original.mean()
                peak = int(np.argmax(np.correlate(original, rendered, mode="valid")))
                aligned = original[peak:peak + len(rendered)]
                norm = float(np.linalg.norm(aligned) * np.linalg.norm(rendered))
                entry["source_offset_error_seconds"] = round(peak / sr - source_pad, 4)
                entry["correlation"] = round(float(np.dot(aligned, rendered) / norm) if norm else 0.0, 3)
            checks.append(entry)
        offset += segment.length
    return checks


def _rms_db(pcm: np.ndarray) -> float:
    """Window level in dBFS (ffmpeg f32 is normalised to +-1.0); -inf when empty."""
    if len(pcm) == 0:
        return -np.inf
    rms = float(np.sqrt(np.mean(np.square(pcm))))
    return 20 * float(np.log10(rms)) if rms > 0 else -np.inf


def _decode_pcm(path: str, start: float, duration: float, sr: int) -> np.ndarray:
    """Mono f32 PCM for one window of a file (empty when it cannot be decoded)."""
    if duration <= 0:
        return np.zeros(0)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{start:.3f}", "-i", path,
         "-t", f"{duration:.3f}", "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True,
    )
    if raw.returncode != 0:
        return np.zeros(0)
    return np.frombuffer(raw.stdout, dtype=np.float32).astype(float)


def unclassified_voice_facts(
    plan: Plan, voices: Sequence[tuple[float, float]], probes: Sequence[dict] = (),
) -> dict:
    """Which unclassified voiced spans the plan keeps, and which it removes.

    A span the plan keeps is already safe; a span it REMOVES is the case worth
    printing, because only a person can say whether that was intended.
    """
    probe_text = {(round(p["start"], 3), round(p["end"], 3)): p.get("text", "") for p in probes}
    removed: list[dict] = []
    for start, end in voices:
        for cut in plan.cuts:
            lo, hi = max(start, cut.start), min(end, cut.end)
            if hi - lo >= UNCLASSIFIED_REPORT_MIN:
                removed.append({
                    "start": round(start, 3), "end": round(end, 3), "reason": cut.reason,
                    "removed_sec": round(hi - lo, 3),
                    "probe": probe_text.get((round(start, 3), round(end, 3)), ""),
                })
                break
    return {
        "spans": [{"start": round(s, 3), "end": round(e, 3),
                   "probe": probe_text.get((round(s, 3), round(e, 3)), "")} for s, e in voices],
        "removed": sorted(removed, key=lambda r: r["start"]),
    }


def _spread(values: Sequence[float]) -> dict:
    """p25 / p50 / p75 / max of a list of seconds, rounded to milliseconds."""
    p25, p50, p75 = (float(x) for x in np.percentile(values, [25, 50, 75]))
    return {"p25": round(p25, 3), "p50": round(p50, 3), "p75": round(p75, 3), "max": round(float(max(values)), 3)}


def pacing_facts(
    plan: Plan, source_words: Sequence, silences: Sequence[tuple[float, float]],
    edges: Sequence[dict] | None, config: Config,
) -> dict:
    """How long the kept pauses are, against the same pauses in the source.

    A pause is what the policy trims: an interior VAD silence (`silences`) of
    more than `pause_above`, classified by the last word that starts before it
    and by its own length. Word-stamp gaps are not pauses: a hum or breath
    between two words fills one with speech-level sound the policy rightly keeps.
    Both words around the silence must survive whole in the main source. `pauses`
    holds, per boundary type, the source silence length and the part of it the
    output keeps. `seam_quiet` is the median speechless audio a seam keeps, per
    cut reason (from `source_edge_facts`). Facts only: nothing here is a verdict.
    """
    kept: list[tuple[float, float, float]] = []  # (source start, source end, output start), main only
    offset = 0.0
    for seg in plan.segments:
        if seg.source == plan.main:
            kept.append((seg.start, seg.end, offset))
        offset += seg.length
    starts = [k[0] for k in kept]
    word_starts = [w.start for w in source_words]

    def survives(w) -> bool:
        i = bisect_right(starts, w.start + 1e-6) - 1
        return i >= 0 and w.end <= kept[i][1] + 1e-6

    src: dict[str, list[float]] = {}
    out: dict[str, list[float]] = {}
    for s, e in silences:
        dur = e - s
        if s <= 0.0 or dur <= config.pause_above:
            continue
        i = bisect_left(word_starts, s) - 1
        if i < 0 or i + 1 >= len(source_words):
            continue
        a, b = source_words[i], source_words[i + 1]
        if not (survives(a) and survives(b)):
            continue
        boundary = pause_boundary(a.text, dur, config)
        src.setdefault(boundary, []).append(dur)
        out.setdefault(boundary, []).append(sum(max(0.0, min(e, ke) - max(s, ks)) for ks, ke, _ in kept))
    pauses = {
        b: {"count": len(src[b]), "source": _spread(src[b]), "output": _spread(out[b])}
        for b in ("mid", "clause", "sentence", "topic") if b in src
    }
    by_reason: dict[str, list[float]] = {}
    for e in edges or []:
        by_reason.setdefault(e["reason"], []).append(e["quiet"])
    seam_quiet = {r: {"count": len(v), "median": round(float(np.median(v)), 3)} for r, v in sorted(by_reason.items())}
    return {"pause_above": config.pause_above, "pauses": pauses, "seam_quiet": seam_quiet}


def color_facts(source_path: str, render_path: str) -> dict:
    """The source's and the render's colour facts, as verify._color compares them."""
    src, ren = probe_color(source_path), probe_color(render_path)
    return {
        "source": {"primaries": src.primaries, "transfer": src.transfer, "matrix": src.matrix,
                   "range": src.range, "pix_fmt": src.pix_fmt, "hdr": src.hdr},
        "render": {"primaries": ren.primaries, "transfer": ren.transfer, "matrix": ren.matrix,
                   "range": ren.range, "pix_fmt": ren.pix_fmt, "hdr": ren.hdr},
    }


def display_size(path: str) -> tuple[int, int]:
    """Width and height as displayed: a phone's rotation tag swaps the stored axes."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height:stream_side_data=rotation", "-of", "json", path],
        capture_output=True, text=True, check=True,
    ).stdout
    s = json.loads(out)["streams"][0]
    w, h = int(s["width"]), int(s["height"])
    rot = next((int(d["rotation"]) for d in s.get("side_data_list", []) if "rotation" in d), 0)
    return (h, w) if abs(rot) % 180 == 90 else (w, h)



def audio_stream_facts(path: str) -> dict:
    """Codec, rate, channels and duration of a file's first audio stream."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_name,sample_rate,channels,duration:format=duration", "-of", "json", path],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)
    streams = info.get("streams") or []
    if not streams:
        raise RuntimeError(f"{path}: no audio stream")
    s = streams[0]
    return {"codec": s.get("codec_name"), "sample_rate": int(s.get("sample_rate") or 0),
            "channels": int(s.get("channels") or 0),
            "duration": float(s.get("duration") or info.get("format", {}).get("duration") or 0.0)}


def project_words(plan: Plan, words: Sequence) -> list[Word]:
    """The source words that survive whole in the main source, on the output timeline.

    This is the plan's EXPECTED transcript, not a measurement of the render: a
    word cut through, or spanning a join, does not survive.
    """
    out: list[Word] = []
    for w in words:
        a = plan.source_to_output(w.start + 1e-6)
        b = plan.source_to_output(w.end - 1e-6)
        if a is None or b is None or abs((b - a) - (w.end - w.start - 2e-6)) > 1e-3:
            continue
        out.append(Word(w.text, a - 1e-6, b + 1e-6, getattr(w, "probability", 1.0), getattr(w, "speaker", None)))
    return out


def collect_facts(
    plan: Plan, render_path: str, *, transcriber: Transcriber | None = None, cache: Cache, workdir: Path,
    source_duration: float, source_silences: list[tuple[float, float]], lufs_target: float = -16.0,
    source_path: str | None = None, voices: Sequence[tuple[float, float]] = (),
    speech_probes: Sequence[dict] = (), config: Config | None = None,
    scale: str | None = None,
    env: Envelope | None = None, floor: float | None = None, source_words: Sequence = (),
    audio_only: bool = False, output_asr: bool = True,
    judged: Sequence[tuple[float, float, str]] | None = None,
) -> dict:
    """Measure the render: durations, loudness, transcript (cached), join grades,
    the dead air that survives the plan, and - when the source path is known -
    where the rendered speech actually sits against that source and what colour
    the render produced. `env`/`floor` are the source envelope and noise floor
    the analysis already computed (edge check and seam quiet); `source_words`
    are its refined words (pacing). Without `env` those facts are None.

    `audio_only` measures a WAV: no picture, so no video/colour/frame facts.
    `output_asr=False` skips transcribing the render; the transcript is then the
    source words projected through the plan, recorded as `output_asr:
    "source-projection"` because it is an expectation, not a measurement.
    `judged` are the judges' word-removing decisions (start, end, reason): the
    `judged_removals` check fails when the plan keeps a word any of them removed.
    `disfluency_findings` is every filler, repeat and restart the render's transcript shows (None when not
    transcribed); the pipeline adds `disfluency_keep` once the output pass has judged its rounds.
    A transcriber that replays one fixed source response is refused as an
    output ASR: it would hand back the source's words for the edited audio."""
    config = config or Config()
    if audio_only:
        v, a = None, audio_stream_facts(render_path)["duration"]
    else:
        v, a = av_durations(render_path)
    wav, sha = prepare_audio(render_path, workdir, force=True)
    if output_asr:
        if transcriber is None:
            raise ValueError("output ASR requested without a transcriber")
        if getattr(transcriber, "replays_fixed_source", False):
            raise RuntimeError(
                f"transcriber {transcriber.name!r} replays one saved source response and cannot "
                "transcribe the edited output; use a live transcriber or disable output ASR (--no-output-asr)")
        transcript = transcribe_cached(cache, sha, transcriber, wav)
    else:
        transcript = Transcript("source-projection", "1", plan.output_duration, project_words(plan, source_words))
    joins = [t for t in plan.joins() if t > 0.05]
    grades = [check_join(render_path, t) for t in joins]
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", render_path],
        capture_output=True, text=True, check=True,
    ).stdout
    words = transcript.words
    runs = _dead_air_runs(plan, source_silences, GAP_MIN)
    edges = source_edge_facts(plan, env, floor, config.quiet_above_floor_db) if env is not None else None
    picture = bool(source_path) and not audio_only
    return {
        "mode": "audio" if audio_only else "video",
        "output_asr": "independent" if output_asr else "source-projection",
        "audio_stream": audio_stream_facts(render_path) if audio_only else None,
        "duration": float(json.loads(out)["format"]["duration"]),
        "video_duration": v, "audio_duration": a,
        "lufs": measure_lufs(render_path), "lufs_target": lufs_target,
        "words": words,
        "join_grades": grades,
        # None = no source envelope was given (a hand render, unit facts).
        "join_edges": edges,
        "pacing": pacing_facts(plan, source_words, source_silences, edges, config) if env is not None else None,
        # Pauses the policy keeps on purpose are not dead air (see dead_air_excess).
        # The budget still assumes most silence was tightened; a cook that keeps
        # far more must raise it via env, e.g. 0.15.
        "gap_budget": max(1.0, float(os.environ.get("TALKINGHEAD_GAP_BUDGET_FRACTION", 0.005)) * source_duration),
        "dead_air": dead_air_excess([(a, b) for a, b, _ in runs], words, config, raw=[r for _, _, r in runs]),
        "quiet_silence_min": quiet_silence_min(config),
        "render_silences": render_silences(wav, quiet_silence_min(config)),
        "render_transcript": transcript.to_dict(),
        # None = the render was not transcribed, so nothing in it can be scanned.
        "disfluency_findings": None if not output_asr else [
            {**f.to_dict(), "source": finding_source(plan, f)} for f in scan_disfluencies(words)],
        # None = the evidence was never asked for (a hand render, unit facts);
        # [] = asked for and unavailable, which the check reports as a warning.
        "source_timing": measure_source_timing(plan, render_path) if source_path else None,
        "unclassified_voice": unclassified_voice_facts(plan, voices, speech_probes),
        "judged_leaks": judged_leaks(plan, source_words, judged) if judged is not None else None,
        "clipped_words": clipped_words(_word_cuts(plan), source_words) if source_words else None,
        "color": color_facts(source_path, render_path) if picture else None,
        # `scale` is the render's deliberate downscale (demo tier); None = native.
        "frame": {"source": list(display_size(source_path)), "render": list(display_size(render_path)),
                  "scale": scale} if picture else None,
    }
