"""From a video file to the run context: extract audio once, transcribe once
(cached by audio hash), analyse once, refine word bounds, run the acoustic
detectors on the shared envelope. Nothing here consults an LLM.

Two results answer what the transcript cannot: `voices` (voiced, speech-level
spans no word covers, which only the waveform can see) and `speech_probes` (a
focused re-transcription of each, with the other backend, so the report can name
what was said instead of only measuring it). The spans are UNCLASSIFIED audio,
not proven words: a speech model may demote one to non-speech, nothing promotes
one."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import __version__
from .audio import FRAME_SEC, envelope, filter_masked, noise_floor, refine_word_bounds, unclassified_voice
from .bursts import detect_voiced_bursts
from .config import Config
from .context import Context
from .judge import Judge, LLMCache
from .noise import VAD_FRAME_SEC, _read_wav, detect_mouth_noises, silero_speech_mask
from .store import Cache, RunDir, sha256_file
from .timeline import MIN_CUT, Cut, SourceInfo
from .transcribe import (
    AudioEvent,
    Transcriber,
    Word,
    audio_duration,
    check_av_mismatch,
    detect_silences,
    find_fluidaudio,
    prepare_audio,
    probe_words,
    transcribe_cached,
)

# v2: detect_silences threshold anchored between noise/speech clusters (was
# 2x p15, which collapsed on long tracks).
ANALYSIS_VERSION = 2

# v1: laughter and unrecognised tags are protected whole (grown over quiet
# edges); sigh/breath/mouth-noise tags are cut candidates. Part of every stage
# memo key through the transcript provenance whenever a run has events.
EVENT_POLICY_VERSION = 1
_LAUGH_STEMS = ("laugh", "chuckl", "giggl", "snicker", "chortl", "cackl", "guffaw", "titter", "haha", "hehe")
_NOISE_STEMS = ("sigh", "breath", "inhale", "exhale", "smack", "tongue", "throat", "cough", "sniff",
                "swallow", "gulp")
# A protected event grows over its quieter onset/decay: frames this far above
# the floor, bridging dips up to EVENT_GAP_SEC, for at most EVENT_GROW_MAX_SEC,
# and never into a neighbouring word.
EVENT_EDGE_ABOVE_FLOOR_DB = 3.0
EVENT_GAP_SEC = 0.15
EVENT_GROW_MAX_SEC = 0.6


def classify_event(text: str) -> str:
    """Event kind: laughter, noise (a known cuttable sound) or unknown (kept whole).

    Laughter wins over noise ("laughs and sighs"): only a recognised sound is
    ever cut, and an unfamiliar tag is never assumed to be disposable.
    """
    t = text.lower()
    if any(s in t for s in _LAUGH_STEMS):
        return "laughter"
    if any(s in t for s in _NOISE_STEMS):
        return "noise"
    return "unknown"


def _reach(db: np.ndarray, threshold: float, frame_sec: float, t: float, limit: float, step: int) -> int:
    """Furthest frame from `t` toward `limit` still above threshold, bridging short dips."""
    i, last = int(t / frame_sec), int(t / frame_sec)
    stop, allowed, quiet = int(limit / frame_sec), int(EVENT_GAP_SEC / frame_sec), 0
    j = i + step
    while 0 <= j < len(db) and (j - stop) * step <= 0:
        if db[j] > threshold:
            last, quiet = j, 0
        else:
            quiet += 1
            if quiet > allowed:
                break
        j += step
    return last


def _grow_span(env, floor: float, start: float, end: float, words) -> tuple[float, float]:
    prev_end = max((w.end for w in words if w.end <= start), default=0.0)
    next_start = min((w.start for w in words if w.start >= end), default=env.duration)
    lo = min(start, max(prev_end, start - EVENT_GROW_MAX_SEC))
    hi = max(end, min(next_start, end + EVENT_GROW_MAX_SEC))
    thr = floor + EVENT_EDGE_ABOVE_FLOOR_DB
    new_start = _reach(env.db, thr, env.frame_sec, start, lo, -1) * env.frame_sec
    new_end = (_reach(env.db, thr, env.frame_sec, end, hi, +1) + 1) * env.frame_sec
    return min(start, new_start), max(end, min(new_end, hi))


def _union(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _overlaps(span: tuple[float, float], others: list[tuple[float, float]]) -> bool:
    return any(span[0] < e and s < span[1] for s, e in others)


def _subtract(start: float, end: float, blocked: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pieces, cursor = [], start
    for s, e in _union(blocked):
        if e <= cursor or s >= end:
            continue
        if s - cursor > MIN_CUT:
            pieces.append((cursor, s))
        cursor = max(cursor, e)
    if end - cursor > MIN_CUT:
        pieces.append((cursor, end))
    return pieces


def events_key(events: list[AudioEvent]) -> str:
    """Digest of the tagged events under the current policy; re-keys decisions on either."""
    rows = [(classify_event(e.text), e.text, round(e.start, 3), round(e.end, 3)) for e in events]
    return hashlib.sha256(json.dumps([EVENT_POLICY_VERSION, rows]).encode()).hexdigest()[:16]


def apply_event_policy(events: list[AudioEvent], words, env, floor: float
                       ) -> tuple[list[tuple[float, float]], list[Cut], list[dict]]:
    """(protected spans, noise-candidate cuts, per-event report rows).

    Protected spans never overlap a candidate: a candidate is trimmed around
    them and around every spoken word.
    """
    rows: list[dict] = []
    grown: list[tuple[float, float]] = []
    for e in events:
        kind = classify_event(e.text)
        row = {"text": e.text, "start": round(e.start, 3), "end": round(e.end, 3), "kind": kind}
        if kind != "noise":
            s, t = _grow_span(env, floor, e.start, e.end, words)
            grown.append((s, t))
            row["protected"] = [round(s, 3), round(t, 3)]
        rows.append(row)
    protected = _union(grown)
    blocked = protected + [(w.start, w.end) for w in words]
    cuts = [Cut(s, t, "noise", e.text, {"source": "scribe audio_event"})
            for e in events if classify_event(e.text) == "noise"
            for s, t in _subtract(e.start, e.end, blocked)]
    return protected, cuts, rows


def analyze(
    video: str,
    run_dir: str | Path,
    config: Config,
    transcriber: Transcriber,
    judge: Judge,
    *,
    cache: Cache,
    guidelines: str,
    topic: str | None = None,
    preferences: str = "",
    allow_av_mismatch: bool = False,
    audio_only: bool = False,
) -> Context:
    """`video` is the recording; with `audio_only` it is an audio file and there
    is no video track to check or to synthesise."""
    run = RunDir(Path(run_dir)).ensure()
    if audio_only:
        v_dur, a_dur = None, audio_duration(video)
    else:
        v_dur, a_dur = check_av_mismatch(video, tolerance=config.av_mismatch_tolerance, allow=allow_av_mismatch)
    wav, audio_sha = prepare_audio(video, run.root / "audio")
    transcript = transcribe_cached(cache, audio_sha, transcriber, wav)

    samples, sr = _read_wav(wav)
    x = np.frombuffer(samples.tobytes(), dtype=np.int16)
    env = envelope(x, sr, FRAME_SEC)
    floor = noise_floor(env.db)
    if v_dur is not None and abs(v_dur - a_dur) > config.av_mismatch_tolerance:
        duration = min(v_dur, a_dur)
    else:
        duration = transcript.duration
    duration = float(duration or len(x) / sr)

    silences = [tuple(s) for s in cache.get_or_compute(
        audio_sha, "silences", {"frame_sec": 0.02}, ANALYSIS_VERSION, lambda: detect_silences(wav)
    )]
    words = refine_word_bounds(transcript.words, env, floor)
    as_words = [Word(w.text, w.start, w.end, 1.0) for w in words]
    env_lists = (env.db.tolist(), env.zcr.tolist())
    # Detector results depend on the (refined) words they were given, so the
    # cache key carries a digest of them, not just the backend name.
    words_key = {"transcript": transcript.backend, "words": hashlib.sha256(
        json.dumps([(w.text, round(w.start, 3), round(w.end, 3)) for w in as_words]).encode()).hexdigest()[:16]}

    # Native sound tags: laughter and unknown tags are protected whole; known
    # sigh/breath tags become cut candidates. Detector output is filtered AFTER
    # the cache (events change what it may claim, not what it measured).
    protected, event_cuts, event_rows = apply_event_policy(transcript.events, words, env, floor)
    ev_key = events_key(transcript.events) if transcript.events else None

    moans = [tuple(m) for m in cache.get_or_compute(
        audio_sha, "moans", words_key, ANALYSIS_VERSION,
        lambda: [(c.start, c.end) for c in detect_voiced_bursts(x, as_words, sample_rate=sr, silences=silences, envelope=env_lists)],
    )]
    moans = [m for m in moans if not _overlaps(m, protected)]
    detector_noise = [Cut(**c) for c in cache.get_or_compute(
        audio_sha, "mouth_noises", words_key, ANALYSIS_VERSION,
        lambda: [{"start": c.start, "end": c.end, "reason": "noise", "text": c.text}
                 for c in detect_mouth_noises(x, as_words, sample_rate=sr, envelope=env_lists)],
    )]
    noise_cuts = [c for c in detector_noise if not _overlaps((c.start, c.end), protected)] + event_cuts
    # Waveform arbitration: voiced speech-level audio no word covers. Moans,
    # mouth noises and protected events are already claimed by their own
    # passes, so they are excluded here and keep their existing treatment.
    claimed = [tuple(m) for m in moans] + [(c.start, c.end) for c in noise_cuts] + protected
    voices_key = words_key if ev_key is None else {**words_key, "events": ev_key}
    voices = [tuple(v) for v in cache.get_or_compute(
        audio_sha, "unclassified_voice", voices_key, ANALYSIS_VERSION,
        lambda: unclassified_voice(env, floor, words, claimed),
    )]
    masked: int | None = None
    if voices:
        # A speech model is the ONLY thing allowed to drop a guarded span, and it
        # is asked only when there is something to confirm: the pass costs a
        # model load, and most recordings have no candidate. None (no model on
        # this machine) leaves every span guarded; `masked` is the count the
        # model demoted to non-speech, for the report.
        mask = cache.get_or_compute(
            audio_sha, "speech_mask", {"frame_sec": VAD_FRAME_SEC}, ANALYSIS_VERSION,
            lambda: silero_speech_mask(x, sr),
        )
        if mask is not None:
            kept = filter_masked(voices, mask)
            masked = len(voices) - len(kept)
            voices = [tuple(v) for v in kept]
    speech_probes = probe_unclassified_voice(cache, audio_sha, wav, transcriber, voices, config)

    source = SourceInfo(id="main", path=str(Path(video).resolve()), sha256=sha256_file(video), duration=duration)
    transcript_info = {"backend": transcript.backend, "version": transcript.version, "words": len(words)}
    if ev_key is not None:
        transcript_info["events"] = ev_key
    ctx = Context(
        config=config, source=source, words=words, env=env, floor=floor, silences=silences, moans=moans,
        judge=judge, llm_cache=LLMCache(run.llm_dir), guidelines=guidelines, topic=topic, preferences=preferences,
        decisions=run.read_json(run.decisions, default={}),
        noise_cuts=noise_cuts, voices=voices, speech_probes=speech_probes, protected_spans=protected,
        provenance={
            "talkinghead": __version__, "analysis_version": ANALYSIS_VERSION,
            "audio_sha256": audio_sha, "video_sha256": source.sha256, "audio_only": audio_only,
            "transcript": transcript_info,
            "floor_dbfs": round(floor, 2), "video_duration": v_dur, "audio_duration": a_dur,
            "unclassified_voice": {"spans": len(voices), "masked_non_speech": masked},
            "audio_events": event_rows,
            "config": config.to_dict(),
        },
    )
    return ctx


# v1: first version; bump when the probe window or the engine choice changes.
SPEECH_PROBE_VERSION = 1
# The probe window is widened past the span so the engines see the word edges
# they need; the span itself is what is reported.
PROBE_PAD_SEC = 0.15


def _probe_backend(transcriber_name: str) -> str | None:
    """The other ASR engine, when this machine has it.

    A second opinion from the engine that already missed the span is worth less.
    Parakeet is optional, so its absence degrades to the run's own engine.
    """
    if transcriber_name not in ("parakeet", "whisper"):
        return None
    if transcriber_name == "whisper":
        return "parakeet" if find_fluidaudio() else "whisper"
    return "whisper"


def probe_unclassified_voice(cache: Cache, audio_sha: str, wav: str, transcriber: Transcriber,
                             voices: list[tuple[float, float]], config: Config) -> list[dict]:
    """Re-transcribe each unclassified voice span once, longest first.

    Bounded by `speech_probe_max` and cached on the audio, so the cost is paid
    once per recording. Only a SUCCESSFUL decode is cached: a missing binary or
    a failed ffmpeg slice is a fact about this moment, not about the recording,
    and caching it as "heard nothing" would turn a transient failure into
    permanent evidence. The text is for the report; the guard that keeps the
    audio is acoustic.
    """
    backend = _probe_backend(getattr(transcriber, "name", ""))
    if not config.speech_probe or backend is None or not voices:
        return []
    spans = sorted(voices, key=lambda s: s[1] - s[0], reverse=True)[: config.speech_probe_max]
    out: list[dict] = []
    for start, end in spans:
        a, b = max(0.0, start - PROBE_PAD_SEC), end + PROBE_PAD_SEC
        params = {"start": round(a, 2), "end": round(b, 2), "backend": backend}
        text = cache.get(audio_sha, "speech_probe", params, SPEECH_PROBE_VERSION)
        if text is None:
            text = probe_words(wav, a, b, backend)
            if text is not None:
                cache.put(audio_sha, "speech_probe", params, SPEECH_PROBE_VERSION, text)
        out.append({"start": round(start, 3), "end": round(end, 3), "backend": backend,
                    "text": text or "", "measured": text is not None})
    return out
