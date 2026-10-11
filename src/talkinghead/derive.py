"""Derive the list of CutSpans to remove from a transcript.

Reads the flat word list produced by transcribe.py and proposes cuts for:
  - silence: pauses longer than pause_above are TRIMMED, centred, to a kept
    length that grows with the pause and is capped by its boundary type
    (mid-sentence < clause < sentence < topic); fully removed at the video edges
  - filler:  hesitation tokens (uh/um/...) whose word span is long enough to
    disrupt flow (>= min_filler_dur); cut edges snap to nearby silence
  - restart: an abandoned sentence restated by the next one (high similarity)

Non-obvious decisions:
  - Silence cuts come from waveform VAD (silences=), never whisper word gaps:
    whisper timestamps drift and drop quiet trailing utterances. The VAD path
    is the only one the CLI uses; the word-gap path is a test-only fallback.
  - Pauses are TRIMMED, not removed: a short quiet sliver is kept at each edge
    of a long pause, preserving natural rhythm and avoiding the harsh
    full-splice sound. Leading/trailing silence is removed outright.
  - Fillers are cut only when the ASR emitted them as tokens (Parakeet does;
    whisper mostly swallows them) AND the token is long enough to matter
    (min_filler_dur). Short fillers stay: cutting them splices mid-flow and
    sounds worse than leaving them.
  - Filler cut edges snap to the nearest VAD silence boundary (snap_window),
    so a cut never lands mid-word.
  - Glued fillers (whisper folding "uhh" into the next token) are recovered by
    re-transcribing the pre-silence chunk (probe=) and cutting if it is a
    filler. Parakeet does not glue, so the CLI only probes on whisper.
  - Restarts compare consecutive sentences on normalized token sequences;
    only the earlier (abandoned) sentence is cut.
  - All candidate cuts are merged when overlapping or within 2*pad, keeping
    the first reason and joining texts, then clipped and dropped if tiny.
  - The pause/filler machinery assumes Parakeet-grade word timing: whisper
    absorbs moans and hesitation vowels into long word spans (the v7
    failure: a 1.72 s "this"), so a whisper-timed word list distorts the
    pause and filler paths. The CLI warns loudly when it runs this pass on
    whisper timing (warn_whisper_timing); the voiced-burst pass (bursts.py)
    is the belt-and-braces for that case.
"""
from __future__ import annotations

import difflib
import sys
from bisect import bisect_left
from collections.abc import Callable

from .config import Config
from .transcribe import Word
from .types import LLM_CUT_REASONS, CutSpan

_WHISPER_TIMING_WARNING = (
    "WARNING: running the pause/filler machinery on whisper word timing "
    "(--asr whisper/auto without fluidaudiocli). Whisper absorbs moans and "
    "hesitation vowels into long word spans, so pause and filler cuts work "
    "off distorted spans; prefer --asr parakeet (see derive module docs). "
    "The voiced-burst pass stays on as a safety net."
)


def warn_whisper_timing(asr_backend: str | None) -> None:
    """Loud stderr warning when the pause/filler machinery runs on whisper.

    Root cause of the v7 absorption failure: the failing run used whisper
    timings, the good sample used Parakeet. Enforcement is a warning, not a
    raise, because whisper is the documented fallback backend.
    """
    if asr_backend == "whisper":
        print(_WHISPER_TIMING_WARNING, file=sys.stderr)

# Hesitation tokens that are cut outright (subject to `min_filler_dur`).
# Deliberately excludes "like"/"actually"/"you know": fluent uses of those
# are normal speech and cutting them reads as choppy.
FILLER_CUT_WORDS: set[str] = {
    "uh",
    "um",
    "uhh",
    "umm",
    "uhm",
    "erm",
    "er",
    "ah",
    "hmm",
    "hm",
    "mmm",
    "mm",
}

# Full filler set (incl. phrases) used by the whisper glue-probe.
FILLER_WORDS: set[str] = FILLER_CUT_WORDS | {
    "like",
    "you know",
    "i mean",
    "sort of",
    "kind of",
    "basically",
    "actually",
    "literally",
}

# Sentence-ending punctuation looked for inside a word's text.
_ENDERS = (".", "!", "?")
# Minimum span length in seconds; anything shorter is noise, not a cut.
_MIN_SPAN = 0.02
# A word token spanning an internal silence longer than this is glued.
_GLUE_MIN_SILENCE = 0.12


def _norm(text: str) -> str:
    """Lowercase and strip punctuation so 'Like,' matches 'like'."""
    return text.strip().strip(".,!?;:\"'()[]{}").lower()


def _sentences(words: list[Word]) -> list[list[int]]:
    """Group word indices into sentences; a word containing . ! ? ends one."""
    groups: list[list[int]] = []
    current: list[int] = []
    for i, w in enumerate(words):
        current.append(i)
        if any(e in w.text for e in _ENDERS):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _snap(t: float, silences: list[tuple[float, float]], window: float) -> float:
    """Snap t to the nearest silence boundary within window; else return t."""
    best, best_d = t, window
    for s, e in silences:
        for cand in (s, e):
            d = abs(cand - t)
            if d < best_d:
                best_d, best = d, cand
    return best


def _speech_burst(
    w: Word,
    silences: list[tuple[float, float]],
    duration: float,
) -> tuple[float, float] | None:
    """VAD speech burst containing w: [end of silence before, start of silence after].

    Returned only when the word overlaps a determinate burst; None when the
    word lands inside a silence span (no burst to cut).
    """
    if not silences:
        return None
    # Locate by the word's CENTER: ASR filler timestamps drift and can straddle
    # a silence boundary even when the voiced filler sits cleanly in a burst.
    center = (w.start + w.end) / 2
    prev_end = 0.0
    for s, e in silences:
        if e <= center:
            prev_end = e
        elif s >= center:
            return (prev_end, s)
        else:
            return None  # center inside a silence; no clean burst
    return (prev_end, duration)


def _filler_cuts(
    words: list[Word],
    silences: list[tuple[float, float]],
    filler_pad: float,
    snap_window: float,
    min_filler_dur: float,
    pad: float,
    duration: float,
) -> list[CutSpan]:
    """Cut hesitation tokens at least `min_filler_dur` long (0 cuts every one).

    Only tokens ASR actually emitted are considered (Parakeet and Scribe
    transcribe fillers; whisper mostly swallows them).

    Cutting is anchored to the VAD speech burst, not the ASR word time: ASR
    filler timestamps drift (Parakeet placed this clip's 'Um' 0.3s early and
    the word-based cut missed its tail). When the burst is basically the
    filler alone (<= filler span + 0.3s), the whole burst is cut plus pad, so
    the span fuses with the surrounding pause trims. Mid-speech fillers fall
    back to word-time + silence snapping.
    """
    cuts: list[CutSpan] = []
    for w in words:
        if _norm(w.text) not in FILLER_CUT_WORDS:
            continue
        if w.end - w.start < min_filler_dur:
            continue
        burst = _speech_burst(w, silences, duration)
        if burst is not None and (burst[1] - burst[0]) <= (w.end - w.start) + 0.3:
            # Isolated filler: the whole VAD burst is the filler.
            start = max(0.0, burst[0] - pad)
            end = min(duration, burst[1] + pad)
        else:
            start = _snap(w.start - filler_pad, silences, snap_window)
            end = _snap(w.end + filler_pad, silences, snap_window)
        if end - start > _MIN_SPAN:
            cuts.append(CutSpan(start, end, "filler", _norm(w.text)))
    return cuts


def _glued_filler_cuts(
    words: list[Word],
    silences: list[tuple[float, float]],
    probe: Callable[[float, float], str],
    fillers: set[str],
    max_filler_gap: float,
    pad: float,
    duration: float,
) -> list[CutSpan]:
    """Cut fillers whisper swallowed into the next word's span.

    Whisper merges "uhh [pause] an" into one token spanning the whole thing.
    For every word whose span fully contains a VAD silence of >= max_filler_gap,
    re-transcribe the pre-silence audio; if it comes back as a filler word, cut
    the filler together with the pause that followed it.
    """
    cuts: list[CutSpan] = []
    for w in words:
        for s, e in silences:
            if not (w.start < s and e < w.end):
                continue
            if e - s < max_filler_gap:
                continue
            chunk = (w.start, s)
            if chunk[1] - chunk[0] < 0.05:
                continue  # nothing audible before the pause; not a filler chunk
            text = probe(chunk[0], chunk[1])
            tokens = [_norm(t) for t in text.split()] if text else []
            if any(t in fillers for t in tokens) or (_norm(text) in fillers and text):
                # Cut the filler plus the pause after it, leaving the real word.
                start = max(0.0, w.start - pad)
                end = min(duration, e - pad)
                if end - start > _MIN_SPAN:
                    cuts.append(CutSpan(start, end, "filler", text or "filler"))
            break  # one glue case per word token is enough
    return cuts


def _restart_cuts(words: list[Word], similarity_threshold: float, pad: float) -> list[CutSpan]:
    """Cut an abandoned sentence when the next one restates it."""
    cuts: list[CutSpan] = []
    groups = _sentences(words)
    for prev, nxt in zip(groups, groups[1:]):
        if len(prev) < 4:
            continue
        prev_tokens = [_norm(words[i].text) for i in prev]
        nxt_tokens = [_norm(words[i].text) for i in nxt]
        ratio = difflib.SequenceMatcher(None, prev_tokens, nxt_tokens).ratio()
        if ratio >= similarity_threshold:
            first, last = words[prev[0]], words[prev[-1]]
            text = " ".join(words[i].text for i in prev)
            cuts.append(CutSpan(first.start - pad, last.end + pad, "restart", text))
    return cuts


def _silence_cuts_from_words(duration: float, words: list[Word], pause_above: float, pad: float) -> list[CutSpan]:
    """Fallback silence cuts from whisper word gaps (test-only; CLI uses VAD)."""
    cuts: list[CutSpan] = []
    # Leading silence.
    if words[0].start - pad > pause_above:
        cuts.append(CutSpan(0.0, words[0].start - pad, "silence"))
    # Gaps between consecutive words.
    for prev, nxt in zip(words, words[1:]):
        if nxt.start - prev.end > pause_above:
            cuts.append(CutSpan(prev.end, nxt.start, "silence"))
    # Trailing silence.
    if duration - (words[-1].end + pad) > pause_above:
        cuts.append(CutSpan(words[-1].end + pad, duration, "silence"))
    return cuts


_CLOSERS = "\"')]}\u201d\u2019"
_DASHES = ("-", "\u2013", "\u2014")


def pause_boundary(word_text: str, raw_sec: float, cfg) -> str:
    """Classify a pause by the word before it: 'mid', 'clause', 'sentence' or
    'topic' (sentence end with a raw pause of at least pause_topic_above)."""
    t = word_text.strip().rstrip(_CLOSERS)
    if t.endswith((".", "?", "!")):
        return "topic" if raw_sec >= cfg.pause_topic_above else "sentence"
    if t.endswith((",", ";", ":") + _DASHES):
        return "clause"
    return "mid"


def pause_cap(boundary: str, cfg) -> float:
    """Longest pause kept at this boundary type."""
    return {
        "mid": cfg.pause_cap_mid,
        "clause": cfg.pause_cap_clause,
        "sentence": cfg.pause_cap_sentence,
        "topic": cfg.pause_cap_topic,
    }[boundary]


def _silence_cuts_from_vad(
    duration: float,
    silences: list[tuple[float, float]],
    words: list[Word],
    pause_above: float,
    cfg,
    pad: float,
) -> list[CutSpan]:
    """Silence cuts from waveform VAD spans.

    Leading/trailing silence is removed outright (padded). Interior pauses
    longer than pause_above are trimmed, centred in the silence, to
    min(raw - 0.02, cap(boundary), pause_keep + pause_keep_ratio * (raw - pause_above)):
    longer pauses keep a little more, up to the cap of the boundary they sit
    on. Shorter interior pauses are left untouched. The boundary word is the
    last word that STARTS before the silence: ASR stamps often end late,
    inside the silence, so ends are not trusted.
    """
    cuts: list[CutSpan] = []
    starts = [w.start for w in words]
    for s, e in silences:
        if s <= 0.0:
            # Leading silence: remove it entirely.
            end = e - pad
            if end > _MIN_SPAN:
                cuts.append(CutSpan(0.0, end, "silence"))
        elif e >= duration:
            # Trailing silence: remove it entirely.
            start = s + pad
            if duration - start > _MIN_SPAN:
                cuts.append(CutSpan(start, duration, "silence"))
        else:
            dur = e - s
            if dur <= pause_above:
                continue
            i = bisect_left(starts, s) - 1
            boundary = pause_boundary(words[i].text, dur, cfg) if i >= 0 else "mid"
            keep = min(
                dur - 0.02,
                pause_cap(boundary, cfg),
                cfg.pause_keep + cfg.pause_keep_ratio * (dur - pause_above),
            )
            mid = (s + e) / 2
            cuts.append(CutSpan(mid - (dur - keep) / 2, mid + (dur - keep) / 2, "silence"))
    return cuts


def derive_cuts(
    duration: float,
    words: list[Word],
    *,
    pause_above: float = 0.35,
    pause: object | None = None,
    max_filler_gap: float = 0.3,
    filler_words: set[str] | None = None,
    pad: float = 0.04,
    similarity_threshold: float = 0.8,
    silences: list[tuple[float, float]] | None = None,
    probe: Callable[[float, float], str] | None = None,
    filler_pad: float = 0.06,
    snap_window: float = 0.12,
    min_filler_dur: float = 0.0,
) -> list[CutSpan]:
    """Propose every CutSpan to remove from a video of the given duration.

    silences: VAD silence spans from transcribe.detect_silences. When provided
        they replace whisper-gap silence detection entirely (ground truth).
    probe: re-transcribes an audio slice; used to classify glued fillers
        (whisper backend only).
    """
    fillers = filler_words if filler_words is not None else FILLER_WORDS
    cuts: list[CutSpan] = []

    if silences is not None:
        cuts.extend(_silence_cuts_from_vad(duration, silences, words, pause_above, pause or Config(), pad))
        if not words and not silences and duration > pause_above:
            # No speech at all and no VAD spans: the whole video is silence.
            cuts.append(CutSpan(0.0, duration, "silence"))
    else:
        if not words:
            # Word-free video, no VAD: the whole thing is silence.
            if duration > pause_above:
                cuts.append(CutSpan(0.0, duration, "silence"))
            return _merge(cuts, duration, pad)
        cuts.extend(_silence_cuts_from_words(duration, words, pause_above, pad))

    if words:
        cuts.extend(_filler_cuts(words, silences or [], filler_pad, snap_window, min_filler_dur, pad, duration))
        if silences and probe:
            cuts.extend(_glued_filler_cuts(words, silences, probe, fillers, max_filler_gap, pad, duration))
        cuts.extend(_restart_cuts(words, similarity_threshold, pad))
    return _merge(cuts, duration, pad)


def _merge(cuts: list[CutSpan], duration: float, pad: float) -> list[CutSpan]:
    """Sort, fuse overlapping/nearby cuts (LLM reason wins on fuse, texts
    joined), clip to the video, and drop spans too short to act on."""
    if not cuts:
        return []
    cuts = sorted(cuts, key=lambda c: (c.start, c.end))
    merged: list[CutSpan] = [CutSpan(cuts[0].start, cuts[0].end, cuts[0].reason, cuts[0].text)]
    # Mechanical reasons (silence/filler) can absorb an LLM judgment span:
    # keep the LLM reason so the review UI shows why the cut was proposed.
    for c in cuts[1:]:
        last = merged[-1]
        if c.start <= last.end + pad * 2:
            last.end = max(last.end, c.end)
            if c.reason in LLM_CUT_REASONS and last.reason not in LLM_CUT_REASONS:
                last.reason = c.reason
            if c.text:
                last.text = f"{last.text} {c.text}".strip()
        else:
            merged.append(CutSpan(c.start, c.end, c.reason, c.text))

    out: list[CutSpan] = []
    for c in merged:
        start, end = max(0.0, c.start), min(duration, c.end)
        if end - start > _MIN_SPAN:
            out.append(CutSpan(start, end, c.reason, c.text))
    return out