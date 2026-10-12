"""Shared LLM plumbing and the flow guard.

`run_llm` and `parse_llm_json` call the judge command, and `chunk_sentences`
splits a transcript for it. `flow_guard` fuses cuts that would leave staccato
jump cuts; it runs last, on the fully merged plan.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass

from .derive import FILLER_CUT_WORDS, _norm
from .transcribe import Word
from .types import MECHANICAL_REASONS, CutSpan

# Speech-time overlap between consecutive chunks, so a cut spanning a chunk
# boundary is still visible in both prompts.
_CHUNK_OVERLAP = 120.0

# Flow guard thresholds (seconds). Two cuts with less kept time between them
# than FLOW_MIN_KEPT read as one rapid-fire jump cut, so they fuse; but ONLY
# when the gap is free of real speech (see _gap_has_speech), because fusing
# across a spoken sentence removes content, not rhythm. FLOW_MIN_FRAGMENT is
# the shortest kept island the guard leaves standing.
FLOW_MIN_KEPT = 2.0
FLOW_MIN_FRAGMENT = 1.5


@dataclass
class Sentence:
    """One transcript sentence with its timing and zero-padded id."""

    sid: str
    start: float
    end: float
    text: str
    # Display name of the one speaker who says the whole sentence; None when unlabelled.
    speaker: str | None = None


def run_llm(cmd: str, prompt: str) -> str:
    """Run the shell command with the prompt on stdin; return stdout."""
    proc = subprocess.run(cmd, shell=True, input=prompt, capture_output=True, text=True)
    if proc.returncode != 0 or not proc.stdout.strip():
        tail = proc.stderr.strip()[-500:]
        raise RuntimeError(f"llm command failed (exit {proc.returncode}): {tail}")
    return proc.stdout


def parse_llm_json(raw: str) -> dict:
    """Parse the LLM stdout as JSON, tolerating ``` fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid LLM JSON: {exc}: {raw[:200]}") from exc


def chunk_sentences(sents: list[Sentence], max_chars: int = 240_000) -> list[list[Sentence]]:
    """Split into chunks of at most max_chars of sentence text.

    Consecutive chunks overlap by ~120s of speech so a cut spanning a chunk
    boundary is still visible in both prompts.
    """
    if not sents:
        return []
    chunks: list[list[Sentence]] = []
    current: list[Sentence] = []
    size = 0
    for s in sents:
        if current and size + len(s.text) + 1 > max_chars:
            chunks.append(current)
            overlap: list[Sentence] = []
            for prev in reversed(current):
                overlap.insert(0, prev)
                if current[-1].end - prev.start >= _CHUNK_OVERLAP:
                    break
            current = overlap
            size = sum(len(x.text) + 1 for x in current)
        current.append(s)
        size += len(s.text) + 1
    chunks.append(current)
    return chunks


def flow_guard(
    cuts: list[CutSpan],
    *,
    words: list[Word] | None = None,
    min_kept: float = FLOW_MIN_KEPT,
    min_fragment: float = FLOW_MIN_FRAGMENT,
) -> list[CutSpan]:
    """Fuse cuts that would leave staccato jump cuts between them.

    Runs last, on the fully merged plan, in two passes:

    1. cross-reason merge: consecutive cuts with less than `min_kept` of kept
       time between them become one span, but ONLY when the gap holds no real
       speech: when `words` is supplied, a gap containing a spoken word
       longer than FLOW_SPEECH_GAP keeps both cuts separate, because fusing
       them would swallow the sentence between (silence trims a second apart
       still fuse; two cuts bracketing a sentence do not).
    2. fragment absorption: a kept island shorter than `min_fragment` between
       two cuts is absorbed into the cut before it.

    With the default thresholds pass 1 already covers every island pass 2
    would catch; pass 2 is the backstop for tuned thresholds
    (`min_fragment` > `min_kept`). Output is sorted by start.
    """
    return _fuse_within(_fuse_within(cuts, min_kept, words), min_fragment, words)


# A gap between two cuts is only fusable when it holds no real speech. ASR
# words in the gap block fusion, EXCEPT filler tokens (uh/um), which are
# cuttable noise the flow guard exists to clean up. Word length is not a
# usable signal: normal spoken words are 0.1-0.3s.
FLOW_SPEECH_GAP = 0.3


def _gap_has_speech(cuts: list[CutSpan], i: int, words: list[Word] | None) -> bool:
    """True when the kept gap between cuts[i] and cuts[i+1] holds spoken words.
    Measured on the word midpoint with STRICT bounds so a word straddling a
    cut edge (or sitting exactly on one) does not block a legitimate fusion,
    and a zero-width gap between adjacent cuts is never blocked. Filler
    tokens (uh, um, ...) never block: they are exactly what adjacent trims
    should be allowed to sweep up.
    """
    if words is None or i + 1 >= len(cuts):
        return False
    gap_start, gap_end = cuts[i].end, cuts[i + 1].start
    for w in words:
        mid = (w.start + w.end) / 2
        if gap_start < mid < gap_end and _norm(w.text) not in FILLER_CUT_WORDS:
            return True
    return False


def _fuse_within(cuts: list[CutSpan], gap: float, words: list[Word] | None = None) -> list[CutSpan]:
    """Fuse cuts separated by less than `gap`, ignoring their reasons.

    Unlike `_merge_same_reason`, a filler cut next to a silence cut fuses too:
    to the viewer both produce the same jump. The surviving span is the union
    and the first cut's reason wins; absorbed text is appended. A gap holding
    real speech (see `_gap_has_speech`) is never fused across when `words` is
    given, so content between two cuts survives.
    """
    out: list[CutSpan] = []
    for c in sorted(cuts, key=lambda c: (c.start, c.end)):
        if out and c.start - out[-1].end < gap and not _gap_has_speech(out, len(out) - 1, words):
            last = out[-1]
            last.end = max(last.end, c.end)
            if c.text and c.text != last.text:
                last.text = f"{last.text} {c.text}".strip()
            # An LLM judgment that lands inside a fused span must own its
            # label: the span now removes real speech, so reporting it as a
            # mechanical "silence"/"filler" trim hides what happened.
            if c.reason not in MECHANICAL_REASONS and last.reason in MECHANICAL_REASONS:
                last.reason = c.reason
        else:
            out.append(CutSpan(c.start, c.end, c.reason, c.text))
    return out
