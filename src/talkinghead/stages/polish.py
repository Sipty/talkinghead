"""Polish: gap tightening and disfluency cuts on the material restructure kept.

Opt-in (`polish = true`). Runs after restructure, on the kept material (the raw cuts plus the block
cuts), and hands its cuts to the edge pass with the rest.

1. Gap tightening (no judge). The silence between two consecutive kept words is cut down to a cap set
   by the boundary it sits on: block boundary, speaker change, sentence end, or inside a sentence. A gap
   is tightened only when both words are kept and no word-removing cut lies in it, so a cut never
   reaches across a word. Laughter and other protected events inside the gap are kept whole.
2. Disfluency judge. Kept sentences are judged in windows (restructure's windows, with read-only
   context either side). The judge names word spans to remove: repeats, false starts, self-corrections,
   fillers, fragments, clutter exchanges and interrupting backchannels. Each window is sampled
   `polish_votes` times and a span is cut only when `polish_agree` samples agree on it.

The stage value holds every cut with its reason, words and votes, so a hand-edited memo is honoured on
replay. `cuts_from` reads it back; `run` produces it.

Output pass (`output_round`, one memo per round). The source transcript cannot show a stutter Scribe merged into
one word, so the rendered cut is transcribed and scanned (`residual.scan`). The judge confirms the flagged spots
in one batched prompt; each confirmed spot maps back to the main source (`residual.to_source`) and is cut with
reason `polish`. The caller re-renders and runs the next round on the new plan.
"""

from __future__ import annotations

import bisect
import hashlib
import inspect
import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .. import residual
from ..context import Context
from ..edges import WORD_CUT_REASONS, word_extent
from ..judge import vote
from ..prompts import retake_table, with_conversation, with_preferences
from ..residual import Finding
from ..semantic import Sentence
from ..spans import cluster_spans
from ..timeline import Cut
from ..types import JUDGED_REASONS
from .restructure import Block, Window, concurrent_safe, windows

GAP_REASON = "gap"
POLISH_REASON = "polish"
KINDS = ("repeat", "false-start", "self-correction", "filler", "fragment", "clutter", "backchannel")
# Gap caps by boundary, most specific first.
BOUNDARIES = ("block", "speaker", "sentence", "within")

# The policy both disfluency passes apply: the source pass names spans of the source transcript, the output pass
# confirms flagged spots of the render. The kinds and the NEVER rules are one text, so the two cannot drift apart.
POLISH_KINDS = (
    "- repeat: a stutter or a repeated word or phrase. Keep the last clean instance. Example: \"I, I, we, we have "
    "a daily, right?\" -> remove \"I, I, we,\" and keep \"we have a daily, right?\".\n"
    "- false-start: an abandoned attempt or restart. Keep the restart. Example: \"I, it's... I went to "
    "Kubernetes\" -> remove \"I, it's...\" and keep \"I went to Kubernetes\".\n"
    "- self-correction: a wrong word or phrase corrected at once. Remove the wrong part and keep the correction.\n"
    "- filler: \"uh\", \"um\", a filler \"like\", \"you know\", \"I mean\" that adds nothing. Example: \"Uh, I could "
    "have just, deleted the whole subscription.\" -> remove \"Uh,\".\n"
    "- fragment: a dangling word cut off before a restart (\"n-\", \"F-\") when the restart follows it.\n"
    "- clutter: a short exchange of mishearing or confusion that the viewer will not miss. Example: S0 \"Me?\", "
    "S1 \"It's... what?\", S0 \"Me?\", S1 \"I.\", then S0 \"Uh-huh. Okay.\" -> remove the four confused turns "
    "and keep \"Uh-huh. Okay.\".\n"
    "- backchannel: a listener's \"yeah\", \"mm\" or \"uh-huh\" that cuts into the speaker's flow. Remove it only when "
    "both turns stay intact and grammatical.\n\n"
)
POLISH_NEVER = (
    "NEVER remove a laugh, a punchline or the set-up of one, the substance of a question or an answer, a claim, "
    "or anything that leaves a sentence ungrammatical. Keep each span tight: start on the first word to remove "
    "and end on the last. A span may cross speakers only inside a clutter exchange. When in doubt, keep.\n\n"
)
POLISH_RULES = (
    "This is one stretch of a talking-head podcast, cut from a long recording. Each sentence line shows its "
    "word ids, like [w0012]. Make the speech snappier by naming word spans to REMOVE. The viewer hears only "
    "what you leave in, so every removal must keep the meaning and the grammar of what stays.\n\n"
    "Answer with spans of word ids from the CORE section only: from_word to to_word, both inclusive. Kinds:\n"
    + POLISH_KINDS + POLISH_NEVER
    + "Answer with cuts: from_word, to_word, kind (one of the kinds above) and why (a few words). An empty list "
    "means nothing to remove.\n"
)
OUTPUT_RULES = (
    "This is the finished output of one talkinghead video: the cut as the viewer hears it, transcribed after the "
    "cut. Each candidate below is a spot a scan flagged as a likely disfluency: a filler, a phrase said twice, or "
    "a word restarted after a fragment. In each context the words a cut would remove are in double brackets, and "
    "the words after them are the ones that stay. For each candidate, answer cut (remove the bracketed words and "
    "the silence before the words that stay) or keep (the words are deliberate emphasis, or carry the meaning). "
    "Judge against these kinds. Their examples name source words, so judge the flagged words in their context.\n\n"
    "Kinds:\n" + POLISH_KINDS + POLISH_NEVER
    + "Answer one verdict per candidate id: cut or keep, and why in a few words. A candidate you do not name counts "
    "as keep.\n"
)

POLISH_SCHEMA = {
    "type": "object",
    "properties": {"cuts": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "from_word": {"type": "string"},
            "to_word": {"type": "string"},
            "kind": {"type": "string", "enum": list(KINDS)},
            "why": {"type": "string"},
        },
        "required": ["from_word", "to_word", "kind", "why"],
    }}},
    "required": ["cuts"],
}


def policy() -> str:
    """Fingerprint of what the disfluency judges are asked, how answers are read, and how gaps are cut.

    The stage memo is keyed on it, so a changed prompt, schema or rule recomputes the cuts instead of
    replaying decisions made under the old rules.
    """
    sources = [inspect.getsource(fn) for fn in (polish_prompt, _judge_window, gap_cuts, _kept_at)]
    text = "\n".join(sources + [repr(POLISH_SCHEMA), POLISH_RULES])
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def polish_prompt(before: str, core: str, after: str, topic: str | None) -> str:
    topic_line = f"Video topic (from the owner): {topic}\n\n" if topic else ""
    return (
        "TASK: POLISH WINDOW\n"
        + topic_line
        + POLISH_RULES
        + "\n--- CONTEXT BEFORE (read only: never answer for these words) ---\n" + (before or "(none)") + "\n\n"
        + "--- CORE (answer for these word ids only) ---\n" + core + "\n\n"
        + "--- CONTEXT AFTER (read only: never answer for these words) ---\n" + (after or "(none)") + "\n"
    )


# ------------------------------------------------------------- span arithmetic


def _coalesce(pairs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sorted, disjoint spans: overlapping or touching pairs fused."""
    out: list[list[float]] = []
    for s, e in sorted(pairs):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


class _Spans:
    """A set of time spans with the queries the gap and disfluency steps need."""

    def __init__(self, pairs: list[tuple[float, float]]) -> None:
        self.spans = _coalesce([(s, e) for s, e in pairs if e > s])
        self.starts = [s for s, _ in self.spans]

    def contains(self, t: float) -> bool:
        i = bisect.bisect_right(self.starts, t) - 1
        return i >= 0 and self.spans[i][0] <= t < self.spans[i][1]

    def overlapping(self, lo: float, hi: float) -> list[tuple[float, float]]:
        i = max(0, bisect.bisect_right(self.starts, lo) - 1)
        out: list[tuple[float, float]] = []
        while i < len(self.spans) and self.spans[i][0] < hi:
            s, e = self.spans[i]
            if e > lo:
                out.append((s, e))
            i += 1
        return out

    def subtract(self, lo: float, hi: float) -> list[tuple[float, float]]:
        """[lo, hi) minus the spans: the pieces left."""
        pieces: list[tuple[float, float]] = []
        cur = lo
        for s, e in self.overlapping(lo, hi):
            if s > cur:
                pieces.append((cur, min(s, hi)))
            cur = max(cur, e)
            if cur >= hi:
                break
        if cur < hi:
            pieces.append((cur, hi))
        return [(a, b) for a, b in pieces if b > a]


def _kept_at(pieces: list[tuple[float, float]], x: float) -> float:
    """The source instant at which `x` seconds of kept audio have been passed, counting from the first piece."""
    acc = 0.0
    for s, e in pieces:
        if x <= acc + (e - s) + 1e-12:
            return s + max(0.0, x - acc)
        acc += e - s
    return pieces[-1][1]


# ------------------------------------------------------------- gap tightening


def _cap(cfg, at_block: bool, speaker_change: bool, sentence_end: bool) -> tuple[float, str]:
    """The longest silence kept at a gap, and the boundary that set it (most specific first)."""
    if at_block:
        return cfg.polish_gap_block, "block"
    if speaker_change:
        return cfg.polish_gap_speaker, "speaker"
    if sentence_end:
        return cfg.polish_gap_sentence, "sentence"
    return cfg.polish_gap_within, "within"


# Fixed distances a gap cut keeps from the word edges it separates. Gap cuts are placed from the word stamps
# alone: the envelope smears refined bounds into restaurant noise, so finalize does not steer them.
GAP_PAD_AFTER = 0.06
GAP_PAD_BEFORE = 0.05


def _extent(w) -> tuple[float, float]:
    """A word's audio extent: its refined bounds widened to its ASR stamps (the edge guards use the same one)."""
    return word_extent(w)


def gap_cuts(ctx: Context, cuts: list[Cut], blocks: list[Block]) -> list[Cut]:
    """Silence cuts for every kept gap longer than its cap.

    `cuts` are the raw cuts plus the disfluency cuts. A gap is the silence between consecutive word extents, and
    both words are kept. No word-removing cut touches it, and no voiced or protected audio lies in the cut. The
    kept audio in the gap (what the other cuts left) is what gets capped: the cut leaves `cap / 2` of kept audio
    after the outgoing word and `cap / 2` before the incoming one, never less than the fixed pads.
    """
    cfg = ctx.config
    words = ctx.words
    extent = [_extent(w) for w in words]
    removing = _Spans([(c.start, c.end) for c in cuts if c.reason in WORD_CUT_REASONS | JUDGED_REASONS])
    any_cut = _Spans([(c.start, c.end) for c in cuts])
    blocked = _Spans(list(ctx.protected_spans) + list(ctx.voices))
    kept = [not removing.contains((a + b) / 2) for a, b in extent]
    sentence_ends = {g[-1] for g in ctx.sentence_words() if g}
    block_ends = sorted(b.end for b in blocks[:-1])
    order = sorted(range(len(words)), key=lambda i: extent[i])
    out: list[Cut] = []
    reach = -math.inf
    for k in range(1, len(order)):
        p, q = order[k - 1], order[k]
        reach = max(reach, extent[p][1])
        lo, hi = reach, extent[q][0]
        if not (kept[p] and kept[q]) or hi <= lo or removing.overlapping(lo, hi):
            continue
        pieces = any_cut.subtract(lo, hi)
        total = sum(b - a for a, b in pieces)
        i_block = bisect.bisect_left(block_ends, lo - 1e-3)
        at_block = i_block < len(block_ends) and block_ends[i_block] <= hi + 1e-3
        speaker_change = (words[p].speaker is not None and words[q].speaker is not None
                          and words[p].speaker != words[q].speaker)
        cap, boundary = _cap(cfg, at_block, speaker_change, p in sentence_ends)
        if not pieces or total - cap < cfg.min_cut_sec:
            continue
        a = max(_kept_at(pieces, cap / 2), lo + GAP_PAD_AFTER)
        b = min(_kept_at(pieces, total - cap / 2), hi - GAP_PAD_BEFORE)
        for x, y in blocked.subtract(a, b):
            if y - x >= cfg.min_cut_sec:
                out.append(Cut(x, y, GAP_REASON, "", {"cap": cap, "boundary": boundary, "words": [p, q]}))
    return out


# ------------------------------------------------------------- disfluency judge


_WID = re.compile(r"^w(\d{4,})$")


def _wid(value: Any, n: int) -> int | None:
    m = _WID.match(str(value or ""))
    if not m:
        return None
    i = int(m.group(1))
    return i if i < n else None


def kept_sentences(ctx: Context, removing: _Spans) -> tuple[list[Sentence], list[list[int]]]:
    """Transcript sentences with the words no word-removing cut takes out, and those words' indices."""
    words = ctx.words
    sents: list[Sentence] = []
    groups: list[list[int]] = []
    for s, idx in zip(ctx.sentences(), ctx.sentence_words()):
        keep = [i for i in idx if not removing.contains((words[i].start + words[i].end) / 2)]
        if not keep:
            continue
        sents.append(Sentence(s.sid, words[keep[0]].start, words[keep[-1]].end,
                              " ".join(words[i].text for i in keep), s.speaker))
        groups.append(keep)
    return sents, groups


def _judge_window(ctx: Context, sents: list[Sentence], groups: list[list[int]], w: Window) -> dict:
    """One window's disfluency spans: the samples' spans clustered, each kept when enough samples agree."""
    cfg = ctx.config
    words = ctx.words
    n = len(words)
    a, b = w.core
    c_lo, c_hi = w.context
    core_ids = {i for g in groups[a:b] for i in g}
    prompt = with_preferences(with_conversation(
        polish_prompt(retake_table(sents[c_lo:a], groups[c_lo:a], words),
                      retake_table(sents[a:b], groups[a:b], words),
                      retake_table(sents[b:c_hi], groups[b:c_hi], words), ctx.topic),
        ctx.conversation), ctx.preferences)
    result = vote(ctx.judge, prompt, POLISH_SCHEMA, k=cfg.polish_votes, cache=ctx.llm_cache)
    per_response: list[list[tuple[int, int]]] = []
    entries: list[tuple[int, int, str, str]] = []
    stats: Counter = Counter()
    for resp in result.responses:
        spans: list[tuple[int, int]] = []
        for item in resp.get("cuts") or []:
            if not isinstance(item, dict):
                stats["bad_ids"] += 1
                continue
            lo_w, hi_w = _wid(item.get("from_word"), n), _wid(item.get("to_word"), n)
            if lo_w is None or hi_w is None or hi_w < lo_w or lo_w not in core_ids or hi_w not in core_ids:
                stats["bad_ids"] += 1
                continue
            if words[hi_w].end - words[lo_w].start > cfg.polish_max_sec:
                stats["overlong"] += 1
                continue
            spans.append((lo_w, hi_w))
            entries.append((lo_w, hi_w, str(item.get("kind") or ""), str(item.get("why") or "").strip()))
            stats["proposed"] += 1
        per_response.append(spans)
    # A word is removed only when `agree` samples remove it (not the intersection of all of them).
    clusters = cluster_spans(per_response, agree=cfg.polish_agree, mode="agreed")
    spans_out = []
    for sp in clusters:
        near = [e for e in entries if e[0] <= sp.b and e[1] >= sp.a]
        kinds = Counter(e[2] for e in near if e[2] in KINDS)
        kind = kinds.most_common(1)[0][0] if kinds else "other"
        why = next((e[3] for e in near if e[2] == kind and e[3]), "")
        spans_out.append({"a": sp.a, "b": sp.b, "votes": sp.votes, "kind": kind, "why": why})
    return {"spans": spans_out, "clusters": len(clusters), "stats": dict(stats)}


def _judge_windows(ctx: Context, sents: list[Sentence], groups: list[list[int]], wins: list[Window]) -> list[dict]:
    def one(i: int) -> dict:
        return _judge_window(ctx, sents, groups, wins[i])

    if ctx.config.polish_parallel > 1 and len(wins) > 1 and concurrent_safe(ctx.judge):
        with ThreadPoolExecutor(max_workers=ctx.config.polish_parallel) as pool:
            return list(pool.map(one, range(len(wins))))
    return [one(i) for i in range(len(wins))]


def disfluency_cuts(ctx: Context, results: list[dict]) -> list[Cut]:
    words = ctx.words
    cuts = [
        Cut(words[sp["a"]].start, words[sp["b"]].end, POLISH_REASON,
            " ".join(w.text for w in words[sp["a"]:sp["b"] + 1]),
            {"words": [sp["a"], sp["b"]], "votes": sp["votes"], "kind": sp["kind"], "why": sp["why"]})
        for res in results for sp in res["spans"]
    ]
    return sorted(cuts, key=lambda c: (c.start, c.end))


# ------------------------------------------------------------- stage


def run(ctx: Context, raw: list[Cut], blocks: list[Block], hand: Sequence[Cut] = ()) -> dict:
    """The stage value: `{"cuts": [Cut as dict], "notes": {...}}`, with every cut's reason, words and votes.

    `hand` are the hand-edited cuts of the stored memo: they are carried over unchanged and count as judged.
    """
    cfg = ctx.config
    removing = _Spans([(c.start, c.end) for c in list(raw) + list(hand)
                       if c.reason in WORD_CUT_REASONS | JUDGED_REASONS])
    sents, groups = kept_sentences(ctx, removing)
    wins = windows(sents, cfg.polish_window_sec, cfg.polish_context_sec)
    results = _judge_windows(ctx, sents, groups, wins) if wins else []
    judged = disfluency_cuts(ctx, results)
    disf = sorted(judged + list(hand), key=lambda c: (c.start, c.end))
    gaps = gap_cuts(ctx, raw + disf, blocks)
    cuts = sorted(gaps + disf, key=lambda c: (c.start, c.end))
    stats: Counter = Counter()
    for res in results:
        stats.update(res["stats"])
    notes = {
        "gaps": len(gaps),
        "gap_sec": round(sum(c.length for c in gaps), 2),
        "gap_boundaries": {k: sum(1 for c in gaps if c.provenance["boundary"] == k) for k in BOUNDARIES},
        "disfluency": len(judged),
        "disfluency_sec": round(sum(c.length for c in judged), 2),
        "disfluency_words": sum(c.provenance["words"][1] - c.provenance["words"][0] + 1 for c in judged),
        "disfluency_kinds": dict(Counter(c.provenance["kind"] for c in judged)),
        "windows": len(wins),
        "proposed": stats["proposed"],
        "dropped_bad_ids": stats["bad_ids"],
        "dropped_overlong": stats["overlong"],
        "hand": len(hand),
        "clusters": sum(res["clusters"] for res in results),
    }
    return {"cuts": [asdict(c) for c in cuts], "notes": notes}


def hand_cuts(memo_path: Path) -> list[Cut]:
    """The hand-edited cuts of a stored polish memo: a re-key carries them over, so they are never lost."""
    if not memo_path.is_file():
        return []
    stored = json.loads(memo_path.read_text())
    return [Cut(**c) for c in stored["value"]["cuts"] if "hand" in c.get("provenance", {})]


def cuts_from(value: dict) -> list[Cut]:
    return [Cut(**c) for c in value["cuts"]]


# ------------------------------------------------------------- output pass

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"verdicts": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "verdict": {"type": "string", "enum": ["cut", "keep"]},
            "why": {"type": "string"},
        },
        "required": ["id", "verdict", "why"],
    }}},
    "required": ["verdicts"],
}
# Words of transcript shown on each side of a finding, in seconds of output time.
OUTPUT_CONTEXT_SEC = 10.0
# A cut that stops this long before the kept word leaves the kept word's onset clear of the cut's edge.
OUTPUT_KEPT_PAD = 0.05


def _candidate_id(k: int) -> str:
    return f"c{k}"


def _context(words: Sequence, f: Finding) -> str:
    """About OUTPUT_CONTEXT_SEC of the transcript around a finding, its removed words in double brackets."""
    lo, hi = f.start - OUTPUT_CONTEXT_SEC / 2, f.end + OUTPUT_CONTEXT_SEC / 2
    parts = []
    for i, w in enumerate(words):
        if w.end < lo or w.start > hi:
            continue
        parts.append(f"[[{w.text.strip()}]]" if f.first <= i <= f.last else w.text.strip())
    return " ".join(parts)


def output_prompt(words: Sequence, found: list[Finding], topic: str | None) -> str:
    """One prompt for the whole round: every finding with its context, judged together."""
    topic_line = f"Video topic (from the owner): {topic}\n\n" if topic else ""
    lines = [f"{_candidate_id(k)} ({f.kind} at {f.start:.1f} s): {_context(words, f)}" for k, f in enumerate(found)]
    return ("TASK: OUTPUT PASS\n" + topic_line + OUTPUT_RULES
            + "\n--- CANDIDATES (answer for these ids only) ---\n" + "\n".join(lines) + "\n")


def output_policy() -> str:
    """Fingerprint of what the output judge is asked, how its answers are read, and how findings map to source.

    The output round's memo is keyed on it, so a changed rule, scan or mapping recomputes the round.
    """
    sources = [inspect.getsource(fn) for fn in (output_prompt, _context, output_round, residual.scan,
                                                residual.to_source)]
    text = "\n".join(sources + [repr(OUTPUT_SCHEMA), OUTPUT_RULES, repr((OUTPUT_CONTEXT_SEC, OUTPUT_KEPT_PAD))])
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def output_round(ctx: Context, plan, words: Sequence, round_no: int) -> dict:
    """One output round: the findings in the render's transcript, the judge's verdicts, and the confirmed cuts.

    `plan` is the plan that was rendered to produce `words`. The stage value holds `cuts` (Cut as dict, reason
    `polish`), `keep` (the judged-keep findings: key and text, so the final check does not flag them again),
    `findings` (every finding with its verdict) and `notes`. A finding with fewer than `polish_agree` cut votes is
    kept; a confirmed finding that cannot map to one main-source span is reported as skipped, never cut.
    """
    cfg = ctx.config
    found = residual.scan(words)
    counts = {"round": round_no, "votes": cfg.polish_votes, "agree": cfg.polish_agree}
    if not found:
        return {"cuts": [], "keep": [], "findings": [], "notes": {**counts, "findings": 0, "cut": 0, "keep": 0,
                                                                  "skipped": 0}}
    prompt = with_preferences(with_conversation(output_prompt(words, found, ctx.topic), ctx.conversation),
                              ctx.preferences)
    result = vote(ctx.judge, prompt, OUTPUT_SCHEMA, k=cfg.polish_votes, cache=ctx.llm_cache)
    cut_votes: Counter = Counter()
    whys: dict[str, str] = {}
    for resp in result.responses:
        seen: set[str] = set()
        for item in resp.get("verdicts") or []:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("id") or "")
            if cid in seen:
                continue
            seen.add(cid)
            if item.get("verdict") == "cut":
                cut_votes[cid] += 1
                if item.get("why"):
                    whys.setdefault(cid, str(item["why"]).strip())
    cuts: list[Cut] = []
    keep: list[dict] = []
    rows: list[dict] = []
    for k, f in enumerate(found):
        cid = _candidate_id(k)
        votes = cut_votes[cid]
        row = {"id": cid, **f.to_dict(), "votes": votes, "why": whys.get(cid, "")}
        if votes < cfg.polish_agree:
            row["decision"] = "keep"
            keep.append({"key": f.key, "text": f.heard, "start": round(f.start, 3), "kind": f.kind})
        else:
            end = f.end - (OUTPUT_KEPT_PAD if f.follows else 0.0)
            src = residual.to_source(plan, f.start, end)
            if src is None:
                row["decision"] = "skipped: crosses a join"
            elif src[1] - src[0] < cfg.min_cut_sec:
                row["decision"] = "skipped: too short"
            else:
                row["decision"] = "cut"
                cuts.append(Cut(src[0], src[1], POLISH_REASON, f.heard,
                                {"pass": "output", "round": round_no, "kind": f.kind,
                                 "output": [round(f.start, 3), round(end, 3)], "votes": votes,
                                 "why": whys.get(cid, ""), "span": [src[0], src[1]]}))
        rows.append(row)
    decided = Counter(r["decision"].split(":")[0] for r in rows)
    notes = {**counts, "findings": len(found), "cut": decided["cut"], "keep": decided["keep"],
             "skipped": decided["skipped"]}
    return {"cuts": [asdict(c) for c in cuts], "keep": keep, "findings": rows, "notes": notes}


def output_cuts_from(value: dict) -> list[Cut]:
    return [Cut(**c) for c in value["cuts"]]

