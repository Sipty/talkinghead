"""Two reads of the finished cut that propose and never cut.

Both read the plan's kept words in the order the viewer hears them, so they see what the judged stages
cannot: those stages each look at the raw recording, one question at a time.

- `repeats`: passages that tell the viewer something an earlier kept passage already said.
- `cold_read`: a viewer who knows only the title marks where they would stop watching.

Each returns rows carrying output and source seconds; the editor decides what to do with a row.
"""

from __future__ import annotations

import bisect
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .judge import Judge, LLMCache, vote
from .prompts import with_preferences
from .timeline import Plan

# A kept pause this long inside one segment ends a line even without punctuation.
LINE_SPLIT_PAUSE = 0.7
# A line is closed after this many words, so one unpunctuated ramble cannot hide a long stretch behind one id.
LINE_MAX_WORDS = 60
_TERMINAL = (".", "?", "!")

REPEATS_SCHEMA = {
    "type": "object",
    "properties": {
        "repeats": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "earlier": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["from", "to", "earlier", "why"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["repeats"],
    "additionalProperties": False,
}

COLD_REASONS = ["slow-start", "off-promise", "repeat", "drags", "confusing", "no-reason-to-care"]

COLD_SCHEMA = {
    "type": "object",
    "properties": {
        "marks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "reason": {"type": "string", "enum": COLD_REASONS},
                    "why": {"type": "string"},
                },
                "required": ["from", "to", "reason", "why"],
                "additionalProperties": False,
            },
        },
        "verdict": {"type": "string"},
    },
    "required": ["marks", "verdict"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Line:
    """One kept sentence-sized stretch: where the viewer hears it and where it sits in the recording."""

    lid: str
    out_start: float
    out_end: float
    src_start: float
    src_end: float
    text: str


def load_words(path: Path, plan: Plan) -> list[dict]:
    """The cook's published source words, refused when they belong to another recording than the plan cuts."""
    data = json.loads(path.read_text())
    main = plan.sources[plan.main]
    sha = (data.get("source") or {}).get("sha256")
    if sha != main.sha256:
        raise ValueError(f"{path} holds the words of another recording than {main.path}; "
                         "pass the source-words.json this plan's cook wrote")
    return data["words"]


def kept_lines(plan: Plan, words: list[dict]) -> list[Line]:
    """The kept words as lines, in output order.

    A word is kept when its midpoint sits inside a main-source segment. Segments are walked in plan order, so
    a reordered plan reads the way it plays. A line ends at sentence punctuation, at a long kept pause, or at
    LINE_MAX_WORDS; it does not end at a cut, because a join inside a sentence is still one sentence to the viewer.
    """
    ordered = sorted(words, key=lambda w: (w["start"] + w["end"]) / 2)
    mids = [(w["start"] + w["end"]) / 2 for w in ordered]
    lines: list[Line] = []
    current: list[tuple[float, float, dict]] = []  # (output start, output end, word)

    def close() -> None:
        if current:
            lines.append(Line(f"L{len(lines) + 1:03d}", current[0][0], current[-1][1],
                              current[0][2]["start"], current[-1][2]["end"],
                              " ".join(w["text"] for _, _, w in current)))
            current.clear()

    clock = 0.0
    for seg in plan.segments:
        if seg.source == plan.main:
            lo, hi = bisect.bisect_left(mids, seg.start), bisect.bisect_left(mids, seg.end)
            prev_end: float | None = None
            for w in ordered[lo:hi]:
                if prev_end is not None and w["start"] - prev_end >= LINE_SPLIT_PAUSE:
                    close()
                start = clock + max(0.0, w["start"] - seg.start)
                current.append((start, clock + min(seg.length, w["end"] - seg.start), w))
                prev_end = w["end"]
                if w["text"].rstrip("\"')”’").endswith(_TERMINAL) or len(current) >= LINE_MAX_WORDS:
                    close()
        clock += seg.length
    close()
    return lines


def _clock(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def line_table(lines: list[Line]) -> str:
    return "\n".join(f"{ln.lid} | {_clock(ln.out_start)} | {ln.text}" for ln in lines)


def _span(row: dict, index: dict[str, int]) -> tuple[int, int] | None:
    """A row's line range, or None when it names an id the table never offered or runs backwards."""
    a, b = index.get(str(row.get("from"))), index.get(str(row.get("to")))
    return (a, b) if a is not None and b is not None and a <= b else None


def _row(lines: list[Line], a: int, b: int) -> dict:
    return {"from": lines[a].lid, "to": lines[b].lid,
            "out_start": round(lines[a].out_start, 2), "out_end": round(lines[b].out_end, 2),
            "src_start": round(lines[a].src_start, 3), "src_end": round(lines[b].src_end, 3),
            "seconds": round(lines[b].out_end - lines[a].out_start, 1),
            "text": " ".join(ln.text for ln in lines[a:b + 1])[:300]}


def repeats_prompt(lines: list[Line]) -> str:
    return (
        "TASK: REPEATS\n"
        "This is the kept cut of a talking-head video, in the order the viewer hears it. Find every passage "
        "that tells the viewer something an EARLIER kept passage already told them: the same point made again, "
        "a second call to action, an \"as I said\" recap, the same example retold. Name the later passage by "
        "its first and last line id, and the earlier line it repeats. A callback that adds something new, a "
        "punchline that pays off a setup, and emphasis inside one sentence are not repeats. When unsure, do "
        "not list it. Empty list when nothing repeats.\n\n"
        f"--- KEPT CUT ---\n{line_table(lines)}\n"
    )


def repeats(lines: list[Line], judge: Judge, *, votes: int, cache: LLMCache | None,
            preferences: str = "") -> list[dict]:
    """Later passages that restate an earlier one, each with the number of samples that named it.

    Samples are merged per line: a run of lines any sample named is one row, scored on its best-backed line.
    """
    index = {ln.lid: i for i, ln in enumerate(lines)}
    result = vote(judge, with_preferences(repeats_prompt(lines), preferences), REPEATS_SCHEMA, k=votes, cache=cache)
    coverage: Counter[int] = Counter()
    detail: dict[int, tuple[str, str]] = {}
    for resp in result.responses:
        seen: set[int] = set()
        for r in resp.get("repeats", []):
            span = _span(r, index)
            earlier = index.get(str(r.get("earlier")))
            # A passage can only repeat something the viewer has already heard.
            if span is None or earlier is None or earlier >= span[0]:
                continue
            for i in range(span[0], span[1] + 1):
                seen.add(i)
                detail.setdefault(i, (lines[earlier].lid, str(r.get("why", ""))))
        coverage.update(seen)
    rows: list[dict] = []
    run: list[int] = []
    for i in sorted(coverage) + [None]:
        if run and (i is None or i != run[-1] + 1):
            best = max(run, key=lambda j: coverage[j])
            earlier, why = detail[best]
            rows.append({**_row(lines, run[0], run[-1]), "votes": coverage[best], "of": votes,
                         "earlier": earlier, "earlier_out": round(lines[index[earlier]].out_start, 2), "why": why})
            run = []
        if i is not None:
            run.append(i)
    return rows


def cold_prompt(lines: list[Line], title: str) -> str:
    return (
        "TASK: COLD READ\n"
        "You clicked a YouTube video because of its title. You know nothing else about it. Below is everything "
        "the speaker says, in order, with the time each line starts. Read it once, the way that viewer would "
        "watch it.\n"
        "Mark every stretch where you would lose interest and think about closing the video. Name its first "
        f"and last line id, pick a reason from {COLD_REASONS}, and say why in one line, from the viewer's seat. "
        "Mark real drop-off risks only, not small flaws; an empty list is a valid answer. Then give a one-line "
        "verdict: does the video deliver what the title promised?\n\n"
        f"TITLE: {title}\n\n--- VIDEO ---\n{line_table(lines)}\n"
    )


def cold_read(lines: list[Line], judge: Judge, title: str, *, votes: int, cache: LLMCache | None) -> dict:
    """Where a viewer who knows only the title would stop watching: one row per mark, per sample."""
    index = {ln.lid: i for i, ln in enumerate(lines)}
    result = vote(judge, cold_prompt(lines, title), COLD_SCHEMA, k=votes, cache=cache)
    marks: list[dict] = []
    for n, resp in enumerate(result.responses):
        for m in resp.get("marks", []):
            span = _span(m, index)
            if span is not None:
                marks.append({**_row(lines, *span), "reason": m["reason"], "why": m["why"], "sample": n})
    marks.sort(key=lambda m: (m["out_start"], m["sample"]))
    return {"title": title, "judge": judge.name, "verdicts": [r.get("verdict", "") for r in result.responses],
            "marks": marks}


def repeats_markdown(rows: list[dict]) -> str:
    out = ["# Repeats in the kept cut", "",
           "Proposals only. `votes` is how many samples named the passage.", "",
           "| at | seconds | votes | repeats | why | text |", "|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {_clock(r['out_start'])} | {r['seconds']} | {r['votes']}/{r['of']} | "
                   f"{r['earlier']} at {_clock(r['earlier_out'])} | {r['why']} | {r['text'][:120]} |")
    return "\n".join(out) + "\n"


def cold_markdown(read: dict) -> str:
    out = [f"# Cold read: {read['title']}", "", f"Reader: {read['judge']}. Proposals only.", ""]
    out += [f"Verdict: {v}" for v in read["verdicts"]]
    out += ["", "| at | seconds | reason | why | text |", "|---|---|---|---|---|"]
    for m in read["marks"]:
        out.append(f"| {_clock(m['out_start'])} | {m['seconds']} | {m['reason']} | {m['why']} | {m['text'][:120]} |")
    return "\n".join(out) + "\n"
