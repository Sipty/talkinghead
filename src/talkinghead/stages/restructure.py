"""Restructure: topic blocks, a payoff gate, and a front-loaded opening.

Opt-in (`restructure = true`). The kept transcript (sentences that no structural,
retake, content or pause cut removes) is cut into blocks in three passes:

1. Windows: each window's core is judged with read-only context on both sides.
   The judge starts blocks inside the core and scores each one (one sample per window).
2. Merge: the window answers partition the kept sentences. A block belongs to the
   window whose core holds its start sentence.
3. Global: one vote over the block list. A block is dropped only when a strict
   majority says it has no payoff. Up to `restructure_front` blocks that stand
   without earlier context are ranked (Borda count) to open the video.

The stage memo holds the blocks with their decisions, so a hand-edited memo is
honoured on replay. `block_cuts` turns decisions into cuts; `reorder` lays the
kept audio out with the openers first and records the block layout in the plan.
"""

from __future__ import annotations

import bisect
import hashlib
import inspect
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

from ..context import Context
from ..edges import EdgeContext
from ..judge import ClaudeCodeJudge, Judge, ShellJudge, vote
from ..prompts import sentence_table, with_preferences
from ..semantic import Sentence
from ..timeline import Cut, Plan, Segment

WINDOW_SCHEMA = {
    "type": "object",
    "properties": {"blocks": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "start": {"type": "string"},
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "payoff": {"type": ["string", "null"]},
            "payoff_sid": {"type": ["string", "null"]},
            "interest": {"type": "integer", "minimum": 1, "maximum": 10},
            "relevance": {"type": "integer", "minimum": 0, "maximum": 10},
            "depends_on_earlier": {"type": "boolean"},
            "depends_note": {"type": "string"},
            "lead_in": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["start", "title", "summary", "payoff", "interest", "relevance", "depends_on_earlier"],
    }}},
    "required": ["blocks"],
}

KEEP_SCHEMA = {
    "type": "object",
    "properties": {"blocks": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "decision": {"type": "string", "enum": ["keep", "drop"]},
            "reason": {"type": "string"},
            "front_rank": {"type": ["integer", "null"], "minimum": 1},
        },
        "required": ["id", "decision", "reason", "front_rank"],
    }}},
    "required": ["blocks"],
}

WINDOW_RULES = (
    "This is one stretch of a long conversation, cut into topic blocks. Each block is a run of "
    "sentences about one point. The video is rebuilt from these blocks: some are dropped, a few "
    "open the video, the rest stay in their original order.\n\n"
    "Start blocks ONLY on sentences in the CORE section. Context sentences are read only: they show "
    "what comes before and after and must not start a block. Blocks cover the sentences in order: a "
    "block runs until the next block starts.\n\n"
    "For each block answer:\n"
    "- start: the sentence id where the block starts (CORE only).\n"
    "- title: a short viewer-facing title.\n"
    "- summary: one line on what the block says.\n"
    "- payoff: what the viewer gets: an insight or take, a story with a punchline, a real laugh, a "
    "concrete tip, or a disagreement settled. null when there is none (a tangent that goes nowhere, "
    "warm-up, small talk without a point).\n"
    "- payoff_sid: the sentence id where the payoff lands, or null.\n"
    "- interest: 1 to 10, how gripping the block is on its own.\n"
    "- relevance: 0 to 10, how well it serves the owner's brief (0 when there is no brief).\n"
    "- depends_on_earlier: true when the block only makes sense after earlier material.\n"
    "- depends_note: why, in a few words (empty when false).\n"
    "- lead_in: sentence ids at the start of the block that only make sense in the original order "
    "(\"as you said\", \"anyway, back to\"). Empty when none.\n"
)

KEEP_RULES = (
    "Below are the topic blocks of one long video, in their original order, each with its scores. "
    "Decide for each block whether it stays, and which blocks open the video.\n\n"
    "Length is no concern. Drop a block ONLY when it has no payoff: a tangent that goes nowhere, "
    "warm-up, or small talk without a point. Keep everything else, even weak blocks.\n\n"
)


def _topic(topic: str | None) -> str:
    return f"Video topic (from the owner): {topic}\n\n" if topic else ""


def _brief(brief: str) -> str:
    if brief.strip():
        return f"--- OWNER BRIEF (follow it) ---\n{brief.strip()}\n\n"
    return "--- OWNER BRIEF ---\nNone given: judge each block for a general audience.\n\n"


def _mmss(t: float) -> str:
    m, s = divmod(t, 60.0)
    return f"{int(m):02d}:{s:05.2f}"


@dataclass
class Block:
    """One topic block: kept sentences `first..last` (indices into the kept list)."""

    id: str
    first: int
    last: int
    start: float
    end: float
    # False for a block no window judged (the lead before the first judged start).
    judged: bool = True
    title: str = ""
    summary: str = ""
    payoff: str | None = None
    payoff_sid: str | None = None
    interest: int = 0
    relevance: int = 0
    depends_on_earlier: bool = False
    depends_note: str = ""
    # [sid, start, end] of each lead-in sentence: cut when the block moves to the front.
    lead_in: list[list] = field(default_factory=list)
    # Speaker share of the block's words, e.g. {"S0": 0.7, "S1": 0.3}; empty without labels.
    share: dict[str, float] = field(default_factory=dict)
    decision: str = "keep"
    reason: str = ""
    front_rank: int | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


def policy() -> str:
    """Fingerprint of what the restructure judges are asked and how answers are read.

    The stage memo is keyed on it, so a changed prompt, schema or aggregation
    recomputes the blocks instead of replaying decisions made under the old rules.
    """
    sources = [inspect.getsource(fn) for fn in (window_prompt, keep_prompt, _window_blocks, _decide)]
    text = "\n".join(sources + [repr(WINDOW_SCHEMA), repr(KEEP_SCHEMA), WINDOW_RULES, KEEP_RULES])
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def concurrent_safe(judge: Judge) -> bool:
    """Only stateless judges may be called from several threads: PanelJudge and FallbackJudge keep per-call state."""
    return isinstance(judge, (ShellJudge, ClaudeCodeJudge))


def kept_sentences(ctx: Context, raw: list[Cut]) -> list[Sentence]:
    """The transcript sentences no cut in `raw` removes (by their midpoint)."""
    spans = [(c.start, c.end) for c in raw]
    return [s for s in ctx.sentences() if not any(a <= (s.start + s.end) / 2 < b for a, b in spans)]


@dataclass(frozen=True)
class Window:
    core: tuple[int, int]     # [lo, hi) sentence indices that may start blocks
    context: tuple[int, int]  # [lo, hi) including the read-only context either side


def windows(sents: list[Sentence], window_sec: float, context_sec: float) -> list[Window]:
    """Consecutive cores of about window_sec - 2*context_sec, cut at sentence boundaries."""
    if not sents:
        return []
    core_sec = window_sec - 2 * context_sec
    cores: list[tuple[int, int]] = []
    lo = 0
    for i in range(1, len(sents)):
        if sents[i].end - sents[lo].start > core_sec:
            cores.append((lo, i))
            lo = i
    cores.append((lo, len(sents)))
    out = []
    for a, b in cores:
        c_lo = a
        while c_lo > 0 and sents[c_lo - 1].end >= sents[a].start - context_sec:
            c_lo -= 1
        c_hi = b
        while c_hi < len(sents) and sents[c_hi].start <= sents[b - 1].end + context_sec:
            c_hi += 1
        out.append(Window((a, b), (c_lo, c_hi)))
    return out


def window_prompt(sents: list[Sentence], w: Window, brief: str, topic: str | None) -> str:
    a, b = w.core
    c_lo, c_hi = w.context
    before = sentence_table(sents[c_lo:a]) or "(none)"
    core = sentence_table(sents[a:b])
    after = sentence_table(sents[b:c_hi]) or "(none)"
    return (
        "TASK: RESTRUCTURE WINDOW\n"
        + _topic(topic)
        + WINDOW_RULES
        + "\n"
        + _brief(brief)
        + "--- CONTEXT BEFORE (read only, do not start blocks here) ---\n" + before + "\n\n"
        + "--- CORE (start blocks here) ---\n" + core + "\n\n"
        + "--- CONTEXT AFTER (read only, do not start blocks here) ---\n" + after + "\n"
    )


def _score(value: Any, top: int) -> int:
    try:
        return min(top, max(0, int(value)))
    except (TypeError, ValueError):
        return 0


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _window_blocks(resp: dict, sents: list[Sentence], sid_index: dict[str, int], w: Window) -> list[dict]:
    """The block starts a window answer makes inside its core; anything else is ignored."""
    out: list[dict] = []
    seen: set[int] = set()
    for item in resp.get("blocks") or []:
        if not isinstance(item, dict):
            continue
        idx = sid_index.get(str(item.get("start", "")))
        if idx is None or not (w.core[0] <= idx < w.core[1]) or idx in seen:
            continue
        seen.add(idx)
        out.append({
            "first": idx,
            "title": str(item.get("title") or "").strip(),
            "summary": str(item.get("summary") or "").strip(),
            "payoff": _text(item.get("payoff")),
            "payoff_sid": _text(item.get("payoff_sid")),
            "interest": _score(item.get("interest"), 10),
            "relevance": _score(item.get("relevance"), 10),
            "depends_on_earlier": bool(item.get("depends_on_earlier")),
            "depends_note": str(item.get("depends_note") or "").strip(),
            "lead_in_sids": [str(x) for x in (item.get("lead_in") or []) if isinstance(x, str)],
        })
    return out


def merge_windows(sents: list[Sentence], judged: list[dict], sid_index: dict[str, int]) -> list[Block]:
    """One contiguous partition of the kept sentences into blocks, ids B01, B02, ..."""
    if not sents:
        return []
    by_first: dict[int, dict] = {}
    for j in judged:
        by_first.setdefault(j["first"], j)
    starts = sorted(by_first)
    if not starts or starts[0] != 0:
        starts = [0] + starts
    blocks: list[Block] = []
    for k, first in enumerate(starts):
        last = starts[k + 1] - 1 if k + 1 < len(starts) else len(sents) - 1
        j = by_first.get(first)
        lead: list[list] = []
        if j is not None:
            idxs = sorted({sid_index[s] for s in j["lead_in_sids"] if s in sid_index})
            lead = [[sents[i].sid, sents[i].start, sents[i].end] for i in idxs if first <= i <= last]
        blocks.append(Block(
            id=f"B{k + 1:02d}", first=first, last=last, start=sents[first].start, end=sents[last].end,
            judged=j is not None,
            title=j["title"] if j else "", summary=j["summary"] if j else "",
            payoff=j["payoff"] if j else None, payoff_sid=j["payoff_sid"] if j else None,
            interest=j["interest"] if j else 0, relevance=j["relevance"] if j else 0,
            depends_on_earlier=j["depends_on_earlier"] if j else False,
            depends_note=j["depends_note"] if j else "", lead_in=lead,
        ))
    return blocks


def _attach_shares(words, blocks: list[Block]) -> None:
    """Each block's speaker share of its words, when the transcript carries speaker labels."""
    spoken = [w for w in words if w.speaker is not None]
    if not spoken:
        return
    for b in blocks:
        counts = Counter(w.speaker for w in spoken if b.start <= (w.start + w.end) / 2 < b.end)
        total = sum(counts.values())
        b.share = {k: round(v / total, 2) for k, v in sorted(counts.items())} if total else {}


def block_row(b: Block) -> str:
    parts = [b.id, f"{_mmss(b.start)}-{_mmss(b.end)} ({b.duration:.0f}s)"]
    if b.share:
        parts.append("speakers " + ", ".join(f"{k} {round(v * 100)}%" for k, v in b.share.items()))
    parts += [f"title: {b.title}", f"summary: {b.summary}", f"payoff: {b.payoff or 'none'}",
              f"interest {b.interest}/10", f"relevance {b.relevance}/10",
              "depends on earlier: " + (f"yes ({b.depends_note})" if b.depends_on_earlier else "no")]
    return " | ".join(parts)


def keep_prompt(blocks: list[Block], brief: str, topic: str | None, front: int) -> str:
    if front > 0:
        opener = (f"Pick at most {front} blocks as openers. An opener must work as the first thing a viewer "
                  "sees, with no earlier context (depends_on_earlier no). front_rank 1 is the best opener; "
                  "give null to every other block.\n\n")
    else:
        opener = "Do not pick any opener: front_rank is null for every block.\n\n"
    rows = "\n".join(block_row(b) for b in blocks)
    return (
        "TASK: RESTRUCTURE KEEP\n"
        + _topic(topic)
        + KEEP_RULES
        + opener
        + "For each block answer: id, decision (\"keep\" or \"drop\"), reason (one line), front_rank.\n\n"
        + _brief(brief)
        + "--- BLOCKS (original order) ---\n" + rows + "\n"
    )


def _rank(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _decide(blocks: list[Block], responses: list[dict], front: int) -> None:
    """Majority keep/drop per block and Borda-ranked openers, written onto the blocks."""
    n = len(responses)
    indexed = [{str(i.get("id")): i for i in (r.get("blocks") or []) if isinstance(i, dict)} for r in responses]
    for b in blocks:
        drops: list[str] = []
        keeps: list[str] = []
        for index in indexed:
            item = index.get(b.id)
            if item is None:
                continue  # a block a sample did not answer is a keep vote
            reason = str(item.get("reason") or "").strip()
            (drops if str(item.get("decision", "keep")).strip().lower() == "drop" else keeps).append(reason)
        dropped = 2 * len(drops) > n
        b.decision = "drop" if dropped else "keep"
        b.reason = " / ".join(sorted({r for r in (drops if dropped else keeps) if r})[:3])
    if front <= 0 or n == 0:
        return
    points = {b.id: 0 for b in blocks}
    ranked_by = {b.id: 0 for b in blocks}
    for index in indexed:
        ranks: dict[str, int] = {}
        for bid, item in index.items():
            rank = _rank(item.get("front_rank"))
            if rank is not None and bid in points:
                ranks[bid] = rank
        for bid, rank in ranks.items():
            ranked_by[bid] += 1
            if rank <= front:
                points[bid] += front + 1 - rank
    eligible = [b for b in blocks if b.decision == "keep" and not b.depends_on_earlier
                and 2 * ranked_by[b.id] > n and points[b.id] > 0]
    eligible.sort(key=lambda b: (-points[b.id], b.first))
    for rank, b in enumerate(eligible[:front], start=1):
        b.front_rank = rank


def _judge_windows(ctx: Context, sents, sid_index, wins: list[Window], prompts: list[str]) -> list[list[dict]]:
    cfg = ctx.config

    def one(i: int) -> list[dict]:
        result = vote(ctx.judge, prompts[i], WINDOW_SCHEMA, k=1, cache=ctx.llm_cache)
        resp = result.responses[0] if result.responses else {}
        return _window_blocks(resp, sents, sid_index, wins[i])

    if cfg.restructure_parallel > 1 and len(wins) > 1 and concurrent_safe(ctx.judge):
        with ThreadPoolExecutor(max_workers=cfg.restructure_parallel) as pool:
            return list(pool.map(one, range(len(wins))))
    return [one(i) for i in range(len(wins))]


def run(ctx: Context, raw: list[Cut]) -> dict:
    """The stage value: `{"windows", "global_prompt_sha", "order", "blocks": [Block as dict]}`.

    `order` is the output sequence of the kept blocks (openers by rank, then source order). Replay
    reads the decisions and front ranks in `blocks`; `order` is derived from them and written for reading.
    """
    cfg = ctx.config
    sents = kept_sentences(ctx, raw)
    sid_index = {s.sid: i for i, s in enumerate(sents)}
    wins = windows(sents, cfg.restructure_window_sec, cfg.restructure_context_sec)
    prompts = [with_preferences(window_prompt(sents, w, cfg.restructure_brief, ctx.topic), ctx.preferences)
               for w in wins]
    judged = [j for lst in _judge_windows(ctx, sents, sid_index, wins, prompts) for j in lst]
    blocks = merge_windows(sents, judged, sid_index)
    _attach_shares(ctx.words, blocks)
    value: dict = {"windows": len(wins), "global_prompt_sha": None}
    votable = [b for b in blocks if b.judged]
    if votable:
        prompt = with_preferences(keep_prompt(votable, cfg.restructure_brief, ctx.topic, cfg.restructure_front),
                                  ctx.preferences)
        result = vote(ctx.judge, prompt, KEEP_SCHEMA, k=cfg.restructure_votes, cache=ctx.llm_cache)
        _decide(votable, result.responses, cfg.restructure_front)
        value["global_prompt_sha"] = result.prompt_sha
    value["order"] = _order(blocks)
    value["blocks"] = [asdict(b) for b in blocks]
    return value


def _order(blocks: list[Block]) -> list[str]:
    front = sorted((b for b in blocks if b.decision == "keep" and b.front_rank is not None),
                   key=lambda b: b.front_rank)
    rest = [b for b in blocks if b.decision == "keep" and b.front_rank is None]
    return [b.id for b in front + rest]


def blocks_from(value: dict) -> list[Block]:
    return [Block(**d) for d in value["blocks"]]


def summary(value: dict) -> dict:
    """Counts for the report's stage notes."""
    blocks = blocks_from(value)
    return {
        "windows": value["windows"],
        "blocks": len(blocks),
        "unjudged": sum(1 for b in blocks if not b.judged),
        "dropped": sum(1 for b in blocks if b.decision == "drop"),
        "front": sum(1 for b in blocks if b.decision == "keep" and b.front_rank is not None),
    }


def block_cuts(blocks: list[Block]) -> list[Cut]:
    """Dropped blocks are word-removing 'tangent' cuts; openers lose their lead-in sentences."""
    cuts: list[Cut] = []
    for b in blocks:
        if b.decision == "drop":
            cuts.append(Cut(b.start, b.end, "tangent", b.title, {"block": b.id, "why": b.reason}))
        elif b.front_rank is not None:
            for sid, a, c in b.lead_in:
                cuts.append(Cut(a, c, "lead-in", sid, {"block": b.id}))
    return cuts


def _quiet_point(ctx: EdgeContext, lo: float, hi: float) -> float:
    """The quietest instant between two sentences: on the frame grid when the render has one."""
    if hi < lo:
        lo, hi = hi, lo
    mid = (lo + hi) / 2
    fs = ctx.env.frame_sec
    if ctx.fps:
        grid = ctx.fps
        points = [k / grid for k in range(math.ceil(lo * grid - 1e-6), math.floor(hi * grid + 1e-6) + 1)]
        points = points or [round(mid * grid) / grid]
    else:
        count = int((hi - lo) / fs) + 1
        points = [lo + i * (hi - lo) / max(1, count - 1) for i in range(count)] if hi > lo else [lo]
    db = ctx.env.db
    n = len(db)

    def level(t: float) -> float:
        i = min(n - 1, max(0, int(round(t / fs))))
        return float(db[max(0, i - 4):min(n, i + 5)].mean())

    return min(points, key=lambda t: (round(level(t), 1), abs(t - mid)))


def reorder(plan: Plan, blocks: list[Block], ctx: EdgeContext) -> Plan:
    """Lay the kept audio out with the openers first, the rest in source order.

    The plan is split at every block boundary (the quietest point between two
    sentences), each piece is owned by one block, openers' pieces come first in
    rank order, and contiguous pieces that stay in source order are re-fused.
    Output length is unchanged. The layout (output range per block) goes into
    `plan.provenance["blocks"]`.
    """
    if not blocks:
        return plan
    bounds = [_quiet_point(ctx, prev.end, cur.start) for prev, cur in zip(blocks, blocks[1:])]
    front = sorted((b for b in blocks if b.decision == "keep" and b.front_rank is not None),
                   key=lambda b: b.front_rank)
    front_ids = {b.id for b in front}
    pieces: list[tuple[Segment, str | None]] = []
    for seg in plan.segments:
        if seg.source != plan.main:
            pieces.append((seg, None))
            continue
        edges = [seg.start, *[t for t in bounds if seg.start < t < seg.end], seg.end]
        for a, c in zip(edges, edges[1:]):
            if c > a:
                owner = blocks[bisect.bisect_right(bounds, (a + c) / 2)].id
                pieces.append((Segment(seg.source, a, c), owner))
    ordered = [p for b in front for p in pieces if p[1] == b.id]
    ordered += [p for p in pieces if p[1] not in front_ids]

    spans: dict[str, list[float]] = {}
    offset = 0.0
    for seg, owner in ordered:
        if owner is not None:
            spans.setdefault(owner, [offset, offset])[1] = offset + seg.length
        offset += seg.length
    segments: list[Segment] = []
    for seg, _ in ordered:
        if segments and segments[-1].source == seg.source and segments[-1].end == seg.start:
            segments[-1] = Segment(seg.source, segments[-1].start, seg.end)
        else:
            segments.append(seg)

    layout = []
    for b in blocks:
        span = spans.get(b.id)
        kept = b.decision == "keep"
        layout.append({
            "id": b.id, "title": b.title, "decision": b.decision, "reason": b.reason,
            "source": [round(b.start, 3), round(b.end, 3)],
            "output": [round(span[0], 3), round(span[1], 3)] if kept and span is not None else None,
            "front_rank": b.front_rank if kept else None,
            "interest": b.interest, "relevance": b.relevance, "judged": b.judged,
        })
    plan.segments = segments
    plan.provenance = {**plan.provenance, "blocks": layout}
    return plan
