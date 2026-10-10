"""Retakes and stumbles, in two passes.

Pass A sweeps the whole transcript for abandoned attempts, but aggregates votes by
SPAN rather than by word. A per-word supermajority turns valid proposals into zero
removed words whenever the samples that agree a retake exists disagree about its
edges by a word or two and cancel each other out.

Pass B: a mechanical scan finds every spoken restart marker ("sorry",
"no wait", "hold on"), then the judge adjudicates each one ALONE, with the
surrounding sentences for context. Focused calls are where the judge is
reliable; a 4361-word needle-hunt is not.

The bias is tiered. On STRONG markers the default is to CUT and the judge must
argue to keep -- a spoken apology is strong evidence of a real retake. On WEAK
markers ("I mean") the "when unsure, keep" asymmetry stands, because ordinary
speech uses these phrases constantly and cutting on them removes real content.
"""

from __future__ import annotations

import re

from ..context import Context
from ..judge import vote
from ..markers import find_markers
from ..prompts import (
    MARKER_SCHEMA,
    RETAKE_SCHEMA,
    annotated_sentence_table,
    marker_prompt,
    marker_window,
    retake_prompt,
)
from ..spans import Span, cluster_spans
from ..timeline import Cut

_WID = re.compile(r"^w(\d{4,})$")


def _wid(s: str, n: int) -> int | None:
    m = _WID.match(str(s))
    if not m:
        return None
    i = int(m.group(1))
    return i if i < n else None


def _quote(words: list, a: int, b: int, pad: int = 0) -> str:
    lo, hi = max(0, a - pad), min(len(words), b + 1 + pad)
    return " ".join(w.text for w in words[lo:hi])


def _pass_a(ctx: Context, table: str) -> tuple[list[Span], dict]:
    """Global sweep; span-level voting, intersection-leaning boundaries."""
    n = len(ctx.words)
    result = vote(ctx.judge, retake_prompt(table, ctx.topic), RETAKE_SCHEMA,
                  k=ctx.config.votes, cache=ctx.llm_cache)
    per_response: list[list[tuple[int, int]]] = []
    dropped_bad = dropped_overlong = proposed = 0
    for resp in result.responses:
        spans: list[tuple[int, int]] = []
        for r in resp.get("retakes", []):
            a, b = _wid(r.get("from_word"), n), _wid(r.get("to_word"), n)
            if a is None or b is None or b < a:
                dropped_bad += 1
                continue
            proposed += 1
            if ctx.words[b].end - ctx.words[a].start > ctx.config.retake_max_sec:
                dropped_overlong += 1
                continue
            spans.append((a, b))
        per_response.append(spans)
    kept = cluster_spans(per_response, agree=ctx.config.agree, mode="narrow")
    notes = {"proposed": proposed, "dropped_bad_ids": dropped_bad,
             "dropped_overlong": dropped_overlong, "spans": len(kept),
             "prompt_sha": result.prompt_sha}
    return kept, notes


def _pass_b(ctx: Context) -> tuple[list[Span], list[dict], dict]:
    """Marker-triggered adjudication, one candidate at a time."""
    words = ctx.words
    n = len(words)
    sents, groups = ctx.sentences(), ctx.sentence_words()
    cands = find_markers(words)
    spans: list[Span] = []
    log: list[dict] = []
    cut = keep = dropped_overlong = 0
    for cand in cands:
        marker_end = cand.word + len(cand.marker.split()) - 1
        window = marker_window(sents, groups, words, cand.word)
        if not window:
            continue
        result = vote(ctx.judge, marker_prompt(window, cand.marker, cand.tier, ctx.topic),
                      MARKER_SCHEMA, k=ctx.config.marker_votes, cache=ctx.llm_cache)
        cut_votes = sum(1 for r in result.responses if r.get("verdict") == "cut")
        keep_votes = sum(1 for r in result.responses if r.get("verdict") == "keep")
        if cand.tier == "strong":
            verdict = "keep" if keep_votes >= ctx.config.marker_agree else "cut"
        else:
            verdict = "cut" if cut_votes >= ctx.config.marker_agree else "keep"
        why = next((r.get("why", "") for r in result.responses if r.get("verdict") == verdict), "")
        entry = {"word": cand.word, "time": round(words[cand.word].start, 2),
                 "marker": cand.marker, "tier": cand.tier, "verdict": verdict,
                 "cut_votes": cut_votes, "keep_votes": keep_votes, "why": why}
        if verdict == "keep":
            keep += 1
            entry["quote"] = _quote(words, cand.word, marker_end, pad=6)
            log.append(entry)
            continue
        # Ranges from the responses that voted to cut; union-leaning, because
        # the guidelines prefer one larger cut over a pile of tiny ones.
        proposals: list[list[tuple[int, int]]] = []
        for r in result.responses:
            if r.get("verdict") != "cut":
                continue
            a, b = _wid(r.get("from_word"), n), _wid(r.get("to_word"), n)
            if a is None or b is None or b < a:
                continue
            # The marker belongs to the abandoned attempt: extend a range that
            # stops short of it, rather than leaving the apology in.
            if a <= marker_end and b < marker_end:
                b = marker_end
            proposals.append([(a, b)])
        clustered = cluster_spans(proposals, agree=1, mode="wide") if proposals else []
        if clustered:
            span = max(clustered, key=lambda s: s.b - s.a)
        else:
            # No usable range: fall back to the marker's sentence head through
            # the marker itself, which is the abandoned attempt by construction.
            home = next((idx for idx in groups if cand.word in idx), None)
            if home is None:
                continue
            span = Span(home[0], marker_end, frozenset({0}))
        if words[span.b].end - words[span.a].start > ctx.config.retake_max_sec:
            dropped_overlong += 1
            entry["verdict"] = "keep"
            entry["why"] = f"span {words[span.b].end - words[span.a].start:.1f}s over retake_max_sec"
            entry["quote"] = _quote(words, cand.word, marker_end, pad=6)
            log.append(entry)
            keep += 1
            continue
        cut += 1
        entry["words"] = [span.a, span.b]
        entry["quote"] = _quote(words, span.a, span.b)
        log.append(entry)
        spans.append(span)
    notes = {"candidates": len(cands), "cut": cut, "keep": keep,
             "dropped_overlong": dropped_overlong}
    return spans, log, notes


def _merge(spans: list[Span]) -> list[tuple[int, int, int]]:
    """Overlapping or adjacent spans fuse; the strongest vote count survives."""
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: (s.a, s.b))
    out: list[list[int]] = [[ordered[0].a, ordered[0].b, ordered[0].votes]]
    for s in ordered[1:]:
        if s.a <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], s.b)
            out[-1][2] = max(out[-1][2], s.votes)
        else:
            out.append([s.a, s.b, s.votes])
    return [(a, b, v) for a, b, v in out]


def run(ctx: Context) -> tuple[list[Cut], dict]:
    words = ctx.words
    table = annotated_sentence_table(ctx.sentences(), ctx.sentence_words(), words)
    a_spans, a_notes = _pass_a(ctx, table)
    b_spans, log, b_notes = _pass_b(ctx)
    cuts = [
        Cut(words[a].start, words[b].end, "retake",
            " ".join(w.text for w in words[a:b + 1]),
            provenance={"words": [a, b], "votes": v})
        for a, b, v in _merge(a_spans + b_spans)
    ]
    notes = {"retakes": {
        "cuts": len(cuts),
        "pass_a": a_notes,
        "pass_b": b_notes,
        "decisions": log,
        # Kept at the top level for report compatibility.
        "dropped_bad_ids": a_notes["dropped_bad_ids"],
        "dropped_overlong": a_notes["dropped_overlong"] + b_notes["dropped_overlong"],
    }}
    return cuts, notes
