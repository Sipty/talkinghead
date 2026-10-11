"""The visuals stage: one hierarchical editorial plan over the FINAL cut.

The judge reads only the finished cut's transcript, so its anchors are
output-timeline phrases and one proposal call returns the argument, chapters
and sparse visuals together. Each anchor records HOW it got its visual
(reuse | researched | quote_card | dropped | unresolved); a meme or evidence
anchor with no asset is dropped, never degraded into a generic text plate.
"""

from __future__ import annotations

from pathlib import Path

from ..context import Context
from ..judge import vote
from ..prompts import (
    EDITORIAL_SCHEMA,
    RESEARCH_SCHEMA,
    editorial_prompt,
    research_prompt,
    transcript_text,
    with_preferences,
)
from ..timeline import Plan
from .anchors import Anchor, find_phrase, find_quote, resolve_output
from .cards import card_sha, reading_seconds, render_card
from .library import Library, tokens
from .plancard import PlanItem, plan_sha, render_plan
from .research import ResearchRequest, chrome_fetcher, download_image, fetch, fetch_image
from .schedule import Placement, free_slot_after, schedule


def _dedup(visuals: list[dict]) -> list[dict]:
    """Drop verbatim repeats of the same (anchor, type) inside one answer."""
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for v in visuals:
        key = (" ".join(str(v.get("anchor_text", "")).lower().split()),
               str(v.get("type", "")))
        if not key[0] or key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


# Two candidates describe the same moment when they sit close together in the
# transcript AND either name the same card, or describe the same thing.
NEAR_WORDS = 60           # roughly 20 s of speech
SAME_DEPICTS = 0.4        # Jaccard over visual_brief tokens


def _same_moment(a: dict, b: dict) -> bool:
    if abs(a["word"] - b["word"]) > NEAR_WORDS:
        return False
    aa, ab = a.get("asset"), b.get("asset")
    # The same asset twice is always one moment, however differently the two
    # candidates described it.
    if aa is not None and ab is not None and aa.id == ab.id:
        return True
    # Steps of one build-up sit a second apart and describe near-identical
    # things; collapsing them would delete the animation.
    sa = aa.sequence if aa is not None else None
    sb = ab.sequence if ab is not None else None
    if sa and sa == sb and aa.id != ab.id:
        return False
    ta, tb = a["title_key"], b["title_key"]
    if ta and ta == tb:
        return True
    da, db = a["depicts_tokens"], b["depicts_tokens"]
    if not da or not db:
        return False
    return len(da & db) / len(da | db) >= SAME_DEPICTS


def collapse_moments(enriched: list[dict]) -> list[list[int]]:
    """Group candidate indices by moment (union-find over `_same_moment`)."""
    parent = list(range(len(enriched)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(enriched)):
        for j in range(i + 1, len(enriched)):
            if _same_moment(enriched[i], enriched[j]):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(enriched)):
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for _, g in sorted(groups.items(), key=lambda kv: min(kv[1]))]


def pick_winner(group: list[int], enriched: list[dict]) -> int:
    """Best candidate for a moment: the highest library match wins.

    On a tie the longer hold wins, because the judge asked for more screen
    time on that topic.
    """
    return max(group, key=lambda i: (enriched[i]["score"],
                                     enriched[i]["hold"],
                                     -enriched[i]["word"]))


def gather_editorial(ctx: Context, words: list) -> tuple[dict, str]:
    """The single proposal call: argument, chapters and visuals in one answer."""
    result = vote(ctx.judge, with_preferences(editorial_prompt(transcript_text(words), ctx.topic,
                                                               ctx.config.plan_chapters), ctx.preferences),
                  EDITORIAL_SCHEMA, k=1, cache=ctx.llm_cache)
    resp: dict = result.responses[0] if result.responses else {}
    return {
        "central_argument": " ".join(str(resp.get("central_argument") or "").split()),
        # Clamp in code, not just the prompt: a model that over-delivers
        # chapters must not overflow the roadmap card.
        "chapters": list(resp.get("chapters") or [])[:ctx.config.plan_chapters],
        "visuals": list(resp.get("visuals") or []),
    }, result.prompt_sha


def research_urls(ctx: Context, items: list[dict]) -> dict[int, str]:
    """One batched call: the judge names a canonical URL per item."""
    if not items:
        return {}
    result = vote(ctx.judge, research_prompt(items, ctx.topic), RESEARCH_SCHEMA,
                  k=1, cache=ctx.llm_cache)
    urls: dict[int, str] = {}
    for resp in result.responses:
        for s in resp.get("sources", []):
            i = s.get("id")
            url = str(s.get("url", "")).strip()
            if isinstance(i, int) and url.startswith("http"):
                urls.setdefault(i, url)
    return urls


def place_plan(ctx: Context, plan: Plan, chapters: list[dict], *,
               words: list, work_dir: Path) -> tuple[list[Placement], dict]:
    """Chapter roadmap cards.

    The opening card sits near the start; every later card begins at its own
    chapter's anchor and owns that corner, so content schedules around it.
    """
    cfg = ctx.config
    work = Path(work_dir)
    notes: dict = {"chapters": len(chapters), "placed": 0, "reveals": []}
    if not chapters:
        notes["note"] = "the editorial judge returned no chapters"
        return [], notes

    titles = [str(c.get("title", "")).strip() for c in chapters]
    by_n: dict[int, dict] = {}
    marks: list[tuple[int, float]] = []
    for n, c in enumerate(chapters, start=1):
        anchor = " ".join(str(c.get("anchor_text", "")).split())
        row = {"n": n, "title": titles[n - 1], "anchor": anchor,
               "purpose": " ".join(str(c.get("purpose", "")).split())}
        i = find_phrase(words, anchor)
        if i is None:
            row.update({"decision": "unresolved",
                        "note": "chapter anchor not found verbatim in the final transcript"})
        else:
            out = float(words[i].start)
            row["chapter_at"] = round(out, 2)
            marks.append((n, out))
        notes["reveals"].append(row)
        by_n[n] = row

    scale = cfg.visuals_scale_card
    taken: list[tuple[float, float]] = []
    out_placements: list[Placement] = []

    # The opening roadmap belongs near the START even when chapter one starts at
    # 0 s; chapter one is on this card, so the bound is the NEXT chapter card.
    limit = plan.output_duration
    if len(marks) > 1:
        limit = min(marks[1][1], limit)
    opening = open_hold = None
    for hold in (cfg.plan_open_hold, cfg.plan_hold, cfg.visuals_min_hold):
        if hold <= 0:
            continue
        slot = free_slot_after(cfg.plan_open_at, hold=hold, taken=taken,
                               min_gap=cfg.visuals_min_gap, limit=limit)
        if slot is not None:
            opening, open_hold = slot, hold
            break
    if opening is not None:
        items = [PlanItem(t, i == 0) for i, t in enumerate(titles)]
        aid = plan_sha(items, "THE PLAN", None, cfg.visuals_avatar)
        path = render_plan(items, work / f"{aid}.png", title="THE PLAN", avatar=cfg.visuals_avatar)
        out_placements.append(Placement(aid, str(path), opening, opening + open_hold, scale))
        taken.append((opening, opening + open_hold))
        notes["opening_at"] = round(opening, 2)
        notes["opening_hold"] = open_hold
        # Only claim chapter one is on the card if it actually resolved.
        row_one = by_n.get(1)
        if row_one is not None and row_one.get("decision") != "unresolved":
            row_one["decision"] = "on the opening card"
            row_one["at"] = round(opening, 2)
    else:
        notes["note"] = "no room for the opening card"

    for n, mark in marks:
        if n == 1:
            continue                       # already highlighted on the opening card
        row = by_n.get(n, {})
        start = float(mark)
        if taken:
            # Only the card before it can push it: the anchor owns the start.
            start = max(start, taken[-1][1] + cfg.visuals_min_gap)
        if start + cfg.plan_hold > plan.output_duration:
            row["decision"] = "no-room"
            row["note"] = (f"chapter starts at {mark:.1f}s; no {cfg.plan_hold:.0f}s window "
                           "left in the cut for its card")
            continue
        items = [PlanItem(t, i == n - 1) for i, t in enumerate(titles)]
        aid = plan_sha(items, "THE PLAN", None, cfg.visuals_avatar)
        path = render_plan(items, work / f"{aid}.png", title="THE PLAN", footer=None, avatar=cfg.visuals_avatar)
        out_placements.append(Placement(aid, str(path), start, start + cfg.plan_hold, scale))
        taken.append((start, start + cfg.plan_hold))
        row["decision"] = "placed"
        row["at"] = round(start, 2)
        row["current"] = n
    notes["placed"] = len(out_placements)
    return out_placements, notes


def _chapter_spans(words: list, chapters: list[dict], output_duration: float) -> dict[int, tuple[int, int, float, float]]:
    """Word-index and output intervals per RESOLVED chapter, from its anchor.

    `{n: (lo, hi, start_out, end_out)}`: [lo, hi) is the chapter's word range,
    [start_out, end_out] its output range (end_out = the next resolved
    chapter's anchor, or the cut end). A chapter with no verbatim anchor is
    absent, which makes a visual referencing it invalid.
    """
    anchors = [find_phrase(words, " ".join(str(c.get("anchor_text", "")).split()))
               for c in chapters]
    spans: dict[int, tuple[int, int, float, float]] = {}
    for idx, i in enumerate(anchors):
        if i is None:
            continue
        nxt = next((anchors[j] for j in range(idx + 1, len(anchors))
                    if anchors[j] is not None), None)
        hi = nxt if nxt is not None and nxt > i else len(words)
        end_out = float(words[hi].start) if hi < len(words) else output_duration
        spans[idx + 1] = (i, hi, float(words[i].start), end_out)
    return spans


def plan_visuals(
    ctx: Context, plan: Plan, *, words: list, assets_dir: str | Path, work_dir: str | Path,
    fetcher=chrome_fetcher, image_fetcher=download_image,
) -> tuple[list[Placement], dict]:
    """Turn one editorial plan into placements on the output timeline.

    `words` is the FINAL cut transcript (`facts["words"]`): its word starts are
    output times, so anchors never pass through the raw Plan. Each visual stays
    inside its declared chapter's word range and output interval.
    """
    cfg = ctx.config
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    library = Library.load(assets_dir)
    editorial, prompt_sha = gather_editorial(ctx, words)
    candidates = _dedup(editorial["visuals"]) if cfg.visuals_content_enabled else []
    chapters = editorial["chapters"]
    spans = _chapter_spans(words, chapters, plan.output_duration) if chapters else {}
    whole = (0, len(words), 0.0, plan.output_duration)

    # Locate and score every candidate first. Both steps are pure and cheap, so
    # duplicates are collapsed BEFORE anything is researched or rasterized.
    enriched: list[dict] = []
    decisions: list[dict] = []
    for v in candidates:
        vtype = str(v.get("type", "")).strip()
        anchor_text = " ".join(str(v.get("anchor_text", "")).split())
        quote = " ".join(str(v.get("quote") or "").split())
        brief = " ".join(str(v.get("visual_brief") or "").split())
        reason = " ".join(str(v.get("editorial_reason") or "").split())
        hold = max(3.0, min(20.0, float(v.get("hold_sec") or 8.0)))
        chapter = v.get("chapter")
        row = {"anchor": anchor_text, "type": vtype, "quote": quote, "chapter": chapter,
               "hold": hold}
        if chapters and (not isinstance(chapter, int) or chapter not in spans):
            row.update({"decision": "dropped", "match": 0.0, "asset": "",
                        "note": "chapter reference is missing, out of range, or has no anchor"})
            decisions.append(row)
            continue
        lo, hi, _start_out, end_out = spans.get(chapter, whole) if chapters else whole
        word = find_phrase(words[lo:hi], anchor_text)
        if word is None:
            row.update({"decision": "unresolved", "match": 0.0, "asset": "",
                        "note": "anchor phrase not found verbatim in its chapter"})
            decisions.append(row)
            continue
        word += lo
        if vtype == "quote_card":
            asset, score = None, 0.0
        else:
            asset, score = library.match(f"{brief} {anchor_text}", ctx.topic)
        enriched.append({
            "anchor": anchor_text, "type": vtype, "quote": quote, "brief": brief,
            "reason": reason, "chapter": chapter, "hold": hold, "word": word,
            "lo": lo, "hi": hi, "until": end_out,
            "score": score, "asset": asset,
            "title_key": " ".join((quote or brief).lower().split()),
            "depicts_tokens": tokens(brief or quote),
        })

    groups = collapse_moments(enriched)
    winner_of: dict[int, int] = {}
    winners: set[int] = set()
    for g in groups:
        w = pick_winner(g, enriched)
        winners.add(w)
        for i in g:
            winner_of[i] = w

    # Losers are recorded as duplicates: they were real candidates for a beat
    # another candidate covers better.
    for i, e in enumerate(enriched):
        if i in winners:
            continue
        w = enriched[winner_of[i]]
        decisions.append({
            "anchor": e["anchor"], "type": e["type"], "chapter": e["chapter"],
            "match": round(e["score"], 2), "asset": "", "decision": "duplicate",
            "note": f"same moment as {w['anchor'][:34]!r} (match {w['score']:.2f})",
        })

    want_shot = [i for i in sorted(winners)
                 if enriched[i]["type"] != "quote_card" and cfg.visuals_research
                 and enriched[i]["score"] < cfg.visuals_match_threshold]
    # research_prompt reads `depicts`; enriched rows describe the brief. The
    # type lets a meme ask for a direct image and evidence for a canonical page.
    urls = research_urls(ctx, [{"type": enriched[i]["type"],
                                "depicts": enriched[i]["brief"] or enriched[i]["anchor"],
                                "anchor_text": enriched[i]["anchor"]}
                               for i in want_shot]) if want_shot else {}
    url_by_candidate = {ci: urls[k] for k, ci in enumerate(want_shot) if k in urls}

    anchors: list[Anchor] = []
    paths: dict[str, str] = {}
    row_of: dict[str, dict] = {}
    for i in sorted(winners):
        e = enriched[i]
        row = {"anchor": e["anchor"], "type": e["type"], "chapter": e["chapter"],
               "match": round(e["score"], 2), "hold": e["hold"]}
        if e["type"] == "quote_card":
            # A quote card renders the words ACTUALLY spoken: find_quote keeps
            # meaningful punctuation, so the model's "$1.50" can never become a
            # spoken "150" and then be rendered as invented text.
            span = find_quote(words[e["lo"]:e["hi"]], e["quote"])
            if span is None:
                row.update({"decision": "unresolved",
                            "note": "quote is not verbatim in the final transcript"})
                decisions.append(row)
                continue
            spoken = " ".join(w.text for w in words[e["lo"] + span[0]:e["lo"] + span[1]])
            aid = f"q-{card_sha(spoken, None, 'term', cfg.visuals_avatar)}"
            path = str(render_card(spoken, None, "term", work / f"{aid}.png", avatar=cfg.visuals_avatar))
            scale = cfg.visuals_scale_card
            hold = max(e["hold"], reading_seconds(spoken, None))
            row.update({"decision": "quote_card", "asset": aid, "quote": spoken})
        elif e["asset"] is not None and e["score"] >= cfg.visuals_match_threshold:
            aid, path = e["asset"].id, str(e["asset"].path)
            scale = e["asset"].scale(card=cfg.visuals_scale_card, shot=cfg.visuals_scale_shot)
            hold = e["hold"]
            row.update({"decision": "reuse", "asset": aid})
        else:
            got = None
            brief = e["brief"] or e["anchor"]
            if i in url_by_candidate:
                aid = f"r-{card_sha(brief, None, 'shot')}"
                req = ResearchRequest(e["anchor"], brief, url_by_candidate[i])
                if e["type"] == "meme":
                    got = fetch_image(req, work / f"{aid}.png", fetcher=image_fetcher)
                else:
                    got = fetch(req, work / f"{aid}.png", fetcher=fetcher)
            if got is not None:
                path, scale, hold = str(got), cfg.visuals_scale_shot, e["hold"]
                row.update({"decision": "researched", "asset": aid, "url": url_by_candidate[i]})
                library.ingest(got, id=aid, depicts=brief, kind="shot")
            else:
                # A meme or a piece of evidence that does not exist is dropped,
                # never degraded into a text card or a page screenshot.
                note = ("no direct meme image found; no generic card"
                        if e["type"] == "meme"
                        else "no matching asset and no source found; no generic card")
                row.update({"decision": "dropped", "note": note})
                decisions.append(row)
                continue
        reused = e["asset"] is not None and row["decision"] == "reuse"
        seq = e["asset"].sequence if reused else None
        order = e["asset"].order if reused else 0
        anchors.append(Anchor(e["anchor"], aid, hold, scale, kind=e["type"], score=e["score"],
                              sequence=seq, order=order, until=e["until"], word=e["word"]))
        paths[aid] = path
        row_of[aid] = row
        decisions.append(row)

    # Anchors resolve DIRECTLY on the final transcript: words[i].start is
    # already an output time.
    resolved = []
    for a in anchors:
        r = resolve_output(a, words, path=paths[a.asset])
        row = row_of[a.asset]
        if r is None:
            row["decision"] = "unresolved"
            row["note"] = "anchor phrase was not found verbatim in the final transcript"
            continue
        row["out_t"] = round(r.out_t, 2)
        resolved.append(r)

    # The chapter cards are placed FIRST and own their corner; content overlays
    # are scheduled around them.
    plan_places: list[Placement] = []
    plan_notes: dict = {}
    if cfg.plan_enabled:
        plan_places, plan_notes = place_plan(ctx, plan, editorial["chapters"],
                                             words=words, work_dir=work)
        plan_notes["prompt_sha"] = prompt_sha

    placements, dropped = schedule(
        resolved, out_duration=plan.output_duration,
        min_hold=cfg.visuals_min_hold, min_gap=cfg.visuals_min_gap,
        reserved=[(p.start, p.end) for p in plan_places])
    placed_assets = {p.asset for p in placements}
    for aid, row in row_of.items():
        if aid not in placed_assets and row["decision"] not in ("unresolved", "dropped"):
            row["decision"] = "dropped"
            row.setdefault("note", "no room in the schedule")

    counts: dict[str, int] = {}
    for d in decisions:
        counts[d["decision"]] = counts.get(d["decision"], 0) + 1
    notes = {
        "central_argument": editorial["central_argument"],
        "content_overlays_enabled": cfg.visuals_content_enabled,
        "candidates": len(candidates), "moments": len(groups), "placed": len(placements),
        "counts": counts, "dropped": dropped, "decisions": decisions,
        "prompt_sha": prompt_sha,
    }
    if cfg.plan_enabled:
        notes["plan"] = plan_notes
        placements = sorted(placements + plan_places, key=lambda p: p.start)
        notes["placed"] = len(placements)
    return placements, notes
