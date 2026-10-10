"""Content cuts: whole sentences the judge says do not earn their runtime,
voted per sentence, capped, then checked by a coherence pass that may veto
or extend a cut (both by supermajority)."""

from __future__ import annotations

from collections import Counter

from ..coherence import build_coherence_prompt
from ..context import Context
from ..judge import vote, vote_items
from ..prompts import COHERENCE_SCHEMA, CONTENT_SCHEMA, content_prompt
from ..timeline import Cut


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for i in indices:
        if out and out[-1][1] == i - 1:
            out[-1] = (out[-1][0], i)
        else:
            out.append((i, i))
    return out


def _limit(cuts: list[Cut], ctx: Context, notes: dict) -> list[Cut]:
    """Drop cuts over the per-cut cap, then the weakest over the content budget."""
    kept: list[Cut] = []
    for c in cuts:
        if c.length > ctx.config.content_max_cut_sec:
            notes["content"]["dropped_overlong"] += 1
        else:
            kept.append(c)
    # The budget is a share of the video for content cuts alone: structural
    # trims and retakes are different decisions and do not consume it.
    budget = ctx.config.content_max_fraction * ctx.duration
    total = sum(c.length for c in kept)
    # Weakest votes go first when over budget.
    while kept and total > budget:
        weakest = min(kept, key=lambda c: (c.provenance["votes"], -c.length))
        kept.remove(weakest)
        total -= weakest.length
        notes["content"]["dropped_over_budget"] += 1
    return kept


def run(ctx: Context, existing: list[Cut]) -> tuple[list[Cut], dict]:
    notes: dict = {"content": {"cuts": 0, "vetoed": 0, "extended": 0, "dropped_overlong": 0,
                               "dropped_over_budget": 0, "enabled": ctx.config.content_enabled}}
    if not ctx.config.content_enabled:
        return [], notes
    sents = ctx.sentences()
    by_sid = {s.sid: i for i, s in enumerate(sents)}
    result = vote(ctx.judge, content_prompt(sents, ctx.topic), CONTENT_SCHEMA,
                  k=ctx.config.votes, cache=ctx.llm_cache)
    coverage: Counter[int] = Counter()
    reasons: dict[int, Counter] = {}
    for resp in result.responses:
        seen: set[int] = set()
        for entry in resp.get("cuts", []):
            for sid in entry.get("ids", []):
                i = by_sid.get(str(sid))
                if i is not None:
                    seen.add(i)
                    reasons.setdefault(i, Counter())[entry.get("reason", "no-value")] += 1
        coverage.update(seen)
    removed = sorted(i for i, c in coverage.items() if c >= ctx.config.agree)

    cuts: list[Cut] = []
    for a, b in _runs(removed):
        start, end = sents[a].start, sents[b].end
        reason = Counter()
        for i in range(a, b + 1):
            reason.update(reasons.get(i, {}))
        cuts.append(Cut(start, end, "content", " ".join(s.text for s in sents[a:b + 1]), provenance={
            "sentences": [sents[a].sid, sents[b].sid],
            "reason": reason.most_common(1)[0][0] if reason else "no-value",
            "votes": min(coverage[i] for i in range(a, b + 1)),
        }))
    cuts = _limit(cuts, ctx, notes)

    # Coherence: does the script still read with these passages gone?
    if cuts:
        prompt = build_coherence_prompt(sents, cuts, ctx.topic or "")
        prompt = prompt.replace("Respond with JSON only:", "TASK: COHERENCE\nRespond with JSON only:")
        coh = vote(ctx.judge, prompt, COHERENCE_SCHEMA, k=ctx.config.votes, cache=ctx.llm_cache)
        vetoes = vote_items(
            [{"v": [{"id": str(e["id"]), "x": "veto"} for e in r.get("vetoes", [])]} for r in coh.responses],
            items_key="v", id_key="id", value_key="x", agree=ctx.config.agree,
        )
        vetoed = {int(i) for i, t in vetoes.items() if t.winner == "veto" and i.isdigit()}
        # extends: a sentence is added when a supermajority adds it to some cut
        adds: Counter[str] = Counter()
        for r in coh.responses:
            seen = {str(sid) for e in r.get("extends", []) for sid in e.get("add_ids", [])}
            adds.update(seen)
        extend_sids = {sid for sid, c in adds.items() if c >= ctx.config.agree and sid in by_sid}
        kept: list[Cut] = []
        for i, c in enumerate(cuts):
            if i in vetoed:
                notes["content"]["vetoed"] += 1
                continue
            kept.append(c)
        for sid in sorted(extend_sids):
            s = sents[by_sid[sid]]
            kept.append(Cut(s.start, s.end, "content", s.text, provenance={"sentences": [sid, sid], "reason": "coherence-extend", "votes": adds[sid]}))
        # Extended sentences can exceed the caps the passages above respected,
        # so the final list is checked again.
        cuts = _limit(sorted(kept, key=lambda c: c.start), ctx, notes)
        notes["content"]["extended"] = sum(c.provenance["reason"] == "coherence-extend" for c in cuts)
        notes["content"]["coherence_prompt_sha"] = coh.prompt_sha
    notes["content"]["cuts"] = len(cuts)
    notes["content"]["prompt_sha"] = result.prompt_sha
    return cuts, notes
