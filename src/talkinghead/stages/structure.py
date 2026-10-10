"""Cold open and outro: the two structural decisions.

Candidates come from code (sentence starts after real pauses in the head
window; sentences in the tail window). The judge picks an id; k votes must
agree. No supermajority -> NeedsDecision (stop and ask), unless the user
already answered (decisions.json / --cold-open / --outro).
"""

from __future__ import annotations

from ..candidates import cold_open_candidates, outro_candidates
from ..context import Context, NeedsDecision
from ..judge import vote, vote_scalar
from ..prompts import COLD_OPEN_SCHEMA, OUTRO_SCHEMA, cold_open_prompt, outro_prompt
from ..timeline import Cut

KEEP_HEAD = "KEEP_HEAD"


def _decide(ctx: Context, key: str, flag: str, prompt: str, schema: dict, answer_key: str,
            options: list[str], labels: dict[str, str]) -> tuple[str | None, dict]:
    user = ctx.decisions.get(key)
    if user is not None:
        return None, {"source": "user", "value": user}
    result = vote(ctx.judge, prompt, schema, k=ctx.config.votes, cache=ctx.llm_cache)
    valid = [r for r in result.responses if str(r.get(answer_key)) in options]
    tally = vote_scalar(valid, answer_key, agree=ctx.config.agree)
    record = {
        "source": "vote", "chosen": tally.winner, "votes": tally.counts,
        "candidates": options, "prompt_sha": result.prompt_sha,
        "why": sorted({str(r.get("why", ""))[:200] for r in valid}),
    }
    if tally.winner is None:
        lines = [f"  {o}: {tally.counts.get(o, 0)} vote(s)  {labels.get(o, '')[:90]}" for o in options
                 if tally.counts.get(o, 0)]
        raise NeedsDecision(
            key, flag,
            f"{key.replace('_', ' ')}: no supermajority ({ctx.config.agree}/{ctx.config.votes}). "
            f"Votes:\n" + "\n".join(lines) + f"\nRerun with {flag} SECONDS to decide; the answer is persisted.",
            [{"id": o, "votes": tally.counts.get(o, 0), "text": labels.get(o, "")} for o in options],
        )
    return tally.winner, record


def run(ctx: Context) -> tuple[list[Cut], dict]:
    sents = ctx.sentences()
    cuts: list[Cut] = []
    decisions: dict = {}

    # -- cold open
    cands = cold_open_candidates(sents, window_sec=ctx.config.cold_open_window_sec)
    options = [KEEP_HEAD] + [s.sid for s in cands]
    labels = {s.sid: s.text for s in cands}
    labels[KEEP_HEAD] = "start on the very first word"
    winner, record = _decide(
        ctx, "cold_open", "--cold-open", cold_open_prompt(sents, cands, ctx.topic), COLD_OPEN_SCHEMA,
        "true_start", options, labels,
    )
    if record["source"] == "user":
        start = float(record["value"])
    else:
        start = 0.0 if winner == KEEP_HEAD else next(s.start for s in sents if s.sid == winner)
    if start > 0.0:
        cuts.append(Cut(0.0, start, "cold-open", provenance={"decision": "cold_open"}))
    record["start"] = start
    decisions["cold_open"] = record

    # -- outro
    tail = outro_candidates(sents, ctx.duration, window_sec=ctx.config.outro_window_sec)
    if tail:
        options = [s.sid for s in tail]
        labels = {s.sid: s.text for s in tail}
        winner, record = _decide(
            ctx, "outro", "--outro", outro_prompt(sents, tail, ctx.topic), OUTRO_SCHEMA,
            "last_sentence", options, labels,
        )
        if record["source"] == "user":
            end = float(record["value"])
        else:
            end = next(s.end for s in sents if s.sid == winner)
        if end < ctx.duration:
            cuts.append(Cut(end, ctx.duration, "outro", provenance={"decision": "outro"}))
        record["end"] = end
        decisions["outro"] = record
    return cuts, decisions
