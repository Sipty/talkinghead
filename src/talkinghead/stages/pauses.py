"""Pauses, fillers and mouth noises: mechanical, no judge. Interior pauses are
trimmed to a length that grows with the pause and is capped by its boundary
(`pause_*` config); leading/trailing silence goes; fillers the ASR
emitted as tokens go; mouth noises in verified quiet go (detector output is
passed in, computed once on the source)."""

from __future__ import annotations

from ..candidates import machinery_candidates
from ..context import Context
from ..timeline import Cut


def run(ctx: Context, noise_cuts: list[Cut]) -> tuple[list[Cut], dict]:
    cfg = ctx.config
    cands = machinery_candidates(
        ctx.duration, ctx.as_words(), ctx.silences,
        pause_above=cfg.pause_above, pause=cfg, min_filler_dur=cfg.min_filler_dur,
    )
    cuts = [Cut(c.start, c.end, c.kind, c.text, provenance={"candidate": c.id}) for c in cands]
    cuts += [Cut(c.start, c.end, "noise", c.text, dict(c.provenance)) for c in noise_cuts]
    counts = {k: sum(1 for c in cuts if c.reason == k) for k in ("silence", "filler", "restart", "noise")}
    return cuts, {"pauses": counts}
