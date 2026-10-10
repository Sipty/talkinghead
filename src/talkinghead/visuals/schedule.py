"""Placement: turn resolved anchors into non-colliding overlay windows.

Overlays share the top-right corner, the one zone of this framing that the speaker
and their equipment do not occupy, so they must not overlap in time. An overlay
squeezed below `min_hold` is dropped rather than flashed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Placement:
    asset: str
    path: str
    start: float
    end: float
    scale: float

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2.0


def thin_collisions(resolved: list, *, min_hold: float, min_gap: float) -> tuple[list, list[str]]:
    """Where two anchors are too close for both, keep the better asset.

    Without this the later anchor simply won, whatever it was. In one demo run that
    dropped a real screenshot (match 0.75) because a generated card 2.7 s later took
    the room: the opposite of earning reuse.
    """
    order = sorted((r for r in resolved if r is not None), key=lambda r: r.out_t)
    dropped: list[str] = []
    changed = True
    while changed and len(order) > 1:
        changed = False
        for k in range(len(order) - 1):
            a, b = order[k], order[k + 1]
            if b.out_t - a.out_t >= min_hold + min_gap:
                continue
            if a.anchor.sequence and a.anchor.sequence == b.anchor.sequence:
                continue
            # Steps of one build-up never compete with each other.
            if a.anchor.sequence and a.anchor.sequence == b.anchor.sequence:
                continue
            loser = a if (a.anchor.score, a.anchor.hold) < (b.anchor.score, b.anchor.hold) else b
            keeper = b if loser is a else a
            dropped.append(
                f"{loser.anchor.asset}: {loser.out_t:.1f}s is within "
                f"{min_hold + min_gap:.1f}s of {keeper.anchor.asset} "
                f"({keeper.out_t:.1f}s, match {keeper.anchor.score:.2f} vs {loser.anchor.score:.2f})")
            order.remove(loser)
            changed = True
            break
    return order, dropped


def _clashes(start: float, end: float, reserved: list[tuple[float, float]], min_gap: float) -> bool:
    """True when [start, end] comes within min_gap of a reserved window."""
    return any(b[0] - min_gap < end and start < b[1] + min_gap for b in reserved)


def _pushed_start(start: float, hold: float, *, blockers: list[tuple[float, float]],
                  min_gap: float, ceiling: float) -> float | None:
    """Earliest start >= `start` where [start, start+hold] clears every blocker.

    `blockers` are the windows already spoken for: placed overlays and any
    RESERVED windows (chapter cards own their corner). None when no clear slot
    fits before `ceiling`.
    """
    t = start
    guard = 0
    while guard <= len(blockers) + 1:
        guard += 1
        clash = next((b for b in blockers
                      if b[0] - min_gap < t + hold and t < b[1] + min_gap), None)
        if clash is None:
            return t
        t = clash[1] + min_gap
        if t + hold > ceiling:
            return None
    return None


def schedule(resolved: list, *, out_duration: float, min_hold: float = 1.5,
             min_gap: float = 2.0, reserved: list[tuple[float, float]] = ()) -> tuple[list[Placement], list[str]]:
    """Place in output-time order; return placements and drop reasons.

    `reserved` windows (chapter cards) own the corner: content yields to them,
    including a library build-up sequence.
    """
    order, dropped = thin_collisions(resolved, min_hold=min_hold, min_gap=min_gap)
    placed: list[Placement] = []
    placed_seq: list[str | None] = []
    reserved = sorted(x for x in reserved if x[1] > x[0])
    for k, r in enumerate(order):
        start = r.out_t
        until = getattr(r.anchor, "until", None)
        nxt = order[k + 1] if k + 1 < len(order) else None
        # A step of a build-up runs straight into the next step: no gap, and no
        # minimum, because it is one visual changing rather than two overlays.
        chained = (nxt is not None and r.anchor.sequence
                   and r.anchor.sequence == nxt.anchor.sequence)
        if chained:
            end = min(nxt.out_t, out_duration, until if until is not None else out_duration)
            if end <= start or _clashes(start, end, reserved, min_gap):
                dropped.append(f"{r.anchor.asset}: build-up step blocked at {start:.1f}s "
                               f"(a chapter card owns the corner)")
            else:
                placed.append(Placement(r.anchor.asset, r.path, start, end, r.anchor.scale))
                placed_seq.append(r.anchor.sequence)
            continue
        ceiling = out_duration
        if until is not None:
            ceiling = min(ceiling, until)
        if nxt is not None:
            ceiling = min(ceiling, nxt.out_t - min_gap)
        # The final step of a build-up abuts its predecessor; the min_gap push
        # applies only between unrelated overlays.
        prev = order[k - 1] if k else None
        continues_chain = (prev is not None and r.anchor.sequence
                           and prev.anchor.sequence == r.anchor.sequence)
        # A chain-abutting step must not treat earlier steps of the SAME
        # sequence as blockers, or a fitting final step is pushed past its
        # ceiling and dropped. Other overlays and reserved cards still block.
        placed_windows = [(p.start, p.end) for p, s in zip(placed, placed_seq)
                          if not (continues_chain and s is not None
                                  and s == r.anchor.sequence)]
        blockers = sorted(placed_windows + list(reserved))
        pushed = _pushed_start(start, r.anchor.hold, blockers=blockers, min_gap=min_gap,
                               ceiling=ceiling)
        if pushed is None or (continues_chain and _clashes(start, start + r.anchor.hold,
                                                           reserved, min_gap)):
            dropped.append(f"{r.anchor.asset}: no room at {r.out_t:.1f}s "
                           f"(need {min_hold:.1f}s clear of the overlays and chapter cards)")
            continue
        if continues_chain and pushed != start:
            dropped.append(f"{r.anchor.asset}: build-up step cannot abut at {start:.1f}s")
            continue
        end = min(pushed + r.anchor.hold, ceiling)
        if end - pushed < min_hold:
            dropped.append(f"{r.anchor.asset}: no room at {r.out_t:.1f}s "
                           f"(only {max(0.0, end - pushed):.2f}s, need {min_hold:.1f}s)")
            continue
        placed.append(Placement(r.anchor.asset, r.path, pushed, end, r.anchor.scale))
        placed_seq.append(r.anchor.sequence)
    return placed, dropped


def free_slot_after(target: float, *, hold: float, taken: list[tuple[float, float]],
                    min_gap: float, limit: float) -> float | None:
    """Earliest start >= target where `hold` seconds are free, up to `limit`.

    The opening card belongs at the START of the video. Reusing the
    before-a-chapter search put it 3:35 in, because the latest free gap before
    chapter one is nearly four minutes deep.
    """
    busy = sorted(taken)
    t = max(0.0, target)
    while t + hold <= limit:
        window = (t, t + hold)
        clash = next((b for b in busy
                      if b[0] < window[1] + min_gap and window[0] < b[1] + min_gap), None)
        if clash is None:
            return round(t, 3)
        t = clash[1] + min_gap
    return None
