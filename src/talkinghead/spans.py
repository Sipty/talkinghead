"""Overlap clustering for span votes.

A retake is a span, and independent samples disagree about its edges by a word
or two. v10 voted per WORD at 4-of-5, so a near miss cancelled an otherwise
unanimous finding: 28 valid proposals produced zero removed words. This votes
per cluster instead.

`narrow` keeps the intersection -- cut less, per the guidelines' "unsure where
a retake ends and the correction begins: keep the wider span". `wide` takes the
union, which the guidelines prefer for messy stretches: "prefer fewer, larger
cuts that remove the whole messy stretch up to the clean restatement".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    a: int
    b: int
    voters: frozenset[int]

    @property
    def votes(self) -> int:
        return len(self.voters)


def cluster_spans(
    per_response: list[list[tuple[int, int]]], *, agree: int, mode: str = "narrow"
) -> list[Span]:
    """Cluster overlapping spans; keep clusters backed by `agree` responses."""
    if mode not in ("narrow", "wide"):
        raise ValueError(f"unknown mode {mode!r}")
    flat: list[tuple[int, int, int]] = []
    for r, spans in enumerate(per_response):
        for a, b in spans:
            flat.append((a, b, r))
    flat.sort()
    clusters: list[list[tuple[int, int, int]]] = []
    for item in flat:
        if clusters and item[0] <= max(x[1] for x in clusters[-1]):
            clusters[-1].append(item)
        else:
            clusters.append([item])
    out: list[Span] = []
    for cl in clusters:
        voters = {r for _, _, r in cl}
        if len(voters) < agree:
            continue
        if mode == "wide":
            a, b = min(x[0] for x in cl), max(x[1] for x in cl)
        else:
            # One response may name overlapping spans, so intersect per
            # response first, then across responses.
            per: dict[int, tuple[int, int]] = {}
            for x0, x1, r in cl:
                if r in per:
                    p0, p1 = per[r]
                    per[r] = (min(p0, x0), max(p1, x1))
                else:
                    per[r] = (x0, x1)
            a, b = max(p[0] for p in per.values()), min(p[1] for p in per.values())
            if a > b:  # no common core; fall back to the union rather than drop
                a, b = min(x[0] for x in cl), max(x[1] for x in cl)
        out.append(Span(a, b, frozenset(voters)))
    return out
