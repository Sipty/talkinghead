"""LLM coherence prompt: does the script still read once the cuts are applied?

Per-sentence judgments cannot see that removing one sentence orphans the next:
cut the credential ("since I have been an engineer...") and the line after it
reads as if being a YouTuber were the qualification. This module shows the judge
the FULL remaining script with each removed passage collapsed into a
`[CUT #n reason]` marker, plus the numbered cut list, and asks for two verdicts,
which `stages/content.py` applies:

    vetoes  - the cut breaks a reference, the logic, or sentence grammar
    extends - removing that sentence strands a neighbour that must go too
"""

from __future__ import annotations

from .prompts import speaker_tag
from .semantic import Sentence
from .types import CutSpan


def _covered(sent: Sentence, cuts: list[CutSpan]) -> int | None:
    """Index of the cut that removes this sentence, or None if it survives.

    A sentence counts as removed only when a MAJORITY of its span is covered
    by cuts (overlaps unioned, so stacked cuts are not double-counted); the
    reported cut is the one overlapping the sentence most. Mid-sentence cuts
    that remove a minority (e.g. a filler inside the sentence) leave the
    sentence in the script instead of displaying the whole sentence as gone.
    Degenerate zero-length sentences fall back to midpoint containment.
    """
    dur = sent.end - sent.start
    if dur <= 0.0:
        mid = (sent.start + sent.end) / 2
        return next((i for i, c in enumerate(cuts) if c.start <= mid <= c.end), None)
    overlaps: list[tuple[float, float, int]] = []
    for i, c in enumerate(cuts):
        s, e = max(sent.start, c.start), min(sent.end, c.end)
        if e > s:
            overlaps.append((s, e, i))
    if not overlaps:
        return None
    overlaps.sort()
    covered = 0.0
    best_i, best_len = overlaps[0][2], -1.0
    cs, ce = overlaps[0][0], overlaps[0][1]
    for s, e, i in overlaps:
        if s > ce:
            covered += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
        if e - s > best_len:
            best_len, best_i = e - s, i
    covered += ce - cs
    return best_i if covered / dur >= 0.5 else None


def _coverage(chunk: list[Sentence], cuts: list[CutSpan]) -> list[tuple[Sentence, int | None]]:
    """Each sentence of the chunk paired with the cut that removes it."""
    return [(s, _covered(s, cuts)) for s in chunk]


def _script_lines(coverage: list[tuple[Sentence, int | None]], cuts: list[CutSpan]) -> list[str]:
    """The chunk as the viewer will hear it: kept sentences, cuts as markers.

    Consecutive sentences removed by the same cut collapse into one marker.
    """
    lines: list[str] = []
    previous: int | None = None
    for s, idx in coverage:
        if idx is None:
            lines.append(f"{s.sid} | {speaker_tag(s)}{s.text}")
        elif idx != previous:
            lines.append(f"[CUT #{idx} {cuts[idx].reason}]")
        previous = idx
    return lines


def _cut_lines(coverage: list[tuple[Sentence, int | None]], cuts: list[CutSpan]) -> list[str]:
    """The numbered cut list, one line per cut marked in this chunk's script.

    Only cuts that remove speech are offered for judgment: a silence trim has
    nothing for the coherence pass to reason about, and listing it invites a
    veto on a cut the viewer would never notice.
    """
    marked = sorted({idx for _, idx in coverage if idx is not None})
    return [
        f"#{i} | {cuts[i].start:.1f}-{cuts[i].end:.1f} | {cuts[i].reason} | {cuts[i].text}"
        for i in marked
    ]


def _join_lines(chunk: list[tuple[Sentence, int | None]], cuts: list[CutSpan]) -> list[str]:
    """The exact splice the viewer will hear at each cut: words before, words after.

    Each cut is locally defensible; the failure mode is two cuts that are fine
    alone but strand a sentence between them ("Money is the number one" losing
    both its continuation and its example). Listing every join with context
    makes that visible without the LLM having to replay the whole script.
    """
    lines: list[str] = []
    coverage = [(s, i) for s, i in chunk if i is not None]
    for i in sorted({i for _, i in coverage}):
        kept = [(s, idx) for s, idx in chunk if idx is None]
        before = [s.text for s, _ in kept if s.end <= cuts[i].start][-2:]
        after = [s.text for s, _ in kept if s.start >= cuts[i].end][:2]
        lines.append(f"#{i} | ...{' '.join(before)} [CUT {cuts[i].reason}] {' '.join(after)}...")
    return lines


def build_coherence_prompt(chunk: list[Sentence], cuts: list[CutSpan], topic: str) -> str:
    """One prompt per chunk: the surviving script, then the numbered cut list."""
    lines: list[str] = []
    if topic:
        lines.append(f"Topic: {topic}")
        lines.append("")
    lines.extend(
        [
            "You are checking whether a talking-head video still makes sense after the",
            "proposed cuts. Below is the remaining script: kept sentences by id, and every",
            "removed passage collapsed into a [CUT #n reason] marker. Read it as the viewer",
            "will hear it, with the marked passages gone.",
            "Report ONLY cuts that break the surviving script:",
            "- dangling reference: a pronoun, \"that\", or a name whose antecedent was cut.",
            "- broken logic: a line that only held up because of a cut sentence, e.g. the",
            "  credentials that justified an opinion are gone, so the next line now claims",
            "  the wrong qualification.",
            "- broken grammar: the cut leaves a mid-thought fragment, or splices two",
            "  half-sentences into nonsense.",
            "- stranded middle: the sentence BETWEEN two cuts loses both its continuation",
            "  and its example, so each cut looks fine alone but together they orphan it.",
            "Two verdicts are available:",
            "- veto: keep this cut's content after all; the script breaks without it.",
            "- extend: the cut is right, but it strands neighbouring sentences that must go too;",
            "  list their ids in add_ids.",
            "Be conservative: a cut that merely tightens the script is fine, leave it alone.",
            'Respond with JSON only: {"vetoes": [{"id": 3, "why": "..."}], '
            '"extends": [{"id": 4, "add_ids": ["S0102"], "why": "..."}]}',
            "Use an empty list for either key when there is nothing to report.",
            "",
            "--- SCRIPT (cuts marked) ---",
        ]
    )
    coverage = _coverage(chunk, cuts)
    lines.extend(_script_lines(coverage, cuts))
    lines.append("")
    lines.append("--- CUTS ---")
    lines.extend(_cut_lines(coverage, cuts))
    lines.append("")
    lines.append("--- JOIN CHECKS (what the viewer hears across each cut) ---")
    lines.extend(_join_lines(coverage, cuts))
    return "\n".join(lines) + "\n"
