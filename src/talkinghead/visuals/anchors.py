"""Anchors on the OUTPUT timeline.

The judge reads only the FINAL cut's transcript, so the matched word's own
`start` IS its output time and nothing maps through the raw `Plan`. A phrase
not copied verbatim does not resolve, and the anchor is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..markers import norm


@dataclass(frozen=True)
class Anchor:
    text: str
    asset: str
    hold: float
    scale: float
    kind: str = "card"
    # Library match score, carried so the scheduler can settle a collision on
    # merit rather than on arrival order.
    score: float = 0.0
    # Members of a sequence chain with no gap: one visual that changes.
    sequence: str | None = None
    order: int = 0
    # Output time the overlay must not outlive (its chapter's end); None means
    # the end of the cut.
    until: float | None = None
    # The chosen absolute word index (chapter-local lookup); when set, the
    # resolver uses it instead of re-searching globally.
    word: int | None = None


@dataclass
class Resolved:
    anchor: Anchor
    path: str
    out_t: float
    word: int


def find_phrase(words: list, phrase: str) -> int | None:
    """Index of the first word of `phrase` in the stream, punctuation-blind."""
    want = [norm(w) for w in phrase.split()]
    want = [w for w in want if w]
    if not want:
        return None
    toks = [norm(w.text) for w in words]
    for i in range(len(toks) - len(want) + 1):
        if toks[i:i + len(want)] == want:
            return i
    return None


# Sentence punctuation at a token's edge; never carries meaning. Currency
# signs, signs and in-token decimals are deliberately NOT in this set, so a
# quote cannot turn a spoken "150" into a rendered "$1.50".
_EDGE_PUNCT = ".,!?;:\"'\u201c\u201d\u2018\u2019()[]\u2026"


def quote_token(text: str) -> str:
    """Normalise one token for QUOTE comparison only.

    Edge punctuation is dropped unconditionally, but `$`, signs and INTERNAL
    decimals stay, so the card cannot equate "$1.50" with "150" while "150."
    still matches a spoken "150".
    """
    return text.lower().strip().lstrip(_EDGE_PUNCT).rstrip(_EDGE_PUNCT)


def find_quote(words: list, quote: str) -> tuple[int, int] | None:
    """Index range [start, end) of a verbatim QUOTE, punctuation-preserving.

    A quote only matches the words actually spoken, so the caller can render
    the transcript span instead of the model's paraphrase of it.
    """
    want = [quote_token(w) for w in quote.split()]
    want = [w for w in want if w]
    if not want:
        return None
    toks = [quote_token(w.text) for w in words]
    for i in range(len(toks) - len(want) + 1):
        if toks[i:i + len(want)] == want:
            return i, i + len(want)
    return None


def resolve_output(anchor: Anchor, words: list, *, path: str = "") -> Resolved | None:
    """Resolve on the OUTPUT timeline: the matched word's start is the output
    time, so there is no source time and no cut to map through.

    Uses the anchor's chosen word index when it has one: a chapter-local lookup
    already picked the RIGHT occurrence, and a global re-search would undo that.
    """
    i = anchor.word if anchor.word is not None else find_phrase(words, anchor.text)
    if i is None or not 0 <= i < len(words):
        return None
    return Resolved(anchor, path, float(words[i].start), i)
