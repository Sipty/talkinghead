"""No cut edge lies inside a kept word: the hackathon geometry, a seam beside a touching word, and the clipped check."""

import numpy as np

from talkinghead.audio import FRAME_SEC, Envelope, RefinedWord
from talkinghead.edges import EdgeContext, _decided_removed, finalize_cuts, keep_words_whole
from talkinghead.timeline import Cut, Plan, SourceInfo
from talkinghead.verify import clipped_words

S1 = "S1"


def _ctx(words: list[RefinedWord], duration: float = 100.0) -> EdgeContext:
    db = np.full(int(duration / FRAME_SEC), -50.0)
    for w in words:
        db[int(w.asr_start / FRAME_SEC): int(w.asr_end / FRAME_SEC)] = -20.0
    env = Envelope(db, np.zeros_like(db), FRAME_SEC, 16000)
    return EdgeContext(duration=duration, env=env, floor=-50.0, words=words, fps=None)


def _overlaps(c: Cut, a: float, b: float) -> bool:
    return c.start < b and c.end > a


# The hackathon case: "hackathon." refines to 89.26-89.36 but its ASR stamp runs to 89.82. A restaurant-noise cut
# sits in that stamp; the filler "Um," and the false start "it was..." are removed. The fused polish cut used to
# start at 89.467, inside the stamp.
HACK_WORDS = [
    RefinedWord("GitLab", 88.92, 89.22, 88.92, 89.22, 1.0, speaker=S1),
    RefinedWord("hackathon.", 89.26, 89.36, 89.26, 89.82, 1.0, speaker=S1),
    RefinedWord("Um,", 90.16, 90.70, 90.30, 90.70, 1.0, speaker=S1),
    RefinedWord("it", 91.02, 91.07, 91.06, 91.12, 1.0, speaker=S1),
    RefinedWord("was...", 91.07, 91.12, 91.16, 91.50, 1.0, speaker=S1),
]


def test_hackathon_geometry_keeps_the_word_and_still_removes_the_filler() -> None:
    raw = [Cut(90.14, 90.76, "filler", "Um,"),
           Cut(89.55, 89.825, "noise"),
           Cut(91.02, 91.12, "polish", "it was...", {"kind": "false-start", "votes": 3})]
    out = finalize_cuts(raw, _ctx(HACK_WORDS))
    assert not any(_overlaps(c, 89.26, 89.82) for c in out), out
    assert not any(_overlaps(c, 88.92, 89.22) for c in out), out
    assert any(c.start <= 90.50 <= c.end for c in out), "the filler must still be removed"


def test_decided_removed_words_are_the_ones_a_word_cut_covers() -> None:
    decided = _decided_removed([Cut(91.02, 91.12, "polish")], HACK_WORDS)
    assert {id(w) for w in HACK_WORDS if w.text in ("it", "was...")} <= decided
    assert id(HACK_WORDS[1]) not in decided


def test_a_seam_beside_a_touching_word_starts_after_that_words_stamp() -> None:
    # The reorder seam (a tangent block) begins 80 ms into "alpha." but its ASR stamp runs to 1.60, touching "beta".
    words = [RefinedWord("alpha.", 1.00, 1.50, 1.00, 1.60, 1.0, speaker="S0"),
             RefinedWord("beta", 1.55, 1.90, 1.55, 1.90, 1.0, speaker="S0")]
    cuts = [Cut(1.52, 3.00, "tangent", "block")]
    decided = _decided_removed(cuts, words)
    out = keep_words_whole(cuts, _ctx(words), decided)
    assert out, "the seam must still remove the block"
    assert all(not _overlaps(c, 1.00, 1.60) for c in out), out
    assert min(c.start for c in out) >= 1.60


def test_a_cut_that_covers_a_kept_word_is_split_around_it() -> None:
    words = [RefinedWord("kept", 2.00, 2.40, 2.00, 2.40, 1.0, speaker="S0")]
    out = keep_words_whole([Cut(0.50, 5.00, "noise")], _ctx(words), set())
    assert [(round(c.start, 6), round(c.end, 6)) for c in out] == [(0.50, 1.95), (2.45, 5.00)]


def test_the_clipped_check_flags_the_hackathon_cut_and_passes_the_fixed_one() -> None:
    source = SourceInfo("main", "/v/in.mp4", "sha", 100.0)
    before = Plan.from_cuts(source, [Cut(89.467, 91.767, "polish", "", {"kind": "false-start"})])
    flagged = clipped_words(before, HACK_WORDS)
    assert [x["text"] for x in flagged] == ["hackathon."]
    assert flagged[0]["cut"] > 0.10 and flagged[0]["reasons"] == ["polish"]
    after = Plan.from_cuts(source, [Cut(89.85, 91.767, "polish", "", {"kind": "false-start"})])
    assert clipped_words(after, HACK_WORDS) == []


def test_a_word_the_stages_removed_is_not_a_kept_word_when_the_edge_chain_shortened_its_cut() -> None:
    # "it" was removed by the judge; the edge chain left its cut covering only the stamp's tail. With the decisions
    # given, "it" is not kept, so its partial cut is not a clipped kept word.
    source = SourceInfo("main", "/v/in.mp4", "sha", 100.0)
    plan = Plan.from_cuts(source, [Cut(91.09, 91.12, "polish", "", {"kind": "false-start"})])
    decisions = [(91.02, 91.12, "polish")]
    assert clipped_words(plan, HACK_WORDS, decisions) == []
    assert [x["text"] for x in clipped_words(plan, HACK_WORDS)] == ["it"]
