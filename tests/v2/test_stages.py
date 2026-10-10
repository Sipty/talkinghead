"""Stages are pure transforms: (context, cuts) -> cuts + decisions. Judges are
faked; every span in the output must come from code-generated candidates."""


import numpy as np
import pytest

from talkinghead.audio import FRAME_SEC, Envelope, RefinedWord
from talkinghead.config import Config
from talkinghead.context import Context, NeedsDecision
from talkinghead.judge import FakeJudge
from talkinghead.stages import content, pauses, retakes, structure
from talkinghead.timeline import Cut, SourceInfo
from talkinghead.transcribe import Word

# Sentences (punctuation or a >= 0.5 s pause ends one): S0000 "Hold on." | S0001 "Are you
# tired?" | S0002 "Um" | S0003 "Are you tired yet?" | S0004 "Second point here." | S0005 "Goodbye."
WORDS = [
    Word("Hold", 0.00, 0.30, 1.0), Word("on.", 0.30, 0.60, 1.0),
    Word("Are", 2.00, 2.20, 1.0), Word("you", 2.20, 2.40, 1.0), Word("tired?", 2.40, 2.90, 1.0),
    Word("Um", 3.60, 3.90, 1.0),
    Word("Are", 4.50, 4.70, 1.0), Word("you", 4.70, 4.90, 1.0), Word("tired", 4.90, 5.20, 1.0), Word("yet?", 5.20, 5.60, 1.0),
    Word("Second", 6.00, 6.40, 1.0), Word("point", 6.40, 6.70, 1.0), Word("here.", 6.70, 7.00, 1.0),
    Word("Goodbye.", 8.00, 8.50, 1.0),
]
DURATION = 12.0
SILENCES = [(0.6, 2.0), (2.9, 3.6), (3.9, 4.5), (5.6, 6.0), (7.0, 8.0), (8.5, 12.0)]


def _ctx(judge, cfg: Config | None = None, decisions: dict | None = None,
         words: list[Word] = WORDS, duration: float = DURATION) -> Context:
    db = np.full(int(duration / FRAME_SEC), -50.0)
    for w in words:
        db[int(w.start / FRAME_SEC): int(w.end / FRAME_SEC)] = -20.0
    env = Envelope(db, np.zeros_like(db), FRAME_SEC, 16000)
    refined = [RefinedWord(w.text, w.start, w.end, w.start, w.end, 1.0) for w in words]
    return Context(
        config=cfg or Config(votes=5, agree=4),
        source=SourceInfo("main", "/v/in.mp4", "sha", duration),
        words=refined, env=env, floor=-50.0, silences=SILENCES, moans=[],
        judge=judge, llm_cache=None, guidelines="GUIDELINES", topic="testing",
        decisions=decisions or {},
    )


# ------------------------------------------------------------- structure

def test_cold_open_and_outro_from_supermajority_votes() -> None:
    judge = FakeJudge(
        [{"true_start": "S0001", "why": "topic lands"}] * 5
        + [{"last_sentence": "S0005", "why": "clean goodbye"}] * 5
    )
    ctx = _ctx(judge)
    cuts, decisions = structure.run(ctx)
    assert Cut(0.0, 2.0, "cold-open") in [Cut(c.start, c.end, c.reason) for c in cuts]
    assert Cut(8.5, DURATION, "outro") in [Cut(c.start, c.end, c.reason) for c in cuts]
    assert decisions["cold_open"]["chosen"] == "S0001" and decisions["cold_open"]["votes"]["S0001"] == 5
    assert decisions["cold_open"]["candidates"][:2] == ["KEEP_HEAD", "S0000"]
    assert judge.calls == 10


def test_cold_open_keep_head_option_produces_no_head_cut() -> None:
    judge = FakeJudge([{"true_start": "KEEP_HEAD", "why": ""}] * 5 + [{"last_sentence": "S0005", "why": ""}] * 5)
    cuts, decisions = structure.run(_ctx(judge))
    assert not [c for c in cuts if c.reason == "cold-open"]
    assert decisions["cold_open"]["chosen"] == "KEEP_HEAD"


def test_cold_open_without_supermajority_stops_and_asks() -> None:
    judge = FakeJudge([{"true_start": "S0001", "why": ""}] * 3 + [{"true_start": "S0002", "why": ""}] * 2)
    with pytest.raises(NeedsDecision) as exc:
        structure.run(_ctx(judge))
    assert exc.value.key == "cold_open"
    assert "--cold-open" in str(exc.value)
    assert "S0001" in str(exc.value) and "S0002" in str(exc.value)


def test_persisted_decision_overrides_the_vote() -> None:
    judge = FakeJudge([{"last_sentence": "S0005", "why": ""}] * 5)   # no cold-open call at all
    cuts, decisions = structure.run(_ctx(judge, decisions={"cold_open": 4.5}))
    head = [c for c in cuts if c.reason == "cold-open"]
    assert head and head[0].end == pytest.approx(4.5)
    assert decisions["cold_open"]["source"] == "user"
    assert judge.calls == 5


# --------------------------------------------------------------- retakes

def test_retakes_are_word_ranges_voted_per_span() -> None:
    # 4 of 5 samples cut "Are you tired?" (w0002-w0004); one sample also cuts "Um".
    # WORDS opens on "Hold on.", a STRONG marker, so pass B runs too: three
    # keep votes leave it alone and only the pass A span is cut.
    answers = [{"retakes": [{"from_word": "w0002", "to_word": "w0004", "why": "first take"}]}] * 4
    answers += [{"retakes": [{"from_word": "w0002", "to_word": "w0005", "why": ""}]}]
    answers += [{"verdict": "keep", "why": "trademark greeting"}] * 3
    cuts, notes = retakes.run(_ctx(FakeJudge(answers)))
    assert [(c.start, c.end, c.reason) for c in cuts] == [(2.0, 2.9, "retake")]
    assert notes["retakes"]["pass_a"]["spans"] == 1
    assert notes["retakes"]["pass_b"]["keep"] == 1


def test_retakes_ignore_unknown_ids_and_overlong_spans() -> None:
    answers = [{"retakes": [{"from_word": "w0000", "to_word": "w0013", "why": "everything"}]}] * 5
    answers += [{"verdict": "keep", "why": "trademark greeting"}] * 3
    cuts, notes = retakes.run(_ctx(FakeJudge(answers), cfg=Config(votes=5, agree=4, retake_max_sec=5.0)))
    assert cuts == []
    assert notes["retakes"]["dropped_overlong"] == 5  # one per sample


# --------------------------------------------------------------- content

def test_content_cuts_are_whole_sentences_and_survive_coherence() -> None:
    answers = [{"cuts": [{"ids": ["S0004"], "reason": "digression", "why": "off topic"}]}] * 5
    answers += [{"vetoes": [], "extends": []}] * 5
    cuts, notes = content.run(_ctx(FakeJudge(answers)), existing=[])
    assert [(c.start, c.end, c.reason) for c in cuts] == [(6.0, 7.0, "content")]
    assert cuts[0].provenance["reason"] == "digression"


def test_content_cut_vetoed_by_coherence_is_dropped_and_reported() -> None:
    answers = [{"cuts": [{"ids": ["S0004"], "reason": "digression", "why": ""}]}] * 5
    answers += [{"vetoes": [{"id": 0, "why": "next line refers to it"}], "extends": []}] * 5
    cuts, notes = content.run(_ctx(FakeJudge(answers)), existing=[])
    assert cuts == []
    assert notes["content"]["vetoed"] == 1


def test_content_caps_drop_overlong_and_over_budget_runs() -> None:
    cfg = Config(votes=5, agree=4, content_max_cut_sec=0.5, content_max_fraction=0.10)
    answers = [{"cuts": [{"ids": ["S0004"], "reason": "digression", "why": ""}]}] * 5
    cuts, notes = content.run(_ctx(FakeJudge(answers), cfg=cfg), existing=[])
    assert cuts == [] and notes["content"]["dropped_overlong"] == 1
    # The budget is a share of the video for CONTENT cuts; a long cold-open trim
    # or retake does not eat it (a real run dropped every content cut that way).
    cfg = Config(votes=5, agree=4, content_max_fraction=0.10)
    answers = [{"cuts": [{"ids": ["S0004"], "reason": "digression", "why": ""}]}] * 5 + [{"vetoes": [], "extends": []}] * 5
    cuts, notes = content.run(_ctx(FakeJudge(answers), cfg=cfg), existing=[Cut(0.0, 5.0, "cold-open")])
    assert len(cuts) == 1 and notes["content"]["dropped_over_budget"] == 0


def _ramble(n_words: int, start: float) -> list[Word]:
    """Unpunctuated words half a second apart; the last one ends the sentence."""
    out = [Word(f"w{i}", start + 0.5 * i, start + 0.5 * i + 0.4, 1.0) for i in range(n_words)]
    out[-1] = Word("end.", out[-1].start, out[-1].end, 1.0)
    return out


def _cut_then_extend(sentence_id: str) -> list[dict]:
    """Five votes cutting S0000, then five coherence passes extending it to `sentence_id`."""
    cut = {"cuts": [{"ids": ["S0000"], "reason": "no-value", "why": ""}]}
    extend = {"vetoes": [], "extends": [{"id": 0, "add_ids": [sentence_id], "why": "strands it"}]}
    return [cut] * 5 + [extend] * 5


def test_coherence_extend_respects_the_cut_cap() -> None:
    # The extend adds a whole 34.9 s sentence. The 30 s cap that bounds passage
    # cuts must bound extends too, so it is dropped and only "Hold on." is cut.
    words = [Word("Hold", 0.0, 0.3, 1.0), Word("on.", 0.3, 0.6, 1.0)] + _ramble(70, 2.0)
    cuts, notes = content.run(_ctx(FakeJudge(_cut_then_extend("S0001")), words=words, duration=40.0),
                              existing=[])
    assert [(c.start, c.end) for c in cuts] == [(0.0, 0.6)]
    assert notes["content"]["dropped_overlong"] == 1 and notes["content"]["extended"] == 0


def test_coherence_extend_respects_the_content_budget() -> None:
    # Budget is 10% of 40 s = 4 s. The extend (7.9 s) passes the cap but not the
    # budget: it is the longest of equal-vote cuts, so it goes and "Hold on." stays.
    words = [Word("Hold", 0.0, 0.3, 1.0), Word("on.", 0.3, 0.6, 1.0)] + _ramble(16, 2.0)
    cuts, notes = content.run(_ctx(FakeJudge(_cut_then_extend("S0001")), words=words, duration=40.0),
                              existing=[])
    assert [(c.start, c.end) for c in cuts] == [(0.0, 0.6)]
    assert notes["content"]["dropped_over_budget"] == 1 and notes["content"]["extended"] == 0



def test_content_disabled_makes_no_llm_calls() -> None:
    judge = FakeJudge([])
    cuts, notes = content.run(_ctx(judge, cfg=Config(content_enabled=False)), existing=[])
    assert cuts == [] and judge.calls == 0


# ---------------------------------------------------------------- pauses

def test_pauses_stage_cuts_silences_and_fillers_mechanically() -> None:
    cuts, notes = pauses.run(_ctx(FakeJudge([])), noise_cuts=[Cut(7.2, 7.4, "noise", "mouth noise: smack")])
    reasons = {c.reason for c in cuts}
    assert {"silence", "filler", "noise"} <= reasons
    filler = [c for c in cuts if c.reason == "filler"]
    assert len(filler) == 1 and 3.4 <= filler[0].start <= 3.6
    # Interior pauses are trimmed, not removed: the 1.4 s pause after "on."
    # (sentence end) keeps its 0.55 s cap, centred.
    lead = [c for c in cuts if c.reason == "silence" and 0.6 <= c.start <= 1.0]
    assert lead and (lead[0].start, lead[0].end) == pytest.approx((0.875, 1.725))
    assert notes["pauses"]["silence"] >= 4
