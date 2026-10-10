"""Check short repairs and protected contrasts through the real retake judge.
Live cases require TALKINGHEAD_JUDGE_TESTS=1 and use production judge wiring.
Allowed word-index deletions distinguish clean repairs from collateral cuts.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass

import numpy as np
import pytest

from talkinghead.audio import FRAME_SEC, Envelope, RefinedWord
from talkinghead.config import Config
from talkinghead.context import Context
from talkinghead.judge import validate
from talkinghead.markers import norm
from talkinghead.stages import retakes
from talkinghead.timeline import SourceInfo

GUIDELINES = pathlib.Path(__file__).parents[2] / "editorial" / "guidelines.md"
TOPIC = "Reviewing pull requests with coding agents: what the agents do well and where they slip"
WORD_STEP = 0.4      # 0.3 s word + 0.1 s gap: a correction arrives with no pause
SENTENCE_GAP = 0.6   # above sentence_split_pause, so sentences split as spoken


@dataclass(frozen=True)
class Case:
    label: str
    sentence: str
    alternatives: tuple[tuple[int, ...], ...]  # acceptable removed word positions
    kept: str

    @property
    def words(self) -> list[str]:
        return self.sentence.split()


def _norm_text(words: list[str]) -> str:
    return " ".join(norm(w) for w in words)


# --- cases that must be cut ------------------------------------------------

WRONG_INFLECTION = Case(
    "single word, no marker: wrong verb form then right one",
    "Each worker pick, picks a task from the queue and runs it.",
    ((2,),), "Each worker picks a task from the queue and runs it.")
WRONG_ARTICLE = Case(
    "single word, no marker: wrong article then right one",
    "The reviewer left a, an honest comment about the naming.",
    ((3,),), "The reviewer left an honest comment about the naming.")
PHRASE_SHARED_SUBJECT = Case(
    "short phrase, shared subject: abandoned 'checks every hour'",
    "Our scheduler checks every hour, checks every ten minutes for new jobs.",
    ((2, 3, 4), (3, 4, 5)), "Our scheduler checks every ten minutes for new jobs.")
HAVE_IT_HAS_IT = Case(
    "short phrase, shared subject: 'Claude Code have it, has it, OMP has it'",
    "Claude Code have it, has it, OMP has it.",
    ((2, 3),), "Claude Code has it, OMP has it.")
REPEATED_FALSE_START = Case(
    "repeated false start of the same opening words",
    "We should, we should start every review from the diff.",
    ((0, 1), (1, 2)), "We should start every review from the diff.")

# --- cases that must stay whole -------------------------------------------

EMPHASIS_DOUBLED_WORD = Case(
    "intentional emphasis: doubled intensifier",
    "That was a very, very long week.",
    ((),), "That was a very, very long week.")
EMPHASIS_ESCALATING = Case(
    "intentional emphasis: escalating repetition",
    "The migration was slow, slow, painfully slow.",
    ((),), "The migration was slow, slow, painfully slow.")
LIST_OF_VERBS = Case(
    "valid list: parallel verb phrases",
    "The agent reads the issue, writes the patch, runs the tests, and opens the pull request.",
    ((),), "The agent reads the issue, writes the patch, runs the tests, and opens the pull request.")
PARALLEL_CLAUSES = Case(
    "valid list: parallel clauses repeating the subject and verb",
    "It runs on every commit, it runs on every merge, and it runs again every night.",
    ((),), "It runs on every commit, it runs on every merge, and it runs again every night.")
NOT_X_BUT_Y = Case(
    "meaningful contrast: not X, we need Y",
    "We do not need more agents, we need better handoffs between the agents we already have.",
    ((),), "We do not need more agents, we need better handoffs between the agents we already have.")
NOT_X_I_MEAN_Y = Case(
    "meaningful correction: not the parser, I mean the lexer, and the difference matters",
    "It was not the parser that failed, I mean the lexer, and that difference matters "
    "because the fix ships in a different package.",
    ((),), "It was not the parser that failed, I mean the lexer, and that difference matters "
           "because the fix ships in a different package.")

# --- neutral sentences: nothing to cut -------------------------------------

def _neutral(n: int, sentence: str) -> Case:
    return Case(f"neutral {n}", sentence, ((),), sentence)


N1 = _neutral(1, "The review queue was backed up for most of the week.")
N2 = _neutral(2, "Nothing else in the pipeline changed after that.")
N3 = _neutral(3, "So I wrote down what each tool actually owns.")
N4 = _neutral(4, "That settled the argument for the whole team.")

# Each transcript is one coherent stretch of talk, run through retakes.run once.
TRANSCRIPTS: dict[str, list[Case]] = {
    "corrections": [N1, WRONG_INFLECTION, WRONG_ARTICLE, N2, PHRASE_SHARED_SUBJECT,
                    HAVE_IT_HAS_IT, N3, REPEATED_FALSE_START, N4],
    "keeps": [N1, EMPHASIS_DOUBLED_WORD, EMPHASIS_ESCALATING, N2, LIST_OF_VERBS,
              PARALLEL_CLAUSES, N3, NOT_X_BUT_Y, NOT_X_I_MEAN_Y, N4],
    "mixed": [N1, WRONG_INFLECTION, EMPHASIS_ESCALATING, N2, HAVE_IT_HAS_IT, LIST_OF_VERBS,
              N3, REPEATED_FALSE_START, NOT_X_I_MEAN_Y, WRONG_ARTICLE, PARALLEL_CLAUSES, N4],
}


def build_context(cases: list[Case], judge, cfg: Config | None = None) -> tuple[Context, list[int]]:
    """A Context over the cases' words; also each case's first word position."""
    words: list[RefinedWord] = []
    offsets: list[int] = []
    t = 0.0
    for case in cases:
        offsets.append(len(words))
        for token in case.words:
            words.append(RefinedWord(token, t, t + 0.3, t, t + 0.3, 1.0))
            t += WORD_STEP
        t += SENTENCE_GAP - (WORD_STEP - 0.3)
    duration = t + 1.0
    db = np.full(int(duration / FRAME_SEC) + 1, -50.0)
    for w in words:
        db[int(w.start / FRAME_SEC): int(w.end / FRAME_SEC)] = -20.0
    env = Envelope(db, np.zeros_like(db), FRAME_SEC, 16000)
    ctx = Context(
        config=cfg or Config(), source=SourceInfo("main", "/v/in.mp4", "sha", duration),
        words=words, env=env, floor=-50.0, silences=[], moans=[], judge=judge,
        llm_cache=None, guidelines=GUIDELINES.read_text(), topic=TOPIC)
    return ctx, offsets


def removed_positions(cuts) -> set[int]:
    out: set[int] = set()
    for c in cuts:
        a, b = c.provenance["words"]
        out.update(range(a, b + 1))
    return out


def grade(cases: list[Case], offsets: list[int], cuts) -> list[dict]:
    """Grade each passage's removed words and surviving text."""
    removed = removed_positions(cuts)
    rows = []
    for case, off in zip(cases, offsets):
        got = tuple(sorted(p - off for p in removed if off <= p < off + len(case.words)))
        kept = _norm_text([w for i, w in enumerate(case.words) if i not in got])
        ok = got in case.alternatives and kept == _norm_text(case.kept.split())
        rows.append({"label": case.label, "ok": ok, "sentence": case.sentence,
                     "removed": list(got), "allowed": [list(a) for a in case.alternatives],
                     "remaining": " ".join(w for i, w in enumerate(case.words) if i not in got)})
    return rows


def _live_judge(cfg: Config):
    from talkinghead.cli import build_judge

    return build_judge(cfg.judge, model=cfg.model, guidelines=GUIDELINES.read_text(),
                       effort=cfg.effort, fallback_command=cfg.fallback_command)


@pytest.mark.judge
@pytest.mark.skipif(not os.environ.get("TALKINGHEAD_JUDGE_TESTS"),
                    reason="set TALKINGHEAD_JUDGE_TESTS=1 to spend judge calls")
@pytest.mark.parametrize("name", list(TRANSCRIPTS))
def test_real_judge_cuts_short_corrections_and_keeps_the_rest(name: str) -> None:
    cfg = Config()
    cases = TRANSCRIPTS[name]
    ctx, offsets = build_context(cases, _live_judge(cfg), cfg)
    cuts, _ = retakes.run(ctx)
    bad = [r for r in grade(cases, offsets, cuts) if not r["ok"]]
    assert not bad, "\n" + "\n".join(
        f"- {r['label']}: removed {r['removed']} allowed {r['allowed']}\n"
        f"    in : {r['sentence']}\n    out: {r['remaining']}" for r in bad)


class _SingleWordPassA:
    """Answers pass A with one single-word range; this checks the consumer, not the judge."""

    name = "single-word"

    def __init__(self, word_id: str) -> None:
        self.word_id = word_id

    def decide(self, prompt: str, schema: dict) -> dict:
        if "TASK: RETAKES" in prompt:
            answer = {"retakes": [{"from_word": self.word_id, "to_word": self.word_id, "why": "x"}]}
        else:
            answer = {"verdict": "keep", "why": ""}
        return validate(answer, schema)


def test_unanimous_single_word_range_cuts_exactly_that_word() -> None:
    """A range with from == to is a one-word cut, not a rejected or widened one."""
    cases = TRANSCRIPTS["corrections"]
    off = 0
    for case in cases:
        if case is WRONG_INFLECTION:
            break
        off += len(case.words)
    target = off + 2  # "pick,"
    ctx, offsets = build_context(cases, _SingleWordPassA(f"w{target:04d}"))
    cuts, notes = retakes.run(ctx)
    assert [c.provenance["words"] for c in cuts] == [[target, target]]
    assert cuts[0].text == "pick,"
    assert notes["retakes"]["dropped_bad_ids"] == 0
    assert [r["ok"] for r in grade(cases, offsets, cuts) if r["label"] == WRONG_INFLECTION.label] == [True]
