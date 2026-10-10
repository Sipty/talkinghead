"""Pass B: marker-triggered adjudication with tiered bias.

The judge is routed by prompt content rather than scripted in sequence, so a
test states what the judge believes about a passage and stays readable when
the number of calls changes.
"""

from __future__ import annotations

import numpy as np

from talkinghead.audio import FRAME_SEC, Envelope, RefinedWord
from talkinghead.config import Config
from talkinghead.context import Context
from talkinghead.judge import validate
from talkinghead.stages import retakes
from talkinghead.timeline import SourceInfo
from talkinghead.transcribe import Word


class RoutingJudge:
    """Answers by matching a substring of the prompt; default is a keep/no-op."""

    name = "routing"

    def __init__(self, routes: dict | None = None) -> None:
        self.routes = routes or {}
        self.calls = 0

    def decide(self, prompt: str, schema: dict) -> dict:
        self.calls += 1
        for key, answer in self.routes.items():
            if key in prompt:
                return validate(answer, schema)
        default = {"retakes": []} if "TASK: RETAKES" in prompt else {"verdict": "keep", "why": "default"}
        return validate(default, schema)


def ctx_for(text: str, judge, *, cfg: Config | None = None, step: float = 0.4) -> Context:
    words = [Word(t, i * step, i * step + 0.3, 1.0) for i, t in enumerate(text.split())]
    duration = len(words) * step + 2.0
    db = np.full(int(duration / FRAME_SEC) + 1, -50.0)
    for w in words:
        db[int(w.start / FRAME_SEC): int(w.end / FRAME_SEC)] = -20.0
    env = Envelope(db, np.zeros_like(db), FRAME_SEC, 16000)
    refined = [RefinedWord(w.text, w.start, w.end, w.start, w.end, 1.0) for w in words]
    return Context(
        config=cfg or Config(votes=3, agree=2, marker_votes=3, marker_agree=2),
        source=SourceInfo("main", "/v/in.mp4", "sha", duration),
        words=refined, env=env, floor=-50.0, silences=[], moans=[],
        judge=judge, llm_cache=None, guidelines="G", topic="testing",
    )


def test_pass_b_cuts_a_strong_marker() -> None:
    # A shortened stumble with a wrong name, restated.
    text = "Scout agent which passes over to a reviewer sorry uh so a scout agent passing over to a planner."
    judge = RoutingJudge({"TASK: MARKER": {"verdict": "cut", "from_word": "w0000",
                                           "to_word": "w0009", "why": "agent named wrong, restated"}})
    cuts, notes = retakes.run(ctx_for(text, judge))
    assert len(cuts) == 1
    assert cuts[0].reason == "retake"
    assert notes["retakes"]["pass_b"]["cut"] == 1
    d = notes["retakes"]["decisions"][0]
    assert (d["marker"], d["tier"], d["verdict"]) == ("sorry", "strong", "cut")


def test_pass_b_keeps_audience_address() -> None:
    # "sorry" addressed to the viewer is not a retake.
    text = "And I blamed Anthropic a lot for it sorry guys this is not your fault I messed up."
    judge = RoutingJudge({"TASK: MARKER": {"verdict": "keep", "why": "addressed to the audience"}})
    cuts, notes = retakes.run(ctx_for(text, judge))
    assert cuts == []
    d = notes["retakes"]["decisions"][0]
    assert (d["marker"], d["verdict"]) == ("sorry", "keep")
    assert d["why"] == "addressed to the audience"


def test_strong_marker_cuts_by_default_when_the_judge_is_silent() -> None:
    """Strong-marker bias: no keep majority means cut."""
    text = "the planner runs first hold on the scout runs first and then the planner."
    cuts, notes = retakes.run(ctx_for(text, RoutingJudge(
        {"TASK: MARKER": {"verdict": "cut", "why": "restart", "from_word": "w0000", "to_word": "w0004"}})))
    assert len(cuts) == 1
    assert notes["retakes"]["decisions"][0]["tier"] == "strong"


def test_weak_marker_keeps_by_default() -> None:
    # "I mean" is ordinary speech far more often than a retake.
    text = "And by vibecode I mean apply strong engineering practices and get great results."
    cuts, notes = retakes.run(ctx_for(text, RoutingJudge()))
    assert cuts == []
    d = notes["retakes"]["decisions"][0]
    assert (d["marker"], d["tier"], d["verdict"]) == ("i mean", "weak", "keep")


def test_marker_is_pulled_into_a_span_that_stops_short_of_it() -> None:
    """A range ending before the apology would leave the apology in."""
    text = "passes over to a reviewer sorry so a scout passes to a planner."
    judge = RoutingJudge({"TASK: MARKER": {"verdict": "cut", "from_word": "w0000",
                                           "to_word": "w0004", "why": "stops before the marker"}})
    cuts, _ = retakes.run(ctx_for(text, judge))
    assert "sorry" in cuts[0].text


def test_decision_log_records_every_candidate() -> None:
    text = "one sorry two and I mean three."
    _, notes = retakes.run(ctx_for(text, RoutingJudge()))
    assert {d["marker"] for d in notes["retakes"]["decisions"]} == {"sorry", "i mean"}


def test_pass_a_span_voting_survives_boundary_disagreement() -> None:
    """Near-miss edges from different samples must not cancel the cut out."""
    text = "alpha beta gamma delta epsilon zeta eta theta."
    answers = [
        {"retakes": [{"from_word": "w0001", "to_word": "w0004", "why": "a"}]},
        {"retakes": [{"from_word": "w0002", "to_word": "w0004", "why": "b"}]},
        {"retakes": [{"from_word": "w0001", "to_word": "w0003", "why": "c"}]},
    ]

    class Seq:
        name = "seq"

        def __init__(self): self.i = 0

        def decide(self, prompt, schema):
            if "TASK: RETAKES" in prompt:
                a = answers[self.i]
                self.i += 1
                return validate(a, schema)
            return validate({"verdict": "keep", "why": ""}, schema)

    cuts, notes = retakes.run(ctx_for(text, Seq()))
    assert notes["retakes"]["pass_a"]["spans"] == 1
    assert len(cuts) == 1
    # narrow mode: the core all three agree on is w0002-w0003
    assert cuts[0].provenance["words"] == [2, 3]


def test_overlong_marker_span_is_dropped_and_reported() -> None:
    text = "one two three four five sorry six seven eight nine ten."
    judge = RoutingJudge({"TASK: MARKER": {"verdict": "cut", "from_word": "w0000",
                                           "to_word": "w0005", "why": "long"}})
    cfg = Config(votes=3, agree=2, marker_votes=3, marker_agree=2, retake_max_sec=0.5)
    cuts, notes = retakes.run(ctx_for(text, judge, cfg=cfg))
    assert cuts == []
    assert notes["retakes"]["pass_b"]["dropped_overlong"] == 1
