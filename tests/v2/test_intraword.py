"""Restarts inside one word: the audio finds where the broken-off part ends, and the cut sits there.

The signals are synthetic, with the restart's place known: a sound repeats after a closure. The placement is then
checked through the output round and the edge pass, so an intra-token cut is neither shrunk nor clipped there."""

import numpy as np
import pytest

from talkinghead import intraword
from talkinghead.audio import FRAME_SEC, Envelope, RefinedWord
from talkinghead.edges import EdgeContext, finalize_cuts
from talkinghead.judge import FakeJudge
from talkinghead.stages import polish
from talkinghead.timeline import Cut, Plan, SourceInfo
from talkinghead.transcribe import Word
from talkinghead.verify import _word_cuts

SR = 16000


def voiced(dur: float, f0: float, formant: float, seed: int) -> np.ndarray:
    """A harmonic sound with one spectral peak at `formant`, faded in and out so it has no clicks."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(dur * SR))) / SR
    x = np.zeros_like(t)
    for h in range(1, 80):
        f = h * f0
        if f > 4000:
            break
        amp = np.exp(-((f - formant) / 300.0) ** 2) + 0.05
        x += amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    x *= np.minimum(1.0, np.minimum(t, dur - t) / 0.01)
    return (0.3 * x / np.max(np.abs(x))).astype(np.float32)


def silence(dur: float) -> np.ndarray:
    return np.zeros(int(round(dur * SR)), np.float32)


A = voiced(0.15, 120, 700, seed=1)    # the word's first sound
B = voiced(0.20, 200, 2500, seed=2)   # what the speaker said before breaking off
C = voiced(0.25, 150, 1500, seed=3)   # the rest of the word, unlike the first sound
RESTART_AT = 0.15 + 0.20 + 0.04       # where the repeated first sound begins, after a 40 ms closure


def stutter() -> np.ndarray:
    """"impl-implement" in shape: the first sound, a broken-off part, a closure, the word again."""
    return np.concatenate([A, B, silence(0.04), A, C])


def no_repeat() -> np.ndarray:
    """A token whose first sound never comes back: an ordinary word."""
    return np.concatenate([A, B, C])


def no_closure() -> np.ndarray:
    """A repeat with no closure: the sound runs straight on, so there is no quiet point to cut at."""
    return np.concatenate([A, B, A, C])


def test_the_restart_is_found_where_the_first_sound_comes_back() -> None:
    found = intraword.locate(stutter())
    assert found is not None and found.found
    assert found.tau == pytest.approx(RESTART_AT, abs=0.02)
    assert found.similarity > 0.95
    # The cut ends in the closure, never in the broken-off sound or the restart.
    assert 0.15 + 0.20 <= found.cut_end <= RESTART_AT + 1e-9


def test_a_token_whose_first_sound_never_comes_back_is_not_claimed() -> None:
    found = intraword.locate(no_repeat())
    assert found is not None and not found.found


def test_a_repeat_with_no_closure_is_not_claimed_because_there_is_no_quiet_point() -> None:
    found = intraword.locate(no_closure())
    assert found is not None and not found.found and found.dip_db < intraword.MIN_DIP_DB


def test_a_token_too_short_for_a_restart_gives_no_candidate() -> None:
    assert intraword.locate(A[:800]) is None


def test_the_split_detection_takes_an_x_y_token_whose_second_part_starts_with_the_first() -> None:
    from talkinghead.residual import scan

    def finds(text: str):
        return [(f.kind, f.heard, f.split) for f in scan([Word(text, 1.0, 1.8, 1.0), Word("next", 2.0, 2.3, 1.0)])]

    assert finds("impl-implement") == [("restart", "impl-implement", True)]
    assert finds("dis-discuss,") == [("restart", "dis-discuss,", True)]
    assert finds("a-away") == [("restart", "a-away", True)]
    assert finds("co-op") == []  # the second part does not start with the first
    assert finds("laser-focused") == []


def _plan() -> Plan:
    return Plan.from_cuts(SourceInfo("main", "/v/in.mp4", "sha", 10.0), [])


def _ctx(judge) -> object:
    from types import SimpleNamespace

    from talkinghead.config import Config

    return SimpleNamespace(config=Config(polish_votes=3, polish_agree=2), judge=judge, llm_cache=None, topic=None,
                           conversation=[], preferences="")


def _verdict(decision: str) -> dict:
    return {"verdicts": [{"id": "c0", "verdict": decision, "why": "test"}]}


def test_an_output_round_cuts_the_broken_off_start_of_a_confirmed_restart(monkeypatch) -> None:
    token = stutter()
    monkeypatch.setattr(intraword, "load_span", lambda path, s, e: token)
    words = [Word("impl-implement", 1.0, 1.0 + token.size / SR, 1.0), Word("next", 3.0, 3.3, 1.0)]
    judge = FakeJudge([_verdict("cut")] * 3)
    value = polish.output_round(_ctx(judge), _plan(), words, 1)
    (cut,) = value["cuts"]
    assert value["findings"][0]["decision"] == "cut"
    assert cut["provenance"]["intra"] is True and cut["provenance"]["kind"] == "restart"
    assert cut["start"] == pytest.approx(1.0)
    assert cut["end"] == pytest.approx(1.0 + intraword.locate(token).cut_end, abs=1e-6)
    assert 1.0 + 0.35 <= cut["end"] <= 1.0 + RESTART_AT + 1e-9


def test_an_output_round_skips_and_reports_a_confirmed_restart_it_cannot_place(monkeypatch) -> None:
    token = no_repeat()
    monkeypatch.setattr(intraword, "load_span", lambda path, s, e: token)
    words = [Word("impl-implement", 1.0, 1.0 + token.size / SR, 1.0), Word("next", 3.0, 3.3, 1.0)]
    value = polish.output_round(_ctx(FakeJudge([_verdict("cut")] * 3)), _plan(), words, 1)
    assert value["cuts"] == []
    assert value["findings"][0]["decision"] == "skipped: no restart in the audio"
    # The sounds share a spectrum, so the match alone is not enough: the cut point has no closure to sit in.
    assert value["findings"][0]["restart"]["dip_db"] < intraword.MIN_DIP_DB


def test_a_placed_restart_survives_the_edge_pass_inside_the_word_it_shortens() -> None:
    # The kept word covers the whole token. Word cuts would be pushed out of it; a placed restart is not.
    db = np.full(int(10.0 / FRAME_SEC), -50.0)
    env = Envelope(db, np.zeros_like(db), FRAME_SEC, 16000)
    word = RefinedWord("impl-implement", 1.0, 1.9, 1.0, 1.9, 1.0)
    ctx = EdgeContext(duration=10.0, env=env, floor=-50.0, words=[word], fps=None)
    placed = Cut(1.0, 1.38, "polish", "impl-implement",
                 {"pass": "output", "kind": "restart", "intra": True, "span": [1.0, 1.38]})
    out = finalize_cuts([placed], ctx)
    assert len(out) == 1
    assert out[0].start == pytest.approx(1.0, abs=1e-6) and out[0].end == pytest.approx(1.38, abs=1e-6)


def test_an_intra_restart_is_not_a_judged_removal_or_a_clipped_word() -> None:
    cut = Cut(1.0, 1.38, "polish", "impl-implement", {"pass": "output", "kind": "restart", "intra": True})
    plan = Plan.from_cuts(SourceInfo("main", "/v/in.mp4", "sha", 10.0), [])
    plan = Plan(sources=plan.sources, main=plan.main, cuts=[cut], segments=plan.segments)
    assert _word_cuts(plan).cuts == []
