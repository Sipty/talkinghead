"""The v11 gate: the eleven labelled retake cases from the v10 source.

Two tiers, because only one of them is free.

The marker SCAN is pure, so it is asserted unconditionally: every labelled
word must be found with the right marker and tier. If this breaks, the
detection path regressed and no judge can save it.

The VERDICTS need a live judge, so they run only under
TALKINGHEAD_JUDGE_TESTS=1. This is the gate the v11 spec puts in front of the
full cook: six cases must be cut and five preserved before a 24-minute render
is worth paying for.

These labels are pinned to ONE source (audio sha b39d336...). Passing here is
a regression guard for this video, not evidence that another video works.
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pytest

from talkinghead.audio import FRAME_SEC, Envelope, RefinedWord
from talkinghead.config import Config
from talkinghead.context import Context
from talkinghead.markers import find_markers
from talkinghead.stages import retakes
from talkinghead.timeline import SourceInfo

SHA = "b39d33626855024d37e55bf962b90f7c381b622a3d753d214646eea1ccfbb801"
TRANSCRIPT = "transcript-929d495f7dccacc3811d9d66.json"
LABELS = pathlib.Path(__file__).parent.parent / "fixtures" / "v11" / "retake-labels.json"


def _cache_path() -> pathlib.Path | None:
    for root in (os.environ.get("TALKINGHEAD_CACHE"),
                 pathlib.Path.home() / ".cache" / "talkinghead-v11",
                 pathlib.Path.home() / ".cache" / "talkinghead-v10",
                 pathlib.Path.home() / ".cache" / "talkinghead"):
        if root is None:
            continue
        p = pathlib.Path(root) / SHA / TRANSCRIPT
        if p.is_file():
            return p
    return None


needs_cache = pytest.mark.skipif(_cache_path() is None,
                                 reason="v10 transcript cache not on this machine")


def _words() -> list[RefinedWord]:
    raw = json.loads(_cache_path().read_text())["value"]["words"]
    return [RefinedWord(d["w"], d["s"], d["e"], d["s"], d["e"], 1.0) for d in raw]


def _labels() -> dict:
    return json.loads(LABELS.read_text())


def test_labels_are_pinned_to_the_expected_source() -> None:
    assert _labels()["audio_sha256"] == SHA


@needs_cache
def test_every_labelled_case_is_found_by_the_scan() -> None:
    found = {c.word: c for c in find_markers(_words())}
    missing = [(c["word"], c["marker"], c["note"]) for c in _labels()["cases"]
               if c["word"] not in found or found[c["word"]].marker != c["marker"]]
    assert not missing, f"scan missed labelled cases: {missing}"


@needs_cache
def test_expected_cuts_are_all_strong_tier() -> None:
    """The owner's flipped bias only reaches STRONG markers, so every case we
    expect to cut must be one -- otherwise it relies on the judge overriding a
    keep default, which is not what the design promises."""
    found = {c.word: c for c in find_markers(_words())}
    weak_cuts = [c["word"] for c in _labels()["cases"]
                 if c["expect"] == "cut" and found[c["word"]].tier != "strong"]
    assert not weak_cuts, f"expected-cut cases on a weak marker: {weak_cuts}"


@needs_cache
def test_no_labelled_keep_is_an_i_mean_that_the_scan_calls_strong() -> None:
    found = {c.word: c for c in find_markers(_words())}
    for c in _labels()["cases"]:
        if c["marker"] == "i mean":
            assert found[c["word"]].tier == "weak"


@needs_cache
@pytest.mark.judge
@pytest.mark.skipif(not os.environ.get("TALKINGHEAD_JUDGE_TESTS"),
                    reason="set TALKINGHEAD_JUDGE_TESTS=1 to spend judge calls")
def test_verdicts_match_the_labels() -> None:
    from talkinghead.cli import build_judge

    words = _words()
    duration = words[-1].end + 1.0
    db = np.full(int(duration / FRAME_SEC) + 1, -50.0)
    env = Envelope(db, np.zeros_like(db), FRAME_SEC, 16000)
    guidelines = (pathlib.Path(__file__).parents[2] / "editorial" / "guidelines.md").read_text()
    cfg = Config()
    ctx = Context(
        config=cfg, source=SourceInfo("main", "/golden-denoised.mp4", SHA, duration),
        words=words, env=env, floor=-54.28, silences=[], moans=[],
        judge=build_judge(cfg.judge, model=cfg.model, guidelines=guidelines,
                          effort=cfg.effort, fallback_command=cfg.fallback_command),
        llm_cache=None, guidelines=guidelines,
        topic="My agentic coding workflow: using several models and coding harnesses "
              "together, what worked and what failed",
    )
    _, notes = retakes.run(ctx)
    got = {d["word"]: d["verdict"] for d in notes["retakes"]["decisions"]}
    wrong = [{"word": c["word"], "expect": c["expect"], "got": got.get(c["word"]), "note": c["note"]}
             for c in _labels()["cases"] if got.get(c["word"]) != c["expect"]]
    assert not wrong, "verdict mismatches:\n" + json.dumps(wrong, indent=2)
