"""Speaker-aware cleanup on a two-person recording (one mixed track).

A speakerless transcript must cook exactly as before: its judge prompts, stage
memo keys, config digest and cut plan are pinned to values computed at the base
commit. Everything speaker-specific is tested with labelled words.
"""

from __future__ import annotations

import contextlib
import hashlib
import json

import numpy as np
import pytest

from talkinghead.audio import FRAME_SEC, Envelope, RefinedWord
from talkinghead.candidates import sentence_groups, sentence_table
from talkinghead.coherence import build_coherence_prompt
from talkinghead.config import Config, load_config
from talkinghead.context import Context, NeedsDecision
from talkinghead.edges import EdgeContext, finalize_cuts, spare_other_speaker
from talkinghead.judge import FakeJudge
from talkinghead.pipeline import _retake_policy, _stage_config, _stage_inputs, edge_context
from talkinghead.prompts import (
    annotated_sentence_table,
    cold_open_prompt,
    content_prompt,
    editorial_prompt,
    marker_prompt,
    marker_window,
    outro_prompt,
    retake_marker_window,
    retake_prompt,
    retake_table,
    transcript_text,
    with_conversation,
    with_preferences,
)
from talkinghead.prompts import (
    sentence_table as prompt_sentence_table,
)
from talkinghead.stages import content, retakes, structure
from talkinghead.timeline import Cut, Plan, SourceInfo
from talkinghead.transcribe import Word

SINGLE = [
    Word("Hold", 0.00, 0.30, 1.0), Word("on.", 0.30, 0.60, 1.0),
    Word("Are", 2.00, 2.20, 1.0), Word("you", 2.20, 2.40, 1.0), Word("tired?", 2.40, 2.90, 1.0),
    Word("Um", 3.60, 3.90, 1.0),
    Word("Sorry", 4.50, 4.80, 1.0), Word("no", 4.80, 4.90, 1.0), Word("wait", 4.90, 5.20, 1.0),
    Word("Second", 6.00, 6.40, 1.0), Word("point", 6.40, 6.70, 1.0), Word("here.", 6.70, 7.00, 1.0),
    Word("Goodbye.", 8.00, 8.50, 1.0),
]
DURATION = 12.0
SILENCES = [(0.6, 2.0), (2.9, 3.6), (3.9, 4.5), (5.2, 6.0), (7.0, 8.0), (8.5, 12.0)]
NAMES = {"S0": "Chavdar (host)", "S1": "brother (DevOps engineer, guest)"}

# Turn changes without punctuation or pauses: S0, S1 "yeah", S0 "right so".
TURNS = [
    Word("So", 0.0, 0.2, 1.0, "S0"), Word("I", 0.2, 0.3, 1.0, "S0"), Word("think", 0.3, 0.6, 1.0, "S0"),
    Word("yeah", 0.7, 0.9, 1.0, "S1"),
    Word("right", 1.0, 1.2, 1.0, "S0"), Word("so.", 1.2, 1.5, 1.0, "S0"),
]


def _digest(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _labelled(words: list[Word], labels: list[str | None]) -> list[Word]:
    return [Word(w.text, w.start, w.end, w.probability, s) for w, s in zip(words, labels)]


def _w(text: str, start: float, end: float, speaker: str | None) -> Word:
    return Word(text, start, end, 1.0, speaker)


def _ctx(words: list[Word], cfg: Config | None = None, judge=None) -> Context:
    db = np.full(int(DURATION / FRAME_SEC), -50.0)
    for w in words:
        db[int(w.start / FRAME_SEC): int(w.end / FRAME_SEC)] = -20.0
    env = Envelope(db, np.zeros_like(db), FRAME_SEC, 16000)
    refined = [RefinedWord(w.text, w.start, w.end, w.start, w.end, 1.0, speaker=w.speaker) for w in words]
    return Context(
        config=cfg or Config(votes=5, agree=4),
        source=SourceInfo("main", "/v/in.mp4", "sha", DURATION),
        words=refined, env=env, floor=-50.0, silences=SILENCES, moans=[],
        judge=judge or FakeJudge([]), llm_cache=None, guidelines="GUIDELINES", topic="testing",
    )


def fingerprints() -> dict:
    """Hashes of everything a speakerless cook sends to a judge, keys its memos with, or cuts."""
    ctx = _ctx(SINGLE)
    sents, groups = ctx.sentences(), ctx.sentence_words()
    window = marker_window(sents, groups, ctx.words, 6)
    table = annotated_sentence_table(sents, groups, ctx.words)
    cuts = [Cut(0.0, 2.0, "cold-open"), Cut(3.6, 3.9, "filler", "Um"), Cut(4.5, 5.2, "retake", "Sorry no wait"),
            Cut(6.0, 6.4, "content", "Second")]
    coherence_cuts = [Cut(4.5, 5.2, "retake", "Sorry no wait")]
    plan = Plan.from_cuts(ctx.source, finalize_cuts(cuts, edge_context(ctx, ctx.config, fps=30)))
    return {
        "sentences": _digest([(s.sid, s.start, s.end, s.text) for s in sents]),
        "cold_open": _digest(with_preferences(cold_open_prompt(sents, sents[:2], "topic"), "- keep laughs")),
        "outro": _digest(outro_prompt(sents, sents[-2:], "topic")),
        "retake_table": _digest(table),
        "marker_window": _digest(window),
        "marker": _digest(marker_prompt(window, "sorry", "strong", "topic")),
        "retake": _digest(retake_prompt(table, "topic")),
        "content": _digest(content_prompt(sents, "topic")),
        "coherence": _digest(build_coherence_prompt(sents, coherence_cuts, "topic")),
        "editorial": _digest(editorial_prompt(transcript_text(ctx.words), "topic", 6)),
        "retake_policy": _retake_policy(),
        "stage_inputs": _digest(_stage_inputs(ctx, "GUIDELINES")),
        "stage_config": {s: _digest(_stage_config(ctx, s)) for s in ("structure", "retakes", "content", "pauses", "visuals")},
        "config_dict": _digest(Config().to_dict()),
        "plan": _digest(plan.to_json()),
    }


# Captured at the base commit, before any speaker code existed.
BASE_COMMIT = {
    "coherence": "49f2a13f9e0d45f5",
    "cold_open": "3cc5ff9a70bee9e8",
    "config_dict": "7f75077128cf4bbe",
    "content": "d11001e1e783383d",
    "editorial": "55f6e016005ae2fe",
    "marker": "f50772bde822030e",
    "marker_window": "d3d66c44a15f1f1c",
    "outro": "5bb70c96a844dd99",
    "plan": "abcea96cae82bd58",
    "retake": "7695db7a8d503e27",
    "retake_policy": "0e9fc19aadb6b87c",
    "retake_table": "9e1811c957b6f97f",
    "sentences": "3f404eed2c59f44d",
    "stage_config": {
        "content": "e7297933ebfcc14d",
        "pauses": "027a2bc564a07543",
        "retakes": "fffac30e8611b1e2",
        "structure": "4b38cf0097098a9c",
        "visuals": "90b041df648b8ce7",
    },
    "stage_inputs": "7e63feae1edd6c79",
}


def test_speakerless_cook_produces_the_base_commit_prompts_keys_and_plan() -> None:
    assert fingerprints() == BASE_COMMIT


# --- sentences ----------------------------------------------------------------

def test_a_turn_change_starts_a_new_sentence_even_without_punctuation_or_pause() -> None:
    assert [s.text for s in sentence_table(TURNS)] == ["So I think", "yeah", "right so."]


def test_speakerless_words_split_exactly_as_before() -> None:
    plain = [Word(w.text, w.start, w.end, w.probability) for w in TURNS]
    assert [s.text for s in sentence_table(plain)] == ["So I think yeah right so."]
    assert sentence_groups(plain) == [[0, 1, 2, 3, 4, 5]]


def test_sentences_carry_display_names_or_raw_labels() -> None:
    named = sentence_table(TURNS, speaker_names={"S0": "Chavdar (host)"})
    assert [s.speaker for s in named] == ["Chavdar (host)", "S1", "Chavdar (host)"]
    assert [s.speaker for s in sentence_table(TURNS)] == ["S0", "S1", "S0"]
    assert sentence_table([Word("hi", 0, 1, 1.0)])[0].speaker is None


def test_context_sentences_use_configured_names() -> None:
    ctx = _ctx(TURNS, cfg=Config(votes=5, agree=4, speaker_names=NAMES))
    assert [s.speaker for s in ctx.sentences()] == [NAMES["S0"], NAMES["S1"], NAMES["S0"]]


# --- judge-facing tables ------------------------------------------------------

def test_sentence_table_tags_each_line_with_the_speaker_name() -> None:
    lines = prompt_sentence_table(sentence_table(TURNS, speaker_names={"S0": "Chavdar (host)"})).split("\n")
    assert lines[0] == "S0000 | 0.00-0.60 | [Chavdar (host)] So I think"
    assert lines[1] == "S0001 | 0.70-0.90 | [S1] yeah"


def test_speakerless_sentence_table_has_no_tags() -> None:
    plain = sentence_table([Word("hi", 0.0, 0.5, 1.0)])
    assert prompt_sentence_table(plain) == "S0000 | 0.00-0.50 | hi"


def test_retake_table_tags_sentences_and_keeps_word_ids() -> None:
    ctx = _ctx(TURNS, cfg=Config(votes=5, agree=4, speaker_names=NAMES))
    table = retake_table(ctx.sentences(), ctx.sentence_words(), ctx.words)
    assert table.split("\n")[0] == f"S0000 | 0.00-0.60 | [{NAMES['S0']}] [w0000]So [w0001]I [w0002]think"


def test_retake_table_without_speakers_is_the_annotated_table() -> None:
    ctx = _ctx(SINGLE)
    sents, groups = ctx.sentences(), ctx.sentence_words()
    assert retake_table(sents, groups, ctx.words) == annotated_sentence_table(sents, groups, ctx.words)


def test_retake_marker_window_tags_speakers_only_when_present() -> None:
    ctx = _ctx(TURNS, cfg=Config(votes=5, agree=4, speaker_names=NAMES))
    window = retake_marker_window(ctx.sentences(), ctx.sentence_words(), ctx.words, 3)
    assert f"[{NAMES['S1']}] [w0003]yeah" in window


# --- conversation preamble ----------------------------------------------------

def test_conversation_preamble_only_when_there_are_speakers() -> None:
    assert with_conversation("TASK: X", []) == "TASK: X"
    led = with_conversation("TASK: X", ["Chavdar (host)", "brother"])
    assert led.startswith("--- CONVERSATION ---\n")
    assert "between Chavdar (host), brother" in led
    assert "refer to the host only" in led
    assert led.endswith("\n\nTASK: X")


def test_context_conversation_lists_speakers_in_order_of_first_speaking() -> None:
    ctx = _ctx(TURNS, cfg=Config(votes=5, agree=4, speaker_names=NAMES))
    assert ctx.conversation == [NAMES["S0"], NAMES["S1"]]
    assert _ctx(SINGLE).conversation == []


class Answering(FakeJudge):
    """Records every prompt and answers by the schema's first property."""

    name = "answering"

    def __init__(self, answers: dict) -> None:
        super().__init__([])
        self.answers = answers
        self.prompts: list[str] = []

    def decide(self, prompt: str, schema: dict) -> dict:
        self.calls += 1
        self.prompts.append(prompt)
        return dict(self.answers[next(iter(schema["properties"]))])


ANSWERS = {
    "retakes": {"retakes": []},
    "verdict": {"verdict": "keep", "why": ""},
    "cuts": {"cuts": []},
    "vetoes": {"vetoes": [], "extends": []},
}
SINGLE_LABELS = ["S0"] * 3 + ["S1"] * 2 + ["S0"] * 4 + ["S1"] * 4


def test_editorial_judges_see_the_preamble_when_speakers_are_present() -> None:
    judge = Answering(ANSWERS)
    ctx = _ctx(_labelled(SINGLE, SINGLE_LABELS), cfg=Config(votes=5, agree=4, speaker_names=NAMES), judge=judge)
    retakes.run(ctx)
    content.run(ctx, existing=[])
    assert judge.prompts, "the stages should have asked the judge"
    assert all(p.count("--- CONVERSATION ---") == 1 for p in judge.prompts)
    assert any(f"[{NAMES['S1']}]" in p for p in judge.prompts)


def test_single_speaker_judges_see_no_preamble() -> None:
    judge = Answering(ANSWERS)
    ctx = _ctx(SINGLE, judge=judge)
    retakes.run(ctx)
    content.run(ctx, existing=[])
    assert judge.prompts
    assert not any("CONVERSATION" in p for p in judge.prompts)


def test_cold_open_prompt_carries_the_preamble_and_tags_with_speakers() -> None:
    judge = Answering({"true_start": {"true_start": "KEEP_HEAD", "why": ""},
                       "last_sentence": {"last_sentence": "S0005", "why": ""}})
    ctx = _ctx(_labelled(SINGLE, SINGLE_LABELS), cfg=Config(votes=5, agree=4, speaker_names=NAMES), judge=judge)
    with contextlib.suppress(NeedsDecision):  # the outro answer need not be a candidate here; only prompts matter
        structure.run(ctx)
    cold = [p for p in judge.prompts if "TASK: COLD-OPEN" in p]
    assert cold and all("--- CONVERSATION ---" in p and f"[{NAMES['S0']}]" in p for p in cold)


# --- overlapping-speech guard -------------------------------------------------

def _edge_ctx(words: list[Word], *, duration: float = DURATION) -> EdgeContext:
    db = np.full(int(duration / FRAME_SEC), -50.0)
    for w in words:
        db[int(w.start / FRAME_SEC): int(w.end / FRAME_SEC)] = -20.0
    env = Envelope(db, np.zeros_like(db), FRAME_SEC, 16000)
    refined = [RefinedWord(w.text, w.start, w.end, w.start, w.end, 1.0, speaker=w.speaker) for w in words]
    return EdgeContext(duration=duration, env=env, floor=-50.0, words=refined, fps=None)


def test_a_backchannel_fully_under_another_speakers_word_is_not_cut() -> None:
    ctx = _edge_ctx([_w("so", 0.8, 1.7, "S0"), _w("mm", 1.0, 1.3, "S1")])
    assert spare_other_speaker([Cut(1.0, 1.3, "filler", "mm")], ctx) == []
    assert ctx.spared_other_speaker == 1


def test_a_filler_overlapping_another_speaker_shrinks_away_from_their_word() -> None:
    ctx = _edge_ctx([_w("um", 1.0, 1.4, "S0"), _w("yeah", 1.2, 1.6, "S1")])
    (cut,) = spare_other_speaker([Cut(1.0, 1.4, "filler", "um")], ctx)
    assert cut.start == 1.0
    assert cut.end == pytest.approx(1.2 - 0.03)
    assert ctx.spared_other_speaker == 1


def test_a_filler_left_with_only_a_sliver_is_dropped() -> None:
    ctx = _edge_ctx([_w("um", 1.0, 1.1, "S0"), _w("yeah", 1.05, 1.9, "S1")])
    assert spare_other_speaker([Cut(1.0, 1.1, "filler", "um")], ctx) == []


def test_a_restart_split_around_another_speakers_word_keeps_both_sides() -> None:
    ctx = _edge_ctx([_w("so", 2.0, 2.2, "S0"), _w("no", 2.4, 2.6, "S1"), _w("so", 2.8, 3.0, "S0")])
    pieces = spare_other_speaker([Cut(2.0, 3.0, "restart")], ctx)
    assert [(round(c.start, 2), round(c.end, 2)) for c in pieces] == [(2.0, 2.37), (2.63, 3.0)]


def test_a_cut_of_the_same_speakers_words_is_untouched() -> None:
    ctx = _edge_ctx([_w("um", 1.0, 1.3, "S0"), _w("so", 1.2, 1.9, "S0")])
    cuts = [Cut(1.0, 1.3, "filler", "um")]
    (cut,) = spare_other_speaker(cuts, ctx)
    assert (cut.start, cut.end, cut.provenance) == (1.0, 1.3, {"speaker": "S0"})
    assert ctx.spared_other_speaker == 0


def test_speakerless_cuts_pass_through_the_guard_unchanged() -> None:
    ctx = _edge_ctx([_w("um", 1.0, 1.3, None), _w("so", 1.2, 1.9, None)])
    cuts = [Cut(1.0, 1.3, "filler", "um")]
    assert spare_other_speaker(cuts, ctx) == cuts
    assert ctx.spared_other_speaker == 0


def test_finalize_spares_a_filler_that_would_have_swallowed_another_speakers_word() -> None:
    # The filler covers half of the S1 word, so the edge steps would remove it whole. The guard
    # keeps it: the cut stops before the S1 word.
    ctx = _edge_ctx([_w("um", 1.0, 1.4, "S0"), _w("yeah", 1.2, 1.6, "S1"), _w("ok", 4.0, 4.4, "S0")])
    out = finalize_cuts([Cut(1.0, 1.4, "filler", "um")], ctx)
    assert ctx.spared_other_speaker >= 1
    assert not any(c.end > 1.2 and c.start < 1.6 for c in out)


def test_a_judged_cut_never_leaves_another_speakers_word_clipped_mid_word() -> None:
    # The retake's last word "so" is S0's; S1's "okay" starts inside it. The S1 word must stay whole,
    # and so must "so": the cut stops before it rather than shaving its tail.
    words = [_w("so", 1.0, 1.5, "S0"), _w("okay", 1.4, 2.4, "S1"), _w("yes", 3.0, 3.4, "S0")]
    out = finalize_cuts([Cut(1.0, 1.7, "retake", "so")], _edge_ctx(words))
    for word in words[:2]:
        overlapped = [c for c in out if c.end > word.start and c.start < word.end]
        assert all(c.start <= word.start and c.end >= word.end for c in overlapped), word.text


def test_a_judged_cut_covering_most_of_another_speakers_word_removes_it_whole() -> None:
    words = [_w("so", 1.0, 1.5, "S0"), _w("okay", 1.4, 2.4, "S1"), _w("yes", 3.0, 3.4, "S0")]
    out = finalize_cuts([Cut(1.0, 2.2, "content", "so okay")], _edge_ctx(words))
    other = words[1]
    assert any(c.start <= other.start and c.end >= other.end for c in out)


# --- config and memo keys -----------------------------------------------------

def test_speaker_names_load_from_the_cook_config(tmp_path) -> None:
    path = tmp_path / "talkinghead.toml"
    path.write_text('[speaker_names]\nS0 = "Chavdar (host)"\nS1 = "brother"\n')
    assert load_config(path).speaker_names == {"S0": "Chavdar (host)", "S1": "brother"}


def test_config_dict_leaves_speaker_names_out_until_they_are_set() -> None:
    assert "speaker_names" not in Config().to_dict()
    assert Config(speaker_names=NAMES).to_dict()["speaker_names"] == NAMES


def test_stage_inputs_key_on_speaker_labels_and_names_only_when_present() -> None:
    assert "speakers" not in _stage_inputs(_ctx(SINGLE), "G")
    named = _ctx(TURNS, cfg=Config(votes=5, agree=4, speaker_names=NAMES))
    renamed = _ctx(TURNS, cfg=Config(votes=5, agree=4, speaker_names={"S0": "Other"}))
    assert _stage_inputs(named, "G")["speakers"] != _stage_inputs(renamed, "G")["speakers"]
