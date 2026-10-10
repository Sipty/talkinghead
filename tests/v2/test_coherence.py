"""Tests for the coherence prompt builder (coherence.py)."""

from __future__ import annotations

from talkinghead.coherence import _covered, build_coherence_prompt
from talkinghead.semantic import Sentence
from talkinghead.types import CutSpan


def make_sents() -> list[Sentence]:
    return [
        Sentence("S0000", 0.0, 3.0, "Since I have been an engineer for ten years."),
        Sentence("S0001", 3.0, 6.0, "I can tell you these credentials matter."),
        Sentence("S0002", 6.0, 9.0, "Anyway, drop a comment below."),
        Sentence("S0003", 20.0, 23.0, "So why would you use model routing."),
    ]


def make_cuts() -> list[CutSpan]:
    """Cut #0 removes the credential sentence, cut #1 the meta-talk."""
    return [
        CutSpan(0.0, 3.0, "ramble", "Since I have been an engineer for ten years."),
        CutSpan(6.0, 9.0, "no-value", "Anyway, drop a comment below."),
    ]


# --- build_coherence_prompt ------------------------------------------------


def test_prompt_shows_surviving_script_with_cut_markers() -> None:
    prompt = build_coherence_prompt(make_sents(), make_cuts(), "model routing")
    assert "Topic: model routing" in prompt
    # Cut sentences collapse into markers, kept sentences appear by id.
    assert "[CUT #0 ramble]" in prompt
    assert "[CUT #1 no-value]" in prompt
    assert "S0001 | I can tell you these credentials matter." in prompt
    assert "S0003 | So why would you use model routing." in prompt
    # A cut sentence's text is not repeated inline in the script section.
    script = prompt.split("--- SCRIPT (cuts marked) ---")[1].split("--- CUTS ---")[0]
    assert "Since I have been an engineer" not in script


def test_prompt_lists_numbered_cuts_and_json_schema() -> None:
    prompt = build_coherence_prompt(make_sents(), make_cuts(), "")
    assert not prompt.startswith("Topic:")
    assert "#0 | 0.0-3.0 | ramble | Since I have been an engineer for ten years." in prompt
    assert "#1 | 6.0-9.0 | no-value | Anyway, drop a comment below." in prompt
    assert '"vetoes"' in prompt and '"extends"' in prompt and '"add_ids"' in prompt


def test_prompt_collapses_a_multi_sentence_cut_into_one_marker() -> None:
    cuts = [CutSpan(0.0, 6.0, "no-value", "two sentences")]
    prompt = build_coherence_prompt(make_sents(), cuts, "")
    assert prompt.count("[CUT #0 no-value]") == 1


def test_prompt_omits_cuts_outside_the_chunk() -> None:
    chunk = make_sents()[:2]  # 0.0-6.0
    cuts = make_cuts() + [CutSpan(300.0, 301.0, "silence", "far away")]
    prompt = build_coherence_prompt(chunk, cuts, "")
    assert "#0 |" in prompt
    assert "far away" not in prompt


def test_prompt_omits_cuts_that_remove_no_speech() -> None:
    # A silence trim between two sentences has nothing to judge, so it is not
    # offered for veto.
    cuts = make_cuts() + [CutSpan(9.5, 10.5, "silence", "")]
    prompt = build_coherence_prompt(make_sents(), cuts, "")
    assert "#2 |" not in prompt


def test_prompt_numbering_uses_global_cut_indices() -> None:
    # Ids must survive chunking: a veto in the last chunk still names the right
    # cut in the whole-plan list.
    chunk = make_sents()[3:]
    cuts = make_cuts() + [CutSpan(20.0, 23.0, "no-value", "why would you use model")]
    prompt = build_coherence_prompt(chunk, cuts, "")
    assert "[CUT #2 no-value]" in prompt
    assert "#2 | 20.0-23.0 | no-value | why would you use model" in prompt
    assert "#0 |" not in prompt


def test_prompt_padding_does_not_mark_a_neighbour_as_cut() -> None:
    # A cut padded past the next sentence's start must not swallow its id:
    # coverage is decided on the fraction of the sentence span covered.
    cuts = [CutSpan(0.0, 3.4, "ramble", "credential")]
    prompt = build_coherence_prompt(make_sents(), cuts, "")
    assert "S0001 | I can tell you these credentials matter." in prompt


def test_prompt_mid_sentence_cut_does_not_mark_whole_sentence_removed() -> None:
    # Regression: coverage used to match on the sentence midpoint, so a
    # mid-sentence cut made the WHOLE sentence display as removed and could
    # trigger a false coherence veto. A cut covering a minority of the
    # sentence must leave the sentence visible in the surviving script.
    sent = Sentence("S0000", 0.0, 6.0, "So we route the request through the model table.")
    cuts = [CutSpan(1.0, 2.5, "filler", "uh so basically")]
    prompt = build_coherence_prompt([sent], cuts, "")
    assert "S0000 | So we route the request" in prompt
    assert "[CUT #0" not in prompt


def test_prompt_marks_sentence_whose_majority_is_cut() -> None:
    # A cut removing most of the sentence still shows the sentence as removed.
    sent = Sentence("S0000", 0.0, 6.0, "So we route the request through the model table.")
    cuts = [CutSpan(1.0, 6.0, "ramble", "route the request through the model table")]
    prompt = build_coherence_prompt([sent], cuts, "")
    assert "[CUT #0 ramble]" in prompt


def test_coverage_reports_the_most_overlapping_cut() -> None:
    # Two cuts jointly cover a sentence; the marker names the bigger overlap.
    sent = Sentence("S0000", 0.0, 4.0, "sentence text")
    cuts = [CutSpan(0.0, 1.0, "filler", "first"), CutSpan(1.5, 4.0, "ramble", "second")]
    assert _covered(sent, cuts) == 1
