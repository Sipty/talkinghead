"""The offline dry-run judge answers every prompt kind the pipeline sends, in a schema-valid way."""

import json
import subprocess
import sys

import pytest

from talkinghead.dryrun_judge import answer
from talkinghead.judge import PREFLIGHT_PROMPT, PREFLIGHT_SCHEMA, validate
from talkinghead.prompts import (
    COHERENCE_SCHEMA,
    COLD_OPEN_SCHEMA,
    CONTENT_SCHEMA,
    EDITORIAL_SCHEMA,
    MARKER_SCHEMA,
    OUTRO_SCHEMA,
    RESEARCH_SCHEMA,
    RETAKE_SCHEMA,
)
from talkinghead.stages.polish import OUTPUT_SCHEMA, POLISH_SCHEMA, output_prompt

CANDIDATES = "--- CANDIDATES ---\n- S0001: Hello there.\n- S0009: Thanks for watching.\n\n--- FULL TRANSCRIPT ---\n- S0001: Hello there.\n"


@pytest.mark.parametrize("prompt, schema", [
    (PREFLIGHT_PROMPT, PREFLIGHT_SCHEMA),
    ("TASK: COLD-OPEN\n" + CANDIDATES, COLD_OPEN_SCHEMA),
    ("TASK: OUTRO\n" + CANDIDATES, OUTRO_SCHEMA),
    ("TASK: RETAKES\n[w0001] hello", RETAKE_SCHEMA),
    ("TASK: MARKER\nThe word 'sorry' appears.", MARKER_SCHEMA),
    ("TASK: CONTENT\n- S0001: hello", CONTENT_SCHEMA),
    ("TASK: COHERENCE\nRespond with JSON only:", COHERENCE_SCHEMA),
    ("TASK: EDITORIAL PLAN\nTRANSCRIPT", EDITORIAL_SCHEMA),
    ("TASK: RESEARCH\n- 0: type=meme", RESEARCH_SCHEMA),
    ("TASK: POLISH WINDOW\nCORE", POLISH_SCHEMA),
])
def test_every_prompt_kind_gets_a_schema_valid_conservative_reply(prompt: str, schema: dict) -> None:
    validate(answer(prompt), schema)


def test_the_outro_keeps_to_the_final_candidate_and_the_cold_open_keeps_the_first_word() -> None:
    assert answer("TASK: OUTRO\n" + CANDIDATES)["last_sentence"] == "S0009"
    assert answer("TASK: COLD-OPEN\n" + CANDIDATES)["true_start"] == "KEEP_HEAD"


def test_the_output_pass_keeps_every_finding_it_is_asked_about() -> None:
    from talkinghead.residual import scan
    from talkinghead.transcribe import Word

    words = [Word("uh,", 0.0, 0.3, 1.0), Word("I,", 0.5, 0.8, 1.0), Word("I", 0.8, 1.1, 1.0)]
    found = scan(words)
    reply = answer(output_prompt(words, found, None))
    validate(reply, OUTPUT_SCHEMA)
    assert [(v["id"], v["verdict"]) for v in reply["verdicts"]] == [(f"c{k}", "keep") for k in range(len(found))]


def test_an_unknown_prompt_fails_loudly_instead_of_guessing() -> None:
    with pytest.raises(SystemExit):
        answer("TASK: SOMETHING NEW\n")


def test_the_module_runs_as_a_shell_judge_on_stdin_and_stdout() -> None:
    proc = subprocess.run([sys.executable, "-m", "talkinghead.dryrun_judge"], input="TASK: RETAKES\n",
                          capture_output=True, text=True, check=True)
    assert json.loads(proc.stdout) == {"retakes": []}
