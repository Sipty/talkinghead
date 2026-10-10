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
])
def test_every_prompt_kind_gets_a_schema_valid_conservative_reply(prompt: str, schema: dict) -> None:
    validate(answer(prompt), schema)


def test_the_outro_keeps_to_the_final_candidate_and_the_cold_open_keeps_the_first_word() -> None:
    assert answer("TASK: OUTRO\n" + CANDIDATES)["last_sentence"] == "S0009"
    assert answer("TASK: COLD-OPEN\n" + CANDIDATES)["true_start"] == "KEEP_HEAD"


def test_an_unknown_prompt_fails_loudly_instead_of_guessing() -> None:
    with pytest.raises(SystemExit):
        answer("TASK: SOMETHING NEW\n")


def test_the_module_runs_as_a_shell_judge_on_stdin_and_stdout() -> None:
    proc = subprocess.run([sys.executable, "-m", "talkinghead.dryrun_judge"], input="TASK: RETAKES\n",
                          capture_output=True, text=True, check=True)
    assert json.loads(proc.stdout) == {"retakes": []}
