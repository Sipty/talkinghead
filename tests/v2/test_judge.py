"""The LLM layer: adapters behind one protocol, schema-validated verdicts,
k-vote stability, per-prompt caching, and pause-and-resume on rate limits."""

import json
import sys
from pathlib import Path

import pytest

from talkinghead.judge import (
    ClaudeCodeJudge,
    FakeJudge,
    JudgeError,
    JudgeUnavailable,
    LLMCache,
    RateLimited,
    ShellJudge,
    judge_model,
    preflight,
    vote,
    vote_items,
    vote_scalar,
    with_model,
)

SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "verdict": {"type": "string", "enum": ["cut", "keep"]}},
                "required": ["id", "verdict"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}


def _resp(*pairs: tuple[str, str]) -> dict:
    return {"verdicts": [{"id": i, "verdict": v} for i, v in pairs]}


def test_fake_judge_validates_against_the_schema() -> None:
    good = FakeJudge([_resp(("c1", "cut"))])
    assert good.decide("p", SCHEMA) == _resp(("c1", "cut"))
    bad = FakeJudge([{"verdicts": [{"id": "c1", "verdict": "maybe"}]}])
    with pytest.raises(JudgeError, match="schema"):
        bad.decide("p", SCHEMA)


def test_shell_judge_pipes_prompt_on_stdin_and_parses_fenced_json(tmp_path: Path) -> None:
    script = tmp_path / "echo_judge.py"
    script.write_text(
        "import sys; p = sys.stdin.read(); assert 'PROMPT-MARK' in p\n"
        "print('```json\\n{\"verdicts\": [{\"id\": \"c1\", \"verdict\": \"keep\"}]}\\n```')\n"
    )
    judge = ShellJudge(f"{sys.executable} {script}")
    assert judge.decide("PROMPT-MARK", SCHEMA) == _resp(("c1", "keep"))


def test_shell_judge_sends_the_guidelines_ahead_of_the_prompt(tmp_path: Path) -> None:
    # A shell command has no system-prompt flag, so the guidelines must ride in stdin.
    script = tmp_path / "guided.py"
    script.write_text(
        "import sys; p = sys.stdin.read(); assert p.index('RULE-MARK') < p.index('PROMPT-MARK')\n"
        "print('{\"verdicts\": []}')\n"
    )
    cmd = f"{sys.executable} {script}"
    assert ShellJudge(cmd, system_prompt="RULE-MARK").decide("PROMPT-MARK", SCHEMA) == {"verdicts": []}
    with pytest.raises(JudgeError):
        ShellJudge(cmd).decide("PROMPT-MARK", SCHEMA)


def test_answers_cached_before_guidelines_were_sent_still_replay(tmp_path: Path) -> None:
    old = tmp_path / "old.py"
    old.write_text("print('{\"verdicts\": [{\"id\": \"c1\", \"verdict\": \"keep\"}]}')\n")
    dead = tmp_path / "dead.py"
    dead.write_text("raise SystemExit(1)\n")
    cache = LLMCache(tmp_path / "llm")
    vote(ShellJudge(f"{sys.executable} {old}"), "p", SCHEMA, k=1, cache=cache)
    # Same judge, now carrying guidelines: the cached sample answers, the command never runs.
    again = vote(ShellJudge(f"{sys.executable} {dead}", system_prompt="rules"), "p", SCHEMA, k=1, cache=cache)
    assert again.responses == [_resp(("c1", "keep"))]


def test_shell_judge_retries_once_on_invalid_json_then_fails(tmp_path: Path) -> None:
    counter = tmp_path / "n"
    counter.write_text("0")
    script = tmp_path / "flaky.py"
    script.write_text(
        f"from pathlib import Path\nn = int(Path({str(counter)!r}).read_text()); Path({str(counter)!r}).write_text(str(n + 1))\n"
        "print('not json' if n == 0 else '{\"verdicts\": []}')\n"
    )
    judge = ShellJudge(f"{sys.executable} {script}")
    assert judge.decide("p", SCHEMA) == {"verdicts": []}
    assert counter.read_text() == "2"
    counter.write_text("0")
    always_bad = tmp_path / "bad.py"
    always_bad.write_text("print('nope')\n")
    with pytest.raises(JudgeError):
        ShellJudge(f"{sys.executable} {always_bad}").decide("p", SCHEMA)


def _argv_probe(tmp_path: Path) -> str:
    """A shell judge that records its own argv and answers valid JSON."""
    script = tmp_path / "argv_judge.py"
    script.write_text(
        "import json, sys\n"
        f"open({str(tmp_path / 'argv')!r}, 'w').write(chr(10).join(sys.argv[1:]))\n"
        "print(json.dumps({'verdicts': [{'id': 'c1', 'verdict': 'keep'}]}))\n"
    )
    return str(script)


def test_shell_judge_model_override_rewrites_the_model_argument(tmp_path: Path) -> None:
    script = _argv_probe(tmp_path)
    judge = ShellJudge(
        f"{sys.executable} {script} -p --no-session --mode text --model callers/model",
        model="override/model",
    )
    assert judge.decide("p", SCHEMA) == _resp(("c1", "keep"))
    assert judge.cmd == f"{sys.executable} {script} -p --no-session --mode text --model override/model"
    assert (tmp_path / "argv").read_text().splitlines() == [
        "-p", "--no-session", "--mode", "text", "--model", "override/model",
    ]


def test_shell_judge_appends_a_model_to_a_command_that_names_none(tmp_path: Path) -> None:
    script = _argv_probe(tmp_path)
    judge = ShellJudge(f"{sys.executable} {script} -p", model="override/model")
    judge.decide("p", SCHEMA)
    assert (tmp_path / "argv").read_text().splitlines() == ["-p", "--model", "override/model"]


def test_shell_judge_without_an_override_uses_the_configured_command_verbatim() -> None:
    cmd = "omp -p --no-session --no-tools --mode text --model openrouter/deepseek/deepseek-v4.1-flash"
    judge = ShellJudge(cmd)
    assert judge.cmd == cmd
    assert judge.model == "openrouter/deepseek/deepseek-v4.1-flash"


def test_a_model_override_keeps_the_rest_of_the_command_character_for_character() -> None:
    # The command is edited, never split and re-joined: quotes and
    # substitutions in it must survive the swap.
    assert with_model("judge --note 'two words' --model old", "new") == (
        "judge --note 'two words' --model new"
    )
    assert with_model('judge -z "$(cat)" --model old', "new") == 'judge -z "$(cat)" --model new'


def test_shell_judges_differing_only_in_model_get_different_names() -> None:
    # The cache key is the judge's name. Identical names would replay the
    # primary's cached answers after a switch to the fallback.
    primary = ShellJudge("omp -p --model openrouter/deepseek/deepseek-v4.1-flash")
    backup = ShellJudge("omp -p --model openrouter/z-ai/glm-5.3-flash")
    assert primary.name == "shell:omp:openrouter/deepseek/deepseek-v4.1-flash"
    assert backup.name == "shell:omp:openrouter/z-ai/glm-5.3-flash"
    assert primary.name != backup.name
    assert ShellJudge("omp -p").name == "shell:omp"


def test_claude_code_judge_builds_a_headless_command_and_reads_structured_output(tmp_path: Path) -> None:
    fake_claude = tmp_path / "claude"
    fake_claude.write_text(
        "#!/bin/sh\n"
        "printf '%s' \"$@\" > " + str(tmp_path / "argv") + "\n"
        "cat > " + str(tmp_path / "stdin") + "\n"
        'echo \'{"is_error": false, "structured_output": {"verdicts": [{"id": "c1", "verdict": "cut"}]}, "result": "..."}\'\n'
    )
    fake_claude.chmod(0o755)
    judge = ClaudeCodeJudge(model="claude-fable-5-1", system_prompt="SYS", claude_bin=str(fake_claude))
    assert judge.decide("the prompt", SCHEMA) == _resp(("c1", "cut"))
    argv = (tmp_path / "argv").read_text()
    for flag in ("-p", "--output-format", "json", "--json-schema", "--model", "claude-fable-5-1",
                 "--no-session-persistence", "--system-prompt", "SYS", "--tools"):
        assert flag in argv
    assert (tmp_path / "stdin").read_text() == "the prompt"


def test_claude_code_judge_maps_usage_limit_errors_to_rate_limited(tmp_path: Path) -> None:
    fake_claude = tmp_path / "claude"
    fake_claude.write_text(
        "#!/bin/sh\ncat > /dev/null\n"
        'echo \'{"is_error": true, "result": "You have hit your usage limit. Resets at 3pm"}\'\n'
    )
    fake_claude.chmod(0o755)
    judge = ClaudeCodeJudge(claude_bin=str(fake_claude))
    with pytest.raises(RateLimited):
        judge.decide("p", SCHEMA)


def test_an_explicit_claude_judge_still_selects_the_claude_adapter() -> None:
    from talkinghead.cli import build_judge

    judge = build_judge("claude", model="claude-opus-5", guidelines="G", effort=None)
    assert isinstance(judge, ClaudeCodeJudge) and judge.model == "claude-opus-5"


def test_vote_items_requires_supermajority_and_ties_resolve_to_none() -> None:
    responses = [
        _resp(("c1", "cut"), ("c2", "cut"), ("c3", "cut")),
        _resp(("c1", "cut"), ("c2", "keep"), ("c3", "cut")),
        _resp(("c1", "cut"), ("c2", "keep"), ("c3", "cut")),
        _resp(("c1", "cut"), ("c2", "cut"), ("c3", "keep")),
        _resp(("c1", "cut"), ("c2", "keep")),  # c3 missing in one sample
    ]
    tally = vote_items(responses, items_key="verdicts", id_key="id", value_key="verdict", agree=4)
    assert tally["c1"].winner == "cut" and tally["c1"].counts == {"cut": 5}
    assert tally["c2"].winner is None and tally["c2"].counts == {"cut": 2, "keep": 3}
    assert tally["c3"].counts == {"cut": 3, "keep": 1}
    assert tally["c3"].winner is None  # 3 of 5 is not a supermajority


def test_vote_scalar_majority() -> None:
    assert vote_scalar([{"pick": "S1"}, {"pick": "S1"}, {"pick": "S2"}, {"pick": "S1"}, {"pick": "S1"}], "pick", agree=4).winner == "S1"
    assert vote_scalar([{"pick": "S1"}, {"pick": "S2"}, {"pick": "S1"}], "pick", agree=3).winner is None


def test_vote_samples_k_times_and_caches_every_sample(tmp_path: Path) -> None:
    judge = FakeJudge([_resp(("c1", "cut"))] * 10)
    cache = LLMCache(tmp_path / "llm")
    r1 = vote(judge, "prompt A", SCHEMA, k=5, cache=cache)
    assert judge.calls == 5
    assert len(r1.responses) == 5
    r2 = vote(judge, "prompt A", SCHEMA, k=5, cache=cache)
    assert judge.calls == 5              # replayed from cache, not re-asked
    assert r2.responses == r1.responses
    assert (tmp_path / "llm" / f"{r1.prompt_sha}.prompt.txt").read_text() == "prompt A"
    assert len(list((tmp_path / "llm").glob(f"{r1.prompt_sha}-*.json"))) == 5
    vote(judge, "prompt B", SCHEMA, k=5, cache=cache)
    assert judge.calls == 10
    # A different judge never replays this judge's answers.
    other = FakeJudge([_resp(("c1", "keep"))] * 5)
    other.name = "other-model"
    vote(other, "prompt A", SCHEMA, k=5, cache=cache)
    assert other.calls == 5


def test_vote_pauses_and_resumes_on_rate_limit(tmp_path: Path) -> None:
    def script():
        yield RateLimited("usage limit")
        yield RateLimited("usage limit")
        while True:
            yield _resp(("c1", "keep"))

    judge = FakeJudge(script())
    waits: list[float] = []
    result = vote(
        judge, "p", SCHEMA, k=3, cache=LLMCache(tmp_path / "llm"),
        sleep=lambda s: waits.append(s), max_wait=3600.0,
    )
    assert len(result.responses) == 3
    assert len(waits) == 2 and waits[1] > waits[0]
    # Samples completed before the pause were persisted: only 3 real answers were needed.
    assert judge.calls == 5


def test_vote_escalates_a_limit_that_outlasts_the_bounded_wait(tmp_path: Path) -> None:
    judge = FakeJudge(RateLimited("usage limit") for _ in range(10))
    waits: list[float] = []
    with pytest.raises(JudgeUnavailable, match="still limited"):
        vote(judge, "p", SCHEMA, k=3, cache=LLMCache(tmp_path / "llm"),
             sleep=lambda s: waits.append(s), max_limit_waits=3)
    assert len(waits) == 3


def _fake_omp(tmp_path: Path, body: str) -> str:
    """An executable named `omp` whose stdout/stderr/exit come from `body`."""
    omp = tmp_path / "omp"
    omp.write_text(f"#!{sys.executable}\nimport json, sys\nsys.stdin.read()\n{body}")
    omp.chmod(0o755)
    return f"{omp} -p --no-session --mode text --model openai-codex/gpt-6.1-sol"


def _events(provider: str, model: str, stop: str = "stop", error: str = "") -> str:
    msg = {"role": "assistant", "provider": provider, "model": model, "stopReason": stop,
           "content": [{"type": "text", "text": '{"ok": true}'}]}
    if error:
        msg["errorMessage"] = error
    return f"print(json.dumps({{'type': 'message_end', 'message': {msg!r}}}))\n"


def test_preflight_accepts_the_pinned_model_and_asks_for_json_events(tmp_path: Path) -> None:
    # The fake only answers when the preflight switched the command to --mode json.
    cmd = _fake_omp(tmp_path, "assert sys.argv[sys.argv.index('--mode') + 1] == 'json'\n"
                    + _events("openai-codex", "gpt-6.1-sol"))
    preflight(ShellJudge(cmd))


@pytest.mark.parametrize("body, why", [
    # omp fuzzy-matches --model, so a reply from another provider is not the pin.
    (_events("openrouter", "openai/gpt-6.1-sol"), "omp served openrouter/openai/gpt-6.1-sol"),
    (_events("openai-codex", "gpt-6.1-sol", stop="error", error="usage limit reached"), "usage limit"),
    ("print('No API key found for openai-codex', file=sys.stderr)\nraise SystemExit(1)\n", "No API key"),
])
def test_preflight_escalates_when_the_pinned_judge_cannot_serve(tmp_path: Path, body: str, why: str) -> None:
    with pytest.raises(JudgeUnavailable, match=why):
        preflight(ShellJudge(_fake_omp(tmp_path, body)))


def test_a_logged_out_judge_mid_run_escalates_instead_of_retrying(tmp_path: Path) -> None:
    cmd = _fake_omp(tmp_path, "print('No API key found for openai-codex', file=sys.stderr)\nraise SystemExit(1)\n")
    with pytest.raises(JudgeUnavailable):
        ShellJudge(cmd).decide("p", SCHEMA)


def test_fallback_judge_switches_to_the_backup_when_the_primary_hits_its_usage_limit() -> None:
    from talkinghead.judge import FallbackJudge

    primary = FakeJudge([_resp(("c1", "cut")), RateLimited("usage limit reached")])
    primary.name = "claude-code:claude-fable-5-1"
    backup = FakeJudge([_resp(("c1", "keep"))] * 3)
    backup.name = "claude-code:claude-opus-5"
    judge = FallbackJudge(primary, backup)
    assert judge.decide("p", SCHEMA) == _resp(("c1", "cut"))
    assert judge.name == "claude-code:claude-fable-5-1"
    assert judge.decide("p", SCHEMA) == _resp(("c1", "keep"))   # primary limited -> backup answers
    assert judge.switched and judge.name == "claude-code:claude-opus-5"
    assert judge.decide("p", SCHEMA) == _resp(("c1", "keep"))   # stays on the backup
    assert primary.calls == 2 and backup.calls == 2


def test_fallback_judge_propagates_a_limited_backup_so_the_vote_waits() -> None:
    from talkinghead.judge import FallbackJudge

    primary = FakeJudge([RateLimited("limit")])
    backup = FakeJudge([RateLimited("limit too")])
    with pytest.raises(RateLimited):
        FallbackJudge(primary, backup).decide("p", SCHEMA)


def test_a_rate_limited_shell_command_switches_to_the_fallback_command(tmp_path: Path) -> None:
    from talkinghead.judge import FallbackJudge

    script = tmp_path / "judge.py"
    script.write_text(
        "import json, sys\n"
        "if sys.argv[1] == 'limited':\n"
        "    print('you have hit your usage limit', file=sys.stderr)\n"
        "    raise SystemExit(1)\n"
        "print(json.dumps({'verdicts': [{'id': 'c1', 'verdict': 'keep'}]}))\n"
    )
    primary = ShellJudge(
        f"{sys.executable} {script} limited --model openrouter/deepseek/deepseek-v4.1-flash")
    backup = ShellJudge(f"{sys.executable} {script} ok --model openrouter/z-ai/glm-5.3-flash")
    judge = FallbackJudge(primary, backup)
    assert judge.name == primary.name != backup.name
    assert judge.decide("p", SCHEMA) == _resp(("c1", "keep"))  # primary limited -> fallback answers
    assert judge.switched and judge.name == backup.name
    assert judge.decide("p", SCHEMA) == _resp(("c1", "keep"))  # stays on the fallback


# ------------------------------------------------------------------- panel


def _panel_imports() -> None:
    from talkinghead.judge import PanelJudge  # noqa: F401


def test_panel_judge_round_robins_samples_across_members() -> None:
    from talkinghead.judge import PanelJudge

    a = FakeJudge([_resp(("c1", "cut"))] * 4)
    b = FakeJudge([_resp(("c1", "keep"))] * 4)
    panel = PanelJudge([a, b])
    assert panel.name == "fake + fake"
    assert [panel.decide("p", SCHEMA) for _ in range(4)] == [
        _resp(("c1", "cut")), _resp(("c1", "keep")),
        _resp(("c1", "cut")), _resp(("c1", "keep")),
    ]
    assert a.calls == 2 and b.calls == 2
    assert panel.last_responder == "fake"


def test_panel_judge_failing_attempt_does_not_consume_the_turn() -> None:
    from talkinghead.judge import PanelJudge

    a = FakeJudge([JudgeError("bad answer"), _resp(("c1", "cut"))])
    b = FakeJudge([_resp(("c1", "keep"))] * 4)
    panel = PanelJudge([a, b])
    # Member a fails on sample 0; vote() retries it on a, then sample 1
    # moves to member b.
    with pytest.raises(JudgeError):
        panel.decide("p", SCHEMA)
    assert panel.decide("p", SCHEMA) == _resp(("c1", "cut"))
    assert panel.decide("p", SCHEMA) == _resp(("c1", "keep"))
    assert a.calls == 2 and b.calls == 1
    assert panel.last_responder == "fake"


def test_panel_judge_needs_members() -> None:
    from talkinghead.judge import PanelJudge

    with pytest.raises(ValueError):
        PanelJudge([])


def test_build_judge_wires_the_panel_and_records_the_responder(tmp_path: Path) -> None:
    from talkinghead.cli import build_judge
    from talkinghead.judge import PanelJudge, ShellJudge

    script = tmp_path / "judge.py"
    script.write_text(
        "import json,sys\n"
        "prompt = sys.stdin.read()\n"
        "print(json.dumps({'retakes': []}))\n"
    )
    judge = build_judge(
        f"shell:{sys.executable} {script}", model=None, guidelines="g", effort=None,
        fallback_command=None, timeout=10.0,
        panel=[f"{sys.executable} {script}", f"{sys.executable} {script}"],
    )
    assert isinstance(judge, PanelJudge)
    assert len(judge.members) == 3
    assert all(isinstance(m, ShellJudge) for m in judge.members)
    cache = LLMCache(tmp_path / "llm")
    vote(judge, "PROMPT", {"type": "object", "properties": {"retakes": {"type": "array"}}, "required": ["retakes"]},
         k=3, cache=cache)
    # Sample i names the member that answered it.
    from talkinghead.judge import prompt_sha
    sha = prompt_sha("PROMPT", judge.name)
    recorded = [json.loads((tmp_path / "llm" / f"{sha}-{i}.json").read_text())["judge"] for i in range(3)]
    assert recorded == [judge.members[i % 3].name for i in range(3)]


def test_build_judge_rejects_a_panel_on_the_claude_judge() -> None:
    from talkinghead.cli import build_judge

    with pytest.raises(ValueError):
        build_judge("claude", model="claude-opus-5", guidelines="g", effort=None, panel=["some cmd"])


def test_a_judge_command_that_is_not_installed_is_named_not_blamed_on_omp() -> None:
    with pytest.raises(JudgeUnavailable, match="judge command not found: no-such-judge-xyz"):
        ShellJudge("no-such-judge-xyz --json").preflight()


def test_a_missing_claude_binary_fails_the_preflight_with_its_name(tmp_path: Path) -> None:
    judge = ClaudeCodeJudge(claude_bin=str(tmp_path / "no-claude"))
    with pytest.raises(JudgeUnavailable, match="judge command not found"):
        preflight(judge)


def test_a_claude_judge_without_a_model_passes_no_model_flag_and_is_labelled_plainly() -> None:
    judge = ClaudeCodeJudge(model=None)
    assert "--model" not in judge.argv({"type": "object"})
    assert judge.name == "claude-code"
    assert ClaudeCodeJudge(model="claude-opus-5").name == "claude-code:claude-opus-5"


def test_a_custom_shell_judge_is_labelled_by_its_own_command_and_model() -> None:
    assert ShellJudge("python3 judge.py").name == "shell:python3"
    assert ShellJudge("omp -p --model a/b").name == "shell:omp:a/b"


def test_judge_model_is_the_given_model_else_the_one_the_command_names() -> None:
    assert judge_model("shell:omp -p --model a/b", None) == "a/b"
    assert judge_model("shell:omp -p --model a/b", "c/d") == "c/d"
    assert judge_model("shell:python3 judge.py", None) is None
    assert judge_model("claude", None) is None
    assert judge_model("claude", "claude-opus-5") == "claude-opus-5"
