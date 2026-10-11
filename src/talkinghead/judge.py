"""The LLM layer: one protocol, several adapters, and the vote.

Design rules that make the pipeline reproducible:
- A judge never emits timestamps. Every schema it fills names candidates the
  code generated (ids) or picks from enumerated options.
- Every answer is validated against a JSON schema before anyone reads it.
- A decision is taken by sampling the same prompt k times and requiring a
  supermajority; anything short of that is "no decision" and the caller
  applies its conservative default (keep).
- Every sample is written to the run's `llm/` directory under the prompt's
  hash, so a resumed or repeated run replays instead of re-asking.
- A usage-limit error pauses the run (persisting progress) and resumes with
  exponential backoff; it never degrades the vote. A limit that outlasts the
  bounded wait, or a judge that cannot be reached at all (credential, model),
  stops the run with JudgeUnavailable so the user decides what happens next.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import jsonschema

from .store import atomic_write_text


class JudgeError(RuntimeError):
    """The judge did not return a usable, schema-valid answer."""


class RateLimited(JudgeError):
    """The provider refused for quota reasons; retry later, do not degrade."""


class JudgeUnavailable(JudgeError):
    """The configured judge cannot answer (credential, model, or a lasting quota).

    Never handled by switching judges: the run stops and the user is told.
    """


# Phrases in a judge command's stderr that mean "this provider will not serve
# you at all", as opposed to "not right now" (the rate-limit markers).
_UNAVAILABLE_MARKERS = (
    "no api key", "api key", "unauthorized", "401", "403", "forbidden",
    "not logged in", "authentication", "model not found", "not found",
)


def looks_unavailable(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _UNAVAILABLE_MARKERS)


_LIMIT_MARKERS = ("usage limit", "rate limit", "rate_limit", "too many requests", "429", "overloaded", "quota")


def looks_rate_limited(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _LIMIT_MARKERS)


def parse_json_text(raw: str) -> Any:
    """JSON from model text, tolerating ``` fences and surrounding prose."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def validate(value: Any, schema: dict) -> dict:
    try:
        jsonschema.validate(value, schema)
    except jsonschema.ValidationError as e:
        raise JudgeError(f"answer failed schema: {e.message}") from e
    return value


class Judge(Protocol):
    name: str

    def decide(self, prompt: str, schema: dict) -> dict: ...


class FakeJudge:
    """Scripted answers for tests; an Exception in the script is raised."""

    name = "fake"

    def __init__(self, responses: Iterable[Any]) -> None:
        self._it: Iterator[Any] = iter(responses)
        self.calls = 0

    def decide(self, prompt: str, schema: dict) -> dict:
        self.calls += 1
        item = next(self._it)
        if isinstance(item, Exception):
            raise item
        return validate(item, schema)


_MODEL_FLAG = re.compile(r"--model(?:\s*=\s*|\s+)\S+")


def command_model(cmd: str) -> str | None:
    """The model a judge command will call, or None if it names none."""
    argv = shlex.split(cmd)
    for i, arg in enumerate(argv):
        if arg == "--model" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--model="):
            return arg[len("--model=") :]
    return None


def with_model(cmd: str, model: str | None) -> str:
    """Point a judge command at another model.

    A "shell:" judge takes its model from its own argv, so an override has to
    be written into the command: the value of `--model` is swapped in place
    and the rest of the command -- flags, provider pinning -- is left alone.
    A command that has no `--model` token gets one appended. Without an
    override the command is used verbatim.
    """
    if not model:
        return cmd
    # Replaced textually, not split and re-joined: a judge command may carry
    # quotes or substitutions that a re-join would rewrite.
    if _MODEL_FLAG.search(cmd):
        return _MODEL_FLAG.sub(f"--model {model}", cmd, count=1)
    if re.search(r"(^|\s)--model\s*$", cmd):
        return re.sub(r"--model\s*$", f"--model {model}", cmd)
    return f"{cmd} --model {model}"


def judge_model(spec: str, model: str | None) -> str | None:
    """The model a judge will call: `model` when given, else the one its shell command names."""
    if model is not None or not spec.startswith("shell:"):
        return model
    return command_model(spec[len("shell:"):])


# Exit status of a shell that cannot find the command it was asked to run.
_NOT_FOUND = 127
# Appended to omp failures: a signed-out omp is the usual cause.
_OMP_HINT = " (if omp is signed out, run `omp login`)"


def _command_head(cmd: str) -> str:
    argv = shlex.split(cmd)
    return argv[0] if argv else ""


def _not_found(cmd: str) -> JudgeUnavailable:
    return JudgeUnavailable(f"judge command not found: {_command_head(cmd)}")


class ShellJudge:
    """Any command that reads the prompt on stdin and prints JSON on stdout."""

    def __init__(self, cmd: str, timeout: float = 1800.0, model: str | None = None) -> None:
        self.cmd = with_model(cmd, model)
        self.timeout = timeout
        self.model = command_model(self.cmd)
        # The model belongs in the name: the cache key is built from the
        # judge's name, so two shell judges differing only in --model must not
        # replay each other's answers.
        head = shlex.split(self.cmd)[0] if self.cmd.strip() else "?"
        self.name = f"shell:{head}" + (f":{self.model}" if self.model else "")

    def _run(self, prompt: str) -> str:
        proc = subprocess.run(
            self.cmd, shell=True, input=prompt, capture_output=True, text=True, timeout=self.timeout
        )
        if proc.returncode == _NOT_FOUND:
            raise _not_found(self.cmd)
        if proc.returncode != 0 or not proc.stdout.strip():
            tail = (proc.stderr or "").strip()[-800:]
            if looks_rate_limited(tail):
                raise RateLimited(tail)
            if looks_unavailable(tail):
                raise JudgeUnavailable(f"{self.name}: {tail}")
            raise JudgeError(f"judge command failed (exit {proc.returncode}): {tail}")
        return proc.stdout

    def preflight(self) -> None:
        """One tiny call before the run, so a missing judge fails in seconds.

        For an omp command the reply is requested as JSON events, which name
        the provider and model that actually answered: omp fuzzy-matches
        `--model`, so a successful reply alone does not prove the named model
        served it. Raises JudgeUnavailable on any failure.
        """
        head = _command_head(self.cmd)
        if Path(head).name != "omp":
            try:
                self.decide(PREFLIGHT_PROMPT, PREFLIGHT_SCHEMA, attempts=1)
            except JudgeUnavailable:
                raise
            except (JudgeError, subprocess.TimeoutExpired) as e:
                raise JudgeUnavailable(f"{self.name}: preflight failed: {e}") from e
            return
        cmd = _MODE_FLAG.sub("--mode json", self.cmd, count=1) if _MODE_FLAG.search(self.cmd) \
            else f"{self.cmd} --mode json"
        try:
            proc = subprocess.run(cmd, shell=True, input=PREFLIGHT_PROMPT, capture_output=True,
                                  text=True, timeout=PREFLIGHT_TIMEOUT)
        except subprocess.TimeoutExpired as e:
            raise JudgeUnavailable(f"{self.name}: no reply within {PREFLIGHT_TIMEOUT:.0f}s") from e
        reply = _last_assistant_message(proc.stdout)
        if reply is None:
            if proc.returncode == _NOT_FOUND:
                raise _not_found(self.cmd)
            tail = (proc.stderr or proc.stdout or "").strip()[-800:]
            raise JudgeUnavailable(f"{self.name}: exit {proc.returncode}: {tail}{_OMP_HINT}")
        if reply.get("stopReason") == "error":
            raise JudgeUnavailable(f"{self.name}: {str(reply.get('errorMessage', 'error'))[:800]}{_OMP_HINT}")
        served = f"{reply.get('provider')}/{reply.get('model')}"
        # A bare pattern ("opus") is a deliberate fuzzy pick; a provider/model
        # pin must be served exactly.
        if self.model and "/" in self.model and served != self.model:
            raise JudgeUnavailable(f"{self.name}: asked for {self.model} but omp served {served}")

    def decide(self, prompt: str, schema: dict, *, attempts: int = 4) -> dict:
        # The task prompts describe what to decide, not how to format it; a
        # literal model answers in plain text and fails the schema. Carry the
        # format contract here and retry: a model that keeps answering in
        # prose kills the whole vote pool if decide ever gives up early.
        instruction = (
            "OUTPUT FORMAT (mandatory): reply with a single JSON object and "
            "nothing else - no prose, no markdown fences. It must satisfy this "
            f"schema:\n{json.dumps(schema)}\n\n"
        )
        last = ""
        for attempt in range(attempts):
            try:
                raw = self._run(instruction + prompt)
            except subprocess.TimeoutExpired:
                # A hung shell judge must not kill the whole run: sol wedges
                # on large whole-transcript prompts and needs a fresh call.
                if attempt == attempts - 1:
                    raise
                prompt += (
                    "\n\nIMPORTANT: your previous attempt was abandoned after "
                    "the time limit. Respond immediately with the JSON object only."
                )
                continue
            last = raw
            try:
                return validate(parse_json_text(raw), schema)
            except (ValueError, JudgeError) as e:
                if attempt == attempts - 1:
                    raise JudgeError(
                        f"judge returned invalid JSON {attempts}x: {raw[:200]!r}"
                    ) from e
                prompt += (
                    "\n\nIMPORTANT: your previous reply was not valid JSON for the "
                    "required schema. Respond with the JSON object only - no prose, "
                    "no markdown."
                )
        raise JudgeError(f"unreachable: {last[:200]!r}")


_MODE_FLAG = re.compile(r"--mode(?:\s*=\s*|\s+)\S+")
PREFLIGHT_PROMPT = 'Reply with exactly this JSON object and nothing else: {"ok": true}'
PREFLIGHT_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
# A one-word reply takes seconds; anything near this means the provider is down.
PREFLIGHT_TIMEOUT = 180.0


def _last_assistant_message(events: str) -> dict | None:
    """The final assistant message from omp's `--mode json` event stream."""
    last = None
    for line in events.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = ev.get("message") if isinstance(ev, dict) and ev.get("type") == "message_end" else None
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            last = msg
    return last


class ClaudeCodeJudge:
    """Claude Code headless (`claude -p`) with structured output.

    Uses the user's Claude subscription; no API key. The prompt goes on stdin
    (a positional prompt after `--tools` would be eaten by that variadic
    flag). `--bare` is deliberately NOT used: it skips the settings that carry
    the login. The system prompt replaces Claude Code's default one, which
    otherwise costs tens of thousands of tokens per call.
    """

    def __init__(
        self,
        model: str | None = None,
        system_prompt: str | None = None,
        claude_bin: str = "claude",
        effort: str | None = None,
        timeout: float = 1800.0,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.claude_bin = claude_bin
        self.effort = effort
        self.timeout = timeout
        self.name = f"claude-code:{model}" if model else "claude-code"

    def argv(self, schema: dict) -> list[str]:
        argv = [
            self.claude_bin, "-p", "--output-format", "json",
            "--json-schema", json.dumps(schema, sort_keys=True),
            "--no-session-persistence",
        ]
        if self.model:
            argv += ["--model", self.model]
        if self.system_prompt:
            argv += ["--system-prompt", self.system_prompt]
        if self.effort:
            argv += ["--effort", self.effort]
        argv += ["--tools", ""]
        return argv

    def decide(self, prompt: str, schema: dict) -> dict:
        try:
            proc = subprocess.run(
                self.argv(schema), input=prompt, capture_output=True, text=True, timeout=self.timeout
            )
        except FileNotFoundError as e:
            raise JudgeUnavailable(f"judge command not found: {self.claude_bin}") from e
        stderr = (proc.stderr or "").strip()[-800:]
        if proc.returncode != 0:
            if looks_rate_limited(stderr):
                raise RateLimited(stderr)
            raise JudgeError(f"claude exited {proc.returncode}: {stderr}")
        try:
            envelope = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise JudgeError(f"claude returned a non-JSON envelope: {proc.stdout[:200]!r}") from e
        if envelope.get("is_error"):
            msg = str(envelope.get("result", ""))
            if looks_rate_limited(msg):
                raise RateLimited(msg)
            raise JudgeError(f"claude error: {msg[:300]}")
        value = envelope.get("structured_output")
        if value is None:
            value = parse_json_text(str(envelope.get("result", "")))
        return validate(value, schema)

    def preflight(self) -> None:
        """A missing binary fails here; a signed-out or limited account fails on one tiny call."""
        if shutil.which(self.claude_bin) is None:
            raise JudgeUnavailable(f"judge command not found: {self.claude_bin}")
        try:
            self.decide(PREFLIGHT_PROMPT, PREFLIGHT_SCHEMA)
        except (JudgeError, subprocess.TimeoutExpired) as e:
            raise JudgeUnavailable(f"{self.name}: preflight failed: {e}") from e


class FallbackJudge:
    """Primary judge with a backup for usage limits.

    The first RateLimited from the primary switches every later call to the
    backup for the rest of the run: a quota is a reason to answer through
    another judge, not to degrade the vote. The name follows the judge
    actually answering, so cached samples record who produced them -- which
    only holds if the two judges are named differently, hence the model in a
    shell judge's name. A limited backup propagates, and the vote waits.
    """

    def __init__(self, primary: Judge, backup: Judge) -> None:
        self.primary = primary
        self.backup = backup
        self.switched = False

    @property
    def name(self) -> str:
        return self.backup.name if self.switched else self.primary.name

    def decide(self, prompt: str, schema: dict) -> dict:
        if not self.switched:
            try:
                return self.primary.decide(prompt, schema)
            except RateLimited as e:
                print(f"talkinghead: {self.primary.name} limited ({str(e)[:80]}); switching to {self.backup.name}",
                      file=sys.stderr)
                self.switched = True
        return self.backup.decide(prompt, schema)


class PanelJudge:
    """A panel of judges: sample i is answered by member i mod n.

    Retake-heavy recordings benefit from more than one model on the vote, so
    the user can add panel members in a config file. Members keep
    their own names: prompt_sha hashes the panel name, so a panel run never
    replays a single-judge run's answers, and `last_responder` records which
    member actually produced a sample so the cache stays attributable.
    A member's RateLimited propagates (the vote pauses and retries it);
    silently rerouting a sample to another model would fake the vote.
    """

    def __init__(self, members: list[Judge]) -> None:
        if not members:
            raise ValueError("panel needs at least one judge")
        self.members = members
        self.last_responder: str = members[0].name
        self._last_prompt: str | None = None
        self._next = 0

    @property
    def name(self) -> str:
        return " + ".join(m.name for m in self.members)

    def decide(self, prompt: str, schema: dict) -> dict:
        # Round-robin per prompt: vote() asks k sequential samples of one
        # prompt. A failed attempt must not consume the turn, or a retry
        # would silently land on the wrong member.
        if prompt != self._last_prompt:
            self._last_prompt = prompt
            self._next = 0
        member = self.members[self._next % len(self.members)]
        answer = member.decide(prompt, schema)
        self.last_responder = member.name
        self._next += 1
        return answer


# ------------------------------------------------------------------- voting


@dataclass
class Tally:
    counts: dict[str, int]
    winner: str | None


def _tally(values: list[str], agree: int) -> Tally:
    counts = Counter(values)
    if not counts:
        return Tally({}, None)
    value, n = counts.most_common(1)[0]
    return Tally(dict(counts), value if n >= agree else None)


def vote_items(
    responses: list[dict], *, items_key: str, id_key: str, value_key: str, agree: int
) -> dict[str, Tally]:
    """Per-id supermajority over list-shaped answers; a missing id counts as no vote."""
    by_id: dict[str, list[str]] = {}
    for resp in responses:
        for item in resp.get(items_key, []):
            by_id.setdefault(str(item[id_key]), []).append(str(item[value_key]))
    return {k: _tally(v, agree) for k, v in sorted(by_id.items())}


def vote_scalar(responses: list[dict], key: str, *, agree: int) -> Tally:
    return _tally([str(r.get(key)) for r in responses if r.get(key) is not None], agree)


class LLMCache:
    """`llm/<prompt sha>.prompt.txt` + `llm/<prompt sha>-<i>.json` per sample."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def save_prompt(self, sha: str, prompt: str) -> None:
        p = self.root / f"{sha}.prompt.txt"
        if not p.is_file():
            atomic_write_text(p, prompt)

    def get(self, sha: str, i: int) -> dict | None:
        p = self.root / f"{sha}-{i}.json"
        return json.loads(p.read_text()) if p.is_file() else None

    def put(self, sha: str, i: int, value: dict, judge_name: str) -> None:
        atomic_write_text(
            self.root / f"{sha}-{i}.json",
            json.dumps({"judge": judge_name, "sample": i, "answer": value}, indent=2, sort_keys=True) + "\n",
        )


@dataclass
class VoteResult:
    prompt_sha: str
    responses: list[dict] = field(default_factory=list)


def prompt_sha(prompt: str, judge_name: str = "") -> str:
    """Cache key: the prompt AND the judge, so switching models never replays
    another model's answers."""
    return hashlib.sha256(f"{judge_name}\n{prompt}".encode()).hexdigest()[:24]


# Backoffs before a usage limit escalates: 60+120+240+480+960 s, about half an
# hour, which rides out overload blips but not an exhausted subscription window.
LIMIT_WAITS = 5


def vote(
    judge: Judge,
    prompt: str,
    schema: dict,
    *,
    k: int = 5,
    cache: LLMCache | None = None,
    sleep: Callable[[float], None] = time.sleep,
    base_wait: float = 60.0,
    max_wait: float = 3600.0,
    max_limit_waits: int = LIMIT_WAITS,
) -> VoteResult:
    """Sample the judge k times, replaying cached samples, pausing on limits.

    A limit still standing after `max_limit_waits` backoffs raises
    JudgeUnavailable: the user hears about it instead of the run idling.
    """
    sha = prompt_sha(prompt, judge.name)
    if cache is not None:
        cache.save_prompt(sha, prompt)
    result = VoteResult(sha)
    for i in range(k):
        cached = cache.get(sha, i) if cache is not None else None
        if cached is not None:
            result.responses.append(cached["answer"])
            continue
        attempt = 0
        while True:
            try:
                answer = judge.decide(prompt, schema)
                break
            except RateLimited as e:
                if attempt >= max_limit_waits:
                    raise JudgeUnavailable(
                        f"{judge.name}: still limited after {attempt} backoffs: {str(e)[:300]}") from e
                wait = min(max_wait, base_wait * (2 ** attempt))
                attempt += 1
                sleep(wait)
        if cache is not None:
            cache.put(sha, i, answer, getattr(judge, "last_responder", judge.name))
        result.responses.append(answer)
    return result


def preflight(judge: Judge) -> None:
    """Prove every judge that will vote can answer, before any hour-long stage.

    A fallback judge is not probed: it only answers once the primary is
    limited, and a single-judge run has none. Raises JudgeUnavailable.
    """
    if isinstance(judge, PanelJudge):
        for member in judge.members:
            preflight(member)
    elif isinstance(judge, FallbackJudge):
        preflight(judge.primary)
    elif isinstance(judge, (ShellJudge, ClaudeCodeJudge)):
        judge.preflight()
