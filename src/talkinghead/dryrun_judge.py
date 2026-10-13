"""Offline dry-run judge: answers every prompt kind the pipeline sends, with no model.

It makes no editorial judgment. The cold open stays at the first word, no
retake, content or coherence cut is proposed, no marker is cut, and the plan
has no chapters or visuals. A run therefore exercises the whole machinery
(transcription, cutting, render, report) without an LLM. Its output is a
check of the install, not a cut anyone should publish.

    TALKINGHEAD_JUDGE='shell:python3 -m talkinghead.dryrun_judge' \\
    TALKINGHEAD_TRANSCRIBER=whisper talkinghead run clip.mov --no-visuals

An unknown prompt exits 2 instead of guessing, so a new prompt kind shows up here first.
"""

from __future__ import annotations

import json
import re
import sys

_CANDIDATES = re.compile(r"^- (\S+?):", re.M)
_CORE_FIRST = re.compile(r"--- CORE \(start blocks here\) ---\n(S\d+) \|")
_BLOCK_ROW = re.compile(r"^(B\d+) \|", re.M)
_FINDING_ID = re.compile(r"^(c\d+) \(", re.M)


def _last_candidate(prompt: str) -> str:
    """The final candidate id: the outro candidates end with the video's last sentence."""
    section = prompt.split("--- CANDIDATES ---", 1)[-1].split("--- FULL TRANSCRIPT ---", 1)[0]
    ids = _CANDIDATES.findall(section)
    return ids[-1] if ids else "KEEP_HEAD"


def answer(prompt: str) -> dict:
    """The conservative reply for one prompt, in the schema that prompt asks for."""
    if "Reply with exactly this JSON object" in prompt:
        return {"ok": True}
    if "TASK: COHERENCE" in prompt:
        return {"vetoes": [], "extends": []}
    if "TASK: COLD-OPEN" in prompt:
        return {"true_start": "KEEP_HEAD", "why": "dry run: keep the first word"}
    if "TASK: OUTRO" in prompt:
        return {"last_sentence": _last_candidate(prompt), "why": "dry run: keep to the last sentence"}
    if "TASK: RETAKES" in prompt:
        return {"retakes": []}
    if "TASK: MARKER" in prompt:
        return {"verdict": "keep", "why": "dry run: keep every marker"}
    if "TASK: CONTENT" in prompt:
        return {"cuts": []}
    if "TASK: POLISH WINDOW" in prompt:
        return {"cuts": []}
    if "TASK: OUTPUT PASS" in prompt:
        # Every finding the render's scan flags is kept: the dry run makes no editorial call.
        return {"verdicts": [{"id": c, "verdict": "keep", "why": "dry run: keep every finding"}
                             for c in _FINDING_ID.findall(prompt)]}
    if "TASK: EDITORIAL PLAN" in prompt:
        return {"central_argument": "dry run: no editorial plan", "chapters": [], "visuals": []}
    if "TASK: RESEARCH" in prompt:
        return {"sources": []}
    if "TASK: RESTRUCTURE WINDOW" in prompt:
        # One block per window, starting at the window's first core sentence: nothing dropped, nothing moved.
        core = _CORE_FIRST.search(prompt)
        if core is None:
            return {"blocks": []}
        return {"blocks": [{"start": core.group(1), "title": "dry run block", "summary": "one block per window",
                            "payoff": None, "payoff_sid": None, "interest": 1, "relevance": 0,
                            "depends_on_earlier": False, "depends_note": "", "lead_in": []}]}
    if "TASK: RESTRUCTURE KEEP" in prompt:
        return {"blocks": [{"id": b, "decision": "keep", "reason": "dry run: keep every block", "front_rank": None}
                           for b in _BLOCK_ROW.findall(prompt)]}
    raise SystemExit("dryrun judge: no answer for this prompt kind")


def main() -> int:
    print(json.dumps(answer(sys.stdin.read())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
