"""Fake --llm for talkinghead tests: prompt on stdin, canned JSON on stdout.

Responses come from the FAKE_TALKINGHEAD_RESPONSES env var (a JSON object
keyed by prompt marker); anything unmatched gets a conservative empty
response, so the machinery passes drive the chain.
"""

import json
import os
import sys

DEFAULT_RESPONSES = {
    "RETAKE-VERDICTS": {"verdicts": [], "cold_open_start": None, "outro_end": None},
    "PROSE-READ": {"cuts": []},
    "JOIN-QA": {"joins": []},
    "BLIND-QA": {"pass": True, "broken": []},
}

overrides = json.loads(os.environ.get("FAKE_TALKINGHEAD_RESPONSES", "null")) or {}
responses = {**DEFAULT_RESPONSES, **overrides}
prompt = sys.stdin.read()

for marker, val in responses.items():
    if marker in prompt:
        print(json.dumps(val))
        break
else:
    raise SystemExit(f"no canned response (env={os.environ.get('FAKE_TALKINGHEAD_RESPONSES', 'unset')[:80]}): {prompt[:120]}")
