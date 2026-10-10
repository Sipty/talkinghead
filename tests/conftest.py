"""Shared fixtures.

Every test starts without the TALKINGHEAD_* variables a developer may have
exported, so the judge, transcriber and config file come only from what the
test sets.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_talkinghead_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("TALKINGHEAD_JUDGE", "TALKINGHEAD_TRANSCRIBER", "TALKINGHEAD_CONFIG"):
        monkeypatch.delenv(name, raising=False)
