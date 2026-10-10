"""Content-addressed cache and the run directory.

Determinism starts here: the transcript, the audio analysis and every LLM
verdict are computed once and stored under the SHA-256 of the audio they were
computed from. A rerun of the same file finds them and never re-samples a
nondeterministic model. The run directory holds the plan, the report, the
persisted structural decisions and a memo per stage keyed by the hash of its
inputs, which is what makes `talkinghead run` resumable without flags.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_CACHE_DIR = Path(os.environ.get("TALKINGHEAD_CACHE", Path.home() / ".cache" / "talkinghead"))


class RunDirConflict(RuntimeError):
    """The run directory is the cook for a different recording."""


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def stable_key(params: Any, version: int) -> str:
    """Hex digest of (params, version) with sorted keys: order-independent."""
    payload = json.dumps({"params": params, "version": version}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def atomic_write_text(path: Path, text: str) -> None:
    """Replace `path` atomically. The temp file is unique per call, so concurrent writers never share one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _dump(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


class Cache:
    """`<root>/<content sha>/<kind>-<key>.json` entries, computed at most once."""

    def __init__(self, root: Path = DEFAULT_CACHE_DIR) -> None:
        self.root = Path(root)

    def path(self, content_sha: str, kind: str, params: Any, version: int) -> Path:
        return self.root / content_sha / f"{kind}-{stable_key(params, version)}.json"

    def get(self, content_sha: str, kind: str, params: Any, version: int) -> Any | None:
        p = self.path(content_sha, kind, params, version)
        if not p.is_file():
            return None
        return json.loads(p.read_text())["value"]

    def put(self, content_sha: str, kind: str, params: Any, version: int, value: Any) -> None:
        atomic_write_text(
            self.path(content_sha, kind, params, version),
            _dump({"kind": kind, "params": params, "version": version, "value": value}),
        )

    def get_or_compute(
        self, content_sha: str, kind: str, params: Any, version: int, compute: Callable[[], Any]
    ) -> Any:
        cached = self.get(content_sha, kind, params, version)
        if cached is not None:
            return cached
        value = compute()
        self.put(content_sha, kind, params, version, value)
        return value


class RunDir:
    """`<video>.talkinghead/` layout for one run."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        # Stage memos recomputed over an older key, in call order: their previous
        # content is kept beside them, because the memos are hand-editable.
        self.superseded: list[str] = []

    # -- layout
    @property
    def manifest(self) -> Path:
        return self.root / "run.json"

    @property
    def plan(self) -> Path:
        return self.root / "plan.json"

    @property
    def report(self) -> Path:
        return self.root / "report.md"

    @property
    def decisions(self) -> Path:
        return self.root / "decisions.json"

    @property
    def llm_dir(self) -> Path:
        return self.root / "llm"

    @property
    def verify_dir(self) -> Path:
        return self.root / "verify"

    @property
    def memo_dir(self) -> Path:
        return self.root / "stages"

    def ensure(self) -> "RunDir":
        for d in (self.root, self.llm_dir, self.verify_dir, self.memo_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def bind(self, source: str | Path, source_sha256: str) -> None:
        """Claim this directory for one recording, on the first cook.

        A cook owns its directory end to end: master, memos, plan, renders,
        report. Pointing a second recording at the same directory would mix
        its renders with the first one's and, worse, hand it the first one's
        persisted structural answers, so a directory records the picture it
        was created for and refuses to change hands. Directories that predate
        the manifest (or were cleaned by hand) are adopted as they are.
        """
        entry = {
            "source": str(Path(source).resolve()),
            "source_sha256": source_sha256,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        stored = self.read_json(self.manifest, default=None)
        if stored is None:
            self.write_json(self.manifest, entry)
            return
        if stored.get("source_sha256") != source_sha256:
            raise RunDirConflict(
                f"{self.root} is the cook directory for {stored.get('source')} "
                f"(sha {str(stored.get('source_sha256'))[:12]}), not for {entry['source']} "
                f"(sha {source_sha256[:12]}). One cook, one directory: pass --out-dir DIR "
                f"to cook the new recording in a directory of its own."
            )

    # -- json helpers
    def write_json(self, path: Path, value: Any) -> None:
        atomic_write_text(path, _dump(value))

    def read_json(self, path: Path, default: Any = None) -> Any:
        if not path.is_file():
            return default
        return json.loads(path.read_text())

    # -- stage memo
    def memo(
        self, stage: str, inputs: Any, version: int, run: Callable[[], Any], *, fresh: bool = False
    ) -> Any:
        """Run `stage` unless an output for the same inputs+version is stored.

        A stored memo whose key no longer matches is COPIED to
        `stages/<stage>.superseded.json` before it is replaced. The memos are
        hand-editable - manual cuts live in them - so a re-key (a config field
        the stage reads, a version bump) must not destroy that work silently:
        the stage lands on `RunDir.superseded`, which the report names.
        """
        key = stable_key(inputs, version)
        path = self.memo_dir / f"{stage}.json"
        if not fresh and path.is_file():
            stored = json.loads(path.read_text())
            if stored.get("key") == key:
                return stored["value"]
        if path.is_file():
            keep = self.memo_dir / f"{stage}.superseded.json"
            if not keep.is_file():
                shutil.copyfile(path, keep)
            if not fresh:
                self.superseded.append(stage)
        value = run()
        self.write_json(path, {"stage": stage, "key": key, "version": version, "value": value})
        return value
