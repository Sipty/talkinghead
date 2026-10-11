#!/usr/bin/env python3
"""Plan the next release from git history: the version to tag and its release notes.

Every push to main is a release (see docs/releasing.md and .github/workflows/release.yml).
The bump is the largest `Release:` trailer among the commits since the last vX.Y.Z tag:

    Release: major   -> X+1.0.0   (breaking change; the maintainer decides)
    Release: minor   -> X.Y+1.0   (a new feature or capability)
    (no trailer)     -> X.Y.Z+1   (fixes, docs, internal changes)

Usage: scripts/release.py [--notes-file PATH]
Prints the next version (without the "v"), or nothing when HEAD is already tagged.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
TRAILER_RE = re.compile(r"^release:\s*(\S+)\s*$", re.IGNORECASE)
# Higher wins when one push carries several commits.
BUMPS = {"patch": 0, "minor": 1, "major": 2}
# Separates commits in `git log` output; never appears in a commit message.
SEP = "\x1e"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def bump_of(message: str) -> str:
    """The bump one commit message asks for; ValueError on an unknown Release: value."""
    for line in message.splitlines():
        m = TRAILER_RE.match(line.strip())
        if m:
            value = m.group(1).lower()
            if value not in BUMPS:
                raise ValueError(f"unknown 'Release: {m.group(1)}' (use minor or major; patch is the default)")
            return value
    return "patch"


def next_version(current: tuple[int, int, int], bump: str) -> tuple[int, int, int]:
    major, minor, patch = current
    if bump == "major":
        return (major + 1, 0, 0)
    if bump == "minor":
        return (major, minor + 1, 0)
    return (major, minor, patch + 1)


def release_notes(messages: list[str]) -> str:
    """One bullet per commit, oldest first: the subject, then the body without trailers."""
    blocks = []
    for message in messages:
        subject, _, body = message.strip().partition("\n")
        lines = [ln for ln in body.strip().splitlines() if not TRAILER_RE.match(ln.strip())]
        body_text = "\n".join(f"  {ln}" if ln.strip() else "" for ln in lines).rstrip()
        blocks.append(f"- {subject.strip()}" + (f"\n{body_text}" if body_text else ""))
    return "\n".join(blocks) + "\n"


def latest_tag() -> str:
    tags = [t for t in _git("tag", "--merged", "HEAD", "--list", "v*").split() if TAG_RE.match(t)]
    if not tags:
        raise SystemExit("no vX.Y.Z tag reachable from HEAD: tag the first release by hand")
    return max(tags, key=lambda t: tuple(int(x) for x in TAG_RE.match(t).groups()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notes-file", type=Path, help="write the release notes here")
    args = parser.parse_args()

    if any(TAG_RE.match(t) for t in _git("tag", "--points-at", "HEAD").split()):
        return 0  # Already released (a re-run of the workflow on the same commit).
    last = latest_tag()
    log = _git("log", "--reverse", f"--format=%B{SEP}", f"{last}..HEAD")
    messages = [m for m in log.split(SEP) if m.strip()]
    if not messages:
        return 0
    try:
        bump = max((bump_of(m) for m in messages), key=BUMPS.__getitem__)
    except ValueError as e:
        print(f"release: {e}", file=sys.stderr)
        return 1
    current = tuple(int(x) for x in TAG_RE.match(last).groups())
    print(".".join(map(str, next_version(current, bump))))
    if args.notes_file:
        args.notes_file.write_text(release_notes(messages))
    return 0


if __name__ == "__main__":
    sys.exit(main())
