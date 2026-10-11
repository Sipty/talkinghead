"""talkinghead: deterministic single-render talking-head post-production.

The release version is the git tag (vX.Y.Z) that CI puts on every push to main
(see docs/releasing.md), read from the clone talkinghead runs from. Between
tags it reads like 1.1.0+3.gabc1234 (three commits past v1.1.0); outside a git
checkout it is "unknown".
"""

import subprocess
from pathlib import Path


def _version() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "describe", "--tags", "--match", "v[0-9]*.[0-9]*.[0-9]*", "--dirty"],
            capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    # v1.1.0 -> 1.1.0; v1.1.0-3-gabc1234-dirty -> 1.1.0+3.gabc1234.dirty
    tag, _, rest = proc.stdout.strip().removeprefix("v").partition("-")
    return f"{tag}+{rest.replace('-', '.')}" if rest else tag


__version__ = _version()
