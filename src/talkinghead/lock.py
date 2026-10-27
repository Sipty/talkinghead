"""The edit lock: binds a plan to the audio master that passed the audio gates.

Audio is iterated on cheap audio-only renders; the picture is rendered once afterwards. The lock is what makes that
safe: it records the plan digest, the approved audio's sha256 and its packet MD5 (the hash that survives a `-c copy`
mux into the mp4), the frame rate the audio was snapped with, and optionally the transcript the audio was verified
with. Anything that renders the picture or times overlays checks the lock first, so an edited plan or a rewritten
master fails loudly instead of shipping a picture that no longer matches the approved audio.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from .store import atomic_write_text, sha256_file
from .timeline import Plan
from .transcribe import Transcript

# Bumped when the lock's shape changes, so an old lock is refused rather than half-read.
LOCK_VERSION = 1


class LockMismatch(RuntimeError):
    """The lock does not describe what is on disk; the message says what differs and where the lock is."""


def audio_stream_md5(path: str | Path) -> str:
    """MD5 of the first audio stream's packets, equal across containers that stream-copied the same packets."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0", "-c", "copy", "-f", "md5", "-"],
        capture_output=True, text=True,
    )
    out = proc.stdout.strip()
    if proc.returncode != 0 or not out.startswith("MD5="):
        raise RuntimeError(f"{path} has no readable audio stream: {proc.stderr.strip() or 'ffmpeg printed no hash'}")
    return out[len("MD5="):]


def plan_digest(plan: Plan) -> str:
    """sha256 of what decides the picture and audio bytes: segments, source hashes and the isolation track.

    Cut text, reasons, decisions, anchors, provenance and the talkinghead version are left out on purpose: they
    describe why the edit was made, not what it renders.
    """
    track = (plan.isolation or {}).get("track") or {}
    payload = {
        "segments": [
            {"source": s.source, "start": round(s.start, 3), "end": round(s.end, 3)} for s in plan.segments
        ],
        "sources": {k: v.sha256 for k, v in sorted(plan.sources.items())},
        "isolation_track": track.get("sha256"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _probe_audio(path: Path) -> dict:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_name,sample_rate,duration:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    d = json.loads(proc.stdout)
    streams = d.get("streams") or []
    if not streams:
        raise RuntimeError(f"{path} has no audio stream")
    s = streams[0]
    duration = s.get("duration") or (d.get("format") or {}).get("duration")
    return {
        "codec": s.get("codec_name"),
        "sample_rate": int(s["sample_rate"]) if s.get("sample_rate") else None,
        "duration": round(float(duration), 3) if duration else None,
    }


def _stored(path: Path, base: Path) -> str:
    """Relative to the lock's directory when inside it, so a moved run dir keeps a valid lock."""
    resolved, root = path.resolve(), base.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def _resolve(stored: str, lock_path: Path) -> Path:
    p = Path(stored)
    return p if p.is_absolute() else (lock_path.parent / p).resolve()


def write_lock(path: Path, plan: Plan, audio: Path, *, fps: float, transcript: Path | None = None) -> dict:
    """Record the approved audio master (and optionally its verified transcript) against the plan."""
    path = Path(path)
    audio = Path(audio)
    lock = {
        "version": LOCK_VERSION,
        "plan_sha": plan_digest(plan),
        "fps": fps,
        "output_duration": round(plan.output_duration, 3),
        "audio": {
            "path": _stored(audio, path.parent),
            "sha256": sha256_file(audio),
            "stream_md5": audio_stream_md5(audio),
            **_probe_audio(audio),
        },
        "transcript": (
            {"path": _stored(Path(transcript), path.parent), "sha256": sha256_file(transcript)}
            if transcript is not None else None
        ),
    }
    atomic_write_text(path, json.dumps(lock, indent=2, sort_keys=True) + "\n")
    return lock


def read_lock(path: Path) -> dict:
    """Load a lock, refusing a missing, unparsable or other-version file."""
    path = Path(path)
    try:
        lock = json.loads(path.read_text())
    except FileNotFoundError:
        raise LockMismatch(f"no edit lock at {path}; run `talkinghead lock` first") from None
    except (OSError, ValueError) as e:
        raise LockMismatch(f"edit lock {path} is unreadable ({e}); rewrite it with `talkinghead lock`") from None
    if not isinstance(lock, dict) or lock.get("version") != LOCK_VERSION:
        got = lock.get("version") if isinstance(lock, dict) else None
        raise LockMismatch(f"edit lock {path} has version {got!r}, expected {LOCK_VERSION}; rewrite it")
    return lock


def check_lock(lock: dict, plan: Plan, *, fps: float, lock_path: Path) -> Path:
    """Fail unless the plan, frame rate and audio file still match the lock; return the audio's absolute path."""
    digest = plan_digest(plan)
    if digest != lock["plan_sha"]:
        raise LockMismatch(
            f"plan changed since the lock: plan {digest[:12]} vs locked {lock['plan_sha'][:12]} ({lock_path}); "
            "re-lock the edited plan"
        )
    if fps != lock["fps"]:
        raise LockMismatch(f"fps {fps} differs from the locked fps {lock['fps']} ({lock_path}); render with the locked rate")
    audio = _resolve(lock["audio"]["path"], lock_path)
    if not audio.is_file():
        raise LockMismatch(f"locked audio {audio} is missing ({lock_path}); re-render the audio and re-lock")
    got = sha256_file(audio)
    if got != lock["audio"]["sha256"]:
        raise LockMismatch(
            f"audio {audio.name} changed since the lock: audio {got[:12]} vs locked "
            f"{lock['audio']['sha256'][:12]} ({lock_path}); re-lock the new master"
        )
    return audio


def locked_transcript(lock: dict, *, lock_path: Path) -> Transcript:
    """The transcript the audio was verified with, after proving its bytes are the recorded ones."""
    rec = lock.get("transcript")
    if not rec:
        raise LockMismatch(f"edit lock {lock_path} records no transcript; lock with a verified transcript")
    path = _resolve(rec["path"], lock_path)
    if not path.is_file():
        raise LockMismatch(f"locked transcript {path} is missing ({lock_path}); re-lock")
    got = sha256_file(path)
    if got != rec["sha256"]:
        raise LockMismatch(
            f"transcript {path.name} changed since the lock: transcript {got[:12]} vs locked "
            f"{rec['sha256'][:12]} ({lock_path}); re-lock"
        )
    return Transcript.from_dict(json.loads(path.read_text()))
