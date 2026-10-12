"""Stage 1: turn a video (or its wav) into timed words + audio truth.

Two layers live here: the raw backends - whisper (mlx-whisper on macOS, faster-whisper
elsewhere) and FluidAudio/Parakeet (macOS only) -
plus the waveform helpers that read a wav directly (extract_wav,
detect_silences), and the Transcriber interface above them, which transcribes
once per audio hash. The transcript is the pipeline's frozen input: whatever
backend produced it, the cache returns the same words for the same audio
forever after, which is what makes a rerun reproduce the same plan.

Non-obvious decisions:
  - Whisper word timestamps are NOT reliable ground truth (they drift, and
    quiet trailing utterances get dropped from the transcript). Silence cuts
    therefore come from waveform-level VAD (detect_silences), not from gaps
    between whisper words.
  - The main whisper run goes WITHOUT a filler prompt: biasing it makes
    whisper hallucinate filler tokens over real silence ("Um, good luck" where
    the audio is just a pause), which then cut into real speech. Fillers are
    recovered instead by re-probing glued word chunks (see derive.py), where a
    filler prompt is applied to the slice that provably contains audio.
  - Parakeet emits fillers as real tokens with native per-token timings, so it
    is the default backend; whisper is explicit opt-in, because its word spans
    absorb hesitations.
  - Scribe v2 is the normal backend, served through OpenRouter (elevenlabs/scribe-v2).
    Native word stamps plus sound tags (audio events: sighs, laughter). A recording too
    long for one request is cut at quiet points into overlapping chunks. Events are kept
    as data on the Transcript; analysis decides which are cuts and which are protected.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import wave
from array import array
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

import numpy as np

from .store import Cache, sha256_file

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
# The same large-v3-turbo weights for faster-whisper, used off macOS.
FASTER_WHISPER_MODEL = "large-v3-turbo"

# Biases whisper to transcribe fillers as tokens instead of swallowing them.
FILLER_PROMPT = "Uh, um, uhh, uhm, erm, er, hmm, like, you know, I mean, actually."


@dataclass
class Word:
    """One recognized word with its timing (seconds) and whisper confidence."""

    text: str
    start: float
    end: float
    probability: float


@dataclass
class AudioEvent:
    """A non-speech sound the backend tagged natively, e.g. "[sighs]"."""

    text: str
    start: float
    end: float


def _probe_duration(video_path: str) -> float:
    """Duration in seconds via ffprobe; RuntimeError with stderr on failure."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as e:
        raise RuntimeError("ffprobe not found; install ffmpeg (e.g. brew install ffmpeg)") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed for {video_path}: {e.stderr.strip()}") from e
    try:
        return float(proc.stdout.strip())
    except ValueError as e:
        raise RuntimeError(f"ffprobe returned no parseable duration for {video_path}: {proc.stdout!r}") from e


def extract_wav(video_path: str) -> str:
    """Extract 16 kHz mono PCM WAV to a tempfile; caller deletes it."""
    fd = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    fd.close()
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", video_path, "-ar", "16000", "-ac", "1", fd.name]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as e:
        Path(fd.name).unlink(missing_ok=True)
        raise RuntimeError("ffmpeg not found; install ffmpeg (e.g. brew install ffmpeg)") from e
    except subprocess.CalledProcessError as e:
        Path(fd.name).unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed extracting audio from {video_path}: {e.stderr.strip()}") from e
    return fd.name


def whisper_engine() -> str:
    """Which whisper engine runs here: mlx-whisper on macOS, faster-whisper elsewhere.

    Both decode the same large-v3-turbo weights. Their transcript versions differ
    (see whisper_version), so a cache never mixes the two engines' output.
    """
    return "mlx" if sys.platform == "darwin" else "faster"


def default_whisper_model() -> str:
    return DEFAULT_MODEL if whisper_engine() == "mlx" else FASTER_WHISPER_MODEL


def whisper_version(model: str) -> str:
    """The transcript version string for `model` on this machine's whisper engine."""
    return model if whisper_engine() == "mlx" else f"faster-whisper;{model}"


def _mlx_whisper():
    """mlx-whisper is Apple-only: it loads only when the whisper backend runs."""
    try:
        import mlx_whisper
    except ImportError as e:
        raise RuntimeError(f"the whisper backend needs mlx-whisper, which did not import ({e})") from None
    return mlx_whisper


_FASTER_MODELS: dict[str, object] = {}


def _faster_whisper(model: str):
    """faster-whisper (CTranslate2, no torch) model, loaded once per process."""
    if model not in _FASTER_MODELS:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError(f"the whisper backend needs faster-whisper, which did not import ({e})") from None
        _FASTER_MODELS[model] = WhisperModel(model, device="auto", compute_type="default")
    return _FASTER_MODELS[model]


def _whisper_engine_handle(model: str):
    """The loaded engine for `model`; RuntimeError when it is not installed."""
    return _mlx_whisper() if whisper_engine() == "mlx" else _faster_whisper(model)


def _whisper_decode(engine, audio: str, model: str, *, prompt: str | None = None,
                    word_timestamps: bool = False) -> dict:
    """One whisper decode in mlx-whisper's result shape: {"text", "segments": [{"text", "words"}]}.

    Both engines are normalised to that shape so the callers parse one format.
    """
    if whisper_engine() == "mlx":
        kwargs: dict = {"path_or_hf_repo": model}
        if word_timestamps:
            kwargs["word_timestamps"] = True
        if prompt is not None:
            kwargs["initial_prompt"] = prompt
        return engine.transcribe(audio, **kwargs)
    segments, _info = engine.transcribe(audio, word_timestamps=word_timestamps, initial_prompt=prompt)
    out = []
    for s in segments:
        words = [{"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                 for w in (s.words or [])]
        out.append({"text": s.text, "words": words})
    return {"text": "".join(s["text"] for s in out), "segments": out}


def transcribe(video_path: str, model: str | None = None, wav_path: str | None = None) -> tuple[float, list[Word]]:
    """Transcribe video audio via this platform's whisper engine; returns (duration, words).

    When wav_path is given the audio is extracted there (caller owns cleanup);
    otherwise a temp wav is created and removed here.
    """
    model = model or default_whisper_model()
    engine = _whisper_engine_handle(model)
    duration = _probe_duration(video_path)
    own_wav = wav_path is None
    wav = wav_path or extract_wav(video_path)
    try:
        result = _whisper_decode(engine, wav, model, word_timestamps=True)
    finally:
        if own_wav:
            Path(wav).unlink(missing_ok=True)
    words: list[Word] = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            words.append(
                Word(
                    text=str(w["word"]),
                    start=float(w["start"]),
                    end=float(w["end"]),
                    probability=float(w.get("probability", 0.0)),
                )
            )
    return duration, words


def find_fluidaudio() -> str | None:
    """Locate fluidaudiocli (NVIDIA Parakeet TDT ASR, macOS CoreML).

    Checked in order: the FLUIDAUDIO_BIN env var, then PATH.
    """
    p = os.environ.get("FLUIDAUDIO_BIN")
    if p and Path(p).exists():
        return p
    return shutil.which("fluidaudiocli")


def transcribe_parakeet(video_path: str, wav_path: str) -> tuple[float, list[Word]]:
    """Transcribe via FluidAudio (NVIDIA Parakeet TDT) with word timestamps.

    Parakeet emits fillers (um/uh/ah) as real tokens with native per-token
    timestamps - no whisper-style gluing or ~0.5s drift. Requires fluidaudiocli
    (see find_fluidaudio); raises RuntimeError when missing or on failure.
    """
    duration = _probe_duration(video_path)
    binary = find_fluidaudio()
    if not binary:
        raise RuntimeError(
            "fluidaudiocli not found: install FluidAudio's fluidaudiocli (see README), put it on PATH, "
            "or set FLUIDAUDIO_BIN to its path."
        )
    fd = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    fd.close()
    try:
        proc = subprocess.run(
            [binary, "transcribe", wav_path, "--word-timestamps", "--output-json", fd.name],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or "").strip().splitlines()[-15:])
            raise RuntimeError(f"fluidaudiocli failed (exit {proc.returncode}):\n{tail}")
        doc = json.loads(Path(fd.name).read_text())
    finally:
        Path(fd.name).unlink(missing_ok=True)
    words = [
        Word(
            text=str(w["word"]),
            start=float(w["startTime"]),
            end=float(w["endTime"]),
            probability=1.0,
        )
        for w in doc.get("wordTimings", [])
    ]
    return duration, words


def detect_silences(wav_path: str, frame_sec: float = 0.02, merge_gap: float = 0.25, min_silence: float = 0.15) -> list[tuple[float, float]]:
    """Voice-activity detection over the raw waveform: returns silent ranges.

    Frame RMS is thresholded 26 dB under the 90th-percentile (speech) frame
    RMS, so the cut adapts to the recording's mic gain but not to track
    composition. Brief speech dips (gaps in speech < merge_gap) are bridged
    so a word never fragments into fake silences; runs shorter than
    min_silence are dropped as noise.

    This is the ground truth for *silence* cuts. Whisper word timestamps must
    never drive silence boundaries: they drift, and a quiet final utterance
    is often dropped entirely, which made the old trailing-silence rule cut
    real speech off the end of videos.
    """
    with wave.open(wav_path, "rb") as w:
        sr = w.getframerate()
        data = w.readframes(w.getnframes())
    samples = array("h")
    samples.frombytes(data)

    frame = max(1, int(frame_sec * sr))
    nframes = len(samples) // frame
    rms: list[float] = []
    for i in range(nframes):
        chunk = samples[i * frame : (i + 1) * frame]
        acc = sum(s * s for s in chunk)
        rms.append((acc / len(chunk)) ** 0.5)
    if not rms:
        return []

    # Speech-referenced threshold: 26 dB under the 90th-percentile (speech)
    # frame RMS. The noise-referenced rule it replaces (2x the 15th
    # percentile) drifts with track composition: a 30-minute cut's p15 sits
    # deeper in the noise than a 6-minute one, so its threshold collapsed
    # from 98 to 67 and swallowed real inter-word pauses (measured: 178
    # silence ranges on a 6-minute cut of a recording vs 38 on the full 31
    # minutes, identical audio). Speech level is stable for one mic in one
    # room: both cuts of this recording measure 96-98 here.
    ordered = sorted(rms)
    speech_level = ordered[min(len(ordered) - 1, int(len(ordered) * 0.90))]
    threshold = max(speech_level * 10 ** (-26.0 / 20.0), 0.006)

    # Speech runs: frames at or above the threshold.
    speech: list[tuple[int, int]] = []
    cur: list[int] | None = None
    for i, r in enumerate(rms):
        if r >= threshold:
            if cur is None:
                cur = [i, i]
            else:
                cur[1] = i
        elif cur is not None:
            speech.append((cur[0], cur[1]))
            cur = None
    if cur is not None:
        speech.append((cur[0], cur[1]))

    # Bridge short dips inside speech so quiet words stay speech.
    merged: list[tuple[int, int]] = []
    for a, b in speech:
        if merged and a * frame_sec - (merged[-1][1] + 1) * frame_sec < merge_gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))

    # Complement: silence runs, keeping only the meaningful ones.
    silences: list[tuple[float, float]] = []
    cursor = 0.0
    for a, b in merged:
        s0 = a * frame_sec
        s1 = (b + 1) * frame_sec
        if s0 - cursor >= min_silence:
            silences.append((cursor, s0))
        cursor = s1
    total = nframes * frame_sec
    if total - cursor >= min_silence:
        silences.append((cursor, total))
    return silences


def _slice_wav(wav_path: str, start: float, end: float) -> str:
    """Cut [start, end) out of a wav into a temp 16 kHz mono file; caller deletes it."""
    fd = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    fd.close()
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-i", wav_path, "-ar", "16000", "-ac", "1", fd.name,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (OSError, subprocess.SubprocessError):
        Path(fd.name).unlink(missing_ok=True)
        raise ValueError(f"cannot slice {wav_path} at {start:.3f}-{end:.3f}") from None
    return fd.name


def _decode_slice(slice_path: str, backend: str, model: str | None) -> str:
    """One ASR decode of an already-sliced wav; "" when the engine fails."""
    if backend == "parakeet":
        try:
            _, words = transcribe_parakeet(slice_path, slice_path)
        except (OSError, RuntimeError):
            return ""
        return " ".join(w.text for w in words).strip()
    model = model or default_whisper_model()
    engine = _whisper_engine_handle(model)
    try:
        result = _whisper_decode(engine, slice_path, model, prompt=FILLER_PROMPT)
    except (OSError, RuntimeError):
        return ""
    return str(result.get("text") or "").strip()


def probe_filler(wav_path: str, start: float, end: float, model: str | None = None) -> str:
    """Re-transcribe one audio slice (used to classify glued filler chunks).

    Small slices are whisper's weak spot (it hallucinates real words over
    short filler bursts), so the filler-biased prompt is essential here.
    Returns the trimmed text, or "" if the slice cannot be decoded.
    """
    try:
        slice_path = _slice_wav(wav_path, start, end)
    except ValueError:
        return ""
    try:
        return _decode_slice(slice_path, "whisper", model)
    finally:
        Path(slice_path).unlink(missing_ok=True)


def probe_words(wav_path: str, start: float, end: float, backend: str, model: str | None = None) -> str:
    """Re-transcribe ONE narrow slice with one named backend: the focused,
    independent second opinion on an uncertain boundary.

    The caller uses it on voiced, speech-level audio no word covers, where the
    transcript may simply have dropped what was said. Both directions of error
    are handled honestly: a hallucinated word only ever protects audio (the
    guard never turns this text into a cut), and an engine that hears nothing
    is not treated as proof the audio is silent. Returns "" when the slice
    cannot be decoded or the engine fails, so an unavailable backend degrades
    to "no second opinion" rather than to a wrong answer.
    """
    try:
        slice_path = _slice_wav(wav_path, start, end)
    except ValueError:
        return ""
    try:
        return _decode_slice(slice_path, backend, model)
    finally:
        Path(slice_path).unlink(missing_ok=True)


# ------------------------------------------------------- cached interface


TRANSCRIPT_CACHE_VERSION = 1


@dataclass
class Transcript:
    backend: str
    version: str
    duration: float
    words: list[Word]
    # Native sound tags; empty for backends that emit none.
    events: list[AudioEvent] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "version": self.version,
            "duration": self.duration,
            "words": [{"w": w.text, "s": w.start, "e": w.end, "p": w.probability} for w in self.words],
            "events": [{"t": e.text, "s": e.start, "e": e.end} for e in self.events],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Transcript":
        return cls(
            backend=str(d["backend"]),
            version=str(d["version"]),
            duration=float(d["duration"]),
            words=[Word(str(w["w"]), float(w["s"]), float(w["e"]), float(w.get("p", 1.0))) for w in d["words"]],
            events=[AudioEvent(str(e["t"]), float(e["s"]), float(e["e"])) for e in d.get("events", [])],
        )


class Transcriber(Protocol):
    name: str
    version: str

    def transcribe(self, wav_path: str) -> Transcript: ...


class FakeTranscriber:
    replays_fixed_source = False

    def __init__(self, words: list[Word], duration: float, name: str = "fake",
                 events: list[AudioEvent] | None = None) -> None:
        self.words, self.duration, self.name, self.version = words, duration, name, "1"
        self.events = list(events or [])
        self.calls = 0

    def transcribe(self, wav_path: str) -> Transcript:
        self.calls += 1
        return Transcript(self.name, self.version, self.duration, list(self.words), list(self.events))


class ParakeetTranscriber:
    """NVIDIA Parakeet TDT via FluidAudio (macOS CoreML); native word timings."""

    name = "parakeet"
    replays_fixed_source = False

    def __init__(self) -> None:
        # fluidaudiocli reads TDT_EMISSION_DELAY_FRAMES from the environment and
        # it shifts every word stamp, so an override must not replay a transcript
        # cached without it. The default key stays as it was: existing caches,
        # and every judge memo keyed on their words, remain valid.
        delay = os.environ.get("TDT_EMISSION_DELAY_FRAMES")
        self.version = "fluidaudio-tdt" if delay is None else f"fluidaudio-tdt;delay={delay}"

    def transcribe(self, wav_path: str) -> Transcript:
        duration, words = transcribe_parakeet(wav_path, wav_path)
        return Transcript(self.name, self.version, duration, words)


class WhisperTranscriber:
    """Whisper: mlx-whisper on macOS, faster-whisper elsewhere; explicit opt-in only (word spans absorb hesitations)."""

    name = "whisper"
    replays_fixed_source = False

    def __init__(self, model: str | None = None) -> None:
        self.model = model or default_whisper_model()
        self.version = whisper_version(self.model)

    def transcribe(self, wav_path: str) -> Transcript:
        # Module-level transcribe (this platform's whisper engine), not this method.
        duration, words = transcribe(wav_path, model=self.model, wav_path=wav_path)
        return Transcript(self.name, self.version, duration, words)


# ------------------------------------------------------------ Scribe v2
#
# Scribe v2 runs through OpenRouter (elevenlabs/scribe-v2) and nothing else:
# OPENROUTER_API_KEY is the only key it reads. Native word stamps plus sound tags
# (audio events: sighs, laughter). Events are kept as data on the Transcript;
# analysis decides which are cuts and which are protected.
#
# OpenRouter rejects a request body over OPENROUTER_BODY_LIMIT_BYTES with HTTP 413.
# A recording that does not fit one request is cut at quiet points into overlapping
# chunks, sent a few at a time, and merged back onto the recording's clock.
# Saved responses (ElevenLabs JSON, via --scribe-response) still replay through
# SavedScribeTranscriber, which reads that format with parse_scribe_response.

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/audio/transcriptions"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
# The only model the OpenRouter route serves.
OPENROUTER_SCRIBE_MODEL = "elevenlabs/scribe-v2"
# OpenRouter's limit on one request body, in bytes.
OPENROUTER_BODY_LIMIT_BYTES = 52_428_800
# Request bodies stay 4 MiB under that limit. Each body is measured exactly before it is
# sent, so the margin only covers anything the gateway counts differently.
SCRIBE_BODY_BUDGET_BYTES = OPENROUTER_BODY_LIMIT_BYTES - 4 * 1024 * 1024
# Seconds of audio each chunk hears beyond its core, on every side that has a neighbour.
# Words are kept only from the core, so this context never duplicates a word.
SCRIBE_CHUNK_OVERLAP_SEC = 20.0
# A seam is looked for this far either side of the evenly spaced target.
SCRIBE_SEAM_SEARCH_SEC = 120.0
# With no silence in the search window, the seam goes at the quietest block this long.
SCRIBE_QUIET_BLOCK_SEC = 0.05
# Chunk requests in flight at once.
SCRIBE_MAX_PARALLEL = 3
# Bump when the response parser changes what words/events it yields.
SCRIBE_PARSER_VERSION = 1
# The cache key of every live Scribe transcript. Changing it re-keys every cached transcript.
# The route and the chunk count are appended to the stored version, not to this key.
SCRIBE_VERSION = f"scribe_v2;word;events;diarize=false;verbatim;parse={SCRIBE_PARSER_VERSION}"
# Scribe stamps may trail the audio end by a few frames.
SCRIBE_TIME_SLACK = 0.1
SCRIBE_TIMEOUT_SEC = 1200
PCM_RATE = 16000
# Bytes before the PCM payload in the WAV this module writes.
WAV_HEADER_BYTES = 44


def _probability_from_logprob(item: dict) -> float:
    logprob = item.get("logprob")
    return 1.0 if logprob is None else math.exp(min(0.0, float(logprob)))


def _probability_from_confidence(item: dict) -> float:
    # OpenRouter reports a per-word confidence in [0, 1]. Nothing decides on Word.probability;
    # a route that reports no score stores 1.0.
    confidence = item.get("confidence")
    return 1.0 if confidence is None else min(1.0, max(0.0, float(confidence)))


def _parse_words(items, *, duration: float, default_kind: str | None, text_key: str,
                 probability) -> tuple[list[Word], list[AudioEvent]]:
    """Words and audio events on the audio's timeline.

    Spacing items carry no content and are skipped. A stamp outside the audio
    means the response belongs to other audio: refuse rather than cut on it.
    """
    if not isinstance(items, list):
        raise ValueError("Scribe response has no 'words' list")
    words: list[Word] = []
    events: list[AudioEvent] = []
    for i, item in enumerate(items):
        kind = item.get("type", default_kind)
        if kind == "spacing":
            continue
        if kind not in ("word", "audio_event"):
            raise ValueError(f"Scribe response item {i} has unknown type {kind!r}")
        try:
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Scribe response item {i} has no usable timestamps") from None
        if not 0.0 <= start <= end <= duration + SCRIBE_TIME_SLACK:
            raise ValueError(
                f"Scribe {kind} {start:.2f}-{end:.2f}s lies outside the {duration:.2f}s audio; "
                "the response belongs to other audio")
        end = min(end, duration)
        start = min(start, end)
        text = str(item.get(text_key, "")).strip()
        if not text:
            continue
        if kind == "word":
            words.append(Word(text, start, end, probability(item)))
        else:
            events.append(AudioEvent(text, start, end))
    return words, events


def parse_scribe_response(native: dict, *, duration: float) -> tuple[list[Word], list[AudioEvent]]:
    """Native ElevenLabs Scribe v2 JSON -> words and audio events on the audio's timeline."""
    items = native.get("words") if isinstance(native, dict) else None
    return _parse_words(items, duration=duration, default_kind=None, text_key="text",
                        probability=_probability_from_logprob)


def parse_openrouter_response(native: dict, *, duration: float) -> tuple[list[Word], list[AudioEvent]]:
    """OpenRouter STT verbose_json -> the same words and events.

    OpenRouter omits `type` on spoken words, so a missing type means "word" on this route only.
    """
    items = native.get("words") if isinstance(native, dict) else None
    return _parse_words(items, duration=duration, default_kind="word", text_key="word",
                        probability=_probability_from_confidence)


def _wav_frames(path: str | Path) -> tuple[str, int] | None:
    """(sha256 of PCM payload, frame count) when `path` is already 16 kHz mono s16 WAV."""
    try:
        with wave.open(str(path), "rb") as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (PCM_RATE, 1, 2):
                return None
            h, frames = hashlib.sha256(), 0
            while block := w.readframes(1 << 16):
                h.update(block)
                frames += len(block) // 2
            return h.hexdigest(), frames
    except (wave.Error, EOFError):
        return None


def pcm_identity(path: str | Path) -> tuple[str, int]:
    """Identity of the audio as the pipeline hears it: sha256 and frame count of
    its canonical 16 kHz mono s16 PCM payload, ignoring WAV header differences.

    Other formats are converted exactly as `prepare_audio` converts them.
    """
    direct = _wav_frames(path)
    if direct is not None:
        return direct
    with tempfile.TemporaryDirectory() as tmp:
        canon = Path(tmp) / "canon.wav"
        _ffmpeg_16k_mono(str(path), canon)
        found = _wav_frames(canon)
    assert found is not None
    return found


def _ascii_key_or_raise(key: str, env_name: str) -> None:
    # Keys travel in HTTP headers, so anything beyond a bare ASCII token is a configuration error.
    if not key or not key.isascii() or any(ord(c) < 33 or ord(c) == 127 for c in key):
        raise RuntimeError(f"{env_name} must be a single ASCII token without whitespace")


class ScribeRequestError(RuntimeError):
    """A Scribe request failed: HTTP error, network failure or timeout. Never carries the key."""


def _send(request: urllib.request.Request, key: str, label: str) -> bytes:
    """Response body, or ScribeRequestError.

    Provider error bodies are not echoed: they can contain request data.
    """
    try:
        with urllib.request.urlopen(request, timeout=SCRIBE_TIMEOUT_SEC) as response:
            return response.read()
    except urllib.error.HTTPError as e:
        e.close()
        raise ScribeRequestError(f"{label} request failed: HTTP {e.code}") from None
    except urllib.error.URLError as e:
        raise ScribeRequestError(f"{label} request failed: {str(e.reason).replace(key, '[REDACTED]')}") from None
    except TimeoutError:
        raise ScribeRequestError(f"{label} request timed out") from None


def _openrouter_body(wav: bytes) -> bytes:
    return json.dumps({
        "model": OPENROUTER_SCRIBE_MODEL,
        "input_audio": {"data": base64.b64encode(wav).decode("ascii"), "format": "wav"},
        "response_format": "verbose_json",
        "timestamp_granularities": ["word"],
    }).encode()


def _openrouter_request(wav: bytes, key: str) -> bytes:
    """POST one WAV to OpenRouter's Scribe endpoint; the response body."""
    _ascii_key_or_raise(key, OPENROUTER_KEY_ENV)
    request = urllib.request.Request(
        OPENROUTER_ENDPOINT, data=_openrouter_body(wav), method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    return _send(request, key, "OpenRouter Scribe")


def require_scribe_key() -> str:
    """The OpenRouter key Scribe v2 needs. Nothing transcribes locally in its place."""
    key = os.environ.get(OPENROUTER_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{OPENROUTER_KEY_ENV} is not set; Scribe v2 runs through OpenRouter "
            "and nothing transcribes locally in its place.")
    return key


def _pcm_wav(pcm: bytes) -> bytes:
    """A 16 kHz mono s16 WAV around raw PCM frames."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(PCM_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def _max_request_frames() -> int:
    """Most PCM frames one request can carry while its whole JSON body stays within the budget."""
    base64_room = SCRIBE_BODY_BUDGET_BYTES - len(_openrouter_body(b""))
    wav_bytes = (base64_room // 4) * 3  # base64 writes 4 characters per 3 bytes
    return (wav_bytes - WAV_HEADER_BYTES) // 2


def _overlap_frames() -> int:
    return round(SCRIBE_CHUNK_OVERLAP_SEC * PCM_RATE)


def _canonical_pcm_wav(path: str | Path, workdir: Path) -> Path:
    """`path` itself when it is 16 kHz mono s16 PCM, else its conversion under `workdir`."""
    try:
        with wave.open(str(path), "rb") as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (PCM_RATE, 1, 2):
                return Path(path)
    except (wave.Error, EOFError):
        pass
    canon = workdir / "canon.wav"
    _ffmpeg_16k_mono(str(path), canon)
    return canon


def _read_frames(path: Path, start: int, end: int) -> bytes:
    with wave.open(str(path), "rb") as w:
        w.setpos(start)
        return w.readframes(end - start)


def _read_pcm(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


@dataclass(frozen=True)
class ScribeChunk:
    """One Scribe request and the core it owns, in PCM frames of the canonical recording.

    The request is the core plus the overlap on each side that the recording has.
    """

    request_start: int
    request_end: int
    core_start: int
    core_end: int


def plan_scribe_chunks(n_frames: int, *, max_request_frames: int, overlap_frames: int,
                       choose_seam: Callable[[int, int, int], int]) -> list[ScribeChunk]:
    """Split `n_frames` into the fewest chunks whose requests each fit `max_request_frames`.

    Cores are contiguous and cover the recording. The first and last cores carry one
    overlap, the middle ones two. `choose_seam(lo, hi, target)` returns the seam between
    two cores anywhere in [lo, hi]; the bounds keep every remaining core within the limit.
    """
    if n_frames <= max_request_frames:
        return [ScribeChunk(0, n_frames, 0, n_frames)]
    edge = max_request_frames - overlap_frames  # first or last core: overlap on one side
    inner = max_request_frames - 2 * overlap_frames  # middle core: overlap on both sides
    if inner <= 0:
        raise ValueError("the request budget leaves no room beyond the chunk overlap")

    def cover(chunks: int) -> int:
        """Most frames that `chunks` consecutive cores can own, the last one an edge core."""
        return (chunks - 1) * inner + edge

    count = 2
    while edge + cover(count - 1) < n_frames:
        count += 1
    seams: list[int] = []
    start = 0
    for j in range(count - 1):
        cap = edge if j == 0 else inner
        lo = max(start + 1, n_frames - cover(count - 1 - j))
        hi = min(start + cap, n_frames - 1)
        seam = choose_seam(lo, hi, n_frames * (j + 1) // count)
        if not lo <= seam <= hi:
            raise ValueError(f"seam {seam} lies outside [{lo}, {hi}]")
        seams.append(seam)
        start = seam
    bounds = list(zip([0, *seams], [*seams, n_frames]))
    return [ScribeChunk(max(0, a - overlap_frames), min(n_frames, b + overlap_frames), a, b)
            for a, b in bounds]


class _SeamFinder:
    """Seams in the longest silence of the search window, else at its quietest block.

    The silences and the waveform are read on the first call, which only happens when a
    recording needs chunks.
    """

    def __init__(self, wav: Path) -> None:
        self._wav = wav
        self._silences: list[tuple[float, float]] | None = None
        self._pcm: np.ndarray | None = None

    def __call__(self, lo: int, hi: int, target: int) -> int:
        if self._silences is None or self._pcm is None:
            self._silences = detect_silences(str(self._wav))
            self._pcm = _read_pcm(self._wav)
        half = round(SCRIBE_SEAM_SEARCH_SEC * PCM_RATE)
        w_lo, w_hi = max(lo, target - half), min(hi, target + half)
        if w_lo > w_hi:
            w_lo, w_hi = lo, hi
        best: tuple[tuple[int, int], int] | None = None
        for s0, s1 in self._silences:
            a = max(w_lo, round(s0 * PCM_RATE))
            b = min(w_hi, round(s1 * PCM_RATE))
            if b > a:
                mid = (a + b) // 2
                rank = (b - a, -abs(mid - target))  # longest clipped silence, then nearest the target
                if best is None or rank > best[0]:
                    best = (rank, mid)
        if best is not None:
            return best[1]
        return self._quietest(w_lo, w_hi, target)

    def _quietest(self, lo: int, hi: int, target: int) -> int:
        block = round(SCRIBE_QUIET_BLOCK_SEC * PCM_RATE)
        count = (hi - lo) // block
        if count == 0:
            return (lo + hi) // 2
        seg = self._pcm[lo: lo + count * block].astype(np.float64).reshape(count, block)
        rms = np.sqrt((seg * seg).mean(axis=1))
        centres = lo + np.arange(count) * block + block // 2
        i = min(range(count), key=lambda k: (rms[k], abs(int(centres[k]) - target)))
        return int(centres[i])


def _owned(start: float, end: float, lo: float, hi: float, last: bool) -> bool:
    """Whether a stamp belongs to the core [lo, hi), judged by its midpoint.

    The last core also owns the audio's final instant.
    """
    mid = (start + end) / 2
    return lo <= mid and (mid < hi or last)


def _merge_chunks(parts: list[tuple[ScribeChunk, list[Word], list[AudioEvent]]],
                  frames: int) -> tuple[list[Word], list[AudioEvent]]:
    """One timeline from the chunks' answers.

    Each chunk's stamps move onto the recording's clock. A word or event is kept only from
    the chunk whose core holds its midpoint, so a word heard in two overlaps is kept once.
    """
    words: list[Word] = []
    events: list[AudioEvent] = []
    for chunk, got_words, got_events in parts:
        shift = chunk.request_start / PCM_RATE
        lo, hi = chunk.core_start / PCM_RATE, chunk.core_end / PCM_RATE
        last = chunk.core_end == frames
        for w in got_words:
            s, e = w.start + shift, w.end + shift
            if _owned(s, e, lo, hi, last):
                words.append(Word(w.text, s, e, w.probability))
        for ev in got_events:
            s, e = ev.start + shift, ev.end + shift
            if _owned(s, e, lo, hi, last):
                events.append(AudioEvent(ev.text, s, e))
    return words, events


def _scribe_chunk(wav: Path, chunk: ScribeChunk, key: str, index: int,
                  total: int) -> tuple[list[Word], list[AudioEvent]]:
    """One request's words and events, on the chunk's own clock (zero at its request start)."""
    pcm = _read_frames(wav, chunk.request_start, chunk.request_end)
    seconds = (chunk.request_end - chunk.request_start) / PCM_RATE
    label = f" (chunk {index + 1} of {total})" if total > 1 else ""
    try:
        native = json.loads(_openrouter_request(_pcm_wav(pcm), key))
        return parse_openrouter_response(native, duration=seconds)
    except ScribeRequestError as e:
        raise ScribeRequestError(f"{e}{label}") from None
    except ValueError as e:
        raise ValueError(f"{e}{label}") from None


_ChunkAnswer = tuple[ScribeChunk, list[Word], list[AudioEvent]]


def _run_chunks(wav: Path, chunks: list[ScribeChunk], key: str) -> list[_ChunkAnswer]:
    """Every chunk's answer, in chunk order. The first failure fails the whole transcription."""
    pool = ThreadPoolExecutor(max_workers=min(SCRIBE_MAX_PARALLEL, len(chunks)))
    try:
        futures = [pool.submit(_scribe_chunk, wav, c, key, i, len(chunks)) for i, c in enumerate(chunks)]
        return [(c, *f.result()) for c, f in zip(chunks, futures)]
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


class ScribeTranscriber:
    """Scribe v2 through OpenRouter, the only route; see the section header.

    The stored version records the route, and the chunk count for a chunked recording.
    Keys are never logged.
    """

    name = "scribe"
    version = SCRIBE_VERSION
    replays_fixed_source = False

    def transcribe(self, wav_path: str) -> Transcript:
        key = require_scribe_key()
        _ascii_key_or_raise(key, OPENROUTER_KEY_ENV)
        with tempfile.TemporaryDirectory(prefix="scribe-") as tmp:
            wav = _canonical_pcm_wav(wav_path, Path(tmp))
            with wave.open(str(wav), "rb") as w:
                frames = w.getnframes()
            chunks = plan_scribe_chunks(frames, max_request_frames=_max_request_frames(),
                                        overlap_frames=_overlap_frames(), choose_seam=_SeamFinder(wav))
            parts = _run_chunks(wav, chunks, key)
        words, events = _merge_chunks(parts, frames)
        version = f"{self.version};route=openrouter"
        if len(chunks) > 1:
            version += f";chunks={len(chunks)}"
        return Transcript(self.name, version, frames / PCM_RATE, words, events)


def _response_sha256(native: dict) -> str:
    return hashlib.sha256(json.dumps(native, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class SavedScribeTranscriber:
    """Replays a real saved Scribe v2 response for the exact audio that produced it.

    `source_audio` is that audio in any WAV (or ffmpeg-readable) form; the pair is
    checked at construction and the audio handed to `transcribe` must have the
    same canonical 16 kHz mono PCM. The version carries both identities, so a
    cache never replays one response for another recording or response. It
    replays a FIXED source, so it is not independent ASR of any other audio.
    """

    name = "scribe"
    replays_fixed_source = True

    def __init__(self, response: str | Path, source_audio: str | Path) -> None:
        native = json.loads(Path(response).read_text())
        self.audio_pcm_sha256, frames = pcm_identity(source_audio)
        self._duration = frames / PCM_RATE
        self._words, self._events = parse_scribe_response(native, duration=self._duration)
        secs = native.get("audio_duration_secs")
        if secs is not None and abs(float(secs) - self._duration) > 0.05:
            raise ValueError(
                f"saved Scribe response covers {float(secs):.3f}s but the source audio is "
                f"{self._duration:.3f}s")
        self.response_sha256 = _response_sha256(native)
        self.version = (f"{SCRIBE_VERSION};saved;response={self.response_sha256[:16]};"
                        f"audio={self.audio_pcm_sha256[:16]}")

    def transcribe(self, wav_path: str) -> Transcript:
        sha, _ = pcm_identity(wav_path)
        if sha != self.audio_pcm_sha256:
            raise ValueError(
                "audio does not match the source the saved Scribe response was made from "
                "(canonical 16 kHz mono PCM differs)")
        return Transcript(self.name, self.version, self._duration, list(self._words), list(self._events))


def transcribe_cached(cache: Cache, wav_sha: str, transcriber: Transcriber, wav_path: str) -> Transcript:
    params = {"backend": transcriber.name, "version": transcriber.version}

    def compute() -> dict:
        got = transcriber.transcribe(wav_path)
        # Keep the route the transcriber recorded (ScribeTranscriber appends ";route=...").
        # The cache key above still uses the static version, so one audio hash maps to one entry.
        return replace(got, backend=transcriber.name).to_dict()

    return Transcript.from_dict(
        cache.get_or_compute(wav_sha, "transcript", params, TRANSCRIPT_CACHE_VERSION, compute)
    )


# ------------------------------------------------------------ input checks


class AVMismatch(RuntimeError):
    """Audio and video tracks disagree on length; the timeline is ambiguous."""


def av_durations(video: str) -> tuple[float, float]:
    """(video seconds, audio seconds) of the first stream of each kind."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration:format=duration",
         "-of", "json", video],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)
    fmt = float(info.get("format", {}).get("duration") or 0.0)
    found = {}
    for s in info.get("streams", []):
        kind = s.get("codec_type")
        if kind in ("video", "audio") and kind not in found:
            found[kind] = float(s.get("duration") or fmt)
    if "video" not in found or "audio" not in found:
        raise RuntimeError(f"{video}: need one video and one audio stream, found {sorted(found)}")
    return found["video"], found["audio"]


def check_av_mismatch(video: str, tolerance: float = 1.0, allow: bool = False) -> tuple[float, float]:
    v, a = av_durations(video)
    if abs(v - a) > tolerance and not allow:
        raise AVMismatch(
            f"{video}: video track {v:.2f}s vs audio track {a:.2f}s differ by {abs(v - a):.2f}s "
            f"(> {tolerance:.1f}s). Fix the master or pass --allow-av-mismatch."
        )
    return v, a


def _ffmpeg_16k_mono(src: str, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", src, "-vn", "-ar", str(PCM_RATE), "-ac", "1",
         "-c:a", "pcm_s16le", "-f", "wav", "-fflags", "+bitexact", "-flags:a", "+bitexact", str(dst)],
        check=True, capture_output=True, text=True,
    )


def audio_duration(path: str) -> float:
    """Seconds of the first audio stream (container duration when the stream has none)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=duration:format=duration", "-of", "json", path],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)
    if not info.get("streams"):
        raise RuntimeError(f"{path}: no audio stream")
    raw = info["streams"][0].get("duration") or info.get("format", {}).get("duration")
    if raw is None:
        raise RuntimeError(f"{path}: audio duration unknown")
    return float(raw)


def prepare_audio(video: str, workdir: str | Path, *, force: bool = False) -> tuple[str, str]:
    """Extract 16 kHz mono PCM once; return (wav path, sha256 of the wav).

    `force` re-extracts even when a wav is on disk: collect_facts measures
    `final.mp4`, so a stale workdir wav is never trusted by existence. The
    transcript cache still keys on the fresh wav's hash.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    wav = workdir / "audio-16k.wav"
    if force or not wav.is_file():
        tmp = wav.with_suffix(".tmp.wav")
        _ffmpeg_16k_mono(video, tmp)
        tmp.replace(wav)
    return str(wav), sha256_file(wav)
