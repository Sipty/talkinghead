"""Whisper engine choice: mlx-whisper on macOS, faster-whisper elsewhere.

The macOS path is the owner's cache identity: its engine arguments and its
transcript version string must stay exactly what they were. The other engine
is exercised through a fake, so these tests never download a model.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from talkinghead import transcribe as tr


class FakeMlx:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def transcribe(self, audio: str, **kwargs) -> dict:
        self.calls.append((audio, kwargs))
        return {"text": " hello there", "segments": [{"text": " hello there", "words": [
            {"word": " hello", "start": 0.0, "end": 0.5, "probability": 0.9},
            {"word": " there", "start": 0.5, "end": 1.0, "probability": 0.8},
        ]}]}


class FakeFaster:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool, str | None]] = []

    def transcribe(self, audio: str, *, word_timestamps: bool = False, initial_prompt: str | None = None):
        self.calls.append((audio, word_timestamps, initial_prompt))
        words = [SimpleNamespace(word=" hello", start=0.0, end=0.5, probability=0.9),
                 SimpleNamespace(word=" there", start=0.5, end=1.0, probability=0.8)]
        segment = SimpleNamespace(text=" hello there", words=words if word_timestamps else None)
        return iter([segment]), SimpleNamespace(language="en")


def _platform(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setattr("sys.platform", name)


def test_macos_keeps_mlx_whisper_and_its_cache_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    _platform(monkeypatch, "darwin")
    assert tr.whisper_engine() == "mlx"
    assert tr.default_whisper_model() == "mlx-community/whisper-large-v3-turbo"
    transcriber = tr.WhisperTranscriber()
    assert (transcriber.name, transcriber.version) == ("whisper", "mlx-community/whisper-large-v3-turbo")


def test_elsewhere_uses_faster_whisper_with_a_version_that_never_shares_a_cache_entry(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _platform(monkeypatch, "linux")
    assert tr.whisper_engine() == "faster"
    assert tr.default_whisper_model() == "large-v3-turbo"
    transcriber = tr.WhisperTranscriber()
    assert transcriber.version == "faster-whisper;large-v3-turbo"
    assert transcriber.version != tr.DEFAULT_MODEL


def test_macos_decode_calls_mlx_exactly_as_before(monkeypatch: pytest.MonkeyPatch) -> None:
    _platform(monkeypatch, "darwin")
    fake = FakeMlx()
    monkeypatch.setattr(tr, "_mlx_whisper", lambda: fake)
    monkeypatch.setattr(tr, "_probe_duration", lambda path: 2.0)
    duration, words = tr.transcribe("in.mov", wav_path="in.wav")
    assert fake.calls == [("in.wav", {"path_or_hf_repo": tr.DEFAULT_MODEL, "word_timestamps": True})]
    assert duration == 2.0
    assert words == [tr.Word(" hello", 0.0, 0.5, 0.9), tr.Word(" there", 0.5, 1.0, 0.8)]


def test_elsewhere_decode_maps_faster_words_to_the_same_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    _platform(monkeypatch, "linux")
    fake = FakeFaster()
    monkeypatch.setattr(tr, "_faster_whisper", lambda model: fake)
    monkeypatch.setattr(tr, "_probe_duration", lambda path: 2.0)
    duration, words = tr.transcribe("in.mov", wav_path="in.wav")
    assert fake.calls == [("in.wav", True, None)]
    assert duration == 2.0
    assert words == [tr.Word(" hello", 0.0, 0.5, 0.9), tr.Word(" there", 0.5, 1.0, 0.8)]


def test_filler_probe_gets_the_filler_prompt_on_the_faster_engine(
        monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _platform(monkeypatch, "linux")
    fake = FakeFaster()
    slice_path = str(tmp_path / "slice.wav")
    monkeypatch.setattr(tr, "_faster_whisper", lambda model: fake)
    monkeypatch.setattr(tr, "_slice_wav", lambda wav, start, end: slice_path)
    assert tr.probe_filler("in.wav", 1.0, 2.0) == "hello there"
    assert fake.calls == [(slice_path, False, tr.FILLER_PROMPT)]


def test_missing_faster_whisper_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _platform(monkeypatch, "linux")
    monkeypatch.setattr(tr, "_FASTER_MODELS", {})
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # makes the import raise ImportError
    with pytest.raises(RuntimeError, match="needs faster-whisper"):
        tr._faster_whisper("large-v3-turbo")
