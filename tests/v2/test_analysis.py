"""Analysis: everything the stages need, computed once from the source and
cached under its audio hash; word bounds refined; detectors fed the shared
envelope so they run in seconds."""

import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from talkinghead.analysis import analyze
from talkinghead.config import Config
from talkinghead.judge import FakeJudge
from talkinghead.store import Cache
from talkinghead.transcribe import AudioEvent, FakeTranscriber, Word


def _speech_video(path: Path) -> None:
    """3 s clip: 440 Hz 'word' 0.5-1.0 and 1.6-2.2 over near-silence."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc=size=160x120:rate=15:duration=3",
         "-f", "lavfi", "-i",
         r"aevalsrc='0.3*sin(440*2*PI*t)*(between(t\,0.5\,1.0)+between(t\,1.6\,2.2))+0.0005*random(0)':s=16000:d=3",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-b:a", "192k", str(path)],
        check=True,
    )


def test_analyze_builds_a_context_with_refined_words_and_caches_the_transcript(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _speech_video(video)
    # ASR stamps deliberately early/late; refinement must snap them to the tones.
    backend = FakeTranscriber([Word("one.", 0.56, 0.90, 1.0), Word("two.", 1.66, 2.05, 1.0)], duration=3.0)
    cache = Cache(tmp_path / "cache")
    ctx = analyze(str(video), tmp_path / "run", Config(), backend, FakeJudge([]), cache=cache, guidelines="G")
    assert backend.calls == 1
    assert [w.text for w in ctx.words] == ["one.", "two."]
    assert ctx.words[0].start == pytest.approx(0.5, abs=0.03) and ctx.words[0].end == pytest.approx(1.0, abs=0.06)
    assert ctx.words[1].end == pytest.approx(2.2, abs=0.06)
    assert ctx.floor < -40.0
    assert any(a <= 1.05 and b >= 1.55 for a, b in ctx.silences)   # the gap between the tones
    assert ctx.duration == pytest.approx(3.0, abs=0.1)
    assert ctx.provenance["transcript"]["backend"] == "fake"
    assert ctx.provenance["audio_sha256"]
    # Second analysis of the same file: transcript and silences come from the cache.
    ctx2 = analyze(str(video), tmp_path / "run2", Config(), backend, FakeJudge([]), cache=cache, guidelines="G")
    assert backend.calls == 1
    assert [(w.start, w.end) for w in ctx2.words] == [(w.start, w.end) for w in ctx.words]


def test_detector_caches_are_keyed_by_the_words_they_saw(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _speech_video(video)
    cache = Cache(tmp_path / "cache")
    a = FakeTranscriber([Word("one.", 0.56, 0.90, 1.0)], duration=3.0)
    b = FakeTranscriber([Word("one.", 0.56, 0.90, 1.0), Word("two.", 1.66, 2.05, 1.0)], duration=3.0, name="b")
    analyze(str(video), tmp_path / "r1", Config(), a, FakeJudge([]), cache=cache, guidelines="G")
    analyze(str(video), tmp_path / "r2", Config(), b, FakeJudge([]), cache=cache, guidelines="G")
    entries = list((tmp_path / "cache").glob("*/moans-*.json"))
    assert len(entries) == 2


# ------------------------------------------------------------ audio events

SR = 16000


def _events_wav(path: Path) -> None:
    """6 s audio: words 0.5-1.0 and 4.0-4.5; a 2.0-3.0 broadband burst (the tagged
    event) whose quieter shoulders run 1.8-2.0 and 3.0-3.3, over near-silence."""
    rng = np.random.default_rng(7)
    t = np.arange(6 * SR) / SR
    x = 0.0005 * rng.standard_normal(t.size)
    for a, b in ((0.5, 1.0), (4.0, 4.5)):
        m = (t >= a) & (t < b)
        x[m] += 0.3 * np.sin(2 * np.pi * 440 * t[m])
    for a, b, amp in ((1.8, 2.0, 0.02), (2.0, 3.0, 0.2), (3.0, 3.3, 0.02)):
        m = (t >= a) & (t < b)
        x[m] += amp * rng.standard_normal(m.sum())
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def _audio_cook(tmp_path: Path, tag: str, start: float = 2.0, end: float = 3.0, cache: Cache | None = None,
                name: str = "fake"):
    wav = tmp_path / "talk.wav"
    if not wav.exists():
        _events_wav(wav)
    backend = FakeTranscriber([Word("one.", 0.5, 1.0, 1.0), Word("two.", 4.0, 4.5, 1.0)], duration=6.0,
                              name=name, events=[AudioEvent(tag, start, end)])
    return analyze(str(wav), tmp_path / "run", Config(), backend, FakeJudge([]),
                   cache=cache or Cache(tmp_path / "cache"), guidelines="G", audio_only=True)


@pytest.mark.parametrize("tag", ["[laughs]", "[chuckles]", "[giggles]", "[applause]"])
def test_laughter_and_unrecognised_events_are_protected_whole_with_their_quiet_edges(tmp_path: Path, tag: str) -> None:
    ctx = _audio_cook(tmp_path, tag)
    assert len(ctx.protected_spans) == 1
    s, e = ctx.protected_spans[0]
    assert s <= 1.85 and e >= 3.25                      # grew over the quiet onset and decay
    assert s >= 1.0 and e <= 4.0                        # never into a neighbouring word
    assert not any(c.start < e and s < c.end for c in ctx.noise_cuts)   # never offered as a cut
    assert not any(m[0] < e and s < m[1] for m in ctx.moans)


@pytest.mark.parametrize("tag", ["[sighs]", "[breathes]", "[lip smack]"])
def test_known_noise_events_are_cut_candidates_not_protected(tmp_path: Path, tag: str) -> None:
    ctx = _audio_cook(tmp_path, tag)
    assert ctx.protected_spans == []
    sound = [c for c in ctx.noise_cuts if c.text == tag]
    assert [(c.start, c.end) for c in sound] == [(2.0, 3.0)]


def test_a_noise_tag_is_never_cut_through_a_spoken_word(tmp_path: Path) -> None:
    ctx = _audio_cook(tmp_path, "[sighs]", start=0.4, end=1.2)
    assert ctx.noise_cuts
    for cut in ctx.noise_cuts:
        assert not any(cut.start < w.end and w.start < cut.end for w in ctx.words)


def test_event_policy_rekeys_the_transcript_identity_only_when_events_exist(tmp_path: Path) -> None:
    laugh = _audio_cook(tmp_path, "[laughs]", name="a")
    sigh = _audio_cook(tmp_path, "[sighs]", name="b")
    assert laugh.provenance["transcript"]["events"] != sigh.provenance["transcript"]["events"]
    plain = analyze(str(tmp_path / "talk.wav"), tmp_path / "plain", Config(),
                    FakeTranscriber([Word("one.", 0.5, 1.0, 1.0)], duration=6.0, name="plain"),
                    FakeJudge([]), cache=Cache(tmp_path / "cache"), guidelines="G", audio_only=True)
    assert "events" not in plain.provenance["transcript"]


def test_audio_only_analysis_has_no_video_track_to_check(tmp_path: Path) -> None:
    ctx = _audio_cook(tmp_path, "[laughs]")
    assert ctx.provenance["audio_only"] is True and ctx.provenance["video_duration"] is None
    assert ctx.source.path == str((tmp_path / "talk.wav").resolve())
    assert ctx.duration == pytest.approx(6.0, abs=0.05)


def test_a_tagged_event_survives_the_transcript_cache_into_the_next_analysis(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "cache")
    first = _audio_cook(tmp_path, "[laughs]", cache=cache)
    again = _audio_cook(tmp_path, "[chuckles]", cache=cache)   # same backend identity: cached transcript wins
    assert again.protected_spans == first.protected_spans != []
    assert again.provenance["transcript"]["events"] == first.provenance["transcript"]["events"]


def _write_wav(path: Path, x: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def _bounded_cook(tmp_path: Path, bursts: list[tuple[float, float, float]], stamp: tuple[float, float],
                  tag: str = "[laughs]", digital: bool = False, total: float = 8.0):
    """`total` s over room tone (0.0005 std) with white-noise bursts (start, end, amplitude) on top.
    With `digital`, the stamp is zeroed except for the bursts; keep it under a quarter of `total`,
    or the noise floor reads digital silence and room tone counts as audible. Words sit at 0.2-0.7
    and in the last 1.5 s."""
    rng = np.random.default_rng(11)
    t = np.arange(int(total * SR)) / SR
    x = 0.0005 * rng.standard_normal(t.size)
    if digital:
        x[(t >= stamp[0]) & (t < stamp[1])] = 0.0
    for a, b, amp in bursts:
        m = (t >= a) & (t < b)
        x[m] += amp * rng.standard_normal(m.sum())
    wav = tmp_path / "talk.wav"
    _write_wav(wav, x)
    backend = FakeTranscriber([Word("one.", 0.2, 0.7, 1.0), Word("two.", total - 1.5, total - 1.0, 1.0)],
                              duration=total, name="bounded", events=[AudioEvent(tag, *stamp)])
    return analyze(str(wav), tmp_path / "run", Config(), backend, FakeJudge([]),
                   cache=Cache(tmp_path / "cache"), guidelines="G", audio_only=True)


def test_a_long_laugh_over_digital_silence_protects_only_the_audible_parts(tmp_path: Path) -> None:
    ctx = _bounded_cook(tmp_path, [(2.0, 2.5, 0.2), (4.0, 4.4, 0.2)], stamp=(1.0, 6.0), digital=True, total=24.0)
    assert len(ctx.protected_spans) == 2                  # one span per audible laugh, not the 5 s stamp
    (s1, e1), (s2, e2) = ctx.protected_spans
    assert 1.9 <= s1 <= 2.05 and 2.45 <= e1 <= 2.7
    assert 3.95 <= s2 <= 4.05 and 4.35 <= e2 <= 4.6
    assert not any(s < 3.8 and 2.8 < e for s, e in ctx.protected_spans)   # the dead air between stays cuttable
    assert ctx.provenance["audio_events"][0]["protected"] == [[round(s1, 3), round(e1, 3)],
                                                             [round(s2, 3), round(e2, 3)]]


def test_a_real_laugh_keeps_its_soft_tail(tmp_path: Path) -> None:
    ctx = _bounded_cook(tmp_path, [(2.0, 2.5, 0.2), (2.5, 3.5, 0.02)], stamp=(1.5, 4.0))
    assert len(ctx.protected_spans) == 1
    s, e = ctx.protected_spans[0]
    assert s <= 2.05                                      # the loud core is kept whole
    assert 3.45 <= e <= 3.75                              # the soft decay is kept; the silence after it is not
