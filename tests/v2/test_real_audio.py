"""Real-audio tests over short clips of the owner's own recording.

Fixtures: tests/fixtures/*.wav (16 kHz mono 16-bit PCM, 6-20 s each) plus
tests/fixtures/manifest.json, which carries, per clip, the Parakeet words
inside it and the ground-truth events from tests/fixtures/candidates.json,
all REBASED to the clip (clip_t = source_t - source_start). The clips may be
stripped before publishing: every test skips when its wav is missing.

Floors. `talkinghead.audio.noise_floor` is the 25th percentile of the 5 ms
envelope; over the 31-minute source that is -54.28 dBFS. Over a 6-20 s clip
the same percentile is not a noise floor (measured -43.5 dBFS on the
speech-dense word-boundary-216.0 clip, -68.8 dBFS on the dead-air-heavy
outro clip), so tests that snap word bounds pass the SOURCE floor from the
manifest and assert the clip-local deviation separately. `detect_silences`
thresholds 26 dB under the 90th-percentile (speech) frame RMS, so clip and
full-file runs agree (the old noise-referenced rule made clip-local silence
geometry disagree with the full file; see the voiced-gap tests).

`refine_word_bounds` (burst-attach contract): the envelope is cut into voiced
bursts (frames above floor+6 dB, quiet under BURST_SPLIT_QUIET_SEC bridged,
runs under BURST_MIN_SEC dropped), each word attaches to the first burst
overlapping its stamp by >= BURST_MIN_SEC (else the nearest within
ATTACH_REACH_SEC), the outer words of a burst take the burst's edges and the
inner words keep their stamps. Confidence: 0.0 unattached, 0.5 inner word,
0.75 one acoustic edge, 1.0 both.

Assertions the data contradicts are marked xfail(strict=True) with the
measured values; see the reasons on each.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from talkinghead.audio import (
    BURST_MIN_SEC,
    BURST_SPLIT_QUIET_SEC,
    FRAME_SEC,
    Envelope,
    envelope,
    noise_floor,
    refine_word_bounds,
    voiced_bursts,
)
from talkinghead.bursts import BURST_MIN_DUR, detect_voiced_bursts
from talkinghead.candidates import (
    Candidate,
    cold_open_candidates,
    machinery_candidates,
    sentence_table,
)
from talkinghead.config import Config
from talkinghead.noise import _read_wav
from talkinghead.transcribe import Word, detect_silences

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
MANIFEST_PATH = FIXTURES / "manifest.json"
MANIFEST: dict = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {"clips": []}
CLIPS: dict[str, dict] = {c["file"]: c for c in MANIFEST["clips"]}
CLIP_NAMES = sorted(CLIPS)
SOURCE_FLOOR: float = MANIFEST.get("source_floor_5ms_dbfs", -54.276)

# The 10 ms RMS analysis in candidates.json used floor+6 dB held >= 40 ms; the
# 5 ms envelope here has one-frame jitter against it, so 0.15 s is generous.
END_TOL = 0.15


@dataclass
class Clip:
    name: str
    meta: dict
    raw: object            # array('h'): what the desilence detectors accept
    samples: np.ndarray
    sr: int
    duration: float
    words: list[Word]
    env: Envelope
    clip_floor: float
    silences: list[tuple[float, float]]

    def event(self, category: str, source_start: float) -> dict:
        for ev in self.meta["events"]:
            if ev["category"] == category and abs(ev["source_start"] - source_start) < 1e-6:
                return ev
        raise KeyError((category, source_start))


@lru_cache(maxsize=None)
def load(name: str) -> Clip:
    path = FIXTURES / name
    if name not in CLIPS or not path.exists():
        pytest.skip(f"real-audio fixture {name} not present")
    raw, sr = _read_wav(str(path))
    samples = np.frombuffer(raw.tobytes(), dtype=np.int16)
    meta = CLIPS[name]
    words = [Word(w["w"], w["s"], w["e"], w["p"]) for w in meta["words"]]
    env = envelope(samples, sr, FRAME_SEC)
    return Clip(
        name, meta, raw, samples, sr, len(samples) / sr, words, env,
        noise_floor(env.db), detect_silences(str(path)),
    )


def refined(clip: Clip, floor: float | None = None):
    return refine_word_bounds(clip.words, clip.env, SOURCE_FLOOR if floor is None else floor)


def refined_word(clip: Clip, text: str, asr_start: float, floor: float | None = None):
    return next(r for r in refined(clip, floor) if r.text == text and abs(r.asr_start - asr_start) < 1e-6)


def candidates(clip: Clip) -> list[Candidate]:
    fixed = replace(Config(), pause_keep=0.20, pause_keep_ratio=0.0)
    return machinery_candidates(clip.duration, clip.words, clip.silences, pause_above=0.35, pause=fixed)


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


# 1. Folded "Hello." --------------------------------------------------------

def test_folded_hello_end_snaps_to_acoustic_end_not_the_parakeet_stamp() -> None:
    """'Hello.' is stamped 15.12-20.80; the voice is 15.46-15.79 followed by
    quiet (-48..-64 dBFS) and a faint second 'Hello' at 16.12-16.42. With the
    source floor the word attaches to the burst 15.46-15.815 (measured), so
    both edges come from the audio and the 5.0 s of folded silence is gone."""
    clip = load("word-boundary-13.0.wav")
    d = clip.event("word_boundaries", 15.12)["detector_output"]
    r = refined_word(clip, "Hello.", d["asr_start"])
    assert r.end < d["asr_end"] - 4.0, "must not keep the 20.80 stamp"
    assert abs(r.end - d["acoustic_end"]) <= END_TOL
    assert abs(r.start - d["acoustic_onset"]) <= END_TOL
    assert r.confidence == 1.0


def test_folded_hello_with_the_clip_local_floor_bridges_the_room_tone() -> None:
    """Why the source floor matters here too: this clip is 60% pause, so its
    25th percentile is -57.44 dBFS and the +6 dB voiced threshold (-51.4) sits
    inside the room tone; one burst then runs 13.40-18.115 and 'Hello.' comes
    back as 13.400-18.115 (measured), 2.3 s past the acoustic end."""
    clip = load("word-boundary-13.0.wav")
    assert clip.clip_floor < SOURCE_FLOOR - 2.5
    d = clip.event("word_boundaries", 15.12)["detector_output"]
    r = refined_word(clip, "Hello.", d["asr_start"], floor=clip.clip_floor)
    assert r.end > d["acoustic_end"] + 2.0
    assert r.start < d["acoustic_onset"] - 1.5


# 2. Early end stamps: "tired" and "for" -------------------------------------

def test_for_refined_end_extends_to_the_acoustic_end() -> None:
    """'for' 219.60-219.76 rings on to 220.42 (+0.66 s) before 'multigentic'
    at 220.56. 'formula' and 'for' share the burst 218.615-220.42, so 'for'
    keeps its start stamp and takes the burst end: measured 219.600-220.420."""
    clip = load("word-boundary-216.0.wav")
    d = clip.event("word_boundaries", 219.6)["detector_output"]
    r = refined_word(clip, "for", d["asr_start"])
    assert r.end - r.asr_end >= 0.3
    assert abs(r.end - d["acoustic_end"]) <= END_TOL
    assert r.start == r.asr_start
    assert r.confidence == 0.75


def test_clip_local_floor_on_a_speech_dense_clip_is_speech_not_noise() -> None:
    """Why the boundary tests pass the source floor: on the 8 s 'for' clip the
    25th percentile of the envelope is -43.49 dBFS (source -54.28), the voiced
    threshold sits at -37.5 dBFS, the ring-out of 'for' splits off into its own
    faint burst and the word ends at 219.865 (measured), 0.55 s early."""
    clip = load("word-boundary-216.0.wav")
    assert clip.clip_floor > SOURCE_FLOOR + 8.0
    d = clip.event("word_boundaries", 219.6)["detector_output"]
    r = refined_word(clip, "for", d["asr_start"], floor=clip.clip_floor)
    assert r.end < d["acoustic_end"] - 0.4


@pytest.mark.xfail(
    strict=True,
    reason=(
        "'tired' 105.20-105.36 rings on to 106.055 and the next word 'of' (stamped 106.08) has its "
        "voice from 106.20: the quiet between them is 0.145 s, under BURST_SPLIT_QUIET_SEC = 0.15, "
        "so 'tired' and 'of' share one voiced burst (105.06-109.935 with the source floor) and as "
        "an inner word 'tired' keeps its stamps: measured 105.200-105.360, confidence 0.5, with "
        "both the source floor and the clip floor (-56.03). Expected an end within 0.15 s of 106.04."
    ),
)
def test_tired_refined_end_extends_to_the_acoustic_end() -> None:
    clip = load("cold-open-100.0.wav")
    d = clip.event("word_boundaries", 105.2)["detector_output"]
    r = refined_word(clip, "tired", d["asr_start"])
    assert r.end - r.asr_end >= 0.3
    assert abs(r.end - d["acoustic_end"]) <= END_TOL


def test_tired_and_of_share_one_burst_because_the_quiet_between_them_is_short() -> None:
    """The measurement behind the xfail above: with the source floor there is
    no burst edge between the 'tired' stamp and the 'of' onset, and the longest
    contiguous quiet run from the ring-out to the 'of' voice is 106.055-106.20
    = 0.145 s, one 5 ms frame short of BURST_SPLIT_QUIET_SEC."""
    clip = load("cold-open-100.0.wav")
    d = clip.event("word_boundaries", 105.2)["detector_output"]
    tired, of = d["asr_start"], d["next_word"]["s"]
    edges = [e for b in voiced_bursts(clip.env, SOURCE_FLOOR) for e in b if tired < e < of + 0.2]
    assert edges == []
    quiet = clip.env.db <= SOURCE_FLOOR + 6.0
    f0, f1 = clip.env.frame_at(d["acoustic_end"] - 0.02), clip.env.frame_at(of + 0.2)
    longest, run = 0, 0
    for q in quiet[f0:f1]:
        run = run + 1 if q else 0
        longest = max(longest, run)
    assert longest * FRAME_SEC == pytest.approx(0.145, abs=FRAME_SEC / 2)
    assert longest * FRAME_SEC < BURST_SPLIT_QUIET_SEC


# 3. Invariants on every fixture ---------------------------------------------

@pytest.mark.parametrize("name", CLIP_NAMES)
@pytest.mark.parametrize("which_floor", ["source", "clip"])
def test_refined_words_never_cross_neighbours(name: str, which_floor: str) -> None:
    """end_i <= start_{i+1} for every neighbouring pair, including the pair
    whose Parakeet stamps overlap each other (pause-588.0: 'call' 595.20-595.36
    and 'it' 595.28-595.44); one RefinedWord per input word, in order, with
    the stamps carried through and a confidence from the documented set."""
    clip = load(name)
    floor = SOURCE_FLOOR if which_floor == "source" else clip.clip_floor
    ref = refined(clip, floor)
    assert [r.text for r in ref] == [w.text for w in clip.words]
    assert all(r.asr_start == w.start and r.asr_end == w.end for r, w in zip(ref, clip.words))
    for a, b in zip(ref, ref[1:]):
        assert a.end <= b.start + 1e-9, (name, a, b)
    assert all(r.confidence in (0.0, 0.5, 0.75, 1.0) for r in ref)


@pytest.mark.parametrize("name", CLIP_NAMES)
def test_refined_boundaries_that_move_land_on_burst_edges(name: str) -> None:
    """Every boundary that differs from its stamp is a voiced-burst edge, the
    previous word's refined end (monotonic pass), or the minimal span
    (start + BURST_MIN_SEC) given to a word whose stamp lies outside its burst;
    inner words (confidence 0.5) keep their stamps clipped into the burst, so
    their span lies inside one burst ('you' 104.96-105.20 becomes 105.06-105.20)."""
    clip = load(name)
    ref = refined(clip)
    bursts = voiced_bursts(clip.env, SOURCE_FLOOR)
    starts = {round(s, 6) for s, _ in bursts}
    ends = {round(e, 6) for _, e in bursts}
    prev_end = None
    for r in ref:
        if r.confidence == 0.5:
            assert any(s - 1e-9 <= r.start and r.end <= e + 1e-9 for s, e in bursts), r
        if r.start != r.asr_start:
            assert (round(r.start, 6) in starts or r.start == prev_end
                    or abs(r.end - r.start - BURST_MIN_SEC) < 1e-6), (name, r)
        if r.end != r.asr_end:
            assert (round(r.end, 6) in ends or r.end == prev_end
                    or abs(r.end - r.start - BURST_MIN_SEC) < 1e-6
                    or abs(r.end - (prev_end or 0.0) - BURST_MIN_SEC) < 1e-6), (name, r)
        prev_end = r.end


def test_refined_spans_are_at_least_burst_min_sec() -> None:
    if not CLIP_NAMES:
        pytest.skip("real-audio fixtures not present")
    short = []
    for name in CLIP_NAMES:
        clip = load(name)
        off = clip.meta["source_start"]
        short += [
            (name, r.text, round(off + r.start, 3), round(off + r.end, 3))
            for r in refined(clip) if r.end - r.start < BURST_MIN_SEC - 1e-9
        ]
    assert short == []


def test_a_word_stamped_before_its_burst_gets_the_burst_start_and_a_minimal_span() -> None:
    """'Are' (104.72-104.96) does not overlap any burst; the nearest is the hook
    burst starting 105.06, of which 'Are' is the first word but 'you' the
    second, so start = 105.06 (burst) and the end is a minimal span inside the
    burst (the stamp end lies before the voice and is useless). This used to
    yield a negative-length word."""
    clip = load("cold-open-100.0.wav")
    are = refined_word(clip, "Are", 104.72 - clip.meta["source_start"])
    bursts = voiced_bursts(clip.env, SOURCE_FLOOR)
    assert not any(overlap(s, e, are.asr_start, are.asr_end) >= BURST_MIN_SEC for s, e in bursts)
    host = next(b for b in bursts if b[0] > are.asr_end)
    assert host[0] - are.asr_end == pytest.approx(0.10, abs=0.01)
    assert are.start == pytest.approx(host[0])
    assert are.end == pytest.approx(are.start + BURST_MIN_SEC)
    assert are.burst_first and not are.burst_last


# 4. Cold open ---------------------------------------------------------------

def test_cold_open_candidates_include_the_sentence_starting_with_are() -> None:
    """The true hook start 'Are you tired...' (104.72) is a candidate. Note what
    the stamps say about it: 'Hello.' 103.60-104.72 and 'Are' 104.72-104.96
    leave a 0.000 s gap, so any pause heuristic on the stamps would drop it
    (the audio has a 0.815 s take boundary there, see the burst test below);
    cold_open_candidates now lists every head-window sentence instead."""
    clip = load("cold-open-100.0.wav")
    ev = clip.event("cold_open", 104.72)
    sents = sentence_table(clip.words)
    assert [s.text for s in sents][:2] == ["Hello.", "Are you tired of Anthropic telling you to fan out the sub"]
    are = next(s for s in sents if s.text.startswith("Are"))
    assert are.start == pytest.approx(ev["start"])
    assert are.start - sents[0].end == 0.0
    cands = cold_open_candidates(sents)
    assert [s.sid for s in cands] == [s.sid for s in sents]
    assert any(s.sid == are.sid for s in cands)


def test_cold_open_split_pause_breaks_the_hook_sentence_at_a_stamp_gap() -> None:
    """sentence_table(split_pause=0.5) splits on STAMP gaps. On the hook the
    stamps leave 0.72 s between 'tired' (ends 105.36) and 'of' (starts
    106.08), but the audio has 'tired' ringing to 106.055 and 'of' from
    106.20: 0.145 s of quiet, no pause. Measured: 'Hello.' | 'Are you tired' |
    'of Anthropic telling you to fan out the sub' - the sentence is cut in
    two, and the real take boundary (Hello. -> Are, 0.815 s) is not a split."""
    clip = load("cold-open-100.0.wav")
    off = clip.meta["source_start"]
    sents = sentence_table(clip.words, 0.5)
    assert [s.text for s in sents] == [
        "Hello.", "Are you tired", "of Anthropic telling you to fan out the sub",
    ]
    assert sents[2].start - sents[1].end == pytest.approx(0.72, abs=1e-6)
    tired = refined_word(clip, "tired", 105.2 - off)
    of = refined_word(clip, "of", 106.08 - off)
    assert of.start - tired.end == pytest.approx(0.72, abs=1e-6), "both inner words keep their stamps"
    assert sentence_table(clip.words, 0.35) == sents


def test_cold_open_take_boundary_is_visible_in_the_voiced_bursts() -> None:
    """What the stamps hide: 'Hello.' and 'Are' sit in different bursts 0.815 s
    apart with the source floor (measured 103.90-104.245 and 105.06-...)."""
    clip = load("cold-open-100.0.wav")
    off = clip.meta["source_start"]
    hello = refined_word(clip, "Hello.", 103.6 - off)
    tired = refined_word(clip, "tired", 105.2 - off)
    assert hello.confidence == 1.0
    assert hello.end == pytest.approx(104.245 - off, abs=FRAME_SEC)
    host = next(b for b in voiced_bursts(clip.env, SOURCE_FLOOR) if b[0] <= tired.asr_start <= b[1])
    assert host[0] - hello.end == pytest.approx(0.815, abs=2 * FRAME_SEC)
    assert host[0] == pytest.approx(105.06 - off, abs=FRAME_SEC)


def test_discarded_first_take_becomes_its_own_sentence_only_when_split_at_pauses() -> None:
    """The abandoned take 'are you tired ... bro it and' (30.8-43.44) carries no
    sentence punctuation: on punctuation alone sentence_table folds the whole
    clip into one sentence 'Have you heard hm hello are you tired ...'. With
    split_pause=0.5 the stamp gaps break it into 'Have you heard' | 'hm' |
    'hello' | 'are you tired ... need to do' | 'loop engineering bro it and'
    (measured), so the take starts a sentence at 30.8 and every piece is a
    cold-open candidate. The 10.48 s pause after 43.44 holds one merged
    detect_silences range on the clip (its breath-level structure sits under
    the speech-referenced threshold)."""
    clip = load("cold-open-26.0.wav")
    ev = clip.event("cold_open", 30.8)
    flat = sentence_table(clip.words)
    assert len(flat) == 1
    assert flat[0].text.startswith("Have you heard hm hello are you tired")
    split = sentence_table(clip.words, 0.5)
    assert [s.text for s in split][:3] == ["Have you heard", "hm", "hello"]
    assert split[3].start == pytest.approx(ev["start"])
    assert split[3].text.startswith("are you tired of Anthropic")
    assert split[-1].end == pytest.approx(ev["end"])
    assert [s.sid for s in cold_open_candidates(split)] == [s.sid for s in split]
    inside = [(a, b) for a, b in clip.silences if a >= ev["end"] - 1e-6]
    assert len(inside) >= 1
    assert sum(b - a for a, b in inside) >= 1.5


# 5. Moans -------------------------------------------------------------------

MOANS = [
    # clip, source start of the moan; measured burst on the clip (source seconds)
    ("retake-114.0.wav", 119.625),   # 119.625-120.035: identical to the full-file cut
    ("moan-1662.0.wav", 1665.12),    # 1665.20-1665.40 vs recorded 1665.12-1665.405
    ("moan-1729.0.wav", 1733.06),    # 1733.06-1733.66 vs recorded 1733.06-1733.67
]


@pytest.mark.parametrize("name,source_start", MOANS)
def test_voiced_burst_detected_inside_the_host_word_in_isolation(name: str, source_start: float) -> None:
    """Measured 3 of 3 moans detected on the isolated clips despite clip-local
    floors of -55.37 / -44.43 / -49.06 dBFS (source -54.28). The 119.6 clip
    additionally reports a second burst 120.035-120.255 inside 'experimenting.'
    that the full-file run (floor 1.1 dB higher) did not."""
    clip = load(name)
    ev = clip.event("moans", source_start)
    hw = ev["detector_output"]["host_word"]
    bursts = detect_voiced_bursts(clip.raw, clip.words, silences=clip.silences)
    hits = [b for b in bursts if overlap(b.start, b.end, ev["start"], ev["end"]) >= BURST_MIN_DUR]
    assert hits, [(b.start, b.end, b.text) for b in bursts]
    assert all(b.reason == "noise" and hw["w"] in b.text for b in hits)
    # The burst never reaches outside the host word's ASR span.
    assert all(b.start >= hw["s"] - 1e-6 and b.end <= hw["e"] + 1e-6 for b in hits)


# 6. Pauses ------------------------------------------------------------------

SILENT_GAPS = [
    # (clip, source gap start). The VAD is acoustic and the ASR gap is loose,
    # so the candidate may sit inside the gap or stretch past it; the contract
    # is the refined-neighbour bound asserted below. 'new' is stamped to
    # 1488.40 but its voice ends at 1487.955 (refined) and 'engineering'
    # starts at 1489.21 (refined) not 1488.96.
    ("pause-588.0.wav", 591.52),
    ("pause-1366.0.wav", 1369.2),
    ("throat-1485.0.wav", 1488.4),
]


@pytest.mark.parametrize("name,source_start", SILENT_GAPS)
def test_silent_gap_yields_a_silence_candidate(name: str, source_start: float) -> None:
    clip = load(name)
    ev = clip.event("pauses", source_start)
    g0, g1 = ev["start"], ev["end"]
    sil = [c for c in candidates(clip) if c.kind == "silence" and overlap(c.start, c.end, g0, g1) > 0]
    assert len(sil) == 1, sil
    c = sil[0]
    assert c.id.startswith("P")
    vad = next((a, b) for a, b in clip.silences if overlap(a, b, g0, g1) > 0)
    assert vad[0] <= c.start + 1e-9 and c.end <= vad[1] + 1e-9
    # derive trims the VAD silence keeping pause_keep = 0.20 s split across the edges.
    assert (vad[1] - vad[0]) - (c.end - c.start) == pytest.approx(0.20, abs=0.011)
    # ...and it never cuts into the refined neighbours' voice.
    prev = max((r for r in refined(clip) if r.end <= g0 + 1e-6), key=lambda r: r.end)
    nxt = min((r for r in refined(clip) if r.start >= g1 - 1e-6), key=lambda r: r.start)
    assert prev.end <= c.start and c.end <= nxt.start, (c, prev, nxt)


@pytest.mark.parametrize(
    "name,source_start",
    [("voiced-gap-253.0.wav", 256.88), ("voiced-gap-362.0.wav", 366.88)],
)
def test_voiced_gap_silence_candidates_are_bounded_by_refined_voice(name: str, source_start: float) -> None:
    """The 'voiced' gaps carry breath-level material, but under the
    speech-referenced VAD threshold their quiet stretches still yield silence
    candidates inside the gap (demo-approved behavior; the old noise-
    referenced full-file threshold, 2x p15 = 67, suppressed them and left the
    full cook loosely cut). Whatever the VAD finds, the candidates never cut
    into the refined neighbours' voice."""
    clip = load(name)
    ev = clip.event("pauses", source_start)
    g0, g1 = ev["start"], ev["end"]
    inside = [c for c in candidates(clip) if c.kind == "silence" and c.start >= g0 and c.end <= g1]
    assert inside, "no silence candidate inside the voiced gap"
    prev = max((r for r in refined(clip) if r.end <= g0 + 1e-6), key=lambda r: r.end)
    nxt = min((r for r in refined(clip) if r.start >= g1 - 1e-6), key=lambda r: r.start)
    assert all(prev.end <= c.start and c.end <= nxt.start for c in inside)


@pytest.mark.parametrize(
    "name,source_start",
    [("voiced-gap-253.0.wav", 256.88), ("voiced-gap-362.0.wav", 366.88)],
)
def test_voiced_gap_carries_audible_material_above_the_source_floor(name: str, source_start: float) -> None:
    """What makes these gaps 'voiced': the 5 ms envelope peaks 11.9 / 22.3 dB
    over the source floor inside the Parakeet gap (recorded peak -42.35 and
    -32.0 dBFS; the clip reproduces them within 0.5 dB), and with the source
    floor the gap contains at least one voiced burst."""
    clip = load(name)
    ev = clip.event("pauses", source_start)
    d = ev["detector_output"]
    f0, f1 = clip.env.frame_at(ev["start"]), clip.env.frame_at(ev["end"])
    peak = float(clip.env.db[f0:f1].max())
    assert peak == pytest.approx(d["peak_dbfs_in_gap"], abs=0.5)
    assert peak - SOURCE_FLOOR >= 10.0
    assert any(overlap(s, e, ev["start"], ev["end"]) >= BURST_MIN_SEC for s, e in voiced_bursts(clip.env, SOURCE_FLOOR))


# 7. Fillers -----------------------------------------------------------------

FILLERS = [
    ("smack-434.0.wav", 443.68, "um"),      # F 443.62-444.28 covers 'Um,' 443.68-444.16
    ("filler-514.0.wav", 517.52, "um"),     # F 517.46-518.22 'um uh' covers 'um' 517.52-517.84
    ("smack-958.0.wav", 962.56, "uh"),      # F 962.50-963.10 covers 'uh,' 962.56-963.04
]


@pytest.mark.parametrize("name,source_start,norm", FILLERS)
def test_filler_candidate_covers_the_recorded_filler(name: str, source_start: float, norm: str) -> None:
    clip = load(name)
    ev = clip.event("fillers", source_start)
    tok = ev["detector_output"]["token"]
    assert tok["e"] - tok["s"] >= 0.15
    fill = [c for c in candidates(clip) if c.kind == "filler" and c.start <= tok["s"] + 1e-6 and c.end >= tok["e"] - 1e-6]
    assert len(fill) == 1, [(c.id, c.start, c.end, c.text) for c in candidates(clip)]
    assert fill[0].id.startswith("F")
    assert norm in fill[0].text.split()
    # A filler cut is the token plus at most ~0.4 s of snapped padding.
    assert fill[0].end - fill[0].start <= (tok["e"] - tok["s"]) + 0.45


def test_filler_clip_matches_the_full_file_cut() -> None:
    """'Um,' 1679.60-1680.08: the full file records filler_cut 1679.54-1680.14
    with no touching silence. The speech-referenced VAD threshold keeps clip
    and full file in agreement, so the 20 s clip reproduces the full-file
    F-candidate exactly (the old clip-local noise-referenced threshold called
    the tail of 'you.' silent and fused the filler into the pause trim)."""
    clip = load("moan-1662.0.wav")
    ev = clip.event("fillers", 1679.6)
    tok = ev["detector_output"]["token"]
    assert not any(a <= tok["s"] <= b for a, b in clip.silences)
    cut = ev["detector_output"]["filler_cut"]
    cov = [c for c in candidates(clip) if c.start <= tok["s"] + 1e-6 and c.end >= tok["e"] - 1e-6]
    assert len(cov) == 1
    assert "um" in cov[0].text.split()
    assert cov[0].kind == "filler"
    assert (cov[0].start, cov[0].end) == pytest.approx((cut["start"], cut["end"]), abs=0.011)


# 8. Outro -------------------------------------------------------------------

def test_goodbye_refined_end_snaps_to_acoustic_end() -> None:
    """'Goodbye.' is stamped 1887.12-1888.16; its voice is the burst
    1887.505-1887.66 (-37..-35 dBFS) after quiet from 1887.22 (-54..-75 dBFS).
    Measured with the source floor: 1887.505-1887.660, confidence 1.0."""
    clip = load("outro-1882.0.wav")
    d = clip.event("outro", 1887.12)["detector_output"]
    r = refined_word(clip, "Goodbye.", d["word"]["s"])
    assert abs(r.end - d["acoustic_end"]) <= END_TOL
    assert r.start - r.asr_start >= 0.3
    assert r.end - r.start >= 0.1
    assert r.confidence == 1.0


def test_goodbye_with_the_clip_local_floor_attaches_to_room_tone() -> None:
    """The dead air after 'Goodbye.' drags the clip's 25th percentile to
    -68.83 dBFS; room tone is then 'voiced' (threshold -62.8), the bursts are
    1882.0-1887.175 and 1887.365-1888.15, and 'Goodbye.' attaches to the first
    by overlap: measured 1887.120-1887.175, 0.49 s before its voice."""
    clip = load("outro-1882.0.wav")
    assert clip.clip_floor < SOURCE_FLOOR - 10.0
    d = clip.event("outro", 1887.12)["detector_output"]
    r = refined_word(clip, "Goodbye.", d["word"]["s"], floor=clip.clip_floor)
    assert r.end < d["acoustic_end"] - 0.4


def test_outro_dead_air_runs_to_the_end_of_the_clip() -> None:
    """After 'Goodbye.' the silences reach the file end (the last range ends at
    the clip duration; 2.70 s of the 3.96 s from 1890.64 is silent on the
    clip); 'day.' before it refines to 1886.64-1886.84 with the source floor
    (its stamp runs to 1887.12)."""
    clip = load("outro-1882.0.wav")
    ev = clip.event("outro", 1890.64)
    assert clip.silences[-1][1] == pytest.approx(clip.duration, abs=FRAME_SEC)
    after = [(a, b) for a, b in clip.silences if a >= ev["start"] - 0.05]
    assert sum(b - a for a, b in after) >= 2.5
    day = refined_word(clip, "day.", 1886.64 - clip.meta["source_start"])
    assert day.end == pytest.approx(1886.84 - clip.meta["source_start"], abs=0.02)


# Extra: the large negative disagreement ---------------------------------------

def test_separate_folded_stamp_is_cut_back_to_the_voice() -> None:
    """'separate.' is stamped 817.60-822.40; the voiced burst (source floor) is
    817.41-818.23: voice to 817.92, a 50 ms dip, a 150 ms breath at
    817.98-818.12 (peak -37 dBFS) that BURST_SPLIT_QUIET_SEC bridges, quiet
    from 818.23. The 40 ms criterion in candidates.json ends the word at
    817.93; the module ends it after the breath at 818.23 (measured). The
    preceding 'uh' (817.20-817.36) takes the burst's start, so 'separate.' is
    an inner-start word and keeps its 817.60 stamp (confidence 0.75)."""
    clip = load("word-boundary-814.0.wav")
    d = clip.event("word_boundaries", 817.6)["detector_output"]
    r = refined_word(clip, "separate.", d["asr_start"])
    assert r.asr_end - r.end >= 4.0
    assert d["acoustic_end"] - 0.01 <= r.end <= d["acoustic_end"] + 0.32
    assert r.start == r.asr_start
    assert r.confidence == 0.75
