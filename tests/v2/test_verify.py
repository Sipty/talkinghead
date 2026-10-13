"""Verification on the render: a declarative check table, fatal or recorded,
no fix loops."""

import subprocess
import wave

import numpy as np
import pytest

from talkinghead import verify
from talkinghead.audio import envelope, noise_floor
from talkinghead.config import Config
from talkinghead.joins import check_join, source_edge_facts
from talkinghead.store import Cache, sha256_file
from talkinghead.timeline import Cut, Plan, SourceInfo
from talkinghead.transcribe import FakeTranscriber, Word, prepare_audio
from talkinghead.verify import CheckResult, collect_facts, run_checks

SRC = SourceInfo("main", "/v/in.mp4", "x", 100.0)


def _plan() -> Plan:
    p = Plan.from_cuts(SRC, [Cut(0.0, 10.0, "cold-open"), Cut(50.0, 52.0, "silence"), Cut(95.0, 100.0, "outro")])
    p.decisions["cold_open"] = {"start": 10.0}
    return p


def _facts(**over) -> dict:
    facts = {
        "duration": 83.0, "video_duration": 83.0, "audio_duration": 83.0,
        "lufs": -16.1,
        # Dense speech: a word every 0.4 s from 0.1 s to the end, no gap >= 0.5 s.
        "words": [Word("Are", 0.10, 0.3, 1.0)] + [Word(f"w{i}", 0.4 * i, 0.4 * i + 0.3, 1.0) for i in range(1, 206)],
        "join_grades": [{"t": 40.0, "grade": "OK"}],
        "gap_budget": 1.0, "dead_air": [],
        "render_silences": {"quiet": [], "digital": []},
    }
    facts.update(over)
    return facts


def test_all_checks_pass_on_a_clean_render() -> None:
    results = run_checks(_plan(), _facts())
    assert all(r.ok for r in results)
    assert {r.name for r in results} >= {
        "residual_gaps", "render_silence", "joins", "join_edges",
        "source_timing", "source_similarity", "unclassified_voice", "color"}


def test_duration_mismatch_is_fatal() -> None:
    results = {r.name: r for r in run_checks(_plan(), _facts(duration=80.0))}
    assert not results["duration"].ok and results["duration"].severity == "fatal"


def test_av_sync_uses_stream_durations() -> None:
    results = {r.name: r for r in run_checks(_plan(), _facts(video_duration=83.0, audio_duration=84.2))}
    assert not results["av_sync"].ok and results["av_sync"].severity == "fatal"


def test_loudness_window_and_cold_open_invariant() -> None:
    results = {r.name: r for r in run_checks(_plan(), _facts(lufs=-13.0))}
    assert not results["loudness"].ok
    late = _facts(words=[Word("Are", 1.2, 1.4, 1.0), Word("bye.", 82.0, 82.5, 1.0)])
    results = {r.name: r for r in run_checks(_plan(), late)}
    assert not results["cold_open"].ok and results["cold_open"].severity == "fatal"


def test_residual_gap_budget_is_fatal_and_join_flags_are_recorded() -> None:
    gappy = _facts(dead_air=_excess([(3.0, 5.0)]), gap_budget=1.0)
    results = {r.name: r for r in run_checks(_plan(), gappy)}
    assert not results["residual_gaps"].ok and results["residual_gaps"].severity == "fatal"
    flagged = _facts(join_grades=[{"t": 40.0, "grade": "CLIPPED: voiced audio within 30ms of join"}])
    results = {r.name: r for r in run_checks(_plan(), flagged)}
    assert not results["joins"].ok and results["joins"].severity == "warn"


def test_check_result_renders_one_line() -> None:
    r = CheckResult("duration", "fatal", True, "82.99s vs 83.00s expected")
    assert "duration" in r.line() and "OK" in r.line()


def test_residual_dead_air_is_measured_from_the_source_vad_through_the_plan() -> None:
    from talkinghead.verify import residual_dead_air

    plan = Plan.from_cuts(SRC, [Cut(10.0, 14.0, "silence"), Cut(30.0, 31.0, "silence")])
    # Source silences: 9.9-14.1 (trimmed to 0.1 s each side by the cut), 20.0-20.8
    # (untouched: 0.8 s of dead air ships), 29.5-31.5 (0.5 s left each side of a join:
    # they meet at the join and add up to 1.0 s), 60.0-60.3 (under the bar).
    silences = [(9.9, 14.1), (20.0, 20.8), (29.5, 31.5), (60.0, 60.3)]
    runs = residual_dead_air(plan, silences, min_gap=0.5)
    assert [(round(a, 2), round(b, 2)) for a, b in runs] == [(16.0, 16.8), (25.5, 26.5)]  # output timeline
    assert sum(b - a for a, b in runs) == pytest.approx(1.8)


def _excess(runs, words=None) -> list[dict]:
    """dead_air facts for silence runs on the output timeline; the default
    transcript is the dense unpunctuated one of `_facts`, so every run is mid-sentence."""
    return verify.dead_air_excess(runs, words or _facts()["words"], Config())


def test_residual_gap_check_counts_only_the_excess_over_the_boundary_cap() -> None:
    results = {r.name: r for r in run_checks(_plan(), _facts(dead_air=_excess([(20.0, 20.8), (25.5, 26.5)]), gap_budget=0.5))}
    # mid-sentence cap 0.40 + 0.10 slack: 0.3 s and 0.5 s of it are dead air.
    assert not results["residual_gaps"].ok and "0.80s" in results["residual_gaps"].detail
    assert results["residual_gaps"].detail.count("mid") == 2
    results = {r.name: r for r in run_checks(_plan(), _facts(dead_air=[], gap_budget=1.0))}
    assert results["residual_gaps"].ok


def test_intended_pauses_are_not_dead_air_but_a_long_mid_sentence_gap_is() -> None:
    """Silence runs as collect_facts builds them: source VAD silence through a plan,
    classified by the transcript word before each run."""
    plan = Plan.from_cuts(SRC, [Cut(10.4, 12.1, "silence")])
    silences = [(10.0, 12.5), (20.0, 20.7), (30.0, 31.5)]
    # Output timeline (1.7 s removed): the runs sit at 10.0-10.8, 18.3-19.0 and 28.3-29.8.
    words = [Word("done.", 9.5, 10.0, 1.0), Word("end.", 17.8, 18.3, 1.0), Word("so", 27.8, 28.3, 1.0)]
    runs = verify._dead_air_runs(plan, silences, verify.GAP_MIN)
    topic, sentence, mid = verify.dead_air_excess(
        [(a, b) for a, b, _ in runs], words, Config(), raw=[r for _, _, r in runs])
    assert (topic["boundary"], sentence["boundary"], mid["boundary"]) == ("topic", "sentence", "mid")
    # 0.80 s kept out of a 2.5 s topic pause: inside the topic cap.
    assert topic["end"] - topic["start"] == pytest.approx(0.8) and topic["excess"] == 0.0
    # 0.70 s after a sentence: 0.05 s beyond cap 0.55 + 0.10 slack, far inside the budget.
    assert sentence["excess"] == pytest.approx(0.05, abs=1e-3)
    # 1.5 s untouched mid-sentence: 1.0 s beyond cap 0.40 + 0.10.
    assert mid["excess"] == pytest.approx(1.0, abs=1e-3)
    results = {r.name: r for r in run_checks(_plan(), _facts(dead_air=[topic, sentence]))}
    assert results["residual_gaps"].ok
    assert "1 of 2" in results["residual_gaps"].detail
    results = {r.name: r for r in run_checks(_plan(), _facts(dead_air=[topic, sentence, mid], gap_budget=0.9))}
    assert not results["residual_gaps"].ok and results["residual_gaps"].severity == "fatal"


def test_a_pause_kept_beyond_its_cap_is_counted_by_the_excess() -> None:
    words = [Word("done.", 5.0, 5.4, 1.0)]
    (run,) = verify.dead_air_excess([(5.4, 6.7)], words, Config(), raw=[1.8])  # topic cap 0.80 + 0.10
    assert run["boundary"] == "topic" and run["excess"] == pytest.approx(0.4, abs=1e-3)
    (run,) = verify.dead_air_excess([(5.4, 6.7)], words, Config())  # same run, raw 1.3 s: sentence
    assert run["boundary"] == "sentence" and run["excess"] == pytest.approx(0.65, abs=1e-3)






def test_render_silence_quiet_runs_share_the_gap_budget() -> None:
    results = {r.name: r for r in run_checks(
        _plan(), _facts(render_silences={"quiet": [(3.0, 4.4)], "digital": []}, gap_budget=1.0))}
    assert not results["render_silence"].ok and results["render_silence"].severity == "fatal"
    assert "1.40s" in results["render_silence"].detail
    results = {r.name: r for r in run_checks(
        _plan(), _facts(render_silences={"quiet": [(3.0, 3.8)], "digital": []}, gap_budget=1.0))}
    assert results["render_silence"].ok


def test_render_silence_digital_muting_is_fatal_regardless_of_budget() -> None:
    results = {r.name: r for r in run_checks(
        _plan(), _facts(render_silences={"quiet": [], "digital": [(50.0, 90.0)]}, gap_budget=100.0))}
    assert not results["render_silence"].ok and results["render_silence"].severity == "fatal"
    assert "50.0-90.0" in results["render_silence"].detail


def test_render_silences_measures_the_render_wav(tmp_path) -> None:
    """Both silence tiers must detect a true mute after a voiced segment."""
    wav = tmp_path / "tone-then-mute.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i",
         "aevalsrc=sin(2*PI*440*t)*between(t\\,0\\,2):s=16000:d=4",
         "-c:a", "pcm_s16le", str(wav)],
        check=True)
    sil = verify.render_silences(str(wav))
    assert sil["digital"] == pytest.approx([(2.0, 4.0)], abs=0.05)
    assert any(abs(a - 2.0) < 0.05 for a, _ in sil["quiet"])


def test_collect_facts_refreshes_a_stale_workdir_wav(tmp_path, monkeypatch) -> None:
    """The render is what is measured: a 16 kHz wav left in the verify workdir
    by an earlier render must be re-extracted, not trusted by existence, or the
    facts and the editorial transcript would describe the wrong file."""
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc=size=160x120:rate=15:duration=2",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(clip)],
        check=True)
    work = tmp_path / "verify"
    work.mkdir()
    stale = b"stale bytes from an earlier render"
    (work / "audio-16k.wav").write_bytes(stale)
    monkeypatch.setattr(verify, "measure_lufs", lambda path: -16.0)
    plan = Plan.from_cuts(SRC, [])
    facts = collect_facts(plan, str(clip),
                          transcriber=FakeTranscriber([Word("hi", 0.1, 0.3, 1.0)], 1.0),
                          cache=Cache(tmp_path / "cache"), workdir=work,
                          source_duration=2.0, source_silences=[])
    assert facts["words"][0].text == "hi"
    wav = work / "audio-16k.wav"
    assert wav.read_bytes() != stale
    _, fresh_sha = prepare_audio(str(clip), tmp_path / "fresh", force=True)
    assert sha256_file(wav) == fresh_sha


# --- source<->render timing ---

def _noise_video(tmp_path, seconds: float, name: str) -> str:
    """A video whose audio is white noise: its autocorrelation is a spike, so a
    cross-correlation peak is the true lag and nothing else."""
    path = tmp_path / name
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"color=c=black:size=160x120:rate=15:duration={seconds}",
         "-f", "lavfi", "-i", f"anoisesrc=r=16000:d={seconds}:c=white",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(path)],
        check=True)
    return str(path)


def test_source_timing_finds_a_clean_render_and_a_shifted_one(tmp_path) -> None:
    """Duration, loudness and silence all survive a render whose audio sits in
    the wrong place; the cross-correlation against the source does not."""
    source = _noise_video(tmp_path, 12.0, "source.mp4")
    same, shifted = tmp_path / "same.mp4", tmp_path / "shifted.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", source, "-c", "copy", str(same)], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "0.15", "-i", source, "-c", "copy", str(shifted)], check=True)
    plan = Plan.from_cuts(SourceInfo("main", source, "sha", 12.0), [Cut(11.0, 12.0, "outro")])

    good = verify.measure_source_timing(plan, str(same))
    assert good and abs(good[0]["source_offset_error_seconds"]) <= verify.SOURCE_TIMING_TOL
    assert good[0]["correlation"] >= verify.SOURCE_CORR_MIN
    results = {r.name: r for r in run_checks(plan, _facts(source_timing=good))}
    assert results["source_timing"].ok and results["source_similarity"].ok

    bad = verify.measure_source_timing(plan, str(shifted))
    assert bad[0]["source_offset_error_seconds"] == pytest.approx(0.15, abs=0.03)
    results = {r.name: r for r in run_checks(plan, _facts(source_timing=bad))}
    assert not results["source_timing"].ok and results["source_timing"].severity == "fatal"


def test_source_similarity_warns_without_stopping_the_run() -> None:
    # The render passes through compressor/loudnorm, so a weak correlation is
    # suspicious rather than conclusive: it is recorded, not fatal.
    facts = _facts(source_timing=[{"output_seconds": 4.0, "source_offset_error_seconds": 0.01,
                                   "correlation": 0.4}])
    results = {r.name: r for r in run_checks(_plan(), facts)}
    assert results["source_timing"].ok
    assert not results["source_similarity"].ok and results["source_similarity"].severity == "warn"


def test_source_timing_reports_missing_evidence_instead_of_passing() -> None:
    """An empty measurement is not a clean bill of health: a plan of short keeps
    or undecodable windows has to say so, not report OK."""
    results = {r.name: r for r in run_checks(_plan(), _facts(source_timing=[]))}
    assert not results["source_timing"].ok and results["source_timing"].severity == "warn"
    assert "not measured" in results["source_timing"].detail
    # No source path was ever given (a hand render, or hand-built facts): the
    # evidence was not expected, so this is a pass.
    assert {r.name: r for r in run_checks(_plan(), _facts())}["source_timing"].ok


def test_source_timing_does_not_invent_an_offset_from_a_silent_window(tmp_path) -> None:
    """Correlating a silent window answers with lag 0, which would read as a
    fabricated -pad offset. A mute is the silence tiers' finding, not a timing
    failure, so the window is reported unmeasured."""
    source = _noise_video(tmp_path, 4.0, "source.mp4")
    muted = tmp_path / "muted.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", source, "-c:v", "copy",
                    "-af", "volume=0", "-c:a", "aac", str(muted)], check=True)
    plan = Plan.from_cuts(SourceInfo("main", source, "sha", 4.0), [Cut(3.5, 4.0, "outro")])
    checks = verify.measure_source_timing(plan, str(muted))
    assert checks and "unmeasured" in checks[0] and "source_offset_error_seconds" not in checks[0]
    results = {r.name: r for r in run_checks(plan, _facts(source_timing=checks))}
    assert not results["source_timing"].ok and results["source_timing"].severity == "warn"
    assert results["source_similarity"].ok


def test_color_check_requires_depth_and_tag_parity_for_an_hdr_source() -> None:
    hdr = {"primaries": "bt2020", "transfer": "arib-std-b67", "matrix": "bt2020nc", "range": "tv",
           "pix_fmt": "p010le", "hdr": True}
    tagged_8bit = dict(hdr, pix_fmt="yuv420p")     # the right tags on an 8-bit stream
    results = {r.name: r for r in run_checks(_plan(), _facts(color={"source": hdr, "render": tagged_8bit}))}
    assert not results["color"].ok and results["color"].severity == "fatal"
    assert "8-bit" in results["color"].detail
    drifted = dict(hdr, matrix="bt709")            # one tag lost on a 10-bit stream
    assert not {r.name: r for r in run_checks(
        _plan(), _facts(color={"source": hdr, "render": drifted}))}["color"].ok
    assert {r.name: r for r in run_checks(
        _plan(), _facts(color={"source": hdr, "render": dict(hdr)}))}["color"].ok


def test_render_silences_tier_room_tone_as_dead_air_not_muting(tmp_path) -> None:
    """Even one PCM quantization step is nonzero audio, not a dropped stream."""
    quiet = tmp_path / "quiet.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
         "aevalsrc=1/32768:s=16000:d=2", "-c:a", "pcm_s16le", str(quiet)],
        check=True)
    sil = verify.render_silences(str(quiet))
    assert sil["digital"] == []
    assert any(b - a >= 1.5 for a, b in sil["quiet"])


def test_source_timing_and_colour_are_measured_from_the_real_files(tmp_path, monkeypatch) -> None:
    clip = _noise_video(tmp_path, 8.0, "clip.mp4")
    plan = Plan.from_cuts(SourceInfo("main", clip, "sha", 8.0), [Cut(7.0, 8.0, "outro")])
    monkeypatch.setattr(verify, "measure_lufs", lambda path: -16.0)
    facts = collect_facts(plan, clip, transcriber=FakeTranscriber([Word("hi", 0.1, 0.3, 1.0)], 1.0),
                          cache=Cache(tmp_path / "cache"), workdir=tmp_path / "verify",
                          source_duration=8.0, source_silences=[], source_path=clip,
                          voices=[(1.0, 1.2)], speech_probes=[])
    assert facts["source_timing"] and facts["color"] is not None
    results = {r.name: r for r in run_checks(plan, facts)}
    assert results["source_timing"].ok and results["color"].ok and results["unclassified_voice"].ok


# --- speech-level audio the transcript does not cover ---

def test_unclassified_voice_facts_split_kept_spans_from_removed_ones() -> None:
    """The reference recording's case: a 0.16 s utterance inside a pause trim
    that no transcript entry named. The report names the removal and the probe's
    words; a span the plan keeps stays out of it."""
    plan = Plan.from_cuts(SRC, [Cut(10.0, 14.0, "silence")])
    voices = [(11.0, 11.16), (40.0, 40.5)]
    probes = [{"start": 11.0, "end": 11.16, "backend": "whisper", "text": "Uh the"},
              {"start": 40.0, "end": 40.5, "backend": "whisper", "text": "mmm"}]
    facts = verify.unclassified_voice_facts(plan, voices, probes)
    assert [s["start"] for s in facts["spans"]] == [11.0, 40.0]
    (removed,) = facts["removed"]
    assert removed["reason"] == "silence" and removed["probe"] == "Uh the"
    assert removed["removed_sec"] == pytest.approx(0.16)

    results = {r.name: r for r in run_checks(plan, _facts(unclassified_voice=facts))}
    assert not results["unclassified_voice"].ok and results["unclassified_voice"].severity == "warn"
    assert "Uh the" in results["unclassified_voice"].detail

    clean = verify.unclassified_voice_facts(plan, voices[1:], probes[1:])
    assert clean["removed"] == []
    assert {r.name: r for r in run_checks(plan, _facts(unclassified_voice=clean))}["unclassified_voice"].ok


def test_color_check_fails_a_silent_hdr_to_sdr_conversion() -> None:
    hdr = {"primaries": "bt2020", "transfer": "arib-std-b67", "matrix": "bt2020nc", "range": "tv",
           "pix_fmt": "p010le", "hdr": True}
    sdr = {"primaries": "bt709", "transfer": "bt709", "matrix": "bt709", "range": "tv",
           "pix_fmt": "yuv420p", "hdr": False}
    results = {r.name: r for r in run_checks(_plan(), _facts(color={"source": hdr, "render": sdr}))}
    assert not results["color"].ok and results["color"].severity == "fatal"
    assert "arib-std-b67" in results["color"].detail
    kept = {r.name: r for r in run_checks(_plan(), _facts(color={"source": hdr, "render": dict(hdr)}))}
    assert kept["color"].ok
    # A 10-bit stream with no tags is SDR by construction: no HDR claim is
    # invented for the source, so the untagged pair passes.
    results = {r.name: r for r in run_checks(
        _plan(), _facts(color={"source": dict(sdr, pix_fmt="yuv420p10le"), "render": sdr}))}
    assert results["color"].ok


# --- source-side edge check, pacing facts ---

SR16 = 16000


def _tone_then_room(tmp_path):
    """3 s of source audio: a 440 Hz tone for the first second, room noise after.
    Returns (int16 samples, envelope, floor)."""
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 30.0, 3 * SR16)
    t = np.arange(SR16) / SR16
    x[:SR16] = 10000.0 * np.sin(2 * np.pi * 440.0 * t)
    x = x.astype(np.int16)
    env = envelope(x, SR16)
    return x, env, noise_floor(env.db)


def _write_wav(path, samples) -> str:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR16)
        w.writeframes(samples.astype(np.int16).tobytes())
    return str(path)


def test_edge_check_flags_a_cut_that_lands_on_a_loud_tone_the_render_grader_passes(tmp_path) -> None:
    """The cut removes 0.8-1.6 s: its left edge is mid-tone (a chopped word), its
    right edge is room noise. What follows the seam in the render is quiet, so
    `check_join` grades it OK; the source-side check sees the chopped left edge."""
    x, env, floor = _tone_then_room(tmp_path)
    src = SourceInfo("main", "/v/in.mp4", "x", 3.0)
    plan = Plan.from_cuts(src, [Cut(0.8, 1.6, "retake")])

    render = _write_wav(tmp_path / "render.wav", np.concatenate([x[: int(0.8 * SR16)], x[int(1.6 * SR16):]]))
    assert check_join(render, 0.8)["grade"] == "OK"

    (edge,) = source_edge_facts(plan, env, floor)
    assert edge["flag"] == "left" and edge["reason"] == "retake"
    assert edge["left"]["voiced"] and not edge["right"]["voiced"]
    assert edge["t"] == pytest.approx(0.8)

    results = {r.name: r for r in run_checks(plan, _facts(join_edges=[edge]))}
    assert not results["join_edges"].ok and results["join_edges"].severity == "warn"
    assert "left (retake)" in results["join_edges"].detail


def test_edge_check_passes_a_cut_between_quiet_edges_and_measures_the_quiet_it_keeps(tmp_path) -> None:
    _, env, floor = _tone_then_room(tmp_path)
    plan = Plan.from_cuts(SourceInfo("main", "/v/in.mp4", "x", 3.0), [Cut(1.2, 1.8, "silence")])
    (edge,) = source_edge_facts(plan, env, floor)
    assert edge["flag"] is None
    # Left keep [0, 1.2): tone to 1.0, then 0.2 s of room. Right keep [1.8, 3.0): all room.
    assert edge["quiet_left"] == pytest.approx(0.2, abs=0.015)
    assert edge["quiet_right"] == pytest.approx(1.2, abs=0.015)
    assert {r.name: r for r in run_checks(plan, _facts(join_edges=[edge]))}["join_edges"].ok


def test_a_tail_that_decayed_below_the_word_peak_is_not_a_chopped_edge(tmp_path) -> None:
    """Loud (well above floor) but 20 dB under the local peak: a natural word end."""
    rng = np.random.default_rng(5)
    x = rng.normal(0.0, 30.0, 2 * SR16)
    x[:SR16 // 2] = 10000.0 * np.sin(2 * np.pi * 440.0 * np.arange(SR16 // 2) / SR16)
    x[SR16 // 2: SR16 // 2 + 640] = 900.0 * np.sin(2 * np.pi * 440.0 * np.arange(640) / SR16)
    env = envelope(x.astype(np.int16), SR16)
    floor = noise_floor(env.db)
    plan = Plan.from_cuts(SourceInfo("main", "/v/in.mp4", "x", 2.0), [Cut(0.54, 1.5, "filler")])
    (edge,) = source_edge_facts(plan, env, floor)
    assert edge["left"]["db"] >= floor + 15 and edge["left"]["peak"] - edge["left"]["db"] > 6
    assert edge["flag"] is None


class _W:  # the refined words only need text/start/end here
    def __init__(self, text, start, end):
        self.text, self.start, self.end = text, start, end


def test_pacing_facts_compare_kept_pauses_with_the_source_per_boundary_type() -> None:
    words = [_W("One.", 0.0, 0.5), _W("Two,", 2.0, 2.5), _W("three", 3.0, 3.4), _W("four", 4.2, 4.6),
             _W("five.", 4.7, 5.0)]
    # VAD silences: 1.5 s after "One." (topic), 0.5 s after "Two," (clause, untouched),
    # 0.8 s after "three" (mid), 0.1 s after "four" (below pause_above, ignored).
    silences = [(0.5, 2.0), (2.5, 3.0), (3.4, 4.2), (4.6, 4.7)]
    plan = Plan.from_cuts(SourceInfo("main", "/v/in.mp4", "x", 6.0),
                          [Cut(0.65, 1.85, "silence"), Cut(3.5, 4.1, "silence")])
    edges = [{"reason": "silence", "quiet": 0.30}, {"reason": "silence", "quiet": 0.20},
             {"reason": "retake", "quiet": 0.02}]
    facts = verify.pacing_facts(plan, words, silences, edges, Config())
    pauses = facts["pauses"]
    assert set(pauses) == {"topic", "clause", "mid"}
    assert pauses["topic"]["source"]["p50"] == pytest.approx(1.5)
    assert pauses["topic"]["output"]["p50"] == pytest.approx(0.30)
    assert pauses["clause"]["output"]["max"] == pytest.approx(0.5)
    assert pauses["mid"]["source"]["p50"] == pytest.approx(0.8)
    assert pauses["mid"]["output"]["p50"] == pytest.approx(0.2)
    assert facts["seam_quiet"] == {"retake": {"count": 1, "median": 0.02},
                                   "silence": {"count": 2, "median": 0.25}}


def test_pacing_ignores_pauses_next_to_a_removed_word() -> None:
    words = [_W("keep.", 0.0, 0.5), _W("gone", 1.5, 1.9), _W("kept", 3.0, 3.4)]
    plan = Plan.from_cuts(SourceInfo("main", "/v/in.mp4", "x", 5.0), [Cut(1.4, 2.0, "content")])
    facts = verify.pacing_facts(plan, words, [(0.5, 1.5)], [], Config())
    # "keep." -> "gone" is 1.0 s in the source, but "gone" is cut: no kept pause to report.
    assert facts["pauses"] == {}


def test_a_hum_between_two_words_is_not_a_kept_pause() -> None:
    """A 1.4 s word-stamp gap holds a 0.45 s hum: the two quiet stretches around it are
    0.38 s and 0.2 s, so the only pause is the 0.38 s one, kept at 0.35 s - under the
    sentence cap. The policy does not trim the hum, so it is not a pause it kept."""
    words = [_W("essentially.", 0.0, 0.5), _W("And", 1.9, 2.1)]
    silences = [(0.5, 0.88), (1.33, 1.53)]  # hum at 0.88-1.33 (voiced, no word)
    plan = Plan.from_cuts(SourceInfo("main", "/v/in.mp4", "x", 3.0), [Cut(0.6, 0.63, "silence")])
    facts = verify.pacing_facts(plan, words, silences, [], Config())
    (only,) = facts["pauses"].items()
    assert only[0] == "sentence" and only[1]["count"] == 1
    assert only[1]["source"]["max"] == pytest.approx(0.38)
    assert only[1]["output"]["max"] == pytest.approx(0.35)


def test_a_downscaled_full_render_is_fatal_but_the_demo_tier_may_scale() -> None:
    native = {"source": [3840, 2160], "render": [3840, 2160], "scale": None}
    shrunk = {"source": [3840, 2160], "render": [1920, 1080], "scale": None}
    demo = {"source": [3840, 2160], "render": [854, 480], "scale": "-2:480"}
    check = {f["render"][0]: {r.name: r for r in run_checks(_plan(), _facts(frame=f))}["resolution"]
             for f in (native, shrunk, demo)}
    assert check[3840].ok
    assert not check[1920].ok and check[1920].severity == "fatal"
    assert check[854].ok


def test_display_size_follows_a_phone_rotation_tag(tmp_path) -> None:
    # iPhone footage stores 2160x3840 with a -90 rotation and displays landscape;
    # comparing stored axes would fail every native render of it.
    flat, rotated = tmp_path / "flat.mp4", tmp_path / "rotated.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=32x64:rate=1",
                    "-frames:v", "1", "-c:v", "libx264", str(flat)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-display_rotation", "-90", "-i", str(flat),
                    "-c", "copy", str(rotated)], check=True)
    assert verify.display_size(str(flat)) == (32, 64)
    assert verify.display_size(str(rotated)) == (64, 32)


# --- audio-only renders: facts and checks for a WAV, and the no-output-ASR boundary ---

def _wav(path, seconds: float, *, source: str = "anoisesrc=r=48000:c=white:a=0.2", codec: str = "pcm_s16le") -> str:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"{source}:d={seconds}",
                    "-ar", "48000", "-c:a", codec, str(path)], check=True)
    return str(path)


def _audio_facts(tmp_path, render: str, source: str, *, plan: Plan, **kw) -> dict:
    return collect_facts(plan, render, cache=Cache(tmp_path / "cache"), workdir=tmp_path / "verify",
                         source_duration=6.0, source_silences=[], source_path=source,
                         audio_only=True, output_asr=False,
                         source_words=[Word("hello", 0.2, 0.6, 1.0)], **kw)


def test_project_words_keeps_only_words_that_survive_whole_on_the_output_timeline() -> None:
    plan = Plan.from_cuts(SourceInfo("main", "/x.wav", "x", 10.0), [Cut(2.0, 3.0, "silence"), Cut(5.0, 5.4, "filler")])
    words = [Word("kept", 0.5, 1.0, 1.0), Word("cut", 2.2, 2.6, 1.0), Word("clipped", 1.8, 2.2, 1.0),
             Word("later", 6.0, 6.5, 1.0)]
    projected = [(w.text, round(w.start, 2), round(w.end, 2)) for w in verify.project_words(plan, words)]
    assert projected == [("kept", 0.5, 1.0), ("later", 4.6, 5.1)]


def test_audio_facts_measure_a_wav_without_inventing_picture_or_asr_facts(tmp_path, monkeypatch) -> None:
    source = _wav(tmp_path / "src.wav", 6.0)
    render = str(tmp_path / "out.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", source, "-t", "4", "-c", "copy", render], check=True)
    monkeypatch.setattr(verify, "measure_lufs", lambda path: -16.0)
    plan = Plan.from_cuts(SourceInfo("main", source, "x", 6.0), [Cut(4.0, 6.0, "outro")])
    facts = _audio_facts(tmp_path, render, source, plan=plan)
    assert facts["mode"] == "audio" and facts["color"] is None and facts["frame"] is None
    assert facts["video_duration"] is None and facts["audio_stream"]["codec"] == "pcm_s16le"
    assert [w.text for w in facts["words"]] == ["hello"]
    results = {r.name: r for r in run_checks(plan, facts)}
    assert not {"av_sync", "color", "resolution"} & set(results)
    assert "NOT transcribed" in results["output_asr"].detail
    assert "not measured" in results["cold_open"].detail
    assert all(r.ok or r.severity != "fatal" for r in results.values()), [r.line() for r in results.values()]
    assert results["source_timing"].ok and "sampled keeps" in results["source_timing"].detail


def test_a_transcriber_that_replays_the_source_is_refused_as_output_asr(tmp_path) -> None:
    class Replays:
        name, version, replays_fixed_source = "saved", "1", True

        def transcribe(self, wav_path):
            raise AssertionError("must not be called")

    source = _wav(tmp_path / "src.wav", 2.0)
    plan = Plan.from_cuts(SourceInfo("main", source, "x", 2.0), [])
    with pytest.raises(RuntimeError, match="--no-output-asr"):
        collect_facts(plan, source, transcriber=Replays(), cache=Cache(tmp_path / "cache"),
                      workdir=tmp_path / "verify", source_duration=2.0, source_silences=[], audio_only=True)


def test_an_audio_render_that_is_not_the_promised_wav_fails_a_fatal_check(tmp_path, monkeypatch) -> None:
    source = _wav(tmp_path / "src.wav", 6.0)
    render = _wav(tmp_path / "out.m4a", 6.0, codec="aac")
    monkeypatch.setattr(verify, "measure_lufs", lambda path: -16.0)
    plan = Plan.from_cuts(SourceInfo("main", source, "x", 6.0), [])
    results = {r.name: r for r in run_checks(plan, _audio_facts(tmp_path, render, source, plan=plan))}
    assert not results["audio_stream"].ok and results["audio_stream"].severity == "fatal"


def test_a_muted_audio_render_is_caught_by_the_silence_and_timing_checks(tmp_path, monkeypatch) -> None:
    source = _wav(tmp_path / "src.wav", 6.0)
    render = _wav(tmp_path / "out.wav", 6.0, source="anullsrc=r=48000:cl=mono")
    monkeypatch.setattr(verify, "measure_lufs", lambda path: -16.0)
    plan = Plan.from_cuts(SourceInfo("main", source, "x", 6.0), [])
    results = {r.name: r for r in run_checks(plan, _audio_facts(tmp_path, render, source, plan=plan))}
    assert not results["render_silence"].ok and "DIGITAL" in results["render_silence"].detail
    assert not results["source_timing"].ok


def _judged_plan(cuts: list[Cut]) -> Plan:
    from talkinghead.timeline import Plan as P
    src = SourceInfo("main", "/v/in.mp4", "sha", 10.0)
    return P.from_cuts(src, cuts)


class _W:
    def __init__(self, text: str, start: float, end: float) -> None:
        self.text, self.start, self.end, self.speaker = text, start, end, None


def test_judged_removals_fails_on_a_kept_word_a_judged_cut_removed_and_names_the_first_few() -> None:
    # The judge removed [2, 4); the plan keeps "yes" (3.0 to 3.4) because a cut chewed the span.
    words = [_W("so", 1.0, 1.5), _W("yes", 3.0, 3.4), _W("right", 3.5, 3.9), _W("next", 6.0, 6.5)]
    plan = _judged_plan([Cut(2.0, 2.9, "silence"), Cut(3.7, 4.0, "silence")])
    judged = [(2.0, 4.0, "tangent")]
    leaks = verify.judged_leaks(plan, words, judged)
    assert [x["text"] for x in leaks] == ["yes"]
    res = verify._judged_removals(plan, {"judged_leaks": leaks})
    assert not res.ok and res.severity == "fatal" and "'yes' at 3.00s (tangent)" in res.detail


def test_judged_removals_passes_when_every_judged_word_stays_removed() -> None:
    words = [_W("so", 1.0, 1.5), _W("yes", 2.1, 2.4), _W("next", 6.0, 6.5)]
    plan = _judged_plan([Cut(2.0, 4.0, "tangent")])
    leaks = verify.judged_leaks(plan, words, [(2.0, 4.0, "tangent")])
    assert leaks == []
    assert verify._judged_removals(plan, {"judged_leaks": leaks}).ok


def test_judged_removals_is_not_measured_without_judged_decisions() -> None:
    res = verify._judged_removals(_judged_plan([]), {})
    assert res.ok and res.severity == "info"
