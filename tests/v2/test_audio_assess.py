"""Assess real recording levels without confusing silence, low noise, or stereo cancellation."""

import json
import subprocess
from pathlib import Path

from talkinghead import audio_assess
from talkinghead.judge import validate

READING = {
    "noise": {"source": "unknown", "severity": "low", "evidence": "no tone is established by the numbers"},
    "leveling": {"verdict": "quiet", "evidence": "integrated loudness sits well below target"},
    "recommendation": "Leave the recording as it is; nothing here calls for processing.",
    "owner_decisions": [],
}


class RecordingJudge:
    name = "recording"

    def __init__(self, reading: dict) -> None:
        self.reading = reading
        self.calls = 0
        self.prompt = ""

    def decide(self, prompt: str, schema: dict) -> dict:
        self.calls += 1
        self.prompt = prompt
        return validate(self.reading, schema)


def _wav(path: Path, filter_graph: str) -> Path:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", filter_graph,
                    "-c:a", "pcm_s16le", str(path)], check=True)
    return path


def test_measures_quiet_broadband_noise(tmp_path: Path) -> None:
    src = _wav(tmp_path / "room.wav", "anoisesrc=color=white:amplitude=0.01:sample_rate=48000:duration=12")
    judge = RecordingJudge(READING)
    out = audio_assess.assess(str(src), judge, None, target_lufs=-16.0, true_peak_dbtp=-1.5,
                              guidelines="G", preferences="")
    facts = out["facts"]
    assert facts["status"] == "measured"
    assert -60 < facts["floor_dbfs"] < -30
    assert facts["quiet_frame_share"] > 0.9
    assert facts["mono_sample_peak_dbfs"] < -30
    assert facts["quiet_spectrum"] is not None and "band_share" in facts["quiet_spectrum"]


def test_silence_is_explicit_null_facts_and_no_non_standard_json(tmp_path: Path) -> None:
    src = _wav(tmp_path / "silent.wav", "anullsrc=sample_rate=48000:duration=3")
    judge = RecordingJudge(READING)
    out = audio_assess.assess(str(src), judge, None, target_lufs=-16.0, true_peak_dbtp=-1.5,
                              guidelines="G", preferences="")
    facts = out["facts"]
    assert facts["status"] == "silence"
    assert all(facts[key] is None for key in audio_assess.LEVEL_KEYS)
    # Strict JSON: the judge is sent exactly these numbers, so none may be NaN or Infinity.
    json.loads(json.dumps(facts, allow_nan=False))


def test_cancelling_stereo_keeps_original_loudness_and_peak(tmp_path: Path) -> None:
    src = tmp_path / "opposite-polarity.wav"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
        "aevalsrc=0.1*sin(2*PI*440*t)|-0.1*sin(2*PI*440*t):s=48000:d=3:c=stereo",
        "-c:a", "pcm_f32le", str(src),
    ], check=True)
    facts = audio_assess.measure(str(src), target_lufs=-16.0, true_peak_dbtp=-1.5)
    assert facts["status"] == "mono_analysis_unavailable"
    assert -30 < facts["input_integrated_lufs"] < -10
    assert -22 < facts["input_true_peak_dbtp"] < -18
    assert facts["gain_to_target_db"] is not None
    assert facts["floor_dbfs"] is None and facts["quiet_spectrum"] is None
    json.dumps(facts, allow_nan=False)
