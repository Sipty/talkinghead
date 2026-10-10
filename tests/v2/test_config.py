"""Configuration overrides and invalid values must fail predictably."""

from pathlib import Path

import pytest

from talkinghead.config import Config, load_config


def test_overlay_corner_is_a_framing_choice_with_a_validated_name() -> None:
    assert load_config(visuals_corner="top-left").visuals_corner == "top-left"
    with pytest.raises(ValueError):
        Config(visuals_corner="top-middle").validate()




def test_toml_overrides_known_keys_and_rejects_unknown(tmp_path: Path) -> None:
    f = tmp_path / "talkinghead.toml"
    f.write_text('pause_above = 0.6\nvotes = 3\nagree = 3\n')
    cfg = load_config(f)
    assert cfg.pause_above == 0.6 and cfg.votes == 3 and cfg.agree == 3
    f.write_text('pause_abvoe = 0.6\n')
    with pytest.raises(ValueError, match="pause_abvoe"):
        load_config(f)


def test_agree_cannot_exceed_votes() -> None:
    with pytest.raises(ValueError):
        Config(votes=3, agree=4).validate()




def test_seam_lengths_cannot_be_negative() -> None:
    for key in ("splice_quiet_sentence", "splice_quiet_clause", "splice_pad"):
        with pytest.raises(ValueError, match="splice_"):
            Config(**{key: -0.01}).validate()


def test_the_flag_beats_the_environment_which_beats_the_config_file(tmp_path: Path, monkeypatch) -> None:
    f = tmp_path / "talkinghead.toml"
    f.write_text('judge = "shell:from-file"\ntranscriber = "whisper"\n')
    cfg = load_config(f)
    assert (cfg.judge, cfg.transcriber) == ("shell:from-file", "whisper")
    monkeypatch.setenv("TALKINGHEAD_JUDGE", "shell:from-env")
    monkeypatch.setenv("TALKINGHEAD_TRANSCRIBER", "parakeet")
    cfg = load_config(f)
    assert (cfg.judge, cfg.transcriber) == ("shell:from-env", "parakeet")
    cfg = load_config(f, judge="shell:from-flag", transcriber=None)
    assert (cfg.judge, cfg.transcriber) == ("shell:from-flag", "parakeet")


def test_an_empty_environment_variable_counts_as_unset(monkeypatch) -> None:
    monkeypatch.setenv("TALKINGHEAD_JUDGE", "")
    assert load_config().judge is None


def test_the_judge_and_transcriber_have_no_default_without_a_file_or_environment() -> None:
    cfg = load_config()
    assert cfg.judge is None and cfg.transcriber is None

