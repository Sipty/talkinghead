"""Head-clip check: overlay rect math, sampling, and the clip decision.

The rect math must mirror build_overlay_command exactly -- a drift there
checks a different rectangle than the bake paints. The detector itself is
Apple Vision and only proven on real footage (see the run reports).
"""

import subprocess
from pathlib import Path

import pytest

from talkinghead.visuals import headclip
from talkinghead.visuals.headclip import overlay_rect, sample_times, summarize
from talkinghead.visuals.schedule import Placement


def _png(path: Path, w: int = 64, h: int = 48) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", f"color=c=red:s={w}x{h}:d=0.1", "-frames:v", "1", str(path)],
        check=True,
    )
    return path


def test_overlay_rect_mirrors_the_renderer_per_corner(tmp_path: Path) -> None:
    card = _png(tmp_path / "card.png", w=64, h=48)  # 0.75 aspect
    p = Placement("a", str(card), 5.0, 9.0, 0.5)
    # 1920x1080 base: w = even(0.5*1920) = 960, h = 960*48/64 = 720.
    assert overlay_rect(p, 1920, 1080, 40, "top-right") == (1920 - 960 - 40, 40, 960, 720)
    assert overlay_rect(p, 1920, 1080, 40, "top-left") == (40, 40, 960, 720)
    assert overlay_rect(p, 1920, 1080, 40, "bottom-right") == (1920 - 960 - 40, 1080 - 720 - 40, 960, 720)
    assert overlay_rect(p, 1920, 1080, 40, "bottom-left") == (40, 1080 - 720 - 40, 960, 720)


def test_overlay_rect_snaps_width_to_even(tmp_path: Path) -> None:
    card = _png(tmp_path / "card.png")
    p = Placement("a", str(card), 0.0, 4.0, 0.335)  # 0.3*1920 = 643.2 -> 644
    x, y, w, h = overlay_rect(p, 1920, 1080, 40, "top-right")
    assert w % 2 == 0 and w == 644


def test_sample_times_stays_off_the_fade_edges() -> None:
    p = Placement("a", "x.png", 10.0, 14.0, 0.4)
    times = sample_times(p)
    assert times[0] >= 10.2 and times[-1] <= 14.0 - 0.2
    # Windows shorter than one step still get a sample: the midpoint.
    short = sample_times(Placement("a", "x.png", 20.0, 20.4, 0.4))
    assert short == [20.2]


def test_summarize_requires_persistence_not_a_flicker() -> None:
    # One noisy sample above threshold: not clipped.
    assert summarize([0.001, 0.03, 0.001]) == (False, 0.03, 1)
    # A real lean-in holds the threshold across samples.
    assert summarize([0.03, 0.05, 0.02]) == (True, 0.05, 2)
    # Clean window: nothing over threshold at all.
    assert summarize([0.0, 0.004]) == (False, 0.004, 0)


def test_a_missing_swiftc_is_a_runtime_error(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(headclip, "HELPER_CACHE", tmp_path / "cache")
    monkeypatch.setattr(headclip.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="swiftc not found"):
        headclip._helper_path()


def test_off_macos_the_head_check_is_skipped_with_a_warning(
        monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    # Apple Vision is the only detector: elsewhere the cook must go on, loudly, without touching the video.
    monkeypatch.setattr("sys.platform", "linux")
    placements = [Placement("a", "x.png", 1.0, 3.0, 0.3)]
    result, _facts = headclip.check_head_clip("missing.mp4", placements, margin=40, corner="top-right",
                                              work_dir=tmp_path)
    assert result.severity == "warn" and result.ok
    assert result.detail.startswith("skipped: head-coverage check needs macOS")
    assert "needs macOS" in capsys.readouterr().err