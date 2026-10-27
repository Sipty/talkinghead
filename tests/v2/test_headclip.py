"""Head-clip check: overlay rect math, sampling, and the clip decision.

The rect math must mirror build_overlay_command exactly -- a drift there
checks a different rectangle than the bake paints. The detector itself is
Apple Vision and only proven on real footage (see the run reports).
"""

import subprocess
from pathlib import Path

import pytest

from talkinghead.render import RenderSettings, render_picture
from talkinghead.timeline import Plan, Segment, SourceInfo
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

# --- check_head_clip_plan: the same gate on source frames, no picture needed -------------------------------------

def _moving_plan(tmp_path: Path):
    """A two-cut plan of a moving source, with a card placed across the second cut."""
    src = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x240:r=25:d=6",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "5", str(src)],
        check=True,
    )
    plan = Plan(sources={"main": SourceInfo("main", str(src), "x", 6.0)}, main="main", cuts=[],
                segments=[Segment("main", 0.0, 2.0), Segment("main", 3.5, 5.5)])
    placements = [Placement("a", str(_png(tmp_path / "card.png")), 0.4, 3.6, 0.3)]
    return plan, placements


def _capture(monkeypatch, coverage=lambda t: 0.0) -> list[dict]:
    """Pretend to be macOS with a detector that records what it is handed and answers per output time."""
    seen: list[dict] = []

    def fake(items: list[dict]) -> dict[str, float]:
        seen.extend(items)
        return {it["frame"]: coverage(it["t"]) for it in items}

    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr(headclip, "_helper_path", lambda: Path("detector"))
    monkeypatch.setattr(headclip, "_coverages", fake)
    return seen


def _psnr(a: str, b: str) -> float:
    proc = subprocess.run(["ffmpeg", "-v", "info", "-i", a, "-i", b, "-lavfi", "psnr", "-f", "null", "-"],
                          capture_output=True, text=True, check=True)
    value = proc.stderr.split("average:")[1].split()[0]
    return float("inf") if value == "inf" else float(value)


def _render_frame(video: Path, t: float, out: Path) -> str:
    assert headclip._grab(video, t, out) is not None
    return str(out)


def test_the_plan_check_hands_the_detector_the_same_frames_as_the_render(monkeypatch, tmp_path: Path) -> None:
    plan, placements = _moving_plan(tmp_path)
    settings = RenderSettings(crf=18, preset="ultrafast", fps=25)
    render = tmp_path / "picture.mp4"
    render_picture(plan, str(render), settings)
    seen = _capture(monkeypatch)
    headclip.check_head_clip_plan(plan, placements, settings=settings, margin=10, corner="top-right",
                                  work_dir=tmp_path / "work")
    assert len(seen) == len(sample_times(placements[0]))
    for item in seen:
        mine = item["frame"]
        same = _psnr(mine, _render_frame(render, item["t"], tmp_path / "same.jpg"))
        later = _psnr(mine, _render_frame(render, item["t"] + 0.5, tmp_path / "later.jpg"))
        assert same > later, f"t={item['t']}: {same} vs {later}"


def test_the_plan_check_gives_the_same_decision_as_the_check_on_the_render(monkeypatch, tmp_path: Path) -> None:
    plan, placements = _moving_plan(tmp_path)
    settings = RenderSettings(crf=18, preset="ultrafast", fps=25)
    render = tmp_path / "picture.mp4"
    render_picture(plan, str(render), settings)
    _capture(monkeypatch, coverage=lambda t: 0.05 if 1.0 <= t <= 2.2 else 0.0)
    kwargs = {"margin": 10, "corner": "top-right", "work_dir": tmp_path / "work"}
    on_plan, plan_facts = headclip.check_head_clip_plan(plan, placements, settings=settings, **kwargs)
    on_render, render_facts = headclip.check_head_clip(render, placements, **kwargs)
    assert on_plan.ok == on_render.ok is False
    assert on_plan.severity == on_render.severity == "fatal"
    assert on_plan.detail == on_render.detail
    assert [p["clipped_times"] for p in plan_facts["placements"]] == \
        [p["clipped_times"] for p in render_facts["placements"]] != [[]]
    assert (plan_facts["frames"], render_facts["frames"]) == ("source", "render")


def test_the_plan_check_rectangle_is_scaled_to_the_renders_frame_size(monkeypatch, tmp_path: Path) -> None:
    plan, placements = _moving_plan(tmp_path)
    settings = RenderSettings(preset="ultrafast", fps=25, scale="-2:120")
    seen = _capture(monkeypatch)
    headclip.check_head_clip_plan(plan, placements, settings=settings, margin=10, corner="top-left",
                                  work_dir=tmp_path / "work")
    x, y, w, h = overlay_rect(placements[0], 160, 120, 10, "top-left")
    grow = headclip.GRAB_WIDTH / 160
    assert seen and all(it["rect"] == [round(x * grow), round(y * grow), round(w * grow), round(h * grow)]
                        for it in seen)


def test_the_plan_check_grabs_through_an_aspect_changing_scale(monkeypatch, tmp_path: Path) -> None:
    # 320x240 source rendered through scale=160:240 (a 2:3 frame): the grab must have the render's aspect,
    # or the rectangle (render coordinates) would cover a different region of the picture.
    plan, placements = _moving_plan(tmp_path)
    settings = RenderSettings(preset="ultrafast", fps=25, scale="160:240")
    seen = _capture(monkeypatch)
    headclip.check_head_clip_plan(plan, placements, settings=settings, margin=10, corner="top-left",
                                  work_dir=tmp_path / "work")
    assert seen
    for it in seen:
        assert headclip._image_dims(it["frame"]) == (headclip.GRAB_WIDTH, headclip.GRAB_WIDTH * 240 // 160)


def test_the_plan_check_skips_samples_past_the_snapped_end(monkeypatch, tmp_path: Path) -> None:
    plan, _ = _moving_plan(tmp_path)
    late = [Placement("a", str(_png(tmp_path / "late.png")), 3.0, 6.0, 0.3)]
    seen = _capture(monkeypatch)
    headclip.check_head_clip_plan(plan, late, settings=RenderSettings(fps=25), margin=10, corner="top-right",
                                  work_dir=tmp_path / "work")
    assert seen and max(it["t"] for it in seen) <= 4.0 - 0.1


def test_off_macos_the_plan_check_warns_and_grabs_nothing(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr("sys.platform", "linux")

    def refuse(*_a, **_k):
        raise AssertionError("nothing may be grabbed off macOS")

    monkeypatch.setattr(headclip, "_grab", refuse)
    monkeypatch.setattr(headclip, "frame_size", refuse)
    plan = Plan(sources={"main": SourceInfo("main", "missing.mp4", "x", 6.0)}, main="main", cuts=[],
                segments=[Segment("main", 0.0, 2.0)])
    result, facts = headclip.check_head_clip_plan(
        plan, [Placement("a", "x.png", 0.4, 1.6, 0.3)], settings=RenderSettings(fps=25), margin=40,
        corner="top-right", work_dir=tmp_path)
    assert result.severity == "warn" and result.ok
    assert result.detail.startswith("skipped: head-coverage check needs macOS")
    assert facts["frames"] == "source" and "needs macOS" in capsys.readouterr().err
