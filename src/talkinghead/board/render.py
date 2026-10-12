"""Render a synchronised tablet canvas card over the verified talking-head cut."""

from __future__ import annotations

import json
import queue
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from ..color import HDR_BITRATE, encode_args, probe_color, write_sdr_to_hdr_cube
from ..timeline import Segment, in_source_order
from ..verify import display_size
from .spec import BoardSpec, BoardView


def source_at_output(t: float, segments: Iterable[Segment]) -> float | None:
    """Map a concatenated output instant to its source instant."""
    offset = 0.0
    for segment in segments:
        if offset <= t < offset + segment.length:
            return segment.start + t - offset
        offset += segment.length
    return None


def appear_alpha(source: float, appear: float | None, duration: float = 0.4) -> float:
    """Return panel opacity at a source-timeline instant."""
    if appear is None:
        return 1.0
    if source < appear:
        return 0.0
    if duration <= 0:
        return 1.0
    return ease((source - appear) / duration)



def map_view_to_output(view: BoardView, segments: Iterable[Segment]) -> list[BoardView]:
    """Project a source-time view through cuts, splitting at removed regions."""
    result = []
    output_start = 0.0
    for segment in segments:
        start, end = max(view.start, segment.start), min(view.end, segment.end)
        if end > start:
            result.append(BoardView(output_start + start - segment.start,
                                    output_start + end - segment.start,
                                    view.rect, view.label))
        output_start += segment.length
    return result
def expand_rect(rect: tuple[float, float, float, float], panel_aspect: float,
                canvas_aspect: float = 1.0) -> tuple[float, float, float, float]:
    """Expand a canvas-normalised rect to panel aspect, shifting it inside 0-1."""
    x, y, w, h = rect
    if panel_aspect <= 0 or canvas_aspect <= 0:
        raise ValueError("panel and canvas aspects must be positive")
    aspect = w * canvas_aspect / h
    if aspect > panel_aspect:
        nh = min(1.0, w * canvas_aspect / panel_aspect)
        ny = min(max(0.0, y + h / 2 - nh / 2), 1 - nh)
        y, h = ny, nh
    else:
        nw = min(1.0, h * panel_aspect / canvas_aspect)
        nx = min(max(0.0, x + w / 2 - nw / 2), 1 - nw)
        x, w = nx, nw
    return x, y, w, h


def ease(progress: float) -> float:
    """Monotonic smoothstep with exact endpoint values."""
    p = min(1.0, max(0.0, progress))
    return p * p * (3.0 - 2.0 * p)


def _mix_rect(a: tuple[float, float, float, float], b: tuple[float, float, float, float], amount: float):
    return tuple(x + (y - x) * amount for x, y in zip(a, b))


def view_rect_at(t: float, views: tuple[BoardView, ...], transition: float):
    """Return the eased canvas-normalised view at a source-timeline instant."""
    full = (0.0, 0.0, 1.0, 1.0)
    if t <= 0:
        return full
    for i, view in enumerate(views):
        fade = min(transition, (view.end - view.start) / 2)
        if i == 0:
            previous, enter_start = full, view.start - fade
            if view.start == 0 and transition > 0 and t < fade:
                return _mix_rect(full, view.rect, ease(t / max(1e-9, fade)))
        else:
            prior = views[i - 1]
            prior_fade = min(transition, (prior.end - prior.start) / 2)
            gap = view.start - prior.end
            if gap <= 2 * transition:
                previous, enter_start = view.rect, view.start
            else:
                previous = full
                enter_start = max(prior.end + prior_fade, view.start - fade)
        next_view = views[i + 1] if i + 1 < len(views) else None
        following = next_view.rect if next_view and next_view.start - view.end <= 2 * transition else full
        if transition > 0 and enter_start <= t < view.start:
            return _mix_rect(previous, view.rect, ease((t - enter_start) / max(1e-9, view.start - enter_start)))
        if i and gap <= 2 * transition and views[i - 1].end <= t < view.start:
            return view.rect
        if view.start <= t < view.end:
            if transition > 0 and t >= view.end - fade:
                return _mix_rect(view.rect, following, ease((t - (view.end - fade)) / max(1e-9, fade)))
            return view.rect
        if transition > 0 and view.end <= t < view.end + fade:
            return following
    return full


def _probe(path: Path, entries: str) -> dict:
    proc = subprocess.run(["ffprobe", "-v", "error", "-show_entries", entries,
                           "-of", "json", str(path)], capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"ffprobe failed for {path}: {proc.stderr.strip()}")
    return json.loads(proc.stdout)

def _tablet_frames(path: Path, canvas: tuple[int, int, int, int]):
    """Yield cropped canvas frames and presentation timestamps from one decode."""
    x, y, width, height = canvas
    cmd = ["ffmpeg", "-v", "info", "-nostdin", "-i", str(path), "-map", "0:v:0",
           "-vf", f"crop={width}:{height}:{x}:{y},showinfo", "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None and process.stderr is not None
    frame_bytes = width * height * 3
    timestamps: queue.Queue[float | None] = queue.Queue()
    errors: list[str] = []

    def read_log():
        pattern = re.compile(rb"pts_time:([-+0-9.eE]+)")
        for line in process.stderr:
            match = pattern.search(line)
            if match:
                timestamps.put(float(match.group(1)))
            elif b"Error" in line or b"error" in line:
                errors.append(line.decode(errors="replace"))
        timestamps.put(None)

    def iterator():
        log_thread = threading.Thread(target=read_log, daemon=True)
        log_thread.start()
        try:
            while True:
                raw = process.stdout.read(frame_bytes)
                if not raw:
                    break
                if len(raw) != frame_bytes:
                    raise RuntimeError("tablet decoder returned an incomplete frame")
                ts = timestamps.get()
                if ts is None:
                    raise RuntimeError("tablet decoder produced a frame without a presentation timestamp")
                yield ts, np.frombuffer(raw, np.uint8).reshape((height, width, 3))
            process.stdout.close()
            log_thread.join()
            code = process.wait()
            if code:
                raise RuntimeError(f"tablet decode failed: {''.join(errors[-20:])}")
        finally:
            if process.poll() is None:
                process.stdout.close()
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if log_thread.is_alive():
                log_thread.join(timeout=5)
            if log_thread.is_alive():
                process.stderr.close()
                log_thread.join(timeout=1)
    return iterator(), cmd





def _sample_grid(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x0, y0 = xs.astype(np.int32), ys.astype(np.int32)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (xs - x0)[None, :, None], (ys - y0)[:, None, None]
    top = image[y0[:, None], x0[None, :]].astype(np.float32) * (1 - fx) + image[y0[:, None], x1[None, :]].astype(np.float32) * fx
    bottom = image[y1[:, None], x0[None, :]].astype(np.float32) * (1 - fx) + image[y1[:, None], x1[None, :]].astype(np.float32) * fx
    return np.clip(top * (1 - fy) + bottom * fy, 0, 255).astype(np.uint8)

def _sdr_to_hdr_rgb(image: np.ndarray, transfer: str) -> np.ndarray:
    """Apply color.py's SDR graphics-white transform to cached panel pixels."""
    rgb = image.astype(np.float32) / 255.0
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    matrix = np.asarray(((0.6274039, 0.32928304, 0.04331307),
                         (0.06909729, 0.9195404, 0.01136232),
                         (0.01639144, 0.08801331, 0.89559525)), dtype=np.float32)
    display = linear @ matrix.T
    if transfer == "arib-std-b67":
        scaled = np.maximum(display, 0) * (203.0 / 1000.0)
        luma = scaled @ np.asarray((0.2627, 0.6780, 0.0593), dtype=np.float32)
        scene = np.zeros_like(scaled)
        np.divide(scaled, np.maximum(luma[..., None], 1e-20) ** (1 / 6), out=scene,
                  where=luma[..., None] > 0)
        signal = np.empty_like(scene)
        low = scene <= 1 / 12
        signal[low] = np.sqrt(3 * scene[low])
        high = ~low
        signal[high] = 0.17883277 * np.log(12 * scene[high] - 0.28466892) + 0.55991073
    elif transfer == "smpte2084":
        relative = np.maximum(display, 0) * (203.0 / 10000.0)
        m = relative ** 0.1593017578125
        signal = ((0.8359375 + 18.8515625 * m) / (1 + 18.6875 * m)) ** 78.84375
    else:
        raise ValueError(f"no SDR-to-HDR transform for {transfer!r}")
    return np.clip(np.rint(signal * 255.0), 0, 255).astype(np.uint8)


def _prepare_canvas(frame: np.ndarray, spec: BoardSpec) -> tuple[np.ndarray, np.ndarray]:
    cx, cy, cw, ch = spec.canvas
    if frame.shape[:2] != (ch, cw):
        raise ValueError("canvas crop does not match tablet frame")
    crop = frame.copy()
    edge = np.concatenate([crop[:4].reshape(-1, 3), crop[-4:].reshape(-1, 3),
                           crop[:, :4].reshape(-1, 3), crop[:, -4:].reshape(-1, 3)])
    paper = np.median(edge, axis=0).astype(np.uint8)
    for mx, my, mw, mh in spec.masks:
        x0, y0 = max(0, mx - cx), max(0, my - cy)
        x1, y1 = min(cw, x0 + mw), min(ch, y0 + mh)
        if x1 > x0 and y1 > y0:
            crop[y0:y1, x0:x1] = paper
    return crop, paper


def _canvas_changed(previous: np.ndarray, current: np.ndarray) -> bool:
    delta = np.abs(current.astype(np.int16) - previous.astype(np.int16)).max(axis=2)
    return np.count_nonzero(delta > 20) > 32


def _card(canvas: np.ndarray, paper: np.ndarray, rect: tuple[float, float, float, float],
          panel_size: tuple[int, int]) -> np.ndarray:
    ch, cw = canvas.shape[:2]
    pw, ph = panel_size
    x, y, rw, rh = expand_rect(rect, pw / ph, cw / ch)
    source_aspect = rw * cw / (rh * ch)
    if source_aspect > pw / ph:
        out_w, out_h = pw, max(1, round(pw / source_aspect))
    else:
        out_h, out_w = ph, max(1, round(ph * source_aspect))
    xs = (x + (np.arange(out_w, dtype=np.float32) + 0.5) * rw / out_w) * (cw - 1)
    ys = (y + (np.arange(out_h, dtype=np.float32) + 0.5) * rh / out_h) * (ch - 1)
    image = np.broadcast_to(paper, (ph, pw, 3)).copy()
    ox, oy = (pw - out_w) // 2, (ph - out_h) // 2
    image[oy:oy + out_h, ox:ox + out_w] = _sample_grid(canvas, xs, ys)
    return image


def _frame_times(segments: list[Segment], fps: float, count: int):
    for i in range(count):
        t = i / fps
        yield t, source_at_output(t, segments)


def render_board(run_dir: Path, spec: BoardSpec, out: Path | None = None) -> Path:
    """Composite the board panel while preserving base audio and HDR colour."""
    started = time.monotonic()
    run_dir = Path(run_dir)
    base = run_dir / "final.mp4"
    if not base.is_file():
        raise FileNotFoundError(f"verified base cut not found: {base}")
    tablet = Path(spec.tablet)
    if not tablet.is_file():
        raise FileNotFoundError(f"tablet recording not found: {tablet}")
    output = Path(out) if out is not None else run_dir / "final-board.mp4"
    if output.resolve() == base.resolve():
        raise ValueError("board output must not overwrite final.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    board_dir = run_dir / "board"
    board_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads((run_dir / "plan.json").read_text())
    segments = [Segment(**s) for s in plan["segments"]]
    if not segments:
        raise ValueError("plan.json has no kept segments")
    if not in_source_order(segments):
        raise ValueError("board overlays need the cut in source order: this plan moves topic blocks "
                         "ahead of earlier material (restructure), so the tablet timeline would run "
                         "backwards. Render the board on a cut without restructure reordering.")
    base_info = _probe(base, "stream=avg_frame_rate,nb_read_frames,duration:format=duration")
    stream = base_info["streams"][0]
    width, height = display_size(str(base))
    num, den = map(int, stream["avg_frame_rate"].split("/"))
    fps = num / den
    duration = float(base_info.get("format", {}).get("duration") or stream.get("duration"))
    count = int(stream.get("nb_read_frames") or round(duration * fps))
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("cannot determine base video dimensions and frame rate")
    px, py, pw, ph = spec.panel
    panel_x, panel_y = round(px * width), round(py * height)
    panel_x -= panel_x % 2
    panel_y -= panel_y % 2
    panel_w, panel_h = max(2, round(pw * width)), max(2, round(ph * height))
    panel_w += panel_w % 2
    panel_h += panel_h % 2
    color = probe_color(base)
    tablet_info = _probe(tablet, "stream=width,height")
    tablet_width, tablet_height = int(tablet_info["streams"][0]["width"]), int(tablet_info["streams"][0]["height"])
    rotation = _probe(tablet, "stream=width,height:stream_side_data=rotation")
    if rotation.get("streams", [{}])[0].get("side_data_list"):
        angle = int(round(float(rotation["streams"][0]["side_data_list"][0].get("rotation", 0))))
        if abs(angle) in (90, 270):
            tablet_width, tablet_height = tablet_height, tablet_width
    canvas_x, canvas_y, canvas_w, canvas_h = spec.canvas
    if canvas_x + canvas_w > tablet_width or canvas_y + canvas_h > tablet_height:
        raise ValueError("canvas crop lies outside tablet display frame")
    frames, tablet_command = _tablet_frames(tablet, spec.canvas)
    current = next(frames, None)
    if current is None:
        raise RuntimeError("tablet recording decoded no frames")
    frame_ts, current_frame = current
    tablet_start = frame_ts
    current_canvas, current_paper = _prepare_canvas(current_frame, spec)
    frame_iter = iter(frames)
    pad = 12
    video_w, video_h = panel_w + 2 * pad, panel_h + 2 * pad
    if color.hdr:
        write_sdr_to_hdr_cube(board_dir / "sdr_to_hdr.cube", color)
    x, y = panel_x - pad, panel_y - pad
    if color.hdr:
        graph_parts = ["[0:v]format=p010le[base]",
                       f"[1:v]format=rgba,scale=out_color_matrix={color.matrix}:out_range={color.range},format=yuva420p10le[panel]",
                       f"[base][panel]overlay=x={x}:y={y}:format=yuv420p10:shortest=1[vout]"]
    else:
        graph_parts = ["[1:v]format=rgba[panel]",
                       f"[0:v][panel]overlay=x={x}:y={y}:format=yuv420:shortest=1[vout]"]
    cmd = ["ffmpeg", "-y", "-v", "error", "-nostdin", "-i", str(base),
           "-f", "rawvideo", "-pixel_format", "rgba", "-video_size", f"{video_w}x{video_h}",
           "-framerate", f"{fps:.12g}", "-i", "pipe:0",
           "-filter_complex", ";".join(graph_parts), "-map", "[vout]", "-map", "0:a?",
           *encode_args(color, crf=18, preset="medium", bitrate=HDR_BITRATE),
           "-c:a", "copy", "-movflags", "+faststart", str(output)]
    encoder = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert encoder.stdin is not None
    yy, xx = np.ogrid[:video_h, :video_w]
    cx, cy = xx - pad, yy - pad
    radius = max(4, int(min(panel_w, panel_h) * 0.035))
    dx = np.maximum(np.maximum(radius - cx, 0), cx - (panel_w - radius - 1))
    dy = np.maximum(np.maximum(radius - cy, 0), cy - (panel_h - radius - 1))
    outside = np.sqrt(dx * dx + dy * dy)
    coverage = (cx >= 0) & (cx < panel_w) & (cy >= 0) & (cy < panel_h)
    card_alpha = np.where(coverage, np.clip((radius + 0.5 - outside) * 255, 0, 255), 0)
    shadow_alpha = np.where(outside <= radius + pad,
                            np.clip((radius + pad - outside) * 80 / pad, 0, 80), 0)
    alpha = np.maximum(card_alpha, shadow_alpha).astype(np.uint8)
    rgba = np.zeros((video_h, video_w, 4), dtype=np.uint8)
    rgba[..., 3] = alpha
    tablet_frame_id = 0
    pending = None
    cached_key = None
    failure: Exception | None = None
    try:
        for _, source_t in _frame_times(segments, fps, count):
            if source_t is None:
                source_t = segments[-1].end
            wanted = source_t + spec.offset
            if wanted >= tablet_start:
                while True:
                    nxt = pending if pending is not None else next(frame_iter, None)
                    pending = None
                    if nxt is None:
                        break
                    if nxt[0] > wanted:
                        pending = nxt
                        break
                    frame_ts, next_frame = nxt
                    next_canvas, next_paper = _prepare_canvas(next_frame, spec)
                    if _canvas_changed(current_canvas, next_canvas):
                        current_frame = next_frame
                        current_canvas, current_paper = next_canvas, next_paper
                        tablet_frame_id += 1
            rect = view_rect_at(source_t, spec.views, spec.transition)
            rect_key = tuple(round(value, 6) for value in rect)
            key = (tablet_frame_id, rect_key)
            if key != cached_key:
                card = _card(current_canvas, current_paper, rect, (panel_w, panel_h))
                if color.hdr:
                    card = _sdr_to_hdr_rgb(card, color.transfer)
                rgba[pad:pad + panel_h, pad:pad + panel_w, :3] = card
                cached_key = key
            rgba[..., 3] = (alpha * appear_alpha(source_t, spec.appear)).astype(np.uint8)
            encoder.stdin.write(rgba.tobytes())
    except (RuntimeError, OSError, ValueError) as exc:
        failure = exc
    finally:
        try:
            encoder.stdin.close()
        except BrokenPipeError as exc:
            failure = failure or exc
        frame_iter.close()
    error = encoder.stderr.read().decode(errors="replace") if encoder.stderr else ""
    code = encoder.wait()
    if failure:
        raise RuntimeError(f"board render failed: {failure}\n{error[-4000:]}") from failure
    if code:
        raise RuntimeError(f"board render failed (exit {code}):\n{error[-4000:]}")
    elapsed = time.monotonic() - started
    commands = {"tablet_decode": tablet_command, "composite": cmd}
    report = {"spec": spec.to_json(), "commands": commands,
              "timings": {"seconds": round(elapsed, 3), "base_duration": duration,
                          "realtime_factor": round(duration / elapsed, 3), "base_frames": count, "fps": fps}}
    (board_dir / "render.json").write_text(json.dumps(report, indent=2) + "\n")
    return output
