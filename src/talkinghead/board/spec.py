"""Board-video specification parsing, validation, and durable editing."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BoardView:
    start: float
    end: float
    rect: tuple[float, float, float, float]
    label: str = ""



@dataclass(frozen=True)
class BoardSpec:
    version: int
    tablet: str
    offset: float
    canvas: tuple[int, int, int, int]
    masks: tuple[tuple[int, int, int, int], ...]
    panel: tuple[float, float, float, float]
    transition: float
    views: tuple[BoardView, ...] = field(default_factory=tuple)
    notes: str = ""
    appear: float | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version, "tablet": self.tablet, "offset": self.offset,
            "canvas": list(self.canvas), "masks": [list(m) for m in self.masks],
            "panel": list(self.panel), "transition": self.transition,
            "views": [{"start": v.start, "end": v.end, "rect": list(v.rect), "label": v.label}
                      for v in self.views], "notes": self.notes,
            "appear": self.appear,
        }



def _number(value: Any, name: str, *, integer: bool = False) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if integer and int(value) != value:
        raise ValueError(f"{name} values must be integers")
    return int(value) if integer else float(value)


def _tuple4(value: Any, name: str, *, integer: bool = False) -> tuple:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{name} must contain exactly four values")
    return tuple(_number(v, f"{name}[{i}]", integer=integer) for i, v in enumerate(value))


def validate(data: dict) -> BoardSpec:
    """Validate schema v1 and return immutable, normalized values."""
    if not isinstance(data, dict):
        raise ValueError("board spec must be a JSON object")
    if "ink" in data:
        raise ValueError("unknown board spec field: ink")
    version = data.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ValueError("version must be 1")
    tablet = data.get("tablet")
    if not isinstance(tablet, str) or not tablet.strip():
        raise ValueError("tablet must be a non-empty path")
    offset = _number(data.get("offset"), "offset")
    canvas = _tuple4(data.get("canvas"), "canvas", integer=True)
    if canvas[0] < 0 or canvas[1] < 0:
        raise ValueError("canvas x and y must be non-negative")
    if canvas[2] <= 0 or canvas[3] <= 0:
        raise ValueError("canvas width and height must be positive")
    appear_data = data.get("appear")
    appear = None if appear_data is None else _number(appear_data, "appear")
    if appear is not None and appear < 0:
        raise ValueError("appear must be non-negative or null")
    masks_data = data.get("masks", [])
    if not isinstance(masks_data, list):
        raise ValueError("masks must be a list")
    masks = tuple(_tuple4(m, f"masks[{i}]", integer=True) for i, m in enumerate(masks_data))
    if any(w <= 0 or h <= 0 for _, _, w, h in masks):
        raise ValueError("mask widths and heights must be positive")
    panel = _tuple4(data.get("panel"), "panel")
    x, y, w, h = panel
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > 1 or y + h > 1:
        raise ValueError("panel must have positive size and lie inside the 0-1 frame")
    transition = _number(data.get("transition", 0.8), "transition")
    if transition < 0:
        raise ValueError("transition must not be negative")
    views_data = data.get("views", [])
    if not isinstance(views_data, list):
        raise ValueError("views must be a list")
    views: list[BoardView] = []
    previous_end = -math.inf
    for i, item in enumerate(views_data):
        if not isinstance(item, dict):
            raise ValueError(f"views[{i}] must be an object")
        start = _number(item.get("start"), f"views[{i}].start")
        end = _number(item.get("end"), f"views[{i}].end")
        if start < 0 or end <= start:
            raise ValueError(f"views[{i}] must have start >= 0 and end > start")
        if start < previous_end:
            raise ValueError("views must be sorted and non-overlapping")
        rect = _tuple4(item.get("rect"), f"views[{i}].rect")
        rx, ry, rw, rh = rect
        if rw <= 0 or rh <= 0 or rx < 0 or ry < 0 or rx + rw > 1 or ry + rh > 1:
            raise ValueError(f"views[{i}].rect must have positive size and lie inside 0-1")
        label = item.get("label", "")
        if not isinstance(label, str):
            raise ValueError(f"views[{i}].label must be a string")
        views.append(BoardView(start, end, rect, label))
        previous_end = end
    notes = data.get("notes", "")
    if not isinstance(notes, str):
        raise ValueError("notes must be a string")
    return BoardSpec(1, tablet, offset, canvas, masks, panel, transition, tuple(views), notes,
                     appear)


def load_spec(path: str | Path) -> BoardSpec:
    try:
        data = json.loads(Path(path).read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"cannot read board spec {path}: {exc}") from exc
    return validate(data)


def save_spec(spec: BoardSpec, path: str | Path) -> Path:
    """Atomically replace a spec, retaining the previous valid file as .bak."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(spec.to_json(), indent=2, ensure_ascii=False) + "\n"
    if dest.is_file():
        backup = dest.with_suffix(dest.suffix + ".bak")
        tmp_backup = backup.with_suffix(backup.suffix + ".tmp")
        tmp_backup.write_bytes(dest.read_bytes())
        os.replace(tmp_backup, backup)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dest.name}.", dir=dest.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, dest)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return dest
