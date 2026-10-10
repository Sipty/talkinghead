"""The plan card: a readable roadmap with the current chapter highlighted."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .cards import FONT, PALETTE, avatar_data_uri, avatar_fingerprint, wrap

WIDTH = 1080
PAD_X = 46
ROW_H = 92
HEAD_H = 104
FOOT_H = 74
NUM_W = 62                 # width of the number column
TEXT_W = WIDTH - PAD_X * 2 - NUM_W - 14
ITEM_SIZE = 43
PLAN_CARD_VERSION = 3


@dataclass(frozen=True)
class PlanItem:
    text: str
    current: bool = False


def _esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fit(text: str) -> tuple[list[str], int]:
    for size in (ITEM_SIZE, 40, 37, 34, 31):
        lines = wrap(text, size, max_lines=2, bold=True, width=TEXT_W)
        if lines:
            return lines, size
    return [text], 31


def plan_svg(items: list[PlanItem], *, title: str = "THE PLAN",
             footer: str | None = None, avatar: str | None = None) -> str:
    rows = [_fit(i.text) for i in items]
    heights = [max(ROW_H, int(len(lines) * s * 1.22) + 34) for lines, s in rows]
    height = HEAD_H + sum(heights) + (FOOT_H if footer else 18)
    uri = avatar_data_uri(avatar)

    p: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}">',
        '<defs>',
        f'<clipPath id="avclip"><circle cx="{WIDTH - PAD_X - 34}" cy="54" r="34"/></clipPath>',
        '</defs>',
        f'<rect x="4" y="4" width="{WIDTH - 8}" height="{height - 8}" rx="30" '
        f'fill="{PALETTE["background"]}" stroke="{PALETTE["border"]}" stroke-width="3"/>',
    ]
    # Header: title, accent rule, and the avatar as a small badge when one is set.
    p.append(f'<text x="{PAD_X}" y="62" font-family="{FONT}" font-size="32" '
             f'font-weight="700" letter-spacing="4" fill="{PALETTE["accent"]}">{_esc(title)}</text>')
    if uri:
        cx = WIDTH - PAD_X - 34
        p.append(f'<circle cx="{cx}" cy="54" r="34" fill="{PALETTE["foreground"]}"/>')
        p.append(f'<image x="{cx - 34}" y="20" width="68" height="68" xlink:href="{uri}" '
                 f'clip-path="url(#avclip)" preserveAspectRatio="xMidYMid meet"/>')
        p.append(f'<circle cx="{cx}" cy="54" r="34" fill="none" '
                 f'stroke="{PALETTE["accent"]}" stroke-width="4"/>')
    p.append(f'<rect x="{PAD_X}" y="84" width="{WIDTH - PAD_X * 2}" height="2" '
             f'fill="{PALETTE["border"]}"/>')

    y = HEAD_H
    for n, (item, (lines, size)) in enumerate(zip(items, rows), start=1):
        rh = heights[n - 1]
        mid = y + rh / 2
        current = "true" if item.current else "false"
        p.append(f'<g data-current="{current}">')
        if item.current:
            p.append(f'<rect x="{PAD_X - 16}" y="{y + 7}" '
                     f'width="{WIDTH - PAD_X * 2 + 32}" height="{rh - 14}" rx="18" '
                     f'fill="{PALETTE["accent"]}" opacity="0.14"/>')
        num_fill = PALETTE["accent"] if item.current else PALETTE["muted_fg"]
        p.append(f'<text x="{PAD_X}" y="{mid + size * 0.34:.1f}" font-family="{FONT}" '
                 f'font-size="{size}" font-weight="700" fill="{num_fill}" '
                 f'opacity="{1 if item.current else 0.72}">{n}</text>')
        tx = PAD_X + NUM_W
        ty = mid - (len(lines) - 1) * size * 0.61 + size * 0.34
        fill = PALETTE["foreground"] if item.current else PALETTE["muted_fg"]
        for ln in lines:
            p.append(f'<text x="{tx}" y="{ty:.1f}" font-family="{FONT}" font-size="{size}" '
                     f'font-weight="600" fill="{fill}">{_esc(ln)}</text>')
            ty += size * 1.22
        p.append('</g>')
        if n < len(items):
            p.append(f'<rect x="{PAD_X}" y="{y + rh - 1}" width="{WIDTH - PAD_X * 2}" '
                     f'height="1.5" fill="{PALETTE["border"]}" opacity="0.55"/>')
        y += rh

    if footer:
        p.append(f'<text x="{PAD_X}" y="{height - FOOT_H + 44}" font-family="{FONT}" '
                 f'font-size="31" fill="{PALETTE["muted_fg"]}">{_esc(footer)}</text>')
    p.append('</svg>')
    return "".join(p)


def plan_sha(items: list[PlanItem], title: str, footer: str | None, avatar: str | None = None) -> str:
    payload = "|".join([f"v{PLAN_CARD_VERSION}", title, footer or ""]
                       + [f"{i.current:d}{i.text}" for i in items])
    fingerprint = avatar_fingerprint(avatar)
    if fingerprint:
        payload += f"|avatar={fingerprint}"
    return "p-" + hashlib.sha256(payload.encode()).hexdigest()[:14]


def render_plan(items: list[PlanItem], out: str | Path, *, title: str = "THE PLAN",
                footer: str | None = None, avatar: str | None = None) -> Path:
    out = Path(out)
    if out.is_file() and out.stat().st_size > 0:
        return out
    if not shutil.which("rsvg-convert"):
        raise RuntimeError("rsvg-convert not found: needed to rasterize the plan card")
    out.parent.mkdir(parents=True, exist_ok=True)
    svg = out.with_suffix(".svg")
    svg.write_text(plan_svg(items, title=title, footer=footer, avatar=avatar))
    proc = subprocess.run(["rsvg-convert", "-w", str(WIDTH), "-o", str(out), str(svg)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"rsvg-convert failed: {proc.stderr.strip()}")
    return out
