"""Generated cards: the tier that works on any video with no assets at all.

The card is a speech bubble, with an optional avatar beside it, so a generated
visual reads as the speaker talking rather than as a caption bolted on. SVG
rendered by `rsvg-convert`, in the channel palette (PALETTE below).
Output is content-addressed, so a re-cook never re-rasterizes.

Text is FITTED, not hoped for. An earlier cut sized the title from a character
count with three buckets and truncated at 40 characters, which overflowed the
plate and cut words in half ("Claude Superpowers skills repository REA"). Here
the title wraps to at most two lines, the size steps down until the longest
line measures inside the bubble, and the bubble grows to fit.
"""

from __future__ import annotations

import base64
import hashlib
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

PALETTE = {
    "accent": "#e2571b",
    "accent_fg": "#171310",
    "background": "#33281d",
    "foreground": "#f6efe4",
    "muted": "#3e3022",
    "muted_fg": "#bcae9b",
    "border": "#52402c",
}
FONT = "system-ui, -apple-system, 'Helvetica Neue', Arial, sans-serif"

# Canvas geometry. Wider than a plain plate because the avatar rides alongside
# the text; the renderer scales the whole card to visuals_scale_card. With an
# avatar it sits on the RIGHT and the bubble opens to its LEFT, tail pointing at it.
AVATAR = 234                    # avatar circle diameter
GAP = 16                        # avatar to tail tip
PAD_X, PAD_TOP, PAD_BOT = 52, 42, 42
TAIL = 34                       # how far the tail juts out of the bubble
WIDTH = 1360                    # canvas stays FIXED so text scales consistently
MARGIN_R = 8
# The bubble shrinks to its text and the whole group is right-anchored inside
# the fixed canvas, so a two-word title gets a compact plate instead of an
# almost-empty one, and the visible card still hugs the top-right corner.
MAX_BUBBLE_W = WIDTH - MARGIN_R - AVATAR - GAP - TAIL - 8
MIN_BUBBLE_W = 340
TEXT_W = MAX_BUBBLE_W - PAD_X * 2

TITLE_SIZES = (72, 64, 58, 52, 46, 42, 38)
SUB_SIZES = (40, 36, 33, 30)
MAX_TITLE_LINES = 2

# Text metrics, calibrated by rendering strings with rsvg-convert and measuring
# the ink extents (see docs: bold measured 0.482 mean / 0.528 max advance per
# character, regular 0.467 / 0.509). A single flat 0.60 overestimated by up to
# 24%, which showed as dead space to the right of a short title. This splits
# characters into three width classes instead and carries a 6% safety margin,
# because overflowing the plate is a visible defect and a few px of extra
# padding is not.
NARROW_CHARS = frozenset("iljtfrI.,'!|:;()[] ")
WIDE_CHARS = frozenset("WMmw@%&0123456789ABCDEFGHJKLNOPQRSTUVXYZ$")
W_NARROW, W_MID, W_WIDE = 0.30, 0.50, 0.62
BOLD_FACTOR = 1.03
SAFETY = 1.06

# The avatar art runs to the edge of its own frame, so it is inset inside the
# circle rather than clipped by it: a circle crop cuts off outer detail.
# Scale of the avatar image relative to its badge circle, tuned at overlay size.
AVATAR_FIT = 1.15


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


@lru_cache(maxsize=4)
def avatar_data_uri(path: str | None) -> str | None:
    """The avatar image at `path` as a data URI, so the SVG is self-contained; None without one."""
    if not path:
        return None
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def avatar_fingerprint(path: str | None) -> str:
    """Content hash of the avatar for cache keys; empty without one."""
    if not path:
        return ""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def text_width(text: str, size: int, *, bold: bool = False) -> float:
    """Estimated rendered width in px, from the calibrated class model."""
    units = 0.0
    for ch in text:
        if ch in NARROW_CHARS:
            units += W_NARROW
        elif ch in WIDE_CHARS:
            units += W_WIDE
        else:
            units += W_MID
    return units * size * SAFETY * (BOLD_FACTOR if bold else 1.0)


def wrap(text: str, size: int, *, max_lines: int, bold: bool = False,
         width: float = TEXT_W) -> list[str] | None:
    """Greedy word wrap. None when it cannot be made to fit in `max_lines`."""
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    cur = words[0]
    if text_width(cur, size, bold=bold) > width:
        return None                      # a single word already overflows
    for w in words[1:]:
        trial = f"{cur} {w}"
        if text_width(trial, size, bold=bold) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
            if text_width(cur, size, bold=bold) > width:
                return None
            if len(lines) == max_lines:
                return None
    lines.append(cur)
    return lines if len(lines) <= max_lines else None


def fit_title(title: str) -> tuple[list[str], int]:
    """Largest size at which the title wraps into at most two lines."""
    for size in TITLE_SIZES:
        lines = wrap(title, size, max_lines=MAX_TITLE_LINES, bold=True)
        if lines:
            return lines, size
    # Nothing fits: trim on a WORD boundary, never mid-word, and mark it.
    size = TITLE_SIZES[-1]
    words = title.split()
    while words:
        candidate = " ".join(words) + "…"
        lines = wrap(candidate, size, max_lines=MAX_TITLE_LINES, bold=True)
        if lines:
            return lines, size
        words.pop()
    return [title[:1]], size


def fit_subtitle(subtitle: str) -> tuple[str, int]:
    for size in SUB_SIZES:
        if text_width(subtitle, size) <= TEXT_W:
            return subtitle, size
    size = SUB_SIZES[-1]
    words = subtitle.split()
    while words:
        candidate = " ".join(words) + "…"
        if text_width(candidate, size) <= TEXT_W:
            return candidate, size
        words.pop()
    return "", size


def card_svg(title: str, subtitle: str | None, kind: str = "term",
             *, avatar: str | None = None, avatar_fit: float = AVATAR_FIT,
             tail: int = TAIL) -> str:
    title_fill = PALETTE["accent"] if kind == "price" else PALETTE["foreground"]
    lines, tsize = fit_title(title)
    sub, ssize = fit_subtitle(subtitle) if subtitle else ("", 0)
    line_h = int(tsize * 1.16)
    bubble_h = PAD_TOP + line_h * len(lines) + (int(ssize * 1.95) if sub else 0) + PAD_BOT
    bubble_h = max(bubble_h, 150)
    uri = avatar_data_uri(avatar)
    # Without an avatar the bubble stands alone on the right margin: no badge, no tail.
    height = (max(bubble_h, AVATAR) if uri else bubble_h) + 8
    by = (height - bubble_h) / 2                    # bubble top
    cy = height / 2                                 # shared centre line

    # Bubble only as wide as its widest line, opening leftward from the avatar.
    widest = max([text_width(ln, tsize, bold=True) for ln in lines] +
                 ([text_width(sub, ssize)] if sub else [0.0]))
    bubble_w = min(MAX_BUBBLE_W, max(MIN_BUBBLE_W, widest + PAD_X * 2))
    av_cx = WIDTH - MARGIN_R - AVATAR / 2
    bubble_right = (av_cx - AVATAR / 2 - GAP - TAIL) if uri else (WIDTH - MARGIN_R)
    bubble_x = bubble_right - bubble_w
    r = AVATAR / 2 - 5
    side = 2 * r * avatar_fit
    p: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{WIDTH}" height="{int(height)}" viewBox="0 0 {WIDTH} {int(height)}">',
    ]
    if uri:
        th = max(18, int(tail * 0.9))
        p.append(f'<defs><clipPath id="avclip"><circle cx="{av_cx}" cy="{cy}" r="{r}"/></clipPath></defs>')
        # Tail pointing back at the avatar.
        p.append(
            f'<path d="M {bubble_right} {cy - th} L {bubble_right + tail} {cy} '
            f'L {bubble_right} {cy + th} Z" fill="{PALETTE["background"]}" '
            f'stroke="{PALETTE["border"]}" stroke-width="3"/>')
    p.append(
        f'<rect x="{bubble_x}" y="{by}" width="{bubble_w}" height="{bubble_h}" rx="30" '
        f'fill="{PALETTE["background"]}" stroke="{PALETTE["border"]}" stroke-width="3"/>')
    if uri:
        # Cover the seam where the tail meets the plate.
        p.append(f'<rect x="{bubble_right - 7}" y="{cy - th + 2}" width="6" '
                 f'height="{2 * th - 4}" fill="{PALETTE["background"]}"/>')
        p.append(f'<circle cx="{av_cx}" cy="{cy}" r="{r}" fill="{PALETTE["foreground"]}"/>')
        p.append(f'<image x="{av_cx - side / 2}" y="{cy - side / 2}" '
                 f'width="{side}" height="{side}" '
                 f'xlink:href="{uri}" clip-path="url(#avclip)" '
                 f'preserveAspectRatio="xMidYMid meet"/>')
        p.append(f'<circle cx="{av_cx}" cy="{cy}" r="{r}" fill="none" '
                 f'stroke="{PALETTE["accent"]}" stroke-width="7"/>')
    # Text.
    y = by + PAD_TOP + int(tsize * 0.80)
    for ln in lines:
        p.append(f'<text x="{bubble_x + PAD_X}" y="{y}" font-family="{FONT}" '
                 f'font-size="{tsize}" font-weight="600" fill="{title_fill}">{_esc(ln)}</text>')
        y += line_h
    if sub:
        p.append(f'<text x="{bubble_x + PAD_X}" y="{y + int(ssize * 0.45)}" '
                 f'font-family="{FONT}" font-size="{ssize}" '
                 f'fill="{PALETTE["muted_fg"]}">{_esc(sub)}</text>')
    p.append('</svg>')
    return "".join(p)


def card_sha(title: str, subtitle: str | None, kind: str, avatar: str | None = None) -> str:
    """Keyed on the text, kind and avatar content, not on the SVG (which embeds the avatar)."""
    payload = f"v2|{kind}|{title}|{subtitle or ''}"
    fingerprint = avatar_fingerprint(avatar)
    if fingerprint:
        payload += f"|avatar={fingerprint}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def reading_seconds(title: str, subtitle: str | None) -> float:
    """How long a viewer needs to actually read this card.

    An earlier cut let a card sit for 1.5 s, which is not enough to notice it,
    let alone read it. Roughly 2.4 words/second plus a beat to notice the card
    arriving and a beat before it leaves.
    """
    words = len(title.split()) + len(str(subtitle or "").split())
    return round(1.6 + words / 2.4, 2)


def render_card(title: str, subtitle: str | None, kind: str, out: str | Path,
                *, avatar: str | None = None, width: int = WIDTH) -> Path:
    """Rasterize to `out`; a no-op when the file already exists."""
    out = Path(out)
    if out.is_file() and out.stat().st_size > 0:
        return out
    if not shutil.which("rsvg-convert"):
        raise RuntimeError("rsvg-convert not found: needed to rasterize generated cards")
    out.parent.mkdir(parents=True, exist_ok=True)
    svg = out.with_suffix(".svg")
    svg.write_text(card_svg(title, subtitle, kind, avatar=avatar))
    proc = subprocess.run(["rsvg-convert", "-w", str(width), "-o", str(out), str(svg)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"rsvg-convert failed: {proc.stderr.strip()}")
    return out


# Words that describe the ARTEFACT rather than the subject. `depicts` is
# written so the matcher can find an asset ("GLM 5.3 Flash benchmark chart"),
# which makes it a poor card title: falling back to it verbatim produced
# "Claude Superpowers skills repository README screenshot" on screen.
ARTEFACT_WORDS = frozenset("""
screenshot screengrab screencap capture page pages docs documentation doc
terminal output console chart graph diagram card plate image images photo
logo website site view panel window dashboard listing snippet
""".split())
LABEL_MAX_WORDS = 6


def label_from_depicts(depicts: str) -> str:
    """A display label from a search description.

    Keeps the leading subject and stops at the first word that names the
    artefact, so "Claude Code sub-agents documentation page screenshot"
    becomes "Claude Code sub-agents".
    """
    words = depicts.split()
    kept: list[str] = []
    for w in words:
        if w.strip(",.:;()").lower() in ARTEFACT_WORDS:
            break
        kept.append(w)
        if len(kept) >= LABEL_MAX_WORDS:
            break
    if not kept:                      # described only as an artefact
        kept = words[:LABEL_MAX_WORDS]
    return " ".join(kept).strip(" ,.:;-")
