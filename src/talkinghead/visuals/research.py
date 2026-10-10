"""The research tier: fetch a real reference for a claim made on camera.

Two kinds of source: a technical_evidence item is SCREENSHOTTED (a real page:
docs, pricing, benchmark), while a meme item must become the actual image, so
its URL has to be a direct image and the downloaded bytes are validated by
magic number. A failure is never fatal: the caller drops the anchor.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path
from urllib.request import Request, urlopen

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp")


@dataclass(frozen=True)
class ResearchRequest:
    anchor_text: str
    depicts: str
    url: str


def chrome_fetcher(url: str, out: Path, *, timeout: float = 90.0) -> bool:
    """Screenshot a page with chrome-devtools-axi, if it is installed."""
    exe = shutil.which("chrome-devtools-axi")
    if exe is None:
        return False
    try:
        subprocess.run([exe, "open", url], capture_output=True, text=True, timeout=timeout)
        proc = subprocess.run([exe, "screenshot", "--out", str(out)],
                              capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0 and out.is_file() and out.stat().st_size > 0


def looks_like_image_url(url: str) -> bool:
    """True for a DIRECT image URL (path ends in an image extension)."""
    return url.split("?", 1)[0].lower().endswith(IMAGE_SUFFIXES)


def image_magic_ok(data: bytes) -> bool:
    """PNG / JPEG / GIF / WebP by magic number, so a page cannot pass as one."""
    return (
        data.startswith(b"\x89PNG\r\n\x1a\n")
        or data.startswith(b"\xff\xd8\xff")
        or data.startswith(b"GIF87a")
        or data.startswith(b"GIF89a")
        or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")
    )


def is_gif(data: bytes) -> bool:
    return data.startswith(b"GIF87a") or data.startswith(b"GIF89a")


def rasterize_gif(data: bytes, out: Path) -> bool:
    """Rasterize a GIF's first frame to a real static PNG with ffmpeg.

    render_visuals feeds each overlay through image2 with -loop/-framerate; a
    GIF demuxer can abort there, so a downloaded GIF is normalized to PNG. The
    temp source is always cleaned up.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    try:
        try:
            tmp.write(data)
            tmp.close()
            proc = subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", tmp.name, "-frames:v", "1", str(out)],
                capture_output=True, text=True)
        except OSError:
            return False
        if proc.returncode != 0 or not out.is_file():
            return False
        return out.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def download_image(url: str, out: Path, *, timeout: float = 30.0) -> bool:
    """Download a direct image URL and write it only if it is really an image."""
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    if not image_magic_ok(data):
        return False
    Path(out).write_bytes(data)
    return True


def fetch(request: ResearchRequest, out: str | Path, *, fetcher=chrome_fetcher) -> Path | None:
    out = Path(out)
    if out.is_file() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        ok = fetcher(request.url, out)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if not ok or not out.is_file() or out.stat().st_size == 0:
        if out.is_file():
            out.unlink()
        return None
    return out


def fetch_image(request: ResearchRequest, out: str | Path, *,
                fetcher=download_image, rasterize=rasterize_gif) -> Path | None:
    """Fetch a meme as the ACTUAL static image; a page URL is rejected outright.

    A GIF is normalized to a real PNG (render loops image2, where a GIF can
    abort). A failed download or conversion drops the asset; nothing is left on
    disk.
    """
    out = Path(out)
    if out.is_file() and out.stat().st_size > 0:
        return out
    if not looks_like_image_url(request.url):
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        ok = fetcher(request.url, out)
    except (OSError, ValueError, HTTPException):
        return None
    if not ok or not out.is_file():
        if out.is_file():
            out.unlink()
        return None
    data = out.read_bytes()
    if not image_magic_ok(data):
        out.unlink()
        return None
    if is_gif(data):
        out.unlink()                          # clear the raw GIF first
        if not rasterize(data, out) or not out.is_file():
            if out.is_file():
                out.unlink()
            return None
    return out
