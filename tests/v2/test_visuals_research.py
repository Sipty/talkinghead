import base64
import shutil
from pathlib import Path

import pytest

from talkinghead.prompts import RESEARCH_SCHEMA, research_prompt
from talkinghead.visuals.research import (
    ResearchRequest,
    fetch,
    fetch_image,
    image_magic_ok,
    is_gif,
    looks_like_image_url,
)


def req(url="https://example.com/pricing"):
    return ResearchRequest("Kimi K three is too expensive", "Kimi K3 pricing", url)


def test_prompt_lists_items_with_what_they_depict():
    p = research_prompt([{"type": "technical_evidence", "depicts": "Kimi K3 pricing",
                          "anchor_text": "Kimi K three"}], "agentic coding")
    assert "Kimi K3 pricing" in p and "Kimi K three" in p
    assert "agentic coding" in p
    assert "type=technical_evidence" in p


def test_prompt_asks_a_meme_for_a_direct_image_not_an_official_page():
    p = research_prompt([{"type": "meme", "depicts": "2000s reaction face",
                          "anchor_text": "that is the whole point"}], None)
    assert "type=meme" in p
    low = p.lower()
    assert "direct image url" in low and ".png" in low
    assert "page url is rejected" in low
    assert "official docs page" in p          # still the technical_evidence rule


def test_prompt_tells_the_judge_to_omit_guesses():
    p = research_prompt([{"type": "technical_evidence", "depicts": "x", "anchor_text": "y"}], None)
    assert "Omit an" in p and "404" in p


def test_schema_requires_a_url_per_id():
    items = RESEARCH_SCHEMA["properties"]["sources"]["items"]
    assert set(items["required"]) == {"id", "url", "why"}


def test_failed_fetch_returns_none(tmp_path):
    assert fetch(req(), tmp_path / "x.png", fetcher=lambda u, p: False) is None


def test_fetcher_exception_is_not_fatal(tmp_path):
    def boom(u, p):
        raise OSError("network down")
    assert fetch(req(), tmp_path / "x.png", fetcher=boom) is None


def test_empty_file_counts_as_failure(tmp_path):
    def empty(u, p):
        Path(p).write_bytes(b"")
        return True
    assert fetch(req(), tmp_path / "x.png", fetcher=empty) is None
    assert not (tmp_path / "x.png").is_file()


def test_successful_fetch_returns_path(tmp_path):
    def ok(u, p):
        Path(p).write_bytes(b"\x89PNG\r\n\x1a\n")
        return True
    got = fetch(req(), tmp_path / "x.png", fetcher=ok)
    assert got is not None and got.is_file()


def test_existing_file_is_reused_without_fetching(tmp_path):
    (tmp_path / "x.png").write_bytes(b"\x89PNG")
    calls = []

    def spy(u, p):
        calls.append(u)
        return True
    fetch(req(), tmp_path / "x.png", fetcher=spy)
    assert calls == []


# ------------------------------------------------------ meme image sourcing

def test_only_direct_image_urls_are_candidates():
    assert looks_like_image_url("https://meme.example/face.png")
    assert looks_like_image_url("https://meme.example/face.gif?v=2")
    assert not looks_like_image_url("https://meme.example/face")
    assert not looks_like_image_url("https://meme.example/post/123")


def test_magic_check_rejects_a_page_body():
    assert image_magic_ok(b"\x89PNG\r\n\x1a\n")
    assert image_magic_ok(b"GIF89a...")
    assert not image_magic_ok(b"<!DOCTYPE html><html>")


def test_fetch_image_requires_a_direct_image_url(tmp_path):
    calls = []

    def spy(u, p):
        calls.append(u)
        return True
    assert fetch_image(req("https://meme.example/face"), tmp_path / "x.png", fetcher=spy) is None
    assert calls == []                        # a page is never fetched as a meme


def test_fetch_image_rejects_html_bytes(tmp_path):
    def page(u, p):
        Path(p).write_bytes(b"<!DOCTYPE html><html>nope</html>")
        return True
    assert fetch_image(req("https://meme.example/face.png"), tmp_path / "x.png", fetcher=page) is None
    assert not (tmp_path / "x.png").is_file()


def test_fetch_image_writes_validated_image_bytes(tmp_path):
    png = b"\x89PNG\r\n\x1a\n" + b"pixels"

    def ok(u, p):
        Path(p).write_bytes(png)
        return True
    got = fetch_image(req("https://meme.example/face.png"), tmp_path / "x.png", fetcher=ok)
    assert got is not None and got.read_bytes() == png


def test_fetch_image_normalizes_a_gif_to_a_real_png(tmp_path):
    gif = b"GIF89a" + b"\x00" * 16
    out = tmp_path / "x.png"
    png = b"\x89PNG\r\n\x1a\n" + b"frame"
    seen = {}

    def gif_fetcher(u, p):
        Path(p).write_bytes(gif)
        return True

    def fake_rasterize(data, path):
        seen["gif"] = data
        Path(path).write_bytes(png)
        return True

    got = fetch_image(req("https://meme.example/face.gif"), out,
                      fetcher=gif_fetcher, rasterize=fake_rasterize)
    assert got is not None
    assert seen["gif"] == gif                       # the GIF went to the converter
    assert out.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not is_gif(out.read_bytes())


def test_fetch_image_drops_and_cleans_when_gif_conversion_fails(tmp_path):
    gif = b"GIF89a" + b"\x00" * 16
    out = tmp_path / "x.png"

    def gif_fetcher(u, p):
        Path(p).write_bytes(gif)
        return True

    got = fetch_image(req("https://meme.example/face.gif"), out, fetcher=gif_fetcher,
                      rasterize=lambda data, path: False)
    assert got is None
    assert not out.is_file()                        # the raw GIF was not left behind


def test_fetch_image_rasterizes_a_real_gif_with_ffmpeg(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    raw = base64.b64decode("R0lGODlhAQABAIAAAAUEBAAAACwAAAAAAQABAAACAkQBADs=")  # 1x1 GIF
    assert is_gif(raw)
    out = tmp_path / "x.png"

    def gif_fetcher(u, p):
        Path(p).write_bytes(raw)
        return True

    got = fetch_image(req("https://meme.example/face.gif"), out, fetcher=gif_fetcher)
    assert got is not None
    assert out.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
