import shutil

import pytest

from talkinghead.visuals.cards import (
    PALETTE,
    TEXT_W,
    card_sha,
    card_svg,
    fit_subtitle,
    fit_title,
    reading_seconds,
    render_card,
    text_width,
    wrap,
)


def test_svg_carries_brand_palette(tmp_path):
    svg = card_svg("GLM 5.3 Flash", "z.ai", "term", avatar=_avatar_png(tmp_path))
    assert PALETTE["accent"] == "#e2571b"
    assert "#e2571b" in svg and "#33281d" in svg and "#f6efe4" in svg


def test_price_kind_puts_the_figure_in_accent():
    assert f'fill="{PALETTE["accent"]}">$6-7' in card_svg("$6-7 / day", "cost", "price")


def test_svg_escapes_markup():
    out = card_svg("a <script>b & c", None, "term")
    assert "<script>" not in out and "&amp;" in out


# ---------------------------------------------------------------- fitting

def test_every_title_line_measures_inside_the_plate():
    """The v11 defect: text overflowed the card."""
    for title in ["Claude Superpowers skills repository README",
                  "Anthropic prompt caching documentation page",
                  "OMP (Oh My Pi) coding harness terminal session",
                  "Claude Code terminal output where Opus stalls",
                  "OpenRouter provider routing page",
                  "GLM 5.3 Flash"]:
        lines, size = fit_title(title)
        assert lines, title
        for ln in lines:
            assert text_width(ln, size, bold=True) <= TEXT_W, (title, ln, size)


def test_long_titles_wrap_rather_than_shrink_to_nothing():
    lines, size = fit_title("Anthropic prompt caching documentation page")
    assert len(lines) == 2
    assert size >= 40


def test_short_titles_keep_the_largest_size():
    from talkinghead.visuals.cards import TITLE_SIZES
    lines, size = fit_title("Vibecode")
    assert lines == ["Vibecode"] and size == TITLE_SIZES[0]


def test_overlong_title_trims_on_a_word_boundary_with_an_ellipsis():
    lines, _ = fit_title(" ".join(["extraordinarily"] * 12))
    joined = " ".join(lines)
    assert joined.endswith("…")
    assert "extraordinaril…" not in joined      # never mid-word


def test_subtitle_is_fitted_too():
    sub, size = fit_subtitle("running cost across multiple concurrent pipelines per day")
    assert text_width(sub, size) <= TEXT_W


def test_wrap_returns_none_when_it_cannot_fit():
    assert wrap("a " * 200, 76, max_lines=2, bold=True) is None


def test_card_grows_taller_for_a_second_line():
    one = card_svg("Vibecode", None, "term")
    two = card_svg("Anthropic prompt caching documentation page", None, "term")

    def h(svg):
        return int(svg.split('height="')[1].split('"')[0])
    assert h(two) > h(one)


# ---------------------------------------------------------------- dwell

def test_reading_time_grows_with_text():
    assert reading_seconds("Vibecode", None) < reading_seconds(
        "Claude Opus 4.8", "the model that changed my mind about vibecoding")


def test_reading_time_never_below_the_notice_threshold():
    """1.5 s was not long enough to notice a card, let alone read it."""
    assert reading_seconds("X", None) >= 2.0


# ---------------------------------------------------------------- raster

def test_sha_is_stable_and_content_addressed():
    assert card_sha("a", None, "term") == card_sha("a", None, "term")
    assert card_sha("a", None, "term") != card_sha("b", None, "term")


@pytest.mark.skipif(not shutil.which("rsvg-convert"), reason="rsvg-convert missing")
def test_render_card_writes_a_png(tmp_path):
    out = render_card("$6-7 / day", "running cost", "price", tmp_path / "c.png")
    assert out.is_file()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.skipif(not shutil.which("rsvg-convert"), reason="rsvg-convert missing")
def test_render_card_is_a_noop_when_present(tmp_path):
    out = render_card("x", None, "term", tmp_path / "c.png")
    first = out.stat().st_mtime_ns
    render_card("x", None, "term", tmp_path / "c.png")
    assert out.stat().st_mtime_ns == first


# ------------------------------------------------- avatar and speech bubble

def _avatar_png(tmp_path) -> str:
    """A real 4x4 PNG on disk, passed to the card builders explicitly."""
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    rows = b"".join(b"\x00" + b"\xf6\xef\xe4" * 4 for _ in range(4))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
    path = tmp_path / "avatar.png"
    path.write_bytes(png)
    return str(path)


def test_a_card_with_an_avatar_has_the_image_and_a_speech_bubble(tmp_path):
    svg = card_svg("GLM 5.3 Flash", "the engineer model", "term", avatar=_avatar_png(tmp_path))
    assert svg.count("<circle") >= 2                 # badge fill + accent ring
    assert "<image" in svg and "clip-path" in svg    # avatar, clipped round
    assert "<path d=" in svg                         # the bubble tail


def test_the_avatar_is_inset_not_clipped_by_the_circle(tmp_path):
    """A circle crop lopped off outer detail, so the art is inset."""
    from talkinghead.visuals.cards import AVATAR, AVATAR_FIT
    assert AVATAR_FIT > 0
    svg = card_svg("X", None, "term", avatar=_avatar_png(tmp_path))
    width = float(svg.split('<image')[1].split('width="')[1].split('"')[0])
    assert width <= AVATAR * 1.2


def _bubble(svg):
    """(x, width) of the bubble plate rect."""
    seg = svg.split('rx="30"')[0]
    x = float(seg.rsplit('<rect x="', 1)[1].split('"')[0])
    w = float(seg.rsplit('width="', 1)[1].split('"')[0])
    return x, w


def test_the_bubble_shrinks_to_its_text():
    """A fixed-width plate left dead space beside a two-word title."""
    _, short_w = _bubble(card_svg("Vibecode", None, "term"))
    _, long_w = _bubble(card_svg("Anthropic prompt caching documentation page", None, "term"))
    assert short_w < long_w


def test_the_avatar_stays_anchored_to_the_right_edge(tmp_path):
    """With an avatar the bubble opens leftward from it. The canvas is fixed so
    text scales consistently, and the avatar is the right-anchored element so a
    compact card still hugs the top-right corner."""
    from talkinghead.visuals.cards import AVATAR, MARGIN_R, WIDTH
    avatar = _avatar_png(tmp_path)
    for title in ["Vibecode", "GLM 5.3 Flash", "Anthropic prompt caching documentation page"]:
        svg = card_svg(title, None, "term", avatar=avatar)
        av_cx = float(svg.split('<circle cx="')[1].split('"')[0])
        assert abs((av_cx + AVATAR / 2) - (WIDTH - MARGIN_R)) < 1.0, title


def test_the_bubble_sits_to_the_left_of_the_avatar(tmp_path):
    from talkinghead.visuals.cards import AVATAR
    avatar = _avatar_png(tmp_path)
    for title in ["Vibecode", "Anthropic prompt caching documentation page"]:
        svg = card_svg(title, None, "term", avatar=avatar)
        av_cx = float(svg.split('<circle cx="')[1].split('"')[0])
        x, w = _bubble(svg)
        assert x + w < av_cx - AVATAR / 2 + 1, title       # bubble ends before the badge
        assert x > 0, title


def test_the_tail_points_right_toward_the_avatar(tmp_path):
    svg = card_svg("GLM 5.3 Flash", None, "term", avatar=_avatar_png(tmp_path))
    x, w = _bubble(svg)
    path = svg.split('<path d="M ')[1].split('"')[0]
    tip_x = float(path.split(" L ")[1].split()[0])
    assert tip_x > x + w, "the tail must jut out of the bubble's RIGHT edge"


def test_a_card_without_an_avatar_has_no_badge_or_tail():
    from talkinghead.visuals.cards import MARGIN_R, WIDTH
    svg = card_svg("Vibecode", None, "term")
    assert "<image" not in svg and "<circle" not in svg and "<path d=" not in svg
    x, w = _bubble(svg)
    assert abs((x + w) - (WIDTH - MARGIN_R)) < 1.0        # the bubble takes the right margin alone


def test_an_avatar_path_that_does_not_exist_is_an_error(tmp_path):
    from talkinghead.visuals.cards import avatar_data_uri
    with pytest.raises(FileNotFoundError):
        avatar_data_uri(str(tmp_path / "missing.png"))


def test_the_card_key_follows_the_avatar_content(tmp_path):
    path = _avatar_png(tmp_path)
    assert card_sha("x", None, "term", path) == card_sha("x", None, "term", path)
    assert card_sha("x", None, "term", path) != card_sha("x", None, "term")


@pytest.mark.skipif(not shutil.which("rsvg-convert"), reason="rsvg-convert missing")
def test_render_card_without_an_avatar_writes_a_png(tmp_path):
    out = render_card("Vibecode", None, "term", tmp_path / "c.png")
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_bubble_never_overruns_the_canvas():
    from talkinghead.visuals.cards import WIDTH
    for title in ["a", "Vibecode", "Anthropic prompt caching documentation page",
                  "x" * 60, "OMP (Oh My Pi) coding harness terminal session"]:
        x, w = _bubble(card_svg(title, "a fairly long subtitle here", "term"))
        assert x >= 0 and x + w <= WIDTH, title


def test_text_metrics_are_calibrated_against_rendered_ink():
    """Measured with rsvg: bold 0.482 mean / 0.528 max advance per character.
    A flat 0.60 overestimated by up to 24%, which showed as dead space."""
    measured = {("Vibecode", 72, True): 304, ("GLM 5.3 Flash", 72, True): 455,
                ("Anthropic prompt caching", 72, True): 828,
                ("the engineer model", 40, False): 326,
                ("running cost, multiple pipelines", 40, False): 524}
    for (text, size, bold), ink in measured.items():
        est = text_width(text, size, bold=bold)
        assert est >= ink * 0.98, (text, est, ink)     # never badly under
        assert est <= ink * 1.20, (text, est, ink)     # and not wildly over


# --------------------------------------------------- labels from depicts

def test_label_stops_at_the_word_naming_the_artefact():
    """`depicts` is written for MATCHING, so using it verbatim as a title put
    "Claude Superpowers skills repository README screenshot" on screen."""
    from talkinghead.visuals.cards import label_from_depicts
    assert label_from_depicts(
        "Claude Code sub-agents documentation page screenshot") == "Claude Code sub-agents"
    assert label_from_depicts(
        "Anthropic prompt caching documentation page") == "Anthropic prompt caching"
    assert label_from_depicts(
        "OMP (Oh My Pi) coding harness terminal screenshot") == "OMP (Oh My Pi) coding harness"


def test_label_is_capped_in_length():
    from talkinghead.visuals.cards import LABEL_MAX_WORDS, label_from_depicts
    long = "one two three four five six seven eight nine ten"
    assert len(label_from_depicts(long).split()) <= LABEL_MAX_WORDS


def test_label_survives_a_description_that_is_only_an_artefact():
    from talkinghead.visuals.cards import label_from_depicts
    assert label_from_depicts("screenshot") != ""


def test_label_leaves_a_clean_subject_alone():
    from talkinghead.visuals.cards import label_from_depicts
    assert label_from_depicts("Kimi K3 pricing") == "Kimi K3 pricing"
