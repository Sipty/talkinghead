"""The visuals stage over ONE editorial plan.

The judge reads only the final cut's transcript and answers once with the
central argument, the chapters and the sparse visuals. These tests pin that
shape: one proposal call, output-timeline anchors, verbatim quotes, chapter
cards that own their corner, and no generic card where a meme or piece of
evidence was asked for.
"""

from __future__ import annotations

import numpy as np

from talkinghead.audio import FRAME_SEC, Envelope, RefinedWord
from talkinghead.config import Config
from talkinghead.context import Context
from talkinghead.judge import validate
from talkinghead.markers import norm
from talkinghead.timeline import Cut, Plan, SourceInfo
from talkinghead.transcribe import Word
from talkinghead.visuals.stage import plan_visuals

TEXT = ("Hello. We have GLM five three Flash which is competing with Claude Opus. "
        "I pay six seven dollars a day for this pipeline. And that is the whole point.")
STEP = 0.4


def final_words():
    return [Word(t, i * STEP, i * STEP + 0.3, 1.0) for i, t in enumerate(TEXT.split())]


def words_from(tokens):
    return [Word(t, i * STEP, i * STEP + 0.3, 1.0) for i, t in enumerate(tokens)]


def word_start(phrase: str) -> float:
    """Start time of `phrase`, matched exactly as the pipeline's own anchor
    resolver does (punctuation-blind), so the fixture can hold 'point.'."""
    words = final_words()
    want = [norm(t) for t in phrase.split()]
    for i in range(len(words) - len(want) + 1):
        if [norm(w.text) for w in words[i:i + len(want)]] == want:
            return words[i].start
    raise AssertionError(f"{phrase!r} not in the fixture")


class VJudge:
    """One editorial-plan answer, plus optional research URLs."""

    name = "vjudge"

    def __init__(self, editorial, sources=None):
        self.editorial, self.sources = editorial, sources or []
        self.prompts = []

    def decide(self, prompt, schema):
        self.prompts.append(prompt)
        if "TASK: EDITORIAL PLAN" in prompt:
            return validate(self.editorial, schema)
        if "TASK: RESEARCH" in prompt:
            return validate({"sources": self.sources}, schema)
        raise AssertionError(f"unexpected prompt: {prompt[:80]}")

    @property
    def editorial_calls(self):
        return [p for p in self.prompts if "TASK: EDITORIAL PLAN" in p]


def editorial(visuals=(), chapters=(), argument="The pipeline is the product."):
    return {"central_argument": argument, "chapters": list(chapters), "visuals": list(visuals)}


def quote_visual(anchor, quote, hold=5.0, chapter=1):
    return {"chapter": chapter, "anchor_text": anchor, "type": "quote_card", "quote": quote,
            "visual_brief": "", "editorial_reason": "payoff", "hold_sec": hold}


def meme_visual(anchor, brief, hold=5.0, chapter=1, type="meme"):
    return {"chapter": chapter, "anchor_text": anchor, "type": type, "quote": "",
            "visual_brief": brief, "editorial_reason": "beat", "hold_sec": hold}


def ctx_for(judge, cfg=None, cuts=()):
    words = final_words()
    duration = len(words) * STEP + 2.0
    db = np.full(int(duration / FRAME_SEC) + 1, -50.0)
    env = Envelope(db, np.zeros_like(db), FRAME_SEC, 16000)
    refined = [RefinedWord(w.text, w.start, w.end, w.start, w.end, 1.0) for w in words]
    ctx = Context(
        config=cfg or Config(marker_votes=1, marker_agree=1, visuals_min_hold=1.0,
                             visuals_min_gap=1.0, plan_enabled=False),
        source=SourceInfo("main", "/v.mp4", "sha", duration),
        words=refined, env=env, floor=-50.0, silences=[], moans=[],
        judge=judge, llm_cache=None, guidelines="G", topic=None,
    )
    return ctx, Plan.from_cuts(ctx.source, list(cuts))


def _lib(tmp_path, *entries):
    tmp_path.mkdir(parents=True, exist_ok=True)
    for aid, depicts, kind in entries:
        (tmp_path / f"{aid}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "manifest.toml").write_text("\n".join(
        f'[[asset]]\nid = "{a}"\nfile = "{a}.png"\ndepicts = "{d}"\nkind = "{k}"\n'
        for a, d, k in entries))
    return tmp_path


# ------------------------------------------------------------------- the plan

def test_one_editorial_answer_supplies_chapters_and_visuals(tmp_path):
    chapters = [{"title": "The setup", "anchor_text": "We have GLM five three", "purpose": "b"},
                {"title": "The bill", "anchor_text": "I pay six seven dollars", "purpose": "b"}]
    judge = VJudge(editorial(
        visuals=[quote_visual("And that is the whole point", "And that is the whole point",
                              chapter=2)],
        chapters=chapters))
    cfg = Config(marker_votes=1, marker_agree=1, visuals_min_hold=1.0, visuals_min_gap=1.0,
                 plan_enabled=True, plan_hold=1.0, plan_open_at=0.5, plan_open_hold=1.0)
    ctx, plan = ctx_for(judge, cfg=cfg)
    _, notes = plan_visuals(ctx, plan, words=final_words(), assets_dir=_lib(tmp_path / "a"),
                            work_dir=tmp_path / "w")
    assert len(judge.editorial_calls) == 1
    assert notes["central_argument"] == "The pipeline is the product."
    assert notes["plan"]["chapters"] == 2
    assert {d["decision"] for d in notes["decisions"]} >= {"quote_card"}


def test_chapter_cards_remain_when_content_overlays_are_disabled(tmp_path):
    chapters = [{"title": "The setup", "anchor_text": "We have GLM five three", "purpose": "b"},
                {"title": "The bill", "anchor_text": "I pay six seven dollars", "purpose": "b"}]
    judge = VJudge(editorial(
        visuals=[quote_visual("And that is the whole point", "And that is the whole point",
                              chapter=2)],
        chapters=chapters))
    cfg = Config(marker_votes=1, marker_agree=1, visuals_content_enabled=False,
                 plan_enabled=True, plan_hold=1.0, plan_open_at=0.5, plan_open_hold=1.0)
    ctx, plan = ctx_for(judge, cfg=cfg)
    placements, notes = plan_visuals(ctx, plan, words=final_words(), assets_dir=_lib(tmp_path / "a"),
                                     work_dir=tmp_path / "w")
    assert placements and all(p.asset.startswith("p-") for p in placements)
    assert notes["content_overlays_enabled"] is False
    assert notes["decisions"] == []
    assert notes["plan"]["chapters"] == 2


def test_visuals_are_recorded_even_with_no_chapters(tmp_path):
    judge = VJudge(editorial(visuals=[quote_visual("And that is the whole point",
                                                   "And that is the whole point")]))
    ctx, plan = ctx_for(judge)
    placements, notes = plan_visuals(ctx, plan, words=final_words(), assets_dir=_lib(tmp_path / "a"),
                                     work_dir=tmp_path / "w")
    assert [p.asset for p in placements] == [notes["decisions"][0]["asset"]]
    assert "plan" not in notes     # plan card disabled


# ---------------------------------------------------------- output anchors

def test_anchor_uses_the_final_word_start_directly(tmp_path):
    """The anchors are output-timeline phrases; a cut earlier in the source
    must NOT shift them through the raw Plan."""
    judge = VJudge(editorial(visuals=[quote_visual("I pay six seven dollars",
                                                   "I pay six seven dollars")]))
    ctx, plan = ctx_for(judge, cuts=[Cut(0.0, 1.0, "cold-open")])
    placements, _ = plan_visuals(ctx, plan, words=final_words(), assets_dir=_lib(tmp_path / "a"),
                                 work_dir=tmp_path / "w")
    assert placements
    assert placements[0].start == word_start("I pay six seven dollars")


def test_an_anchor_the_judge_invented_is_reported_unresolved(tmp_path):
    judge = VJudge(editorial(visuals=[quote_visual("this phrase was never spoken aloud",
                                                   "this phrase was never spoken aloud")]))
    ctx, plan = ctx_for(judge)
    placements, notes = plan_visuals(ctx, plan, words=final_words(), assets_dir=_lib(tmp_path / "a"),
                                     work_dir=tmp_path / "w")
    assert placements == []
    assert notes["decisions"][0]["decision"] == "unresolved"


# ------------------------------------------------------------- quote cards

def test_a_quote_card_requires_verbatim_wording(tmp_path):
    judge = VJudge(editorial(visuals=[quote_visual("And that is the whole point",
                                                   "and thus the entire thesis")]))
    ctx, plan = ctx_for(judge)
    placements, notes = plan_visuals(ctx, plan, words=final_words(), assets_dir=_lib(tmp_path / "a"),
                                     work_dir=tmp_path / "w")
    assert placements == []
    row = notes["decisions"][0]
    assert row["decision"] == "unresolved"
    assert "verbatim" in row["note"]


def test_a_quote_card_is_rendered_from_the_exact_spoken_words(tmp_path):
    judge = VJudge(editorial(visuals=[quote_visual("And that is the whole point",
                                                   "And that is the whole point")]))
    ctx, plan = ctx_for(judge)
    placements, notes = plan_visuals(ctx, plan, words=final_words(), assets_dir=_lib(tmp_path / "a"),
                                     work_dir=tmp_path / "w")
    assert placements and placements[0].start == word_start("And that is the whole point")
    # The card shows the transcript's own tokens, so it keeps the sentence period.
    assert notes["decisions"][0]["quote"] == "And that is the whole point."
    assert (tmp_path / "w" / f"{notes['decisions'][0]['asset']}.png").is_file()


def test_a_quote_card_never_triggers_research(tmp_path):
    judge = VJudge(editorial(visuals=[quote_visual("And that is the whole point",
                                                   "And that is the whole point")]),
                   sources=[{"id": 0, "url": "https://example.com/x", "why": "y"}])
    ctx, plan = ctx_for(judge)
    plan_visuals(ctx, plan, words=final_words(), assets_dir=_lib(tmp_path / "a"),
                 work_dir=tmp_path / "w")
    assert not any("TASK: RESEARCH" in p for p in judge.prompts)


def test_a_quote_must_match_meaningful_punctuation(tmp_path):
    """Punctuation-blind matching would accept a spoken '150' for '$1.50' and
    then render invented text; the quote must keep the currency sign."""
    words = words_from(["I", "pay", "$1.50", "a", "day", "minus", "-10", "points"])
    judge = VJudge(editorial(visuals=[quote_visual("I pay $1.50 a day", "150")]))
    ctx, plan = ctx_for(judge)
    placements, notes = plan_visuals(ctx, plan, words=words, assets_dir=_lib(tmp_path / "a"),
                                     work_dir=tmp_path / "w")
    assert placements == []
    assert notes["decisions"][0]["decision"] == "unresolved"


def test_a_quote_renders_the_matched_spoken_text(tmp_path):
    """The card is built from the transcript tokens, not the model's string."""
    words = words_from(["I", "pay", "$1.50", "a", "day", "minus", "-10", "points"])
    judge = VJudge(editorial(visuals=[quote_visual("I pay $1.50 a day", "$1.50")]))
    ctx, plan = ctx_for(judge)
    placements, notes = plan_visuals(ctx, plan, words=words, assets_dir=_lib(tmp_path / "a"),
                                     work_dir=tmp_path / "w")
    assert placements
    assert notes["decisions"][0]["quote"] == "$1.50"


# ------------------------------------------------- meme / technical evidence

def test_a_missing_meme_becomes_no_card_not_a_generic_one(tmp_path):
    judge = VJudge(editorial(visuals=[meme_visual("I pay six seven dollars",
                                                  "a rare 2000s reaction meme")]))
    cfg = Config(marker_votes=1, marker_agree=1, visuals_research=False,
                 visuals_min_hold=1.0, visuals_min_gap=1.0, plan_enabled=False)
    ctx, plan = ctx_for(judge, cfg=cfg)
    placements, notes = plan_visuals(ctx, plan, words=final_words(), assets_dir=_lib(tmp_path / "a"),
                                     work_dir=tmp_path / "w")
    assert placements == []
    assert notes["decisions"][0]["decision"] == "dropped"
    assert "no generic card" in notes["decisions"][0]["note"]
    # nothing was rasterized into a text plate for the same words
    assert not list((tmp_path / "w").glob("c-*.png"))


def test_failed_research_drops_the_evidence_rather_than_swapping_in_a_card(tmp_path):
    judge = VJudge(editorial(visuals=[meme_visual("I pay six seven dollars", "pricing page",
                                                  type="technical_evidence")]),
                   sources=[{"id": 0, "url": "https://example.com/x", "why": "y"}])
    ctx, plan = ctx_for(judge)
    placements, notes = plan_visuals(ctx, plan, words=final_words(), assets_dir=_lib(tmp_path / "a"),
                                     work_dir=tmp_path / "w", fetcher=lambda u, o: False)
    assert placements == []
    assert notes["decisions"][0]["decision"] == "dropped"


def test_research_path_builds_its_prompt_and_ingests_the_result(tmp_path):
    judge = VJudge(
        editorial(visuals=[meme_visual("GLM five three Flash which is competing",
                                       "GLM 5.3 Flash benchmark page",
                                       type="technical_evidence")]),
        sources=[{"id": 0, "url": "https://z.ai/bench", "why": "canonical"}])
    ctx, plan = ctx_for(judge)
    fetched = []

    def fetcher(url, out):
        fetched.append(url)
        out.write_bytes(b"\x89PNG\r\n\x1a\n")
        return True

    placements, notes = plan_visuals(ctx, plan, words=final_words(),
                                     assets_dir=_lib(tmp_path / "a"),
                                     work_dir=tmp_path / "w", fetcher=fetcher)
    assert fetched == ["https://z.ai/bench"]
    assert any("TASK: RESEARCH" in p for p in judge.prompts)
    assert notes["decisions"][0]["decision"] == "researched"
    assert len(placements) == 1
    assert 'kind = "shot"' in (tmp_path / "a" / "manifest.toml").read_text()


def test_a_meme_research_request_carries_its_type(tmp_path):
    judge = VJudge(editorial(visuals=[meme_visual("I pay six seven dollars",
                                                  "a 2000s reaction face")]))
    ctx, plan = ctx_for(judge)
    _, notes = plan_visuals(ctx, plan, words=final_words(), assets_dir=_lib(tmp_path / "a"),
                            work_dir=tmp_path / "w")
    research = next(p for p in judge.prompts if "TASK: RESEARCH" in p)
    assert "type=meme" in research
    assert notes["decisions"][0]["decision"] == "dropped"


def test_a_meme_is_sourced_as_the_image_not_a_page(tmp_path):
    """A meme must land as the actual image bytes, never a page screenshot."""
    judge = VJudge(editorial(visuals=[meme_visual("I pay six seven dollars",
                                                  "a 2000s reaction face")]),
                   sources=[{"id": 0, "url": "https://meme.example/face.png", "why": "y"}])
    ctx, plan = ctx_for(judge)
    png = b"\x89PNG\r\n\x1a\n" + b"imagedata"

    def image_fetcher(url, out):
        out.write_bytes(png)
        return True

    placements, notes = plan_visuals(ctx, plan, words=final_words(), assets_dir=_lib(tmp_path / "a"),
                                     work_dir=tmp_path / "w", image_fetcher=image_fetcher)
    row = notes["decisions"][0]
    assert row["decision"] == "researched"
    assert (tmp_path / "w" / f"{row['asset']}.png").read_bytes() == png
    assert len(placements) == 1


def test_a_meme_page_url_is_rejected_not_screenshotted(tmp_path):
    judge = VJudge(editorial(visuals=[meme_visual("I pay six seven dollars",
                                                  "a 2000s reaction face")]),
                   sources=[{"id": 0, "url": "https://meme.example/face", "why": "y"}])
    ctx, plan = ctx_for(judge)

    def boom(url, out):
        raise AssertionError("a page must not be fetched as a meme")

    placements, notes = plan_visuals(ctx, plan, words=final_words(), assets_dir=_lib(tmp_path / "a"),
                                     work_dir=tmp_path / "w", image_fetcher=boom)
    assert placements == []
    assert notes["decisions"][0]["decision"] == "dropped"
    assert "direct meme image" in notes["decisions"][0]["note"]


def test_research_is_skipped_when_a_library_asset_already_matches(tmp_path):
    judge = VJudge(editorial(visuals=[meme_visual("GLM five three Flash which is competing",
                                                  "GLM 5.3 Flash benchmark page",
                                                  type="technical_evidence")]),
                   sources=[{"id": 0, "url": "https://z.ai/bench", "why": "c"}])
    ctx, plan = ctx_for(judge)
    lib = _lib(tmp_path / "a", ("s-zbench", "GLM 5.3 Flash benchmark page", "shot"))
    _, notes = plan_visuals(ctx, plan, words=final_words(), assets_dir=lib, work_dir=tmp_path / "w",
                            fetcher=lambda u, o: (_ for _ in ()).throw(AssertionError("fetched")))
    assert notes["decisions"][0]["decision"] == "reuse"
    assert not any("TASK: RESEARCH" in p for p in judge.prompts)


# ------------------------------------------------------------- chapter cards

def test_chapter_cards_start_at_their_anchors(tmp_path):
    chapters = [{"title": "Intro", "anchor_text": "We have GLM five three", "purpose": "b"},
                {"title": "The bill", "anchor_text": "I pay six seven dollars", "purpose": "b"},
                {"title": "The point", "anchor_text": "And that is the whole point", "purpose": "b"}]
    judge = VJudge(editorial(chapters=chapters))
    cfg = Config(marker_votes=1, marker_agree=1, visuals_min_hold=0.4, visuals_min_gap=0.4,
                 plan_enabled=True, plan_hold=0.4, plan_open_at=0.0, plan_open_hold=0.4)
    ctx, plan = ctx_for(judge, cfg=cfg)
    places, notes = plan_visuals(ctx, plan, words=final_words(), assets_dir=_lib(tmp_path / "a"),
                                 work_dir=tmp_path / "w")
    plan_starts = [round(p.start, 6) for p in places]
    assert round(word_start("I pay six seven dollars"), 6) in plan_starts
    assert round(word_start("And that is the whole point"), 6) in plan_starts
    first = next(r for r in notes["plan"]["reveals"] if r["n"] == 1)
    assert first["decision"] == "on the opening card"


def test_chapter_cards_reserve_the_corner_against_content(tmp_path):
    chapters = [{"title": "Intro", "anchor_text": "We have GLM five three", "purpose": "b"},
                {"title": "The bill", "anchor_text": "I pay six seven dollars", "purpose": "b"}]
    judge = VJudge(editorial(
        visuals=[quote_visual("I pay six seven dollars", "I pay six seven dollars a day",
                              chapter=2)],
        chapters=chapters))
    cfg = Config(marker_votes=1, marker_agree=1, visuals_min_hold=0.4, visuals_min_gap=0.4,
                 plan_enabled=True, plan_hold=0.4, plan_open_at=0.0, plan_open_hold=0.4)
    ctx, plan = ctx_for(judge, cfg=cfg)
    places, _ = plan_visuals(ctx, plan, words=final_words(), assets_dir=_lib(tmp_path / "a"),
                             work_dir=tmp_path / "w")
    # The chapter card sits exactly on the anchor; the quote card yields.
    assert any(p.start == word_start("I pay six seven dollars") for p in places)
    for x, y in zip(places, places[1:]):
        assert y.start >= x.end + cfg.visuals_min_gap - 1e-6


def test_a_repeated_phrase_resolves_inside_its_declared_chapter(tmp_path):
    """'shared beat' appears in both chapters; a chapter-2 visual must land on
    the chapter-2 occurrence, not the earlier one."""
    words = words_from(["intro", "one", "shared", "beat", "middle", "two",
                        "shared", "beat", "final", "three"])
    chapters = [{"title": "One", "anchor_text": "intro one", "purpose": "b"},
                {"title": "Two", "anchor_text": "middle two", "purpose": "b"}]
    judge = VJudge(editorial(
        visuals=[quote_visual("shared beat", "shared beat", chapter=2)], chapters=chapters))
    ctx, plan = ctx_for(judge)
    places, notes = plan_visuals(ctx, plan, words=words, assets_dir=_lib(tmp_path / "a"),
                                 work_dir=tmp_path / "w")
    assert notes["decisions"][0]["decision"] == "quote_card"
    assert places and places[0].start == words[6].start


def test_a_visual_cannot_be_pushed_into_the_next_chapter(tmp_path):
    """A chapter-1 overlay that cannot fit before chapter 2 is dropped, never
    pushed across the chapter boundary."""
    words = words_from(["intro", "one", "alpha", "beta", "middle", "two", "gamma", "delta"])
    chapters = [{"title": "One", "anchor_text": "intro one", "purpose": "b"},
                {"title": "Two", "anchor_text": "middle two", "purpose": "b"}]
    judge = VJudge(editorial(
        visuals=[quote_visual("alpha beta", "alpha beta", chapter=1, hold=20.0)],
        chapters=chapters))
    ctx, plan = ctx_for(judge, cfg=Config(marker_votes=1, marker_agree=1, plan_enabled=True,
                                          plan_hold=0.4, plan_open_at=0.0, plan_open_hold=0.4,
                                          visuals_min_hold=1.0, visuals_min_gap=0.4))
    places, notes = plan_visuals(ctx, plan, words=words, assets_dir=_lib(tmp_path / "a"),
                                 work_dir=tmp_path / "w")
    row = notes["decisions"][0]
    assert row["decision"] == "dropped"
    ch2_at = words[4].start
    assert not [p for p in places if p.asset == row.get("asset") and p.end > ch2_at]


# ------------------------------------------------------------------ prompt

def test_opening_roadmap_appears_even_when_chapter_one_starts_at_zero(tmp_path):
    """Chapter one at 0 s must not suppress the roadmap: chapter one is shown
    ON the opening card, so the card still starts near plan_open_at."""
    chapters = [{"title": "Intro", "anchor_text": "Hello. We have", "purpose": "b"},
                {"title": "The bill", "anchor_text": "I pay six seven dollars", "purpose": "b"}]
    judge = VJudge(editorial(chapters=chapters))
    cfg = Config(marker_votes=1, marker_agree=1, visuals_min_hold=0.4, visuals_min_gap=0.4,
                 plan_enabled=True, plan_hold=0.4, plan_open_at=1.0, plan_open_hold=0.4)
    ctx, plan = ctx_for(judge, cfg=cfg)
    _, notes = plan_visuals(ctx, plan, words=final_words(), assets_dir=_lib(tmp_path / "a"),
                            work_dir=tmp_path / "w")
    assert notes["plan"]["opening_at"] == 1.0
    first = next(r for r in notes["plan"]["reveals"] if r["n"] == 1)
    assert first["decision"] == "on the opening card"


def test_chapters_are_clamped_to_plan_chapters_in_code(tmp_path):
    anchors = ["Hello. We have", "I pay six seven dollars",
               "And that is the whole point", "We have GLM five"]
    chapters = [{"title": f"Chapter {i}", "anchor_text": a, "purpose": "b"}
                for i, a in enumerate(anchors, start=1)]
    judge = VJudge(editorial(chapters=chapters))
    cfg = Config(marker_votes=1, marker_agree=1, visuals_min_hold=0.4, visuals_min_gap=0.4,
                 plan_enabled=True, plan_hold=0.4, plan_open_at=0.0, plan_open_hold=0.4,
                 plan_chapters=2)
    ctx, plan = ctx_for(judge, cfg=cfg)
    _, notes = plan_visuals(ctx, plan, words=final_words(), assets_dir=_lib(tmp_path / "a"),
                            work_dir=tmp_path / "w")
    assert notes["plan"]["chapters"] == 2
    assert len(notes["plan"]["reveals"]) == 2


def test_the_prompt_carries_only_the_final_cut_transcript(tmp_path):
    """The editor must never see the raw source: a phrase removed by a cut
    must not appear in the prompt, whatever the source transcript holds."""
    judge = VJudge(editorial(visuals=[quote_visual("And that is the whole point",
                                                   "And that is the whole point")]))
    ctx, plan = ctx_for(judge, cuts=[Cut(word_start("GLM five three Flash which"),
                                         word_start("Claude Opus.") + 0.3, "content")])
    final = [w for w in final_words()
             if not (word_start("GLM five three Flash which") <= w.start
                     < word_start("Claude Opus.") + 0.3)]
    plan_visuals(ctx, plan, words=final, assets_dir=_lib(tmp_path / "a"), work_dir=tmp_path / "w")
    prompt = judge.editorial_calls[0]
    assert "GLM five three Flash" not in prompt
    assert "Claude Opus" not in prompt
    assert "I pay six seven dollars" in prompt
