"""The chapter plan card: a readable roadmap whose later cards begin at their
own chapter's anchor and own that corner."""

from __future__ import annotations

import numpy as np

from talkinghead.audio import FRAME_SEC, Envelope, RefinedWord
from talkinghead.config import Config
from talkinghead.context import Context
from talkinghead.timeline import Plan, SourceInfo
from talkinghead.transcribe import Word
from talkinghead.visuals.plancard import PlanItem, plan_sha, plan_svg
from talkinghead.visuals.stage import place_plan

TEXT = ("one two three four five six seven eight nine ten "
        "alpha beta gamma delta epsilon zeta eta theta iota kappa "
        "lambda mu nu xi omicron pi rho sigma tau upsilon "
        "phi chi psi omega red green blue white black grey")


def ctx_for(cfg=None, step=1.0):
    words = [Word(t, i * step, i * step + 0.6, 1.0) for i, t in enumerate(TEXT.split())]
    duration = len(words) * step + 4.0
    db = np.full(int(duration / FRAME_SEC) + 1, -50.0)
    env = Envelope(db, np.zeros_like(db), FRAME_SEC, 16000)
    refined = [RefinedWord(w.text, w.start, w.end, w.start, w.end, 1.0) for w in words]
    return Context(
        config=cfg or Config(plan_hold=4.0, visuals_min_gap=2.0, plan_revealed_at_open=1,
                            plan_open_at=1.0, plan_open_hold=5.0, visuals_min_hold=2.0),
        source=SourceInfo("main", "/v.mp4", "sha", duration),
        words=refined, env=env, floor=-50.0, silences=[], moans=[],
        judge=None, llm_cache=None, guidelines="G", topic="testing",
    )


def a_plan(ctx):
    return Plan.from_cuts(ctx.source, [])


CH = [{"title": "First thing", "anchor_text": "alpha beta gamma", "purpose": "b"},
      {"title": "Second thing", "anchor_text": "lambda mu nu", "purpose": "b"},
      {"title": "The payoff", "anchor_text": "phi chi psi", "purpose": "b"}]


# ------------------------------------------------------------------ the card

def test_every_title_is_sharp_and_only_the_current_row_is_highlighted():
    svg = plan_svg([PlanItem("current", True), PlanItem("later", False)])
    assert "current" in svg and "later" in svg
    assert "feGaussianBlur" not in svg
    assert 'filter="url(#blur)"' not in svg
    assert svg.count('data-current="true"') == 1


def test_default_plan_card_has_no_clickbait_footer():
    svg = plan_svg([PlanItem("current", True), PlanItem("later", False)])
    assert "stay to the end" not in svg
    assert "last one cost me" not in svg


def test_the_sha_changes_as_the_current_chapter_changes():
    one = plan_sha([PlanItem("a", True), PlanItem("b", False)], "THE PLAN", None)
    two = plan_sha([PlanItem("a", False), PlanItem("b", True)], "THE PLAN", None)
    assert one != two


def test_the_sharp_card_does_not_reuse_the_legacy_blurred_asset():
    items = [PlanItem("The five-agent pipeline I actually run", True),
             PlanItem("The skill that grills you first", False),
             PlanItem("My exact models, and my $10 day", False),
             PlanItem("Why your reviewer must be another lab", False),
             PlanItem("It shipped a whole epic overnight", False),
             PlanItem("The caching mistake that torched my quota", False)]
    assert plan_sha(items, "THE PLAN", None) != "p-c36f7d12bedf71"


def test_every_row_is_numbered_and_present():
    svg = plan_svg([PlanItem("a", True), PlanItem("bbb", False), PlanItem("ccc", False)])
    assert ">bbb<" in svg and ">ccc<" in svg      # hidden, not omitted
    assert ">3<" in svg


# ------------------------------------------------------------- placement

def test_opening_card_highlights_the_first_chapter(tmp_path):
    ctx = ctx_for()
    places, notes = place_plan(ctx, a_plan(ctx), CH, words=ctx.as_words(), work_dir=tmp_path)
    assert notes["chapters"] == 3
    assert notes["opening_at"] is not None
    assert places and places[0].start < places[-1].start


def test_one_current_chapter_card_per_chapter_after_the_opening(tmp_path):
    ctx = ctx_for()
    places, notes = place_plan(ctx, a_plan(ctx), CH, words=ctx.as_words(), work_dir=tmp_path)
    # opening (chapter 1 current) + one card for chapters 2 and 3
    assert len(places) == 3
    assert [r.get("current") for r in notes["reveals"] if r.get("decision") == "placed"] == [2, 3]


def test_each_later_card_begins_at_its_chapter_anchor(tmp_path):
    ctx = ctx_for()
    _, notes = place_plan(ctx, a_plan(ctx), CH, words=ctx.as_words(), work_dir=tmp_path)
    for row in notes["reveals"]:
        if row.get("decision") == "placed":
            assert row["at"] == row["chapter_at"]


def test_a_card_is_pushed_only_by_the_card_before_it(tmp_path):
    """The anchor owns the start; only an earlier card can push it."""
    ch = [{"title": "A", "anchor_text": "alpha beta gamma", "purpose": "b"},
          {"title": "B", "anchor_text": "delta epsilon zeta", "purpose": "b"}]
    ctx = ctx_for()
    places, _ = place_plan(ctx, a_plan(ctx), ch, words=ctx.as_words(), work_dir=tmp_path)
    assert len(places) == 2
    assert places[1].start >= places[0].end + ctx.config.visuals_min_gap


def test_no_room_at_the_end_of_the_cut_is_recorded(tmp_path):
    cfg = Config(plan_hold=10.0, visuals_min_gap=2.0, plan_revealed_at_open=1,
                 plan_open_at=1.0, plan_open_hold=5.0, visuals_min_hold=2.0)
    ctx = ctx_for(cfg)
    ch = [{"title": "A", "anchor_text": "alpha beta gamma", "purpose": "b"},
          {"title": "Late", "anchor_text": "white black grey", "purpose": "b"}]
    _, notes = place_plan(ctx, a_plan(ctx), ch, words=ctx.as_words(), work_dir=tmp_path)
    blocked = [r for r in notes["reveals"] if r.get("decision") == "no-room"]
    assert blocked, "a chapter with no room for its card must be reported"
    assert "no" in blocked[0]["note"] and "window" in blocked[0]["note"]


def test_an_unresolvable_chapter_anchor_is_reported(tmp_path):
    ch = [{"title": "Ghost", "anchor_text": "this was never said", "purpose": "b"}] + CH
    ctx = ctx_for()
    _, notes = place_plan(ctx, a_plan(ctx), ch, words=ctx.as_words(), work_dir=tmp_path)
    assert any(r.get("decision") == "unresolved" for r in notes["reveals"])


def test_no_chapters_is_not_fatal(tmp_path):
    ctx = ctx_for()
    places, notes = place_plan(ctx, a_plan(ctx), [], words=ctx.as_words(), work_dir=tmp_path)
    assert places == [] and notes["placed"] == 0
    assert "no chapters" in notes["note"]


def test_the_opening_card_is_at_the_START_of_the_video(tmp_path):
    ctx = ctx_for()
    _, notes = place_plan(ctx, a_plan(ctx), CH, words=ctx.as_words(), work_dir=tmp_path)
    assert notes["opening_at"] <= ctx.config.plan_open_at + 1.0


def test_chapter_one_is_marked_as_shown_on_the_opening_card(tmp_path):
    ctx = ctx_for()
    _, notes = place_plan(ctx, a_plan(ctx), CH, words=ctx.as_words(), work_dir=tmp_path)
    first = next(r for r in notes["reveals"] if r["n"] == 1)
    assert first["decision"] == "on the opening card"


def test_the_opening_hold_shortens_rather_than_skipping(tmp_path):
    """A video too short for the full opening hold still gets its roadmap."""
    cfg = Config(plan_hold=4.0, visuals_min_gap=1.0, plan_revealed_at_open=1,
                 plan_open_at=1.0, plan_open_hold=100.0, visuals_min_hold=2.0)
    ctx = ctx_for(cfg)
    _, notes = place_plan(ctx, a_plan(ctx), CH, words=ctx.as_words(), work_dir=tmp_path)
    assert notes.get("opening_at") is not None
    assert notes["opening_hold"] < 100.0
