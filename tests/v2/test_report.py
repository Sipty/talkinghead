"""The report renders what verification measured: pacing is a table, never a verdict."""

from talkinghead import verify
from talkinghead.config import Config
from talkinghead.report import build_report
from talkinghead.timeline import Cut, Plan, SourceInfo


class _W:
    def __init__(self, text, start, end):
        self.text, self.start, self.end = text, start, end


def test_report_renders_the_pacing_tables_from_pacing_facts() -> None:
    plan = Plan.from_cuts(SourceInfo("main", "/v/in.mp4", "x", 6.0), [Cut(0.65, 1.85, "silence")])
    words = [_W("One.", 0.0, 0.5), _W("Two", 2.0, 2.5)]
    pacing = verify.pacing_facts(plan, words, [(0.5, 2.0)], [{"reason": "silence", "quiet": 0.31}], Config())
    text = build_report(plan=plan, notes={}, checks=[], status="OK", render_cmd=None, pacing=pacing)
    assert "## Pacing" in text
    assert "| topic | 1 | 1.50 / 1.50 / 1.50 / 1.50 | 0.30 / 0.30 / 0.30 / 0.30 |" in text
    assert "| silence | 1 | 0.31 |" in text
    assert text.index("## Verification") < text.index("## Pacing") < text.index("## Cuts")


def test_report_without_pacing_facts_has_no_pacing_section() -> None:
    plan = Plan.from_cuts(SourceInfo("main", "/v/in.mp4", "x", 6.0), [])
    text = build_report(plan=plan, notes={}, checks=[], status="OK", render_cmd=None)
    assert "## Pacing" not in text
