"""One report per run: what was decided and by how many votes, every cut with
its reason and provenance, the verification table, and honest totals (removed
seconds must equal the duration delta)."""

from __future__ import annotations

from .timeline import Plan
from .verify import CheckResult


def _fmt(t: float) -> str:
    m, s = divmod(t, 60.0)
    return f"{int(m):02d}:{s:05.2f}"


def _plan_card(notes: dict) -> list[str]:
    """The roadmap card: which chapter reveals landed, and which had no room."""
    pl = (notes.get("visuals") or {}).get("plan") or {}
    reveals = pl.get("reveals") or []
    if not reveals:
        return []
    rows = ["## Plan card", "",
            f"{pl.get('chapters', 0)} chapters, {pl.get('placed', 0)} cards placed"
            + (f", opening at {pl.get('opening_at')}s" if pl.get("opening_at") is not None else "")
            + ".", ""]
    if pl.get("note"):
        rows += [f"- **WARN**: {pl['note']}", ""]
    rows += ["| # | chapter | at | chapter starts | state | note |",
             "|---|---|---|---|---|---|"]
    for r in reveals:
        at = r.get("at")
        rows.append(
            f"| {r.get('n')} | {' '.join(str(r.get('title', '')).split())[:40]} "
            f"| {_fmt(at) if at is not None else ''} "
            f"| {_fmt(r['chapter_at']) if r.get('chapter_at') is not None else ''} "
            f"| {r.get('decision', '')} | {' '.join(str(r.get('note', '')).split())[:54]} |")
    return rows + [""]


def _retake_decisions(notes: dict) -> list[str]:
    """Every marker candidate with its verdict, quote and reason.

    An unlabelled video has no ground truth, so this is a DECISION LOG, not an
    eval: it becomes an eval only once the owner confirms or corrects the
    verdicts. Calling it an eval before that would hide regressions.
    """
    log = (notes.get("retakes") or {}).get("decisions") or []
    if not log:
        return []
    cut = sum(1 for d in log if d["verdict"] == "cut")
    rows = ["## Retake decisions", "",
            f"{len(log)} marker candidates: {cut} cut, {len(log) - cut} kept. "
            "Correct anything wrong here and the corrections become this video's labels.", "",
            "| word | time | marker | tier | verdict | quote | why |",
            "|---|---|---|---|---|---|---|"]
    for d in log:
        quote = " ".join(str(d.get("quote", "")).split())[:70]
        why = " ".join(str(d.get("why", "")).split())[:60]
        rows.append(f"| w{d['word']:04d} | {_fmt(d['time'])} | {d['marker']} | {d['tier']} "
                    f"| **{d['verdict']}** | {quote} | {why} |")
    return rows + [""]


def _asset_decisions(notes: dict) -> list[str]:
    """How each anchor got its visual: reuse must be earned, never assumed."""
    v = notes.get("visuals") or {}
    log = v.get("decisions") or []
    if not log:
        return []
    counts = v.get("counts") or {}
    tally = ", ".join(f"{k}={n}" for k, n in sorted(counts.items()))
    rows = ["## Asset decisions", "",
            f"{v.get('placed', 0)} overlays placed from {v.get('candidates', 0)} candidates ({tally}).", ""]
    rows += ["| anchor | decision | match | asset | note |", "|---|---|---|---|---|"]
    for d in log:
        anchor = " ".join(str(d.get("anchor", "")).split())[:52]
        note = " ".join(str(d.get("note", "")).split())[:58]
        flag = "**" if d["decision"] in ("unresolved", "dropped") else ""
        rows.append(f"| {anchor} | {flag}{d['decision']}{flag} | {d.get('match', '')} "
                    f"| {d.get('asset', '')} | {note} |")
    dropped = v.get("dropped") or []
    if dropped:
        rows += ["", "Dropped for lack of room:", ""] + [f"- {x}" for x in dropped]
    return rows + [""]


def _dist(d: dict) -> str:
    return " / ".join(f"{d[k]:.2f}" for k in ("p25", "p50", "p75", "max"))


def _pacing(pacing: dict | None) -> list[str]:
    """Kept pause lengths against the source's, and the quiet each cut reason
    leaves at its seam. Facts for the editor's ear; never a verdict."""
    if not pacing:
        return []
    rows = ["## Pacing", ""]
    pauses = pacing.get("pauses") or {}
    if pauses:
        rows += [f"Pauses of at least {pacing.get('pause_above', 0.0):.2f}s in the source, by the boundary they sit on "
                 "(seconds, p25 / p50 / p75 / max).", "",
                 "| boundary | n | source | kept |", "|---|---|---|---|"]
        rows += [f"| {b} | {p['count']} | {_dist(p['source'])} | {_dist(p['output'])} |" for b, p in pauses.items()]
        rows.append("")
    seams = pacing.get("seam_quiet") or {}
    if seams:
        rows += ["Median speechless audio kept at a seam, by cut reason.", "",
                 "| cut reason | joins | median quiet (s) |", "|---|---|---|"]
        rows += [f"| {r} | {s['count']} | {s['median']:.2f} |" for r, s in seams.items()]
        rows.append("")
    return rows



def build_report(
    *, plan: Plan, notes: dict, checks: list[CheckResult], status: str, render_cmd: str | None,
    question: str | None = None, pacing: dict | None = None,
) -> str:
    src = plan.sources[plan.main]
    removed = sum(c.length for c in plan.cuts)
    rows = ["# talkinghead report", ""]
    rows.append(f"- status: **{status}**")
    rows.append(f"- source: {src.path}")
    rows.append(f"- source duration: {src.duration:.2f}s; output: {plan.output_duration:.2f}s; removed: {removed:.2f}s in {len(plan.cuts)} cuts")
    prov = plan.provenance
    if prov:
        t = prov.get("transcript", {})
        rows.append(f"- transcript: {t.get('backend')} ({t.get('words')} words); floor {prov.get('floor_dbfs')} dBFS")
        rows.append(f"- audio sha256: {prov.get('audio_sha256', '')[:16]}; talkinghead {prov.get('talkinghead')}")
    rows.append("")
    if question:
        rows += ["## Decision needed", "", "```", question, "```", ""]
    rows += ["## Decisions", ""]
    for key, d in plan.decisions.items():
        if d.get("source") == "user":
            rows.append(f"- {key}: {d.get('value')}s (supplied by the user)")
        else:
            votes = ", ".join(f"{k}={v}" for k, v in sorted(d.get("votes", {}).items(), key=lambda kv: -kv[1]))
            rows.append(f"- {key}: {d.get('chosen')} (votes: {votes})")
            for why in d.get("why", [])[:3]:
                rows.append(f"    - {why}")
    rows.append("")
    rows += ["## Stage notes", ""]
    for stage, n in notes.items():
        rows.append(f"- {stage}: " + ", ".join(f"{k}={v}" for k, v in n.items() if k not in ("prompt_sha", "coherence_prompt_sha")))
    rows.append("")
    rows += _plan_card(notes)
    rows += _retake_decisions(notes)
    rows += _asset_decisions(notes)
    rows += ["## Verification", ""]
    rows += [c.line() for c in checks] or ["- not run"]
    rows.append("")
    rows += _pacing(pacing)
    rows += ["## Cuts", "", "| start | end | dur | reason | votes | text |", "|---|---|---|---|---|---|"]
    for c in plan.cuts:
        votes = c.provenance.get("votes", "")
        rows.append(f"| {_fmt(c.start)} | {_fmt(c.end)} | {c.length:.2f} | {c.reason} | {votes} | {' '.join(c.text.split())[:80]} |")
    rows.append("")
    if render_cmd:
        rows += ["## Render", "", "```", render_cmd, "```", ""]
    return "\n".join(rows)
