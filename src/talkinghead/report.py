"""One report per run: what was decided and by how many votes, every cut with
its reason and provenance, the verification table, and honest totals (removed
seconds must equal the duration delta)."""

from __future__ import annotations

from collections import Counter

from .isolate import report_facts as isolation_facts
from .timeline import Plan
from .verify import CheckResult


def _fmt(t: float) -> str:
    m, s = divmod(t, 60.0)
    return f"{int(m):02d}:{s:05.2f}"


def _polish(plan: Plan, notes: dict) -> list[str]:
    """What the polish stage removed: gap tightening and disfluency cuts, read off the final plan's cuts."""
    if "polish" not in notes:
        return []
    gaps = [c for c in plan.cuts if c.reason == "gap"]
    disf = sorted((c for c in plan.cuts if c.reason == "polish"), key=lambda c: c.start)
    words = sum(c.provenance["words"][1] - c.provenance["words"][0] + 1 for c in disf if "words" in c.provenance)
    kinds = Counter(c.provenance.get("kind", "") for c in disf)
    judged = notes["polish"]
    rows = ["## Polish", "",
            f"- gaps: {len(gaps)} cuts, {sum(c.length for c in gaps):.2f} s of silence removed between kept words",
            f"- disfluencies: {len(disf)} cuts, {words} words, {sum(c.length for c in disf):.2f} s removed",
            "- kinds: " + (", ".join(f"{k} {v}" for k, v in kinds.most_common()) or "none"),
            f"- judge: {judged.get('windows', 0)} windows, {judged.get('proposed', 0)} spans proposed, "
            f"{judged.get('dropped_bad_ids', 0)} bad ids, {judged.get('dropped_overlong', 0)} overlong, "
            f"{judged.get('clusters', 0)} clusters kept by agreement", "",
            "First disfluency cuts, in source order:", "",
            "| start | end | kind | votes | text |", "|---|---|---|---|---|"]
    for c in disf[:15]:
        rows.append(f"| {_fmt(c.start)} | {_fmt(c.end)} | {c.provenance.get('kind', '')} | "
                    f"{c.provenance.get('votes', '')} | {' '.join(c.text.split())[:90]} |")
    rows += [""]
    if "output" in judged:
        rows += _output_pass(judged["output"])
    return rows + [""]


def _isolation(plan: Plan) -> list[str]:
    """The isolated track: the trim, the mix and every word the unit swallowed and the original put back."""
    if not plan.isolation:
        return []
    data = isolation_facts(plan)
    rows = ["## Isolation", "",
            f"- latency trimmed: {data['lag_samples']} samples ({data['lag_ms']:.2f} ms), measured at "
            f"{len(data['points'])} points",
            f"- mix: {data['mix']:.2f} isolated, {1 - data['mix']:.2f} original; dropout gain {data['gain']:.3f}",
            f"- helper {data['helper']}; track sha256 {plan.isolation['track']['sha256'][:16]}",
            f"- repaired words: {len(data['repaired'])}", ""]
    if data["repaired"]:
        rows += ["| source | output | word | ratio |", "|---|---|---|---|"]
        for r in data["repaired"]:
            out = "cut" if r["output_start"] is None else _fmt(r["output_start"])
            rows.append(f"| {_fmt(r['start'])} | {out} | {r['text']} | {r['ratio']:.3f} |")
        rows.append("")
    return rows


def _table(head: list[str], items: list[list[str]], limit: int = 15) -> list[str]:
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    rows += ["| " + " | ".join(item) + " |" for item in items[:limit]]
    if len(items) > limit:
        rows.append(f"| ... {len(items) - limit} more | | | |")
    return rows + [""]


def _output_pass(output: dict) -> list[str]:
    """The output pass: each round's counts, the findings it skipped, what the final render still shows, and the
    findings the judge kept on purpose."""
    rounds = output.get("rounds") or []
    rows = ["### Output pass", ""]
    if output.get("skipped"):
        rows += [f"- skipped: {output['skipped']}", ""]
    for r in rounds:
        rows.append(f"- round {r.get('round')}: {r.get('findings', 0)} findings, {r.get('cut', 0)} cut, "
                    f"{r.get('keep', 0)} judged keep, {r.get('skipped', 0)} skipped "
                    f"({r.get('votes', '')} votes, {r.get('agree', '')} to agree)")
    rows.append("")
    skipped = [x for r in rounds for x in r.get("rows", []) if str(x.get("decision", "")).startswith("skipped")]
    if skipped:
        rows += ["Confirmed but not cut (the span could not be mapped to one main-source span):", ""]
        rows += _table(["output time", "text", "reason"],
                       [[_fmt(x["start"]), " ".join(str(x["heard"]).split())[:60], x["decision"]] for x in skipped])
    residual = output.get("residual")
    if residual is None:
        rows += ["- residual check: not measured (the render was not transcribed)", ""]
    elif not residual:
        rows += ["- residual check: none left in the render's transcript outside the judged-keep list", ""]
    else:
        rows += [f"Residual in the final render, not judged keep ({len(residual)}):", ""]
        rows += _table(["output time", "kind", "text"],
                       [[_fmt(x["start"]), x["kind"], " ".join(str(x["heard"]).split())[:60]] for x in residual])
    keep = output.get("keep") or []
    if keep:
        rows += [f"Judged keep, left in on purpose ({len(keep)}):", ""]
        rows += _table(["output time", "kind", "text"],
                       [[_fmt(x["start"]), x["kind"], " ".join(str(x["text"]).split())[:60]] for x in keep])
    return rows


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


def _blocks(plan: Plan) -> list[str]:
    """Topic blocks: where each came from, where it landed in the cut, and why it was kept or dropped."""
    blocks = plan.provenance.get("blocks")
    if not blocks:
        return []

    def clean(text: object, width: int) -> str:
        return " ".join(str(text).replace("|", "/").split())[:width]

    rows = ["## Blocks", "", "| id | source | output | dur | title | interest/relevance | decision | reason | front |",
            "|---|---|---|---|---|---|---|---|---|"]
    kept_out = dropped_src = 0.0
    n_kept = n_dropped = 0
    for b in blocks:
        s0, s1 = b["source"]
        out = b["output"]
        if b["decision"] == "drop":
            n_dropped += 1
            dropped_src += s1 - s0
        else:
            n_kept += 1
            if out:
                kept_out += out[1] - out[0]
        out_txt = f"{_fmt(out[0])}-{_fmt(out[1])}" if out else "-"
        rows.append(f"| {b['id']} | {_fmt(s0)}-{_fmt(s1)} | {out_txt} | {s1 - s0:.1f}s | {clean(b['title'], 60)} | "
                    f"{b['interest']}/{b['relevance']} | {b['decision']} | {clean(b['reason'], 80)} | "
                    f"{b['front_rank'] or ''} |")
    rows += ["", f"Kept: {kept_out / 60:.2f} min of output in {n_kept} blocks. "
                 f"Dropped: {dropped_src / 60:.2f} min of source in {n_dropped} blocks.", ""]
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
        if stage == "isolation":
            continue
        rows.append(f"- {stage}: " + ", ".join(f"{k}={v}" for k, v in n.items() if k not in ("prompt_sha", "coherence_prompt_sha")))
    rows.append("")
    rows += _plan_card(notes)
    rows += _retake_decisions(notes)
    rows += _asset_decisions(notes)
    rows += ["## Verification", ""]
    rows += [c.line() for c in checks] or ["- not run"]
    rows.append("")
    rows += _pacing(pacing)
    rows += _blocks(plan)
    rows += _polish(plan, notes)
    rows += _isolation(plan)
    rows += ["## Cuts", "", "| start | end | dur | reason | votes | text |", "|---|---|---|---|---|---|"]
    for c in plan.cuts:
        votes = c.provenance.get("votes", "")
        rows.append(f"| {_fmt(c.start)} | {_fmt(c.end)} | {c.length:.2f} | {c.reason} | {votes} | {' '.join(c.text.split())[:80]} |")
    rows.append("")
    if render_cmd:
        rows += ["## Render", "", "```", render_cmd, "```", ""]
    return "\n".join(rows)
