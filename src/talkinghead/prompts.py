"""Prompt builders. Every prompt = editorial guidelines (stable prefix, cached
by the provider) + the whole transcript + one task block naming the ids the
judge may answer with. Judges never see or emit timestamps as answers."""

from __future__ import annotations

from .audio import RefinedWord
from .semantic import Sentence


def word_id(i: int) -> str:
    return f"w{i:04d}"


def sentence_table(sents: list[Sentence]) -> str:
    return "\n".join(f"{s.sid} | {s.start:.2f}-{s.end:.2f} | {s.text}" for s in sents)


def transcript_text(words: list) -> str:
    """A plain word stream: the editorial judge reads only the FINAL cut, so
    there are no ids and no source timestamps to leak cut material."""
    return " ".join(str(w.text) for w in words)


def annotated_sentence_table(sents: list[Sentence], groups: list[list[int]], words: list[RefinedWord]) -> str:
    """Sentences with every word tagged by id, for word-range answers."""
    lines = []
    for s, idx in zip(sents, groups):
        tagged = " ".join(f"[{word_id(i)}]{words[i].text}" for i in idx)
        lines.append(f"{s.sid} | {s.start:.2f}-{s.end:.2f} | {tagged}")
    return "\n".join(lines)


def system_prompt(guidelines: str) -> str:
    return (
        "You are the editor of a talking-head YouTube video. You work from a word-level "
        "transcript and answer ONLY with the JSON object the task asks for, naming the ids "
        "the task offers. You never invent ids or timestamps.\n\n" + guidelines
    )


def with_preferences(prompt: str, preferences: str) -> str:
    """Lead a task prompt with the owner's approved editing preferences.

    They travel in the prompt body, not a system prompt, so every judge
    adapter receives them and the per-sample cache key changes with them.
    No preferences leaves the prompt byte-identical, so earlier answers replay.
    """
    if not preferences.strip():
        return prompt
    return f"--- OWNER EDITING PREFERENCES (follow these) ---\n{preferences.strip()}\n\n{prompt}"


def _topic_line(topic: str | None) -> str:
    return f"Video topic (from the owner): {topic}\n\n" if topic else ""


def cold_open_prompt(sents: list[Sentence], candidates: list[Sentence], topic: str | None) -> str:
    cand = "\n".join(f"- {s.sid}: {s.text[:160]}" for s in candidates)
    return (
        _topic_line(topic)
        + "TASK: COLD-OPEN\n"
        "Pick the sentence where the finished video should start (the true start, per the "
        "guidelines: the take the owner committed to and continued from; everything before it is "
        "warmup or rehearsal). Answer with one of the candidate ids below, or KEEP_HEAD if the "
        "video should start on its very first word.\n\n"
        f"--- CANDIDATES ---\n{cand}\n\n"
        f"--- FULL TRANSCRIPT ---\n{sentence_table(sents)}\n"
    )


COLD_OPEN_SCHEMA = {
    "type": "object",
    "properties": {"true_start": {"type": "string"}, "why": {"type": "string"}},
    "required": ["true_start", "why"],
    "additionalProperties": False,
}


def outro_prompt(sents: list[Sentence], candidates: list[Sentence], topic: str | None) -> str:
    cand = "\n".join(f"- {s.sid}: {s.text[:160]}" for s in candidates)
    return (
        _topic_line(topic)
        + "TASK: OUTRO\n"
        "Pick the LAST sentence the finished video should contain (the landing take or the clean "
        "sign-off, per the guidelines). Everything after it is dead air, mumbling, or an abandoned "
        "second ending and is cut. Answer with one of the candidate ids below.\n\n"
        f"--- CANDIDATES ---\n{cand}\n\n"
        f"--- FULL TRANSCRIPT ---\n{sentence_table(sents)}\n"
    )


OUTRO_SCHEMA = {
    "type": "object",
    "properties": {"last_sentence": {"type": "string"}, "why": {"type": "string"}},
    "required": ["last_sentence", "why"],
    "additionalProperties": False,
}


def retake_prompt(table: str, topic: str | None) -> str:
    return (
        _topic_line(topic)
        + "TASK: RETAKES\n"
        "Find live retakes and stumbles per the guidelines: an attempt he abandoned and then said "
        "again. For each, give the word range of the ABANDONED attempt only (from the first word to "
        "the last word to remove, inclusive, using the [wNNNN] ids). Keep the correction. When he "
        "marks the restart out loud (\"sorry\", \"no wait\", \"hold on\", \"scratch that\"), that "
        "marker is part of the abandoned attempt: the range must END at the marker, not before it, "
        "so the apology goes with the take it belongs to. Include clear single-word and short "
        "within-sentence self-corrections, even without a restart marker or pause. Remove the "
        "wrong word or abandoned short phrase and any repeated words belonging to that attempt, "
        "while preserving the corrected wording and its shared subject. Keep intentional emphasis, "
        "list items, and contrasts whose meaning depends on both versions. A spoken correction "
        "is not a retake when the speaker explains why the distinction matters: keep both versions "
        "and the explanation. For example, 'not the cache, I mean the database, and that "
        "distinction matters' is meaningful contrast, not disposable repair. Before proposing "
        "a range, read the surviving sentence: do not strand a negation on the correction, "
        "reverse the claim, or duplicate connecting words. If a clean deletion cannot preserve "
        "the intended meaning, keep the passage. Do not list pauses, fillers, or content you "
        "merely find weak; those are other tasks. Empty list when there are none.\n\n"
        f"--- TRANSCRIPT (word ids in brackets) ---\n{table}\n"
    )


RETAKE_SCHEMA = {
    "type": "object",
    "properties": {
        "retakes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from_word": {"type": "string"},
                    "to_word": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["from_word", "to_word", "why"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["retakes"],
    "additionalProperties": False,
}

CONTENT_REASONS = ["redundant", "false-start", "self-talk", "digression", "no-value", "warmup"]


def content_prompt(sents: list[Sentence], topic: str | None) -> str:
    return (
        _topic_line(topic)
        + "TASK: CONTENT\n"
        "Read the whole transcript as prose and apply the guidelines' post-hook standard. List the "
        "sentences (by id) that do not earn their runtime, grouped into passages, each with a reason "
        f"from {CONTENT_REASONS} and a one-line why. Cut whole sentences only. When unsure, do not "
        "list it. Empty list when everything earns its place.\n\n"
        f"--- TRANSCRIPT ---\n{sentence_table(sents)}\n"
    )


CONTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "cuts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ids": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string", "enum": CONTENT_REASONS},
                    "why": {"type": "string"},
                },
                "required": ["ids", "reason", "why"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["cuts"],
    "additionalProperties": False,
}

COHERENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "vetoes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "why": {"type": "string"}},
                "required": ["id", "why"],
                "additionalProperties": False,
            },
        },
        "extends": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "add_ids": {"type": "array", "items": {"type": "string"}},
                    "why": {"type": "string"},
                },
                "required": ["id", "add_ids", "why"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["vetoes", "extends"],
    "additionalProperties": False,
}


def marker_window(sents, groups, words, marker_word: int, *, context: int = 2) -> str:
    """The marker's sentence plus `context` sentences either side, word-tagged.

    Pass B adjudicates one candidate at a time: a focused window is where the
    judge is reliable, unlike a 4361-word sweep.
    """
    home = next((k for k, idx in enumerate(groups) if marker_word in idx), None)
    if home is None:
        return ""
    lo, hi = max(0, home - context), min(len(groups), home + context + 1)
    return annotated_sentence_table(sents[lo:hi], groups[lo:hi], words)


def marker_prompt(window: str, marker: str, tier: str, topic: str | None) -> str:
    bias = (
        "This is a STRONG retake marker. The default is to CUT: answer \"keep\" only if this is "
        "clearly not a retake -- for example he is addressing the audience (\"sorry guys, this is "
        "not your fault\"), or the correction only makes sense if the viewer hears the wrong "
        "version first."
        if tier == "strong" else
        "This is a WEAK marker that is usually ordinary speech -- a discourse filler (\"but I mean, "
        "you're not looking at this anyway\") or a definition (\"by vibecode, I mean...\"). The "
        "default is to KEEP: answer \"cut\" only if there is a plain abandoned attempt followed by "
        "a restatement of the same point."
    )
    return (
        _topic_line(topic)
        + "TASK: MARKER\n"
        f"The word {marker!r} appears in the passage below. Decide whether it marks a live retake "
        "-- an attempt he abandoned and then said again.\n\n"
        f"{bias}\n\n"
        "When you answer \"cut\", give the word range of the ABANDONED attempt. The marker "
        f"({marker!r}) is part of the abandoned attempt and MUST be inside the range; the "
        "correction that follows it is kept. When you answer \"keep\", omit the range.\n\n"
        f"--- PASSAGE (word ids in brackets) ---\n{window}\n"
    )


MARKER_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["cut", "keep"]},
        "from_word": {"type": "string"},
        "to_word": {"type": "string"},
        "why": {"type": "string"},
    },
    "required": ["verdict", "why"],
    "additionalProperties": False,
}


VISUAL_TYPES = ["quote_card", "meme", "technical_evidence"]


def editorial_prompt(transcript: str, topic: str | None, max_chapters: int) -> str:
    """One hierarchical plan over the final cut: the argument, the chapters,
    and the sparse mid-chapter visuals, answered together so the visuals obey
    the chapters instead of being chosen by keyword."""
    return (
        _topic_line(topic)
        + "TASK: EDITORIAL PLAN\n"
        "Below is the FINAL CUT of a talking-head video exactly as the viewer hears it, in order, "
        "with retakes and cuts already removed. Read the whole transcript first and work out the "
        "central argument and the story beats before you answer. Then make ONE plan for the "
        "video's on-screen graphics.\n\n"
        "You answer as the channel's editor, not as a brand. The voice is brazen, technically "
        "experienced, direct and funny, steeped in older-internet and 2000s meme culture. "
        "Technical credibility beats a laugh. Never sound corporate or motivational: no \"unlock\", "
        "\"supercharge\", \"game-changer\" or \"journey\", no generic creator slogans, no invented "
        "jokes, no inspirational paraphrase. Never explain a joke, and never put up a card that "
        "just restates the sentence under it.\n\n"
        "Return exactly three things in one JSON object:\n\n"
        "1. central_argument: one sentence stating what the video is actually arguing. Specific "
        "and arguable, in the host's own terms, not a topic summary.\n\n"
        f"2. chapters: the video's broad sections, in order, at most {max_chapters}. Three to five "
        "substantial chapters is usual; the maximum is a ceiling, not a quota, and fewer is fine. "
        "If the host outlines what he will cover near the beginning, follow his outline. Each "
        "chapter has:\n"
        "   - title: broad but specific, short and scannable, channel-native. Say what the section "
        "delivers, not what it is about. Never vague (\"The setup\"), never a dramatic label for an "
        "ordinary claim, never a slogan.\n"
        "   - anchor_text: 4 to 10 words copied VERBATIM from the transcript at the instant the "
        "chapter STARTS. Copy them exactly as written, punctuation and stumbles included; do not "
        "tidy or paraphrase them or the anchor will not be found.\n"
        "   - purpose: one short line on the job the chapter does in the argument.\n"
        "   The chapters become a roadmap card shown near the start and again at each chapter "
        "start with the current row highlighted, so the titles must stand on their own.\n\n"
        "3. visuals: a SPARSE set of mid-chapter overlays. There is no quota: several clean "
        "minutes with nothing on screen are a valid answer, and one strong visual beats three "
        "filler ones. Use at most one visual per beat. Each visual has:\n"
        "   - chapter: the 1-based number of the chapter it sits inside.\n"
        "   - anchor_text: 4 to 10 words copied VERBATIM from the transcript at the moment the "
        "visual should appear.\n"
        "   - type: one of quote_card, meme, technical_evidence.\n"
        "     - quote_card: a payoff or summary beat where the host's own wording IS the payoff. "
        "Set quote to that wording VERBATIM, one or two readable lines. Never a sentence that "
        "merely restates the point; only quote what already lands.\n"
        "     - meme: amplify a funny, sarcastic, frustrated, self-deprecating or absurd beat that "
        "is ALREADY happening. visual_brief describes the image to source, not a second punchline. "
        "Never invent a joke the host did not make.\n"
        "     - technical_evidence: a claim that is materially better seen than heard (a real "
        "chart, benchmark, price, terminal, repository). visual_brief describes the exact source. "
        "Never decoration, never a logo, never a generic screenshot.\n"
        "   - quote: the verbatim wording for a quote_card; empty string for meme or "
        "technical_evidence.\n"
        "   - visual_brief: what the overlay shows, in words someone could search for.\n"
        "   - editorial_reason: one line on why this beat earns a visual.\n"
        "   - hold_sec: 3 to 20 seconds, matching how long the beat lasts.\n\n"
        f"--- FINAL CUT TRANSCRIPT ---\n{transcript}\n"
    )


EDITORIAL_SCHEMA = {
    "type": "object",
    "properties": {
        "central_argument": {"type": "string"},
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "anchor_text": {"type": "string"},
                    "purpose": {"type": "string"},
                },
                "required": ["title", "anchor_text", "purpose"],
                "additionalProperties": False,
            },
        },
        "visuals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chapter": {"type": "integer"},
                    "anchor_text": {"type": "string"},
                    "type": {"type": "string", "enum": VISUAL_TYPES},
                    "quote": {"type": "string"},
                    "visual_brief": {"type": "string"},
                    "editorial_reason": {"type": "string"},
                    "hold_sec": {"type": "number"},
                },
                "required": ["chapter", "anchor_text", "type", "quote", "visual_brief",
                             "editorial_reason", "hold_sec"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["central_argument", "chapters", "visuals"],
    "additionalProperties": False,
}


def research_prompt(items: list[dict], topic: str | None) -> str:
    listing = "\n".join(
        f"- {i}: type={it.get('type', 'technical_evidence')} depicts={it['depicts']!r} "
        f"(said as: {it['anchor_text']!r})" for i, it in enumerate(items))
    return (
        _topic_line(topic)
        + "TASK: RESEARCH\n"
        "Each item below wants a real image rather than a text plate. For a technical_evidence "
        "item, give the single best public URL to screenshot -- an official docs page, pricing "
        "page, model card, benchmark page or repository; prefer the canonical source over a blog "
        "about it. For a meme item, give a stable DIRECT image URL ending in .png, .jpg, .jpeg, "
        ".gif or .webp; a page URL is rejected because a page screenshot is not a meme.\n\n"
        "Omit an item entirely when you do not know a specific URL; a guess that 404s is worse "
        "than no visual at all.\n\n"
        f"--- ITEMS ---\n{listing}\n"
    )


RESEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "url": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["id", "url", "why"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["sources"],
    "additionalProperties": False,
}
