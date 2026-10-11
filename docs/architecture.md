# Architecture

One run is a single pass: input guard -> align -> analyse -> four stages -> edge
finalization -> one render -> machine verification -> visuals overlays -> report.
`run_pipeline` in `src/talkinghead/pipeline.py` is the whole orchestration; every
stage below is a function it calls in order.

Two things make the run reproducible rather than merely repeatable:

- **Nothing is computed twice.** Deterministic results (transcript, audio analysis,
  render) live in content-addressed caches; judged results are memoised per stage and
  every individual LLM sample is stored.
- **The plan is the deliverable.** `plan.json` holds the edit decision list, so a run
  can be re-rendered exactly without re-deciding anything
  (`talkinghead render plan.json --out out.mp4`, `src/talkinghead/render.py`).

## Where state lives

| store | path | keyed on | holds |
|---|---|---|---|
| cache (global) | `~/.cache/talkinghead` (`TALKINGHEAD_CACHE`), `src/talkinghead/store.py` | `<sha256 of the content>/<kind>-<stable_key>`, 24-hex key from the sorted JSON of the params + a version | `transcript`, `silences`, `moans`, `mouth_noises`, `unclassified_voice` (voiced speech-level audio no word covers), `speech_mask`, `speech_probe` |
| run dir | `<video stem>.talkinghead/` next to the video, or `--out-dir` (`store.RunDir`) | one per recording: `run.json` records the source sha256 and `bind` refuses a different picture, so two cooks never share a directory | `plan.json`, `render.json`, `decisions.json`, `run.json`, `preferences.md`, `audio/`, `llm/`, `stages/`, `verify/`, plus `master.mov` and `align.json` when `--audio` was given |
| stage memos | `<run>/stages/<stage>.json` | `stable_key(inputs, STAGE_VERSION)`: transcription + **the config fields that stage reads** (`pipeline.STAGE_CONFIG_KEYS`) + guidelines + judge name + topic + decisions + preferences (only when non-empty) | the stage's cuts and notes, plus `stages/<stage>.superseded.json` when a re-key replaced one |
| LLM samples | `<run>/llm/<prompt sha>.prompt.txt` and `<prompt sha>-<i>.json` (`judge.LLMCache`) | the prompt **and the judge name**, so switching models never replays another model's answers | every prompt and every sampled answer |

`STAGE_VERSION` and `VISUALS_STAGE_VERSION` (`pipeline.py`) and `ANALYSIS_VERSION`
(`analysis.py`) are bumped when a stage's logic changes, which invalidates its memos
without touching the input hashes. `--fresh` drops stage memos and the render memo for
one run; LLM samples stay cached, because resampling a nondeterministic model to get a
different verdict is the opposite of what a rerun should do.

Stage memos key on the config a stage READS, not on the whole dict
(`pipeline.STAGE_CONFIG_KEYS`; `tests/v2/test_stages.py` checks the lists against the
`config.<field>` reads in each stage module). A waveform or probe setting therefore cannot
recompute a judge's work, and a pause policy change re-derives the pauses without asking
the judge again. The stages not listed (visuals) keep the full dict. When a memo IS
replaced because its key moved, the previous file is copied to
`stages/<stage>.superseded.json` first and the stage is named in the report's stage notes:
the memos are hand-editable, so a re-key must not discard manual cuts without a trace.

## The stages

| # | stage | decides | driver | writes |
|---|---|---|---|---|
| 0 | input guard | video/audio duration agreement, or the actual audio duration in audio-only mode | deterministic | - |
| 1 | align (only with `--audio`) | where the external recording sits on the video timeline, plus its clock drift | deterministic | `master.mov`, `align.json` |
| 2 | analyse | words, native sound events, protected laughter, silences, boundaries and acoustic noise candidates | deterministic | `audio/audio-16k.wav` + cache entries |
| 3 | structure | the true start (cold open) and the outro sentence | judge, supermajority | stage memo |
| 4 | retakes | which abandoned attempts and marked restarts are removed | judge, two passes | stage memo |
| 5 | content | which whole sentences do not earn their runtime (and a coherence veto) | judge + coherence pass | stage memo |
| 6 | pauses | pause trims, fillers, similarity-detected restarts, mouth noises | deterministic | stage memo |
| 7 | edges | final span of every cut | deterministic | `plan.json` |
| 8 | render | picture and mastered audio, or mastered audio alone | deterministic | `final.mp4` or `final.wav`, `render.json` |
| 9 | verify | did the render keep the plan's promises | deterministic, declarative checks | `verify/facts.json` |
| 10 | visuals | what to overlay, where, from which asset | judge | `final-visuals.mp4`, `verify/visuals/` |
| 11 | report | - | deterministic | `report.md` |

### 0. Input guard

Video analysis calls `transcribe.check_av_mismatch` before anything else.
If the video and audio durations differ by more than `Config.av_mismatch_tolerance`, the run stops unless `--allow-av-mismatch` is supplied.
Audio-only analysis measures the real audio stream and does not manufacture a video duration or picture.

### 1. Align

`src/talkinghead/align.py` (ported from the retired `audiolign`). `align()` measures and
then remuxes; `analyze()` measures only. The measurement is a coarse pass (30 ms RMS
envelopes, full cross-correlation) followed by GCC-PHAT on 30 s windows spread across
the timeline, median-filtered and fitted with a line whose slope is the drift in ppm.
The video stream is copied untouched unless a display rotation has to be baked into the
pixels; the external audio is encoded to AAC 256k. `--fix-drift` adds an `atempo`
time-stretch of `1/(1+drift)` so the recording's clock matches the video's, and the tail is
padded when the audio runs out first.

In a run, the result is `master.mov` plus `align.json` in the run dir, and every later
stage reads the master as its single source. The memo key is (video sha256, audio
sha256, `fix_drift`).

### 2. Analyse

Still in `analysis.analyze`, still no LLM:

- `transcribe.prepare_audio` extracts canonical 16 kHz mono PCM once to `<run>/audio/audio-16k.wav` and caches a structured `Transcript`.
  Three backends: Scribe v2 (ElevenLabs or OpenRouter), Parakeet and Whisper. The transcriber has no built-in default: it is named by `TALKINGHEAD_TRANSCRIBER`, `--transcriber` or the config file.
  The transcript includes words and native audio events rather than dropping non-word sounds.
  Saved Scribe responses replay only after their original request audio matches the current canonical PCM, with response identity included in the replay cache version.
  Parakeet's `TDT_EMISSION_DELAY_FRAMES` override still joins its cache key.
- `audio.envelope` / `audio.noise_floor` give the dB and zero-crossing envelope and the
  file's floor; `transcribe.detect_silences` finds silences from the waveform (a
  speech-referenced threshold, never from gaps between ASR words);
  `audio.refine_word_bounds` moves each word's edges to where the voice actually starts
  and stops.
- `bursts.detect_voiced_bursts` finds vocalisations the ASR swallowed into word spans;
  `noise.detect_mouth_noises` labels smacks, breaths and throat clears with the dB + ZCR
  classifier calibrated in `docs/noise-detector-spec.md`.
- `audio.unclassified_voice` collects the voiced, speech-level spans no refined word and
  no already-claimed event covers: audio no transcript accounts for, which no word-driven
  rule can see. They are UNCLASSIFIED rather than proven words (level and voicing gates
  pass a hum as easily as a word), so nothing promotes one: when a speech model is
  available it may only DEMOTE a span (`audio.filter_masked`, run once and only when there
  is a candidate), and probe text is evidence, never a cut.
  `analysis.probe_unclassified_voice` re-transcribes each surviving span (longest first,
  capped by `speech_probe_max`, successful decodes cached) with the OTHER backend, so the
  report can name what was heard instead of only measuring it.

Everything lands in `Context` (`context.py`), which is frozen for the rest of the run.

Complete laughter, chuckles and giggles become `Context.protected_spans`, not noise cuts.
Unknown native sound events are kept conservatively; recognized sighs, breaths and mouth sounds remain removal candidates subject to the ordinary guards.
Protected intervals reach final edge placement so pause, noise and neighboring retake cuts cannot erase quieter parts of a laugh.


### 3. Structure (judge)

`stages/structure.py`. The candidates come from code: `candidates.cold_open_candidates`
is every sentence starting inside `cold_open_window_sec` (300 s), plus `KEEP_HEAD`;
`outro_candidates` is every sentence in the last `outro_window_sec` (120 s).
The judge never emits a timestamp - it picks a sentence id, and code turns that id into
the cut. `judge.vote` takes `Config.votes` (5) samples and `vote_scalar` needs
`Config.agree` (4) identical answers.

No supermajority raises `NeedsDecision` (`context.py`), which the CLI reports and exits 3
with the flag that answers it (`--cold-open`, `--outro`). A supplied answer is written to
`decisions.json` and returned as `source: "user"` from then on, so the question is asked
at most once per run dir. `structure.run` returns the `cold-open` and `outro` cuts.

### 4. Retakes (judge, two passes)

`stages/retakes.py`.

- **Pass A** sweeps the whole transcript once and aggregates the five samples by *span*
  (`spans.cluster_spans`) rather than by word, because per-word voting made agreeing
  samples with edges one word apart cancel each other out.
- **Pass B** is mechanical detection followed by focused judgment: `markers.find_markers`
  finds every spoken restart marker, and the judge adjudicates each marker alone with its
  surrounding sentences (`marker_votes` 3, `marker_agree` 2). STRONG markers (an apology,
  "no wait", "start over") default to cut and require the judge to argue for keeping;
  WEAK markers ("I mean", "actually") keep the standing "when unsure, keep" default. The
  tiers and their evidence live in `editorial/guidelines.md`.

Both passes produce `Cut`s with `reason="retake"` and the vote count in `provenance`.
Clear single-word and short within-sentence corrections are included even without a spoken restart marker.
The abandoned wording is removed while the corrected wording and shared subject remain.
Intentional emphasis, valid lists and meaningful contrasts are kept.


### 5. Content (judge + coherence)

`stages/content.py`. The judge grades the sentence table; a sentence is removed when
`Config.agree` of `Config.votes` samples name it. Removal is capped twice - per contiguous
passage (`content_max_cut_sec`, 30 s) and in total (`content_max_fraction`, 10 % of the
source) - and the weakest-voted passage is dropped first when the budget is exceeded.
Surviving cuts then go through `coherence.build_coherence_prompt`: a second judged pass
that can veto a cut or extend it, both by the same supermajority. `--no-content` skips the
whole stage.

### 6. Pauses (deterministic)

`stages/pauses.py`, the only fully mechanical stage. `candidates.machinery_candidates`
turns the calibrated derive pass into three kinds of cut: pause trims (interior pauses keep
a boundary-typed, length-scaled amount split across the two edges, leading and trailing
silence goes entirely), filler cuts for every hesitation token emitted as a word (`min_filler_dur`, default 0, sets an optional length floor), and `restart` cuts - an abandoned sentence that the next one
restates, found by token similarity. The mouth-noise cuts computed during analysis are
appended unchanged.

A pause of raw length `raw` keeps `min(raw - 0.02, cap(boundary), pause_keep +
pause_keep_ratio * (raw - pause_above))`. `pause_keep` (0.30 s) was chosen on a reference narrative cut, where
~0.15 s of room on each side of a join was the difference between a smooth splice and a clipped one; it is now the amount kept at the
trigger, and longer pauses keep `pause_keep_ratio` (0.25) of the excess. A fixed value made
every pause the same length, so an 11 s chapter break became 0.6 s.

The cap depends on the boundary the pause sits on (`derive.pause_boundary`, `pause_cap`):
mid-sentence 0.40 s, clause 0.45 s, sentence end 0.55 s, and a sentence end after a raw
pause of at least `pause_topic_above` (1.5 s) counts as a topic change, 0.80 s. The caps
sit near the speaker's own p75-p90 natural pauses, so a topic change keeps a longer beat
than a hesitation.

### 7. Edges (deterministic)

`edges.finalize_cuts` receives the union of all four stages and is the only place cut
edges are finalized.
The pass merges overlapping cuts, protects kept speech, steers away from vocalisations, and places edges in genuine quiet.
Valley placement is followed by tight within-sentence repairs, completed-sentence tail retention and non-repair clause pacing.
Flow guards and optional frame snapping run before the final protected-event reconciliation and micro-filler safety check.

Within-sentence repair seams retain phoneme protection without adding sentence or clause pacing.
Completed-sentence seams retain 80 ms of real source tail, with outward rounding on a frame grid.
Protected sound intervals are reconciled after cut fusion and snapping, so preserving a laugh is not undone by a later pass.


Edge clearance depends on the cut. `silence` and `noise` cuts keep `pad = pause_keep / 2`
(0.15 s) from the neighbouring word, so a trimmed pause leaves room on both sides.
Word-removing cuts (`content`, `retake`, `cold-open`, `outro`, `join`, `filler`) only keep
`splice_pad` (0.03 s) clear of kept voiced audio, because the join is crossfaded and the
edge is placed by ear-relevant quiet rather than by a fixed margin.
`snap_to_valleys` moves each such edge to the quietest envelope frame within
`splice_valley_window` (0.12 s) of its word boundary. Candidates sit on the frame grid
because the final `snap_to_frames` would otherwise move a hand-picked valley off its
minimum.
Scribe supplies native timestamps; local Parakeet stamps may be coarser, so the acoustic guard remains useful for either backend.
Within-sentence retakes cannot move inward of the abandoned word or acquire a clause/sentence pacing pause.
`keep_sentence_tail` retains the 80 ms completed-sentence source tail where that much source room exists, rounded outward for video.
`keep_splice_quiet` applies the existing 0.18 s clause pacing only to non-repair word cuts.
`keep_splice_lead_in` then moves a cut's end earlier, through quiet audio only and never over a removed word, so `splice_lead_in` (0.06 s) of source sits before the next kept onset. `refine_joins` applies the same rule to the incoming edge.
Natural continuous-speech pauses are not shortened solely to force an 80 ms edit seam.
`drop_voiced_fillers` can then drop a `filler` cut of at most `filler_keep_voiced_max`
(default 0, so off) whose edge still sits on voiced audio, judged by `joins.edge_state`, the same
predicate verify's `join_edges` uses: when enabled, a micro-filler glued into running speech is kept
because the tight splice its removal makes can be more noticeable than the filler.
Retake, content and join cuts stay even with a voiced edge, since they remove real words.

`protect_words` guards two kinds of kept audio, and the difference matters. Words are
always protected, and so are `Context.voices`: the voiced spans no word covers. That
protection is what the waveform buys the edit. On the reference narrative cut a 0.16 s
fragment had no transcript entry at all, and a pause trim spanning it was the only thing
standing between it and deletion. A `silence` or `noise` cut may SPLIT around such a span
(its whole job is removing non-speech, so eating voice is its own bug); an editorial cut
(`content`, `retake`, `cold-open`, `outro`, `join`, `filler`) only has its EDGES kept clear
of it, because splitting a passage the judge chose to remove would ship a fragment of it.
`flow_guard` counts a guarded span as speech too, or it would fuse two trims across one and
delete it. What an editorial cut swallows is reported instead (see `unclassified_voice` in
§9). `Config.voice_guard` turns the whole guard off.

`Context.protected_spans` is separate from the optional unidentified-voice guard.
It preserves full native laughter/unknown-event intervals for every cut reason, including an event inside an otherwise removed passage.
Cut fusion and frame snapping cannot bridge or trim these protected intervals.


`timeline.Plan.from_cuts` then derives the kept segments from the merged cuts and
`plan.json` is written: cuts with reasons and provenance, the segment list, the
structural decisions, the anchors, and the provenance block (hashes, backend, floor,
effective config) that stage memos are keyed on.

### 8. Render (deterministic, once)

`render.render_plan` builds three ffmpeg runs: a picture pass (one CFR decode per source
run, `fps` then chained half-open `select` filters that drop the removed spans, frame-snapped),
an audio pass (segments decoded and joined in numpy, then the broadcast chain - highpass, compressor, de-esser, `loudnorm` to
`Config.loudness_lufs` - applied exactly once), then a stream-copy mux.
The picture pass avoids per-segment trim fan-out, which stalled on a native 4K HLG recording.
The branch queue was a suspected cause; the observed failure was no output growth for 20 minutes.
`render.join_audio` overlap-adds the segments with equal-power (sine/cosine) crossfades of
`splice_crossfade_sec` (0.020 s) at every join, instead of fading each segment to zero,
which dipped ~23 dB where both sides were voiced.
Each side reads a handle of removed source audio past its edge, half the crossfade, so the
two overlap across the cut and the rendered audio length still equals the plan's; a handle
is clamped to half the removed span and half of each adjacent segment so it never reaches a
neighbour's own audio.
Output boundaries are rounded from the running total, so rounding never accumulates, and
the program start and end get a 10 ms fade so the first and last sample cannot click.
The join is numpy rather than an `atrim`/`concat` graph because a graph of ~200 links is
known to hang. The split exists
because ffmpeg 9.0.1 emits digitally silent audio from a combined `concat=v=1:a=1` graph
for some source regions.
Both passes use the same frame-snapped segment boundaries; audio is processed and encoded once.
`docs/render-proof.md` records why closed `between` intervals are unsafe: they remove one extra frame per grid-aligned cut.

Colour is probed from the sources (`color.probe_color`): an HLG/PQ source is re-encoded
as 10-bit with its own primaries/matrix/transfer and `-map_metadata -1`, so a Dolby
Vision record on the source cannot ride into an edited picture; an SDR source keeps the
libx264/yuv420p encode. No tone-map LUT is ever applied implicitly.

The `render.json` memo is keyed on the segment list, the encoder settings and
`render.RENDERER_VERSION` (the version of the render chain itself, which the plan and
settings cannot express): changing an unrelated config value must not re-encode a
finished render, but changing the chain must.

### 9. Verify (deterministic)

`verify.collect_facts` measures the finished file - container/video/audio duration, LUFS,
a fresh transcript of the render, join grades (`joins.check_join`), a two-sided source edge check (`join_edges`:
a chopped voiced word on either side of a cut, judged on the source), residual dead air
(`verify.residual_dead_air`: the source VAD silences that survive the plan; the gap budget
counts only the excess over the pause caps, so pauses the policy keeps on purpose never
count), pacing facts (`verify.pacing_facts`: kept pause lengths against the source's), source-to-output audio timing (`verify.measure_source_timing`), and color preservation (`verify.color_facts`).
It writes the measurements to `verify/facts.json`.
The report's `## Pacing` section prints the pacing facts (kept pause lengths by boundary type, and the quiet each cut reason leaves at its seam) as facts for the editor's ear, never a verdict.

`verify.run_checks` evaluates twelve declarative checks (`CHECKS` in `verify.py`). Fatal:
`duration`, `av_sync`, `loudness`, `cold_open` (first word within 0.5 s), `residual_gaps`,
`render_silence` (measured on the render; the mute threshold is below one 16-bit PCM quantization step), `source_timing` (a
sampled keep's audio more than 60 ms from where the plan put it) and `color` (a source's
HDR colour silently converted, including right tags on an 8-bit stream). Warnings: `joins`, `join_edges`,
`source_similarity` (a sample that no longer correlates with the source, which the
broadcast chain can legitimately cause) and `unclassified_voice` (voiced audio no word
covers that the plan removes, with the re-transcription's text when one ran).

A measurement that could not be taken is never reported as a pass: `source_timing` says
`not measured` (and warns) when no keep is long enough, no window carries alignable audio,
or nothing decodes, and distinguishes that from a run that never had a source path to
measure. Any failing fatal check writes `report.md` with status FAILED and raises
`PipelineError`, which the CLI turns into exit 1. The visuals pass never runs after a fatal
check.

### 10. Visuals (judge, second pass)

`visuals/stage.plan_visuals` chooses anchors, cards and the chapter plan card, memoised
under `VISUALS_STAGE_VERSION` because it is judged and therefore must not be re-decided;
screenshot research (`--no-research`) and chapter selection are single-sample judge calls.
`visuals/render.render_visuals` composites the overlays in a second ffmpeg run over
`final.mp4` with **the audio stream copied**, which is why every check above keeps its
meaning and why `final.mp4` stays shippable without the overlays. The base's colour is
probed: an SDR cut keeps the libx264/yuv420p encode, an HLG/PQ cut keeps its 10 bits and
its tags, and the SDR overlay assets are converted into that system with the cube from
`color.write_sdr_to_hdr_cube` (SDR white to BT.2408 graphics white) rather than tagged as
if they were already HDR. With no overlays the pass is a stream copy, so there is nothing
to convert.
`visuals/verify.verify_visuals` proves that promise: the audio is untouched and the
duration did not move, plus a contact sheet of frames at overlay midpoints and in clean
control gaps, all under `verify/visuals/`. `--no-visuals` skips the stage.

### 11. Report

`report.build_report` writes `report.md`: status, source and duration accounting,
decisions with their vote tallies, stage notes, per-marker retake decisions, how each
visual anchor got its asset, the verification table, every cut with its reason and votes,
and the render command. `talkinghead inspect OUT_DIR` prints it.

## Judge adapters

`judge.py` defines one protocol (`decide(prompt, schema) -> dict`) and four
implementations: `ShellJudge` (any command: prompt on stdin, JSON on stdout; it leads every
prompt with the system prompt (role + guidelines) and the schema contract, and retries on invalid JSON),
`ClaudeCodeJudge` (headless `claude -p`, subscription, structured output),
`FallbackJudge` (switches to a second judge when the primary reports a usage limit) and
`FakeJudge` for tests. `Config.judge` selects the adapter; `Config.model` rewrites the
command's `--model` token (`judge.with_model`) rather than replacing the command, so
provider pinning and other flags survive an override.

Judges are always asked to classify code-generated candidates or to pick ids from a table
in front of them. No judge answer becomes a timestamp.
