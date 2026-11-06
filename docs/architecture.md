# Architecture

One run is a single pass: input guard -> align -> analyse -> four stages -> edge
finalization -> audio master -> machine verification of the audio -> lock -> visuals planning and head-clip on
source frames -> one picture render with the overlays inside it -> picture verification -> report.
`run_pipeline` in `src/talkinghead/pipeline.py` is the whole orchestration; every
stage below is a function it calls in order.

Two things make the run reproducible rather than merely repeatable:

- **Nothing is computed twice.** Deterministic results (transcript, audio analysis,
  render) live in content-addressed caches; judged results are memoised per stage and
  every individual LLM sample is stored.
- **The plan is the deliverable.** `plan.json` holds the edit decision list, so a run
  can be re-rendered exactly without re-deciding anything: render the audio master
  (`talkinghead render plan.json --audio-only --out audio-master.m4a`), lock it
  (`talkinghead lock`), then render the picture (`talkinghead render plan.json --out out.mp4 --lock edit-lock.json`,
  `src/talkinghead/render.py`). A video render needs the lock.

## Where state lives

| store | path | keyed on | holds |
|---|---|---|---|
| cache (global) | `~/.cache/talkinghead` (`TALKINGHEAD_CACHE`), `src/talkinghead/store.py` | `<sha256 of the content>/<kind>-<stable_key>`, 24-hex key from the sorted JSON of the params + a version | `transcript`, `silences`, `moans`, `mouth_noises`, `unclassified_voice` (voiced speech-level audio no word covers), `speech_mask`, `speech_probe` |
| run dir | `<video stem>.talkinghead/` next to the video, or `--out-dir` (`store.RunDir`) | one per recording: `run.json` records the source sha256 and `bind` refuses a different picture, so two cooks never share a directory | `plan.json`, `render.json`, `audio-master.m4a`, `audio-render.json`, `edit-lock.json` (and `edit-lock.hold` between a `--stop-at-lock` run and the run that renders its picture), `final-visuals.mp4` or `final.mp4`, `decisions.json`, `run.json`, `preferences.md`, `audio/`, `llm/`, `stages/`, `verify/` (with `locked-transcript.json`), plus `master.mov` and `align.json` when `--audio` was given |
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
| 2 | analyse | words, native sound events, protected laughter, silences, boundaries and acoustic noise candidates | deterministic | `audio/audio-16k.wav` (re-extracted on every run), `audio/source-words.json` + cache entries |
| 3 | structure | the true start (cold open) and the outro sentence | judge, supermajority | stage memo |
| 4 | retakes | which abandoned attempts and marked restarts are removed | judge, two passes | stage memo |
| 5 | content | which whole sentences do not earn their runtime (and a coherence veto) | judge + coherence pass | stage memo |
| 6 | pauses | pause trims, fillers, similarity-detected restarts, mouth noises | deterministic | stage memo |
| 6b | restructure (opt-in: `restructure = true`) | topic blocks: which have no payoff (dropped), which open the cut (moved first) | judge: one sample per window, then one vote | stage memo, `plan.json` layout |
| 6c | polish (on by default; `polish = false` opts out) | gap tightening between kept words (deterministic); word spans a judge names as disfluent; output pass: fillers, repeats and fragment restarts the render still has | judge per window, k-vote agreement (gaps need no judge); output pass judges one batch per round | stage memo, one per output round |
| 7 | edges | final span of every cut | deterministic | `plan.json` |
| 8 | render | the mastered audio alone (`audio-master.m4a`), then the picture once, with the overlays inside it and the master muxed unchanged | deterministic | `audio-master.m4a`, `audio-render.json`, `final-visuals.mp4` or `final.mp4`, `render.json`, or `final.wav` for an audio-only cook |
| 9 | verify | did the render keep the plan's promises: the audio table on the master, then the picture table on the picture | deterministic, declarative checks | `verify/facts.json`, `edit-lock.json`, `verify/locked-transcript.json` |
| 10 | visuals | what to overlay, where, from which asset; planned on the locked transcript, head-clip checked on source frames, drawn in stage 8's encode | judge | `visuals/` plan memo, `verify/visuals/` |
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
  Three backends: Scribe v2 (through OpenRouter), Parakeet and Whisper. The transcriber has no built-in default: it is named by `TALKINGHEAD_TRANSCRIBER`, `--transcriber` or the config file.
  Scribe splits a recording longer than one OpenRouter request into overlapping chunks cut at silences. `plan_scribe_chunks` uses the fewest chunks, and each word is kept from the chunk that owns its midpoint.
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
With voice isolation on, the silence the pauses stage and the dead-air check read comes from the isolated track (16 kHz, unrepaired, cached per source). The pauses memo keys on that silence; stages after it recompute only when their input cuts change.

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

A silence-only cut must remove more than `min_pause_cut` (0.6 s) to be made. A trim that small would read as a visible jump on picture,
so the pause stays whole; longer dead air is still trimmed to its boundary cap. The dead-air check (`verify.dead_air_excess`)
allows a pause up to its cap plus `min_pause_cut` plus `DEAD_AIR_SLACK`, and the render-silence check counts runs from
`pause_cap_topic + min_pause_cut + DEAD_AIR_SLACK` upward. `min_pause_cut` is in the pauses and polish memo keys.

### 6b. Restructure (judge, opt-in)

`stages/restructure.py`, enabled by `restructure = true`. The kept transcript (sentences no earlier cut
removes) is judged in windows of `restructure_window_sec` with `restructure_context_sec` of read-only context
each side, one sample per window. The windows' answers partition the sentences into topic blocks. One vote over
the block list (`restructure_votes` samples, strict majority) drops a block only when it has no payoff. Up to
`restructure_front` blocks that stand without earlier context move to the front, Borda-ranked and majority-backed.
Dropped blocks become `tangent` cuts; an opener's lead-in sentences become `lead-in` cuts. `restructure_brief`
is the owner's brief, sent verbatim to both passes.

Only after `refine_joins` does `restructure.reorder` split the plan at each block boundary and
lay the openers first. A boundary is a seam, so it is placed at apply time (the stored block starts and
ends are never changed): the nearest sentence gap within six sentences where the sentence before ends a
thought (`.`, `?`, `!`, not a dash or an ellipsis), the sentence after opens one (not a lowercase
continuation), and no speaker's unfinished sentence is split across a backchannel. A gap with a removed span
between its sentences is cut there, so no kept segment is split; otherwise the quietest frame is used, and only
where both pieces at that frame are at least 0.15 s. The plan then runs out of source order: the picture and
the audio join in list order, the join at a moved block's old place is a crossfade, and
`plan.provenance["blocks"]` records each block's output range, the source span it now owns (`placed`) and
how far its start moved (`seam_moved`). Board overlays refuse such a plan. The stage memo holds the blocks
with their decisions, so a hand-edited decision is honoured on replay.

### 6c. Polish (judge, on by default)

`stages/polish.py`, on unless the cook sets `polish = false`, runs after restructure on the kept material (the raw cuts plus
the block cuts). Two source-side parts, both returning cuts that the edge pass places with the rest:

- **Gap tightening** (no judge). The silence between two consecutive kept words is cut to a cap set by the boundary
  it sits on, most specific first: block boundary (`polish_gap_block`, 0.45 s), speaker change (`polish_gap_speaker`,
  0.20 s), sentence end (`polish_gap_sentence`, 0.30 s), inside a sentence (`polish_gap_within`, 0.15 s). The cut leaves
  half the cap on each side. A gap is tightened only when both words are kept and no word-removing cut lies in it, and
  protected events inside it are kept whole. A gap cut must remove more than `min_pause_cut` (0.6 s), so a gap is trimmed only
  when the trim would be longer than that; a protected event that splits a gap can leave both halves uncut. The gap cut has reason `gap`: it removes no words.
- **Disfluency judge**. Kept sentences are judged in windows of `polish_window_sec` with `polish_context_sec` of
  read-only context. The judge names word spans to remove (repeats, false starts, self-corrections, fillers, fragments,
  clutter exchanges, interrupting backchannels). Each window is sampled `polish_votes` times and a span is cut only when
  `polish_agree` samples agree on it (`spans.cluster_spans`, narrow). Spans longer than `polish_max_sec` are dropped.
  Cuts have reason `polish`, a judged word-removing reason that the `judged_removals` check verifies.

The stage memo holds every cut with its reason, words and votes, so a hand-edited memo is honoured on replay. With
`polish = false`, no polish key reaches the config dict or the plan provenance, so a cook that opts out keeps its keys.
Polish knobs stay out of every other stage's memo key (`pipeline._stage_config`): tuning polish re-keys only the polish
stages and the plan they produce, never the structure, retake, content, pause or visuals memos.
On a plan-only rebuild the polish judge is the one judge allowed to answer (`--plan-only`), since its memo is built there.

#### Output pass (`residual.py`, rounds in `stages/polish.py`)

The source transcript cannot show a stutter Scribe merged into one word ("I, I, I" comes back as one "I"), so after
each render the pipeline transcribes the render and scans that transcript (`residual.scan`) for three things. With voice isolation on, the transcript comes from an original-audio companion of the render, because the unit changes which words and stamps the transcriber returns; the isolated render is transcribed only for its word count.

- fillers: uh, um, erm, hmm, hm, mm, in any case or spelling;
- a 1-3 word phrase said twice in a row, after normalising case and punctuation ("I, I", "it's, it's");
- a fragment restarted: a token ending in a hyphen (or a 2-4 letter prefix) followed by a word that starts with it.

Each round's findings go to the judge in one batched prompt (`vote`, `polish_votes` samples, `polish_agree` to cut),
with about 10 s of transcript around each finding and the flagged words in double brackets. The rules are the same text
the source pass uses (`POLISH_KINDS`, `POLISH_NEVER`). A confirmed finding maps to one main-source span
(`residual.to_source`) and is cut with reason `polish` and `pass: output` in its provenance: the first copy and the gap
before the kept copy, ending `OUTPUT_KEPT_PAD` before the kept word's onset. A span may cross a join that runs forward
in the source (a gap cut sits there, and the cut runs through the removed silence). A span crossing a join that runs
backward (a moved block) is reported and not cut. Output-pass filler cuts settle their voiced edges like source fillers
(`edges.settle_filler_edges`); a filler the source transcript never named can still keep a tail, which the residual check reports.

A fragment restarted inside one token ("impl-implement") is placed on the source audio, not the render. `intraword.locate` aligns the token's onset within `ALIGN_SEC` (80 ms) of Scribe's stamp, since the stamp can start and end tens of milliseconds off the sound. It then takes the restart as the candidate whose onset matches best (similarity at least `MIN_SIMILARITY`, 0.90) among those whose cut point is a closure at least `MIN_DIP_DB` (8 dB) below the loudest frame before it. A token that does not qualify is skipped, and the report says why: `skipped: no restart in the audio` (the match or the closure failed), `skipped: too short for a restart`, `skipped: crosses a join`, or `skipped: too short` (the cut would be below `min_cut_sec`). A skipped finding stays in the residual list. A broken-off fragment the output pass confirms ("impl-" in the render) keeps its confirmed span in the source token: the rest of the token is kept speech, never removed with the fragment.

These thresholds were set on one recording: four of its five stutters pass with their numbers steady across stamp offsets, the fifth is not placed, and about one ordinary word in ten also passes the onset test. A placement therefore needs Scribe's X-Xy token and the judge's confirmation first. Nothing guarantees that every stutter is removed or that no word is clipped: read the residual list and the placed cuts in the report.

The rounds run up to `polish_output_rounds` (default 2, 0 disables). Each round re-renders the audio master, transcribes the
render again and stops early when a round confirms nothing. The round's memo is keyed on the cuts it builds on, so a
plan-only rebuild replays it without transcribing. The final check `residual_disfluencies` (warn, never fatal) lists
findings the last render still shows that the judge did not keep, with output times. The report's Polish section
lists every round, the skipped findings, the residuals and the judged-keep list.

A round runs one more time when the final render shows a repeat or restart that no round covered (its transcript changed after the cuts): one extra judged round at most, and only when `polish_output_rounds` is above 0. Plan-only replays that stored round.

The pass runs only when the render is transcribed. With `--no-output-asr` it is skipped, and the report says so.

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
Word-removing cuts (`content`, `retake`, `cold-open`, `outro`, `join`, `filler`, `tangent`, `lead-in`) only keep
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

`enforce_min_segments` runs last in `finalize_cuts`. A kept segment shorter than `min_segment_sec` (0.5 s) is repaired: a sliver
with no kept word is absorbed by growing an automatic neighbouring cut over it, and a sliver holding a kept word gets an
adjacent pause-only cut (reason `silence` or `gap`, no word inside) restored, so the word joins the kept material beyond it.
Hand cuts are never changed, and a word-removing cut is never reopened. A sliver with no legal repair stays and is listed in
`plan.provenance["min_segment"]["unfixed"]`; verify's `min_segment` check FAILs the plan. `refine_joins` does not move a join
so that a neighbouring segment shrinks below the minimum.

`trim_opening` runs after `restructure.reorder` on the plan's first segment: silence kept before the output's first word is cut back to `pad`, as leading silence always was, so a cold open or a reordered block that now leads starts on its word. The trimmed span is added to `plan.cuts`, and the trim is skipped where it would take voiced audio, laughter or a hand cut.

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
It preserves full native laughter/unknown-event intervals for every mechanical cut reason, so a laugh is never
trimmed by a silence, filler, noise or join cut. A judged cut (`JUDGED_REASONS`: cold-open, outro, tangent,
lead-in, retake, content) is different: its span is a decision, so an event inside it goes too, and only its
two edges are refined. Cut fusion and frame snapping cannot bridge or trim the protected intervals of mechanical cuts.

A judged decision is also held: `hold_judged` puts back any word a judged cut removes (by coverage: the cut covers at
least half the word's audio, the same test the `judged_removals` verify check uses) that an edge step had pulled out of the span. Fusing
cuts never loses a judged reason to a mechanical cut that comes first.


`timeline.Plan.from_cuts` then derives the kept segments from the merged cuts and
`plan.json` is written: cuts with reasons and provenance, the segment list, the
structural decisions, the anchors, and the provenance block (hashes, backend, floor,
effective config) that stage memos are keyed on.

### 7b. Audio assessment (advisory, memoised, alongside the editorial stages)

`audio_assess.measure` reads the recorded source audio as it is (before isolation): integrated loudness, true peak and
LRA from loudnorm's analysis of the original channels, and a 10 ms level envelope (floor, quiet-frame share, speech p90)
and the average spectrum of windows where every frame is quiet (band shares, 50/60 Hz tone ratios). The envelope, the
spectrum and `mono_sample_peak_dbfs` are computed on a mono downmix, which can sum channels above the original peak, so
only the original-channel true peak is clipping evidence. `audio_assess.assess` hands those numbers to the configured
judge once (no fallback), with the owner's preferences in the prompt and a guidelines fingerprint, and returns its
reading. It reads only the source, so `run_pipeline` runs it on a worker thread beside the editorial stages when
`stages.restructure.concurrent_safe(judge)` is true, and inline otherwise. The stage memo (`stages/audio-assessment.json`) keys on the source file's sha, the judge name, the two level
targets, a guidelines digest and a preferences digest, plus `ASSESS_VERSION`; no editorial memo re-keys. Plan-only
rebuilds replay a stored reading whose key matches and never call the judge. The reading is written to the report's
Audio section; no stage reads it, so it cannot enable isolation, denoise or compression.

### 8. Render (deterministic: the audio master, then the picture once)

The audio and the picture are separate renders, so the audio can be judged, listened to and locked before any picture
exists.

**Audio master.** `render.render_audio_plan` writes `audio-master.m4a` (AAC 192k, 48 kHz): segments snapped to the
picture's frame grid (`timemap.snapped_segments`, the same cut the picture uses), decoded and joined in numpy, then one fixed gain
measured on the joined audio by `render._loudness_gain_db`: the gain aims at `Config.loudness_lufs` and never takes the
true peak past -1.5 dBTP, so a peak-limited source lands quieter. The default master has no compressor, EQ, highpass or
dynamic normalisation (`loudnorm` only measures). Verification fails a master that is louder
than target and reports a quieter one as a warning.
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
known to hang. The audio and the picture are separate runs because ffmpeg 9.0.1 emits digitally silent audio from a
combined `concat=v=1:a=1` graph for some source regions.
`audio-render.json` is the master's memo, `{key, cmd, sha256}`: `pipeline.audio_memo_key` covers the snapped segments,
the source and isolation-track hashes, the audio-relevant settings and `render.RENDERER_VERSION`. The master replays
only when the file exists, its sha256 equals the stored one and the key matches, so an overwritten master is rendered
again, never trusted. Picture-only settings (`crf`, `preset`, `scale`) do not enter it.

**The lock.** After the audio checks pass, `lock.write_lock` writes `edit-lock.json`: `lock.plan_digest` (the segments,
the sources' sha256 and the isolation track's), the frame rate, and the master's sha256 and audio-packet MD5.
`lock.check_lock` refuses a plan, fps or master that moved. A cook keeps a matching lock byte for byte and re-locks a
changed plan on its own, unless the lock is held: `--stop-at-lock` writes `edit-lock.hold` beside it, and while the hold
exists only a `--stop-at-lock` run may re-lock, so the plan the owner listened to cannot change under them. The full run
that renders the held lock's picture deletes the hold. Hand workflows make the same file with `talkinghead lock`.

**Picture.** `render.render_plan(..., audio_file=<locked master>, overlays=...)` runs one picture pass (one CFR decode
per source run, `fps` then chained half-open `select` filters that drop the removed spans, frame-snapped, no audio) and
then a `-c copy` mux of the master's first audio stream, packets unchanged. The picture pass avoids per-segment trim
fan-out, which stalled on a native 4K HLG recording.
The branch queue was a suspected cause; the observed failure was no output growth for 20 minutes.
With overlays, `visuals.render.overlay_graph` extends the same graph, so the overlays are part of this one encode and
the deliverable is `final-visuals.mp4`; with none it is `final.mp4`. A cook renders one of the two; a `--no-visuals` rerun on the same lock adds `final.mp4` beside `final-visuals.mp4` for the board, and a picture from an earlier lock is removed.
`docs/render-proof.md` records why closed `between` intervals are unsafe: they remove one extra frame per grid-aligned cut.

Colour is probed from the sources (`color.probe_color`): an HLG/PQ source is re-encoded
as 10-bit with its own primaries/matrix/transfer and `-map_metadata -1`, so a Dolby
Vision record on the source cannot ride into an edited picture; an SDR source keeps the
libx264/yuv420p encode. No tone-map LUT is ever applied implicitly.

The picture's `render.json` memo is keyed on the segment list, the picture settings, the locked master's sha256, the
overlays' identity (each asset's sha256) and the deliverable's name, plus `render.RENDERER_VERSION` (the version of the
render chain itself, which the plan and settings cannot express): changing an unrelated config value must not
re-encode a finished render, but changing the chain must.

**Companions.** Animations made outside the pipeline anchor on `audio/source-words.json` (the recording's words, in
source seconds). `talkinghead timings` maps an anchor to output time with `timemap.map_span` (`kept`, `trimmed`,
`split` or `cut`, from the same snapped cut), and `talkinghead render --lock L --window S:E` renders the picture of
`timemap.window_segments(plan, S, E)`, the speaker footage under the animation. Both refuse a plan the lock does not
match.

### 9. Verify (deterministic, two tables)

The audio table runs on `audio-master.m4a`, before any picture. `verify.collect_facts` measures it: audio stream
(AAC, 48 kHz), duration, LUFS, a fresh transcript of the master (`prepare_audio(force=True)`, so a rewritten master is
never judged on a stale WAV), join grades (`joins.check_join`), a two-sided source edge check (`join_edges`:
a chopped voiced word on either side of a cut, judged on the source), residual dead air
(`verify.residual_dead_air`: the source VAD silences that survive the plan; the gap budget
counts only the excess over the pause caps, so pauses the policy keeps on purpose never
count), pacing facts (`verify.pacing_facts`: kept pause lengths against the source's), and source-to-output audio timing (`verify.measure_source_timing`).
It writes the measurements to `verify/facts.json`.
The report's `## Pacing` section prints the pacing facts (kept pause lengths by boundary type, and the quiet each cut reason leaves at its seam) as facts for the editor's ear, never a verdict.

`verify.run_checks` evaluates the declarative audio checks (`CHECKS` in `verify.py`). Fatal:
`audio_stream`, `duration`, `loudness`, `cold_open` (first word within 0.5 s), `residual_gaps`,
`render_silence` (measured on the master; the mute threshold is below one 16-bit PCM quantization step) and `source_timing` (a
sampled keep's audio more than 60 ms from where the plan put it). Warnings: `joins`, `join_edges`,
`source_similarity` (a sample that no longer correlates with the source, which the
broadcast chain can legitimately cause) and `unclassified_voice` (voiced audio no word
covers that the plan removes, with the re-transcription's text when one ran).

The picture table (`verify.run_picture_checks`, `PICTURE_CHECKS`) runs on the finished picture and measures only what
the picture can change, from `verify.collect_picture_facts`: `av_sync`, `color` (a source's HDR colour silently
converted, including right tags on an 8-bit stream), `resolution` (the render's displayed size against the source
through the configured scale) and `audio_locked`. `audio_locked` is fatal unless the MD5 of the picture's audio
packets (`lock.audio_stream_md5`, `ffmpeg -map 0:a:0 -c copy -f md5 -`) equals the locked master's; packets survive the
`-c copy` mux, so a changed byte anywhere in the audio fails it. These facts land under `"picture"` in
`verify/facts.json`.

A measurement that could not be taken is never reported as a pass: `source_timing` says
`not measured` (and warns) when no keep is long enough, no window carries alignable audio,
or nothing decodes, and distinguishes that from a run that never had a source path to
measure. Any failing fatal check writes `report.md` with status FAILED and raises
`PipelineError`, which the CLI turns into exit 1. A fatal audio check stops the cook before the lock; neither the
visuals planning nor the picture runs after it.

### 10. Visuals (judge, planned before the picture)

`visuals/stage.plan_visuals` chooses anchors, cards and the chapter plan card on the locked transcript
(`verify/locked-transcript.json`, the words the approved master says), memoised under `VISUALS_STAGE_VERSION` because
it is judged and therefore must not be re-decided; screenshot research (`--no-research`) and chapter selection are
single-sample judge calls. It runs after the lock and before any picture exists.

`visuals/headclip.check_head_clip_plan` then checks the overlay rectangle against the speaker on SOURCE frames: each
sample's output time is located through `timemap.locate` (the snapped cut, the same one the picture renders) and the
frame is read from the recording. A fatal `visuals_head_clip` stops the cook before any picture, because the fix is a
corner, scale or margin change in the cook's config, which re-renders the picture anyway. A `--stop-at-lock` run does
both steps before it stops, so the owner sees the overlay plan and the head-clip result next to the master.

The overlays are drawn by the picture encode itself (stage 8): `visuals/render.overlay_inputs` adds one looped input per
still and `overlay_graph` composites them with `enable=between(t,..)` windows over the cut's own `[vout]`. The base's
colour is probed: an SDR cut keeps the libx264/yuv420p encode, an HLG/PQ cut keeps its 10 bits and its tags, and the
SDR overlay assets are converted into that system with the cube from `color.write_sdr_to_hdr_cube` (SDR white to
BT.2408 graphics white) rather than tagged as if they were already HDR. There is no second encode and no stream-copy
pass over a finished file.
`visuals/verify.verify_visuals` writes a contact sheet of frames at overlay midpoints and in clean control gaps under
`verify/visuals/` (`visuals_frames`); the audio and duration promises it used to check are now `audio_locked` and the
picture table. `--no-visuals` skips the stage, and the deliverable is then `final.mp4`.

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
