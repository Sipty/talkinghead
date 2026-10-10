# Render proof (Milestone 0): which ffmpeg cut-render strategy is safe at 31 minutes?

Measured, not remembered. Script: `scripts/render_proof.py`. Outputs (`*.render.json`, `*.measure.json`, full ffmpeg logs) go to `--out-dir` (about 15 GB, never in the repo).

- ffmpeg: `ffmpeg version 9.0.1 Copyright (c) 2000-2026 the FFmpeg developers` (Homebrew, Apple M5, 10 cores, 32 GB)
- Source: `<SOURCE_VIDEO>` (passed as `--source`), measured:
  1920x1080 h264, `r_frame_rate` = `avg_frame_rate` = 30/1, aac 48 kHz stereo.
  Container 1894.603 s; video track 56678 frames = 1889.267 s; audio track 1894.603 s.
  The video PTS list was checked frame by frame: 56678 frames, every delta exactly 1/30 s,
  no gaps. **This file is true CFR**, so the VFR phone-footage pitfall does not apply to it;
  nothing below tests VFR handling.
- Proof encode for every strategy: `-c:v libx264 -preset ultrafast -crf 28 -pix_fmt yuv420p -c:a aac -b:a 128k -ar 48000`.
- Two renders ran concurrently (one ffmpeg process each; strategy C's segments were encoded
  sequentially inside its slot). Wall times are therefore comparable between strategies but
  about 1.5x slower than a lone render would be. `P2j_A` additionally shared the machine with
  ad-hoc probing and is inflated (see its row).

## Strategies

Cuts are source-time spans to remove; keeps are their complement in `[0, 1894.603]`.

- **A** `filter_complex`: for each keep `[0:v]trim=start:end,setpts=PTS-STARTPTS`,
  `[0:a]atrim,asetpts=PTS-STARTPTS,afade in 40 ms,afade out 40 ms`, then `concat=n=N:v=1:a=1`.
  Mirrors `src/talkinghead/render.py::_video_graph`/`_audio_graph` without the audio chain.
- **B** single pass `-vf "fps=30,select='not(between(t,a1,b1)+...)',setpts=N/FRAME_RATE/TB"
  -af "aselect='<same>',asetpts=N/SR/TB"`.
- **B2** (added during the proof) identical to B but the expression is split into chained
  `select`/`aselect` filters of 40 `between()` terms each, because B cannot be parsed above 99
  terms (see P2). `select` does not rewrite timestamps, so chained filters all see the original `t`.
- **C** encode each keep to its own mp4 with `-ss S -to E` before `-i` (accurate seek, re-encode),
  write a concat list, join with `ffmpeg -f concat -safe 0 -i list.txt -c copy`.

## Plans

- **P0** no cuts (plain re-encode).
- **P1** 15 cuts of 1.0 s at t = 120, 240, ..., 1800 s (removes 15.0 s; 16 keeps).
- **P2** 208 cuts of 0.6 s at t = 9, 18, ..., 1872 s (removes 124.8 s; 209 keeps). All boundaries
  fall exactly on frame times (9k and 9k+0.6 are multiples of 1/30 s).
- **P2j** (added) P2 with every boundary shifted by +13.7 ms so no boundary lies on the frame grid,
  the realistic case for ASR-derived cut times. Added after P2 showed that B's and C's behaviour
  depends on boundary phase.

## Measurement method (and its limits)

1. **Duration**: ffprobe container duration, video-stream duration/`nb_frames`/`avg_frame_rate`,
   audio-stream duration, each minus `expected = 1894.603 - removed`. Because the source video
   track ends 5.336 s before the audio, the video column always reads about -5336 ms against that
   expectation; the value in parentheses is the video duration against
   `video_track_duration - removed_within_video_track`, which is the achievable figure.
2. **Dropouts**: whole output audio decoded to 16 kHz mono (timestamp-conformed with
   `aresample=async=1:min_hard_comp=0.001`), RMS in 50 ms frames, runs >= 1.5 s below -60 dBFS
   are candidates; each is mapped to source time (through the keeps, corrected by the measured
   audio offset) and is a dropout only if the source has no pause >= 80 % as long within +-1 s
   (source threshold -55 dBFS, a 5 dB margin because the AAC re-encode pushes near-silence lower).
   Every render produced 14-16 candidates and **all were matched to real pauses in the source**.
   An earlier, stricter version of this check (no drift correction, no margin) flagged 12
   "dropouts" in `P2_C`, 1 in `P1_C`/`P2_B2`; each was traced by hand and was either a real pause
   mis-mapped because the output had drifted, or a pause hovering at -59..-72 dBFS that the
   re-encode pushed under -60 dBFS while a single -59 dBFS frame in strategy A's output broke the
   run. No audio content was missing in any of them. The detector cannot see dropouts shorter than
   1.5 s or partial attenuation.
   A second audio metric, **sequential excess**, is the raw sequential decode length minus the
   timestamp-conformed length: > 0 means the file carries more samples than its timestamps account for.
3. **A/V sync, video track**: six probes at 5/20/40/60/80/97 % of the output, each a 3 s window
   that never crosses a splice. The spec asked for 3 s clearance from cuts; P2/P2j keeps are only
   8.4 s long, so there the window is centred in its keep with 2.7 s clearance (reported per table).
   Output frames (160x90 gray, `-fps_mode passthrough`; ffmpeg 9 has removed `-vsync`) are matched
   against a 19 s source window (+-8 s search range, needed because B2 drifts by seconds) by
   full-frame mean absolute difference; the integer-frame match is then refined to sub-millisecond
   with the real frame PTS from ffprobe. The spec's method, cross-correlating per-frame *mean luma*
   sequences, is also computed and stored, but it disagreed with the full-frame match on 1-2 of 6
   probes in every render (e.g. -232 frames at r = 0.89 at t = 1717 s in `P2_A`, where full-frame
   MAD gave 0 frames with a peak ratio of 0.09): on static talking-head footage the mean-luma
   series carries too little information. The full-frame MAD peak ratio (best/median, lower is
   sharper) was <= 0.13 on every probe except `P2_B2` (max 0.30) and `P2j_C` (max 0.23).
   **Audio** offset: FFT cross-correlation of the same 3 s output window against the same source
   window, energy-normalised. Peak ratio (peak / next best outside +-10 ms) was >= 1.8 everywhere
   except `P2j_C` probe 4 (1.4), which is flagged below.
   Sign convention everywhere: **positive = output content appears later than the plan says**.
   **Lipsync error = video offset - audio offset** (negative = video leads / lips ahead of sound).
4. `avg_frame_rate` of the output as reported by ffprobe.

## Results

Duration deltas are output minus expected, in ms. Per-probe columns list the six probes in order.

### P0 (no cuts)

Probe output times (s): 93, 377, 756, 1135, 1514, 1836

| strategy | wall | duration delta ms: container / video (vs video track) / audio | dropouts | audio seq. excess ms | video offset ms per probe | audio offset ms per probe | lipsync error ms per probe | avg_frame_rate |
|---|---|---|---|---|---|---|---|---|
| A | 2.7 min | +0.0 / -5336 (+0.0) / +0.0 | 0 | +0.0 | +0.0 / +0.0 / +0.0 / +0.0 / +0.0 / +0.0 | -0.0 / -0.0 / -0.0 / -0.0 / -0.0 / -0.0 | +0.0 / +0.0 / +0.0 / +0.0 / +0.0 / +0.0 | 30/1 |
| B | 2.7 min | +0.0 / -5336 (+0.0) / +0.0 | 0 | +0.0 | +0.0 / +0.0 / +0.0 / +0.0 / +0.0 / +0.0 | -0.0 / -0.0 / -0.0 / -0.0 / -0.0 / -0.0 | +0.0 / +0.0 / +0.0 / +0.0 / +0.0 / +0.0 | 30/1 |
| C | 2.9 min | +21 / -5336 (+0.0) / +21 | 0 | +0.0 | +21 / +21 / +21 / +21 / +21 / +21 | +21 / +21 / +21 / +21 / +21 / +21 | +0.0 / +0.0 / +0.0 / +0.0 / +0.0 / +0.0 | 30/1 |

Encoder and timestamps are clean: A and B reproduce the source to the sample (56678 frames,
1889.267 s video, 1894.603 s audio). C's `-c copy` join drops the first segment's AAC edit list
(1024 priming samples = 21.333 ms), so both tracks start 21.35 ms late (video `start_time`
0.021354); lipsync is unaffected.

### P1 (15 cuts, boundaries on the frame grid)

Probe output times (s) [splice clearance s]: 92 [24.5], 374 [16.4], 750 [35.3], 1126 [54.3], 1502 [42.8], 1822 [35.7]

| strategy | wall | duration delta ms: container / video (vs video track) / audio | dropouts | audio seq. excess ms | video offset ms per probe | audio offset ms per probe | lipsync error ms per probe | avg_frame_rate |
|---|---|---|---|---|---|---|---|---|
| A | 2.8 min | +0.0 / -5336 (+0.0) / +0.0 | 0 | +0.0 | +0.0 / +0.0 / +0.0 / +0.0 / +0.0 / +0.0 | -0.0 / -0.0 / -0.0 / -0.0 / -0.0 / -0.0 | +0.0 / +0.0 / +0.0 / +0.0 / +0.0 / +0.0 | 30/1 |
| B | 2.5 min | -40 / -5836 (-500) / -40 | 0 | +0.0 | +0.0 / -100 / -200 / -300 / -400 / -500 | -0.0 / -8.0 / -16 / -24 / -32 / -40 | +0.0 / -92 / -184 / -276 / -368 / -460 | 30/1 |
| C | 2.6 min | +21 / -5336 (+0.0) / +21 | 0 | +581 | +21 / +21 / +21 / +21 / +21 / +21 | +21 / +21 / +21 / +21 / +21 / +21 | +0.0 / +0.0 / +0.0 / +0.0 / +0.0 / +0.0 | 30/1 |

B loses exactly one extra video frame per cut: `between(t,a,b)` is closed on both ends, and with
a and b on the frame grid it removes 31 frames for a 1.0 s cut instead of 30. Audio frames
(256 samples at this point of the graph, 5.33 ms) are quantised the same way but the error is
smaller (-2.67 ms per cut). Lips run ahead of sound by 33 ms per cut: -460 ms after 15 cuts.
C's video is exact, but the sequential decode of its audio is 581 ms longer than its timestamps:
each segment's AAC priming (1024 samples) and end padding (up to one 1024-sample frame; here
896 samples because 119 s is not a whole number of AAC frames) survive the stream-copy join with
overlapping timestamps (38.8 ms of stray samples per splice).

### P2 (208 cuts, boundaries on the frame grid)

Probe output times (s) [splice clearance s]: 87 [2.7], 356 [2.7], 709 [2.7], 1062 [2.7], 1414 [2.7], 1717 [2.7]

| strategy | wall | duration delta ms: container / video (vs video track) / audio | dropouts | audio seq. excess ms | video offset ms per probe | audio offset ms per probe | lipsync error ms per probe | avg_frame_rate |
|---|---|---|---|---|---|---|---|---|
| A | 10.3 min | -0.0 / -5336 (-0.0) / -0.0 | 0 | +0.0 | +0.0 / +0.0 / +0.0 / +0.0 / -0.0 / -0.0 | -0.0 / -0.0 / -0.0 / -0.0 / -0.0 / -0.0 | +0.0 / +0.0 / +0.0 / +0.0 / -0.0 / -0.0 | 30/1 |
| B | FAILED: `Error while parsing expression ... Cannot allocate memory` | - | - | - | - | - | - | - |
| B2 | 2.5 min | -555 / -12269 (-6933) / -555 | 0 | +0.0 | -333 / -1400 / -2800 / -4833 / -6233 / -7433 | -37 / -123 / -224 / -325 / -448 / -544 | -296 / -1277 / -2576 / -4508 / -5785 / -6889 | 30/1 |
| C | 2.0 min | +21 / -5336 (-0.0) / +21 | 0 | +5544 | +21 / +21 / +21 / +21 / +21 / +21 | +21 / +21 / +21 / +21 / +21 / +21 | +0.0 / +0.0 / +0.0 / +0.0 / +0.0 / +0.0 | 30/1 |

- **A**: 209 trim/atrim branches, 52934 frames (= 56678 - 208 x 18 exactly), audio 1769.803 s
  (exact), zero offset at every probe, zero dropouts. A reported "filter_complex concat with
  20+ segments silences audio mid-file" **did not reproduce** with ffmpeg 9.0.1 and this graph
  (which has no audio chain / loudnorm after the concat). Splice timestamps are seamless
  (video 8.9667 -> 9.0000, audio contiguous). Cost: the filter graph fans every decoded frame out
  to all 209 branches; wall time 617 s vs 160 s for one keep.
- **B**: ffmpeg's expression parser accepts at most **99** `between()` terms; 100 fails with
  `Invalid argument`, more with `Cannot allocate memory` (tested 50 ok, 99 ok, 100/104/120/208
  fail). A single-expression select is unusable for a pause-cut plan.
- **B2**: renders, but loses one extra frame per cut as in P1: 52726 frames instead of 52934,
  video 6.93 s short, audio 0.56 s short. Video leads audio by 33 ms per cut, reaching **-6.9 s**
  at t = 1717 s. The probe values are exact frame counts in the first three probes (10, 42, 84
  cuts passed) but 19 frames larger than "one per cut" in probes 4-6; a 1 s-step scan showed why:
  once the accumulated drift D exceeds the time between probe and next cut, the output has already
  jumped past the cut while the plan has not, so for D seconds the measured offset is D plus the
  cut length (18 frames + 1). Both readings are genuine perceived errors, the second being what a
  viewer sees at every cut once drift exceeds a few seconds.
- **C**: video exact, lipsync 0.0 ms at every probe, but 5.54 s of stray audio samples
  (26.7 ms per splice: 1024 priming + 256 padding samples) with squeezed/overlapping timestamps
  (at the first splice: packets of duration 21 us and 18.6 ms carrying 1024 samples each). A
  timestamp-following decoder (ffmpeg with `-ss`, browsers, QuickTime) drops them and hears a
  constant +21 ms; a sample-sequential decoder (`ffmpeg -i out -f wav` and anything built on that
  path without async resampling) plays them and drifts 26.7 ms later per splice, 5.5 s by the end.
  ffmpeg prints `non monotonically increasing dts` for every splice when it reads the file.

### P2j (208 cuts, boundaries off the frame grid)

Probe output times (s) [splice clearance s]: 87 [2.7], 356 [2.7], 709 [2.7], 1062 [2.7], 1415 [2.7], 1717 [2.7]

| strategy | wall | duration delta ms: container / video (vs video track) / audio | dropouts | audio seq. excess ms | video offset ms per probe | audio offset ms per probe | lipsync error ms per probe | avg_frame_rate |
|---|---|---|---|---|---|---|---|---|
| A | 14.4 min (inflated, see top) | +20 / -5336 (-0.0) / +20 | 0 | +0.0 | +0.0 / +0.0 / +0.0 / +0.0 / -0.0 / -0.0 | +20 / +20 / +20 / +20 / +20 / +20 | -20 / -20 / -20 / -20 / -20 / -20 | 30/1 |
| B2 | 2.7 min | -0.0 / -5336 (-0.0) / -0.0 | 0 | +0.0 | +0.0 / +0.0 / +0.0 / +0.0 / -0.0 / -0.0 | +5.3 / +5.3 / +11 / -5.3 / -0.0 / +11 | -5.3 / -5.3 / -11 / +5.3 / -0.0 / -11 | 30/1 |
| C | 2.6 min | +6941 / +1597 (+6933) / +6941 | 0 | -1368 | +355 / +1421 / +2788 / +4788 / +6221 / +7421 | +341 / +1408 / +2801 / +4201 / +6208 / +7408 | +14 / +14 / -13 / +587 (low confidence) / +14 / +14 | 42792960/1432037 (29.88) |

- **A**: video exact; audio and container 20 ms long, and the audio offset is +20 ms from the
  first probe on and never grows. The concat filter pads the shorter stream of each segment to the
  longer one; the first keep `[0, 9.0137]` has 271 video frames (9.0333 s) against 9.0137 s of
  audio, so 20 ms of silence is inserted once. Every later keep is exactly 8.4 s = 252 frames on
  both tracks, so nothing accumulates. In general each cut whose boundaries are off the grid
  contributes a sub-frame pad (< 33 ms) that keeps the two tracks synchronous at the next segment
  start; lipsync error stays bounded by one frame and does not accumulate.
- **B2**: with boundaries off the grid the closed interval no longer double-counts: exactly 18
  frames per cut, audio quantisation (+-5.3 ms per cut) averages out (total 0.0 ms), lipsync
  within +-11 ms. So B's drift is entirely a boundary-phase effect: catastrophic when the planner
  snaps cuts to frames, benign when it does not. Untested mitigation: half-open expressions
  (`gte(t,a)*lt(t,b)` instead of `between`).
- **C**: `-ss 9.6137 -to 18.0137` yields 8.4333 s of video (frames straddle both boundaries, first
  frame at +33 ms) against 8.400 s of audio; the concat demuxer advances by the longer stream, so
  every splice inserts a ~33 ms hole in both tracks: +6.94 s total, a 66.7 ms video packet at each
  splice, `avg_frame_rate` 29.88 (the output is no longer CFR), and the picture reaches t = 1717 s
  7.4 s later than the plan. Lipsync itself stays at +-14 ms except probe 4 (+587 ms), where the
  audio cross-correlation peak ratio was 1.4 (weakest of the run; the video match was clear at
  0.23) and 587 ms is one cut length off, so that value is most likely a mis-pick of the
  correlation rather than a real glitch; it was not re-verified by ear.

### The video-track tail (video 5.336 s shorter than audio in the source)

All strategies preserve it unchanged: every output's video track ends 5.336 s before its audio
(e.g. P0: video 1889.267 s, audio 1894.603 s, 56678 frames in all three). Nobody pads, nobody
truncates; players hold the last frame under the remaining audio. A cut inside the tail
(t > 1889.27) would remove audio only; none of the plans has one.

## Superseded: the picture pass now uses half-open B2

The per-segment picture graph stalled on a 3840x2160 10-bit HLG source: the picture file did not grow for 20 minutes.
ffmpeg had several GB resident and system-wide swap usage exceeded 15 GB; branch buffering was a suspected cause, not an isolated measurement.
`render.py` now renders the picture as B2 with the mitigation noted under P2j: `fps` first, then chained `select` filters of at most 40 half-open `gte(t,a)*lt(t,b)` terms.
Edges are moved half a frame earlier so grid-aligned cuts remove exactly their frames.
The current audio path joins PCM in numpy and encodes once; it does not use `aselect`.
The recommendation below remains as the record of the earlier 1080p measurement.
`tests/v2/test_render.py::test_many_frame_grid_cuts_keep_the_exact_frames` checks the frame
sequence across more cuts than one expression holds.
Each run retains bounded clone-tail padding and stops decoding at its finite endpoint.
The non-CFR (`fps=0`) path retains per-segment trim behavior.

Real-source smoke proof: a cook's `verify/` directory (`<VIDEO_DIR>/<cook>/verify/`).
The 3840x2160 HLG export has 28,897 frames at 30 fps, matching the snapped plan.
Six picture probes at output 40, 200, 400, 600, 800 and 940 seconds matched their expected source frame (zero-frame offset; grayscale MAE 0.12-0.153).
Audio correlation over 24 kept windows was within 1 ms of the plan.
The audio track has a 67 ms terminal overhang, not accumulated content drift.
This full-source render preceded the boundary follow-up; short-tail padding, reordered runs and fps=0 were exercised separately by the final render tests.
The final suite passed 640 tests with two skips and one expected failure.


## Recommendation

**Adopt strategy A (filter_complex trim/atrim + concat), i.e. the graph already in
`src/talkinghead/render.py`, for the full-length render.** Numbers: exact frame count and audio
length in all four plans (largest duration error 20 ms, from one sub-frame pad), 0.0 ms video
offset at all 24 probes, lipsync 0.0 ms (grid-aligned cuts) or a constant, non-accumulating
-20 ms (off-grid cuts), 0 dropouts at 209 segments, seamless splice timestamps, CFR 30/1 output.

Do not adopt B: it cannot be expressed above 99 cuts, and even chained (B2) it drifts 33 ms per
cut whenever a boundary lands on the frame grid (-6.9 s at the end of P2). Do not adopt C as
specified: the stream-copy join leaves 27-39 ms of overlapping priming/padding samples per
splice (5.5 s in P2) that some decoders play, and with off-grid boundaries it inserts a 33 ms
hole per splice (+6.9 s, VFR output). Both C problems are muxing problems, not encoding
problems; a C variant that joins audio through a filter graph (or snaps segment boundaries to
the frame grid and re-muxes audio without priming) might recover its 5x speed advantage, but
that is a new design and was not measured.

Residual risks with A, with the numbers behind them:

1. **Render time grows with segment count**: 160-169 s for 1-16 keeps, 617 s for 209 keeps at
   `-preset ultrafast` (10.3 min for a 29.5 min output). The extra cost is filter-graph fan-out,
   not encoding, so at `-preset medium` it is a smaller fraction, but a 500-cut plan should be
   expected to take 20-30 min at ultrafast on this machine. Mitigations to test if that matters:
   render in chunks of <= 100 keeps and join the chunks with a second concat *filter* pass (not
   the concat demuxer), or `-filter_complex_threads`.
2. **Sub-frame cut boundaries cost up to one frame of lipsync per segment** (20 ms measured in
   P2j, constant). It does not accumulate, but it is avoidable: snap cut times to the frame grid
   before rendering; with grid-aligned boundaries A was exact to the sample (P1, P2).
3. **The earlier mid-file silence report was not reproduced here.** This proof ran the graph without
   the audio chain (`highpass, acompressor, equalizer, loudnorm`) that `render.py` appends after
   the concat. Before trusting the final-render path, re-run `measure` on one 200-cut render made
   through `render.py` with the chain on; the measurement half of the script works on any file
   plus its keep list. If the silence returns, the suspect is the chain, not the concat.
4. **Wall-time figures are from 2 concurrent renders**; single-render times will be lower and the
   ratios between strategies roughly preserved.
5. **The source is CFR.** Nothing here says what A does with VFR phone footage; if that input
   returns, add `fps=30` in front of the trims and re-measure.

## Reproduce

```
uv run python scripts/render_proof.py all --source <SOURCE_VIDEO> --out-dir <OUT_DIR> --plans P0 P1 P2 --strategies A B C
uv run python scripts/render_proof.py all --source <SOURCE_VIDEO> --out-dir <OUT_DIR> --plans P2 --strategies B2
uv run python scripts/render_proof.py all --source <SOURCE_VIDEO> --out-dir <OUT_DIR> --plans P2j --strategies A B2 C
uv run python scripts/render_proof.py report --out-dir <OUT_DIR>
```

Total render time about 55 min; measurement about 1.5 min per render. Smoke test on the first
60 s: `... all --source <SOURCE_VIDEO> --out-dir <scratch> --plans P2 --strategies A B C --head 60`.
