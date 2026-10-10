# Mouth-noise detector: calibration data

## Goal
Detect and cut tongue smacks, breaths, and throat clears in talking-head audio,
as a repeatable pass in the pipeline. Editors used to cut these by hand
today ("they look like jumps in the audiograph"). This module automates that.

## Ground-truth events (hand-labelled, located acoustically in one calibration render)
Render timestamps where a listener heard noise; acoustic signature measured from
16kHz mono, 5ms frames, dB = 20*log10(RMS), ZCR = mean sign-flip rate:

| render t | kind | signature |
|---|---|---|
| 15.02 | smack | <=0.2s burst, ZCR > 0.4, sits in 300ms gap between words |
| 51.09 | smack | same, but embedded between closely-spaced words (skip) |
| 52.28 | smack | same, core in 240ms gap, edges brush word tails |
| 62.0-62.3 | smack x2 | short ZCR 0.6-0.7 clicks |
| 83.0 | smack | 0.01-0.02s clicks, ZCR ~1.0 |
| 88.0 | smack | 0.12s ZCR 0.55 + 0.01s ZCR 0.97 |
| 122.25 | smack | in gap 122.20-122.50 between 'previous,' and 'I' |
| 139.39 | smack | in gap 139.36-139.66 between 'latest,' and 'greatest' |
| 147.0 | breath | 0.3-0.5s, ZCR < 0.2, 10-16dB over floor, between sentences |
| 183.3 | breath | same, between 'enough.' and "Let's" |
| 207.2 | breath | same pattern |
| 222.0 | breath | same pattern |
| 251.0 | throat clear | 0.5s burst at -13.7dB in a real quiet gap ASR swallowed into a word span |

## Classifier (validated against the ground truth)
Working on 5ms frames; local noise floor = 25th percentile of the whole file
(adaptive per-file floor worked; 1.5s sliding floor over-segmented):
  burst = contiguous frames above floor+8dB, ending when below floor+2
  SMACK: duration <= 0.2s AND mean ZCR > 0.4
  BREATH: 0.15s <= duration <= 0.6s AND mean ZCR < 0.22 AND peak < 16dB over floor
  THROAT: like breath but stronger/longer; ASR often swallows it, so do not
          trust word spans alone - check the audio envelope for a quiet gap
          around the burst.

## Safety rules (non-negotiable)
1. Never cut a span that overlaps a word's voiced core. ASR word spans are
   generous: check the acoustic envelope, not just the word table. A noise cut
   must start after the previous word's energy has actually dropped below
   floor+4dB and end before the next word's energy rises above floor+8dB.
2. If a burst is embedded in continuous speech with <150ms of true quiet on
   either side, LEAVE IT (a clipped word is worse than a small smack).
3. Merge noise cuts into adjacent existing cuts when within pad distance.
4. Reason label: "noise" with kind in the text ("mouth noise: smack/breath/throat").
5. All noise cuts still pass the review gate like every other cut.

## Validation harness
Run the detector on a cut render; report remaining word-gap noises. Success =
zero SMACK/BREATH events in word gaps of the rendered output (measured on the calibration render:
13 clean gap events + 5 embedded; target: all gap events removed, embedded
ones flagged for human review in the report).

## Calibration findings

1. Silero VAD gate MISLABELS smacks/breaths as speech (they are voiced): with
   the VAD gate at prob<0.5, only 2 of ~16 hand-confirmed noises passed. The
   envelope+word-core path found all of them. Default: run envelope-gated;
   keep the VAD as an optional pre-filter for future tuning.
2. Bursts embedded in continuous speech (no 150ms quiet either side) must be
   DUCKED, not cut: cutting clips word cores (verified acoustically). Ducking
   via ffmpeg `volume='if(between(t,a,b),0.30,1)':eval=frame` on the rendered
   output dropped embedded smacks/breaths ~12dB with no speech loss (659 words
   intact before/after). Open item: fold ducking into render as a first-class plan
   field (e.g. plan.ducks = [{start, end, gain}]) instead of post-processing.

## Implementation
`src/talkinghead/noise.py` (`detect_mouth_noises`), called once from
`analysis.analyze` on the shared envelope; its cuts enter the plan through the pauses
stage (`src/talkinghead/stages/pauses.py`), which has no per-run switch for it. The
classifier above labels candidates, but the candidate GATE is Silero VAD (`silero-vad` + `onnxruntime`, optional extra `vad`): a
burst must sit inside a run of frames the model scores below 0.5 speech
probability. Without silero-vad installed the pass falls back to envelope-only
gaps and says so on stderr. Cut edges always come from the 5 ms envelope, never
from the 32 ms VAD grid or from word spans.

Note: this is the opposite of what finding 1 above recommends. The shipped pass keeps the
VAD as the candidate gate (with the envelope-only fallback) and relies on the labelling
thresholds to reject speech; finding 1 measured the gate alone on this ground truth and
found it weak, which is why the fallback exists.
