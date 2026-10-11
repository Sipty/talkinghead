# talkinghead

talkinghead turns one talking-head recording into a finished cut and a report that says why each cut was made.
It removes material: warmup, retakes, digressions, long pauses and filler words.
It does not write, re-record or reorder anything.
An LLM judge that you choose makes the editorial calls, and every call is recorded with its vote count.
The cutting, mastering and checks all run on your own machine.

Runs on macOS, Linux and Windows (WSL2).
Point your AI coding agent at this README and it can set everything up.

## What you get

Every run writes a run directory next to the source video, named `<video stem>.talkinghead/`.

| file | what it is |
|---|---|
| `final.mp4` | the cut. Shippable on its own |
| `final-visuals.mp4` | the same cut with the overlay pass applied (quote cards, screenshots, chapter roadmap) |
| `plan.json` | the edit decision list: every cut with its reason, the kept segments, and the config that produced them |
| `report.md` | the decisions with their vote counts, every cut, the verification table, and the render command |

An excerpt of `report.md` from the smoke test in Quick start (a 30 second synthetic clip):

```
# talkinghead report

- status: **OK**
- source duration: 30.00s; output: 16.40s; removed: 13.60s in 4 cuts
- transcript: whisper (46 words)

## Decisions

- outro: S0004 (votes: S0004=5)
    - dry run: keep to the last sentence

## Verification

- duration: OK: 16.40s rendered vs 16.40s planned (delta +0.00s, tol 0.50s)
- loudness: OK: -16.1 LUFS (target -16 +-1)
- cold_open: OK: first word ' This' at 0.00s (<= 0.5s required)
- joins: OK: 3/3 joins clean
- resolution: OK: render 1280x720 vs source 1280x720 as displayed
```

The smoke test used the offline dry-run judge, so its cuts are mechanical.
A real judge makes the editorial cuts.

## Requirements

Run each check command in a terminal before you start.
talkinghead runs on macOS on Apple Silicon, on Linux, and on Windows through WSL2 with Ubuntu.
Native Windows is not supported.

### Install the system tools

**macOS (Apple Silicon, Homebrew)**

```bash
brew install ffmpeg librsvg uv
swiftc --version || xcode-select --install   # for visuals; the installer opens a window
```

**Linux (Ubuntu or Debian)**

```bash
sudo apt-get update
sudo apt-get install -y git ffmpeg librsvg2-bin espeak-ng fonts-liberation curl ca-certificates
curl -LsSf https://astral.sh/uv/install.sh | sh
. "$HOME/.local/bin/env"
```

Ubuntu 24.04's ffmpeg 6.1 is enough to run talkinghead, HDR sources included.
Only the developer test suite needs ffmpeg 7.1 or newer, for its HDR test fixtures: install a newer static build if you run the tests (for example from https://github.com/BtbN/FFmpeg-Builds/releases).

**Windows (WSL2)**

1. In PowerShell as administrator, run `wsl --install -d Ubuntu`, then restart when asked.
2. Open the Ubuntu app and follow the Linux steps there.
3. Keep the clone in the Linux home (`~/talkinghead`). File access under `/mnt/c` is much slower.

The Linux steps were verified on Ubuntu 24.04 with its stock ffmpeg 6.1, on both x86_64 and arm64, in Docker.
WSL2 runs the same Ubuntu, but WSL2 itself has not been run end to end yet.

### Check the tools and requirements

| requirement | needed | check command | notes |
|---|---|---|---|
| ffmpeg and ffprobe | always | `ffmpeg -version \| head -1` and `ffprobe -version \| head -1` | 7.1 or newer for the test suite |
| uv (Python project manager) | always | `uv --version` | |
| Python 3.11 or newer | always | `uv run python --version` | on Linux and WSL2 use `uv sync --python 3.12`, since faster-whisper has no wheel for the newest Python releases |
| librsvg (`rsvg-convert`) | with visuals (the default) | `rsvg-convert --version` | `brew install librsvg` or `apt install librsvg2-bin` |
| `swiftc` | with visuals, macOS only | `swiftc --version` | `xcode-select --install` |
| fonts | with visuals, Linux only | `fc-match "Liberation Sans"` | `apt install fonts-liberation` |
| a judge | always | see [Judge values](#judge-values) | Claude Code, omp, or your own command |
| a Scribe key (`ELEVENLABS_API_KEY` or `OPENROUTER_API_KEY`) | with `--transcriber scribe` | `echo "${ELEVENLABS_API_KEY:+elevenlabs set} ${OPENROUTER_API_KEY:+openrouter set}"` prints what is set; an empty line means neither is set | your provider account |
| Whisper model (about 1.5 GB) | with `--transcriber whisper` | downloaded on first use; macOS: `du -sh ~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo`; Linux and WSL2: `du -sh ~/.cache/huggingface/hub/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo` | the model comes from Hugging Face |
| `fluidaudiocli` (Parakeet) | with `--transcriber parakeet`, macOS only | `command -v fluidaudiocli` or `echo "$FLUIDAUDIO_BIN"` | a macOS build of FluidAudio's `fluidaudiocli`, on PATH or named in `FLUIDAUDIO_BIN` |
| Silero VAD (optional) | no | `uv run python -c "import silero_vad"` | `uv sync --extra vad` |
| 10-bit HEVC encoder (for HDR sources) | only for 10-bit HLG or PQ sources | `ffmpeg -hide_banner -encoders \| grep -E "hevc_videotoolbox\|libx265"` | macOS uses `hevc_videotoolbox`; elsewhere `libx265`, which is slower |
| espeak-ng | only for `scripts/make_demo_clip.sh` on Linux | `espeak-ng --version` | `apt install espeak-ng`; macOS uses `say` |

Off macOS these things differ:

- Whisper runs faster-whisper on the CPU with the same large-v3-turbo weights. Its transcripts carry a different version string, so the two engines never share a cache entry.
- HDR output is encoded in software with libx265. It keeps the same 10-bit HLG or PQ tags.
- The head-coverage check needs Apple Vision. Elsewhere it is skipped with a warning, so check the framing of overlays yourself.
- Parakeet runs on macOS only.

Without Silero VAD the mouth-noise pass falls back to envelope-only gap detection, and the run prints one stderr line that says so.
That line is not an error.

## Quick start

Do the smoke test first.
It needs no API keys and no judge subscription, so it proves the install works before anything is spent.

### 1. Install
Install the system tools for your platform first: see [Requirements](#requirements).

```bash
git clone https://github.com/Sipty/talkinghead.git
cd talkinghead
uv sync          # on Linux and WSL2: uv sync --python 3.12
uv run talkinghead --help
```

talkinghead is not a pip package.
Run every command in this README from the clone, with `uv run`.

### 2. Smoke test (offline, a few minutes)

This makes a 30 second synthetic clip with macOS `say` (espeak-ng on Linux), then cuts it with the dry-run judge and the Whisper transcriber.
The dry-run judge answers every prompt with the most conservative answer and calls no model.
The first run downloads the Whisper model (about 1.5 GB) from Hugging Face.

```bash
mkdir -p /tmp/talkinghead-smoke
scripts/make_demo_clip.sh /tmp/talkinghead-smoke/demo.mp4
TALKINGHEAD_JUDGE='shell:python3 -m talkinghead.dryrun_judge' \
TALKINGHEAD_TRANSCRIBER=whisper \
uv run talkinghead run /tmp/talkinghead-smoke/demo.mp4 --no-visuals
```

When it finishes it prints two paths and exits 0:

```
final: /tmp/talkinghead-smoke/demo.talkinghead/final.mp4
report: /tmp/talkinghead-smoke/demo.talkinghead/report.md
```

Open `demo.talkinghead/final.mp4` and `demo.talkinghead/report.md`.
The report status should say `OK`.
The output is about 16 seconds, not 30. The dry-run judge keeps the last sentence, which ends near 18.5 s, and cuts the silence after it. Pauses between sentences are trimmed as well.

### 3. A real run

Choose a judge and a transcriber (see [Configuration](#configuration)) and set them in your shell.
For example, Claude Code as the judge and Scribe v2 for transcription:

```bash
export TALKINGHEAD_JUDGE=claude
export TALKINGHEAD_TRANSCRIBER=scribe
export ELEVENLABS_API_KEY=...        # or OPENROUTER_API_KEY
uv run talkinghead run ~/Movies/talk.mov --audio ~/Movies/mic.wav --topic "one line about the video"
```

`--audio` is optional.
Use it when a separate microphone recording exists (see [Two inputs](#two-inputs)).
`--topic` is one line about the video. It goes into every judgment prompt, and it is the cheapest way to improve the judge's answers.

The run directory is `talk.talkinghead/`, next to `talk.mov`.
Open `talk.talkinghead/final.mp4` and `talk.talkinghead/report.md`.

To preview a long recording first, add `--demo 6`.
That renders only the first 6 minutes at draft quality, and its files are prefixed `demo-`.

## Configuration

Set the judge and the transcriber in the environment, on the command line, or in a config file.
Precedence, highest first: command-line flag, environment variable, config file, built-in default.
Neither the judge nor the transcriber has a built-in default, so a run stops with a clear message until you name them.

| variable | required | values | what it controls | example |
|---|---|---|---|---|
| `TALKINGHEAD_JUDGE` | yes, unless `--judge` or `judge` in the config file is set | `claude`, or `shell:<command>` | the LLM that votes on every editorial decision | `export TALKINGHEAD_JUDGE=claude` |
| `TALKINGHEAD_TRANSCRIBER` | yes, unless `--transcriber` or `transcriber` in the config file is set | `scribe`, `parakeet`, `whisper` | the speech-to-text backend | `export TALKINGHEAD_TRANSCRIBER=whisper` |
| `ELEVENLABS_API_KEY` | for `scribe`, unless `OPENROUTER_API_KEY` is set | ElevenLabs API key | the ElevenLabs route of Scribe v2 (used first when set) | `export ELEVENLABS_API_KEY=...` |
| `OPENROUTER_API_KEY` | for `scribe`, unless `ELEVENLABS_API_KEY` is set | OpenRouter API key | the OpenRouter route of Scribe v2: the only route when no ElevenLabs key is set, and the fallback after an ElevenLabs HTTP or network error | `export OPENROUTER_API_KEY=...` |
| `TALKINGHEAD_CONFIG` | no | path to a TOML file | the config file to load (same as `--config`) | `export TALKINGHEAD_CONFIG=$HOME/talkinghead.toml` |
| `TALKINGHEAD_CACHE` | no | directory; default `~/.cache/talkinghead` | where the content-addressed cache lives | `export TALKINGHEAD_CACHE=/Volumes/scratch/th-cache` |
| `TALKINGHEAD_PROFILE` | no | directory, created on first use | your private editing profile: approved preferences and the feedback log (see [Editing profile](#editing-profile)) | `export TALKINGHEAD_PROFILE=$HOME/talkinghead-profile` |
| `FLUIDAUDIO_BIN` | no, unless `fluidaudiocli` is not on PATH | path to `fluidaudiocli` | the Parakeet binary | `export FLUIDAUDIO_BIN=/opt/fluidaudio/fluidaudiocli` |
| `TDT_EMISSION_DELAY_FRAMES` | no, advanced | integer, passed through | a word-timing shift in `fluidaudiocli` (Parakeet); setting it changes the Parakeet cache key | `export TDT_EMISSION_DELAY_FRAMES=2` |
| `TALKINGHEAD_GAP_BUDGET_FRACTION` | no, advanced | fraction of source duration; default `0.005` | the dead-air budget in the verification table; raise it for a cook that keeps far more pause than the policy trims | `export TALKINGHEAD_GAP_BUDGET_FRACTION=0.15` |
| `HF_HOME` | no | directory | read by the Hugging Face libraries, not by talkinghead; moves the Whisper model cache | `export HF_HOME=/Volumes/scratch/hf` |

### Judge values

`claude` runs Claude Code headless (`claude -p`) on your Claude subscription.
It needs `claude` on PATH and a signed-in account (`claude --version` shows the install).
With no model named, Claude Code uses its own default.
`--model` (or `model` in the config file) sets Claude Code's model, using a name Claude Code accepts, such as an alias like `opus`.
`--effort` sets its effort level.

`shell:<command>` runs any command that reads the prompt on stdin and prints one JSON object on stdout.
For example, omp, which needs a login (`omp login`):

```bash
export TALKINGHEAD_JUDGE='shell:omp -p --no-session --no-tools --mode text --model anthropic/claude-opus-5-5'
```

`--model` rewrites the `--model` token of a shell command.
With no `--model`, the command runs as written.

`shell:python3 -m talkinghead.dryrun_judge` is the offline dry-run judge.
It calls no model and makes no editorial decision. It exists to check the install, so do not publish its cuts.

Before transcription, every run sends one tiny call to the judge.
A missing command, a signed-out account, or a model the provider does not serve stops the run before transcription starts, with exit 4.

Some providers restrict how their subscriptions may be used from third-party tools, so pick a judge route your plan allows.
This README does not give legal advice on those terms.

### Transcriber values

`scribe` sends the 16 kHz mono audio of the whole recording to a provider.
It uses ElevenLabs when `ELEVENLABS_API_KEY` is set, and otherwise OpenRouter's `elevenlabs/scribe-v2` through `OPENROUTER_API_KEY`.
If the ElevenLabs call fails with an HTTP or network error and an OpenRouter key is set, the run falls back to OpenRouter.
A malformed response is an error on either route and is not retried on the other.
The route that answered is recorded in the transcript's version.

`parakeet` runs NVIDIA Parakeet TDT through `fluidaudiocli` on macOS. It is macOS only.
Its word timings are tighter, which matters for the machinery cuts, so it is the better local choice.

`whisper` runs `mlx-community/whisper-large-v3-turbo` through mlx-whisper on macOS. Elsewhere it runs the same model through faster-whisper on the CPU.
The model is downloaded from Hugging Face once, on first use (about 1.5 GB).
Whisper's word spans absorb some hesitations, and the CLI says so when you choose it.

### Config file

`talkinghead.example.toml` lists the most used settings as commented lines. Where a setting has a default, the line shows that default.
Copy it, uncomment what you change, and point talkinghead at the copy with `--config FILE` or `TALKINGHEAD_CONFIG`.
Unknown keys are an error, so a typo stops the run instead of being ignored.
A `talkinghead.toml` in the repository root is not loaded automatically.

The settings most people change first are `votes` and `agree` (the vote size), `content_enabled`, `visuals_enabled`, `visuals_research`, `visuals_corner`, `crf` and `preset` (final quality), and `judge_timeout`.

### Editorial guidelines

`editorial/guidelines.md` is one creator's example rule set.
The whole file goes to the judge with every call, so it decides what the judge values: as the system prompt with `claude`, and at the head of the prompt on stdin with a `shell:` judge.
Guidelines are not part of a judge answer's cache key, so after you edit them a rerun replays answers already given; the stages that depend on them are recomputed from those answers.
Replace it with your own rules in one of two ways:

```bash
uv run talkinghead run talk.mov --guidelines ~/my-rules.md
```

or set `guidelines = "/path/to/my-rules.md"` in the config file.
A relative `guidelines` path resolves inside the clone.

### Editing profile

The editing profile is a private directory that remembers how you like your videos cut.
Point `TALKINGHEAD_PROFILE` (or `--profile DIR`) at it; nothing is created until you first use it.
It holds two files:

| file | what it holds |
|---|---|
| `preferences.md` | the rules you approved, one `- rule` per line. Every judge prompt carries them, for every judge |
| `feedback.jsonl` | every note you recorded, with the source time it points at and its review status |

Feedback comes in three kinds, and each goes to a different place:

| kind | example | where it goes |
|---|---|---|
| `video` (default) | "cut the aside at 3:10" | the log only; it is about this video |
| `preference` | "too salesy" with the lesson "Commit to the pitch" | a proposed lesson, waiting for your approval |
| `defect` | "the 'the' at 7:03 is clipped" | a fixture and a test stub in the clone (see [Other commands](#other-commands)); a tool bug is fixed in code, never learned as a preference |

```bash
uv run talkinghead feedback talk.talkinghead "cut the aside" --at 3:10
uv run talkinghead feedback talk.talkinghead "too salesy" --kind preference --lesson "Commit to the pitch."
uv run talkinghead lessons                         # list proposed lessons
uv run talkinghead lessons accept 2 --text "Commit to the pitch: pain, cost, solution."
uv run talkinghead lessons reject 3
```

`--at` is a time in the cut you watched; the log stores the matching source time, so the note still points at the right moment after a re-cut.
Nothing is learned silently: a lesson becomes a preference only through `lessons accept`, which appends it to `preferences.md`.
You can also edit `preferences.md` by hand.

Each cook keeps a snapshot of the preferences it was judged with, in `<video stem>.talkinghead/preferences.md`.
Changing the profile affects new cooks only, so a rerun of an existing cook replays its cached judge answers.
To re-judge an existing cook with the current profile, add `--refresh-preferences`; only the judge calls whose prompt changed run again.
A cook made before the profile existed is frozen without preferences until you refresh it.
With no profile, or an empty `preferences.md`, the prompts are unchanged.

### Filler words

Every hesitation the transcriber writes as a word (uh, um, erm, hmm, hm, mm and similar) is cut by default, however short.
Two config keys bring back a gentler policy:

| key | default | effect |
|---|---|---|
| `min_filler_dur` | `0.0` | fillers shorter than this many seconds are kept (the earlier default was `0.15`) |
| `filler_keep_voiced_max` | `0.0` | a filler up to this long whose edge runs straight into speech is kept, because the tight splice can sound worse than the filler (the earlier default was `0.35`) |

`min_filler_dur` is part of the pauses stage's memo key, so rerunning a cook made under the old default recomputes that mechanical stage.
No judge call runs again, and the replaced memo is kept as `stages/pauses.superseded.json`.

### Overlay assets

`editorial/assets/` is a small library of diagrams made for this project (MIT, see `editorial/assets/CREDITS.md`).
The overlay pass places an asset where the judge's anchors match it.
To use your own library, set `assets_dir` in the config file or pass `--assets-dir DIR`.
Describe your images in a `manifest.toml` next to them, the same way `editorial/assets/manifest.toml` does.
`visuals_avatar` names an optional square PNG or JPEG that is drawn as a mascot on quote cards.

## Running it with an AI agent

Paste this into Claude Code, Codex, omp or another coding agent, from inside your clone of the repository:

```
Set up and run talkinghead from the README in this repository.
1. Check every requirement with the check commands in the README's Requirements section, and install what is missing with the commands the README gives. Use the platform's package manager (Homebrew on macOS, apt on Linux) and uv only.
2. Run the Quick start smoke test first, exactly as written, and report its exit code and the status line of report.md.
3. Then ask me which judge (claude or a shell command) and which transcriber (scribe, parakeet or whisper) I want, and which API key environment variables I have set.
4. Never write an API key into a file. Keep keys in environment variables in the shell you run commands in.
5. Before you run anything that sends audio or transcript text to a provider, or uses my subscription, tell me what it sends and wait for my yes.
6. Run the real command only after I confirm. Do not change the editorial guidelines unless I ask.
```

## How a run works

1. **Align (only with `--audio`).** The external recording is measured against the video, and a master file is written with that recording as its audio.
2. **Transcribe and analyse.** The audio is transcribed once and cached. Silences, voiced spans and mouth noises are measured from the waveform.
3. **Decide.** Three judged stages choose the cuts: the cold open and outro, the retakes, and the content. The judge votes on candidates that the code generated. Code turns each pick into a cut, so the judge never writes a timestamp. A mechanical stage then handles pauses, fillers and mouth noises.
4. **Render and check.** One ffmpeg render produces the cut and the mastered audio. The output is then measured against the plan: duration, loudness, the first word, join quality, dead air and colour.
5. **Overlay and report.** An optional second pass adds the overlays with the audio copied, so the checks still hold. `report.md` records every decision.

The stage-by-stage map is in [`docs/architecture.md`](docs/architecture.md).

## Two inputs

A phone records the picture and a scratch audio track.
A separate microphone often records a cleaner voice.
`--audio` takes that microphone recording.
talkinghead measures where it sits on the video timeline, remuxes the phone's picture with the microphone audio into `<run>/master.mov`, and cuts that master.
The phone's own audio track is not used.

The measurement is written to `align.json` in the run directory:

- `offset_sec`: the video time at which the recording starts. Positive means the recording started late, so its head is padded with silence. Negative means it started early, so its head is trimmed.
- `drift_ppm`: the clock difference across the whole file. `--fix-drift` time-stretches the audio to cancel it, which matters for recordings longer than a few minutes.
- `confidence`: how clearly the winning match stood out. Below 3x the alignment is weak, and a run refuses to cut on it unless you pass `--allow-bad-align`.

## Other commands

| command | what it does |
|---|---|
| `talkinghead align VIDEO AUDIO [-o OUT] [--json] [--fix-drift] [--yes]` | measure the offset and drift, then remux the microphone audio onto the video. It asks before remuxing, unless `--yes` is given. The output defaults to `VIDEO.aligned.mov` next to the video. `--json` prints the result as JSON on stdout after the remux, and sends the report to stderr. This command never refuses on low confidence; it only reports |
| `talkinghead render PLAN.json --out OUT.mp4` | re-render a finished plan exactly, with no judgment. `--audio-only --out OUT.wav` renders an audio-only plan |
| `talkinghead inspect RUN_DIR` | print a run's `report.md` |
| `talkinghead feedback RUN_DIR "note" [--at TIME] [--kind video\|preference\|defect] [--lesson TEXT]` | record feedback on a cut; see [Editing profile](#editing-profile). `--kind defect --at TIME` also turns what you heard into a fixture and a test stub under the clone's `tests/` |
| `talkinghead lessons [list\|accept ID [--text TEXT]\|reject ID]` | review proposed lessons and promote approved ones to `preferences.md` |
| `talkinghead board` and `talkinghead board-page` | experimental; see [Experimental: board videos](#experimental-board-videos) |

Every subcommand accepts `--help`.

## Time and cost

These are measured examples, not guarantees.

| run | source | judge | transcriber | wall time | machine |
|---|---|---|---|---|---|
| smoke test (Quick start step 2), Whisper model already downloaded | 30 s | dry run | Whisper, `--no-visuals` | 49 s | M-series Mac, fresh clone |
| same | 30 s | dry run | Whisper, `--no-visuals` | 51 s | Ubuntu 24.04 arm64 in Docker on that Mac |
| same | 30 s | dry run | Whisper, `--no-visuals` | 111 s | Ubuntu 24.04 x86_64, emulated in Docker (a native x86 machine is faster) |
| smoke clip with visuals | 30 s | Claude Code (`claude`) | Whisper | 4 min 7 s | M-series Mac |
| real cook | 13 min 38 s | LLM judge | Parakeet, transcript cache warm | 29 min | M5 |
| real cook | 23 min 44 s | LLM judge | Parakeet, transcript cache warm | 50 min | M5 |

A rerun of the same command in the same run directory replays the cached decisions. The smoke test replayed in 9 seconds.

Judge calls: each whole-transcript decision asks the judge five times and needs four identical answers.
One maintainer cook of a 49 minute final made 139 judge samples across 43 distinct prompts, including several reruns.

Costs are not quoted here.
Scribe, the judge and any subscription are billed to your own account by their providers, so check current pricing before a long cook.
The local transcribers and the dry-run judge cost nothing beyond electricity.

## Output and exit codes

| exit | meaning |
|---|---|
| 0 | done; `report.md` says `OK` |
| 1 | an error, including a failed verification check (`report.md` says `FAILED`) |
| 3 | a decision is needed: the judge had no 4-of-5 agreement on a structural question (see Troubleshooting) |
| 4 | the judge is unavailable: command missing, signed out, a model the provider does not serve, or a usage limit that outlasted the wait |
| 130 | interrupted; rerun the same command to resume from the cache |

## Troubleshooting

Every message below is printed to stderr.
After you fix the cause, rerun the same command.
Finished stages and judge answers replay from the cache.

**Exit 3: `cold open: no supermajority` or `outro: no supermajority`.**
The judge's five votes did not reach four identical answers.
The message lists the votes and the flag that answers the question.
Read the candidates in the message, then rerun with a time in seconds, for example `--cold-open 42` or `--outro 310`.
The answer is saved to `decisions.json` in the run directory, and the question is not asked again for that run directory.
To change a saved answer, edit `decisions.json` in the run directory.

**Exit 4: `JUDGE UNAVAILABLE: judge command not found: <name>`.**
The judge command is not on PATH.
Install it, or fix `TALKINGHEAD_JUDGE`.

**Exit 4: `JUDGE UNAVAILABLE` with a sign-in or usage message.**
For omp, run `omp login`.
For Claude Code, sign in with `claude`.
A usage limit still standing after five backoffs (about 31 minutes) ends the run with this exit code.
The run never switches to another model by itself, so wait for the limit to reset and rerun.

**`video track ... vs audio track ... differ by ... Fix the master or pass --allow-av-mismatch`.**
The video and audio streams of the source differ in length by more than 1 second.
This often happens when a file was muxed with `-shortest`.
Export the master with matching track lengths, or pass `--allow-av-mismatch` if you accept the drift.

**`alignment is untrustworthy: match confidence ... below the ... this config requires`.**
`--audio` did not line up with the video clearly enough to cut against.
Check the sync on a known event in both files.
If it is right, pass `--allow-bad-align`.

**`missing tool(s): ...`.**
The message names each missing tool and its install command.
For visuals, `rsvg-convert` is needed (`brew install librsvg`, or `apt install librsvg2-bin`), and on macOS `swiftc` too (`xcode-select --install`).
Pass `--no-visuals` to skip them.

**`fluidaudiocli not found`.**
Put `fluidaudiocli` on PATH or set `FLUIDAUDIO_BIN`, or use `--transcriber whisper`. Parakeet runs on macOS only.

**`Neither ELEVENLABS_API_KEY nor OPENROUTER_API_KEY is set`.**
Scribe needs one of them.
Set one, or use `--transcriber parakeet` or `--transcriber whisper`.

**`ElevenLabs Scribe request failed: HTTP 401`.**
The ElevenLabs key was rejected, or the account has no quota.
With `OPENROUTER_API_KEY` set, the run falls back to OpenRouter and says so.
Without it, the run stops with this message.

**`RunDirConflict` (`... is the cook directory for ...`).**
A run directory belongs to one recording.
Pass `--out-dir` with a new directory for the other recording.

**`verification failed: ...` (exit 1).**
A fatal check failed. The run writes `report.md` with status `FAILED` and exits 1, and the overlay pass does not run.
`final.mp4` is still written, but it failed a check, so do not publish it.
Read the failing line in the verification table of `report.md` and fix its cause, which is usually the source or the alignment.

**`silero-vad unavailable; ...`.**
Informational. Run `uv sync --extra vad` if you want the mouth-noise pass gated by Silero VAD.

## Privacy

What leaves your Mac depends on your choices:

- **Scribe** (`--transcriber scribe`): the 16 kHz mono audio of the whole recording goes to ElevenLabs or OpenRouter.
- **The judge**: the transcript sentences, the guidelines, your editing preferences, the topic and the candidate lists go to the judge's provider. With `claude` that is Anthropic, through Claude Code. With a shell command it is whatever provider that command calls. The dry-run judge sends nothing.
- **Overlay research**: when visuals are on, the URLs the judge proposes are fetched, and pages are screenshotted with `chrome-devtools-axi` if it is installed. `--no-research` turns this off.
- **Whisper and Parakeet**: the audio stays on your machine. Whisper downloads its model from Hugging Face once.

What stays local: the video, the master, the render, the verification and the report.

Where data is kept:

- `~/.cache/talkinghead/` holds the content-addressed cache: transcripts (the words spoken), silences, and voice analysis, each keyed on the hash of the audio it came from. It also holds `bin/`, a helper cache for the head-clip check.
- `<video stem>.talkinghead/` holds the run: the plan, the report, the preferences snapshot, every judge prompt and answer under `llm/`, and the stage memos. Prompts include transcript text, so treat a run directory like the recording itself.
- The editing profile directory holds your preferences and every feedback note, with the run directory and source path it refers to.
- The Whisper model lives in the Hugging Face cache, under `~/.cache/huggingface/` unless `HF_HOME` is set.

## Cleanup and uninstall

Remove one run:

```bash
rm -rf talk.talkinghead
```

Remove the shared cache.
It holds the Scribe transcripts, so removing it means paying for those transcripts again.

```bash
rm -rf ~/.cache/talkinghead
```

Remove the Whisper model on macOS:

```bash
rm -rf ~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo
```

Remove the Whisper model on Linux and WSL2:

```bash
rm -rf ~/.cache/huggingface/hub/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo
```

Remove the program: delete the clone, which includes the virtual environment in `.venv/`.

```bash
cd ..
rm -rf talkinghead
```

## Experimental: board videos

`talkinghead board` and `talkinghead board-page` render and edit a tablet panel next to the speaker.
They are experimental, and the spec format may change before a stable release.
`board-page` opens a browser on port 8765 unless you pass `--no-open`.
The spec is `<RUN_DIR>/board/board.json`. It holds the tablet recording's path, an `offset` that aligns the tablet's clock with the phone's, and views (time ranges with a rectangle on the canvas).
`board` renders `final-board.mp4` in the run directory.

## Limitations

- Parakeet (`fluidaudiocli`) and the Apple Vision head-coverage check run on macOS only. Elsewhere the head check is skipped with a warning.
- HDR output encodes in software with libx265 off macOS. It is slower than the Mac hardware encoder.
- Windows runs through WSL2 only. Native Windows is not supported.
- The Whisper engine differs by platform (mlx-whisper on macOS, faster-whisper elsewhere), so the two platforms do not share transcript caches.
- The cut is only as good as the judge and the guidelines. Read the report before you publish, and watch the cut.
- HDR is tested on 10-bit HLG sources. Other HDR formats are untested.
- Long cooks are slow, and they cost money in judge calls and transcription.
- `talkinghead feedback --kind defect` writes into the clone's `tests/` directory, so it works only from a clone.
- The default overlay library and the example guidelines are examples. Replace them for your own channel.
- talkinghead runs from a clone. It is not published as a package, and its bundled files (the guidelines and the assets) are found relative to the clone.

## Development

```bash
uv sync
uv run pytest -q
uvx ruff check   # rule set pinned in pyproject.toml; the tree is clean
```

Some tests call a live judge.
They are skipped unless `TALKINGHEAD_JUDGE_TESTS=1` is set, and that setting needs a working judge.
The test suite removes `TALKINGHEAD_JUDGE`, `TALKINGHEAD_TRANSCRIBER`, `TALKINGHEAD_CONFIG` and `TALKINGHEAD_PROFILE` for each test, so your shell settings do not leak into it.

The code is in `src/talkinghead/`.
`scripts/render_proof.py` is the maintainer's 31-minute render measurement. It reads `TALKINGHEAD_RENDER_PROOF_SOURCE` and `TALKINGHEAD_RENDER_PROOF_OUT_DIR` when its `--source` and `--out-dir` flags are not given.
[`docs/architecture.md`](docs/architecture.md) maps each stage to its file.
[`docs/render-proof.md`](docs/render-proof.md) records the measurement behind the render strategy.
[`docs/noise-detector-spec.md`](docs/noise-detector-spec.md) records the mouth-noise calibration.
[`docs/releasing.md`](docs/releasing.md) explains how every push becomes a tagged release and how the version number is chosen.

## Updating

Every change to talkinghead ships as a tagged release with notes on the repository's Releases page.
`talkinghead --version` shows the version you have.
To update a clone:

```bash
git pull
uv sync
```

## Glossary

- **Cook**: one run of talkinghead on one recording, and its run directory.
- **Judge**: the LLM that makes the editorial decisions, through `claude` or a shell command.
- **Vote**: the judge is asked five times for a whole-transcript decision, and four identical answers make it a decision. Restart markers use three asks and two answers.
- **Stage**: one decision step: cold open and outro, retakes, content, then the mechanical pauses and edges.
- **Memo**: a stage's saved result, keyed on the hash of its inputs, so a rerun replays it.
- **HLG**: hybrid log-gamma, an HDR transfer format that many phone cameras record in. talkinghead keeps 10-bit HLG as 10-bit HLG.
- **LUFS**: loudness units relative to full scale. The final audio targets -16 LUFS.
- **Overlay**: a picture drawn over the finished cut: a quote card, a screenshot or the chapter roadmap.
- **Board**: an experimental panel that shows a tablet canvas beside the speaker.

## License

MIT. See [LICENSE](LICENSE).
