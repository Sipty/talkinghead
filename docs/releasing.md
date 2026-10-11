# Releasing

Every push to `main` is a release.
The `release` workflow (`.github/workflows/release.yml`) tags the next `vX.Y.Z` and publishes a GitHub Release whose notes are the pushed commit messages.
Nobody tags or edits versions by hand.

## How the version is chosen

`scripts/release.py` reads every commit since the last `vX.Y.Z` tag and takes the largest bump any of them asks for, with a `Release:` trailer on the last line of the commit message:

| trailer | bump | use it for |
|---|---|---|
| none | patch: `1.1.0` -> `1.1.1` | fixes, docs, internal changes, small tweaks |
| `Release: minor` | minor: `1.1.3` -> `1.2.0` | a new feature or capability a user would notice |
| `Release: major` | major: `1.4.0` -> `2.0.0` | a breaking change; only the maintainer decides this |

An unknown value (`Release: minr`) fails the workflow instead of guessing.

## Writing commits that read well as release notes

The release notes are the commit subjects and bodies, oldest first, without the trailer.
Write the subject for a user: what changed, not which file.
Put the detail a user needs in the body.

```
Run on Linux and Windows (WSL2)

Whisper uses faster-whisper off macOS and HDR sources encode with libx265.
The head-coverage check is skipped off macOS with a warning.

Release: minor
```

## Versions inside the tool

`talkinghead --version` reads the tag from the clone (`git describe`).
Between releases it shows the commits since the last tag, for example `1.1.0+3.gabc1234`.
The `version` field in `pyproject.toml` is a placeholder and is not the release version.
