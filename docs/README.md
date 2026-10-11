# docs

The architecture map and the measurement records the live code cites. One line per file.

## Live

| file | what it is |
|---|---|
| `README.md` | this index |
| `architecture.md` | stage-by-stage map of the pipeline: what each stage decides, judge-driven vs deterministic, what it writes, where the cache lives |
| `noise-detector-spec.md` | calibration data and safety rules behind the mouth-noise detector; `src/talkinghead/noise.py` cites it and the `## Implementation` section tracks the live wiring |
| `render-proof.md` | the measurement that chose the render strategy (filter_complex trim/atrim + concat over `select` and per-segment concat) at 31 minutes and 209 segments; `src/talkinghead/render.py` cites it and still uses strategy A |
| `releasing.md` | how every push to main becomes a tagged GitHub Release, and how `Release:` commit trailers pick the version bump |

`editorial/guidelines.md` is not in this directory on purpose: it is the judge's editorial rule set, a pipeline input rather than documentation.
