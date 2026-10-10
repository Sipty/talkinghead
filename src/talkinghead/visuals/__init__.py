"""Visuals: corner overlays placed over the finished cut.

The cut pipeline is untouched. Visuals run as a SECOND ffmpeg pass over
`final.mp4` with the audio stream copied, so duration, av_sync, loudness,
cold_open and residual_gaps keep their exact meaning and a bad overlay can
never cost the cut. `final-visuals.mp4` is the deliverable.
"""

from __future__ import annotations
