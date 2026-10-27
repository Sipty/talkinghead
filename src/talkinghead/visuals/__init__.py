"""Visuals: corner overlays placed over the finished cut.

The cut pipeline is untouched. The overlays are composited inside the cut's one
picture encode (`render.render_plan(..., overlays=...)`), muxed with the locked
audio master, so the audio checks keep their exact meaning and no second lossy
generation exists. `final-visuals.mp4` is the deliverable.
"""

from __future__ import annotations
