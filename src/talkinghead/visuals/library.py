"""The asset library: what we already have, and whether it fits.

Overlays used to be hand-aimed at one video, so nothing carried forward. Here
each asset declares in a manifest WHAT IT DEPICTS, and an anchor earns an
asset only by clearing a match score. Below the threshold the caller produces a
fresh asset instead of reusing the closest old one: silently reusing a loose match
puts a picture on screen that does not depict the claim.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

_WORD = re.compile(r"[a-z0-9.]+")
# Words that carry no topical signal; kept small on purpose, since dropping
# too much makes short `depicts` strings match everything.
_STOP = frozenset("""
a an the and or of for to in on at by with from as is are was were be been it its this that
these those he she they you i we his her their your our my me him them us if then than so
but not no yes do does did done have has had will would can could should may might must
about into over under out up down more most less least very just also too only own same
""".split())


@dataclass(frozen=True)
class Asset:
    id: str
    path: Path
    depicts: str
    kind: str                          # "shot" | "card"
    # Phrases this asset should appear on, in preference order. An asset with
    # hints places itself when one of them survives the cut, instead of waiting
    # for the judge to think of it. Without hints, a judge that names only some
    # of the pipeline stages leaves the asset unplaced.
    anchors: tuple[str, ...] = ()
    # Members of a sequence are ONE visual that changes, not several competing
    # overlays: a build-up chains with no gap between steps.
    sequence: str | None = None
    order: int = 0
    hold: float | None = None          # explicit dwell, when the asset wants one

    def scale(self, *, card: float, shot: float) -> float:
        return shot if self.kind == "shot" else card


def tokens(text: str) -> set[str]:
    return {t.strip(".") for t in _WORD.findall(text.lower())} - _STOP - {""}


def score(depicts: str, anchor: str, topic: str | None = None) -> float:
    """Overlap of `depicts` against the anchor, as a fraction of `depicts`.

    Asymmetric on purpose: an anchor sentence is long and an asset's
    description is short, so Jaccard would punish a perfect asset for the
    anchor's extra words. What matters is whether what the asset depicts is
    being talked about.
    """
    d, a = tokens(depicts), tokens(anchor)
    if not d:
        return 0.0
    if topic:
        a = a | tokens(topic)
    return len(d & a) / len(d)


class Library:
    def __init__(self, root: Path, assets: list[Asset]) -> None:
        self.root = Path(root)
        self.assets = assets

    @classmethod
    def load(cls, root: str | Path) -> "Library":
        root = Path(root)
        manifest = root / "manifest.toml"
        entries: list[Asset] = []
        if manifest.is_file():
            data = tomllib.loads(manifest.read_text())
            for e in data.get("asset", []):
                p = root / e["file"]
                if not p.is_file():   # a manifest entry with no file is skipped, not fatal
                    continue
                entries.append(Asset(
                    e["id"], p, e["depicts"], e.get("kind", "card"),
                    anchors=tuple(e.get("anchors", ())),
                    sequence=e.get("sequence"), order=int(e.get("order", 0)),
                    hold=float(e["hold"]) if e.get("hold") is not None else None))
        return cls(root, entries)

    def match(self, anchor: str, topic: str | None = None) -> tuple[Asset | None, float]:
        best: tuple[Asset | None, float] = (None, 0.0)
        for a in self.assets:
            s = score(a.depicts, anchor, topic)
            if s > best[1]:
                best = (a, s)
        return best

    def ingest(self, path: str | Path, *, id: str, depicts: str, kind: str = "card") -> Asset:
        """Copy a file in and append its manifest entry, so it is reusable."""
        import shutil

        src = Path(path)
        self.root.mkdir(parents=True, exist_ok=True)
        dest = self.root / f"{id}{src.suffix}"
        if src.resolve() != dest.resolve():
            shutil.copyfile(src, dest)
        manifest = self.root / "manifest.toml"
        existing = manifest.read_text() if manifest.is_file() else ""
        if f'id = "{id}"' not in existing:
            block = (f'\n[[asset]]\nid = "{id}"\nfile = "{dest.name}"\n'
                     f'depicts = {_toml_str(depicts)}\nkind = "{kind}"\n')
            manifest.write_text(existing.rstrip("\n") + "\n" + block if existing.strip() else block.lstrip("\n"))
        asset = Asset(id, dest, depicts, kind)
        self.assets.append(asset)
        return asset


def _toml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
