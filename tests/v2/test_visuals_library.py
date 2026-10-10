from talkinghead.visuals.library import Library, score


def _lib(tmp_path, *entries):
    lines = []
    for id, depicts, kind in entries:
        (tmp_path / f"{id}.png").write_bytes(b"x")
        lines.append(f'[[asset]]\nid = "{id}"\nfile = "{id}.png"\ndepicts = "{depicts}"\nkind = "{kind}"\n')
    (tmp_path / "manifest.toml").write_text("\n".join(lines))
    return Library.load(tmp_path)


def test_match_above_threshold(tmp_path):
    lib = _lib(tmp_path, ("s-zbench", "GLM 5.3 Flash benchmark chart Claude Opus", "shot"))
    asset, s = lib.match("GLM 5.3 Flash which is competing with Claude Opus 4.8", None)
    assert asset.id == "s-zbench"
    assert s >= 0.70


def test_unrelated_text_scores_low(tmp_path):
    lib = _lib(tmp_path, ("s-cache", "Anthropic prompt caching documentation", "shot"))
    _, s = lib.match("Kimi K three is fantastic but it is too expensive", None)
    assert s < 0.70


def test_best_of_several_wins(tmp_path):
    lib = _lib(tmp_path,
               ("s-omp", "oh-my-pi omp GitHub repository", "shot"),
               ("s-cache", "Anthropic prompt caching time to live", "shot"))
    asset, _ = lib.match("my time to live was set to five minutes instead of one hour for the cache")
    assert asset.id == "s-cache"


def test_ingest_appends_to_manifest(tmp_path):
    (tmp_path / "manifest.toml").write_text("")
    (tmp_path / "src.png").write_bytes(b"x")
    lib = Library.load(tmp_path)
    lib.ingest(tmp_path / "src.png", id="p1-scout", depicts="agent pipeline scout stage", kind="card")
    assert 'id = "p1-scout"' in (tmp_path / "manifest.toml").read_text()
    assert Library.load(tmp_path).assets[0].id == "p1-scout"


def test_ingest_is_idempotent(tmp_path):
    (tmp_path / "manifest.toml").write_text("")
    (tmp_path / "src.png").write_bytes(b"x")
    lib = Library.load(tmp_path)
    lib.ingest(tmp_path / "src.png", id="a", depicts="x", kind="card")
    Library.load(tmp_path).ingest(tmp_path / "src.png", id="a", depicts="x", kind="card")
    assert (tmp_path / "manifest.toml").read_text().count('id = "a"') == 1


def test_missing_file_is_skipped(tmp_path):
    (tmp_path / "manifest.toml").write_text(
        '[[asset]]\nid = "gone"\nfile = "gone.png"\ndepicts = "nothing"\nkind = "card"\n')
    assert Library.load(tmp_path).assets == []


def test_scale_follows_kind(tmp_path):
    lib = _lib(tmp_path, ("s", "a", "shot"), ("c", "b", "card"))
    by = {a.id: a.scale(card=0.28, shot=0.34) for a in lib.assets}
    assert by == {"s": 0.34, "c": 0.28}


def test_score_is_asymmetric_toward_depicts():
    """A short, exact description should not be punished by a long anchor."""
    assert score("prompt caching", "the whole long sentence about prompt caching here") == 1.0
