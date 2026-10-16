"""MakeHuman morph targets (voxgen.mh_targets): catalogue, resolution, parsing, application.
Offline: only the targets committed under assets/makehuman/targets are touched."""
import numpy as np
import pytest

from voxgen import mh, mh_targets as T


def test_catalogue_names_sides_and_pairs():
    cat = T.catalogue()
    m = cat["nose-scale-vert"]
    assert m.files() == ("nose/nose-scale-vert-decr.target", "nose/nose-scale-vert-incr.target")
    assert cat["head-oval"].files() == (None, "head/head-oval.target")
    assert "nose-trans" not in cat and "nose-trans-down|up" in cat      # ambiguous stem, canonical name
    assert T.resolve("nose-trans-up") == (cat["nose-trans-down|up"], 1)
    assert T.resolve("nose-trans-down")[1] == -1
    assert "Gender" not in cat and "Age" not in cat                      # macros skipped
    assert T.expand_pairs({"ear-scale": 1.5}) == {"l-ear-scale": 1.5, "r-ear-scale": 1.5}
    with pytest.raises(KeyError):
        T.expand_pairs({"nose-size": 1.0})
    files = dict(T.files_for({"nose-scale-vert": -0.5, "head-oval": 0.5, "nose-trans-down": 0.3, "ear-scale": 0}))
    assert files == {"nose/nose-scale-vert-decr.target": 0.5, "head/head-oval.target": 0.5,
                     "nose/nose-trans-down.target": 0.3}


def test_parse_and_apply_synthetic():
    idx, off = T.parse_target("# comment\n5 0.1 0 0\n7 0 -0.2 0.3\n\n")
    assert idx.tolist() == [5, 7] and off.shape == (2, 3)
    with pytest.raises(ValueError):
        T.parse_target("19158 0 0 0")
    with pytest.raises(ValueError):
        T.parse_target("1 0 0")
    v = np.zeros((T.N_VERTICES, 3))
    out = v.copy()
    np.add.at(out, idx, off * 2.0)
    assert out[7].tolist() == [0.0, -0.4, 0.6] and (out != 0).sum() == 3
    with pytest.raises(ValueError):
        T.apply_morphs(np.zeros((10, 3)), {})


def test_bundled_targets_are_valid_and_morph_moves_landmarks():
    rels = sorted(p.relative_to(T.ASSET_TARGETS).as_posix() for p in T.ASSET_TARGETS.rglob("*.target"))
    assert rels, "no bundled targets"
    for rel in rels:
        idx, off = T.parse_target(T.target_path(rel, download=False).read_text())
        assert idx.size > 0 and np.abs(off).max() < 2.0, rel          # offsets are decimetres
    base = mh.load().landmarks()
    nosey = mh.load(morphs={"nose-scale-vert": 1.6, "chin-prominent": -0.5}).landmarks()
    assert nosey["nose_tip"][1] < base["nose_tip"][1] - 0.005         # a longer nose ends lower
    assert abs(nosey["head_height"] - base["head_height"]) < 0.03     # the rest stays put
    assert abs(nosey["eye_l"][0] + nosey["eye_r"][0]) < 1e-6          # still symmetric
    ears = mh.load(morphs={"ear-scale": 1.8}).landmarks()
    assert ears["ear_x"] > base["ear_x"] + 0.008
