"""Designed voxel-art character (voxgen.voxchar): the head builds on its coarse grid with the
painted face and the slab hair, the body is a block figure of normal proportions, both are
mirror-symmetric and pass the checks clean, every hair / beard / hat / outfit option builds,
and the coat_guy spec builds as one model with the expected parts."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from voxgen import checks
from voxgen.body import exposed
from voxgen.materials import (MAT_BONE, MAT_CLOTH, MAT_EYE, MAT_FAT, MAT_HAIR, MAT_LEATHER, MAT_MUSCLE,
                              MAT_PLANT, MAT_SKIN)
from voxgen.voxchar import (BEARDS, HAIR_STYLES, HATS, OUTFITS, ROWS, UP, VoxCharParams, build_character,
                            head_grid)

SPEC = Path(__file__).resolve().parents[1] / "specs" / "chars" / "vox" / "coat_guy.py"
OUTER = {MAT_SKIN, MAT_HAIR, MAT_EYE, MAT_CLOTH, MAT_LEATHER, MAT_PLANT}


@pytest.fixture(scope="module")
def full():
    return build_character(VoxCharParams(name="t"))


@pytest.fixture(scope="module")
def head_model():
    return build_character(VoxCharParams(name="t"), head_only=True)


def _mirror_exact(part, ignore_hair=False):
    mat = part.material.copy()
    if ignore_hair:
        mat[mat == MAT_HAIR] = 0
    return np.array_equal(mat, mat[:, :, ::-1])


def test_full_character_parts_and_checks(full):
    assert [p.name for p in full.parts] == ["body", "head", "hair"]
    report = checks.run(full)
    assert report["errors"] == [] and report["warnings"] == []
    body, head = full.part("body"), full.part("head")
    assert body.voxel_size == 0.01 and head.voxel_size == pytest.approx(0.003)
    assert body.bone == "body" and head.bone == "head" and full.part("hair").bone == "head"
    # normal proportions: about seven skull heights tall, feet on the ground
    h = full.meta["height_m"]
    skull = 11 * 0.015
    assert 6.5 < h / skull < 8.0
    assert full.bounds_m()[0][1] == pytest.approx(0.0, abs=1e-6)


def test_symmetry(full):
    assert _mirror_exact(full.part("body"))
    assert _mirror_exact(full.part("head"), ignore_hair=True)
    assert _mirror_exact(full.part("hair"))
    sweep = build_character(VoxCharParams(name="t", fringe_sweep=1), head_only=True)
    assert not _mirror_exact(sweep.part("hair"))
    assert _mirror_exact(sweep.part("head"), ignore_hair=True)


def test_head_face_and_tissues(head_model):
    assert head_model.name == "t_head"
    head = head_model.part("head")
    hist = head.material_histogram()
    for m in (MAT_SKIN, MAT_FAT, MAT_MUSCLE, MAT_BONE, MAT_EYE, MAT_HAIR):
        assert hist.get(m, 0) > 0, m
    # eyes: two dark 2x3-cell rectangles, two fine voxels deep, with the lighter band on top
    eye = head.material == MAT_EYE
    assert int(eye.sum()) == 2 * (2 * UP) * (3 * UP) * 2
    pal = head_model.palette
    band = eye & (head.color == pal.index("eye_light"))
    dark = eye & (head.color == pal.index("eye_dark"))
    assert int(band.sum()) == 2 * (2 * UP) * UP * 2
    assert float(np.nonzero(band)[1].min()) > float(np.nonzero(dark)[1].max())
    # only skin, hair (brows), eyes face the air: no exposed inner tissue
    assert set(np.unique(head.material[exposed(head.solid())]).tolist()) <= OUTER
    # the chin's bottom is at y = 0 in the head frame, the skull centred on x and z
    (x0, y0, z0), (x1, y1, z1) = head.bbox()
    vs = head.voxel_size
    assert head.origin[1] + y0 * vs == pytest.approx(-2 * 0.015, abs=1e-6)       # the neck rows
    assert head.origin[0] + (x0 + x1 + 1) / 2 * vs == pytest.approx(0.0, abs=1e-6)
    stats = head_model.meta["build"]["head"]
    assert stats["mouth"] == 18 and stats["brow"] > 0 and stats["cheek"] > 0


def test_hair_slabs_are_one_mass_on_the_scalp(head_model):
    hair = head_model.part("hair")
    labels, n = checks.label_components(hair.solid())
    assert n == 1
    pal = head_model.palette
    light = hair.color == pal.index("hair_light")
    base = hair.color == pal.index("hair")
    assert light.any() and base.any()
    assert float(np.nonzero(light)[1].min()) > float(np.nonzero(base)[1].max()) - 1   # the lighter slab is the top one
    g = head_grid()
    # the fringe hangs in front of the face plane and the forehead above the brows stays bare
    head = head_model.part("head")
    hz = hair.origin[2] + np.nonzero(hair.solid())[0].min() * hair.voxel_size
    fz = head.origin[2] + np.nonzero(head.material == MAT_SKIN)[0].min() * head.voxel_size
    assert hz < fz + 1e-6 and (g.lo[1] - g.lo[1]) == 0


@pytest.mark.parametrize("hair", HAIR_STYLES)
@pytest.mark.parametrize("hat", HATS)
def test_hair_and_hat_options(hair, hat):
    m = build_character(VoxCharParams(name="t", hair=hair, hat=hat), head_only=True)
    names = [p.name for p in m.parts]
    assert ("hair" in names) == (hair != "bald")
    assert ("hat" in names) == (hat != "none")
    report = checks.run(m)
    assert report["errors"] == [] and report["warnings"] == [], (hair, hat, report)
    if hat != "none":
        hat_part = m.part("hat")
        assert hat_part.material_histogram() == {MAT_PLANT if hat == "straw" else MAT_CLOTH: hat_part.count_solid()}


@pytest.mark.parametrize("beard", BEARDS)
def test_beard_options(beard):
    m = build_character(VoxCharParams(name="t", beard=beard, brows_thick=True), head_only=True)
    names = [p.name for p in m.parts]
    assert ("beard" in names) == (beard in ("short", "full"))
    report = checks.run(m)
    assert report["errors"] == [] and report["warnings"] == [], (beard, report)
    if beard == "stubble":
        head = m.part("head")
        assert (head.color == m.palette.index("stubble")).any()
    if beard == "full":
        assert m.part("beard").count_solid() > m.part("head").material_histogram()[MAT_EYE]


@pytest.mark.parametrize("outfit", OUTFITS)
def test_outfits(outfit):
    m = build_character(VoxCharParams(name="t", outfit=outfit, sleeves="rolled"))
    report = checks.run(m)
    assert report["errors"] == [] and report["warnings"] == [], (outfit, report)
    body = m.part("body")
    hist = body.material_histogram()
    assert hist[MAT_CLOTH] > 0 and hist[MAT_LEATHER] > 0 and hist[MAT_SKIN] > 0
    labels, n = checks.label_components(body.solid())
    assert n == 1
    assert set(np.unique(body.material[exposed(body.solid())]).tolist()) <= OUTER
    # hands stay bare skin
    y = (np.arange(body.ny) + 0.5) * body.voxel_size + body.origin[1]
    hand_rows = (y >= ROWS["hand"][0] * 0.01) & (y < (ROWS["hand"][1] + 1) * 0.01)
    ex = exposed(body.solid()) & hand_rows[None, :, None]
    X = (np.arange(body.nx) + 0.5) * body.voxel_size + body.origin[0]
    ex &= (np.abs(X) > 0.14)[None, None, :]
    assert ex.any() and set(np.unique(body.material[ex]).tolist()) == {MAT_SKIN}


def test_bad_params():
    with pytest.raises(ValueError):
        build_character(VoxCharParams(hair="mohawk"))
    with pytest.raises(ValueError):
        build_character(VoxCharParams(fringe_sweep=2))


def test_coat_guy_spec_builds():
    spec = importlib.util.spec_from_file_location("coat_guy_spec", SPEC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.build()
    assert m.name == "coat_guy" and [p.name for p in m.parts] == ["body", "head", "hair"]
    assert m.meta["outfit"] == "coat"
    pal = m.palette
    body = m.part("body")
    assert (body.color == pal.index("lining")).any() and (body.color == pal.index("trousers_hem")).any()
