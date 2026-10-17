"""Designed voxel-art character (voxgen.voxchar): the head builds as a rounded box with the
painted face (white-block eyes with lid and pupil, proud brows, two nose blocks, the mouth line
with the tooth row behind it, ear notches) and the hair, the body is a block figure of normal
proportions, both are mirror-symmetric and pass the checks clean, every hair / beard / hat /
outfit option builds, and the coat_guy / barber specs build as one model with the expected parts."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from voxgen import checks
from voxgen.body import exposed
from voxgen.materials import (MAT_AIR, MAT_BONE, MAT_CLOTH, MAT_EYE, MAT_FAT, MAT_HAIR, MAT_LEATHER, MAT_MUSCLE,
                              MAT_PLANT, MAT_SKIN, MAT_TOOTH)
from voxgen.voxchar import (BACK, BEARDS, HAIR_STYLES, HATS, HW, OUTFITS, ROWS, TOP, UP, VoxCharParams,
                            build_character)

SPECS = Path(__file__).resolve().parents[1] / "specs" / "chars" / "vox"
OUTER = {MAT_SKIN, MAT_HAIR, MAT_EYE, MAT_CLOTH, MAT_LEATHER, MAT_PLANT}


def load_spec(name: str):
    """(the spec module, its built model)."""
    spec = importlib.util.spec_from_file_location(f"{name}_spec", SPECS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, mod.build()


def overlap(a, b) -> bool:
    """Whether two parts of one voxel size share a solid voxel (their arrays are cropped differently)."""
    vs = a.voxel_size
    off = tuple(int(round((b.origin[i] - a.origin[i]) / vs)) for i in range(3))
    for z, y, x in zip(*np.nonzero(a.solid())):
        bx, by, bz = x - off[0], y - off[1], z - off[2]
        if 0 <= bx < b.nx and 0 <= by < b.ny and 0 <= bz < b.nz and b.material[bz, by, bx]:
            return True
    return False


def touching(a, b) -> bool:
    """Whether some solid voxel of ``a`` has a 6-neighbour solid in ``b``."""
    vs = a.voxel_size
    off = tuple(int(round((b.origin[i] - a.origin[i]) / vs)) for i in range(3))
    for z, y, x in zip(*np.nonzero(a.solid())):
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            bx, by, bz = x + dx - off[0], y + dy - off[1], z + dz - off[2]
            if 0 <= bx < b.nx and 0 <= by < b.ny and 0 <= bz < b.nz and b.material[bz, by, bx]:
                return True
    return False


def at(part, X: int, Y: int, Z: int) -> int:
    """Material of the part at skull-frame coordinates (x 0..44, y 0..54, z 0..49)."""
    vs = part.voxel_size
    ix = int(round(X - part.origin[0] / vs - HW - 0.5))
    iy = int(round(Y - part.origin[1] / vs))
    iz = int(round(Z - part.origin[2] / vs - (BACK + 1) / 2))
    if not (0 <= ix < part.nx and 0 <= iy < part.ny and 0 <= iz < part.nz):
        return MAT_AIR
    return int(part.material[iz, iy, ix])


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
    # eyes: two white 10x8 blocks two voxels deep, the dark lid rows on top, a 4x4 pupil under
    # the lid with one gleam voxel
    eye = head.material == MAT_EYE
    assert int(eye.sum()) == 2 * 10 * 8 * 2
    pal = head_model.palette
    lid = eye & (head.color == pal.index("eye_lid"))
    white = eye & (head.color == pal.index("eye_white"))
    pupil = eye & (head.color == pal.index("eye_dark"))
    gleam = eye & (head.color == pal.index("eye_light"))
    assert int(lid.sum()) == 2 * 10 * 2 * 2 and int(pupil.sum()) == 2 * 4 * 4 * 2 - 2 and int(gleam.sum()) == 2
    assert int(white.sum()) == int(eye.sum()) - 80 - 64
    assert float(np.nonzero(lid)[1].min()) > float(np.nonzero(pupil)[1].max())
    # the tooth row hides behind the mouth line: eight 2x2x2 teeth, none exposed
    teeth = head.material == MAT_TOOTH
    assert int(teeth.sum()) == 8 * 2 * 2 * 2
    assert not (teeth & exposed(head.solid())).any()
    assert at(head, 20, 17, 2) == MAT_TOOTH and at(head, 22, 17, 2) == MAT_FAT and at(head, 20, 17, 0) == MAT_SKIN
    # only skin, hair (brows), eyes face the air: no exposed inner tissue
    assert set(np.unique(head.material[exposed(head.solid())]).tolist()) <= OUTER
    # the chin's bottom is at y = 0 in the head frame, the skull centred on x and z
    (x0, y0, z0), (x1, y1, z1) = head.bbox()
    vs = head.voxel_size
    assert head.origin[1] + y0 * vs == pytest.approx(-2 * 0.015, abs=1e-6)       # the neck rows
    assert head.origin[0] + (x0 + x1 + 1) / 2 * vs == pytest.approx(0.0, abs=1e-6)
    stats = head_model.meta["build"]["head"]
    assert stats["mouth"] == 22 and stats["brow"] > 0 and stats["cheek"] > 0 and stats["teeth"] == 64


def test_head_form_is_a_rounded_box_with_details(head_model):
    head = head_model.part("head")
    # the face plane is flat skin, the box's edges and corners are cut (45-degree staircases)
    assert at(head, HW, 48, 0) == MAT_SKIN and at(head, HW, TOP, 20) == MAT_SKIN
    assert at(head, 0, 30, 0) == MAT_AIR and at(head, 2, 30, 0) == MAT_AIR and at(head, 3, 30, 0) == MAT_SKIN
    assert at(head, 0, TOP, 20) == MAT_AIR and at(head, HW, TOP, BACK) == MAT_AIR and at(head, 0, 0, 0) == MAT_AIR
    assert at(head, HW, TOP, BACK - 8) == MAT_SKIN and at(head, HW, TOP - 8, BACK) == MAT_SKIN
    # the jaw is narrower than the cranium below the ear line
    assert at(head, 44, 45, 25) == MAT_SKIN and at(head, 44, 10, 25) == MAT_AIR and at(head, 40, 10, 25) == MAT_SKIN
    # ears: blocks on the side planes with a two-voxel notch in the outer face
    assert at(head, 46, 28, 21) == MAT_SKIN and at(head, 48, 28, 21) == MAT_SKIN
    assert at(head, 48, 31, 24) == MAT_AIR and at(head, 46, 31, 24) == MAT_SKIN
    # the nose: the tip block five voxels proud with softened front edges, the bridge two proud
    assert at(head, HW, 27, -5) == MAT_SKIN and at(head, HW, 24, -5) == MAT_AIR and at(head, HW, 24, -4) == MAT_SKIN
    assert at(head, HW, 34, -2) == MAT_SKIN and at(head, HW, 34, -3) == MAT_AIR
    # brows: hair material two voxels proud of the face, the inner end a row lower
    assert at(head, 30, 44, -1) == MAT_HAIR and at(head, 30, 44, -2) == MAT_HAIR and at(head, 30, 44, -3) == MAT_AIR
    assert at(head, 26, 42, -1) == MAT_HAIR and at(head, 34, 42, -1) == MAT_AIR
    # the mouth line with its corners a voxel up (colour only)
    pal = head_model.palette
    mouth = head.color == pal.index("mouth")
    ys = np.nonzero(mouth)[1]
    assert int(mouth.sum()) == 22 and ys.max() - ys.min() == 2


def test_hair_slabs_are_one_mass_on_the_scalp(head_model):
    hair = head_model.part("hair")
    labels, n = checks.label_components(hair.solid())
    assert n == 1
    pal = head_model.palette
    light = hair.color == pal.index("hair_light")
    base = hair.color == pal.index("hair")
    assert light.any() and base.any()
    assert float(np.nonzero(light)[1].min()) > float(np.nonzero(base)[1].max()) - 1   # the lighter slab is the top one
    # the shell is three voxels thick in front of the face plane, the quiff one more
    head = head_model.part("head")
    hz = hair.origin[2] + np.nonzero(hair.solid())[0].min() * hair.voxel_size
    fz = head.origin[2] + np.nonzero(head.material == MAT_EYE)[0].min() * head.voxel_size
    assert hz == pytest.approx(fz - 4 * head.voxel_size, abs=1e-6)
    assert at(hair, HW, 56, -3) == MAT_HAIR and at(hair, HW, 56, -6) == MAT_AIR


@pytest.mark.parametrize("hair", HAIR_STYLES)
@pytest.mark.parametrize("hat", HATS)
def test_hair_and_hat_options(hair, hat):
    m = build_character(VoxCharParams(name="t", hair=hair, hat=hat), head_only=True)
    names = [p.name for p in m.parts]
    assert ("hair" in names) == (hair != "bald")
    assert ("hat" in names) == (hat != "none")
    report = checks.run(m)
    assert report["errors"] == [] and report["warnings"] == [], (hair, hat, report)
    assert ("feather" in names) == (hat == "beret")
    if hat != "none":
        hat_part = m.part("hat")
        assert hat_part.material_histogram() == {MAT_PLANT if hat == "straw" else MAT_CLOTH: hat_part.count_solid()}
        if hair != "bald":
            assert not overlap(hat_part, m.part("hair"))


@pytest.mark.parametrize("beard", BEARDS)
def test_beard_options(beard):
    m = build_character(VoxCharParams(name="t", beard=beard, brows_thick=True), head_only=True)
    names = [p.name for p in m.parts]
    assert ("beard" in names) == (beard in ("short", "full"))
    assert m.part("head").material_histogram()[MAT_HAIR] == 2 * 16 * 5 * 2 + 2 * 3 * 2   # the thick brows
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


def test_mustache_sits_under_the_nose_over_the_mouth_gap():
    m = build_character(VoxCharParams(name="t", beard="short", mustache=True), head_only=True)
    beard = m.part("beard")
    assert at(beard, HW, 21, -3) == MAT_HAIR and at(beard, HW, 17, -3) == MAT_AIR    # the mouth gap stays open
    assert at(beard, 11, 16, -3) == MAT_HAIR and at(beard, HW, 5, -2) == MAT_HAIR     # the ends hang; the chin is covered
    assert at(beard, HW, 40, -3) == MAT_AIR
    assert _mirror_exact(beard)
    report = checks.run(m)
    assert report["errors"] == [] and report["warnings"] == []


def test_barber_spec_builds():
    mod, m = load_spec("barber")
    assert m.name == "barber" and [p.name for p in m.parts] == ["body", "head", "hair", "beard", "hat", "feather"]
    report = checks.run(m)
    assert report["errors"] == [] and report["warnings"] == [], report
    hat, feather, hair = m.part("hat"), m.part("feather"), m.part("hair")
    assert hat.material_histogram() == {MAT_CLOTH: hat.count_solid()}
    assert feather.material_histogram() == {MAT_PLANT: feather.count_solid()}
    assert not hat.symmetric and _mirror_exact(hair) and _mirror_exact(m.part("beard"))
    # the feather touches the beret without entering it; the hair peeks out under the band
    assert not overlap(feather, hat) and touching(feather, hat)
    assert feather.bbox()[1][1] + feather.origin[1] / feather.voxel_size > hat.bbox()[1][1] + hat.origin[1] / hat.voxel_size
    assert set(np.unique(m.part("head").material[exposed(m.part("head").solid())]).tolist()) <= OUTER
    h = build_character(mod.PARAMS, head_only=True)      # the head frame, for coordinates
    assert at(h.part("hair"), HW, 52, -1) == MAT_HAIR and at(h.part("hat"), HW, 57, -6) == MAT_CLOTH
    assert at(h.part("hat"), HW, 70, 25) == MAT_CLOTH and at(h.part("hat"), HW, 52, -6) == MAT_AIR


def test_coat_guy_spec_builds():
    _, m = load_spec("coat_guy")
    assert m.name == "coat_guy" and [p.name for p in m.parts] == ["body", "head", "hair"]
    assert m.meta["outfit"] == "coat"
    pal = m.palette
    body = m.part("body")
    assert (body.color == pal.index("lining")).any() and (body.color == pal.index("trousers_hem")).any()
