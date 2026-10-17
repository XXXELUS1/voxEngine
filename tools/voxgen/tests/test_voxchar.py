"""Designed voxel-art character (voxgen.voxchar): the head builds as coarse cubes whose edges are
exact 1:1 staircases of five fine voxels (45-degree planes for the facet renderer: the front and
back edges, the jaw steps, the chin, the top and the back), with the painted face (white-block
eyes with lid and pupil, proud brows, the nose block with its top sloped, the mouth line with the
tooth row behind it, ear notches) and the hair, the body is a block figure of normal proportions,
both are mirror-symmetric and pass the checks clean, every hair / beard / hat / outfit option
builds, and the coat_guy spec (and the barber's former voxchar params) build as one model with the expected parts."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from voxgen import checks
from voxgen.body import exposed
from voxgen.materials import (MAT_AIR, MAT_BONE, MAT_CLOTH, MAT_EYE, MAT_FAT, MAT_HAIR, MAT_LEATHER, MAT_MUSCLE,
                              MAT_PLANT, MAT_SKIN, MAT_TOOTH)
from voxgen.voxchar import (BACK, BEARDS, BERET_ROWS, HAIR_STYLES, HATS, HW, NOSE, OUTFITS, ROWS, TOP, UP,
                            VoxCharParams, build_character, chamfer_edge)

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


def first_z(part, X: int, Y: int, lo: int = -8, hi: int = 12) -> int | None:
    """The front-most solid z of the column (X, Y) in skull-frame coordinates."""
    return next((z for z in range(lo, hi) if at(part, X, Y, z) != MAT_AIR), None)


def outer_x(part, Y: int, Z: int) -> int | None:
    """The outermost solid x (from the -x side) of the row (Y, Z)."""
    return next((x for x in range(-6, HW) if at(part, x, Y, Z) != MAT_AIR), None)


def back_z(part, X: int, Y: int) -> int | None:
    return next((z for z in range(BACK + 12, 0, -1) if at(part, X, Y, z) != MAT_AIR), None)


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


def test_chamfer_edge_is_an_exact_staircase_of_five():
    d = np.arange(12)
    cut = chamfer_edge(d[None, :], d[:, None])            # distances from face a (columns) and face b (rows)
    assert cut.shape == (12, 12) and int(cut.sum()) == 15  # the triangle 5 + 4 + 3 + 2 + 1
    first = np.argmax(~cut, axis=0)                        # the first kept row per layer of face a
    assert first.tolist()[:7] == [5, 4, 3, 2, 1, 0, 0]    # one voxel per step, five steps, then the flat face
    assert not chamfer_edge(np.array([-1]), np.array([0])).any()
    assert int(chamfer_edge(d[None, :], d[:, None], run=3).sum()) == 6


def test_head_form_is_coarse_cubes_with_45_degree_edges(head_model):
    head = head_model.part("head")
    # the face plane is flat skin; the top face is flat
    assert at(head, HW, 48, 0) == MAT_SKIN and at(head, HW, TOP, 20) == MAT_SKIN
    # the front vertical edge: a 1:1 staircase of five fine steps spanning one coarse cell
    assert [first_z(head, x, 30) for x in range(0, 7)] == [5, 4, 3, 2, 1, 0, 0]
    assert at(head, -1, 30, 0) == MAT_AIR
    # the top-front edge: the same staircase up the forehead
    assert [first_z(head, HW, y) for y in range(49, 55)] == [0, 1, 2, 3, 4, 5]
    # the jaw: one cell narrower per side through a 45-degree plane under the ear line (y 25 -> 20),
    # again at the chin (15 -> 10), and the chin's bottom chamfered on the sides (5 -> 0)
    assert [outer_x(head, y, 25) for y in (25, 24, 22, 20, 17, 15)] == [0, 1, 3, 5, 5, 5]
    assert [outer_x(head, y, 25) for y in (14, 12, 10, 7, 5)] == [6, 8, 10, 10, 10]
    assert [outer_x(head, y, 25) for y in (4, 2, 0)] == [11, 13, 15]
    # the back: two 45-degree cells at the top-back and the bottom-back, straight between
    assert [back_z(head, HW, y) for y in (44, 45, 49, 50, 54)] == [BACK, BACK - 1, BACK - 5, BACK - 6, BACK - 10]
    assert [back_z(head, HW, y) for y in (10, 9, 5, 0)] == [BACK, BACK - 1, BACK - 5, BACK - 10]
    assert at(head, HW, TOP, BACK) == MAT_AIR and at(head, 0, TOP, 20) == MAT_AIR and at(head, 0, 0, 0) == MAT_AIR
    # ears: blocks on the side planes with a two-voxel notch in the outer face
    assert at(head, 46, 28, 21) == MAT_SKIN and at(head, 48, 28, 21) == MAT_SKIN
    assert at(head, 48, 31, 24) == MAT_AIR and at(head, 46, 31, 24) == MAT_SKIN
    # the nose: a block one cell proud whose top is a 45-degree slope from the bridge (y 33 on the
    # face plane) down to the front top edge (y 28), the bottom flat, two dark nostril voxels
    x0, x1, y0, y1, nz = NOSE
    tops = {z: max(y for y in range(y0 - 2, y1 + 2) if at(head, HW, y, z) != MAT_AIR) for z in range(-nz, 0)}
    assert tops == {z: y1 - nz + (z + nz) for z in range(-nz, 0)}
    assert all(min(y for y in range(y0 - 2, y1 + 2) if at(head, HW, y, z) != MAT_AIR) == y0 for z in range(-nz, 0))
    assert at(head, x0, 26, -nz) == MAT_SKIN and at(head, x0 - 1, 26, -nz) == MAT_AIR and at(head, x1 + 1, 26, -nz) == MAT_AIR
    pal = head_model.palette
    nostril = head.color == pal.index("line")
    assert int(nostril.sum()) == 2 and at(head, HW + 2, y0, -nz) == MAT_SKIN
    assert head_model.meta["build"]["head"]["nostril"] == 2
    # brows: hair material two voxels proud of the face, three rows a row above the lids, the
    # inner end a row lower
    assert at(head, 30, 43, -1) == MAT_HAIR and at(head, 30, 43, -2) == MAT_HAIR and at(head, 30, 43, -3) == MAT_AIR
    assert at(head, 30, 42, -1) == MAT_HAIR and at(head, 30, 45, -1) == MAT_AIR and at(head, 30, 41, -1) == MAT_AIR
    assert at(head, 26, 41, -1) == MAT_HAIR and at(head, 34, 41, -1) == MAT_AIR and at(head, 40, 43, -1) == MAT_AIR
    # the mouth line with its corners a voxel up (colour only)
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
    assert m.part("head").material_histogram()[MAT_HAIR] == 2 * 15 * 5 * 2 + 2 * 3 * 2   # the thick brows
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
    assert at(beard, HW, 21, -2) == MAT_HAIR and at(beard, HW, 21, -3) == MAT_AIR     # two voxels thick
    assert at(beard, HW, 17, -2) == MAT_AIR                                           # the mouth gap stays open
    assert at(beard, 11, 16, -2) == MAT_HAIR and at(beard, HW, 5, -2) == MAT_HAIR     # the ends hang; the chin is covered
    assert at(beard, HW, 40, -2) == MAT_AIR
    # the beard follows the jaw's chamfers: three voxels at the chin, two on the jaw's side plane
    assert [at(beard, HW, 5, z) for z in (-1, -2, -3, -4)] == [MAT_HAIR] * 3 + [MAT_AIR]
    assert [at(beard, x, 22, 10) for x in (3, 2, 1, 0)] == [MAT_AIR, MAT_HAIR, MAT_HAIR, MAT_AIR]
    assert at(beard, 4, 18, 10) == MAT_HAIR and at(beard, 4, 18, 24) == MAT_AIR      # the top edge falls toward the nape
    assert at(beard, 9, 8, 24) == MAT_HAIR and at(beard, 9, 8, 31) == MAT_AIR
    assert _mirror_exact(beard)
    report = checks.run(m)
    assert report["errors"] == [] and report["warnings"] == []


def test_voxchar_barber_builds():
    # the barber spec itself now builds with voxgen.blockhead; these are its former voxchar params
    params = VoxCharParams(name="barber", skin_tone="warm", hair_tone="chestnut", hair="chunky", beard="short",
                           mustache=True, brows_thick=True, hat="beret", outfit="apron", sleeves="rolled")
    m = build_character(params)
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
    h = build_character(params, head_only=True)         # the head frame, for coordinates
    hat, hair = h.part("hat"), h.part("hair")
    assert at(hair, HW, 51, -1) == MAT_HAIR and at(hat, HW, 52, -6) == MAT_AIR
    # the beret: the band one cell outside the skull, the flare 1:1 to two cells, the top closing
    # over three cells on the right and the front, two on the left and the back (the slouch)
    y0, y1 = BERET_ROWS
    assert at(hat, HW, y0 + 2, -UP) == MAT_CLOTH and at(hat, HW, y0 + 2, -UP - 1) == MAT_AIR
    assert [first_z(hat, HW, y, lo=-14) for y in range(y0 + UP, y0 + 2 * UP + 1)] == [-5, -6, -7, -8, -9, -10]
    assert at(hat, HW, 70, 25) == MAT_CLOTH
    assert at(hat, 39, y1, 25) == MAT_CLOTH and at(hat, 40, y1, 25) == MAT_AIR       # right: 10 -> -5
    assert at(hat, 0, y1, 25) == MAT_CLOTH and at(hat, -1, y1, 25) == MAT_AIR        # left: 10 -> 0
    assert [first_z(hat, HW, y, lo=-14) for y in (y0 + 3 * UP, y0 + 4 * UP, y1)] == [-9, -4, 5]   # front closing: 1:1 from the row after y0 + 3 cells
    assert at(hat, 20, y1 + 2, 25) == MAT_AIR                                        # crumples are one voxel
    # chunky hair: an even shell of three with tufts two voxels prouder at the nape and the temples
    assert at(hair, HW, 20, BACK + 3) == MAT_HAIR and at(hair, HW, 20, BACK + 4) == MAT_AIR
    assert at(hair, HW, 22, BACK + 5) == MAT_HAIR and at(hair, HW, 26, BACK + 5) == MAT_AIR
    assert at(hair, -5, 30, 8) == MAT_HAIR and at(hair, -5, 28, 8) == MAT_AIR and at(hair, -4, 27, 8) == MAT_AIR
    assert (hair.color == h.palette.index("hair_light")).any()


def test_coat_guy_spec_builds():
    _, m = load_spec("coat_guy")
    assert m.name == "coat_guy" and [p.name for p in m.parts] == ["body", "head", "hair"]
    assert m.meta["outfit"] == "coat"
    pal = m.palette
    body = m.part("body")
    assert (body.color == pal.index("lining")).any() and (body.color == pal.index("trousers_hem")).any()
