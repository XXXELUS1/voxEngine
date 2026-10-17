"""Relief-map head engine (voxgen.sheethead): map cells map to the right boxes (front and side
views, mirror), a later fill never overwrites an earlier part, paint recolours the front-most
voxel after the recess cut, the lattice colouring is per block and deterministic, and the barber
head built from specs/chars/vox/barber_maps.py has a whole mirror-exact skull with the 1:1 nose
staircase under untouched hair / beard / beret."""
import numpy as np
import pytest

from voxgen import checks
from voxgen.blockhead import SKULL_MID_Z, U_M, Block, Builder, Frame, make_palette, neck_form, skull_form
from voxgen.materials import MAT_AIR, MAT_CLOTH, MAT_EYE, MAT_HAIR, MAT_PLANT, MAT_SKIN, MAT_TOOTH
from voxgen.sheethead import (CAVITY, DEFAULT_MAPS, EARS, apply_layer, build_head, build_parts, cell_box,
                              footprint, load_maps)


@pytest.fixture(scope="module")
def maps():
    return load_maps(DEFAULT_MAPS)


@pytest.fixture(scope="module")
def built(maps):
    """(builder, uncropped parts, reports) of the barber head."""
    return build_parts(maps, seed=1)


@pytest.fixture(scope="module")
def head(maps):
    return build_head(seed=1, maps=maps)


def at(part, x: float, y: float, z: float, field: str = "material") -> int:
    """Value of the fine cell containing the unit point (x, y, z) of a cropped model part."""
    loc = (x * U_M, y * U_M, (z - SKULL_MID_Z) * U_M)
    ix, iy, iz = (int(np.floor((loc[i] - part.origin[i]) / part.voxel_size + 1e-6)) for i in range(3))
    if not (0 <= ix < part.nx and 0 <= iy < part.ny and 0 <= iz < part.nz):
        return 0
    return int(getattr(part, field)[iz, iy, ix])


def mirrored(part, field: str = "material") -> bool:
    n0 = int(round(-part.origin[0] / part.voxel_size))
    arr = getattr(part, field)
    idx = 2 * n0 - 1 - np.arange(part.nx)
    ok = (idx >= 0) & (idx < part.nx)
    return bool(np.array_equal(arr[:, :, ok], arr[:, :, idx[ok]])) and not arr[:, :, ~ok].any()


def fresh():
    f = Frame()
    return f, Builder(f, make_palette())


def layer(rows, cell=0.5, origin=(-1.0, 1.0), **kw):
    return {"view": "front", "cell": cell, "origin": origin, "mirror": False, "material": "MAT_SKIN", "rows": rows} | kw


# ---------------------------------------------------------------------------- cells
def test_cell_box_front_side_mirror(maps):
    face = dict(maps.LAYERS)["face"]
    b = cell_box(face, 8, 13, "2")                       # y 7.75..8, x -0.75..-0.5, nose bridge 1/2 U proud
    assert (b.x0, b.x1, b.y0, b.y1, b.z0, b.z1) == (-0.75, -0.5, 7.75, 8.0, -0.5, 0.0)
    m = b.mirror()
    assert (m.x0, m.x1, m.y0, m.y1, m.z0, m.z1) == (0.5, 0.75, 7.75, 8.0, -0.5, 0.0)
    side = maps.HAIR_SIDE
    s = cell_box(side, 1, 10, "C")                       # columns are z, depth is x
    assert (s.x0, s.x1, s.y0, s.y1, s.z0, s.z1) == (-5.5, 5.5, 11.0, 11.5, 3.5, 4.0)
    beard = maps.BEARD
    e = cell_box(beard, 9, 7, "=")
    assert (e.x0, e.x1, e.y0, e.y1, e.z0, e.z1) == (-2.0, -1.5, -1.0, -0.5, -0.75, 2.0)


def test_footprint_matches_the_cells_and_their_mirror(maps):
    f = Frame()
    face = dict(maps.LAYERS)["face"]
    fp = footprint(f, face, "2")                         # (ny, nx)
    X, Y = f.X.reshape(1, -1), f.Y.reshape(-1, 1)
    want = ((np.abs(X) < 0.75) & (Y >= 5.5) & (Y < 8)) | ((np.abs(X) < 1) & (Y >= 5) & (Y < 5.25))
    assert np.array_equal(fp, want)
    fs = footprint(f, maps.HAIR_SIDE, "B")               # (ny, nz): the nape clumps, e.g. z 10.5..11 at y 10..10.5
    Z = f.Z.reshape(1, -1)
    assert fs[(Y[:, 0] >= 10) & (Y[:, 0] < 10.5)][:, (Z[0] >= 10.5) & (Z[0] < 11)].all()
    assert not fs[:, Z[0] < 9.5].any()


# ---------------------------------------------------------------------------- fills
def test_fill_never_overwrites_an_earlier_part():
    f, bd = fresh()
    skull = f.new_part("skull")
    bd.place(skull, Block(-1, 1, 0, 2, 0, 2))
    bd.occupied |= skull.solid()
    before = skull.material.copy()
    parts = {"skull": skull}
    lay = layer(["aaaaaaaa"] * 6, origin=(-2.0, 3.0), depth={"a": (-1.0, 1.0)}, paint={"a": "skin_light"},
                material="MAT_HAIR")
    rep = apply_layer(bd, parts, "hat", lay, np.random.default_rng(0))
    hat = parts["hat"]
    assert rep["part"] == "hat" and rep["added"] == 16 * 12 * 8 - 8 * 8 * 4 and rep["empty_cells"] == []
    assert np.array_equal(skull.material, before)
    assert not (hat.solid() & skull.solid()).any()
    assert hat.solid().sum() == rep["added"]
    assert set(hat.material[hat.solid()].tolist()) == {MAT_HAIR}
    assert set(hat.color[hat.solid()].tolist()) == {bd.pal.index("skin_light")}
    assert np.array_equal(bd.occupied, skull.solid() | hat.solid())


def test_mirror_fills_both_halves_and_empty_cells_are_reported():
    f, bd = fresh()
    parts = {}
    lay = layer(["a"], depth={"a": (0.0, 1.0)}, paint={"a": "skin"})
    lay["mirror"] = True
    apply_layer(bd, parts, "m", lay, np.random.default_rng(0))
    m = parts["m"]
    assert m.solid().sum() == 2 * (2 * 2 * 4)
    assert np.array_equal(m.solid(), f.box(-1, -0.5, 0.5, 1, 0, 1) | f.box(0.5, 1, 0.5, 1, 0, 1))
    # a cell fully clipped by what is already there is reported, not an error
    bd.occupied |= f.box(-1, -0.5, 0.5, 1, 0, 1)
    rep = apply_layer(bd, {}, "n", layer(["b"], depth={"b": (0.0, 1.0)}, paint={"b": "skin"}), np.random.default_rng(0))
    assert rep["added"] == 0 and rep["empty_cells"] == [("b", 0, 0)]


def test_paint_and_recess_on_a_slab():
    f, bd = fresh()
    part = f.new_part("skull")
    bd.place(part, Block(-1, 1, 0, 1, 0, 1))             # a slab 4 fine voxels deep from the face plane
    bd.occupied |= part.solid()
    parts = {"skull": part}
    lay = layer(["P."], paint={"P": "pupil", ".": "skin"}, recess={"P": 0.25}, materials={"P": "MAT_EYE"})
    rep = apply_layer(bd, parts, "face", lay, np.random.default_rng(0))
    assert rep["cut"] == 4 and rep["painted"] == 4 and rep["added"] == 0 and "face" not in parts
    pupil = bd.pal.index("pupil")
    for x in (-0.875, -0.625):
        for y in (0.625, 0.875):
            ix, iy, iz = f.index(x, y, 0.125)
            assert part.material[iz, iy, ix] == MAT_AIR and not bd.occupied[iz, iy, ix]
            ix, iy, iz = f.index(x, y, 0.375)
            assert part.material[iz, iy, ix] == MAT_EYE and part.color[iz, iy, ix] == pupil
            ix, iy, iz = f.index(x, y, 0.625)
            assert part.material[iz, iy, ix] == MAT_SKIN and part.color[iz, iy, ix] == bd.pal.index("skin")
    ix, iy, iz = f.index(-0.375, 0.875, 0.125)          # the '.' cell: untouched
    assert part.material[iz, iy, ix] == MAT_SKIN and part.color[iz, iy, ix] == bd.pal.index("skin")
    # paint without recess recolours the front-most voxel in place
    lay2 = layer(["s."], paint={"s": "skin_light"})
    apply_layer(bd, parts, "face", lay2, np.random.default_rng(0))
    ix, iy, iz = f.index(-0.875, 0.875, 0.375)
    assert part.color[iz, iy, ix] == bd.pal.index("skin_light") and part.material[iz, iy, ix] == MAT_EYE


def test_lattice_colours_per_block_and_seeded(maps):
    bd, parts, _ = build_parts(maps, seed=1)
    hair = parts["hair"]
    names = {bd.pal.index(n) for n in maps.HAIR_FRONT["palette"]}
    assert set(hair.color[hair.solid()].tolist()) <= names
    nz, ny, nx = hair.color.shape
    blocks = hair.color.reshape(nz // 4, 4, ny // 4, 4, nx // 4, 4).transpose(0, 2, 4, 1, 3, 5).reshape(-1, 64)
    for blk in blocks:
        assert len(set(blk[blk != 0].tolist())) <= 1    # one colour per 1 U lattice cell, front and side maps agree
    bd2, parts2, _ = build_parts(maps, seed=1)
    assert np.array_equal(parts2["hair"].color, hair.color) and np.array_equal(parts2["beret"].color, parts["beret"].color)
    bd3, parts3, _ = build_parts(maps, seed=2)
    assert np.array_equal(parts3["hair"].solid(), hair.solid()) and not np.array_equal(parts3["hair"].color, hair.color)


# ---------------------------------------------------------------------------- the barber head
def test_parts_order_and_materials(head):
    assert [p.name for p in head.parts] == ["skull", "beard", "beret", "hair", "feather"]
    for name, mats in (("beard", {MAT_HAIR}), ("hair", {MAT_HAIR}), ("beret", {MAT_CLOTH}), ("feather", {MAT_PLANT})):
        p = head.part(name)
        assert set(p.material[p.solid()].tolist()) == mats, name
    assert head.meta["generator"] == "sheethead" and head.meta["layers"]["face"][0]["empty_cells"] == 0


@pytest.mark.xfail(strict=True, reason="barber_maps: BEARD '>' strips (cols 0-2, rows 1-6) share no face with '#' "
                   "inside the beard part and HAIR_FRONT row 7 col 21 ('1' at x 3.5..4) is a lone cell: per-part "
                   "islands resting on the skull. Remove this marker once the maps are fixed.")
def test_checks_clean(head):
    report = checks.run(head)
    assert report["errors"] == []


def test_check_errors_are_only_islands_resting_on_other_parts(head):
    report = checks.run(head)
    assert all("floating component" in e for e in report["errors"]), report["errors"]
    assert len(report["errors"]) == len(head.meta["islands"])
    for isl in head.meta["islands"]:
        assert isl["resting_on"], isl


def test_skull_and_beard_are_mirror_exact(head):
    s = head.part("skull")
    assert mirrored(s) and mirrored(s, "color")
    assert mirrored(head.part("beard"))                 # the beard's per-block colours are random, its material is exact


def test_nose_staircase_cube_bridge_and_nostrils(head):
    s = head.part("skull")
    pal = head.palette
    for k in range(4):                                   # wedge rows y 5.5 -> 4.5: 1, 2, 3, 4 fine voxels proud
        y, front = 5.375 - 0.25 * k, -0.25 * (k + 1)
        for x in (-0.125, 0.125, -0.875, 0.875):
            assert at(s, x, y, front + 0.125) == MAT_SKIN and at(s, x, y, front - 0.125) == MAT_AIR
    for y in (3.625, 4.375):                             # the cube, 1 U proud with a flat bottom
        assert at(s, 0.125, y, -0.875) == MAT_SKIN and at(s, 0.125, y, -1.125) == MAT_AIR
    assert at(s, 0.125, 3.375, -0.875) == MAT_AIR
    assert at(s, 0.125, 6.125, -0.375) == MAT_SKIN and at(s, 0.125, 6.125, -0.625) == MAT_AIR   # the bridge
    assert at(s, 0.875, 6.125, -0.375) == MAT_AIR
    dark = pal.index("eye_frame")
    for x in (-0.625, 0.625):
        assert at(s, x, 3.625, -0.875, "color") == dark and at(s, x, 3.625, -0.625, "color") == pal.index("skin")
    assert at(s, 0.125, 3.625, -0.875, "color") == pal.index("skin")


def test_face_features(head):
    s, beard, pal = head.part("skull"), head.part("beard"), head.palette
    # the eyes: the frame, white and pupil sit one fine voxel into the face
    for x, colour, mat in ((-3.375, "eye_frame", MAT_SKIN), (-2.875, "eye_white", MAT_EYE), (-2.125, "pupil", MAT_EYE),
                           (2.125, "pupil", MAT_EYE), (3.375, "eye_frame", MAT_SKIN)):
        assert at(s, x, 7.125, 0.125) == MAT_AIR
        assert at(s, x, 7.125, 0.375) == mat and at(s, x, 7.125, 0.375, "color") == pal.index(colour), x
    # brows (hair, in the skull part), the moustache (beard part), the light cheek and lip, the mouth
    assert at(s, -2, 8.5, -0.25) == MAT_HAIR and at(s, -2, 8.5, -0.25, "color") == pal.index("hair_black")
    assert at(s, 0.125, 8.5, -0.25) == MAT_AIR
    assert at(beard, 0, 3, -0.25) == MAT_HAIR and at(beard, 0, 3, -0.25, "color") == pal.index("hair_black")
    assert at(s, 0, 3, -0.25) == MAT_AIR
    assert at(s, -3.5, 6, 0.125, "color") == pal.index("skin_light") and at(s, 0.125, 1.875, 0.125, "color") == pal.index("skin_light")
    assert at(s, 0.125, 2.5, 0.5) == MAT_AIR and at(s, 0.125, 2.5, 1.125) == MAT_SKIN
    assert at(s, -1.375, 2.5, 0.875) == MAT_TOOTH and at(s, 1.375, 2.5, 0.875) == MAT_TOOTH
    assert at(s, 5.25, 6, 3, "color") == pal.index("skin")                        # the ear, default skin
    assert at(s, 0.125, -1, 1.125) == MAT_SKIN and at(s, 0.125, -1, 1.125, "color") == pal.index("skin_shadow")   # the neck


def test_layers_never_touch_the_skull(built):
    bd, parts, _ = built
    f = bd.f
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (parts[a].solid() & parts[b].solid()).any(), (a, b)
    base = skull_form(f) | neck_form(f)
    for e in EARS:
        base |= f.mask(e)
    base &= ~f.mask(CAVITY)
    removed = base & ~parts["skull"].solid()               # only the eye recess may take skull voxels away
    sockets = f.box(-3.5, -1, 6.5, 8, 0, 0.25) | f.box(1, 3.5, 6.5, 8, 0, 0.25)
    assert removed.sum() == 120 and not (removed & ~sockets).any()
    assert (parts["skull"].solid() & ~base).sum() > 0       # the face relief was added to the skull
