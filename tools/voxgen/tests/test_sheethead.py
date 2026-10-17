"""Relief-map head engine (voxgen.sheethead): map cells map to the right boxes (front, side and
profile views, mirror), a profile cell fills |x| < half-width, a shell cell hugs what is already
placed, a later fill never overwrites an earlier part, paint recolours the front-most voxel after
the 2-fine recess cut (re-skinning what the cut exposed), the lattice colouring is per block and
deterministic, the frame is computed from the maps, and the barber head built from
specs/chars/vox/barber_maps.py has a whole mirror-exact skull drawn by its profile map (nose tip
at z -2, 1/2 U proud chin, notched ears, open mouth with teeth, dome, occiput, neck) under
untouched hair / beard / beret; the bald spec is the skull and face alone."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from voxgen import checks
from voxgen.blockhead import U_M, Block, Builder, Frame, make_palette
from voxgen.materials import (MAT_AIR, MAT_CLOTH, MAT_EYE, MAT_FAT, MAT_HAIR, MAT_MUSCLE, MAT_PLANT, MAT_CARTILAGE, MAT_SKIN,
                              MAT_TOOTH)
from voxgen.sheethead import (CAVITY_X, CAVITY_Y, DEFAULT_MAPS, EAR_NOTCHES, EARS, LIP_THICK, SKULL_MID_Z, TEETH_X,
                              apply_layer, build_head, build_parts, cell_box, dilate, fills, footprint,
                              frame_extent, load_maps, make_frame)

BALD_SPEC = DEFAULT_MAPS.with_name("bald.py")


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


@pytest.fixture(scope="module")
def bald():
    spec = importlib.util.spec_from_file_location("bald_spec", BALD_SPEC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build()


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


def exposed_tissue(part) -> int:
    """Fat / muscle voxels of a part with an air 6-neighbour (the skin shell must hide them all)."""
    solid = part.solid()
    air_next = np.zeros_like(solid)
    for ax in (0, 1, 2):
        for d in (1, -1):
            air_next |= np.roll(~solid, d, axis=ax)
    return int((np.isin(part.material, (MAT_FAT, MAT_MUSCLE)) & air_next).sum())


def fresh():
    f = Frame()
    return f, Builder(f, make_palette())


def layer(rows, cell=0.5, origin=(-1.0, 1.0), **kw):
    return {"view": "front", "cell": cell, "origin": origin, "mirror": False, "material": "MAT_SKIN", "rows": rows} | kw


# ---------------------------------------------------------------------------- cells
def test_cell_box_front_side_profile_mirror(maps):
    face = maps.FACE
    b = cell_box(face, 4, 2, "b")                        # y 8.75..9, x -3.5..-3.25, brow 1/2 U proud
    assert (b.x0, b.x1, b.y0, b.y1, b.z0, b.z1) == (-3.5, -3.25, 8.75, 9.0, -0.5, 0.0)
    m = b.mirror()
    assert (m.x0, m.x1, m.y0, m.y1, m.z0, m.z1) == (3.25, 3.5, 8.75, 9.0, -0.5, 0.0)
    s = cell_box(maps.HAIR_SIDE, 1, 10, "C")             # side map: columns are z, depth is x
    assert (s.x0, s.x1, s.y0, s.y1, s.z0, s.z1) == (-5.5, 5.5, 11.0, 11.5, 3.5, 4.0)
    p = cell_box(maps.SKULL, 1, 14, "2")                 # profile map: the dome's second row, half-width 2
    assert (p.x0, p.x1, p.y0, p.y1, p.z0, p.z1) == (-2.0, 2.0, 13.0, 13.5, 4.5, 5.0)
    n = cell_box(maps.SKULL, 17, 1, "1")                 # the nose tip cell: y 5..5.5, z -2..-1.5, |x| < 1
    assert (n.x0, n.x1, n.y0, n.y1, n.z0, n.z1) == (-1.0, 1.0, 5.0, 5.5, -2.0, -1.5)
    assert fills(maps.SKULL)["D"] == (-4.5, 4.5) and fills(maps.BEARD) == {}
    with pytest.raises(ValueError):
        fills({"view": "profile", "depth": {"a": (0, 1)}})
    with pytest.raises(ValueError):
        fills({"view": "front", "width": {"a": 1}})


def test_footprint_matches_the_cells_and_their_mirror(maps):
    f = Frame()
    fp = footprint(f, maps.FACE, "b")                    # (ny, nx): the brows, both halves
    X, Y = f.X.reshape(1, -1), f.Y.reshape(-1, 1)
    want = (np.abs(X) >= 0.5) & (np.abs(X) < 3.5) & (Y >= 8) & (Y < 9)
    assert np.array_equal(fp, want)
    Z = f.Z.reshape(1, -1)
    fs = footprint(f, maps.HAIR_SIDE, "B")               # (ny, nz): the nape clumps, e.g. z 15..15.5 at y 10..10.5
    assert fs[(Y[:, 0] >= 10) & (Y[:, 0] < 10.5)][:, (Z[0] >= 15) & (Z[0] < 15.5)].all()
    assert not fs[:, Z[0] < 14].any()
    fn = footprint(f, maps.SKULL, "1")                   # (ny, nz): the nose cube z -2..0 at y 4.5..6 (and the crown)
    assert fn[(Y[:, 0] >= 4.5) & (Y[:, 0] < 6)][:, (Z[0] >= -2) & (Z[0] < 0)].all()
    assert not fn[:, Z[0] < -2].any() and not fn[(Y[:, 0] >= 8) & (Y[:, 0] < 13)].any()


# ---------------------------------------------------------------------------- fills
def test_profile_fill_is_the_half_width_box():
    f, bd = fresh()
    parts = {}
    lay = {"view": "profile", "cell": 0.5, "origin": (0.0, 1.0), "material": "MAT_SKIN", "colour": "skin_light",
           "width": {"2": 2.0, "A": 1.5}, "rows": ["2A"]}
    rep = apply_layer(bd, parts, "skull", lay, np.random.default_rng(0))
    p = parts["skull"]
    want = f.box(-2, 2, 0.5, 1, 0, 0.5) | f.box(-1.5, 1.5, 0.5, 1, 0.5, 1)
    assert np.array_equal(p.solid(), want) and rep["added"] == 112 and rep["empty_cells"] == []
    assert set(p.material[want].tolist()) == {MAT_SKIN} and set(p.color[want].tolist()) == {bd.pal.index("skin_light")}
    assert np.array_equal(bd.occupied, want)


def test_dilate_is_the_l1_ball_without_wrap():
    m = np.zeros((7, 7, 7), bool)
    m[3, 3, 3] = True
    assert dilate(m, 2).sum() == 25 and dilate(m, 1).sum() == 7 and dilate(m, 0).sum() == 1
    edge = np.zeros((3, 3, 3), bool)
    edge[0, 1, 1] = True
    assert dilate(edge, 1).sum() == 6 and not dilate(edge, 1)[2].any()


def test_shell_hugs_what_is_already_placed():
    f, bd = fresh()
    skull = f.new_part("skull")
    bd.place(skull, Block(-1, 1, 0, 2, 0, 2))
    bd.occupied |= skull.solid()
    parts = {"skull": skull}
    lay = layer(["s"] * 2, cell=1.0, origin=(-2.0, 2.0), shell={"s": 0.5}, paint={"s": "skin_light"},
                material="MAT_HAIR")
    rep = apply_layer(bd, parts, "beard", lay, np.random.default_rng(0))
    beard = parts["beard"]
    foot = f.box(-2, -1, 0, 2, -10, 20)                  # the one 1 U column of the map
    want = dilate(skull.solid(), 2) & ~skull.solid() & foot
    assert np.array_equal(beard.solid(), want) and rep["added"] == int(want.sum()) > 0
    assert at_unit(f, beard, -1.125, 1, 1) == MAT_HAIR and at_unit(f, beard, -1.375, 1, 1) == MAT_HAIR
    assert at_unit(f, beard, -1.625, 1, 1) == MAT_AIR                          # 0.5 U = 2 fine thick
    assert at_unit(f, beard, -1.375, 1, -0.125) == MAT_AIR and at_unit(f, beard, -1.125, 1, -0.125) == MAT_HAIR  # 45 deg corner
    assert at_unit(f, beard, -0.5, 1, -0.125) == MAT_AIR                       # outside the footprint
    assert not (beard.solid() & skull.solid()).any() and rep["empty_cells"] == []
    with pytest.raises(ValueError):
        apply_layer(bd, {}, "x", layer(["s"], shell={"s": 0.5}, depth={"s": (0, 1)}, paint={"s": "skin"}),
                    np.random.default_rng(0))


def at_unit(f: Frame, part, x: float, y: float, z: float) -> int:
    ix, iy, iz = f.index(x, y, z)
    return int(part.material[iz, iy, ix])


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
    fat = f.box(-1, 1, 0, 1, 0.5, 1)                     # skin 2 fine, fat behind
    part.material[fat] = MAT_FAT
    parts = {"skull": part}
    lay = layer(["P_"], paint={"P": "pupil", "_": "skin_light"}, recess={"P": 0.5, "_": 0.5}, materials={"P": "MAT_EYE"})
    rep = apply_layer(bd, parts, "face", lay, np.random.default_rng(0))
    assert rep["cut"] == 16 and rep["painted"] == 8 and rep["added"] == 0 and "face" not in parts
    pupil = bd.pal.index("pupil")
    for x in (-0.875, -0.625):
        for y in (0.625, 0.875):
            for z in (0.125, 0.375):                     # 2 fine cut away
                ix, iy, iz = f.index(x, y, z)
                assert part.material[iz, iy, ix] == MAT_AIR and not bd.occupied[iz, iy, ix]
            ix, iy, iz = f.index(x, y, 0.625)            # the new front-most voxel: eye, pupil colour
            assert part.material[iz, iy, ix] == MAT_EYE and part.color[iz, iy, ix] == pupil
            ix, iy, iz = f.index(x, y, 0.875)
            assert part.material[iz, iy, ix] == MAT_FAT
    for x in (-0.375, -0.125):                           # '_' without a material: the old surface's material (skin)
        ix, iy, iz = f.index(x, 0.875, 0.625)
        assert part.material[iz, iy, ix] == MAT_SKIN and part.color[iz, iy, ix] == bd.pal.index("skin_light")
        ix, iy, iz = f.index(x, 0.875, 0.375)
        assert part.material[iz, iy, ix] == MAT_AIR
    # paint without recess recolours the front-most voxel in place and keeps its material
    lay2 = layer(["s."], paint={"s": "skin_light"})
    apply_layer(bd, parts, "face", lay2, np.random.default_rng(0))
    ix, iy, iz = f.index(-0.875, 0.875, 0.625)
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


# ---------------------------------------------------------------------------- the frame
def test_frame_covers_the_maps(maps):
    (x0, x1), (y0, y1), (z0, z1) = frame_extent(maps.LAYERS)
    assert x0 <= -9.5 and x1 >= 9.5 and y0 <= -3 and y1 >= 23 and z0 <= -3 and z1 >= 17.5
    assert x0 == -x1 and all(v == int(v) for v in (x0, x1, y0, y1, z0, z1))
    f = make_frame(maps.LAYERS)
    assert f.origin_m == (x0 * U_M, y0 * U_M, (z0 - SKULL_MID_Z) * U_M) and SKULL_MID_Z == 6.5
    (bx0, bx1), _, (bz0, bz1) = frame_extent([("skull", maps.SKULL)])
    assert bx1 == -bx0 >= 6.5 and bz0 <= -2.5 and bz1 >= 13.5 and bx1 < x1
    with pytest.raises(ValueError):
        make_frame([("face", maps.FACE)])


# ---------------------------------------------------------------------------- the barber head
def test_parts_order_and_materials(head):
    assert [p.name for p in head.parts] == ["skull", "beard", "beret", "hair", "feather"]
    for name, mats in (("beard", {MAT_HAIR}), ("hair", {MAT_HAIR}), ("beret", {MAT_CLOTH}), ("feather", {MAT_PLANT})):
        p = head.part(name)
        assert set(p.material[p.solid()].tolist()) == mats, name
    assert head.meta["generator"] == "sheethead" and head.meta["layers"]["face"][0]["empty_cells"] == 0
    assert head.meta["mid_z"] == SKULL_MID_Z and head.meta["layers"]["face"][0]["cut"] == 2 * 10 * 6 * 2 + 8 * 2 * 2


def test_checks_clean(head):
    """Hair/beard islands resting on the skull count as attached (checks.other_parts_anchor)."""
    report = checks.run(head)
    assert report["errors"] == []
    for isl in head.meta["islands"]:
        assert isl["resting_on"], isl
    assert [w for w in report["warnings"] if "enclosed air pocket" not in w] == []     # the sealed mouth, by design
    assert len(report["warnings"]) == 1 and exposed_tissue(head.part("skull")) == 0


def test_skull_and_beard_are_mirror_exact(head):
    s = head.part("skull")
    assert mirrored(s) and mirrored(s, "color")
    assert mirrored(head.part("beard"))                 # the beard's per-block colours are random, its material is exact


def test_skull_profile_from_the_map(head):
    s = head.part("skull")
    for y in (4.625, 5.375):                             # the nose cube: tip at z -2, |x| < 1
        for x in (-0.125, 0.125, 0.875):
            assert at(s, x, y, -1.875) == MAT_SKIN and at(s, x, y, -2.125) == MAT_AIR
        assert at(s, 1.125, y, -1.875) == MAT_AIR
    for y, front in ((6.25, -1.5), (6.75, -1.0), (7.5, -0.5)):   # the 45-degree wedge in 1/2 U steps
        assert at(s, 0.125, y, front + 0.125) == MAT_SKIN and at(s, 0.125, y, front - 0.125) == MAT_AIR
    assert at(s, 0.125, 8.25, 0.125) == MAT_SKIN and at(s, 0.125, 8.25, -0.125) == MAT_AIR   # the bridge ends at y 8
    assert at(s, 0.125, 1.75, -0.375) == MAT_SKIN and at(s, 0.125, 1.75, -0.625) == MAT_AIR  # chin 1/2 U proud
    assert at(s, 1.875, 1.75, -0.375) == MAT_SKIN and at(s, 2.125, 1.75, -0.375) == MAT_AIR  # ... and 4 wide
    assert at(s, 3.875, 5, 0.125) == MAT_SKIN and at(s, 4.125, 5, 0.125) == MAT_AIR         # cheek plane 8 wide
    assert at(s, 4.125, 5, 0.625) == MAT_SKIN                                                # 9 wide from z 0.5
    assert at(s, 4.875, 9, 6) == MAT_SKIN and at(s, 5.125, 9, 6) == MAT_AIR                 # sides 10 wide
    for y, z in ((11.25, 1.5), (12.25, 2.5), (13.25, 4.5)):                                  # the curved dome front (v3)
        assert at(s, 0.125, y, z + 0.125) == MAT_SKIN and at(s, 0.125, y, z - 0.125) == MAT_AIR
    assert at(s, 1.875, 13.25, 6) == MAT_SKIN and at(s, 2.125, 13.25, 6) == MAT_AIR         # the dome: y 13..13.5 |x| < 2
    assert at(s, 0.875, 13.75, 6.125) == MAT_SKIN and at(s, 1.125, 13.75, 6.125) == MAT_AIR  # ... y 13.5..14 |x| < 1
    assert at(s, 0.125, 14.25, 6) == MAT_AIR
    assert at(s, 0.125, 8.75, 12.875) == MAT_SKIN and at(s, 0.125, 8.75, 13.125) == MAT_AIR  # the back at z 13
    assert at(s, 0.125, 2.25, 11.375) == MAT_SKIN and at(s, 0.125, 2.25, 11.625) == MAT_AIR  # the occiput curves in (z 11.5 at y 2..2.5)
    assert at(s, 0.125, -1, 5.625) == MAT_SKIN and at(s, 0.125, -1, 5.625, "color") == head.palette.index("skin_shadow")
    assert at(s, 0.125, -1, 5.375) == MAT_AIR and at(s, 0.125, -1, 11.625) == MAT_AIR        # neck z 5.5..11.5
    assert at(s, 0.125, -1, 11.375) == MAT_SKIN and at(s, 0.125, -1, 8) != MAT_AIR     # back face, solid inside
    assert at(s, 2.375, -1, 8) == MAT_SKIN and at(s, 2.625, -1, 8) == MAT_AIR                # ... 5 wide


def test_ears_mouth_and_nostrils(head):
    s, pal = head.part("skull"), head.palette
    skin, shadow = pal.index("skin"), pal.index("skin_shadow")
    for sx in (-1, 1):                                   # ears |x| 5..6, z 7..10, y 4..7, with the notch
        assert at(s, sx * 5.75, 4.125, 7.5) == MAT_CARTILAGE and at(s, sx * 5.75, 4.125, 7.5, "color") == skin   # bottom (ears are cartilage: cubic in the smooth renders)
        assert at(s, sx * 5.875, 4.5, 7.5) == MAT_CARTILAGE and at(s, sx * 5.875, 4.5, 7.5, "color") == skin   # outer face
        assert at(s, sx * 5.25, 5.5, 7.125) != MAT_AIR and at(s, sx * 6.125, 4.5, 7.5) == MAT_AIR
        assert at(s, sx * 5.75, 5.5, 8.5) == MAT_AIR                                          # the notch void
        assert at(s, sx * 5.375, 5.5, 8.5) == MAT_CARTILAGE and at(s, sx * 5.375, 5.5, 8.5, "color") == shadow   # floor
        assert at(s, sx * 5.75, 5.5, 7.875) == MAT_SKIN and at(s, sx * 5.75, 5.5, 7.875, "color") == shadow  # wall
        assert at(s, sx * 5.75, 6.125, 8.5, "color") == shadow and at(s, sx * 5.75, 6.375, 8.5, "color") == skin
    # the mouth: 5 U wide, 1 U deep from the face plane behind closed lips (1 fine of skin, the mouth
    # line on the slit row y 3.5..3.75 only), teeth on the back wall
    for x in (0.125, -2.375, 2.375):
        assert at(s, x, 3.625, 0.125) == MAT_SKIN and at(s, x, 3.625, 0.125, "color") == shadow     # the slit row
        assert at(s, x, 3.375, 0.125) == MAT_SKIN and at(s, x, 3.375, 0.125, "color") == skin       # the lower lip row
        for z in (0.375, 0.625):
            assert at(s, x, 3.5, z) == MAT_AIR and at(s, x, 3.375, z) == MAT_AIR
    assert at(s, -2.625, 3.625, 0.125, "color") == skin and at(s, 0.125, 3.875, 0.125, "color") == skin
    assert at(s, -2.625, 3.5, 0.375) == MAT_SKIN and at(s, 0.125, 3.125, 0.375) == MAT_SKIN
    assert at(s, 0.125, 3.5, 1.125) == MAT_SKIN and LIP_THICK == 0.25
    for k in TEETH_X:
        assert at(s, (k + 1) / 4, 3.5, 0.875) == MAT_TOOTH, k
    for x in (-1.875, -0.375, 0.375, 1.875):             # the 1-fine gaps
        assert at(s, x, 3.5, 0.875) == MAT_AIR, x
    assert at(s, 0.125, 3.5, -0.125) == MAT_AIR          # nothing of the skull in front of the lips
    assert at(head.part("beard"), 0.125, 3.5, -0.125) == MAT_HAIR   # ... the moustache covers them
    dark = pal.index("eye_frame")                        # nostrils: the cube's bottom-front fine voxels at x +-0.5
    for x in (-0.625, 0.625):
        assert at(s, x, 4.625, -1.375, "color") == dark and at(s, x, 4.625, -1.125, "color") == dark
        assert at(s, x, 4.625, -1.625, "color") == skin and at(s, x, 4.875, -1.375, "color") == skin
    assert at(s, 0.125, 4.625, -1.375, "color") == skin


def test_face_features(head):
    s, beard, pal = head.part("skull"), head.part("beard"), head.palette
    # the eyes: 2 fine cut, the frame / white / pupil painted on the new front-most voxel, skin behind it
    for x, colour, mat in ((-3.375, "eye_frame", MAT_EYE), (-2.875, "eye_white", MAT_EYE), (-2.125, "pupil", MAT_EYE),
                           (2.125, "pupil", MAT_EYE), (3.375, "eye_frame", MAT_EYE)):
        assert at(s, x, 6.625, 0.125) == MAT_AIR and at(s, x, 6.625, 0.375) == MAT_AIR
        assert at(s, x, 6.625, 0.625) == mat and at(s, x, 6.625, 0.625, "color") == pal.index(colour), x
        assert at(s, x, 6.625, 0.875) == MAT_SKIN
    assert at(s, -1.125, 6.625, 0.375) == MAT_AIR and at(s, -0.875, 6.625, 0.375) == MAT_SKIN   # socket's inner wall re-skinned
    assert at(s, -2, 7.625, 0.125) == MAT_SKIN                                                   # lid skin between brow and eye
    # brows (hair, in the skull part), the moustache (beard part), the light cheek and lip, the philtrum
    assert at(s, -2, 8.5, -0.25) == MAT_HAIR and at(s, -2, 8.5, -0.25, "color") == pal.index("hair_black")
    assert at(s, 0.125, 8.5, -0.25) == MAT_AIR
    assert at(beard, 0, 3.5, -0.25) == MAT_HAIR and at(beard, 0, 3.5, -0.25, "color") == pal.index("hair_black")
    assert at(s, -3.5, 5.5, 0.125, "color") == pal.index("skin_light") and at(s, 0.125, 2.75, 0.125, "color") == pal.index("skin_light")
    assert at(s, 0.125, 4.25, 0.125) == MAT_AIR and at(s, 0.125, 4.25, 0.375) == MAT_AIR
    assert at(s, 0.125, 4.25, 0.625) == MAT_SKIN and at(s, 0.125, 4.25, 0.625, "color") == pal.index("skin")
    assert at(s, 0.125, 4.25, 0.875) == MAT_SKIN


def test_beard_is_a_shell_on_the_skull(built):
    bd, parts, reports = built
    f = bd.f
    skull, beard = parts["skull"].solid(), parts["beard"].solid()
    moustache = beard & f.box(-2.5, 2.5, 2.5, 4, -0.5, 0)                     # from the face layer
    base = skull | moustache                                                   # what the shell grew from
    shell = beard & ~moustache
    assert not (shell & ~dilate(base, 3)).any() and (shell & ~dilate(base, 2)).any()
    assert at_unit(f, parts["beard"], -1, 0.25, -1.125) == MAT_HAIR and at_unit(f, parts["beard"], -1, 0.25, -1.375) == MAT_AIR  # 3 fine before the chin (z -0.5)
    assert at_unit(f, parts["beard"], -1, -0.375, 0.25) == MAT_HAIR            # '=' hangs under the chin
    assert reports["beard"][0]["added"] == int(shell.sum()) > 0


def test_layers_never_touch_the_skull(built, maps):
    bd, parts, _ = built
    f = bd.f
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (parts[a].solid() & parts[b].solid()).any(), (a, b)
    fresh_bd = Builder(f, bd.pal)
    tmp = {}
    apply_layer(fresh_bd, tmp, "skull", maps.SKULL, np.random.default_rng(0))
    base = tmp["skull"].solid()
    for e in EARS:
        base |= f.mask(e)
    base &= ~f.box(CAVITY_X[0], CAVITY_X[1], CAVITY_Y[0], CAVITY_Y[1], LIP_THICK, 1)
    for n in EAR_NOTCHES:
        base &= ~f.mask(n)
    removed = base & ~parts["skull"].solid()               # only the eye and philtrum recesses take skull voxels away
    sockets = f.box(-3.5, -1, 6, 7.5, 0, 0.5) | f.box(1, 3.5, 6, 7.5, 0, 0.5) | f.box(-1, 1, 4, 4.5, 0, 0.5)
    assert removed.sum() == 2 * 10 * 6 * 2 + 8 * 2 * 2 and not (removed & ~sockets).any()
    assert (parts["skull"].solid() & ~base).sum() == 192 + len(TEETH_X) * 2 * 2   # the brows and the teeth


# ---------------------------------------------------------------------------- the bald spec
def test_bald_is_the_skull_and_face_alone(bald, head):
    assert bald.name == "bald" and [p.name for p in bald.parts] == ["skull"]
    s = bald.part("skull")
    assert mirrored(s) and mirrored(s, "color") and exposed_tissue(s) == 0
    report = checks.run(bald)
    assert report["errors"] == [] and [w for w in report["warnings"] if "enclosed air pocket" not in w] == []
    assert at(s, 0.125, 3.625, 0.125) == MAT_CARTILAGE and at(s, 0.125, 3.5, -0.125) == MAT_AIR   # the closed lips show (the slit row is cubic cartilage)
    assert at(s, 0.125, 3.625, 0.125, "color") == bald.palette.index("skin_shadow")            # ... with the mouth line
    assert at(s, 0.125, 3.5, 0.625) == MAT_AIR and at(s, 0.125, 3.5, 0.875) == MAT_TOOTH      # teeth hidden behind
    assert at(s, -2, 8.5, -0.25) == MAT_HAIR                                                 # brows stay
    assert at(s, 0.125, 5, -1.875) == MAT_SKIN and at(s, 5.75, 4.5, 7.5) == MAT_CARTILAGE
    assert np.array_equal(s.material, head.part("skull").material)                           # the same skull
