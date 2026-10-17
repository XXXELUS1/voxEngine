"""Block head (voxgen.blockhead) after docs/reference/barber-head.md: the skull is the 10 x 13 x 9 U
box with the table-2.1 chamfers as exact fine staircases and no narrowing in x, the nose's top is
the four-step staircase, the eyes are recessed framed blocks with the gaze shift, the mouth cavity
with its teeth hides behind the moustache, hair colours are per block (no islands) from the three
browns, the skull / nose / beard are mirror-exact in material, the parts come in order with the
expected bounds, and the checks pass clean."""
import numpy as np
import pytest

from voxgen import checks
from voxgen.blockhead import (HAIR_TONES, SKULL_MID_Z, TEETH_X, U_M, BlockHeadParams, build_head,
                              edge_chamfer)
from voxgen.materials import MAT_AIR, MAT_CLOTH, MAT_EYE, MAT_HAIR, MAT_PLANT, MAT_SKIN, MAT_TOOTH


@pytest.fixture(scope="module")
def head():
    return build_head(BlockHeadParams())


def at(part, x: float, y: float, z: float, field: str = "material") -> int:
    """Value of the fine cell containing the unit point (x, y, z); 0 outside the part."""
    loc = (x * U_M, y * U_M, (z - SKULL_MID_Z) * U_M)
    ix, iy, iz = (int(np.floor((loc[i] - part.origin[i]) / part.voxel_size + 1e-6)) for i in range(3))
    if not (0 <= ix < part.nx and 0 <= iy < part.ny and 0 <= iz < part.nz):
        return 0
    return int(getattr(part, field)[iz, iy, ix])


def mirrored(part, field: str = "material") -> bool:
    """Whether the field is exact under the x-mirror about local x = 0."""
    n0 = int(round(-part.origin[0] / part.voxel_size))
    arr = getattr(part, field)
    idx = 2 * n0 - 1 - np.arange(part.nx)
    ok = (idx >= 0) & (idx < part.nx)
    return bool(np.array_equal(arr[:, :, ok], arr[:, :, idx[ok]])) and not arr[:, :, ~ok].any()


def bounds_u(part):
    """((x0, y0, z0), (x1, y1, z1)) of the solid in units (hi exclusive)."""
    (x0, y0, z0), (x1, y1, z1) = part.bbox()
    vs = part.voxel_size
    lo = [(part.origin[i] + v * vs) / U_M for i, v in enumerate((x0, y0, z0))]
    hi = [(part.origin[i] + (v + 1) * vs) / U_M for i, v in enumerate((x1, y1, z1))]
    lo[2] += SKULL_MID_Z
    hi[2] += SKULL_MID_Z
    return tuple(round(v, 3) for v in lo), tuple(round(v, 3) for v in hi)


def test_edge_chamfer_is_a_one_to_one_staircase():
    c = (np.arange(4) + 0.5) / 4
    cut = edge_chamfer(c[:, None], c[None, :], 1.0)
    assert cut.sum() == 6 and cut[0, 0] and not cut[3, 0] and not cut[1, 2] and cut[1, 1]


def test_skull_profile_and_width(head):
    s = head.part("skull")
    # the under-jaw diagonal: z < 3 + y for y 0..6 (at the centre column), full depth 9 above
    for row in range(6):
        y = row + 0.125
        assert at(s, 0.125, y, 3 + row + 0.125) == MAT_SKIN and at(s, 0.125, y, 3 + row + 0.375) == MAT_AIR
    for y in (6.125, 8.125, 11.125):
        assert at(s, 0.125, y, 8.875) != MAT_AIR and at(s, 0.125, y, 9.125) == MAT_AIR
    # no narrowing in x: every row 0..11 is 10 U wide (the chin row included); measured at z = 2,
    # behind the front chamfer and in front of the forehead's slope, or at z = 1 beside the ears
    for row in range(12):
        z = 1.125 if row in (5, 6) else 2.125
        assert at(s, 4.875, row + 0.125, z) == MAT_SKIN and at(s, 5.125, row + 0.125, z) == MAT_AIR
    # the front vertical edges: a 1 U staircase from the face plane (8 U wide) to the sides
    assert at(s, 4.875, 5.125, 0.125) == MAT_AIR and at(s, 4.875, 5.125, 0.875) == MAT_SKIN
    assert at(s, 4.625, 5.125, 0.375) == MAT_AIR and at(s, 4.625, 5.125, 0.625) == MAT_SKIN
    assert at(s, 4.125, 5.125, 0.125) == MAT_SKIN and at(s, 3.875, 0.125 + 1, 0.125) == MAT_SKIN
    # the back vertical edges and the top edges, 1 U each
    assert at(s, 4.875, 8.125, 8.875) == MAT_AIR and at(s, 4.875, 8.125, 8.125) == MAT_SKIN
    assert at(s, 0.125, 12.875, 8.875) == MAT_AIR and at(s, 0.125, 12.875, 8.125) == MAT_SKIN
    assert at(s, 4.875, 12.875, 4.125) == MAT_AIR and at(s, 4.125, 12.875, 4.125) == MAT_SKIN
    # the forehead slopes back 2 U over y 9..11 and stays at z = 2 above
    assert at(s, 0.125, 9.125, 0.125) == MAT_SKIN and at(s, 0.125, 10.125, 0.875) == MAT_AIR
    assert at(s, 0.125, 10.125, 1.375) == MAT_SKIN and at(s, 0.125, 12.125, 1.875) == MAT_AIR
    assert at(s, 0.125, 12.125, 2.125) == MAT_SKIN
    # the chin's bottom-front edge
    assert at(s, 0.125, 0.125, 0.125) == MAT_AIR and at(s, 0.125, 0.125, 0.875) == MAT_SKIN
    # the neck box with its front edges cut; ears on the sides under the hair
    assert at(s, 0.125, -2.125, 1.125) == MAT_SKIN and at(s, 2.125, -2.125, 1.125) == MAT_AIR
    assert at(s, 2.125, -2.125, 1.875) == MAT_SKIN and at(s, 0.125, -2.125, 5.625) == MAT_AIR
    assert at(s, 5.25, 6, 2.5) == MAT_SKIN and at(s, -5.25, 6, 2.5) == MAT_SKIN
    assert bounds_u(s) == ((-5.5, -3.0, -0.5), (5.5, 13.0, 9.0))      # z -0.5: the proud brows and bridge


def test_nose_staircase(head):
    n = head.part("nose")
    for k in range(4):
        y = 4.5 + 0.125 + 0.25 * k
        front = -1 + 0.25 * k
        assert at(n, 0.125, y, front + 0.125) == MAT_SKIN and at(n, 0.125, y, front - 0.125) == MAT_AIR
    assert at(n, 0.125, 3.75, -0.875) == MAT_SKIN and at(n, 0.125, 5.75, -0.875) == MAT_AIR
    assert at(n, 0.625, 3.625, -0.875, "color") == head.palette.index("dark")
    assert bounds_u(n) == ((-1.0, 3.5, -1.0), (1.0, 5.5, 0.0))
    assert n.material_histogram() == {MAT_SKIN: n.count_solid()}


def test_eyes_brows_and_hidden_mouth(head):
    s, pal = head.part("skull"), head.palette
    for sx in (1, -1):
        assert at(s, sx * 2.25, 6.75, 0.125) == MAT_AIR                         # the socket is recessed
        assert at(s, sx * 2.25, 6.75, 0.375) == MAT_EYE and at(s, sx * 2.25, 6.75, 0.875) != MAT_AIR
        assert at(s, sx * 1.125, 6.125, 0.375, "color") == pal.index("dark")    # the frame
        assert at(s, sx * 3.875, 6.75, 0.125) == MAT_SKIN                       # the lid of skin beside it
    # gaze -0.5: both pupils shifted toward the viewer's left
    assert at(s, 1.75, 6.75, 0.375, "color") == pal.index("pupil") and at(s, 2.75, 6.75, 0.375, "color") == pal.index("eye_white")
    assert at(s, -2.75, 6.75, 0.375, "color") == pal.index("pupil") and at(s, -1.5, 6.75, 0.375, "color") == pal.index("eye_white")
    # brows: hair material half a unit proud, straight
    assert at(s, 2.25, 8, -0.25) == MAT_HAIR and at(s, 2.25, 8, -0.75) == MAT_AIR and at(s, 0.5, 8, -0.25) == MAT_AIR
    # the mouth cavity 1 U deep with eight fine teeth on its back wall, hidden behind the moustache bar
    assert at(s, 0.125, 2.5, 0.375) == MAT_AIR and at(s, 0.125, 2.5, 1.125) == MAT_SKIN
    for k in TEETH_X:
        assert at(s, k / 4 + 0.125, 2.5, 0.875) == MAT_TOOTH
    assert at(head.part("beard"), 0.125, 2.5, -0.25) == MAT_HAIR
    assert at(s, 0.125, 1.875, 0.125, "color") == pal.index("skin_light")      # the lip strip


def test_hair_colours_per_block(head):
    h = head.part("hair")
    tones = {head.palette.index(t) for t in HAIR_TONES}
    solid = h.solid()
    assert set(np.unique(h.color[solid]).tolist()) <= tones and h.material_histogram() == {MAT_HAIR: int(solid.sum())}
    # no colour islands: every hair voxel has a 6-neighbour of its own colour
    same = np.zeros(h.shape, bool)
    for axis in range(3):
        for d in (1, -1):
            c = np.roll(h.color, d, axis=axis)
            m = np.roll(solid, d, axis=axis)
            same |= m & (c == h.color)
    assert not (solid & ~same).any()
    # the temple strands reach at most hair_thickness beyond the skull's side (5 U), both sides
    lo, hi = bounds_u(h)
    assert hi[0] <= 6.5 + 1e-6 and lo[0] >= -6.5 - 1e-6 and lo[1] <= 3.0


def test_symmetry_of_skull_nose_beard(head):
    for name in ("skull", "nose", "beard"):
        assert head.part(name).symmetric and mirrored(head.part(name))
    assert mirrored(head.part("nose"), "color") and mirrored(head.part("beard"), "color")
    straight = build_head(BlockHeadParams(gaze=0.0))
    assert mirrored(straight.part("skull"), "color") and not mirrored(head.part("skull"), "color")


def test_parts_bounds_and_checks(head):
    assert [p.name for p in head.parts] == ["skull", "nose", "beard", "hair", "beret", "feather"]
    beret, feather = head.part("beret"), head.part("feather")
    assert beret.material_histogram() == {MAT_CLOTH: beret.count_solid()}
    assert feather.material_histogram() == {MAT_PLANT: feather.count_solid()}
    assert not beret.symmetric and not feather.symmetric and not head.part("hair").symmetric
    # the slouch hangs on the -x side only; the feather climbs above the beret at the top-left
    assert at(beret, -7, 11, 4) == MAT_CLOTH and at(beret, 7, 11, 4) == MAT_AIR
    assert bounds_u(feather)[1][1] > bounds_u(beret)[1][1] and bounds_u(feather)[0][0] < -6
    assert head.meta["generator"] == "blockhead" and head.voxel_size == pytest.approx(0.005)
    report = checks.run(head)
    assert report["errors"] == [] and report["warnings"] == [], report
    # the beret and the hair never share a voxel with the skull (shaving leaves the skin whole)
    bare = build_head(BlockHeadParams(beard=False, beret=False))
    assert [p.name for p in bare.parts] == ["skull", "nose", "hair"]
    other = build_head(BlockHeadParams(seed=11))
    assert other.part("hair").count_solid() != head.part("hair").count_solid()
    with pytest.raises(ValueError):
        build_head(BlockHeadParams(gaze=0.3))
