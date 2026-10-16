"""The MakeHuman-based head (voxgen.mh, meshvox, head_from_mesh)."""
import numpy as np
import pytest

from voxgen import checks, meshvox, mh
from voxgen.body import dilate, exposed
from voxgen.head_from_mesh import HeadParams, build_head
from voxgen.materials import MAT_AIR, MAT_BONE, MAT_EYE, MAT_HAIR, MAT_SKIN, MAT_TOOTH
from voxgen.objmesh import ObjMesh


@pytest.fixture(scope="module")
def head():
    return build_head(0.005, HeadParams(), name="t")


def test_asset_and_landmarks():
    assert mh.ASSET.exists()
    h = mh.load()
    lm = h.landmarks()
    assert 0.20 < lm["head_height"] < 0.27
    assert abs(lm["eye_l"][0] + lm["eye_r"][0]) < 1e-6          # symmetric about x = 0
    assert lm["eye_l"][2] < lm["neck_base"][2]                  # face toward -Z
    assert lm["chin"][1] < lm["eye_y"] < lm["hairline_y"] < lm["head_top"]
    assert h.body().vertices[:, 1].min() >= -1e-6                # feet at y = 0


def test_cube_voxelize_and_fill():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float) * 0.05 + 0.0025
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                  [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3]], np.int32)
    cube = ObjMesh(v, f, np.zeros(len(f), np.int32), ["cube"])
    grid = meshvox.VoxelGrid.from_box((-0.02, -0.02, -0.02), (0.07, 0.07, 0.07), 0.005, centre_x=False)
    solid, info = meshvox.voxelize_solid(cube, grid)
    assert not info["leak"] and info["interior"] > 0
    assert 1200 <= int(solid.sum()) <= 1400           # an 11-voxel cube, shell inclusive
    # an open box leaks
    open_box = ObjMesh(v, f[2:], np.zeros(len(f) - 2, np.int32), ["cube"])
    _, info2 = meshvox.voxelize_solid(open_box, grid)
    assert info2["leak"]
    # closing the hole fixes it
    _, info3 = meshvox.voxelize_solid(meshvox.close_holes(open_box), grid)
    assert not info3["leak"]


def test_head_mh_builds_with_all_tissues(head):
    part = head.part("head")
    hist = part.material_histogram()
    for m in (MAT_SKIN, MAT_BONE, MAT_EYE, MAT_TOOTH, MAT_HAIR):
        assert hist.get(m, 0) > 0, m
    assert 20_000 < part.count_solid() < 80_000
    assert part.origin[0] == pytest.approx(-part.nx * part.voxel_size / 2)   # x-centred grid
    assert -0.05 < part.origin[1] < 0.0                                      # chin near y = 0 (neck stub below)


def test_head_mh_bone_never_exposed_and_hair_on_skin(head):
    part = head.part("head")
    solid = part.material != MAT_AIR
    ex = exposed(solid)
    assert not (ex & (part.material == MAT_BONE)).any()
    hair = part.material == MAT_HAIR
    touch_skin = dilate(part.material == MAT_SKIN) & hair
    assert touch_skin.sum() > 0.2 * hair.sum()
    # every hair voxel is 6-connected to the skin through hair
    report = checks.run(head)
    assert report["errors"] == [], report["errors"]
    assert not any("hair" in w for w in report["warnings"]), report["warnings"]


def test_head_mh_eyes_visible_and_symmetric(head):
    part = head.part("head")
    eye = part.material == MAT_EYE
    assert (eye & exposed(part.material != MAT_AIR)).sum() >= 4
    mat = part.material
    assert np.array_equal(mat, mat[:, :, ::-1])
