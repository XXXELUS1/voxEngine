"""The MakeHuman-based heads: voxgen.mh, meshvox, head_from_mesh, and the hybrid head
(voxgen.hybridhead: coarse form from the morphed mesh, fine designed details)."""
import numpy as np
import pytest

from voxgen import checks, meshvox, mh
from voxgen.body import dilate, exposed
from voxgen import mh_targets
from voxgen.body import exposed as exposed_mask
from voxgen.head_from_mesh import MID_MORPHS, HeadParams, build_head, make_palette
from voxgen.hybridhead import DEFAULT_MORPHS, HybridHeadParams, build_hybrid_head, quantize_form
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


# ---------------------------------------------------------------------------- mid caricature + relief

@pytest.fixture(scope="module")
def mid():
    return build_head(0.003, HeadParams.mid(1.2), name="t_mid")


def test_mid_targets_are_bundled():
    for rel, _w in mh_targets.files_for(MID_MORPHS):
        assert mh_targets.target_path(rel, download=False).is_file(), rel


def test_mid_params():
    p = HeadParams.mid(1.9, morphs={"nose-scale-vert": 0.3}, stubble=True)
    assert p.relief and p.flat_tints and p.clean_surface and p.stubble
    assert p.morphs["nose-scale-vert"] == pytest.approx(MID_MORPHS["nose-scale-vert"] * 1.9 + 0.3, abs=1e-3)
    assert p.lip_half_height > HeadParams.mid(0.65).lip_half_height           # strong: five lip rows
    assert not HeadParams.mid(0.65).stubble and HeadParams.stylized().stubble  # mid presets: stubble off


def test_mid_relief_is_one_voxel_geometry(mid):
    part = mid.part("head")
    mat, col = part.material, part.color
    pal = make_palette(True)
    assert np.array_equal(mat, mat[:, :, ::-1])
    report = checks.run(mid)
    assert report["errors"] == [] and report["warnings"] == [], report
    r = mid.meta["relief"]
    assert r["ridge"] >= 12 and r["lips"] >= 24 and r["nostril_inner"] == 4   # two 2-voxel slots, 2 deep
    solid = mat != MAT_AIR
    nz, ny, nx = mat.shape
    xc = nx // 2
    front_z = lambda y: int(np.argmax(solid[:, y, xc]))
    front_col = lambda y: int(col[front_z(y), y, xc])
    # the mouth line is a one-voxel recess between the two protruding lip rows (centre column)
    rows = [y for y in range(ny // 2) if solid[:, y, xc].any() and front_col(y) == pal.index("line")]
    assert len(rows) == 1, rows
    my = rows[0]
    assert front_col(my + 1) == pal.index("lips") and front_col(my - 1) == pal.index("lips")
    assert front_z(my + 1) == front_z(my) - 1 and front_z(my - 1) == front_z(my) - 1
    # the brows protrude as hair voxels in the brow tone, sitting on skin, above the eyes
    eye = mat == MAT_EYE
    eye_top = int(np.nonzero(eye)[1].max())
    brow = (mat == MAT_HAIR) & (col == pal.index("hair_deep"))
    brow &= (np.arange(ny)[None, :, None] > eye_top) & (np.arange(ny)[None, :, None] <= eye_top + 8)
    brow &= (np.arange(nz)[:, None, None] < nz // 2) & (np.abs(np.arange(nx) - xc)[None, None, :] <= 20)  # the face, not the sideburns
    assert brow.sum() >= 20
    assert (dilate(brow) & (mat == MAT_SKIN)).sum() >= brow.sum() // 2
    # the eye opening has a flat top under the dark lid line: the top eye row is as wide as the one below
    zs, ys, xs = np.nonzero(eye & exposed(solid))
    top, below = xs[ys == ys.max()], xs[ys == ys.max() - 1]
    assert top.size >= 0.7 * below.size
    # nostril slots: air pockets open to the front, walls in the dark tone next to them
    hole_walls = (col == pal.index("line")) & (mat == MAT_SKIN) & (np.arange(ny)[None, :, None] < my + 12) \
        & (np.arange(ny)[None, :, None] > my + 2)
    assert hole_walls.sum() >= 4


# ---------------------------------------------------------------------------- hybrid head

@pytest.fixture(scope="module")
def hybrid():
    return build_hybrid_head(HybridHeadParams(), name="t_hybrid")


def test_hybrid_targets_are_bundled():
    for rel, _w in mh_targets.files_for(DEFAULT_MORPHS):
        assert mh_targets.target_path(rel, download=False).is_file(), rel


def test_hybrid_builds_symmetric_and_layered(hybrid):
    part = hybrid.part("head")
    mat = part.material
    hist = part.material_histogram()
    assert hist[MAT_EYE] == 30 and hist[MAT_HAIR] > 1000 and hist[MAT_SKIN] > hist[MAT_EYE]
    assert np.array_equal(mat, mat[:, :, ::-1])
    solid = mat != MAT_AIR
    ex = exposed_mask(solid)
    assert not (ex & (mat == MAT_BONE)).any()
    assert part.voxel_size == 0.005 and -0.05 < part.origin[1] < 0.0        # chin near y = 0, neck below
    report = checks.run(hybrid)
    assert report["errors"] == [] and report["warnings"] == [], report
    d = hybrid.meta["details"]
    assert d["eye"] == 30 and d["nostril"] == 2 and d["brow"] == 16 and d["mouth"] > 0 and d["strands"] > 0
    assert hybrid.meta["form"]["bumps_removed"] >= 0 and hybrid.meta["form"]["leak"] is False
    # the form is 1 cm blocks: skin + inner tissues (hair and the fine details aside) is 2x2x2-aligned
    # except where the details carved it: compare the coarse majority with the fine field
    body = solid & (mat != MAT_HAIR)
    coarse = body[::2, ::2, ::2]
    up = coarse.repeat(2, 0).repeat(2, 1).repeat(2, 2)
    assert (body != up).sum() < 0.02 * body.sum()


def test_hybrid_quantize_removes_bumps_and_pits():
    s = np.zeros((9, 9, 9), bool)
    s[2:, :, :] = True
    s[1, 4, 4] = True          # a lone bump
    s[2, 1, 1] = False         # a one-voxel pit
    q, info = quantize_form(s)
    assert info == {"bumps_removed": 1, "pits_filled": 1}
    assert not q[1, 4, 4] and q[2, 1, 1] and q[2:].all()
    assert np.array_equal(q, q[:, :, ::-1])


def test_hybrid_knobs():
    m = build_hybrid_head(HybridHeadParams(hair=False, stubble=True), name="t_bald")
    part = m.part("head")
    assert MAT_HAIR not in part.material_histogram()
    assert m.palette.index("stubble") in set(np.unique(part.color).tolist())
    with pytest.raises(ValueError):
        build_hybrid_head(HybridHeadParams(), fine_voxel=0.003)
