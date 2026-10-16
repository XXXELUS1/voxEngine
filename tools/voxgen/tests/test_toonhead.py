"""The designed voxel-art head (voxgen.toonhead): every preset builds, is mirror-symmetric,
has eyes, hair on skin, honest layers and passes the checks; the ``real`` preset draws its face
with the 5 mm vocabulary of voxgen.facedetail (tested on a flat slab too)."""
import numpy as np
import pytest

from voxgen import checks
from voxgen.body import exposed
from voxgen.facedetail import FaceDetailParams, FaceLandmarks, apply_face_details
from voxgen.materials import MAT_AIR, MAT_BONE, MAT_EYE, MAT_HAIR, MAT_SKIN
from voxgen.toonhead import PRESETS, ToonHeadParams, build_toon_head


@pytest.fixture(scope="module", params=PRESETS)
def head(request):
    return build_toon_head(ToonHeadParams.preset_params(request.param), name="t_" + request.param)


def test_builds_and_passes_checks(head):
    part = head.part("head")
    hist = part.material_histogram()
    for m in (MAT_SKIN, MAT_BONE, MAT_EYE, MAT_HAIR):
        assert hist.get(m, 0) > 0, m
    if ToonHeadParams.preset_params(head.meta["preset"]).fine_details:
        assert hist[MAT_EYE] == 2 * 5 * 3                     # two human-sized 5 x 3 fine eyes
    else:
        assert hist[MAT_EYE] >= 2 * 2 * 2 * 2 * 3             # two eyes of at least 3 x 2 design voxels
    assert part.voxel_size == 0.005
    assert part.origin[1] == pytest.approx(-ToonHeadParams.preset_params(head.meta["preset"]).neck_rows * 0.01)
    report = checks.run(head)
    assert report["errors"] == [], report["errors"]
    assert report["warnings"] == [], report["warnings"]


def test_symmetric_and_layered(head):
    part = head.part("head")
    mat = part.material
    assert np.array_equal(mat, mat[:, :, ::-1])                # the material field mirrors exactly
    solid = mat != MAT_AIR
    ex = exposed(solid)
    assert not (ex & (mat == MAT_BONE)).any()                  # bone never on the surface
    assert (ex & (mat == MAT_SKIN)).sum() > 0.5 * (ex & (mat != MAT_HAIR)).sum()
    # hair sits on skin or hair, and every design voxel became a 2x2x2 fine block (a hair/eye
    # voxel is never split: steps are two fine voxels)
    hair = mat == MAT_HAIR
    coarse = hair[::2, ::2, ::2]
    assert np.array_equal(hair, coarse.repeat(2, 0).repeat(2, 1).repeat(2, 2))


def test_knobs_and_errors():
    p = ToonHeadParams.preset_params("toon", hair_style="bald", stubble=True, expression="happy")
    m = build_toon_head(p)
    hist = m.part("head").material_histogram()
    assert MAT_HAIR not in hist or hist[MAT_HAIR] < 200       # bald: at most the protruding brows
    assert m.palette.get("stubble") in set(np.unique(m.part("head").color).tolist())
    with pytest.raises(ValueError):
        ToonHeadParams.preset_params("giant")
    with pytest.raises(ValueError):
        build_toon_head(ToonHeadParams(), fine_voxel=0.003)


def test_real_preset_fine_face():
    m = build_toon_head(ToonHeadParams.preset_params("real"), name="t_real")
    d = m.meta["details"]
    assert d["eye"] == 30 and d["lid"] == 10 and d["nostril"] == 2 and d["brow"] == 16
    assert d["mouth"] > 0 and d["lips"] > 0 and d["bridge"] > 0 and d["ear"] == 4 and d["cheek"] > 0
    part = m.part("head")
    pal = m.palette
    cols = set(np.unique(part.color).tolist())
    assert {pal.index("iris"), pal.index("highlight"), pal.index("lips"), pal.index("line")} <= cols
    assert np.count_nonzero(part.color == pal.index("highlight")) == 2
    assert np.count_nonzero(part.color == pal.index("iris")) == 16       # 2 x 3 x 3 minus the highlights
    # canon: the eye centre sits at mid-height of the head (chin row 0 .. crown)
    eyes = np.nonzero(part.material == MAT_EYE)
    chin = (part.material != MAT_AIR).any(axis=(0, 2))
    rows = np.nonzero(chin)[0]
    y_eye = (eyes[1].mean() - (-part.origin[1] / part.voxel_size))
    assert 0.42 < y_eye / (rows.max() - (-part.origin[1] / part.voxel_size)) < 0.58


def flat_face(nx=24, ny=30, nz=12):
    """A skin slab facing -z with air in front: the simplest surface for the detail pass."""
    mat = np.zeros((nz, ny, nx), np.uint8)
    col = np.zeros((nz, ny, nx), np.uint8)
    mat[4:, :, :] = MAT_SKIN
    col[4:, :, :] = 1
    return mat, col


COLORS = {"skin": 1, "skin_shadow": 2, "skin_ruddy": 3, "lips": 4, "line": 5, "brow": 6,
          "eye_white": 7, "iris": 8, "highlight": 9}


def test_facedetail_on_a_slab():
    mat, col = flat_face()
    lm = FaceLandmarks(eye_x=5.5, eye_y=20, brow_y=24, nose_top_y=19, nose_base_y=13, nostril_x=1.5,
                       mouth_y=8, mouth_half_w=5.5, ear_y=15, ear_z=8, cheek_y=13)
    st = apply_face_details(mat, col, lm, COLORS, FaceDetailParams(cheeks=True, iris_in=1))
    assert np.array_equal(mat, mat[:, :, ::-1])                      # material mirrors exactly
    assert st["eye"] == 30 and (mat == MAT_EYE).sum() == 30
    # the eye is a one-voxel recess: air at z = 4 where the white sits at z = 5
    eye = mat == MAT_EYE
    assert eye[5].sum() == 30 and (mat[4][eye[5]] == MAT_AIR).all()
    assert (col == 8).sum() == 16 and (col == 9).sum() == 2            # iris minus highlight; one highlight per eye
    hl = np.nonzero(col == 9)
    assert hl[1].tolist() == [21, 21] and (hl[2] < 12).sum() == 1      # top row; one per side (screen-left columns)
    assert (col[4, 22] == 5).sum() == 10                               # lid line: one row, 5 per eye
    assert st["brow"] == 16 and (col[4, 24] == 6).sum() == 10 and (col[4, 25] == 6).sum() == 6
    assert st["bridge"] > 0 and (mat[3] == MAT_SKIN).sum() == st["bridge"]   # ridge protrudes one voxel
    assert st["nostril"] == 2 and (col[4, 13] == 5).sum() == 2
    assert (col[4, 8] == 5).sum() == 10 and (col[4, 9] == 5).sum() == 2     # mouth line + corners up
    assert (col[4, 9] == 4).sum() == 8 and (col[4, 7] == 4).sum() == 10     # upper lip thinner, lower lip
    assert st["cheek"] == 16 and st["ear"] == 4                             # 1 x 2 spot on each side face of the slab
    with pytest.raises(ValueError):
        apply_face_details(np.zeros((4, 4, 5), np.uint8), np.zeros((4, 4, 5), np.uint8), lm, COLORS)


def test_facedetail_relief_on_a_slab():
    mat, col = flat_face()
    lm = FaceLandmarks(eye_x=5.5, eye_y=20, brow_y=24, nose_top_y=19, nose_base_y=13, nostril_x=1.5,
                       mouth_y=8, mouth_half_w=5.5, ear_y=15, ear_z=8, cheek_y=13)
    st = apply_face_details(mat, col, lm, COLORS, FaceDetailParams(relief=True))
    assert np.array_equal(mat, mat[:, :, ::-1])
    # every relief step is exactly one voxel: new voxels only at z = 3 (in front of the slab face at z = 4)
    assert not (mat[:3] != MAT_AIR).any()
    assert st["brow_relief"] == st["brow"] == 16 and ((mat[3] == MAT_HAIR) & (col[3] == 6)).sum() == 16
    assert st["lid_ridge"] == 10 and ((mat[3, 23] == MAT_SKIN) & (col[3, 23] == 1)).sum() == 10   # row above the lid line
    assert (col[4, 22] == 5).sum() == 10                                                      # the lid line stays
    assert st["lips_relief"] == st["lips"] == 18 and (col[3, 9] == 4).sum() == 8 and (col[3, 7] == 4).sum() == 10
    assert (mat[3, 8] == MAT_AIR).all()                                                       # the mouth line is the recess
    # nostrils: two voxels deep, the hole's floor and walls dark, the rim around it untouched
    assert st["nostril"] >= 6 and (mat[4, 13] == MAT_AIR).sum() == 2 and (mat[5, 13] == MAT_AIR).sum() == 2
    hole_x = np.nonzero(mat[5, 13] == MAT_AIR)[0]
    assert (col[6, 13, hole_x] == 5).all() and (col[4, 13, hole_x - 1] == 1).all() and (col[4, 13, hole_x + 1] == 1).all()
