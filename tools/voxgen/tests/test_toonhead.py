"""The designed voxel-art head (voxgen.toonhead): every preset builds, is mirror-symmetric,
has eyes, hair on skin, honest layers and passes the checks; the ``real`` preset draws its face
with the 5 mm vocabulary of voxgen.facedetail (tested on a flat slab too); the ``box`` preset
is a chamfered box with a painted face, eye / sunglasses / hair parts, every form step two fine
voxels."""
import numpy as np
import pytest

from voxgen import checks
from voxgen.body import exposed
from voxgen.facedetail import FaceDetailParams, FaceLandmarks, apply_face_details
from voxgen.head_from_mesh import upsample
from voxgen.materials import MAT_AIR, MAT_BONE, MAT_EYE, MAT_GLASS, MAT_HAIR, MAT_SKIN
from voxgen.toonhead import BOX_EYES, BOX_HAIR_STYLES, PRESETS, ToonHeadParams, build_toon_head, design_box_head


@pytest.fixture(scope="module", params=PRESETS)
def head(request):
    return build_toon_head(ToonHeadParams.preset_params(request.param), name="t_" + request.param)


def test_builds_and_passes_checks(head):
    part = head.part("head")
    hist = part.material_histogram()
    p = ToonHeadParams.preset_params(head.meta["preset"])
    if p.preset == "box":
        # eyes and hair are parts of their own; the head is skin over bone
        assert [q.name for q in head.parts] == ["head", "eye_l", "eye_r", "hair"]
        assert hist.get(MAT_SKIN, 0) > 0 and hist.get(MAT_BONE, 0) > 0 and MAT_EYE not in hist and MAT_HAIR not in hist
        assert set(head.part("hair").material_histogram()) == {MAT_HAIR}
        assert set(head.part("eye_l").material_histogram()) == {MAT_EYE}
    else:
        for m in (MAT_SKIN, MAT_BONE, MAT_EYE, MAT_HAIR):
            assert hist.get(m, 0) > 0, m
    if p.fine_details:
        assert hist[MAT_EYE] == 2 * 5 * 3                     # two human-sized 5 x 3 fine eyes
    elif p.preset != "box":
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


# ---------------------------------------------------------------------------- the box preset

@pytest.fixture(scope="module")
def box_shades():
    return build_toon_head(ToonHeadParams.preset_params("box", eyes="shades", hair_style="quiff"), name="t_box_shades")


def test_box_form_is_whole_design_voxels():
    """Every step of the form is two fine voxels: the bald, dots head is exactly the x2 upsample
    of the design (no pocket, no sockets); the design itself is an even grid with the nose a
    2 x 3 x 1 block one row above the mouth and the face one flat plane."""
    p = ToonHeadParams.preset_params("box", eyes="dots", hair_style="bald")
    d = design_box_head(p)
    assert d.even and d.shape[2] % 2 == 0
    m = build_toon_head(p, name="t_box_bald")
    assert [q.name for q in m.parts] == ["head"]
    head = m.part("head")
    assert np.array_equal(head.solid(), upsample(d.solid(), 2))
    # the face plane: in front of the form only the nose block (one design voxel deep)
    front = int(np.argmax(d.solid().any(axis=(1, 2))))
    nose = d.solid()[front]
    assert nose.sum() == p.nose_w * p.nose_h and not d.solid()[:front].any()
    rows = np.nonzero(nose.any(axis=1))[0]
    assert rows.min() == p.nose_y + d.y0 and rows.max() == p.nose_y + p.nose_h - 1 + d.y0
    assert p.nose_y == p.mouth_y + 2                              # one empty row between the nose and the mouth
    plane = d.solid()[front + 1]
    assert plane[p.mouth_y + d.y0:p.brow_y + 2 + d.y0, d.hx - p.jaw_w // 2 + 1:d.hx + p.jaw_w // 2 - 1].all()
    f = m.meta["build"]["face"]
    assert f["mouth"] == 16 and f["lower_lip"] == 6 and f["nostril"] == 2 and f["brow"] == 24 and f["eye_dots"] == 16
    assert f["band"] > 0 and f["shade"] > f["band"]
    pal = m.palette
    skin_cols = {pal.name_of(int(c)) for c in np.unique(head.color[head.material == MAT_SKIN])}
    assert skin_cols == {"skin", "skin_shadow", "lip_soft", "line", "hair", "eye_dark"}
    report = checks.run(m)
    assert report["errors"] == [] and report["warnings"] == []


def test_box_balls_quiff(head):
    if head.meta["preset"] != "box":
        pytest.skip("the module fixture's box head")
    b = head.meta["build"]
    assert b["face"]["lid"] == 8 and "eye_dots" not in b["face"] and "brow" in b["face"]
    for key in ("eye_l", "eye_r"):
        eye = head.part(key)
        assert eye.voxel_size == pytest.approx(0.003) and eye.bone == key and b[key]["ball"] > 100
        names = {head.palette.name_of(int(c)) for c in np.unique(eye.color[eye.solid()])}
        assert {"eye_white", "iris", "pupil", "highlight"} <= names
    seams = head.meta["seams"]
    assert seams["hair"]["overlap"] == 0 and seams["hair"]["contact"] > 0
    hair = head.part("hair")
    tones = {head.palette.name_of(int(c)) for c in np.unique(hair.color[hair.solid()])}
    assert tones == {"hair", "hair_light", "hair_deep"}
    q = b["hair"]["quiff"]
    assert q["steps"] >= 4 and q["spikes"] == 4 and b["hair"]["pocket"] > 0 and b["hair"]["ridge"] == 0
    # the stair: the hair's front at the centre column steps back one design voxel per design row
    H = hair.solid()
    ix = H.shape[2] // 2
    oy = int(round((hair.origin[1] - head.part("head").origin[1]) / 0.005))
    y0, y1 = q["rows"]
    front = [int(np.argmax(H[:, r - oy, ix])) for r in range(y0, y1 + 1)]
    assert front[1::2] == front[0::2] and all(b_ - a == 2 for a, b_ in zip(front[0::2], front[2::2]))
    mesh = head.meta["mesh"]
    assert set(mesh["parts"]) == {"head", "eye_l", "eye_r", "hair"} and mesh["quads"] > 0


def test_box_shades_part(box_shades):
    assert [q.name for q in box_shades.parts] == ["head", "shades", "hair"]
    shades = box_shades.part("shades")
    assert set(shades.material_histogram()) == {MAT_GLASS} and shades.bone == "head"
    n, = {int(v) for v in shades.material_histogram().values()}
    assert n == box_shades.meta["build"]["shades"] > 0
    labels, count = checks.label_components(shades.solid())
    assert count == 1                                                  # the lenses wrap the corners in one piece
    seams = box_shades.meta["seams"]["shades"]
    assert seams["overlap"] == 0 and seams["contact"] > 0.75 * n        # on the face (the wrap's outer voxels sit on the lenses)
    f = box_shades.meta["build"]["face"]
    assert "brow" not in f and f["eye_dots"] == 16                      # no brows; dots under the glasses
    assert f["band"] > 0
    report = checks.run(box_shades)
    assert report["errors"] == [] and report["warnings"] == []


def test_box_params_validation():
    assert BOX_EYES == ("balls", "dots", "shades") and BOX_HAIR_STYLES == ("quiff", "short", "slick", "bald")
    with pytest.raises(ValueError):
        build_toon_head(ToonHeadParams.preset_params("box", eyes="laser"))
    with pytest.raises(ValueError):
        build_toon_head(ToonHeadParams.preset_params("box", hair_style="bangs"))
    with pytest.raises(ValueError):
        build_toon_head(ToonHeadParams.preset_params("box", cranium_w=15))
    short = build_toon_head(ToonHeadParams.preset_params("box", hair_style="short"), name="t_box_short")
    slick = build_toon_head(ToonHeadParams.preset_params("box", hair_style="slick"), name="t_box_slick")
    assert "quiff" not in short.meta["build"]["hair"] and short.meta["build"]["hair"]["ridge"] == 0
    assert slick.part("hair").count_solid() != short.part("hair").count_solid()


def test_box_specs_build():
    import importlib.util
    from pathlib import Path
    specs = Path(__file__).resolve().parents[1] / "specs" / "chars"
    names = sorted(q.stem for q in specs.glob("box_*.py"))
    assert names == ["box_balls_quiff", "box_balls_short", "box_balls_slick", "box_dots_short", "box_shades_quiff"]
    for name in ("box_shades_quiff", "box_dots_short"):
        spec = importlib.util.spec_from_file_location(name, specs / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model = mod.build()
        assert model.name == name and model.meta["preset"] == "box"
        assert model.meta["eyes"] == name.split("_")[1] and model.meta["hair_style"] == name.split("_")[2]
        assert checks.run(model)["errors"] == []
