"""Reference-style head (voxgen.refhead) and the designed hair styles (voxgen.hairstyle): every
style builds on a synthetic skull and on the real form, the hair is one piece 6-adjacent to the
scalp (no overlap), the painted face has exactly its minimal features, the eyeballs are spheres
with iris / pupil / highlight, the coarse form has no one-voxel terrace, everything mirrors."""
import numpy as np
import pytest

from voxgen import checks, meshvox
from voxgen.body import exposed, shift
from voxgen.hairstyle import STYLES, Clump, HairParams, build_hair, hair_floor
from voxgen.hybridhead import quantize_form
from voxgen.materials import MAT_AIR, MAT_BONE, MAT_EYE, MAT_HAIR, MAT_SKIN
from voxgen.multires_head import MultiresParams, build_form
from voxgen.refhead import EYE_OPTIONS, RefHeadParams, build_ref_head

# ---------------------------------------------------------------------------- synthetic skull


def synthetic_head(vs: float = 0.005):
    """An ellipsoid skull (17 x 22 x 20 cm) with ear boxes, chin at y = 0, plus the landmarks
    build_hair needs (metres)."""
    dims = tuple(int(round(m / vs)) for m in (0.30, 0.30, 0.34))
    grid = meshvox.VoxelGrid((-0.15, -0.03, -0.17), vs, dims)
    X, Y, Z = grid.centres()
    cy, cz = 0.11, 0.0
    form = ((X / 0.085) ** 2 + ((Y - cy) / 0.11) ** 2 + ((Z - cz) / 0.10) ** 2) <= 1.0
    ear = (np.abs(X) > 0.08) & (np.abs(X) < 0.098) & (Y > 0.09) & (Y < 0.13) & (Z > 0.0) & (Z < 0.03)
    form |= ear
    lm = {"brow_y": 0.13, "head_top": 0.22, "ear_z": (0.0, 0.03), "ear_y": (0.09, 0.13), "ear_inner_x": 0.08,
          "eye_y": 0.115, "eye_x": 0.03}
    return grid, form, lm


@pytest.mark.parametrize("style", STYLES)
def test_hair_styles_on_synthetic_skull(style):
    grid, form, lm = synthetic_head()
    res = build_hair(form, grid, lm, style)
    hair = res.hair
    assert hair.any() and not (hair & form).any()
    assert np.array_equal(hair, hair[:, :, ::-1])                      # mirror-symmetric
    labels, n = checks.label_components(hair)
    assert n == 1                                                      # one piece
    support = form | hair
    touch = np.zeros(hair.shape, bool)
    for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        touch |= shift(support, *d)
    assert (hair & ~touch).sum() == 0                                  # 6-adjacent to skin or hair
    assert (hair & shift(form, 0, -1, 0)).sum() > 0                    # hair directly on the scalp
    assert not (res.lit & res.deep).any()
    assert res.lit.sum() > 0 and res.deep.sum() > 0
    assert (res.lit & ~hair).sum() == 0 and (res.deep & ~hair).sum() == 0
    X, Y, Z = grid.centres()
    assert not (hair & (Y < res.floor - 1e-6)).any()                   # nothing below the floor
    # the brow stays bare: no hair on the front below the forehead hairline
    y_hl = lm["brow_y"] + HairParams.for_style(style).hairline_frac * (lm["head_top"] - lm["brow_y"])
    assert not (hair & (Z < -0.05) & (np.abs(X) < 0.04) & (Y < y_hl - 1e-6)).any()
    st = res.stats
    if style == "quiff":
        assert st["quiff"]["steps"] > 0 and st["quiff"]["spikes"] == 4 and st["clumps"] == 0 and st["ridge"] == 0
    elif style == "messy":
        assert st["clumps"] > 0 and st["ridge"] == 0 and "quiff" not in st
    else:
        assert st["ridge"] > 0 and st["clumps"] == 0


def test_quiff_is_a_leaning_stair_and_floor_is_piecewise_flat():
    grid, form, lm = synthetic_head()
    plain = build_hair(form, grid, lm, "quiff", quiff=None)
    quiff = build_hair(form, grid, lm, "quiff")
    X, Y, Z = grid.centres()
    rise = float(Y[quiff.hair].max() - Y[plain.hair].max())
    assert 0.03 <= rise <= 0.06                                        # the shoulder 2 cm + three teeth steps
    q = quiff.stats["quiff"]
    assert q["steps"] >= 3 and q["spikes"] == 4 and q["stair"] > 0
    # the front of the stair (centre column) is a 45-degree diagonal: every step row is set back
    # exactly one step from the row below, and a step is two voxels tall at 5 mm; nothing hangs
    # in front of it (no vertical front wall above the ledge)
    H = quiff.hair
    nz, ny, nx = H.shape
    ix = nx // 2
    y0, y1 = q["rows"]
    top = int(np.nonzero(form[:, :, ix].any(axis=0))[0].max())        # the skull's top row: the stair is free above it
    y0 = y0 + ((top + 1 - y0 + 1) // 2) * 2
    front = [int(np.argmax(H[:, r, ix])) for r in range(y0, y1 + 1)]
    assert len(front) >= 4
    assert front[1::2] == front[0::2]                                  # pairs of rows share a step
    assert all(b - a == 2 for a, b in zip(front[0::2], front[2::2]))   # one step (2 voxels) back per step up
    # below, the stair hugs the round skull: hair right in front of the form on every row above the ledge
    for r in range(q["rows"][0], y0):
        if form[:, r, ix].any():
            assert H[int(np.argmax(form[:, r, ix])) - 1, r, ix]
    assert not (quiff.lit & ~H).any() and (quiff.lit & H & (Y > Y[plain.hair].max())).sum() > 0
    fl = hair_floor(X, Z, lm, HairParams.for_style("quiff"))
    assert len(np.unique(np.round(fl, 6))) <= 6                        # a handful of flat levels
    one = build_hair(form, grid, lm, "quiff", quiff=None, clumps=[Clump(0.0, 0.02, 0.03, 0.03, 0.03, 0.0, 0.01, 0.01)])
    assert one.stats["clumps"] == 1 and "quiff" not in one.stats


def test_hairline_override_and_floor_quantum():
    grid, form, lm = synthetic_head()
    res = build_hair(form, grid, lm, "short", hairline_y=0.17, floor_quantum=0.01, temple_drop=0.01)
    levels = sorted(set(np.round(np.unique(res.floor), 6).tolist()))
    assert all(abs(v * 100 - round(v * 100)) < 1e-6 for v in levels)   # every level on a 1 cm row
    X, Y, Z = grid.centres()
    assert float(res.floor[0, 0, grid.dims[0] // 2]) == pytest.approx(0.17)   # the forehead at the given hairline
    wide = build_hair(form, grid, lm, "short", ridge_period=4, ridge_width=2)
    assert wide.stats["ridge"] > res.stats["ridge"]


def test_hair_at_3mm_same_style():
    grid, form, lm = synthetic_head(0.003)
    res = build_hair(form, grid, lm, "quiff")
    assert res.stats["voxel_mm"] == 3.0 and res.stats["quiff"]["spikes"] == 4
    labels, n = checks.label_components(res.hair)
    assert n == 1


# ---------------------------------------------------------------------------- the real head

@pytest.fixture(scope="module")
def quiff_dots():
    return build_ref_head(RefHeadParams(eyes="dots", hair_style="quiff"), name="t_quiff_dots")


@pytest.fixture(scope="module")
def short_balls():
    return build_ref_head(RefHeadParams(eyes="balls", hair_style="short"), name="t_short_balls")


@pytest.fixture(scope="module")
def quiff_balls_3mm():
    return build_ref_head(RefHeadParams(eyes="balls", hair_style="quiff", hair_voxel=0.003), name="t_q3")


def test_params_validation():
    with pytest.raises(ValueError):
        RefHeadParams(eyes="laser")
    with pytest.raises(ValueError):
        RefHeadParams(hair_style="mohawk")
    assert EYE_OPTIONS == ("dots", "balls")


def test_parts_checks_and_seams(quiff_dots, short_balls, quiff_balls_3mm):
    for model, names in ((quiff_dots, ["head", "hair"]), (short_balls, ["head", "eye_l", "eye_r", "hair"]),
                         (quiff_balls_3mm, ["head", "eye_l", "eye_r", "hair"])):
        assert [p.name for p in model.parts] == names
        assert model.part("head").voxel_size == 0.005 and model.part("head").bone == "head"
        assert model.part("hair").bone == "head"
        report = checks.run(model)
        assert report["errors"] == [], report["errors"]
        seams = model.meta["seams"]
        for part, s in seams.items():
            assert s["contact"] > 0, part
        assert seams["hair"]["overlap"] == 0
        mesh = model.meta["mesh"]
        assert set(mesh["parts"]) == set(names) and mesh["quads"] > 0
    assert quiff_balls_3mm.part("hair").voxel_size == pytest.approx(0.003)
    assert quiff_dots.part("hair").voxel_size == pytest.approx(0.005)
    assert quiff_balls_3mm.part("hair").count_solid() > 3 * quiff_dots.part("hair").count_solid()


def test_hair_part_on_the_pocketed_scalp(quiff_dots):
    head, hair = quiff_dots.part("head"), quiff_dots.part("hair")
    b = quiff_dots.meta["build"]["hair"]
    assert b["pocket"] > 0 and b["quiff"]["steps"] >= 3 and b["quiff"]["spikes"] == 4
    hist = hair.material_histogram()
    assert set(hist) == {MAT_HAIR}
    pal = quiff_dots.palette
    tones = {pal.name_of(int(c)) for c in np.unique(hair.color[hair.solid()])}
    assert tones == {"hair", "hair_light", "hair_deep"}
    # the pocket: skin under the hair, never bone exposed to it
    assert head.material_histogram().get(MAT_BONE, 0) > 0
    ex = exposed(head.solid())
    assert (ex & (head.material == MAT_BONE)).sum() == 0


def test_face_is_painted_not_sculpted(quiff_dots):
    f = quiff_dots.meta["build"]["face"]
    assert f["mouth"] == 2 * 8 and f["nostril"] == 2 and f["eye_dots"] == 2 * 4 * 2 and f["brow"] == 2 * 6 * 2
    assert f["lower_lip"] == 2 * 3 and f["shade"] > 0
    head = quiff_dots.part("head")
    hist = head.material_histogram()
    assert MAT_EYE not in hist                                         # dots are colour on skin
    pal = quiff_dots.palette
    skin_cols = {pal.name_of(int(c)) for c in np.unique(head.color[head.material == MAT_SKIN])}
    assert skin_cols <= {"skin", "skin_shadow", "lip_soft", "line", "eye_dark", "hair"}
    # mirror symmetry of the materials and of the painted skin (teeth alternate their tint)
    assert np.array_equal(head.material, head.material[:, :, ::-1])
    skin_col = np.where(head.material == MAT_SKIN, head.color, 0)
    assert np.array_equal(skin_col, skin_col[:, :, ::-1])


def test_eyeballs_are_spheres_in_sockets(short_balls):
    pal = short_balls.palette
    for key in ("eye_l", "eye_r"):
        eye = short_balls.part(key)
        assert eye.voxel_size == pytest.approx(0.003) and eye.bone == key
        assert set(eye.material_histogram()) == {MAT_EYE}
        names = {pal.name_of(int(c)) for c in np.unique(eye.color[eye.solid()])}
        assert {"eye_white", "iris", "pupil", "highlight"} <= names
        s = short_balls.meta["build"][key]
        assert s["ball"] > 100 and s["stalk"] > 0
    f = short_balls.meta["build"]["face"]
    assert "eye_dots" not in f and f["lid"] == 2 * 4
    head = short_balls.part("head")
    assert (head.material == MAT_EYE).sum() == 0
    assert np.array_equal(head.material, head.material[:, :, ::-1])
    # the carved sockets are air pockets open to the front (no enclosed pocket warnings)
    report = checks.run(short_balls)
    assert not any("pocket" in w for w in report["warnings"]), report["warnings"]


def test_coarse_form_has_no_one_voxel_terrace():
    from voxgen import mh
    from voxgen.head_from_mesh import HeadParams
    p = RefHeadParams()
    human = mh.load(morphs=p.morphs)
    mp = MultiresParams(form_voxel=0.01, head=HeadParams(hair=True, hair_thickness=0.061), margin=0.02)
    _grid, solid, info = build_form(human, human.landmarks(), mp)
    again, q = quantize_form(solid)
    assert q["bumps_removed"] == 0 and q["pits_filled"] == 0 and np.array_equal(again, solid)
    assert info["components"] == 1 and solid.shape[2] % 2 == 0


def test_specs_build():
    import importlib.util
    from pathlib import Path
    specs = Path(__file__).resolve().parents[1] / "specs" / "chars"
    names = sorted(p.stem for p in specs.glob("ref_*.py"))
    assert names == ["ref_messy_dots", "ref_quiff_balls", "ref_quiff_balls_3mm", "ref_quiff_dots", "ref_short_balls",
                     "ref_slick_balls"]
    spec = importlib.util.spec_from_file_location("ref_slick_balls", specs / "ref_slick_balls.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.build()
    assert model.name == "ref_slick_balls" and model.meta["build"]["hair"]["style"] == "slick"
    assert model.meta["build"]["hair"]["ridge"] > 0
