"""Town characters (voxgen.character, voxgen.body_from_mesh, the facial hair of voxgen.hairstyle
and the per-character knobs of voxgen.sculpthead): the body voxelizes clean and headless with
feet on the ground, every outfit is one symmetric shell outside the body, the facial hair /
tonsure / scarf are one piece on the head, the semantic knobs move the metre parameters the
right way, a whole character builds, passes the checks and stands its head on the neck, and the
five town specs build."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from voxgen import checks, meshvox
from voxgen.body import exposed
from voxgen.body_from_mesh import BUILDS, OUTFITS, BodyParams, Outfit, build_body, clothes
from voxgen.character import CharacterParams, build_character
from voxgen.hairstyle import BEARD_STYLES, beard_mask, jaw_mask
from voxgen.materials import MAT_BONE, MAT_CLOTH, MAT_LEATHER, MAT_SKIN
from voxgen.model import Model
from voxgen.sculpthead import SculptParams, build_sculpt_head, head_dims, landmarks, sculpt_form
from voxgen.toonhead import ToonHeadParams, make_palette

SPECS = Path(__file__).resolve().parents[1] / "specs" / "chars" / "town"


@pytest.fixture(scope="module")
def palette():
    return make_palette(ToonHeadParams.preset_params("box"))


@pytest.fixture(scope="module")
def body(palette):
    return build_body(BodyParams(), palette=palette)


@pytest.fixture(scope="module")
def head5():
    """A 5 mm head's form, grid and landmarks for the facial-hair tests (fast)."""
    p = SculptParams(hair_style="bald", eyes="dots").resolved()
    part, solid, _, _ = sculpt_form(p)
    grid = meshvox.VoxelGrid(tuple(part.origin), part.voxel_size, part.dims)
    return solid, grid, landmarks(p)


def test_body_is_clean_headless_and_grounded(body):
    part = body.part
    assert part.voxel_size == 0.01 and part.bone == "body" and part.symmetric
    assert part.nx % 2 == 0 and part.origin[0] == pytest.approx(-part.nx * 0.005)
    solid = part.solid()
    assert np.array_equal(solid, solid[:, :, ::-1])
    labels, n = checks.label_components(solid)
    assert n == 1
    X, Y, Z = part.coords()
    assert float(Y[solid].min()) < 0.015                              # feet on the ground
    assert float(Y[solid].max()) < body.landmarks["neck_top"]          # the head is gone
    assert float(Y[solid].max()) > body.landmarks["neck_y"]            # the neck stub stays
    assert float(Z[solid & (Y > 1.0)].min()) < -0.15                   # the forearms point forward (-Z)
    hist = part.material_histogram()
    assert all(hist.get(m, 0) > 0 for m in (MAT_SKIN, MAT_BONE))
    assert not (exposed(solid) & (part.material != MAT_SKIN)).any()    # the cut face is skin too
    assert body.stats["clean"]["components"] == 1 and body.stats["voxelize"]["leak"] is False
    m = Model("t_body", 0.01, body.part.model.palette if body.part.model else make_palette(ToonHeadParams.preset_params("box")))
    m.add_part(part.copy())
    report = checks.run(m)
    assert report["errors"] == [] and report["warnings"] == []


@pytest.mark.parametrize("build", BUILDS)
def test_builds_morph_the_body(build, palette):
    b = build_body(BodyParams(build=build), palette=palette)
    assert b.stats["solid"] > 40000 and b.landmarks["neck_top"] > 1.3
    if build == "plump":
        assert b.stats["solid"] > build_body(BodyParams(build="lean"), palette=palette).stats["solid"]


@pytest.mark.parametrize("kind", OUTFITS)
def test_outfits(kind, palette):
    b = build_body(BodyParams(build="normal"), palette=palette)
    before = b.part.color.copy()
    shell, stats = clothes(b, Outfit(kind=kind, patches=(kind == "tunic")), palette)
    assert stats["boots"] > 0
    assert (b.part.color != before).sum() > 1000                      # trousers and boots painted
    if kind == "tunic":
        assert shell is None and stats["tunic"] > 0 and stats["patches"] > 0 and stats["belt"] > 0
        return
    assert shell is not None and shell.name == "clothes" and shell.bone == "body" and shell.symmetric
    mats = set(shell.material_histogram())
    assert MAT_CLOTH in mats and (mats == {MAT_CLOTH} or mats == {MAT_CLOTH, MAT_LEATHER})
    s = shell.solid()
    assert np.array_equal(s, s[:, :, ::-1])
    labels, n = checks.label_components(s)
    assert n == 1
    # the shell is outside the body and touches it
    m = Model("t_" + kind, 0.01, palette)
    m.add_part(b.part.copy())
    m.add_part(shell.copy())
    from voxgen.multires_head import seam_stats
    seams = seam_stats(m, "body")["clothes"]
    assert seams["overlap"] == 0 and seams["contact"] > 0.2 * seams["solid"]      # the skirt fill floats between the legs
    report = checks.run(m)
    assert report["errors"] == [] and report["warnings"] == []
    if kind in ("robe", "dress"):
        X, Y, Z = shell.coords()
        assert float(Y[s].min()) < 0.12                                # down to the ankles
        assert (s & (np.abs(X) < 0.03) & (Y < 0.6)).any()              # the skirt fills between the legs
    if kind == "apron":
        assert stats["straps"] > 0 and stats["shell"] > 1000


@pytest.mark.parametrize("style", [s for s in BEARD_STYLES if s != "stubble"])
def test_beards_are_one_piece_outside_the_form(style, head5):
    form, grid, lm = head5
    beard, stats = beard_mask(form, grid, lm, style)
    assert stats["solid"] == int(beard.sum()) > 20
    assert not (beard & form).any()
    assert np.array_equal(beard, beard[:, :, ::-1])
    labels, n = checks.label_components(beard)
    assert n == 1
    X, Y, Z = grid.centres()
    assert float(Y[beard].max()) < lm["ear_y"][1] + 0.002             # never above the ear top (the sideburn end)
    centre = beard & (np.abs(X) <= lm["mouth_half_w"] + 0.012)
    assert float(Y[centre].max()) < lm["nose_base_y"] + grid.voxel_size   # under the nose in the middle
    if style == "mustache":
        assert float(Y[beard].min()) > lm["mouth_y"]                   # above the mouth only
    if style == "full":
        assert float(Y[beard].min()) < lm["chin_y"] - 0.015            # hangs under the chin
    zone = jaw_mask(form, grid, lm)
    assert not (zone & (np.abs(Y - lm["mouth_y"]) <= 0.007) & (np.abs(X) <= lm["mouth_half_w"])).any()


def test_tonsure_and_headscarf():
    m = build_sculpt_head(SculptParams(hair_style="short", tonsure=0.05, eyes="dots"), name="t_tonsure")
    hair = m.part("hair")
    X, Y, Z = hair.coords()
    h = hair.solid()
    assert m.meta["build"]["hair"]["tonsure_removed"] > 500
    assert not (h & (X * X + (Z - 0.004) ** 2 < 0.04 ** 2) & (Y > 0.19)).any()   # bald crown
    assert (h & (Y > 0.17)).any()                                      # the ring remains
    report = checks.run(m)
    assert report["errors"] == [] and report["warnings"] == []
    w = build_sculpt_head(SculptParams(hair_style="short", headscarf=True, eyes="dots", sex="female"), name="t_scarf")
    assert [q.name for q in w.parts] == ["head", "hair", "scarf"]
    scarf = w.part("scarf")
    assert set(scarf.material_histogram()) == {MAT_CLOTH} and scarf.bone == "head"
    s = scarf.solid()
    assert np.array_equal(s, s[:, :, ::-1]) and checks.label_components(s)[1] == 1
    seams = w.meta["seams"]["scarf"]
    assert seams["overlap"] == 0 and seams["contact"] > 0
    Xs, Ys, Zs = scarf.coords()
    assert float(Ys[s].max()) > float(w.part("hair").coords()[1][w.part("hair").solid()].max())   # over the hair
    assert w.meta["build"]["scarf"]["tail"] > 50 and w.meta["build"]["scarf"]["hem"] > 50
    report = checks.run(w)
    assert report["errors"] == [] and report["warnings"] == []


def test_knobs_and_sex_resolve():
    base = SculptParams().resolved()
    assert base.jaw_angle[0] + base.jaw_r <= 0.85 * base.cranium_r[0] + 0.004     # the jaw limit
    assert base.brow_r <= 0.008 and base.brow_inner_x >= 0.012 and base.hollow_r == 0.0
    assert base.lips_y < base.wing_c[1] - base.wing_r and base.lips_y > 0.035           # lower third
    f = SculptParams(sex="female").resolved()
    assert f.sex == "male" and f.jaw_angle[0] < base.jaw_angle[0] and f.nose_tip_r < base.nose_tip_r
    assert f.lips_r > base.lips_r and f.brow_r < base.brow_r and f.neck_r < base.neck_r
    k = SculptParams(cranium_scale=(1.1, 1.05, 1.0), jaw_width=1.1, chin_size=1.2, nose_size=1.2, ear_size=1.1,
                     cheekbone=1.2).resolved()
    assert k.cranium_r[0] == pytest.approx(base.cranium_r[0] * 1.1) and k.crown_y > base.crown_y
    assert k.jaw_angle[0] > base.jaw_angle[0] and k.chin_r > base.chin_r and k.nose_tip_r > base.nose_tip_r
    assert k.ear_r[1] > base.ear_r[1] and k.cheek_r[0] > base.cheek_r[0]
    assert k.resolved() == k                                           # idempotent
    with pytest.raises(ValueError):
        SculptParams(sex="other").resolved()
    # the 3 mm grid: an even column count, a coarse grid that nests
    p3 = SculptParams(voxel=0.003)
    nx, ny, nz = head_dims(p3, 0.003)
    assert nx % 2 == 0 and (nx, ny, nz) == tuple(2 * v for v in head_dims(p3, 0.006))
    assert p3.eye_open_cols == 7 and p3.eye_open_rows == 3 and SculptParams().eye_open_cols == 4


@pytest.fixture(scope="module")
def barber():
    head = SculptParams(voxel=0.003, hair_style="short", beard="mustache", skin_tone="light", hair_tone="brown")
    return build_character(CharacterParams(name="t_barber", head=head, body=BodyParams(build="lean"),
                                           outfit=Outfit(kind="apron"), scar=True, nose_tint=True))


def test_character_builds_and_stands_on_its_neck(barber):
    m = barber
    assert [q.name for q in m.parts] == ["body", "clothes", "head", "eye_l", "eye_r", "hair", "beard"]
    report = checks.run(m)
    assert report["errors"] == [], report["errors"]
    assert report["warnings"] == [], report["warnings"]
    head, body = m.part("head"), m.part("body")
    assert head.voxel_size == 0.003 and body.voxel_size == 0.01 and head.sdf is not None
    hb = head.bbox()
    head_bottom = head.origin[1] + hb[0][1] * head.voxel_size
    body_top = body.origin[1] + (body.bbox()[1][1] + 1) * body.voxel_size
    assert abs(head_bottom - body_top) < 0.011                        # the neck stands on the stub
    assert abs(head.origin[0] + head.nx * head.voxel_size / 2) < 1e-9  # still x-centred
    # no head voxel inside the body
    X, Y, Z = head.coords()
    assert float(Y[head.solid()].min()) >= body_top - 1e-6
    b = m.meta["build"]
    assert b["head"]["scar"] > 5 and b["head"]["nose_tint"] > 20 and b["head"]["beard"]["style"] == "mustache"
    assert b["clothes"]["kind"] == "apron" and b["body"]["solid"] > 40000
    ms = m.meta["mesh"]
    assert set(ms["parts"]) == {q.name for q in m.parts} and ms["quads"] > 0 and ms["solid"] == m.count_solid()
    # the scar is colour only: the material stays mirror-symmetric
    assert np.array_equal(head.material, head.material[:, :, ::-1])
    assert not np.array_equal(head.color, head.color[:, :, ::-1])
    hm = build_character(CharacterParams(name="t_barber", head=SculptParams(voxel=0.003, hair_style="short")), head_only=True)
    assert hm.name == "t_barber_head" and [q.name for q in hm.parts] == ["head", "eye_l", "eye_r", "hair"]


@pytest.mark.parametrize("name", ["barber", "knight", "monk", "drunk", "widow"])
def test_town_specs_build(name):
    path = SPECS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"town_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    m = module.build()
    assert m.name == f"town_{name}" and m.part("body") and m.part("head")
    report = checks.run(m)
    assert report["errors"] == [] and report["warnings"] == []
    expect = {"barber": "beard", "monk": "hair", "widow": "scarf", "knight": "clothes", "drunk": "hair"}[name]
    assert m.get_part(expect) is not None
    if name == "monk":
        assert m.meta["build"]["head"]["hair"]["tonsure_removed"] > 0
    if name == "knight":
        assert m.meta["build"]["head"]["scar"] > 0 and MAT_LEATHER in m.part("clothes").material_histogram()
    if name == "drunk":
        assert m.meta["build"]["head"]["nose_tint"] > 0 and m.meta["build"]["head"]["beard"]["style"] == "stubble"
