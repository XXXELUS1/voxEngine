"""The sculpted head (voxgen.sculpthead): every eyes / hair / grid combination builds as a
multi-part model, the form is mirror-exact and one piece, the checks pass (no floating, honest
layers, bone never on the surface), the eyes and the hair are there, the sunglasses are one
piece on the face, the coarse grid is whole 1 cm blocks, the specs build."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from voxgen import checks, mesh, sdfield
from voxgen.body import exposed
from voxgen.materials import MAT_AIR, MAT_BONE, MAT_CARTILAGE, MAT_EYE, MAT_GLASS, MAT_HAIR, MAT_SKIN
from voxgen.sculpthead import (EYES, GRIDS, HAIR_STYLES, PRESETS, SculptParams, build_sculpt_head, form_field,
                               head_part, sculpt_form)


@pytest.fixture(scope="module")
def balls_quiff():
    return build_sculpt_head(SculptParams.preset_params("sculpt", eyes="balls", hair_style="quiff"), name="t_sculpt")


@pytest.fixture(scope="module")
def shades_quiff():
    return build_sculpt_head(SculptParams.preset_params("sculpt", eyes="shades", hair_style="quiff"), name="t_shades")


def test_builds_and_passes_checks(balls_quiff):
    m = balls_quiff
    assert [q.name for q in m.parts] == ["head", "eye_l", "eye_r", "hair"]
    head = m.part("head")
    assert head.voxel_size == 0.005 and head.nx % 2 == 0 and head.origin[0] == pytest.approx(-head.nx * 0.0025)
    hist = head.material_histogram()
    for mat in (MAT_SKIN, MAT_BONE, MAT_CARTILAGE):
        assert hist.get(mat, 0) > 0, mat
    assert MAT_EYE not in hist and MAT_HAIR not in hist
    assert set(m.part("hair").material_histogram()) == {MAT_HAIR}
    assert set(m.part("eye_l").material_histogram()) == {MAT_EYE}
    report = checks.run(m)
    assert report["errors"] == [], report["errors"]
    assert report["warnings"] == [], report["warnings"]
    b = m.meta["build"]
    assert b["form"]["components"] == 1 and b["form"]["grid"] == "fine"
    assert b["face"]["mouth"] == 20 and b["face"]["lower_lip"] == 8 and b["face"]["nostril"] == 2
    assert b["face"]["brow"] == 24 and b["face"]["lid"] == 8 and b["face"]["band"] > 0
    assert b["hair"]["pocket"] > 0 and b["hair"]["quiff"]["spikes"] == 4
    assert m.meta["seams"]["hair"]["overlap"] == 0 and m.meta["seams"]["hair"]["contact"] > 0
    assert set(m.meta["mesh"]["parts"]) == {"head", "eye_l", "eye_r", "hair"} and m.meta["mesh"]["quads"] > 0


def test_symmetric_layered_and_in_frame(balls_quiff):
    head = balls_quiff.part("head")
    mat = head.material
    assert np.array_equal(mat, mat[:, :, ::-1])
    solid = mat != MAT_AIR
    ex = exposed(solid)
    assert not (ex & (mat == MAT_BONE)).any()                     # bone never on the surface
    assert (ex & (mat == MAT_SKIN)).sum() > 0.9 * ex.sum()        # the surface is skin
    # the chin's bottom near y = 0 (the rows below are the neck), the face toward -Z, the crown near crown_y
    p = SculptParams()
    X, Y, Z = head.coords()
    front = solid & (np.abs(X) < 0.02)
    assert float(Z[front].min()) < p.nose_tip[2]                  # the nose tip is the front-most point
    assert abs(float(Y[solid & (np.abs(X) < 0.03) & (Z < -0.05)].min())) < 0.012
    assert abs(float(Y[solid].max()) - p.crown_y) < 0.01
    # the nose and the ears are cartilage inside, nothing else
    cart = mat == MAT_CARTILAGE
    assert (np.abs(X[cart]) < 0.03).sum() > 0 and (np.abs(X[cart]) > p.ear_inner_x).sum() > 0
    assert not (cart & (np.abs(X) >= 0.03) & (np.abs(X) <= p.ear_inner_x)).any()


def test_form_is_smooth_sdf():
    """The field is a signed distance (negative inside, |gradient| ~ 1 away from the blends) and
    the surface has no voxel-size crumbs: one component on both grids."""
    p = SculptParams()
    part = head_part(p, 0.005)
    f = form_field(part, p)
    assert f.dtype == np.float32 and (f <= 0).sum() > 20000
    gz, gy, gx = np.gradient(f)
    g = np.sqrt(gx * gx + gy * gy + gz * gz)
    near = (np.abs(f) < 1.0) & (np.abs(f) > 0.0)
    assert 0.7 < float(np.median(g[near])) < 1.3
    for grid in GRIDS:
        _, solid, field, info = sculpt_form(SculptParams(grid=grid))
        assert info["components"] == 1 and info["crumbs_removed"] == 0
        assert np.array_equal(solid, solid[:, :, ::-1])
        assert (field is None) == (grid == "coarse")
    assert p.hollow_r == 0.0                                      # no cheek hollow / nasolabial creases


def test_sdf_field_is_the_exact_form(balls_quiff):
    """The head carries the exact signed distance of the form (int8 sixteenths of a voxel): the
    mask is exactly ``sdf < 0``, it round-trips through .vxm, it matches ``form_field`` away from
    the carved pockets, and the Hermite nets read it (a vertex of the cheek sits on the zero set)."""
    head = balls_quiff.part("head")
    assert head.sdf is not None and head.sdf.dtype == np.int8 and head.fields[-1] == "sdf"
    assert sdfield.is_consistent(head.sdf, head.solid())
    p = SculptParams()
    f = form_field(head, p)
    carved = (f <= 0) & ~head.solid()
    assert carved.sum() == balls_quiff.meta["build"]["sdf"]["carved"] > 0
    q = sdfield.dequantize(head.sdf)
    far_from_carve = np.sqrt(sdfield.edt_sq(carved)) > 5.0          # the carve distance there is > 4.5
    diff = np.abs(q - f)[(np.abs(f) < 4.0) & far_from_carve]
    assert diff.max() <= 1.5 / 16 + 1e-6                         # quantization + the consistency clamp
    m = mesh.surface_nets(head, balls_quiff.palette)
    assert m.hermite and m.vertex_normals is not None and m.n_faces > 0
    # no vertex sits farther than a quarter voxel from the exact zero set (sampled trilinearly)
    v = (m.positions - np.asarray(head.origin, np.float32)) / head.voxel_size - 0.5   # voxel-centre coords
    v = np.clip(v, 0, np.array(head.shape[::-1]) - 1.001)
    i0 = np.floor(v).astype(int)
    fr = v - i0
    val = np.zeros(len(v), np.float32)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = (fr[:, 0] if dx else 1 - fr[:, 0]) * (fr[:, 1] if dy else 1 - fr[:, 1]) * (fr[:, 2] if dz else 1 - fr[:, 2])
                val += w * q[i0[:, 2] + dz, i0[:, 1] + dy, i0[:, 0] + dx]
    cheek = (np.abs(m.positions[:, 0]) > 0.04) & (m.positions[:, 1] > 0.06) & (m.positions[:, 1] < 0.12) \
        & (m.positions[:, 2] < -0.03)
    assert cheek.sum() > 50 and np.abs(val[cheek]).max() < 0.25


def test_coarse_grid_is_whole_blocks():
    m = build_sculpt_head(SculptParams.preset_params("sculpt", eyes="dots", hair_style="bald", grid="coarse"), name="t_coarse")
    assert [q.name for q in m.parts] == ["head"] and m.meta["grid"] == "coarse"
    solid = m.part("head").solid()
    coarse = solid[::2, ::2, ::2]
    assert np.array_equal(solid, coarse.repeat(2, 0).repeat(2, 1).repeat(2, 2))
    report = checks.run(m)
    assert report["errors"] == [] and report["warnings"] == []
    assert m.meta["build"]["face"]["eye_dots"] == 16 and "brow" in m.meta["build"]["face"]


def test_shades_part(shades_quiff):
    m = shades_quiff
    assert [q.name for q in m.parts] == ["head", "shades", "hair"]
    shades = m.part("shades")
    assert set(shades.material_histogram()) == {MAT_GLASS} and shades.bone == "head"
    n = shades.count_solid()
    assert n == m.meta["build"]["shades"] > 100
    labels, count = checks.label_components(shades.solid())
    assert count == 1
    assert np.array_equal(shades.material, shades.material[:, :, ::-1])
    seams = m.meta["seams"]["shades"]
    assert seams["overlap"] == 0 and seams["contact"] > 0.4 * n
    f = m.meta["build"]["face"]
    assert "brow" not in f and f["eye_dots"] == 16 and f["band"] > 0
    report = checks.run(m)
    assert report["errors"] == [] and report["warnings"] == []


@pytest.mark.parametrize("eyes,hair,preset", [("balls", "short", "sculpt"), ("dots", "slick", "sculpt_broad")])
def test_other_combinations(eyes, hair, preset):
    m = build_sculpt_head(SculptParams.preset_params(preset, eyes=eyes, hair_style=hair), name=f"t_{eyes}_{hair}")
    assert m.meta["preset"] == preset and m.meta["eyes"] == eyes and m.meta["hair_style"] == hair
    assert checks.run(m)["errors"] == []
    hair_part = m.part("hair")
    tones = {m.palette.name_of(int(c)) for c in np.unique(hair_part.color[hair_part.solid()])}
    assert tones == {"hair", "hair_light", "hair_deep"}
    assert m.meta["seams"]["hair"]["overlap"] == 0


def test_params_validation():
    assert PRESETS == ("sculpt", "sculpt_broad") and EYES == ("balls", "dots", "shades")
    assert HAIR_STYLES == ("quiff", "short", "slick", "bald") and GRIDS == ("fine", "coarse")
    with pytest.raises(ValueError):
        SculptParams.preset_params("cube")
    with pytest.raises(ValueError):
        build_sculpt_head(SculptParams(eyes="laser"))
    with pytest.raises(ValueError):
        build_sculpt_head(SculptParams(hair_style="bangs"))
    with pytest.raises(ValueError):
        build_sculpt_head(SculptParams(grid="medium"))


def test_specs_build():
    specs = Path(__file__).resolve().parents[1] / "specs" / "chars"
    names = sorted(q.stem for q in specs.glob("sculpt_*.py"))
    assert names == ["sculpt_balls_quiff", "sculpt_balls_quiff_coarse", "sculpt_balls_short", "sculpt_dots_slick",
                     "sculpt_shades_quiff"]
    for name in ("sculpt_balls_quiff_coarse", "sculpt_dots_slick"):
        spec = importlib.util.spec_from_file_location(name, specs / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model = mod.build()
        assert model.name == name and model.meta["generator"] == "sculpthead"
        assert model.meta["eyes"] == name.split("_")[1] and model.meta["hair_style"] == name.split("_")[2]
        assert model.meta["grid"] == ("coarse" if name.endswith("_coarse") else "fine")
        assert checks.run(model)["errors"] == []
