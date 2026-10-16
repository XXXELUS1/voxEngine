import numpy as np

from helpers import layered_ball
from voxgen import checks, sdf
from voxgen.materials import MAT_BONE, MAT_FAT, MAT_HAIR, MAT_SKIN


def test_clean_model_has_no_errors():
    model, p = layered_ball(hair=True)
    rep = checks.run(model)
    assert rep["errors"] == []
    assert rep["warnings"] == []
    st = rep["stats"]["parts"]["ball"]
    assert st["floating"]["components"] == 1
    assert st["symmetry_x"]["asymmetric_pct"] == 0
    assert st["hollow"]["pockets"] == 0
    assert st["layer_order"]["hair_unattached"] == 0


def test_floating_blob_detected():
    model, p = layered_ball()
    p.paint(sdf.sphere(p, (0.07, 0.07, 0.0), 0.006), MAT_SKIN, model.palette.index("skin"))  # loose blob
    p.material[0, 0, 0] = MAT_SKIN
    p.color[0, 0, 0] = 1  # a single speck
    rep = checks.run(model)
    st = rep["stats"]["parts"]["ball"]["floating"]
    assert st["components"] == 3 and st["floating"] == 2
    assert any("floating component" in e for e in rep["errors"])     # the blob (> 2 voxels)
    assert any("floating component of 1" in w for w in rep["warnings"])  # the speck


def test_symmetry_check():
    model, p = layered_ball()
    assert checks.run(model, which=["symmetry_x"])["errors"] == []
    p.paint(sdf.sphere(p, (0.045, 0.02, 0), 0.012), MAT_SKIN, 1)  # bump on one side only
    rep = checks.run(model, which=["symmetry_x"])
    pct = rep["stats"]["parts"]["ball"]["symmetry_x"]["asymmetric_pct"]
    assert pct > 0.5
    assert rep["warnings"] or rep["errors"]
    # hair asymmetry is ignored
    model2, p2 = layered_ball(hair=True, seed=7)
    assert checks.run(model2, which=["symmetry_x"])["stats"]["parts"]["ball"]["symmetry_x"]["asymmetric_pct"] == 0


def test_hollow_layer_order_budget_palette():
    model, p = layered_ball()
    p.clear(sdf.sphere(p, (0, 0, 0), 0.01))  # enclosed air pocket
    exposed_before = p.material.copy()
    p.material[p.material == MAT_SKIN] = MAT_FAT  # strip the skin: fat exposed
    p.material[16, 16, 16 + 12] = MAT_HAIR  # a hair voxel floating in air, away from skin
    p.color[16, 16, 16 + 12] = 1
    p.budget = 100
    rep = checks.run(model)
    msgs = " | ".join(rep["errors"] + rep["warnings"])
    assert "air pocket" in msgs
    assert "fat voxels are exposed" in msgs
    assert "hair voxels are not 6-adjacent" in msgs
    assert "exceed the budget" in msgs
    # palette: colour index outside the palette
    p.color[p.material == MAT_FAT] = 200
    rep = checks.run(model, which=["palette_size"])
    assert any("outside the palette" in e for e in rep["errors"])


def test_label_components_basic():
    m = np.zeros((5, 5, 5), bool)
    m[0, 0, 0:3] = True
    m[4, 4, 4] = True
    m[2, 2, 2] = True
    m[2, 3, 2] = True
    labels, n = checks.label_components(m)
    assert n == 3
    assert labels[0, 0, 0] == labels[0, 0, 2] == 1  # largest first
    assert labels[2, 2, 2] == labels[2, 3, 2]
    assert labels[4, 4, 4] != labels[2, 2, 2]
