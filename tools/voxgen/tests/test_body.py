import numpy as np

from helpers import layered_ball, palette
from voxgen import Model, Part, body, sdf
from voxgen.materials import MAT_AIR, MAT_BONE, MAT_FAT, MAT_HAIR, MAT_MUSCLE, MAT_SKIN


def test_fill_layers_order_outer_skin_inner_fat_bone_core():
    model, p = layered_ball(n=40, r=0.08)
    solid = p.solid()
    # every voxel touching air is skin
    assert np.count_nonzero(body.exposed(solid) & (p.material != MAT_SKIN)) == 0
    # along the +x axis from the centre: bone ... muscle, fat, skin, air
    row = p.material[20, 20, 20:]
    names = [int(v) for v in row]
    assert names[0] == MAT_BONE
    last_solid = max(i for i, v in enumerate(names) if v != MAT_AIR)
    assert names[last_solid] == MAT_SKIN
    assert names[last_solid - 1] == MAT_FAT
    assert MAT_MUSCLE in names[: last_solid - 1]
    # skin is a watertight 1-voxel shell: every skin voxel touches air (26-neighbourhood)
    skin = p.material == MAT_SKIN
    air_nb = body.neighbour_count(~solid, 26)
    assert np.count_nonzero(skin & (air_nb == 0)) == 0
    assert skin.sum() < 2.5 * body.exposed(solid).sum()
    # bone core never replaces the outer layer
    assert np.count_nonzero(body.exposed(solid) & (p.material == MAT_BONE)) == 0
    # colours come from the palette entries named like the materials
    assert p.color[20, 20, 20] == model.palette.index("bone")


def test_grow_hair_only_outside_skin_and_attached():
    model, p = layered_ball(n=32, r=0.06)
    before = p.material.copy()
    n = body.grow_hair(p, sdf.half_space(p, (0, 0.02, 0), (0, -1, 0)), (2, 5), "hair", direction="normal", seed=3)
    assert n > 0
    hair = p.material == MAT_HAIR
    assert hair.sum() == n
    # hair only went into former air voxels; nothing else changed
    assert np.all(before[hair] == MAT_AIR)
    assert np.array_equal(p.material[~hair], before[~hair])
    # every hair voxel is attached to skin or hair, and heights are 1..5
    h = body.hair_height(p)
    assert h[hair].min() >= 1 and h[hair].max() <= 5
    assert np.count_nonzero(hair & (h < 0)) == 0
    # fixed direction grows straight columns up
    model2, p2 = layered_ball(n=32, r=0.06)
    body.grow_hair(p2, sdf.half_space(p2, (0, 0.045, 0), (0, -1, 0)), 3, "hair", direction=(0, 1, 0), jitter=0.0)
    hz, hy, hx = np.nonzero(p2.material == MAT_HAIR)
    assert hy.min() > 16


def test_grow_hair_density_and_stop_at_solid():
    model, p = layered_ball(n=32, r=0.06)
    n_full = body.grow_hair(p.copy(), sdf.half_space(p, (0, 0.03, 0), (0, -1, 0)), 3, "hair", density=1.0, jitter=0)
    n_half = body.grow_hair(p.copy(), sdf.half_space(p, (0, 0.03, 0), (0, -1, 0)), 3, "hair", density=0.5, jitter=0, seed=2)
    assert 0 < n_half < n_full
    # hair growing downward from the top of the ball into the ball is blocked by the skin
    n_down = body.grow_hair(p.copy(), sdf.half_space(p, (0, 0.03, 0), (0, -1, 0)), 3, "hair", direction=(0, -1, 0), jitter=0)
    assert n_down == 0


def test_tint_and_paint():
    model, p = layered_ball(n=24, r=0.04)
    pal = model.palette
    RED = pal.add("red", (200, 50, 50))
    n = body.tint_region(p, sdf.sphere(p, (0.04, 0, 0), 0.015), RED)
    assert n > 0
    assert np.all(p.material[p.color == RED] == MAT_SKIN)
    k = body.paint(p, sdf.sphere(p, (0, 0, 0), 0.02), MAT_MUSCLE, "muscle", only=[MAT_BONE])
    assert k > 0 and np.count_nonzero(p.material == MAT_BONE) < 10
