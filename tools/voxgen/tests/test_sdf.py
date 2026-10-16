import math

import numpy as np

from voxgen import Part, sdf


def grid(n=32, vs=1.0):
    return Part("g", (n, n, n), vs, origin=(-n * vs / 2,) * 3)


def test_sphere_volume_within_5_percent():
    p = grid(32)
    s = sdf.sphere(p, (0, 0, 0), 10)
    vol = int((s <= 0).sum())
    assert abs(vol - 4 / 3 * math.pi * 1000) / (4 / 3 * math.pi * 1000) < 0.05
    # distance units are voxels: the centre voxel is ~10 voxels deep
    assert -10.5 < s[16, 16, 16] < -9


def test_sphere_in_metres_converts_to_voxels():
    p = grid(32, vs=0.005)
    s = sdf.sphere(p, (0, 0, 0), 0.05)
    assert abs(int((s <= 0).sum()) - 4 / 3 * math.pi * 1000) / (4 / 3 * math.pi * 1000) < 0.05


def test_primitive_volumes():
    p = grid(40)
    assert abs(int((sdf.box(p, (0, 0, 0), (10, 10, 10)) <= 0).sum()) - 1000) <= 0
    cyl = int((sdf.cylinder(p, (0, -10, 0), (0, 10, 0), 5) <= 0).sum())
    assert abs(cyl - math.pi * 25 * 20) / (math.pi * 25 * 20) < 0.05
    cone = int((sdf.cone(p, (0, -10, 0), (0, 10, 0), 8, 2) <= 0).sum())
    assert abs(cone - math.pi * 20 / 3 * (64 + 16 + 4)) / (math.pi * 20 / 3 * 84) < 0.05
    tor = int((sdf.torus(p, (0, 0, 0), 8, 3) <= 0).sum())
    assert abs(tor - 2 * math.pi ** 2 * 8 * 9) / (2 * math.pi ** 2 * 72) < 0.06
    ell = int((sdf.ellipsoid(p, (0, 0, 0), (10, 6, 8)) <= 0).sum())
    assert abs(ell - 4 / 3 * math.pi * 480) / (4 / 3 * math.pi * 480) < 0.05
    cap = int((sdf.capsule(p, (-5, 0, 0), (5, 0, 0), 4) <= 0).sum())
    assert abs(cap - (math.pi * 16 * 10 + 4 / 3 * math.pi * 64)) / 770 < 0.06


def test_ops_and_mirror():
    p = grid(32)
    a = sdf.sphere(p, (-5, 0, 0), 6)
    b = sdf.sphere(p, (5, 0, 0), 6)
    u, i, d = sdf.union(a, b), sdf.intersect(a, b), sdf.subtract(a, b)
    assert (u <= 0).sum() > (a <= 0).sum() > (d <= 0).sum() > 0
    assert (i <= 0).sum() < (a <= 0).sum()
    su = sdf.smooth_union(a, b, 3)
    assert (su <= 0).sum() >= (u <= 0).sum()
    # mirror_x on the grid: a shape authored at +x appears on both sides, exactly symmetric
    one = sdf.sphere(sdf.mirror_x(p), (8, 0, 0), 4)
    mask = one <= 0
    assert np.array_equal(mask, mask[:, :, ::-1])
    assert mask.sum() == 2 * (sdf.sphere(p, (8, 0, 0), 4) <= 0).sum()
    # mirror_x on a field: union with the flipped copy
    f = sdf.mirror_x(sdf.sphere(p, (8, 0, 0), 4))
    assert np.array_equal(f <= 0, mask)


def test_transforms():
    p = grid(32)
    moved = sdf.sphere(sdf.translate(p, (6, 0, 0)), (0, 0, 0), 4)
    assert np.array_equal(moved <= 0, sdf.sphere(p, (6, 0, 0), 4) <= 0)
    rot = sdf.box(sdf.rotate(p, (0, 0, 90)), (0, 0, 0), (20, 4, 4))
    zz, yy, xx = np.nonzero(rot <= 0)
    assert yy.max() - yy.min() >= 18 and xx.max() - xx.min() <= 4
    big = sdf.sphere(sdf.scale(p, 2.0), (0, 0, 0), 4)
    assert abs((big <= 0).sum() - (sdf.sphere(p, (0, 0, 0), 8) <= 0).sum()) < 40


def test_noise_is_deterministic_bounded_and_symmetric_on_mirrored_grid():
    p = grid(32, vs=0.005)
    n1 = sdf.noise(p, scale=0.03, amplitude=0.005, seed=4)
    n2 = sdf.noise(p, scale=0.03, amplitude=0.005, seed=4)
    assert np.array_equal(n1, n2)
    assert n1.std() > 0.1 and np.abs(n1).max() <= 1.0 + 1e-5  # amplitude 0.005 m = 1 voxel
    assert not np.array_equal(n1, sdf.noise(p, scale=0.03, amplitude=0.005, seed=5))
    ns = sdf.noise(sdf.mirror_x(p), scale=0.03, amplitude=0.005, seed=4)
    assert np.allclose(ns, ns[:, :, ::-1])
