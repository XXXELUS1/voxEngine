"""voxgen.sdfield: exact EDT, signed distance of binary volumes, the int8 quantization that stays
consistent with the mask, carve / add updates."""
import numpy as np

from voxgen import Part, sdf, sdfield
from voxgen.materials import MAT_SKIN


def test_edt_is_exact_against_brute_force():
    rng = np.random.default_rng(1)
    m = rng.random((6, 7, 9)) < 0.12
    d2 = sdfield.edt_sq(m)
    pts = np.argwhere(m)
    g = np.stack(np.meshgrid(*[np.arange(k) for k in m.shape], indexing="ij"), -1).reshape(-1, 3)
    bf = ((g[:, None, :] - pts[None, :, :]) ** 2).sum(-1).min(1).reshape(m.shape)
    assert np.array_equal(d2, bf.astype(np.float32))
    assert np.isinf(sdfield.edt_sq(np.zeros((3, 3, 3), bool))).all()


def test_signed_distance_of_a_sphere_mask():
    n, vs, r = 32, 0.005, 0.05
    p = Part("s", (n, n, n), vs, origin=(-n * vs / 2,) * 3)
    f = sdf.sphere(p, (0, 0, 0), r)
    solid = f <= 0
    d = sdfield.signed_distance_vox(solid)
    assert d.dtype == np.float32 and (d[solid] < 0).all() and (d[~solid] > 0).all()
    near = np.abs(f) < 3.0
    assert np.abs(d - f)[near].max() < 0.75           # stairs: half a voxel plus the face offset
    assert np.abs(d - f)[near].mean() < 0.3
    dm = sdfield.distance_field(solid, vs)
    assert dm.dtype == np.float32 and abs(float(dm.min()) / vs + 127 / 16) < 1e-4   # saturates at the int8 range
    assert np.abs(dm / vs - f)[near].mean() < 0.3     # the blur softens, it does not drift


def test_quantize_is_consistent_and_bounded():
    rng = np.random.default_rng(2)
    solid = rng.random((5, 6, 7)) < 0.5
    d = rng.normal(0, 3, solid.shape).astype(np.float32) * 4   # wildly inconsistent on purpose
    q = sdfield.quantize(d, solid)
    assert q.dtype == np.int8 and sdfield.is_consistent(q, solid)
    assert q.min() >= -127 and q.max() <= 127 and (q[solid] <= -1).all() and (q[~solid] >= 0).all()
    assert (sdfield.quantize(np.full(solid.shape, 0.03, np.float32), solid)[~solid] == 0).all()
    assert np.allclose(sdfield.dequantize(sdfield.quantize(np.full((2, 2, 2), -1.25), np.ones((2, 2, 2), bool))), -1.25)


def test_update_sdf_carve_and_add():
    n, vs = 24, 0.005
    p = Part("s", (n, n, n), vs, origin=(-n * vs / 2,) * 3)
    f = sdf.sphere(p, (0, 0, 0), 0.04)
    p.material[f <= 0] = MAT_SKIN
    p.sdf = sdfield.quantize(f, p.solid())
    before = p.solid()
    X, Y, Z = p.coords()
    p.clear(Y > 0.02)                                   # carve the cap through Part.clear
    after = p.solid()
    assert after.sum() < before.sum() and sdfield.is_consistent(p.sdf, after)
    q = sdfield.dequantize(p.sdf)
    # the carved cap's face is the new surface: the row under the cut reads about -0.5, the row above +0.5
    row = lambda y: int(round((y - p.origin[1]) / vs - 0.5))
    assert abs(float(np.median(q[:, row(0.0175), :][after[:, row(0.0175), :]])) + 0.5) < 0.2
    assert abs(float(np.median(q[:, row(0.0225), :][before[:, row(0.0225), :]])) - 0.5) < 0.2
    assert np.abs(q - f)[(Y < -0.01) & (np.abs(f) < 3)].max() <= 1.5 / 16 + 1e-6   # untouched far from the cut
    p.paint(Y > 0.02, MAT_SKIN, 1)                       # add it back
    assert sdfield.is_consistent(p.sdf, p.solid())
    q8 = p.sdf
    assert sdfield.update_sdf(q8, p.solid(), p.solid()) is q8           # nothing changed: untouched
