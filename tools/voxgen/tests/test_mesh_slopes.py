"""Slope mesher (``voxgen.mesh_slopes``): staircases of fine voxels become planar facets, lone steps
stay cubes, the mesh stays watertight. Synthetic numpy solids only (fast)."""
import math

import numpy as np
import pytest

from voxgen.mesh_slopes import slope_mesh


def stairs(H, nx=4):
    """(nz, ny, nx) solid whose column z holds ``H[z]`` voxels (treads facing +y, steps along z)."""
    H = np.asarray(H)
    ny = int(H.max()) + 1
    y = np.arange(ny)
    s2 = y[None, :] < H[:, None]  # (nz, ny)
    return np.broadcast_to(s2[:, :, None], (len(H), ny, nx)).copy()


def face_normals(m):
    q = m.quads
    p = m.positions.astype(np.float64)
    n = np.cross(p[q[:, 2]] - p[q[:, 0]], p[q[:, 3]] - p[q[:, 1]])
    tri = q[:, 2] == q[:, 3]
    n[tri] = np.cross(p[q[tri, 1]] - p[q[tri, 0]], p[q[tri, 2]] - p[q[tri, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def face_areas(m):
    q = m.quads
    p = m.positions.astype(np.float64)
    tri = q[:, 2] == q[:, 3]
    a = 0.5 * np.linalg.norm(np.cross(p[q[:, 2]] - p[q[:, 0]], p[q[:, 3]] - p[q[:, 1]]), axis=1)
    a[tri] = 0.5 * np.linalg.norm(np.cross(p[q[tri, 1]] - p[q[tri, 0]], p[q[tri, 2]] - p[q[tri, 0]]), axis=1)
    return a


def tilted_vertices(m, normal, tol=1e-6):
    """Vertices of the faces whose unit normal is ``normal`` (the fitted plane) and their count."""
    n = face_normals(m)
    sel = np.linalg.norm(n - np.asarray(normal, np.float64), axis=1) < tol
    return m.positions[np.unique(m.quads[sel])].astype(np.float64), int(sel.sum())


def assert_plane(m, w, r, n_faces_expected):
    """All faces tilted by the pattern (tread w, riser r along +z, treads facing +y) lie on one
    plane: their vertices satisfy ``w * y - r * z = const`` to 1e-6."""
    normal = np.array([0.0, w, -r]) / math.hypot(w, r)
    V, count = tilted_vertices(m, normal)
    assert count == n_faces_expected, count
    c = w * V[:, 1] - r * V[:, 2]
    assert np.ptp(c) < 1e-6, c
    assert m.is_closed()
    assert (face_areas(m) > 1e-9).all()


# ----------------------------------------------------------------------------- A: 1:1 and lone steps
def test_1_1_staircase_is_one_45_degree_plane():
    solid = stairs(8 - np.arange(8))  # 8 treads descending along +z
    st = {}
    m = slope_mesh(solid, step=1, min_run=2, stats=st)
    normal = np.array([0.0, 1.0, 1.0]) / math.sqrt(2)
    V, count = tilted_vertices(m, normal)
    assert count == 8 * 4 and np.ptp(V[:, 1] + V[:, 2]) < 1e-6  # y + z = const on the whole stair
    assert abs(float(V[0, 1] + V[0, 2]) - 8.5) < 1e-6
    n = face_normals(m)
    riser = np.linalg.norm(n - [0, 0, -1], axis=1) < 1e-6
    assert (m.positions[m.quads[riser]][:, :, 2] < 0.5).all()  # no riser left in the stair region
    assert st["risers_dropped"] == 7 * 4 and st["triangles"] == 14 and st["faces_dropped_crossed"] == 0
    assert m.is_closed() and (face_areas(m) > 1e-9).all()
    assert abs(m.signed_volume() - solid.sum()) < 1e-6  # the symmetric cut keeps the volume


def test_lone_step_stays_cubes():
    solid = stairs([4, 4, 4, 4, 5, 5, 5, 5])
    st = {}
    m = slope_mesh(solid, step=1, min_run=2, stats=st)
    assert st["faces_sloped"] == 0 and st["risers_dropped"] == 0 and st["triangles"] == 0
    assert st["faces_out"] == st["faces_in"] and m.is_closed()
    assert np.array_equal(m.positions, np.rint(m.positions))
    assert abs(m.signed_volume() - solid.sum()) < 1e-6


def test_two_steps_need_min_run_two():
    solid = stairs([4, 4, 4, 5, 6, 6, 6, 6])  # one 1-wide tread between two flats: 2 steps
    st = {}
    m = slope_mesh(solid, step=1, min_run=2, stats=st)
    assert st["faces_sloped"] == 4 and st["risers_dropped"] == 0 and m.is_closed()
    assert np.isclose(np.abs(m.positions[:, 1] - np.rint(m.positions[:, 1])).max(), 0.5)  # half risers
    st3 = {}
    m3 = slope_mesh(solid, step=1, min_run=3, stats=st3)
    assert st3["faces_sloped"] == 0 and np.array_equal(m3.positions, np.rint(m3.positions))


def test_bad_arguments():
    with pytest.raises(ValueError):
        slope_mesh(np.ones((2, 2, 2), bool), step=0)
    with pytest.raises(ValueError):
        slope_mesh(np.ones((2, 2, 2), bool), min_run=0)
    assert slope_mesh(np.zeros((2, 2, 2), bool)).n_faces == 0


# ----------------------------------------------------------------------------- B: 2:1, 1:2, step=2
@pytest.mark.parametrize("w,r,step", [(2, 1, 1), (1, 2, 1), (2, 2, 2), (4, 2, 2), (2, 4, 2)])
def test_ratio_staircases_are_planes(w, r, step):
    n_treads = 8
    z = np.arange(n_treads * w)
    solid = stairs(r + r * (z // w))  # ascending along +z: tread w, riser r
    st = {}
    m = slope_mesh(solid, step=step, min_run=2, stats=st)
    # the longer faces own the step: treads for w >= r, the risers (facing -z) for w < r
    n_tilted = n_treads * w * 4 if w >= r else (n_treads - 1) * r * 4 + r * 4
    assert st["coarse_phase"] == (0, 0, 0)  # the stairs are step-granular: fitted at the coarse scale
    assert_plane(m, w, r, n_tilted // step ** 2)
    assert st["faces_dropped_crossed"] == 0
    assert abs(m.signed_volume() - solid.sum()) < 1e-6


def test_fine_staircase_is_not_a_step2_pattern():
    solid = stairs(8 - np.arange(8))
    st = {}
    m = slope_mesh(solid, step=2, min_run=2, stats=st)
    assert st["coarse_phase"] is None  # not 2-granular: the fine path, scaled patterns only
    assert st["faces_sloped"] == 0 and np.array_equal(m.positions, np.rint(m.positions))


def test_granular_phase_is_found_at_any_offset():
    from voxgen.mesh_slopes import granular_phase
    z = np.arange(16)
    solid = stairs(2 + 2 * (z // 2), nx=6)
    assert granular_phase(solid, 2) == (0, 0, 0)
    shifted = np.pad(solid, ((1, 1), (1, 1), (1, 1)))  # the same blocks on an odd grid
    assert granular_phase(shifted, 2) == (1, 1, 1)
    st = {}
    m = slope_mesh(shifted, step=2, min_run=2, stats=st)
    assert st["coarse_phase"] == (1, 1, 1) and m.is_closed()
    assert_plane(m, 2, 2, 8 * 2 * 6 // 4)
    assert abs(m.signed_volume() - shifted.sum()) < 1e-6
    shifted[1, 1, 1] = False  # one fine voxel breaks the granularity
    assert granular_phase(shifted, 2) is None


# ----------------------------------------------------------------------------- C: closedness
def test_ellipsoid_is_watertight():
    nx, ny, nz = 20, 14, 24
    z, y, x = np.mgrid[0:nz, 0:ny, 0:nx]
    solid = ((x + 0.5 - nx / 2) / (nx / 2)) ** 2 + ((y + 0.5 - ny / 2) / (ny / 2)) ** 2 \
        + ((z + 0.5 - nz / 2) / (nz / 2)) ** 2 < 1.0
    half = solid[::2, ::2, ::2]  # the same blob drawn at 2-voxel steps for step=2
    solid2 = np.repeat(np.repeat(np.repeat(half, 2, 0), 2, 1), 2, 2)
    for step, s in ((1, solid), (2, solid2)):
        st = {}
        m = slope_mesh(s, step=step, min_run=2, stats=st)
        assert m.is_closed(), st
        assert (face_areas(m) > 1e-9).all()
        assert np.isfinite(m.positions).all()
        assert st["faces_sloped"] > 0 and st["risers_dropped"] > 0
        assert 0.9 * s.sum() < m.signed_volume() < 1.1 * s.sum()


def test_mixed_run_has_no_holes():
    solid = stairs([1, 2, 3, 4, 4, 5, 5, 6, 6])  # a 1:1 run followed by a 2:1 run
    st = {}
    m = slope_mesh(solid, step=1, min_run=2, stats=st)
    assert m.is_closed() and (face_areas(m) > 1e-9).all()
    assert st["risers_dropped"] == 5 * 4 and st["faces_dropped_crossed"] == 0
    n = face_normals(m)
    assert (np.linalg.norm(n - np.array([0, 1, -1]) / math.sqrt(2), axis=1) < 1e-6).sum() == 3 * 4
    assert (np.linalg.norm(n - np.array([0, 2, -1]) / math.sqrt(5), axis=1) < 1e-6).sum() == 6 * 4
