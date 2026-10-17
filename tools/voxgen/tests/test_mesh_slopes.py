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


# ----------------------------------------------------------------------------- D: step > 1 on a solid with fine detail
def face_orientation(m, stats):
    """(dot of every face's geometric normal with its voxel face direction, face areas)."""
    from voxgen.mesh_slopes import DIRS
    n = face_normals(m)
    d = stats["face_dir"]
    ax = np.array([DIRS[i][0] for i in d])
    sg = np.array([DIRS[i][1] for i in d])
    return n[np.arange(len(n)), ax] * sg, face_areas(m)


def assert_sound(m, stats, sliver_area=0.0):
    """Watertight, no zero-area face, every face facing the way of the voxel face it came from
    (never inverted; folded sideways only for slivers up to ``sliver_area``, the mitre tips of the
    chamfer mode)."""
    assert m.is_closed()
    dot, area = face_orientation(m, stats)
    assert (area > 1e-9).all(), int((area <= 1e-9).sum())
    assert not (dot < -1e-9).any(), int((dot < -1e-9).sum())
    side = np.abs(dot) <= 1e-9
    assert (area[side] <= sliver_area).all(), (int(side.sum()), float(area[side].max()))


def coarse_blob(step=3, detail=True):
    """A step-granular ellipsoid; with ``detail`` a 1-voxel brow block on its front and a 1-voxel
    pit on top break the granularity (the bald_sdf skull: 3-voxel steps plus a painted face)."""
    nx, ny, nz = 8, 6, 9
    z, y, x = np.mgrid[0:nz, 0:ny, 0:nx]
    s = ((x + 0.5 - nx / 2) / (nx / 2)) ** 2 + ((y + 0.5 - ny / 2) / (ny / 2)) ** 2 \
        + ((z + 0.5 - nz / 2) / (nz / 2)) ** 2 < 1.0
    f = np.repeat(np.repeat(np.repeat(s, step, 0), step, 1), step, 2)
    if detail:
        top = int(np.where(f.any(axis=(0, 2)))[0].max())
        zf = int(np.where(f[:, top, :].any(axis=1))[0].min())
        f[zf - 1, top - 4:top - 2, f.shape[2] // 2 - 2:f.shape[2] // 2 + 2] = True
        f[f.shape[0] // 2, top, f.shape[2] // 2] = False
    return f


def test_coarse_blob_with_fine_detail_is_sound():
    """Regression: the fine-grid fit of a non-granular solid at step 3 left holes and folded faces
    (bald_sdf, --facet-step 3: black triangles on the crown and the lower back); now the uniform
    blocks are fitted at the block scale and the mixed blocks stay fine cubes."""
    from voxgen.mesh_slopes import granular_phase
    solid = coarse_blob(3, detail=True)
    assert granular_phase(solid, 3) is None
    st = {}
    m = slope_mesh(solid, step=3, min_run=2, stats=st)
    assert_sound(m, st)
    assert st["coarse_phase"] is None and st["mixed_blocks"] > 0
    assert st["faces_sloped"] > 0 and st["faces_fine"] > 0 and st["faces_subdivided"] > 0
    assert 0.97 * solid.sum() < m.signed_volume() < 1.03 * solid.sum()
    st2 = {}
    m2 = slope_mesh(coarse_blob(3, detail=False), step=3, min_run=2, stats=st2)
    assert_sound(m2, st2)
    assert st2["coarse_phase"] == (0, 0, 0) and st2["mixed_blocks"] == 0 and st2["faces_fine"] == 0


def test_diagonal_staircase_keeps_its_volume():
    """A corner cell of two 1:1 runs (along x and z): the wall between the treads is dropped once,
    by the earlier-ranked sheet; welding it from both sheets chained whole staircases into one
    vertex (the crown corners of bald_sdf)."""
    N = 5
    x = np.arange(N)[None, :]
    z = np.arange(N)[:, None]
    top = 3 + x + z
    y = np.arange(int(top.max()) + 1)[None, :, None]
    solid = y < top[:, None, :]
    st = {}
    m = slope_mesh(solid, step=1, min_run=2, stats=st)
    assert_sound(m, st)
    assert abs(m.signed_volume() - solid.sum()) < 0.5


@pytest.mark.skipif(not __import__("pathlib").Path("godot/assets/models/bald_sdf.vxm").exists()
                    and not __import__("pathlib").Path("../../godot/assets/models/bald_sdf.vxm").exists(),
                    reason="bald_sdf.vxm is generated (make assets)")
def test_bald_sdf_skull_is_sound_at_step_3():
    from pathlib import Path
    from voxgen import io_vxm, materials as M
    from voxgen.mesh_slopes import chamfer_mesh
    path = next(p for p in (Path("godot/assets/models/bald_sdf.vxm"), Path("../../godot/assets/models/bald_sdf.vxm"))
                if p.exists())
    part = io_vxm.read(path).parts[0]
    solid = part.material != M.MAT_AIR
    st = {}
    assert_sound(slope_mesh(solid, step=3, min_run=2, stats=st), st)
    assert st["faces_sloped"] > 100
    st = {}
    m = chamfer_mesh(solid, step=3, min_run=2, stats=st)
    assert m.is_closed() and st["chamfer_edges"] > 1000
    dot, area = face_orientation(m, st)
    assert not (dot < -1e-9).any()
    assert area[np.abs(dot) <= 1e-9].sum() < 0.005 * area.sum()  # sideways slivers at the mitre tips only


# ----------------------------------------------------------------------------- E: chamfer mode
def stairs_on_base(H, nx=8):
    """:func:`stairs` whose outer walls are no step faces: heights 4 + 3k, flats of 4 at the ends."""
    return stairs(H, nx)


def area_of(m, normal, tol=1e-6):
    n = face_normals(m)
    sel = np.linalg.norm(n - np.asarray(normal, np.float64), axis=1) < tol
    return float(face_areas(m)[sel].sum()), sel


def test_chamfer_1_1_staircase_of_3_steps_is_one_45_degree_plane():
    from voxgen.mesh_slopes import chamfer_mesh
    solid = stairs_on_base([4] * 4 + [7] * 3 + [10] * 3 + [13] * 3 + [16] * 4)
    st = {}
    m = chamfer_mesh(solid, step=3, min_run=2, stats=st)
    assert_sound(m, st)
    assert st["chamfer_edges"] == 8 * 8  # 4 risers x (concave bottom + convex top) x 8 unit edges
    a45, sel = area_of(m, np.array([0.0, 1.0, -1.0]) / math.sqrt(2))
    V = m.positions[np.unique(m.quads[sel])].astype(np.float64)
    assert np.ptp(V[:, 1] - V[:, 2]) < 1e-6 and abs(float(V[0, 1] - V[0, 2]) - 1.5) < 1e-6  # y - z = 1.5
    assert abs(a45 - 12 * math.sqrt(2) * 8) < 1e-6  # from z = 2.5 to 14.5: treads and risers all consumed
    flat, fsel = area_of(m, [0.0, 1.0, 0.0])
    P = m.positions[m.quads[fsel]][:, :, 2]
    assert ((P <= 2.5 + 1e-6) | (P >= 14.5 - 1e-6)).all()  # no flat left between the chamfers


def test_chamfer_2_1_staircase_alternates_45_degree_quads_and_flats():
    from voxgen.mesh_slopes import chamfer_mesh
    solid = stairs_on_base([4] * 4 + [7] * 6 + [10] * 6 + [13] * 6 + [16] * 4)
    st = {}
    m = chamfer_mesh(solid, step=3, min_run=2, stats=st)
    assert_sound(m, st)
    a45, sel = area_of(m, np.array([0.0, 1.0, -1.0]) / math.sqrt(2))
    assert abs(a45 - 4 * 3 * math.sqrt(2) * 8) < 1e-6  # 4 risers, each 1.5 + 1.5 of 45 degrees
    V = m.positions[m.quads[sel]].astype(np.float64)
    planes = sorted(set(np.round(V[:, :, 1] - V[:, :, 2], 6).ravel().tolist()))
    assert planes == [-7.5, -4.5, -1.5, 1.5]
    flat, fsel = area_of(m, [0.0, 1.0, 0.0])
    assert abs(flat - (2.5 + 3 + 3 + 3 + 2.5) * 8) < 1e-6  # the 6-wide treads keep 3 of flat
    assert abs(m.signed_volume() - solid.sum()) < 1e-6  # carves and fills cancel


def test_chamfer_1_voxel_step_stays_cubes_at_step_3():
    from voxgen.mesh_slopes import chamfer_mesh
    solid = stairs_on_base([6] * 5 + [7] * 5)
    st = {}
    m = chamfer_mesh(solid, step=3, min_run=2, stats=st)
    assert st["chamfer_edges"] == 0 and st["faces_sloped"] == 0
    assert np.array_equal(m.positions * 2, np.rint(m.positions * 2)) and m.is_closed()
    assert abs(m.signed_volume() - solid.sum()) < 1e-6


def test_chamfer_narrow_tread_keeps_both_edges_sharp():
    """A 2-voxel tread between two 3-voxel risers cannot lose 1.5 voxels to each edge: both stay
    sharp, the risers' other edges are chamfered."""
    from voxgen.mesh_slopes import chamfer_mesh
    solid = stairs_on_base([4] * 5 + [7] * 2 + [10] * 5)
    st = {}
    m = chamfer_mesh(solid, step=3, min_run=2, stats=st)
    assert_sound(m, st)
    assert st["chamfer_edges"] == 2 * 8
    flat, fsel = area_of(m, [0.0, 1.0, 0.0])
    P = m.positions[m.quads[fsel]].astype(np.float64)
    tread = np.isclose(P[:, :, 1], 7.0).all(1)
    assert abs(face_areas(m)[fsel][tread].sum() - 2 * 8) < 1e-6


def test_chamfer_block_on_a_base_has_chamfers_fillets_and_mitred_corners():
    """A 5 x 3 x 5 block on a thick base: its top edges are carved (the walls are 3 tall), its
    bottom edges filled, two fillets meet at every bottom corner in a mitre; the vertical edges
    (faces 5 wide) stay sharp."""
    from voxgen.mesh_slopes import chamfer_mesh
    base = np.zeros((14, 9, 14), bool)
    base[:, :4, :] = True
    block = base.copy()
    block[4:9, 4:7, 4:9] = True
    st = {}
    m = chamfer_mesh(block, step=3, min_run=2, stats=st)
    assert_sound(m, st, sliver_area=0.1)
    assert st["chamfer_edges"] == 8 * 5 and st["chamfer_cushion_cells"] > 0 and st["checkerboards_filled"] == 0
    n = np.round(face_normals(m), 3)
    kinds = {tuple(v) for v in n}
    for a in (0, 2):
        for sgn in (1, -1):
            v = np.zeros(3)
            v[a], v[1] = sgn * 0.707, 0.707
            assert tuple(v) in kinds  # the 45 degree chamfers of the top edges and the fillets
    a45 = face_areas(m)[np.abs(n[:, 1] - 0.707) < 1e-3].sum()
    # 8 edges of 5 with a 1.5 wide 45 degree strip each, minus the corner cells (claimed by two
    # chamfers: facets sloped along both axes) and the mitre tetrahedra
    assert 8 * (5 - 3) * 1.5 * math.sqrt(2) < a45 < 8 * 5 * 1.5 * math.sqrt(2)


def test_chamfer_cube_of_one_step_is_sound():
    """A 3-cube on a base: every face is consumed by its chamfers (a degenerate case: the fillets
    lose their walls and are dropped); the mesh stays sound."""
    from voxgen.mesh_slopes import chamfer_mesh
    base = np.zeros((12, 9, 12), bool)
    base[:, :4, :] = True
    cube = base.copy()
    cube[4:7, 4:7, 4:7] = True
    st = {}
    m = chamfer_mesh(cube, step=3, min_run=2, stats=st)
    assert_sound(m, st, sliver_area=0.1)
    assert st["chamfer_edges"] == 12 * 3
