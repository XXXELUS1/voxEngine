import numpy as np
import pytest

from helpers import layered_ball
from voxgen import mesh
from voxgen.materials import MAT_HAIR


def test_culled_quads_volume_equals_voxel_count():
    model, p = layered_ball(n=24, r=0.05)
    m = mesh.culled_quads(p, model.palette)
    expected = p.count_solid() * p.voxel_size ** 3
    assert m.n_faces > 0 and m.is_closed()
    assert abs(m.signed_volume() - expected) <= 1e-6 * expected
    assert m.positions.dtype == np.float32 and m.quads.dtype == np.int32
    assert m.face_colors.shape == (m.n_faces, 4) and m.face_colors.dtype == np.uint8
    assert m.corner_ao.shape == (m.n_faces, 4) and (m.corner_ao > 0).all() and (m.corner_ao <= 1).all()
    assert len(m.face_material) == m.n_faces and not m.face_smooth.any()
    # positions are metres in local space: the ball is centred on the origin
    assert np.allclose(m.positions.mean(0), 0.0, atol=p.voxel_size)


def test_surface_nets_closed_and_volume_close():
    model, p = layered_ball(n=24, r=0.05)
    m = mesh.surface_nets(p, model.palette)
    assert m.n_faces > 0 and m.is_closed() and m.face_smooth.all()
    assert m.vertex_colors is not None and m.vertex_colors.shape == (len(m.positions), 4)
    expected = p.count_solid() * p.voxel_size ** 3
    assert 0.85 * expected < m.signed_volume() < 1.05 * expected


def test_surface_nets_free_clamp_restores_volume_and_inflate_grows():
    model, p = layered_ball(n=24, r=0.05)
    expected = p.count_solid() * p.voxel_size ** 3
    raw = mesh.surface_nets(p, model.palette, smooth_passes=0)
    cell = mesh.surface_nets(p, model.palette, smooth_passes=3, clamp="cell")
    free = mesh.surface_nets(p, model.palette, smooth_passes=3, clamp="free")
    assert cell.is_closed() and free.is_closed() and len(free.positions) == len(raw.positions)
    # cell-clamped Laplacian shrinks the ball, the free clamp restores the unsmoothed volume
    assert cell.signed_volume() < 0.98 * raw.signed_volume()
    assert abs(free.signed_volume() - raw.signed_volume()) < 0.005 * raw.signed_volume()
    assert 0.95 * expected < free.signed_volume() < 1.05 * expected
    # free vertices never stray far from their cell (half a voxel of play on a 10-voxel ball)
    assert np.abs(free.positions - cell.positions).max() < 0.6 * p.voxel_size
    # inflate pushes the surface outward along the normals: the radius grows by about that much
    puffed = mesh.surface_nets(p, model.palette, smooth_passes=3, clamp="free", inflate=0.5)
    r_free, r_puff = np.linalg.norm(free.positions, axis=1), np.linalg.norm(puffed.positions, axis=1)
    assert 0.4 * p.voxel_size < (r_puff - r_free).mean() < 0.6 * p.voxel_size
    with pytest.raises(ValueError):
        mesh.surface_nets(p, model.palette, clamp="loose")


def test_surface_nets_default_smooths_hair():
    model, p = layered_ball(n=24, r=0.05, hair=True)
    m = mesh.surface_nets(p, model.palette)
    assert m.n_faces > 0 and m.face_smooth.all() and not (~m.face_smooth).any()
    assert (m.face_material == MAT_HAIR).any() and m.signed_volume() > 0
    # thin hair strands give naive surface nets edges shared by 4 quads (not a 2-manifold), but no holes
    q = m.quads
    e = np.stack([np.concatenate([q[:, i] for i in range(4)]), np.concatenate([q[:, (i + 1) % 4] for i in range(4)])], 1)
    e.sort(axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    assert (counts % 2 == 0).all() and (counts >= 2).all()


def test_surface_nets_keeps_hair_cubic():
    model, p = layered_ball(n=24, r=0.05, hair=True)
    m = mesh.surface_nets(p, model.palette, keep_cubic_materials=mesh.CUBIC_MATERIALS)
    assert m.n_faces > 0
    assert (m.face_material[~m.face_smooth] == MAT_HAIR).all()
    assert (m.face_material[m.face_smooth] != MAT_HAIR).all()
    assert m.face_smooth.any() and (~m.face_smooth).any()


def _edge_weight(m, ew, c0, c1, part):
    """Weight of the edge between the vertices at integer corners c0 and c1."""
    corners = np.rint((m.positions - np.asarray(part.origin, np.float32)) / part.voxel_size).astype(int)
    v0 = int(np.flatnonzero((corners == c0).all(1))[0])
    v1 = int(np.flatnonzero((corners == c1).all(1))[0])
    hit = (ew.edges == (min(v0, v1), max(v0, v1))).all(1)
    assert hit.sum() == 1, (c0, c1)
    return float(ew.weights[hit][0])


def test_step_edge_weights_marks_steps_not_cubes_and_min_run_skips_lone_bumps():
    from voxgen import Part
    from voxgen.materials import MAT_SKIN
    # a 4x4x4 cube: no step edge; "all" weights its 12 x 4 unit edges
    cube = Part("cube", (6, 6, 6), 0.01)
    cube.material[1:5, 1:5, 1:5] = MAT_SKIN
    m = mesh.culled_quads(cube, [(200, 150, 120)], jitter=0.0)
    ew = mesh.step_edge_weights(m, cube)
    assert ew.weights.sum() == 0 and ew.stats["edges_90"] == 48 and ew.stats["edges_nonmanifold"] == 0
    assert mesh.step_edge_weights(m, cube, mode="all").weights.sum() == 48
    # a staircase of three 1-voxel steps (x 1..3 holds columns of height 3, 2, 1; 4 voxels wide in z)
    st = Part("stairs", (5, 5, 6), 0.01)
    for x, h in ((1, 3), (2, 2), (3, 1)):
        st.material[1:5, 1:1 + h, x] = MAT_SKIN
    m = mesh.culled_quads(st, [(200, 150, 120)], jitter=0.0)
    ew = mesh.step_edge_weights(m, st)
    assert ew.stats["edges_step"] > 0
    for z in range(1, 5):
        assert _edge_weight(m, ew, (2, 4, z), (2, 4, z + 1), st) == 1.0   # convex: top tread / riser
        assert _edge_weight(m, ew, (2, 3, z), (2, 3, z + 1), st) == 1.0   # concave: riser / next tread
    assert _edge_weight(m, ew, (1, 1, 1), (1, 1, 2), st) == 0.0           # bottom edge of the tall column
    # a lone one-voxel bump on a thick slab: a step only without the run rule
    slab = Part("slab", (8, 7, 8), 0.01)
    slab.material[1:7, 1:5, 1:7] = MAT_SKIN
    slab.material[4, 5, 4] = MAT_SKIN
    m = mesh.culled_quads(slab, [(200, 150, 120)], jitter=0.0)
    assert mesh.step_edge_weights(m, slab, min_run=1).weights.sum() == 12
    assert mesh.step_edge_weights(m, slab).weights.sum() == 0
