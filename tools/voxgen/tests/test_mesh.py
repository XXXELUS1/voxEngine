import numpy as np

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
