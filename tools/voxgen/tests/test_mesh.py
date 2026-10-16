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


def test_surface_nets_keeps_hair_cubic():
    model, p = layered_ball(n=24, r=0.05, hair=True)
    m = mesh.surface_nets(p, model.palette)
    assert m.n_faces > 0
    assert (m.face_material[~m.face_smooth] == MAT_HAIR).all()
    assert (m.face_material[m.face_smooth] != MAT_HAIR).all()
    assert m.face_smooth.any() and (~m.face_smooth).any()
