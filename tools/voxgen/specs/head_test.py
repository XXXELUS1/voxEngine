"""The head the Godot lab scene loads (godot/assets/models/head_test.vxm): 5 mm voxels.

Since the MakeHuman import this is the mesh-based head (voxgen.head_from_mesh), the
same as specs/head_mh.py; the old SDF head is still available as
``_human_head.build_head`` (specs/calibration/head_{3,5,8}mm.py)."""
from voxgen.head_from_mesh import HeadParams, build_head


def build():
    return build_head(0.005, HeadParams(), name="head_test")
