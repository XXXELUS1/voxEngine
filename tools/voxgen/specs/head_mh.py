"""Head voxelized from the MakeHuman base mesh at 5 mm (voxgen.head_from_mesh)."""
from voxgen.head_from_mesh import HeadParams, build_head


def build():
    return build_head(0.005, HeadParams(), name="head_mh")
