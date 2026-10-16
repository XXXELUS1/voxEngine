"""Calibration: the MakeHuman-based head at 3 mm voxels (compare with head_mh at 5 mm)."""
from voxgen.head_from_mesh import HeadParams, build_head


def build():
    return build_head(0.003, HeadParams(), name="head_mh_3mm")
