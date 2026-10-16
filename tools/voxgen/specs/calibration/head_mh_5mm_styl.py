"""Calibration: the neutral MakeHuman head at 5 mm voxels in the stylized cubes look
(HeadParams.stylized, no morphs) - the 5 mm counterpart of head_mh_3mm for the facet
(chamfered steps) study: does the chamfer keep the voxel identity at 5 mm?"""
from voxgen.head_from_mesh import HeadParams, build_head


def build():
    return build_head(0.005, HeadParams.stylized(), name="head_mh_5mm_styl")
