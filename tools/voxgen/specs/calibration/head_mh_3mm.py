"""Calibration: the neutral MakeHuman head at 3 mm voxels in the stylized cubes look
(HeadParams.stylized, no morphs) - the reference the caricatures in specs/chars are
measured against. Compare with head_mh at 5 mm."""
from voxgen.head_from_mesh import HeadParams, build_head


def build():
    return build_head(0.003, HeadParams.stylized(), name="head_mh_3mm")
