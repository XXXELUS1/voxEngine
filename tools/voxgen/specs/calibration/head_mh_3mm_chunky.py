"""Calibration: the neutral 3 mm stylized head with the form voxelized at 6 mm and
upsampled x2 (HeadParams.form_voxel_mult = 2): chunky, deliberate steps; the details
(eyes, lips, brows, hair edge, nostrils) still carved on the 3 mm grid. Compare with
head_mh_3mm (form at 3 mm, cleaned)."""
from voxgen.head_from_mesh import HeadParams, build_head


def build():
    return build_head(0.003, HeadParams.stylized(form_voxel_mult=2), name="head_mh_3mm_chunky")
