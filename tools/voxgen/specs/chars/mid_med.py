"""Caricature "mid", med: the middle between the realistic MakeHuman head and the old SDF head.
The stylized 3 mm head with feature relief (HeadParams.relief: one-voxel lid ridge, lips,
brows, two-voxel nostril holes) and the MID_MORPHS caricature at strength 1.2
(brow ridge, nose, lips, jaw, cheekbones, ears, hooded lids). Compare with
specs/calibration/head_mh_3mm.py (no morphs, no relief) and the two other strengths."""
from voxgen.head_from_mesh import HeadParams, build_head


def build():
    return build_head(0.003, HeadParams.mid(1.2), name="mid_med")
