"""Caricature "Nosey": a long, deep, blunt nose on a receding chin.

MakeHuman modifiers (voxgen.mh_targets) with weights beyond +-1; 3 mm voxels, the
stylized cubes look (HeadParams.stylized).
"""
from voxgen.head_from_mesh import HeadParams, build_head

MORPHS = {
    "nose-scale-vert": 1.6,
    "nose-scale-depth": 1.3,
    "nose-point-width": 0.8,
    "chin-prominent": -0.5,
}


def build():
    return build_head(0.003, HeadParams.stylized(morphs=MORPHS), name="nosey")
