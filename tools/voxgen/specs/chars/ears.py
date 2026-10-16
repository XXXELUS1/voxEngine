"""Caricature "Ears": huge ears standing out from a long oval head with a short chin.

MakeHuman modifiers (voxgen.mh_targets); ear modifiers are expanded to both sides.
3 mm voxels, stylized cubes look.
"""
from voxgen.head_from_mesh import HeadParams, build_head

MORPHS = {
    "ear-scale": 1.8,          # -> l-ear-scale + r-ear-scale
    "ear-wing": 1.0,           # outward
    "ear-flap": 1.0,           # outward
    "chin-height": -0.6,
    "head-oval": 0.5,
}


def build():
    return build_head(0.003, HeadParams.stylized(morphs=MORPHS), name="ears")
