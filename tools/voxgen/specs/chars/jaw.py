"""Caricature "Jaw": a wide, heavy, jutting jaw on a thick neck, cheekbones to match.

MakeHuman modifiers (voxgen.mh_targets); paired l-/r- modifiers are expanded for both
sides. 3 mm voxels, stylized cubes look.
"""
from voxgen.head_from_mesh import HeadParams, build_head

MORPHS = {
    "chin-width": 1.5,
    "chin-jaw-drop": 0.8,
    "chin-prominent": 1.2,
    "cheek-bones": 0.6,        # -> l-cheek-bones + r-cheek-bones
    "neck-scale-horiz": 0.6,
}


def build():
    return build_head(0.003, HeadParams.stylized(morphs=MORPHS), name="jaw")
