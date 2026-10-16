"""Caricature "jaw" at 5 mm voxels (same morphs as jaw.py, stylized look) for the facet study."""
from voxgen.head_from_mesh import HeadParams, build_head

from jaw import MORPHS


def build():
    return build_head(0.005, HeadParams.stylized(morphs=MORPHS), name="jaw_5mm")
