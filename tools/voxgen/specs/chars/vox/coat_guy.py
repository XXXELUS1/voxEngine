"""Our version of the owner's winter-coat reference (voxgen.voxchar): a box head with the
flat painted face and slab hair, a 7-head block body in a white coat with lapels and brown
lining over a white shirt, jeans-blue trousers with a lighter hem, white block shoes."""
from voxgen.voxchar import VoxCharParams, build_character

PARAMS = VoxCharParams(name="coat_guy", skin_tone="light", hair_tone="brown", hair="slabs", fringe_sweep=-1,
                       beard="none", hat="none", outfit="coat")


def build():
    return build_character(PARAMS)
