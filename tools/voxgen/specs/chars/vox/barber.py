"""The barber of the owner's concept sheet (voxgen.voxchar): a warm-skinned head of coarse cubes
with 45-degree edges (render it with the ``facet`` style: every chamfer is a 1:1 staircase of five
fine voxels), the flat painted face, the block nose with its top sloped, thick brows, a short
chestnut beard following the jaw with the moustache, chunky hair with tufts peeking from under a
red beret with a white feather; the body (apron, rolled sleeves, tools) is the next pass and here
still comes from the generic ``apron`` outfit."""
from voxgen.voxchar import VoxCharParams, build_character

PARAMS = VoxCharParams(name="barber", skin_tone="warm", hair_tone="chestnut", hair="chunky", beard="short",
                       mustache=True, brows_thick=True, hat="beret", outfit="apron", sleeves="rolled")


def build():
    return build_character(PARAMS)
