"""The barber of the owner's concept sheet (voxgen.voxchar): a warm-skinned box head with the
flat painted face, thick brows, a short chestnut beard with the moustache and chunky hair
peeking from under a red beret with a white feather; the body (apron, rolled sleeves, tools)
is the next pass and here still comes from the generic ``apron`` outfit."""
from voxgen.voxchar import VoxCharParams, build_character

PARAMS = VoxCharParams(name="barber", skin_tone="warm", hair_tone="chestnut", hair="chunky", beard="short",
                       mustache=True, brows_thick=True, hat="beret", outfit="apron", sleeves="rolled")


def build():
    return build_character(PARAMS)
