"""The barber: lean, a short cut and a mustache, a linen apron over a blue tunic (voxgen.character)."""
from voxgen.body_from_mesh import BodyParams, Outfit
from voxgen.character import CharacterParams, build_character
from voxgen.sculpthead import SculptParams


def build():
    head = SculptParams(voxel=0.003, quantize=True, hair_style="short", beard="mustache", skin_tone="light", hair_tone="brown",
                        iris_tone="brown", nose_size=1.05, lip_fullness=0.9, chin_size=0.95)
    outfit = Outfit(kind="apron", tones={"under": (84, 104, 128), "trousers": (70, 62, 56)})
    return build_character(CharacterParams(name="town_barber", head=head, body=BodyParams(build="lean"), outfit=outfit))
