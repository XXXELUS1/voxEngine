"""The drunk: a red nose, stubble, messy red hair, a patched tunic (voxgen.character)."""
from voxgen.body_from_mesh import BodyParams, Outfit
from voxgen.character import CharacterParams, build_character
from voxgen.sculpthead import SculptParams


def build():
    head = SculptParams(voxel=0.003, quantize=True, hair_style="messy", beard="stubble", skin_tone="light", hair_tone="red",
                        iris_tone="brown", nose_size=1.15, nose_width=1.1, ear_size=1.05, lip_fullness=1.05)
    outfit = Outfit(kind="tunic", patches=True, tones={"base": (112, 96, 72), "shade": (84, 70, 52),
                                                       "trim": (68, 46, 30), "trousers": (56, 52, 50)})
    return build_character(CharacterParams(name="town_drunk", head=head, body=BodyParams(build="normal"), outfit=outfit,
                                           nose_tint=True))
