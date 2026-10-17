"""The knight: a squarer jaw (within the limits), cropped hair, a scar on the cheek, a quilted
gambeson with a leather belt over a broad body (voxgen.character)."""
from voxgen.body_from_mesh import BodyParams, Outfit
from voxgen.character import CharacterParams, build_character
from voxgen.sculpthead import SculptParams


def build():
    head = SculptParams(voxel=0.003, quantize=True, hair_style="short", skin_tone="tan", hair_tone="black", iris_tone="blue",
                        cranium_scale=(1.04, 1.0, 1.0), jaw_width=1.1, chin_size=1.1, cheekbone=1.1, neck_scale=1.1,
                        hair=dict(top=0.009, side=0.006, back=0.008, ledge=0.006))
    outfit = Outfit(kind="gambeson", tones={"trousers": (62, 66, 70)})
    return build_character(CharacterParams(name="town_knight", head=head, body=BodyParams(build="broad"), outfit=outfit,
                                           scar=True))
