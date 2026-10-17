"""The widow: female, older (grey hair, no wrinkles), a headscarf over the hair, a dark dress
(voxgen.character)."""
from voxgen.body_from_mesh import BodyParams, Outfit
from voxgen.character import CharacterParams, build_character
from voxgen.sculpthead import SculptParams


def build():
    head = SculptParams(voxel=0.003, quantize=True, sex="female", hair_style="short", headscarf=True, skin_tone="dark",
                        hair_tone="grey", iris_tone="brown", ear_size=0.95, cheekbone=1.05,
                        scarf_tones=((78, 76, 90), (54, 52, 64)))
    outfit = Outfit(kind="dress", tones={"boots": (40, 34, 32)})
    return build_character(CharacterParams(name="town_widow", head=head, body=BodyParams(build="female", boot_top=0.09),
                                           outfit=outfit))
