"""The monk: a round cranium, a tonsure, a brown robe with a rope belt over a plump body (voxgen.character)."""
from voxgen.body_from_mesh import BodyParams, Outfit
from voxgen.character import CharacterParams, build_character
from voxgen.sculpthead import SculptParams


def build():
    head = SculptParams(voxel=0.003, quantize=True, hair_style="short", tonsure=0.055, skin_tone="pale", hair_tone="brown",
                        iris_tone="green", cranium_scale=(1.06, 1.02, 1.0), cranium_power=2.2, chin_size=1.05,
                        nose_size=1.1, lip_fullness=1.1, cheekbone=0.9)
    outfit = Outfit(kind="robe", tones={"boots": (52, 40, 32)})
    return build_character(CharacterParams(name="town_monk", head=head, body=BodyParams(build="plump", boot_top=0.06),
                                           outfit=outfit))
