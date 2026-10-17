"""Box head (voxgen.toonhead box preset): the reference twin - a chamfered 1 cm box form with
a painted face, eyes `shades`, hair `quiff` as a 5 mm voxgen.hairstyle part."""
from voxgen.toonhead import ToonHeadParams, build_toon_head


def build():
    return build_toon_head(ToonHeadParams.preset_params("box", eyes="shades", hair_style="quiff"), name="box_shades_quiff")
