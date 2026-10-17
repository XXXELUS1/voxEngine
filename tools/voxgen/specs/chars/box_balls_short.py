"""Box head (voxgen.toonhead box preset): a chamfered 1 cm box form with
a painted face, eyes `balls`, hair `short` as a 5 mm voxgen.hairstyle part."""
from voxgen.toonhead import ToonHeadParams, build_toon_head


def build():
    return build_toon_head(ToonHeadParams.preset_params("box", eyes="balls", hair_style="short"), name="box_balls_short")
