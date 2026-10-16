"""Designed voxel-art head, human proportions: the ``normal`` preset of voxgen.toonhead
(1 cm design grid upsampled x2 to 5 mm; short hair, neutral expression)."""
from voxgen.toonhead import ToonHeadParams, build_toon_head


def build():
    return build_toon_head(ToonHeadParams.preset_params("normal"), name="styl_normal")
