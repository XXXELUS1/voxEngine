"""Designed voxel-art head, canon proportions: the ``real`` preset of voxgen.toonhead (1 cm form
with a boxier cranium and a wide chin, the face drawn at 5 mm by voxgen.facedetail)."""
from voxgen.toonhead import ToonHeadParams, build_toon_head


def build():
    return build_toon_head(ToonHeadParams.preset_params("real"), name="styl_real")
