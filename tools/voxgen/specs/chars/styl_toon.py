"""Designed voxel-art head, cartoon proportions: the ``toon`` preset of voxgen.toonhead
(bigger cranium, big eyes, small jaw, bigger nose and ears; bangs)."""
from voxgen.toonhead import ToonHeadParams, build_toon_head


def build():
    return build_toon_head(ToonHeadParams.preset_params("toon"), name="styl_toon")
