"""Sculpted stylised head (voxgen.sculpthead, preset `sculpt`): a smooth SDF form - egg cranium
with a flat forehead, brow ridge, cheekbones, tapered jaw, chin bulge, wedge nose, lip bulge,
rounded ears - the SDF sampled at 10 mm and upsampled x2 (the box preset's step size), the face painted, eyes `balls`,
hair `quiff` as a 5 mm voxgen.hairstyle part."""
from voxgen.sculpthead import SculptParams, build_sculpt_head


def build():
    return build_sculpt_head(SculptParams.preset_params("sculpt", eyes="balls", hair_style="quiff", grid="coarse"), name="sculpt_balls_quiff_coarse")
