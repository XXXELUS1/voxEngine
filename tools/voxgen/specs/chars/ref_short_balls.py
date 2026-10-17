"""Reference-style head (voxgen.refhead): the 1 cm form with a minimal painted face, eyes as
`balls`, hair `short` as a 5 mm part (voxgen.hairstyle)."""
from voxgen.refhead import RefHeadParams, build_ref_head


def build():
    return build_ref_head(RefHeadParams(eyes="balls", hair_style="short", hair_voxel=0.005), name="ref_short_balls")
