"""Reference-style head (voxgen.refhead): the 1 cm form with a minimal painted face, eyes as
`balls`, hair `quiff` as a 3 mm part (voxgen.hairstyle)."""
from voxgen.refhead import RefHeadParams, build_ref_head


def build():
    return build_ref_head(RefHeadParams(eyes="balls", hair_style="quiff", hair_voxel=0.003), name="ref_quiff_balls_3mm")
