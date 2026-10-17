"""Multi-resolution "mid" head: the skin form at 5 mm (morphed MakeHuman mesh, closed/opened,
mirrored, quantized: no one-voxel terraces), the details as their own 3 mm parts - two eyeballs
(spheres, r = 12 mm, iris + pupil + one highlight, bones ``eye_l`` / ``eye_r``), the lips, the
brows and the short haircut - each 6-adjacent to the form, which is carved where a part sits.
Same face as ``specs/chars/mid_v2.py`` (MID_MORPHS at 1.2 with the chin fix). The parts are
meant to rotate / animate later; ``model.meta["mesh"]`` and ``model.meta["seams"]`` carry the
voxel / quad counts and the overlap / contact counts per part."""
from voxgen.multires_head import MultiresParams, build_multires_head


def build():
    return build_multires_head(MultiresParams(), name="multi_med")
