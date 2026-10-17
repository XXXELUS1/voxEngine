"""Caricature "mid" v2 (3 mm, one part): ``mid_med`` after the owner's review of the mid sheet.
The chin is a normal chin again (``CHIN_FIX``: chin-prominent 0.2, chin-width 0.3, no jaw drop),
the brows are separated (inner ends 5 voxels apart), the hair is trimmed 12 mm above the ears so
the ears show. Everything else as ``specs/chars/mid_med.py`` (MID_MORPHS at 1.2, relief, flat
tints). Compare with ``specs/chars/multi_med.py``: the same face as a 5 mm form + 3 mm detail parts.
``model.meta["mesh"]`` carries the cell / solid / quad counts for the performance question."""
from voxgen.head_from_mesh import HeadParams, build_head
from voxgen.multires_head import mesh_stats


def build():
    model = build_head(0.003, HeadParams.mid_v2(1.2), name="mid_v2")
    model.meta["mesh"] = mesh_stats(model)
    return model
