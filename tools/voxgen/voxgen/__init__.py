"""voxgen: offline voxel model generator (SDF specs -> .vxm / .vox / previews / checks).

Quick start for a spec file::

    from voxgen import Model, Part, Palette, sdf, body
    from voxgen.materials import *

    def build():
        pal = Palette()
        WOOD = pal.add("wood", (150, 105, 60))
        m = Model("stool", voxel_size=0.015, palette=pal)
        p = m.add_part(Part.centered("stool", (0.4, 0.5, 0.4), 0.015))
        seat = sdf.cylinder(p, (0, 0.42, 0), (0, 0.45, 0), 0.17)
        p.paint(seat, MAT_WOOD, WOOD)
        return m
"""

__version__ = "0.1.0"

from . import body, checks, materials, sdf  # noqa: E402
from .model import Grid, Model, Part, as_mask  # noqa: E402
from .palette import Palette  # noqa: E402

__all__ = ["Model", "Part", "Grid", "Palette", "as_mask", "materials", "sdf", "body", "checks", "__version__"]
