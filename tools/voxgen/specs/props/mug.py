"""Ceramic mug with a handle (voxel 10 mm). Sits on y = 0, centred on x/z."""
from voxgen import Model, Palette, Part, sdf
from voxgen.materials import MAT_CERAMIC

R, H, WALL = 0.045, 0.105, 0.012


def build():
    pal = Palette()
    GLAZE = pal.add("glaze", (120, 96, 72))
    RIM = pal.add("glaze_light", (176, 150, 120))
    INNER = pal.add("inner", (208, 190, 160))
    m = Model("mug", voxel_size=0.01, palette=pal)
    p = m.add_part(Part.centered("mug", (0.18, 0.14, 0.14), 0.01))
    outer = sdf.cylinder(p, (0, 0, 0), (0, H, 0), R, round=0.006)
    bore = sdf.cylinder(p, (0, 0.015, 0), (0, H + 0.02, 0), R - WALL)
    cup = sdf.subtract(outer, bore)
    handle = sdf.torus(p, (R + 0.004, H * 0.5, 0), 0.03, 0.011, axis="z")
    handle = sdf.intersect(handle, sdf.half_space(p, (R - 0.005, 0, 0), (-1, 0, 0)))  # outer half, ends inside the wall
    p.paint(sdf.union(cup, handle), MAT_CERAMIC, GLAZE)
    # lighter rim band and the visible inside
    p.paint(sdf.intersect(cup, sdf.half_space(p, (0, H - 0.02, 0), (0, -1, 0))), MAT_CERAMIC, RIM, only=[MAT_CERAMIC])
    p.paint(sdf.intersect(cup, sdf.offset(bore, 1.0)), MAT_CERAMIC, INNER, only=[MAT_CERAMIC])
    return m
