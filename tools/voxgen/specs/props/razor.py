"""Straight razor, open, lying flat on the table (voxel 4 mm): steel blade + wooden scales.

Along +X: the blade from x = -0.075 to the pivot at x = 0, the handle from 0 to 0.105.
Sits on y = 0, centred on x/z.
"""
from voxgen import Model, Palette, Part, sdf
from voxgen.materials import MAT_METAL, MAT_WOOD


def build():
    pal = Palette()
    STEEL = pal.add("steel", (178, 182, 188))
    EDGE = pal.add("steel_bright", (222, 226, 230))
    WOOD = pal.add("handle", (96, 62, 36))
    PIN = pal.add("brass", (190, 150, 70))
    m = Model("razor", voxel_size=0.004, palette=pal)
    p = m.add_part(Part.centered("razor", (0.20, 0.03, 0.05), 0.004))
    y = 0.006  # resting height of the blade mid-plane
    # blade: flat box with a rounded nose, thin bright edge along -Z
    blade = sdf.box(p, (-0.040, y, 0.0), (0.075, 0.004, 0.022), round=0.002)
    nose = sdf.cylinder(p, (-0.0775, y - 0.003, 0.0), (-0.0775, y + 0.003, 0.0), 0.011)
    blade = sdf.union(blade, nose)
    tang = sdf.box(p, (0.004, y, 0.004), (0.02, 0.004, 0.010), round=0.001)
    p.paint(sdf.union(blade, tang), MAT_METAL, STEEL)
    edge = sdf.box(p, (-0.040, y, -0.0105), (0.072, 0.004, 0.003))
    p.paint(edge, MAT_METAL, EDGE, only=[MAT_METAL])
    # scales (handle): two wooden slabs with a gap, pivot pin at the tang
    scale_lo = sdf.box(p, (0.055, y - 0.004, 0.004), (0.105, 0.0045, 0.016), round=0.002)
    scale_hi = sdf.box(p, (0.055, y + 0.004, 0.004), (0.105, 0.0045, 0.016), round=0.002)
    p.paint(sdf.union(scale_lo, scale_hi), MAT_WOOD, WOOD, only=[0])
    pin = sdf.cylinder(p, (0.008, y - 0.007, 0.004), (0.008, y + 0.007, 0.004), 0.0025)
    p.paint(pin, MAT_METAL, PIN)
    return m
