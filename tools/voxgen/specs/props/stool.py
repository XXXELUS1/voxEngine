"""Three-legged wooden stool (voxel 15 mm). Sits on y = 0, centred on x/z."""
import math

from voxgen import Model, Palette, Part, sdf
from voxgen.materials import MAT_WOOD

SEAT_R = 0.17
SEAT_Y = 0.44          # top of the seat
SEAT_T = 0.035         # seat thickness
LEG_R = 0.022


def build():
    pal = Palette()
    LIGHT = pal.add("wood_light", (166, 118, 68))
    DARK = pal.add("wood_dark", (118, 78, 42))
    m = Model("stool", voxel_size=0.015, palette=pal)
    p = m.add_part(Part.centered("stool", (0.46, 0.46, 0.46), 0.015))
    seat = sdf.cylinder(p, (0, SEAT_Y - SEAT_T, 0), (0, SEAT_Y, 0), SEAT_R, round=0.01)
    p.paint(seat, MAT_WOOD, LIGHT)
    # three splayed legs + a ring of stretchers between them
    feet = []
    for i in range(3):
        a = math.radians(90 + i * 120)
        top = (0.10 * math.cos(a), SEAT_Y - SEAT_T + 0.005, 0.10 * math.sin(a))
        foot = (0.17 * math.cos(a), 0.0, 0.17 * math.sin(a))
        feet.append((top, foot))
        p.paint(sdf.cylinder(p, foot, top, LEG_R), MAT_WOOD, DARK, only=[0])
    for i in range(3):
        (t1, f1), (t2, f2) = feet[i], feet[(i + 1) % 3]
        a1 = tuple(t1[k] + (f1[k] - t1[k]) * 0.55 for k in range(3))
        a2 = tuple(t2[k] + (f2[k] - t2[k]) * 0.55 for k in range(3))
        p.paint(sdf.cylinder(p, a1, a2, 0.014), MAT_WOOD, LIGHT, only=[0])
    return m
