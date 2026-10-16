"""Small synthetic models shared by the tests."""
import numpy as np

from voxgen import Model, Palette, Part, body, sdf
from voxgen.materials import MAT_BONE, MAT_FAT, MAT_HAIR, MAT_MUSCLE, MAT_SKIN


def palette():
    pal = Palette()
    pal.add("skin", (224, 180, 150))
    pal.add("fat", (238, 214, 140))
    pal.add("muscle", (150, 40, 45))
    pal.add("bone", (235, 225, 200))
    pal.add("hair", (60, 40, 30))
    return pal


def layered_ball(n=32, vs=0.005, r=0.06, hair=False, seed=1):
    """A sphere with skin/fat/muscle layers and a bone core; optional hair on top."""
    pal = palette()
    m = Model("ball", vs, pal)
    half = n * vs / 2
    p = m.add_part(Part("ball", (n, n, n), vs, origin=(-half, -half, -half), bone="", symmetric=True))
    s = sdf.sphere(p, (0, 0, 0), r)
    body.fill_layers(p, s, [(MAT_SKIN, 1), (MAT_FAT, 1), (MAT_MUSCLE, None)], core=(MAT_BONE, sdf.sphere(p, (0, 0, 0), r * 0.5)))
    if hair:
        body.grow_hair(p, sdf.half_space(p, (0, r * 0.4, 0), (0, -1, 0)), (2, 5), "hair", direction="normal", seed=seed)
    return m, p
