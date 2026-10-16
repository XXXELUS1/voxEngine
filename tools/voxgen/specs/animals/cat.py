"""Cat: a small standing cat at 10 mm voxels (~35 voxels long) for the facet (chamfered steps)
render study - the owner's reference is a chamfered-voxel black cat with cyan eyes.

Three body shades in height bands (dark belly, base, light back), cyan eyes (2 voxels each), a pink
nose (2 voxels). Generic material: a prop, not a body - no layers, no damage. Stands on y = 0,
faces -Z, centred on x/z.
"""
from voxgen import Model, Palette, Part, sdf
from voxgen.materials import MAT_EYE, MAT_GENERIC

VS = 0.01


def build() -> Model:
    pal = Palette()
    base = pal.add("fur", (44, 50, 64))
    belly = pal.add("fur_dark", (28, 31, 42))
    back = pal.add("fur_light", (70, 80, 100))
    eye = pal.add("eye", (70, 220, 255))
    nose = pal.add("nose", (232, 120, 140))
    m = Model("cat", voxel_size=VS, palette=pal)
    p = m.add_part(Part.centered("cat", (0.20, 0.36, 0.52), VS))
    X, Y, Z = p.coords()

    # body: shoulders -> hips capsule, chest and rump ellipsoids
    fur = sdf.union(sdf.capsule(p, (0, 0.17, -0.07), (0, 0.165, 0.08), 0.05),
                    sdf.ellipsoid(p, (0, 0.165, -0.08), (0.052, 0.058, 0.055)),
                    sdf.ellipsoid(p, (0, 0.17, 0.085), (0.05, 0.055, 0.05)))
    # head sphere, neck, muzzle box
    fur = sdf.union(fur, sdf.sphere(p, (0, 0.235, -0.18), 0.05),
                    sdf.capsule(p, (0, 0.19, -0.11), (0, 0.225, -0.16), 0.03),
                    sdf.box(p, (0, 0.215, -0.23), (0.05, 0.035, 0.05)))
    # ears: stacked shrinking boxes (stepped pyramids)
    for sx in (-1, 1):
        for w, y in ((0.03, 0.275), (0.02, 0.29), (0.01, 0.30)):
            fur = sdf.union(fur, sdf.box(p, (sx * 0.028, y, -0.175), (w, 0.015, w)))
    # legs: capsules + paw boxes
    for sx in (-1, 1):
        for zc in (-0.075, 0.075):
            fur = sdf.union(fur, sdf.capsule(p, (sx * 0.03, 0.17, zc), (sx * 0.03, 0.02, zc), 0.02),
                            sdf.box(p, (sx * 0.03, 0.015, zc - 0.012), (0.045, 0.03, 0.06)))
    # tail: a chain of spheres curving up and back
    for i in range(14):
        t = i / 13.0
        fur = sdf.union(fur, sdf.sphere(p, (0.0, 0.17 + 0.12 * t * t, 0.11 + 0.14 * t - 0.04 * t * t),
                                        0.02 - 0.006 * t))
    p.paint(fur, MAT_GENERIC, base)

    # shades by height: dark belly, base body, lighter back and ears
    solid = p.solid()
    p.paint(solid & (Y < 0.13), MAT_GENERIC, belly)
    p.paint(solid & (Y >= 0.215) & (Z > -0.12) & (Z < 0.12), MAT_GENERIC, back)
    p.paint(solid & (Y >= 0.27), MAT_GENERIC, back)
    # eyes: 2 voxels each on the head front; nose: 2 voxels on the muzzle tip
    p.paint(solid & (abs(abs(X) - 0.025) < 0.011) & (abs(Y - 0.245) < 0.006) & (Z < -0.21), MAT_EYE, eye)
    p.paint(solid & (abs(X) < 0.011) & (abs(Y - 0.225) < 0.006) & (Z < -0.25), MAT_GENERIC, nose)
    return m
