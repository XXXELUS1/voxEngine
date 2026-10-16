"""Parametric layered human head (shared by head_test.py and the calibration specs).

Everything is authored in metres in the head's local space: Y up, the face looks
toward -Z, the chin is near y = 0 and the head is centred on x/z. ``build_head`` returns
a Model with one part "head" (bone "head") whose grid is x-centred so that mirror
symmetry about x = 0 is exact (``sdf.mirror_x`` is used for all paired features).

Tissue (outside -> in): skin (~5 mm), fat (~5 mm), muscle, then bone (skull ellipsoid +
mandible + maxilla), teeth (two rows of small boxes), cartilage (ears, nose), eyes
(spheres with an iris dot) and vessels (thin capsules in the fat under the temples and
on the neck). Hair: columns of varying length on the scalp swept back, 1-voxel stubble
on the jaw and cheeks, 1-voxel eyebrows. A subtle symmetric value-noise displacement
keeps the silhouette organic.

Variants: "default" (hair + stubble + brows), "bald" (no scalp hair), "shaved" (no stubble).
"""

from __future__ import annotations

from voxgen import Model, Palette, Part, body, sdf
from voxgen.materials import (MAT_BONE, MAT_CARTILAGE, MAT_EYE, MAT_FAT, MAT_HAIR, MAT_MUSCLE, MAT_SKIN,
                              MAT_TOOTH, MAT_VESSEL)

# Metre box the head occupies (hair needs headroom above the crown, the neck stub a bit below the chin).
HEAD_LO = (-0.12, -0.04, -0.13)
HEAD_HI = (0.12, 0.24, 0.13)


def make_palette() -> Palette:
    """Muted earthy palette: skin with a darker and a ruddier variant, lips, tissues, hair."""
    pal = Palette()
    pal.add("skin", (224, 180, 150))
    pal.add("skin_dark", (209, 165, 136))
    pal.add("skin_ruddy", (229, 166, 142))
    pal.add("lips", (206, 138, 122))
    pal.add("fat", (238, 214, 140))
    pal.add("muscle", (150, 40, 45))
    pal.add("bone", (235, 225, 200))
    pal.add("tooth", (245, 240, 220))
    pal.add("tooth_dark", (232, 224, 198))
    pal.add("cartilage", (225, 185, 160))
    pal.add("vessel", (120, 20, 30))
    pal.add("hair", (60, 40, 30))
    pal.add("stubble", (82, 58, 44))
    pal.add("eye_white", (240, 240, 235))
    pal.add("iris", (70, 90, 110))
    return pal


def build_head(voxel_size_m: float = 0.005, dims_hint=None, variant: str = "default", name: str | None = None) -> Model:
    """Build the head at the given voxel size (metres). ``dims_hint`` = (nx, ny, nz) forces the
    grid size (kept x/z-centred); otherwise the grid covers HEAD_LO..HEAD_HI."""
    vs = float(voxel_size_m)
    pal = make_palette()
    model = Model(name or f"head_{round(vs * 1000)}mm", voxel_size=vs, palette=pal,
                  meta={"variant": variant})
    if dims_hint:
        nx, ny, nz = dims_hint
        lo = (-nx * vs / 2, HEAD_LO[1], -nz * vs / 2)
        part = Part("head", (nx, ny, nz), vs, origin=lo, bone="head", symmetric=True)
    else:
        part = Part.from_extent("head", HEAD_LO, HEAD_HI, vs, bone="head", symmetric=True)
    part.budget = 400_000 if vs < 0.004 else 250_000
    model.add_part(part)

    g = part                 # plain grid
    gm = sdf.mirror_x(part)  # author paired features on +X, they appear on both sides
    vox = lambda metres: max(1, int(round(metres / vs)))  # thickness in voxels, >= 1
    k_big, k_small = part.m2v(0.016), part.m2v(0.007)     # blend radii in voxels

    # ---------------------------------------------------------------- skin surface
    cranium = sdf.ellipsoid(g, (0, 0.125, 0.012), (0.074, 0.082, 0.092))
    face_plate = sdf.box(g, (0, 0.108, -0.050), (0.118, 0.080, 0.075), round=0.024)  # flat forehead / eye region
    jaw = sdf.ellipsoid(g, (0, 0.060, -0.008), (0.060, 0.070, 0.078))
    neck = sdf.cylinder(g, (0, -0.08, 0.015), (0, 0.05, 0.015), 0.046)
    head = sdf.smooth_union(cranium, face_plate, k_big)
    head = sdf.smooth_union(head, jaw, k_big)
    head = sdf.smooth_union(head, neck, k_big)
    # brow ridge and cheekbones: gentle bumps
    brow = sdf.capsule(gm, (0.012, 0.136, -0.082), (0.050, 0.130, -0.070), 0.010)
    cheek = sdf.sphere(gm, (0.050, 0.090, -0.050), 0.022)
    head = sdf.smooth_union(head, sdf.union(brow, cheek), k_big)
    # nose: bridge capsule, tip sphere, nostril wings (mirrored)
    bridge = sdf.capsule(g, (0, 0.118, -0.080), (0, 0.078, -0.097), 0.010)
    tip = sdf.sphere(g, (0, 0.071, -0.100), 0.013)
    wing = sdf.sphere(gm, (0.012, 0.067, -0.090), 0.010)
    nose = sdf.union(bridge, tip, wing)
    head = sdf.smooth_union(head, nose, k_small)
    # ears: flattened ellipsoids on the sides, tilted back a little
    ear_c = (0.081, 0.110, 0.012)
    ear_grid = sdf.rotate(gm, (0, 0, -8), pivot=ear_c)
    ear = sdf.ellipsoid(ear_grid, ear_c, (0.007, 0.032, 0.021))
    head = sdf.smooth_union(head, ear, k_small)
    # organic silhouette: subtle symmetric noise
    head = head + sdf.noise(gm, scale=0.035, amplitude=0.0018, seed=3, octaves=2)
    # recesses: eye sockets (shallow almonds), ear canals, the lip line
    socket = sdf.ellipsoid(gm, (0.029, 0.118, -0.0905), (0.018, 0.011, 0.012))
    head = sdf.smooth_subtract(head, socket, part.m2v(0.004))
    canal = sdf.sphere(gm, (0.085, 0.108, 0.014), 0.0055)
    head = sdf.subtract(head, canal)
    lip_line = sdf.box(g, (0, 0.036, -0.088), (0.042, 0.0055, 0.014))
    head = sdf.subtract(head, lip_line)
    nostril = sdf.sphere(gm, (0.008, 0.060, -0.097), 0.004)
    head = sdf.subtract(head, nostril)

    # ---------------------------------------------------------------- bone / cartilage cores
    skull = sdf.ellipsoid(g, (0, 0.125, 0.012), (0.061, 0.069, 0.079))
    mandible = sdf.capsule(gm, (0.0, 0.022, -0.052), (0.048, 0.034, 0.012), 0.009)
    ramus = sdf.capsule(gm, (0.048, 0.034, 0.012), (0.050, 0.095, 0.014), 0.008)
    maxilla = sdf.box(g, (0, 0.055, -0.058), (0.052, 0.018, 0.032), round=0.005)
    bone = sdf.union(skull, mandible, ramus, maxilla)
    ear_core = sdf.ellipsoid(ear_grid, ear_c, (0.0035, 0.025, 0.015))
    nose_core = sdf.union(sdf.capsule(g, (0, 0.118, -0.080), (0, 0.080, -0.096), 0.005),
                          sdf.sphere(g, (0, 0.072, -0.099), 0.0075))
    cart = sdf.union(ear_core, nose_core)

    # ---------------------------------------------------------------- tissue layers
    t_skin, t_fat = vox(0.005), vox(0.005)
    body.fill_layers(part, head,
                     [(MAT_SKIN, t_skin, "skin"), (MAT_FAT, t_fat, "fat"), (MAT_MUSCLE, None, "muscle")],
                     core=[(MAT_BONE, bone, "bone"), (MAT_CARTILAGE, cart, "cartilage")])

    # teeth: two rows of small boxes along the dental arch, behind the lips
    inner = [MAT_FAT, MAT_MUSCLE, MAT_BONE]
    n_teeth = 4  # per side
    pitch = 0.0055
    for i in range(n_teeth):
        x = (i + 0.5) * pitch
        z = -0.0765 + 0.014 * (x / (n_teeth * pitch)) ** 2
        col = "tooth" if i % 2 == 0 else "tooth_dark"
        upper = sdf.box(gm, (x, 0.0455, z), (0.0046, 0.0095, 0.0065))
        lower = sdf.box(gm, (x, 0.0325, z + 0.0015), (0.0046, 0.0085, 0.0065))
        body.paint(part, upper, MAT_TOOTH, col, only=inner)
        body.paint(part, lower, MAT_TOOTH, col, only=inner)

    # eyes: spheres whose front cap sits in the socket recess, iris as a tinted dot
    eye = sdf.sphere(gm, (0.029, 0.118, -0.0712), 0.0125)
    body.paint(part, eye, MAT_EYE, "eye_white")
    iris = sdf.cylinder(gm, (0.029, 0.118, -0.120), (0.029, 0.118, -0.080), 0.0052)
    body.tint_region(part, iris, "iris", materials=[MAT_EYE])

    # vessels: capsules that only replace fat/muscle, i.e. they follow the layer under the skin
    temple = sdf.capsule(gm, (0.068, 0.100, -0.040), (0.064, 0.150, -0.022), 0.0065)
    carotid = sdf.capsule(gm, (0.028, -0.050, -0.020), (0.034, 0.045, -0.010), 0.0065)
    body.paint(part, sdf.union(temple, carotid), MAT_VESSEL, "vessel", only=[MAT_FAT, MAT_MUSCLE])

    # skin tints: ruddy cheeks / nose / ears, darker eye shadow and under the jaw, lips
    body.tint_region(part, sdf.sphere(gm, (0.046, 0.086, -0.058), 0.026), "skin_ruddy")
    body.tint_region(part, sdf.sphere(g, (0, 0.072, -0.096), 0.017), "skin_ruddy")
    body.tint_region(part, sdf.ellipsoid(ear_grid, ear_c, (0.012, 0.034, 0.024)), "skin_ruddy")
    body.tint_region(part, sdf.ellipsoid(gm, (0.029, 0.117, -0.088), (0.020, 0.012, 0.014)), "skin_dark")
    body.tint_region(part, sdf.half_space(g, (0, 0.004, 0), (0, 1, 0)), "skin_dark")
    body.tint_region(part, sdf.ellipsoid(g, (0, 0.036, -0.090), (0.025, 0.0105, 0.012)), "lips")

    # ---------------------------------------------------------------- hair
    if variant != "bald":
        scalp = sdf.ellipsoid(g, (0, 0.135, 0.020), (0.10, 0.11, 0.115))
        hairline = sdf.half_space(g, (0, 0.165, -0.08), (0, 1, 0.40))  # region below the tilted hairline plane
        scalp = sdf.subtract(scalp, hairline)
        scalp = sdf.subtract(scalp, sdf.box(gm, (0.090, 0.105, 0.012), (0.05, 0.078, 0.062)))  # keep the ears clear
        body.grow_hair(part, scalp, (vox(0.015), vox(0.040)), "hair", direction="normal",
                       sweep=(0.0, -0.35, 0.45), jitter=0.3, density=0.8, seed=11, coherence=0.03)
    brows = sdf.capsule(gm, (0.013, 0.138, -0.088), (0.046, 0.134, -0.076), 0.0055)
    body.grow_hair(part, brows, 1, "hair", direction="normal", jitter=0.0, seed=5)
    if variant != "shaved":
        beard = sdf.ellipsoid(g, (0, 0.050, -0.008), (0.075, 0.075, 0.095))
        beard = sdf.intersect(beard, sdf.half_space(g, (0, 0.082, 0), (0, 1, 0)))       # below the cheekbones
        beard = sdf.intersect(beard, sdf.half_space(g, (0, 0, 0.035), (0, 0, 1)))       # not on the neck back
        beard = sdf.intersect(beard, sdf.half_space(g, (0, 0.000, 0), (0, -1, 0)))      # above the chin bottom
        beard = sdf.subtract(beard, sdf.ellipsoid(g, (0, 0.038, -0.090), (0.030, 0.015, 0.02)))  # not on the lips
        beard = sdf.subtract(beard, sdf.sphere(g, (0, 0.072, -0.096), 0.020))                 # not on the nose
        body.grow_hair(part, beard, 1, "stubble", direction="normal", jitter=0.0, density=0.45, seed=23)
    return model
