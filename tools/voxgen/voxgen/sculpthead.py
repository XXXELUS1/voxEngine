"""Sculpted stylised head: a smooth SDF form in metres on the 5 mm grid, never a box, never a
mesh scan.

Why: the owner's reference (a voxel-art character with a quiff and sunglasses) is not built as
a chamfered box (``voxgen.toonhead`` ``box``) - its face is *constructed* like a stylised smooth
head: an egg-shaped cranium with a flat-ish forehead, a soft brow ridge, defined cheekbones,
cheeks slightly hollow below them, a jaw tapering to a rounded chin with a small chin bulge,
the nose a rounded wedge with a round tip and nostril wings, the lips a soft bulge with a
philtrum, rounded ears. Human proportions, mildly stylised: not big eyes, not a box, not a
realistic skull. So the form here is a sum of smooth-unioned primitives (``voxgen.sdf``), every
blend radius 8-20 mm so the surface has no creases, no noise displacement; the smooth render
(``voxgen render --styles mix``, surface nets) reads it as a sculpture while the engine keeps
5 mm voxels to shave and cut.

Construction (:func:`form_field`, metres; Y up, chin's bottom near y = 0, face toward -Z,
mirror-symmetric about x = 0 by evaluating the paired primitives on ``sdf.mirror_x``):
  cranium    superellipsoid (``cranium_power`` 2.4: between an egg and a rounded box), the front
             cut by the tilted forehead plane (``face_z`` at the brow, receding ``forehead_tilt``
             degrees toward the crown) with a soft seam (``smooth_intersect``);
  brow ridge horizontal capsule across the forehead's base, smooth-unioned;
  mid-face   an ellipsoid (the maxilla: cheek to cheek, eyes to mouth), two cheekbone ellipsoids,
             an optional weak cheek hollow (``smooth_subtract``) below them, optional shallow
             eye-socket dents (``socket_depth``; the eyeball parts carve their own windows);
  jaw        a mandible capsule per side from the jaw angle under the ear to beside the chin, the
             lower-face ellipsoid filling between them, a chin sphere for the bulge;
  nose       a cone bridge (root narrower than the end) from the nasion to the tip, a tip sphere,
             two wing spheres; blended into the face with its own radius;
  lips       a horizontal capsule a few mm proud of the upper lip, a philtrum groove subtracted
             above it (one voxel deep at 5 mm);
  ears       flattened ellipsoids on the skull sides with a concha dent;
  neck       a cylinder, blended into the jaw.
``grid="fine"`` samples the field at 5 mm; ``grid="coarse"`` samples it at 10 mm, keeps the
largest component and upsamples x2 (every step two fine voxels, as the box preset) - the two are
compared in the smooth render.

Everything else is the box preset's vocabulary: the head part carries skin / fat / muscle / bone
by depth (nose and ears cartilage), the face is colour only (:func:`paint_face`: a 2 x 8 mouth
line + lower-lip tone, two nostril voxels under the nose, the shade band under the brow ridge
(or the sunglasses), the under-faces and the neck in the shade tone, 6 x 2 brows in the hair
tone), eyes ``balls`` (3 mm eyeball parts in carved sockets, ``voxgen.refhead.carve_sockets``) /
``dots`` (4 x 2 dark voxels) / ``shades`` (a MAT_GLASS part: a flat lens plane one voxel in front
of the eye windows that follows the face two voxels thick where the face curves away), hair a
5 mm ``voxgen.hairstyle`` part on the sculpted scalp (hairline at the top of the forehead plane,
the form carved one voxel under it). Parts ``head``, ``eye_l`` / ``eye_r`` or ``shades``, ``hair``;
``meta["build"]``, ``meta["seams"]``, ``meta["mesh"]`` as the other multi-part heads.

Decisions (the specs are silent on a sculpted head): the tissue layering is the box preset's
depth rule (skin 1, fat 2, muscle 3, bone >= 4 voxels; ``head_from_mesh.layer_tissues`` needs a
MakeHuman body for its teeth, so it is not reused); the palette is the box preset's
(``toonhead.make_palette``); the painter is a metre-landmark copy of ``toonhead.paint_box_face``
(its signature takes the 1 cm design grid).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from . import materials as M
from . import meshvox, sdf
from .body import dilate, exposed, shift
from .checks import label_components
from .facedetail import _cols, _rect, front_face, hole_walls, x_coords
from .hairstyle import HairParams, build_hair, ear_zone, hair_floor
from .head_from_mesh import depth_map, keep_largest_component, upsample
from .model import Model, Part
from .toonhead import ToonHeadParams, make_palette

Vec3 = tuple[float, float, float]
PRESETS = ("sculpt", "sculpt_broad")
EYES = ("balls", "dots", "shades")
HAIR_STYLES = ("quiff", "short", "slick", "bald")
GRIDS = ("fine", "coarse")
FINE_VOXEL = 0.005
COARSE_VOXEL = 0.010
#: hairstyle.HairParams overrides shared by every style on the sculpted scalp (metres): a round
#: sweep hugging the skull, floor levels on 5 mm rows, the temples on the forehead plane's edge
HAIR_COMMON = dict(top=0.015, side=0.010, back=0.012, ledge=0.010, shape_power=4.0, floor_quantum=0.005,
                   temple_x=0.045, temple_drop=0.010, side_in=0.006, ear_clearance=0.008, sideburn_depth=0.020)
HAIR = {
    "quiff": dict(lit_band=0.04),
    "short": dict(ridges="", lit_band=0.02),
    "slick": dict(back=0.02, nape_drop=0.01, temple_drop=-0.005, ridges=""),
}


@dataclass
class SculptParams:
    """Knobs of :func:`build_sculpt_head`; every length in metres, every blend ``*_k`` too.
    Paired features (``cheek_c``, ``jaw_angle`` ...) are given for x > 0 and mirrored."""
    preset: str = "sculpt"
    grid: str = "fine"               # "fine" (5 mm) | "coarse" (10 mm, upsampled x2)
    eyes: str = "balls"              # "balls" | "dots" | "shades"
    hair_style: str = "quiff"        # "quiff" | "short" | "slick" | "bald"
    skin_tone: str = "light"
    hair_tone: str = "black"
    iris_tone: str = "brown"
    # -- part extent (metres; x and z symmetric about 0, multiples of 1 cm so the coarse grid nests)
    extent_x: float = 0.10
    extent_y: tuple[float, float] = (-0.03, 0.33)
    extent_z: float = 0.13
    # -- cranium
    cranium_c: Vec3 = (0.0, 0.145, 0.004)
    cranium_r: Vec3 = (0.076, 0.090, 0.096)
    cranium_power: float = 2.4
    face_z: float = -0.086           # the forehead plane at the brow line
    forehead_tilt: float = 7.0       # degrees the plane recedes toward the crown
    forehead_k: float = 0.012
    # -- brow ridge
    brow_y: float = 0.138
    brow_half_w: float = 0.046
    brow_z: float = -0.084
    brow_r: float = 0.012
    brow_k: float = 0.012
    # -- mid-face
    face_c: Vec3 = (0.0, 0.098, -0.034)
    face_r: Vec3 = (0.064, 0.050, 0.052)
    face_k: float = 0.015
    cheek_c: Vec3 = (0.046, 0.106, -0.058)
    cheek_r: Vec3 = (0.025, 0.018, 0.028)
    cheek_k: float = 0.020
    hollow_c: Vec3 = (0.074, 0.076, -0.048)
    hollow_r: float = 0.020          # 0 disables the cheek hollow
    hollow_k: float = 0.015
    socket_c: Vec3 = (0.031, 0.116, -0.101)
    socket_r: Vec3 = (0.018, 0.011, 0.020)
    socket_depth: float = 0.003      # 0 disables the socket dents
    socket_k: float = 0.008
    # -- jaw and chin
    jaw_angle: Vec3 = (0.057, 0.082, 0.012)
    jaw_chin: Vec3 = (0.030, 0.022, -0.066)
    jaw_r: float = 0.020
    jaw_k: float = 0.012
    lower_c: Vec3 = (0.0, 0.050, -0.030)
    lower_r: Vec3 = (0.058, 0.046, 0.056)
    chin_c: Vec3 = (0.0, 0.024, -0.077)
    chin_r: float = 0.022
    chin_k: float = 0.012
    # -- nose
    nasion: Vec3 = (0.0, 0.124, -0.083)
    nose_tip: Vec3 = (0.0, 0.093, -0.103)
    nose_root_r: float = 0.007
    nose_end_r: float = 0.010
    nose_tip_r: float = 0.012
    wing_c: Vec3 = (0.013, 0.088, -0.091)
    wing_r: float = 0.0085
    nose_k: float = 0.008            # between the nose pieces
    nose_face_k: float = 0.010       # nose into the face
    # -- lips
    lips_y: float = 0.065
    lips_half_w: float = 0.020
    lips_z: float = -0.075
    lips_r: float = 0.009
    lips_k: float = 0.008
    philtrum_r: float = 0.004        # 0 disables
    philtrum_y: tuple[float, float] = (0.073, 0.083)
    philtrum_z: float = -0.082
    # -- ears
    ear_c: Vec3 = (0.076, 0.110, 0.012)
    ear_r: Vec3 = (0.010, 0.026, 0.016)
    ear_k: float = 0.006
    concha_c: Vec3 = (0.090, 0.108, 0.013)
    concha_r: float = 0.008          # 0 disables
    ear_inner_x: float = 0.065       # the hair / socket code treats |x| beyond this as the ear zone
    # -- neck
    neck_r: float = 0.042
    neck_z: float = 0.015
    neck_top: float = 0.060
    neck_k: float = 0.020
    # -- landmarks of the paint, the eyes and the hair
    eye_x: float = 0.030
    eye_y: float = 0.115
    hairline_y: float = 0.175
    crown_y: float = 0.235
    mouth_half_cols: float = 3.5     # fine |X| of the outer mouth column (3.5 -> 8 columns = 4 cm)
    lower_lip: bool = True
    brow_shade_rows: int = 1
    shade_rows: int = 2
    shades_y: tuple[float, float] = (0.105, 0.130)
    shades_half_w: float = 0.055
    shades_thick: int = 2            # voxels the lens follows the face where it curves away
    hair: dict = field(default_factory=dict)
    # -- eyeball parts (voxgen.refhead.carve_sockets reads these names)
    eye_radius: float = 0.0095
    iris_radius: float = 0.005
    pupil_radius: float = 0.0022
    eye_voxel: float = 0.003
    eye_protrude: float = 0.0
    eye_open_cols: int = 4
    eye_open_rows: int = 2

    @classmethod
    def preset_params(cls, preset: str = "sculpt", **overrides) -> "SculptParams":
        """``sculpt`` (the defaults) or ``sculpt_broad`` (a wider jaw and cheekbones, a heavier
        chin and brow: the broad-faced variant of the same construction)."""
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}, expected one of {PRESETS}")
        p = cls(preset=preset)
        if preset == "sculpt_broad":
            p = replace(p, cranium_r=(0.079, 0.090, 0.096), cranium_power=2.8,
                        jaw_angle=(0.061, 0.080, 0.012), jaw_r=0.023, jaw_chin=(0.034, 0.022, -0.066),
                        lower_r=(0.064, 0.046, 0.056), chin_r=0.025, cheek_c=(0.052, 0.108, -0.058),
                        cheek_r=(0.022, 0.017, 0.026), brow_r=0.013, face_r=(0.069, 0.050, 0.052),
                        ear_c=(0.079, 0.110, 0.012), ear_inner_x=0.068, concha_c=(0.093, 0.108, 0.013))
        return replace(p, **overrides)


# ---------------------------------------------------------------------------- the form
def form_field(grid, p: SculptParams) -> np.ndarray:
    """Signed distance (voxels, negative inside) of the whole head on ``grid`` (a Part or Grid
    of :mod:`voxgen.sdf`): the construction described in the module doc."""
    vs = float(grid.voxel_size)
    k = lambda m: m / vs
    mir = sdf.mirror_x(grid)
    tilt = math.radians(p.forehead_tilt)
    # cranium with the forehead plane
    cranium = _superellipsoid(grid, p.cranium_c, p.cranium_r, p.cranium_power)
    plane = sdf.half_space(grid, (0.0, p.brow_y, p.face_z), (0.0, math.sin(tilt), -math.cos(tilt)))
    head = sdf.smooth_intersect(cranium, plane, k(p.forehead_k))
    # brow ridge
    brow = sdf.capsule(grid, (-p.brow_half_w, p.brow_y, p.brow_z), (p.brow_half_w, p.brow_y, p.brow_z), p.brow_r)
    head = sdf.smooth_union(head, brow, k(p.brow_k))
    # mid-face and cheekbones
    head = sdf.smooth_union(head, sdf.ellipsoid(grid, p.face_c, p.face_r), k(p.face_k))
    head = sdf.smooth_union(head, sdf.ellipsoid(mir, p.cheek_c, p.cheek_r), k(p.cheek_k))
    # jaw: mandible capsules, the lower face between them, the chin bulge
    jaw = sdf.capsule(mir, p.jaw_angle, p.jaw_chin, p.jaw_r)
    jaw = sdf.smooth_union(jaw, sdf.ellipsoid(grid, p.lower_c, p.lower_r), k(p.jaw_k))
    head = sdf.smooth_union(head, jaw, k(p.jaw_k))
    head = sdf.smooth_union(head, sdf.sphere(grid, p.chin_c, p.chin_r), k(p.chin_k))
    # neck
    neck = sdf.cylinder(grid, (0.0, p.extent_y[0] - 0.02, p.neck_z), (0.0, p.neck_top, p.neck_z), p.neck_r)
    head = sdf.smooth_union(head, neck, k(p.neck_k))
    # dents: cheek hollow, eye sockets (before the nose and lips, which must stay whole)
    if p.hollow_r > 0:
        head = sdf.smooth_subtract(head, sdf.sphere(mir, p.hollow_c, p.hollow_r), k(p.hollow_k))
    if p.socket_depth > 0:
        head = sdf.smooth_subtract(head, sdf.ellipsoid(mir, p.socket_c, p.socket_r), k(p.socket_k))
    # nose: cone bridge, tip sphere, wings
    nose = sdf.cone(grid, p.nasion, p.nose_tip, p.nose_root_r, p.nose_end_r)
    nose = sdf.smooth_union(nose, sdf.sphere(grid, p.nose_tip, p.nose_tip_r), k(p.nose_k))
    nose = sdf.smooth_union(nose, sdf.sphere(mir, p.wing_c, p.wing_r), k(p.nose_k))
    head = sdf.smooth_union(head, nose, k(p.nose_face_k))
    # lips with the philtrum groove
    lips = sdf.capsule(grid, (-p.lips_half_w, p.lips_y, p.lips_z), (p.lips_half_w, p.lips_y, p.lips_z), p.lips_r)
    head = sdf.smooth_union(head, lips, k(p.lips_k))
    if p.philtrum_r > 0:
        groove = sdf.capsule(grid, (0.0, p.philtrum_y[0], p.philtrum_z), (0.0, p.philtrum_y[1], p.philtrum_z), p.philtrum_r)
        head = sdf.subtract(head, groove)
    # ears with the concha dent
    ear = sdf.ellipsoid(mir, p.ear_c, p.ear_r)
    if p.concha_r > 0:
        ear = sdf.subtract(ear, sdf.sphere(mir, p.concha_c, p.concha_r))
    head = sdf.smooth_union(head, ear, k(p.ear_k))
    return head


def _superellipsoid(grid, center: Vec3, radii: Vec3, power: float) -> np.ndarray:
    """Superellipsoid |x/rx|^p + |y/ry|^p + |z/rz|^p = 1 as an approximate SDF (voxels): the
    implicit value is converted to a distance along the ray from the centre, exact on the axes."""
    X, Y, Z = grid.coords()
    vs = np.float32(grid.voxel_size)
    cx, cy, cz = (np.float32(v) for v in center)
    rx, ry, rz = (np.float32(v) for v in radii)
    x, y, z = X - cx, Y - cy, Z - cz
    q = np.float32(power)
    f = (np.abs(x) / rx) ** q + (np.abs(y) / ry) ** q + (np.abs(z) / rz) ** q
    r = np.sqrt(x * x + y * y + z * z)
    t = np.maximum(f, 1e-9) ** (-1.0 / q)                  # the surface sits at t * (x, y, z)
    return ((r * (1.0 - t)) / vs).astype(np.float32)


def landmarks(p: SculptParams) -> dict:
    """Metre landmarks for :mod:`voxgen.hairstyle` and :func:`voxgen.refhead.carve_sockets`."""
    ex, ey, ez = p.ear_c
    rx, ry, rz = p.ear_r
    return {"brow_y": p.brow_y, "head_top": p.crown_y, "ear_z": (ez - rz, ez + rz), "ear_y": (ey - ry, ey + ry),
            "ear_inner_x": p.ear_inner_x, "eye_x": p.eye_x,
            "eye_l": (-p.eye_x, p.eye_y, p.face_z), "eye_r": (p.eye_x, p.eye_y, p.face_z)}


def head_part(p: SculptParams, voxel: float) -> Part:
    """An empty, x/z-centred head part over the extent (``nx`` even, so the mirror is exact)."""
    return Part.from_extent("head", (-p.extent_x, p.extent_y[0], -p.extent_z), (p.extent_x, p.extent_y[1], p.extent_z),
                            voxel, bone="head", symmetric=True)


def sculpt_form(p: SculptParams) -> tuple[Part, np.ndarray, dict]:
    """The solid of the head on the 5 mm part: sampled at 5 mm (``fine``) or at 10 mm, largest
    component, upsampled x2 (``coarse``). Mirror-exact, one component."""
    if p.grid not in GRIDS:
        raise ValueError(f"grid must be one of {GRIDS}, got {p.grid!r}")
    part = head_part(p, FINE_VOXEL)
    if p.grid == "fine":
        solid = form_field(part, p) <= 0.0
    else:
        coarse = head_part(p, COARSE_VOXEL)
        solid = form_field(coarse, p) <= 0.0
        labels, n = label_components(solid)
        if n > 1:
            solid = labels == 1
        solid = upsample(solid, 2)
        if solid.shape != part.shape:
            raise ValueError(f"coarse grid {coarse.shape} x2 does not nest in the fine grid {part.shape}: "
                             f"the extents must be multiples of 1 cm")
    solid &= solid[:, :, ::-1]
    labels, n = label_components(solid)
    crumbs = 0
    if n > 1:
        crumbs = int((labels > 1).sum())
        solid = labels == 1
    return part, solid, {"grid": p.grid, "components": int(n), "crumbs_removed": crumbs, "solid": int(solid.sum())}


def cartilage_mask(part: Part, p: SculptParams) -> np.ndarray:
    """Where the inner tissue is cartilage: the nose (bridge to wings) and the ears."""
    X, Y, Z = part.coords()
    ex, ey, ez = p.ear_c
    rx, ry, rz = p.ear_r
    ears = (np.abs(X) > p.ear_inner_x) & (np.abs(Y - ey) < ry + 0.004) & (np.abs(Z - ez) < rz + 0.004)
    nose = (np.abs(X) < p.wing_c[0] + p.wing_r + 0.002) & (Y > p.wing_c[1] - p.wing_r - 0.002) & (Y < p.nasion[1] + 0.004) \
        & (Z < p.face_z + 0.004)
    return ears | nose


# ---------------------------------------------------------------------------- paint (colour only)
def paint_face(mat: np.ndarray, col: np.ndarray, part: Part, p: SculptParams, C) -> dict[str, int]:
    """The box preset's painted face on the sculpted form, in place: mouth line (2 rows x 8
    columns) and the lower-lip tone, two nostril voxels under the nose, the shade band under the
    brow ridge (two rows under the sunglasses), the under-faces of the face and the neck in the
    shade tone, 6 x 2 brows in the hair tone (outer third one row higher), eye dots."""
    nz, ny, nx = mat.shape
    vs = part.voxel_size
    X = x_coords(nx)
    row = lambda y: int(math.floor((y - part.origin[1]) / vs + 1e-6))
    zidx = lambda z: int(math.floor((z - part.origin[2]) / vs + 1e-6))
    skin = mat == M.MAT_SKIN
    filled = mat != M.MAT_AIR
    front = front_face(filled) & skin
    ex = p.eye_x / vs                                       # fine |X| of the eye centre (6.0 at 5 mm)
    stats: dict[str, int] = {}

    my = row(p.lips_y - vs)                                 # the two rows straddling the lips' centre
    mouth = front & _rect(mat.shape, _cols(X, 0, p.mouth_half_cols), my, my + 1)
    col[mouth] = C("line")
    stats["mouth"] = int(mouth.sum())
    if p.lower_lip:
        lip = front & _rect(mat.shape, _cols(X, 0, p.mouth_half_cols - 1), my - 1, my - 1)
        col[lip] = C("lip_soft")
        stats["lower_lip"] = int(lip.sum())

    nose_base = p.wing_c[1] - p.wing_r
    nostril = front & _rect(mat.shape, _cols(X, 1.5, 1.5), row(nose_base) - 1, row(nose_base) - 1)
    col[nostril] = C("line")
    stats["nostril"] = int(nostril.sum())

    # the shade band under the brow ridge (under the sunglasses), the under-faces, the neck
    n_band = p.shade_rows if p.eyes == "shades" else p.brow_shade_rows
    b0 = (row(p.shades_y[0]) if p.eyes == "shades" else row(p.brow_y - p.brow_r)) - n_band
    face_cols = _cols(X, 0, p.ear_inner_x / vs)
    shade = front & face_cols[None, None, :] & _rect(mat.shape, _cols(X, 0, 1e9), b0, b0 + n_band - 1)
    stats["band"] = int(shade.sum())
    zi = np.arange(nz)[:, None, None]
    rows = np.arange(ny)[None, :, None]
    # under-faces only where a real overhang casts a shadow: the nose, the brow ridge, the chin
    # (every stair step of a curved form faces down too; shading them all would speckle the jaw)
    ax = np.abs(X)[None, None, :] * vs
    nose_zone = (ax < p.wing_c[0] + p.wing_r + 0.004) & (rows >= row(nose_base - 0.012)) & (rows <= row(p.nose_tip[1] + 0.006)) \
        & (zi < zidx(p.face_z + 0.004))
    brow_zone = (ax <= p.brow_half_w + 0.004) & (rows >= row(p.brow_y - p.brow_r - 0.006)) & (rows <= row(p.brow_y)) \
        & (zi < zidx(p.face_z + 0.010))
    chin_zone = (rows >= row(0.0)) & (rows <= row(p.chin_c[1] - 0.004)) & (zi < zidx(p.chin_c[2] + 0.03))
    under = skin & ~shift(filled, 0, 1, 0) & (nose_zone | brow_zone | chin_zone)
    shade |= under
    shade |= exposed(filled) & skin & (rows < row(0.0))
    shade &= col == C("skin")
    col[shade] = C("skin_shadow")
    stats["shade"] = int(shade.sum())

    if p.eyes != "shades":
        b = row(p.brow_y) - 1
        inner = _cols(X, ex - 2.5, ex + 0.5)
        outer = _cols(X, ex + 1.5, ex + 2.5)
        brow = front & (_rect(mat.shape, inner, b, b + 1) | _rect(mat.shape, outer, b + 1, b + 2))
        col[brow] = C("hair")
        stats["brow"] = int(brow.sum())
    if p.eyes in ("dots", "shades"):
        ey = row(p.eye_y - vs)
        dots = front & _rect(mat.shape, _cols(X, ex - 1.5, ex + 1.5), ey, ey + 1)
        col[dots] = C("eye_dark")
        stats["eye_dots"] = int(dots.sum())
    return stats


def hair_params(p: SculptParams) -> HairParams:
    """The style's defaults, the sculpt overrides, the hairline at the forehead plane's top, then
    the spec's ``hair`` overrides."""
    over = {**HAIR_COMMON, **HAIR.get(p.hair_style, {}), "hairline_y": p.hairline_y, **p.hair}
    return HairParams.for_style(p.hair_style, **over)


def shades_block(form: np.ndarray, part: Part, p: SculptParams) -> np.ndarray:
    """The sunglasses: a lens plane one voxel in front of the eye windows' front, spanning
    ``shades_half_w`` on the rows ``shades_y``, ``shades_thick`` voxels deep (less where the face
    is closer); where the face curves away from the plane the lens follows the local front (the
    wrap hugs the temples), where the face is in front of it (the nose bridge) the lens front
    bulges one voxel proud of the nose. Rows start above the nose tip, where the bridge's slope
    is one voxel per column, so the part stays one piece. Nothing inside the form."""
    nz, ny, nx = form.shape
    vs = part.voxel_size
    X = x_coords(nx)
    row = lambda y: int(math.floor((y - part.origin[1]) / vs + 1e-6))
    y0, y1 = row(p.shades_y[0]), row(p.shades_y[1]) - 1
    rect = _rect(form.shape, _cols(X, 0, p.shades_half_w / vs - 0.5), y0, y1)
    ff = front_face(form) & rect
    has = ff.any(axis=0)                                            # (ny, nx)
    zfront = np.where(has, np.argmax(ff, axis=0), nz)               # the face's front z per column
    win = _cols(X, (p.eye_x - 0.01) / vs, (p.eye_x + 0.01) / vs)
    z_plane = int(zfront[y0:y1 + 1][:, win].min()) - 1
    zi = np.arange(nz)[:, None, None]
    # the lens front: the plane, bulging forward where the face is in front of it (the nose
    # bridge); the lens is ``shades_thick`` deep from its front unless the face is closer
    lens_front = np.minimum(z_plane, zfront - 1).astype(np.int64)
    # each column also reaches forward to its x / y neighbours' front, so a one-voxel step of the
    # front (the nose's slope) never breaks the 6-connectivity of the one-voxel-thin bridge
    lf = np.where(has, lens_front, nz)
    reach = lf.copy()
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        reach = np.minimum(reach, np.roll(np.roll(lf, dy, axis=0), dx, axis=1))
    lens_front = np.where(has, np.minimum(lens_front, reach), nz)
    block = rect & has[None] & (zi >= lens_front[None]) & (zi < np.minimum(zfront, lens_front + p.shades_thick)[None]) & ~form
    block |= block[:, :, ::-1]
    labels, n = label_components(block)
    if n > 1:
        block = labels == 1
    return block


# ---------------------------------------------------------------------------- the builder
def build_sculpt_head(params: SculptParams | None = None, name: str | None = None) -> Model:
    """The sculpted head as a multi-part model in the local frame of the other heads: metres, the
    chin's bottom near y = 0, centred on x / z, face toward -Z. Parts ``head`` (5 mm, tissues by
    depth), ``eye_l`` / ``eye_r`` (3 mm balls) or ``shades``, ``hair`` (5 mm)."""
    from .multires_head import backfill_z, crop_part, mesh_stats, resample, seam_stats
    from .refhead import carve_sockets
    p = params if params is not None else SculptParams()
    if p.eyes not in EYES:
        raise ValueError(f"eyes must be one of {EYES}, got {p.eyes!r}")
    if p.hair_style not in HAIR_STYLES:
        raise ValueError(f"hair_style must be one of {HAIR_STYLES}, got {p.hair_style!r}")
    pal = make_palette(ToonHeadParams.preset_params("box", skin_tone=p.skin_tone, hair_tone=p.hair_tone,
                                                    iris_tone=p.iris_tone))
    C = pal.index
    part, form, finfo = sculpt_form(p)
    vs = part.voxel_size
    nz, ny, nx = part.shape
    grid = meshvox.VoxelGrid(tuple(part.origin), vs, part.dims)
    X, Y, Z = grid.centres()
    lm = landmarks(p)
    cart = cartilage_mask(part, p)
    stats: dict = {"form": finfo}

    # ---- hair on the scalp, the pocket one voxel deep under it
    hair = None
    hres = None
    if p.hair_style != "bald":
        hres = build_hair(form, grid, lm, hair_params(p))
        hair = hres.hair
        floor = hair_floor(X, Z, lm, hres.params)
        under = form & dilate(hair) & (Y >= floor) & ~ear_zone(X, Y, Z, lm)
        pocket = under | under[:, :, ::-1]
        form &= ~pocket
        hair |= pocket
        stats["hair"] = dict(hres.stats, pocket=int(pocket.sum()))

    # ---- sockets (before the tissues, so their walls are skin)
    balls: dict = {}
    inner = np.zeros(form.shape, bool)
    if p.eyes == "balls":
        form, balls, inner = carve_sockets(form, grid, lm, p, C)

    # ---- tissues by depth, then the painted face (colour only)
    mat = np.where(form, np.uint8(M.MAT_SKIN), np.uint8(0)).astype(np.uint8)
    col = np.where(form, np.uint8(C("skin")), np.uint8(0)).astype(np.uint8)
    depth = depth_map(form)
    for lvl, material, colour in ((2, M.MAT_FAT, "fat"), (3, M.MAT_MUSCLE, "muscle")):
        sel = form & (depth == lvl)
        mat[sel], col[sel] = material, C(colour)
    sel = form & (depth >= 4)
    mat[sel], col[sel] = M.MAT_BONE, C("bone")
    sel = form & cart & (depth >= 2)
    mat[sel], col[sel] = M.MAT_CARTILAGE, C("cartilage")
    stats["face"] = paint_face(mat, col, part, p, C)
    skin = mat == M.MAT_SKIN
    if inner.any():
        col[hole_walls(inner, skin)] = C("line")
    if p.eyes == "balls":
        Xf = x_coords(nx)
        ey = int(math.floor((p.eye_y - vs - part.origin[1]) / vs + 1e-6))
        half = (p.eye_open_cols - 1) / 2
        ex = p.eye_x / vs
        lid = front_face(form) & skin & _rect(form.shape, _cols(Xf, ex - half, ex + half), ey + 2, ey + 2)
        col[lid] = C("line")
        stats["face"]["lid"] = int(lid.sum())

    model = Model(name or f"sculpt_{p.eyes}_{p.hair_style}", vs, pal,
                  meta={"generator": "sculpthead", "preset": p.preset, "grid": p.grid, "eyes": p.eyes,
                        "hair_style": p.hair_style})
    head = part
    head.material[:] = mat
    head.color[:] = col
    model.add_part(head)
    model.meta["crumbs_removed"] = keep_largest_component(head)
    zero = (0.0, 0.0, 0.0)

    # ---- eyeballs with their stalks (as voxgen.refhead)
    for key, (egrid, emat, ecol, ball, centre) in balls.items():
        form_e = resample(form, grid, egrid)
        Xe, Ye, Ze = egrid.centres()
        vse = egrid.voxel_size
        axis = (np.abs(Xe - centre[0]) <= vse) & (np.abs(Ye - centre[1]) <= vse)
        back = ball & ~shift(ball, 0, 0, -1) & axis & (Ze > centre[2])
        stalk = backfill_z(back, form_e, 8) & ~ball
        emat[stalk] = M.MAT_EYE
        ecol[stalk] = C("eye_white")
        model.add_part(crop_part(key, egrid, emat, ecol, zero, key, False))
        stats[key] = {"ball": int(ball.sum()), "stalk": int(stalk.sum())}

    # ---- sunglasses
    if p.eyes == "shades":
        block = shades_block(form, part, p)
        smat = np.where(block, M.MAT_GLASS, 0).astype(np.uint8)
        scol = np.where(block, C("shade_dark"), 0).astype(np.uint8)
        model.add_part(crop_part("shades", grid, smat, scol, zero, "head", True))
        stats["shades"] = int(block.sum())

    # ---- the hair part: base, lit top faces, deep undersides / nape
    if hair is not None:
        hmat = np.where(hair, M.MAT_HAIR, 0).astype(np.uint8)
        hcol = np.where(hair, C("hair"), 0).astype(np.uint8)
        hcol[hres.deep & hair] = C("hair_deep")
        hcol[hres.lit & hair] = C("hair_light")
        model.add_part(crop_part("hair", grid, hmat, hcol, zero, "head", True))

    model.meta["build"] = stats
    model.meta["seams"] = seam_stats(model)
    model.meta["mesh"] = mesh_stats(model)
    return model
