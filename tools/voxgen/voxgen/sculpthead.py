"""Sculpted stylised head: a smooth SDF form in metres on a fine grid (5 mm, or 3 mm for the
town characters), never a box, never a mesh scan.

Why: the owner's reference (a voxel-art character with a quiff and sunglasses) is not built as
a chamfered box (``voxgen.toonhead`` ``box``) - its face is *constructed* like a stylised smooth
head: an egg-shaped cranium with a flat-ish forehead, a subtle brow ridge, defined cheekbones,
a jaw narrower than the cranium tapering to one rounded chin, the nose a rounded wedge with a
round tip and nostril wings, the lips a soft bulge with a philtrum, rounded ears. Human
proportions, mildly stylised: small human eyes, no monobrow, no hollow cheeks, no wide jaw, no
age lines (an older character is told by grey hair, never by wrinkles). The form is a sum of
smooth-unioned primitives (``voxgen.sdf``), every blend radius 8-24 mm so the surface has no
creases, no noise displacement. The head part also carries the exact ``sdf`` field
(:mod:`voxgen.sdfield`, int8 sixteenths of a voxel, carved where the hair pocket, the eye sockets,
the quantization and the crumb removal cut the form, so ``(sdf < 0) == solid`` exactly); the
renders never use it (the owner rejected smooth rendering: cuts go mushy) - it is data for the
engine, and the game look is the ``facet`` style (45-degree chamfers on one-voxel steps) on the
3 mm grid, where the chamfers are small and the form reads stylised-but-faceted.

Construction (:func:`form_field`, metres; Y up, chin's bottom near y = 0, face toward -Z,
mirror-symmetric about x = 0 by evaluating the paired primitives on ``sdf.mirror_x``):
  cranium    superellipsoid (``cranium_power`` 2.4: between an egg and a rounded box), the front
             cut by the tilted forehead plane (``face_z`` at the brow, receding ``forehead_tilt``
             degrees toward the crown) with a soft seam (``smooth_intersect``);
  brow ridge one capsule *per side* from ``brow_inner_x`` to ``brow_half_w`` (nothing over the
             nasion, so the two brows never read as one bar), r <= 8 mm, softly blended;
  mid-face   an ellipsoid (the maxilla: cheek to cheek, eyes to mouth), two cheekbone ellipsoids,
             an optional cheek hollow (``hollow_r``, off: the subtract read as nasolabial
             creases), optional shallow eye-socket dents (``socket_depth``);
  jaw        a mandible capsule per side from the jaw angle under the ear to beside the chin
             (the mandible ~0.85 of the cranium's width at the ear line), the lower-face
             ellipsoid filling between them, a chin sphere blended with a large radius so the
             underside of the jaw is one surface - one chin, not three lobes;
  nose       a cone bridge (root narrower than the end) from the nasion to the tip, a tip sphere,
             two wing spheres; blended into the face with its own radius; moderate by default;
  lips       a horizontal capsule ~2 mm proud of the surrounding face in the lower third of the
             face (one quarter of the nose-base-to-chin distance below the nose), a short
             philtrum groove subtracted between it and the nose (one voxel deep);
  ears       flattened ellipsoids on the skull sides with a concha dent;
  neck       a cylinder, blended into the jaw.
``grid="fine"`` samples the field at ``voxel``; ``grid="coarse"`` samples it at 2 x ``voxel``,
keeps the largest component and upsamples x2 (every step two fine voxels). ``quantize`` cleans
the binary form with the 26-neighbour rule of :func:`voxgen.hybridhead.quantize_form` (no
one-wide terrace, no lone pit) for the faceted look; the ``sdf`` field is carved to match.

Per-character knobs (dimensionless, resolved into the metre fields by :meth:`SculptParams.
resolved`): ``cranium_scale`` (w, h, d), ``jaw_width``, ``chin_size``, ``nose_size``,
``nose_width``, ``lip_fullness``, ``ear_size``, ``cheekbone``, ``neck_scale`` and the ``sex`` hint
(``"female"``: narrower jaw, smaller chin and nose, fuller lips, a smoother brow ridge, a
thinner neck). Everything else is the box preset's vocabulary: the head part carries skin / fat /
muscle / bone by depth in metres (nose and ears cartilage), the face is colour only
(:func:`paint_face`: a 2-row mouth line 5 cm wide + lower-lip tone, nostril dots under the nose,
the shade band under the brow ridge (or the sunglasses), the under-faces and the neck in the
shade tone, two separate brow strips in the hair tone with a gap of >= 2 x ``brow_inner_x``),
eyes ``balls`` (3 mm eyeball parts in carved sockets, ``voxgen.refhead.carve_sockets``; the
opening 20 x 10 mm whatever the grid) / ``dots`` / ``shades``, hair a ``voxgen.hairstyle`` part
on the sculpted scalp at the head's voxel size (hairline at the top of the forehead plane, the
form carved one voxel under it), and the facial-hair vocabulary of :mod:`voxgen.hairstyle`:
``beard`` (``stubble`` paints the jaw mask, ``short`` / ``full`` / ``mustache`` / ``goatee`` are a
hair-material part ``beard``), ``tonsure`` (bald crown disc in the hair) and ``headscarf`` (a
cloth part ``scarf`` over the hair with a tail at the nape; a band of hair shows under the hem).
Parts ``head``, ``eye_l`` / ``eye_r`` or ``shades``, ``hair``, ``beard``, ``scarf``;
``meta["build"]``, ``meta["seams"]``, ``meta["mesh"]`` as the other multi-part heads.

Decisions (the specs are silent on a sculpted head): tissue depths in metres (skin 5 mm, fat to
13 mm, muscle to 20 mm, bone deeper; cartilage in the nose and ears), so the 3 mm head keeps a
two-voxel skin for the razor; the palette is the box preset's (``toonhead.make_palette``); the
painter is a metre-landmark version of ``toonhead.paint_box_face``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from . import materials as M
from . import meshvox, sdf, sdfield
from .body import dilate, exposed, shift
from .checks import label_components
from .facedetail import _cols, _rect, front_face, hole_walls, x_coords
from .hairstyle import (BEARD_STYLES, HairParams, beard_mask, build_hair, ear_zone, hair_floor, headscarf_mask,
                        jaw_mask)
from .head_from_mesh import depth_map, keep_largest_component, upsample
from .model import Model, Part
from .toonhead import ToonHeadParams, make_palette

Vec3 = tuple[float, float, float]
PRESETS = ("sculpt", "sculpt_broad")
EYES = ("balls", "dots", "shades")
HAIR_STYLES = ("quiff", "short", "slick", "messy", "bald")
GRIDS = ("fine", "coarse")
SEXES = ("male", "female")
FINE_VOXEL = 0.005
COARSE_VOXEL = 0.010
#: hairstyle.HairParams overrides shared by every style on the sculpted scalp (metres): a round
#: sweep hugging the skull, the temples on the forehead plane's edge (floor_quantum = the voxel)
HAIR_COMMON = dict(top=0.015, side=0.010, back=0.012, ledge=0.010, shape_power=4.0,
                   temple_x=0.045, temple_drop=0.010, side_in=0.006, ear_clearance=0.008, sideburn_depth=0.020)
#: flat-topped styles get no lit tone at all (lit_band < 0): at 3 mm the crown terrace is a wide
#: plateau and a lighter plateau reads as a bald disc; the lit tone stays on clumps and the quiff
HAIR = {
    "quiff": dict(lit_band=0.04),
    "short": dict(ridges="", lit_band=-1.0),
    "slick": dict(back=0.02, nape_drop=0.01, temple_drop=-0.005, ridges="", lit_band=-1.0),
    "messy": dict(lit_band=0.012),
}
#: tissue depths from the surface (metres): skin, fat, muscle; bone deeper than the last
TISSUE_DEPTHS = ((M.MAT_SKIN, 0.005, "skin"), (M.MAT_FAT, 0.013, "fat"), (M.MAT_MUSCLE, 0.020, "muscle"))


@dataclass
class SculptParams:
    """Knobs of :func:`build_sculpt_head`; every length in metres, every blend ``*_k`` too.
    Paired features (``cheek_c``, ``jaw_angle`` ...) are given for x > 0 and mirrored. The
    dimensionless knobs at the end are applied by :meth:`resolved`."""
    preset: str = "sculpt"
    voxel: float = FINE_VOXEL        # the head grid (5 mm; the town characters use 3 mm)
    grid: str = "fine"               # "fine" (voxel) | "coarse" (2 x voxel, upsampled x2)
    quantize: bool = False           # 26-neighbour cleanup of the binary form (no one-wide terraces)
    eyes: str = "balls"              # "balls" | "dots" | "shades"
    hair_style: str = "quiff"        # "quiff" | "short" | "slick" | "messy" | "bald"
    beard: str = ""                  # "" | "stubble" | "short" | "full" | "mustache" | "goatee"
    tonsure: float = 0.0             # > 0: radius of the bald crown disc (metres)
    headscarf: bool = False          # a cloth part over the hair
    scarf_tones: tuple = ((74, 72, 84), (52, 50, 60))   # (cloth, hem) sRGB of the scarf
    skin_tone: str = "light"
    hair_tone: str = "black"
    iris_tone: str = "brown"
    # -- part extent (metres; x and z symmetric about 0; the dims are rounded so a coarse grid nests)
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
    # -- brow ridge (one capsule per side, nothing over the nasion)
    brow_y: float = 0.138
    brow_inner_x: float = 0.014      # the ridge and the painted strip start here (gap = 2 x this)
    brow_half_w: float = 0.046
    brow_z: float = -0.084
    brow_r: float = 0.007
    brow_k: float = 0.020
    # -- mid-face
    face_c: Vec3 = (0.0, 0.098, -0.034)
    face_r: Vec3 = (0.061, 0.050, 0.052)
    face_k: float = 0.015
    cheek_c: Vec3 = (0.045, 0.106, -0.058)
    cheek_r: Vec3 = (0.024, 0.018, 0.028)
    cheek_k: float = 0.020
    hollow_c: Vec3 = (0.074, 0.076, -0.048)
    hollow_r: float = 0.0            # 0 disables the cheek hollow (off: it read as creases)
    hollow_k: float = 0.015
    socket_c: Vec3 = (0.031, 0.116, -0.101)
    socket_r: Vec3 = (0.018, 0.011, 0.020)
    socket_depth: float = 0.003      # 0 disables the socket dents
    socket_k: float = 0.008
    # -- jaw and chin (the mandible ~0.85 of the cranium width at the ear line)
    jaw_angle: Vec3 = (0.047, 0.082, 0.012)
    jaw_chin: Vec3 = (0.022, 0.024, -0.064)
    jaw_r: float = 0.018
    jaw_k: float = 0.018
    lower_c: Vec3 = (0.0, 0.052, -0.030)
    lower_r: Vec3 = (0.052, 0.046, 0.056)
    chin_c: Vec3 = (0.0, 0.024, -0.074)
    chin_r: float = 0.019
    chin_k: float = 0.024
    # -- nose
    nasion: Vec3 = (0.0, 0.124, -0.083)
    nose_tip: Vec3 = (0.0, 0.093, -0.101)
    nose_root_r: float = 0.007
    nose_end_r: float = 0.0095
    nose_tip_r: float = 0.011
    wing_c: Vec3 = (0.012, 0.088, -0.090)
    wing_r: float = 0.008
    nose_k: float = 0.008            # between the nose pieces
    nose_face_k: float = 0.010       # nose into the face
    # -- lips
    lips_y: float = 0.062
    lips_half_w: float = 0.025
    lips_z: float = -0.075
    lips_r: float = 0.008
    lips_k: float = 0.008
    philtrum_r: float = 0.004        # 0 disables
    philtrum_y: tuple[float, float] = (0.070, 0.080)
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
    # -- landmarks of the paint, the eyes and the hair (metres; the painter converts per grid)
    eye_x: float = 0.030
    eye_y: float = 0.115
    hairline_y: float = 0.175
    crown_y: float = 0.235
    mouth_half_w: float = 0.025      # the mouth line spans +- this (10 columns at 5 mm)
    mouth_rows: int = 2
    brow_rows: int = 2
    lower_lip: bool = True
    band_m: float = 0.004            # the shade band under the brow ridge is this tall (one row)
    shade_rows: int = 2              # rows under the sunglasses
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
    eye_open_w: float = 0.020        # the socket opening (metres): 4 x 2 columns at 5 mm, 7 x 3 at 3 mm
    eye_open_h: float = 0.010
    # -- per-character knobs (dimensionless, see ``resolved``)
    sex: str = "male"
    cranium_scale: Vec3 = (1.0, 1.0, 1.0)
    jaw_width: float = 1.0
    chin_size: float = 1.0
    nose_size: float = 1.0
    nose_width: float = 1.0
    lip_fullness: float = 1.0
    ear_size: float = 1.0
    cheekbone: float = 1.0
    neck_scale: float = 1.0

    @property
    def eye_open_cols(self) -> int:
        return max(2, int(round(self.eye_open_w / self.voxel)))

    @property
    def eye_open_rows(self) -> int:
        return max(1, int(round(self.eye_open_h / self.voxel)))

    @property
    def mouth_half_cols(self) -> float:
        """Fine |X| of the outer mouth column (half-integer column coordinates)."""
        return self.mouth_half_w / self.voxel - 0.5

    @classmethod
    def preset_params(cls, preset: str = "sculpt", **overrides) -> "SculptParams":
        """``sculpt`` (the defaults) or ``sculpt_broad`` (a wider jaw and cheekbones, a heavier
        chin: the broad-faced variant of the same construction, still within the jaw limit)."""
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}, expected one of {PRESETS}")
        p = cls(preset=preset)
        if preset == "sculpt_broad":
            p = replace(p, cranium_r=(0.079, 0.090, 0.096), cranium_power=2.8, jaw_width=1.08, chin_size=1.1,
                        cheekbone=1.15, ear_c=(0.079, 0.110, 0.012), ear_inner_x=0.068, concha_c=(0.093, 0.108, 0.013))
        return replace(p, **overrides)

    def resolved(self) -> "SculptParams":
        """The metre parameters with the dimensionless knobs and the ``sex`` hint applied (and
        reset to 1 / ``"male"``, so resolving twice changes nothing)."""
        if self.sex not in SEXES:
            raise ValueError(f"sex must be one of {SEXES}, got {self.sex!r}")
        p = self
        jaw, chin, nose, lips, brow_r, brow_k, neck = p.jaw_width, p.chin_size, p.nose_size, p.lip_fullness, p.brow_r, \
            p.brow_k, p.neck_scale
        if p.sex == "female":
            jaw *= 0.93
            chin *= 0.90
            nose *= 0.90
            lips *= 1.15
            brow_r *= 0.7
            brow_k *= 1.2
            neck *= 0.90
        sx, sy, sz = p.cranium_scale
        v = lambda a, b, c: (a, b, c)
        cran_c = p.cranium_c
        cran_r = v(p.cranium_r[0] * sx, p.cranium_r[1] * sy, p.cranium_r[2] * sz)
        crown = cran_c[1] + cran_r[1]
        jw = jaw * (1.0 + 0.5 * (sx - 1.0))                      # a wider cranium widens the jaw a little
        cb = p.cheekbone
        out = replace(
            p,
            sex="male", cranium_scale=(1.0, 1.0, 1.0), jaw_width=1.0, chin_size=1.0, nose_size=1.0, nose_width=1.0,
            lip_fullness=1.0, ear_size=1.0, cheekbone=1.0, neck_scale=1.0,
            cranium_r=cran_r, crown_y=crown,
            hairline_y=p.brow_y + (p.hairline_y - p.brow_y) * sy,
            brow_half_w=p.brow_half_w * sx, brow_r=brow_r, brow_k=brow_k,
            face_r=v(p.face_r[0] * (0.5 + 0.5 * jw) * sx, p.face_r[1], p.face_r[2]),
            cheek_c=v(p.cheek_c[0] * sx + 0.004 * (cb - 1.0), p.cheek_c[1], p.cheek_c[2]),
            cheek_r=v(p.cheek_r[0] * cb, p.cheek_r[1] * cb, p.cheek_r[2] * cb),
            jaw_angle=v(p.jaw_angle[0] * jw, p.jaw_angle[1], p.jaw_angle[2]),
            jaw_chin=v(p.jaw_chin[0] * jw, p.jaw_chin[1], p.jaw_chin[2]),
            jaw_r=p.jaw_r * (0.6 + 0.4 * jw),
            lower_r=v(p.lower_r[0] * jw, p.lower_r[1], p.lower_r[2]),
            chin_r=p.chin_r * chin, chin_c=v(p.chin_c[0], p.chin_c[1], p.chin_c[2] - 0.010 * (chin - 1.0)),
            nose_root_r=p.nose_root_r * nose, nose_end_r=p.nose_end_r * nose * math.sqrt(p.nose_width),
            nose_tip_r=p.nose_tip_r * nose, wing_r=p.wing_r * nose * math.sqrt(p.nose_width),
            nose_tip=v(p.nose_tip[0], p.nose_tip[1], p.nose_tip[2] - 0.012 * (nose - 1.0)),
            wing_c=v(p.wing_c[0] * p.nose_width * nose, p.wing_c[1], p.wing_c[2] - 0.006 * (nose - 1.0)),
            lips_r=p.lips_r * lips, lips_z=p.lips_z - 0.003 * (lips - 1.0),
            ear_c=v(p.ear_c[0] * sx, p.ear_c[1], p.ear_c[2]),
            ear_r=v(p.ear_r[0], p.ear_r[1] * p.ear_size, p.ear_r[2] * p.ear_size),
            ear_inner_x=p.ear_inner_x * sx, concha_c=v(p.concha_c[0] * sx, p.concha_c[1], p.concha_c[2]),
            concha_r=p.concha_r * p.ear_size,
            neck_r=p.neck_r * neck,
            eye_x=p.eye_x * (0.5 + 0.5 * sx),
        )
        ear_edge = out.ear_c[0] + out.ear_r[0] + 0.008
        if ear_edge > out.extent_x:
            out = replace(out, extent_x=math.ceil(ear_edge / 0.01) * 0.01)
        return out


# ---------------------------------------------------------------------------- the form
def form_field(grid, p: SculptParams) -> np.ndarray:
    """Signed distance (voxels, negative inside) of the whole head on ``grid`` (a Part or Grid
    of :mod:`voxgen.sdf`): the construction described in the module doc. ``p`` is used as given
    (call :meth:`SculptParams.resolved` first for the semantic knobs)."""
    vs = float(grid.voxel_size)
    k = lambda m: m / vs
    mir = sdf.mirror_x(grid)
    tilt = math.radians(p.forehead_tilt)
    # cranium with the forehead plane
    cranium = _superellipsoid(grid, p.cranium_c, p.cranium_r, p.cranium_power)
    plane = sdf.half_space(grid, (0.0, p.brow_y, p.face_z), (0.0, math.sin(tilt), -math.cos(tilt)))
    head = sdf.smooth_intersect(cranium, plane, k(p.forehead_k))
    # brow ridge: one capsule per side, the nasion left alone (no monobrow)
    brow = sdf.capsule(mir, (p.brow_inner_x, p.brow_y, p.brow_z), (p.brow_half_w, p.brow_y, p.brow_z), p.brow_r)
    head = sdf.smooth_union(head, brow, k(p.brow_k))
    # mid-face and cheekbones
    head = sdf.smooth_union(head, sdf.ellipsoid(grid, p.face_c, p.face_r), k(p.face_k))
    head = sdf.smooth_union(head, sdf.ellipsoid(mir, p.cheek_c, p.cheek_r), k(p.cheek_k))
    # jaw: mandible capsules, the lower face between them, one chin (large blend: one underside)
    jaw = sdf.capsule(mir, p.jaw_angle, p.jaw_chin, p.jaw_r)
    jaw = sdf.smooth_union(jaw, sdf.ellipsoid(grid, p.lower_c, p.lower_r), k(p.jaw_k))
    jaw = sdf.smooth_union(jaw, sdf.sphere(grid, p.chin_c, p.chin_r), k(p.chin_k))
    head = sdf.smooth_union(head, jaw, k(p.jaw_k))
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
    """Metre landmarks for :mod:`voxgen.hairstyle` (hair, beard, scarf) and
    :func:`voxgen.refhead.carve_sockets`."""
    ex, ey, ez = p.ear_c
    rx, ry, rz = p.ear_r
    return {"brow_y": p.brow_y, "head_top": p.crown_y, "ear_z": (ez - rz, ez + rz), "ear_y": (ey - ry, ey + ry),
            "ear_inner_x": p.ear_inner_x, "eye_x": p.eye_x,
            "eye_l": (-p.eye_x, p.eye_y, p.face_z), "eye_r": (p.eye_x, p.eye_y, p.face_z),
            "mouth_y": p.lips_y, "mouth_half_w": p.lips_half_w, "nose_base_y": p.wing_c[1] - p.wing_r,
            "chin_y": p.chin_c[1] - p.chin_r, "chin_z": p.chin_c[2], "neck_z": p.neck_z, "neck_r": p.neck_r,
            "crown_z": p.cranium_c[2]}


def head_dims(p: SculptParams, voxel: float) -> tuple[int, int, int]:
    """(nx, ny, nz) of the head part at ``voxel``: ``nx`` even (an exact mirror), every count a
    multiple of two of the coarse grid's counts, so ``grid="coarse"`` nests whatever the voxel."""
    cv = 2.0 * p.voxel
    nx_c = 2 * int(math.ceil(p.extent_x / cv - 1e-6))
    ny_c = int(math.ceil((p.extent_y[1] - p.extent_y[0]) / cv - 1e-6))
    nz_c = int(math.ceil(2.0 * p.extent_z / cv - 1e-6))
    f = int(round(cv / voxel))
    return (nx_c * f, ny_c * f, nz_c * f)


def head_part(p: SculptParams, voxel: float) -> Part:
    """An empty, x-centred head part over the extent (``nx`` even, so the mirror is exact)."""
    nx, ny, nz = head_dims(p, voxel)
    origin = (-nx * voxel / 2, p.extent_y[0], -p.extent_z)
    return Part("head", (nx, ny, nz), voxel, origin=origin, bone="head", symmetric=True)


def sculpt_form(p: SculptParams) -> tuple[Part, np.ndarray, np.ndarray | None, dict]:
    """The solid of the head on the fine part: sampled at ``voxel`` (``fine``) or at twice it,
    largest component, upsampled x2 (``coarse``); optionally quantized (26-neighbour rule).
    Mirror-exact, one component. Returns ``(part, solid, field, info)``; ``field`` is the exact
    signed distance in voxels on the fine grid (None for ``coarse``, whose blocks are the point).
    ``p`` must be resolved."""
    if p.grid not in GRIDS:
        raise ValueError(f"grid must be one of {GRIDS}, got {p.grid!r}")
    part = head_part(p, p.voxel)
    field = None
    if p.grid == "fine":
        field = form_field(part, p)
        solid = field <= 0.0
    else:
        coarse = head_part(p, 2.0 * p.voxel)
        solid = form_field(coarse, p) <= 0.0
        labels, n = label_components(solid)
        if n > 1:
            solid = labels == 1
        solid = upsample(solid, 2)
        if solid.shape != part.shape:
            raise ValueError(f"coarse grid {coarse.shape} x2 does not nest in the fine grid {part.shape}")
    solid &= solid[:, :, ::-1]
    info: dict = {"grid": p.grid, "voxel_mm": round(p.voxel * 1000, 2)}
    if p.quantize:
        from .hybridhead import quantize_form
        solid, qinfo = quantize_form(solid)
        info["quantize"] = qinfo
    labels, n = label_components(solid)
    crumbs = 0
    if n > 1:
        crumbs = int((labels > 1).sum())
        solid = labels == 1
    info.update(components=int(n), crumbs_removed=crumbs, solid=int(solid.sum()))
    return part, solid, field, info


def form_sdf(field: np.ndarray, solid: np.ndarray) -> np.ndarray:
    """The int8 ``sdf`` field of the finished head: the exact ``field`` (voxels) carved by the
    voxels the build removed from ``field <= 0`` (hair pocket, eye sockets, quantization, mirror,
    crumbs: ``max(d, -d_removed)``, the removed cubes' faces become the surface) and unioned with
    any it added, quantized consistently with ``solid`` (:func:`voxgen.sdfield.quantize`)."""
    d = np.asarray(field, np.float32)
    removed = (d <= 0.0) & ~solid
    added = solid & (d > 0.0)
    if removed.any():
        d = np.maximum(d, -sdfield.signed_distance_vox(removed))
    if added.any():
        d = np.minimum(d, sdfield.signed_distance_vox(added))
    return sdfield.quantize(d, solid)


def cartilage_mask(part: Part, p: SculptParams) -> np.ndarray:
    """Where the inner tissue is cartilage: the nose (bridge to wings) and the ears."""
    X, Y, Z = part.coords()
    ex, ey, ez = p.ear_c
    rx, ry, rz = p.ear_r
    ears = (np.abs(X) > p.ear_inner_x) & (np.abs(Y - ey) < ry + 0.004) & (np.abs(Z - ez) < rz + 0.004)
    nose = (np.abs(X) < p.wing_c[0] + p.wing_r + 0.002) & (Y > p.wing_c[1] - p.wing_r - 0.002) & (Y < p.nasion[1] + 0.004) \
        & (Z < p.face_z + 0.004)
    return ears | nose


def layer_tissues(mat: np.ndarray, col: np.ndarray, form: np.ndarray, cart: np.ndarray, vs: float, C) -> None:
    """Skin / fat / muscle / bone by depth in metres (``TISSUE_DEPTHS``: a tissue reaches to its
    depth, bone is deeper than the last), cartilage in ``cart`` below the skin; in place."""
    depth = depth_map(form)
    bounds = [int(math.ceil(d / vs - 1e-6)) for _, d, _ in TISSUE_DEPTHS]
    mat[form] = M.MAT_SKIN
    col[form] = C("skin")
    lo = bounds[0]
    for (material, _, colour), hi in zip(TISSUE_DEPTHS[1:], bounds[1:]):
        sel = form & (depth > lo) & (depth <= hi)
        mat[sel], col[sel] = material, C(colour)
        lo = hi
    sel = form & (depth > lo)
    mat[sel], col[sel] = M.MAT_BONE, C("bone")
    sel = form & cart & (depth > bounds[0])
    mat[sel], col[sel] = M.MAT_CARTILAGE, C("cartilage")


# ---------------------------------------------------------------------------- paint (colour only)
def paint_face(mat: np.ndarray, col: np.ndarray, part: Part, p: SculptParams, C) -> dict[str, int]:
    """The painted face on the sculpted form, in place, every landmark in metres: the mouth line
    (``mouth_rows`` rows x ``2 x mouth_half_w``) and the lower-lip tone, nostril dots under the
    nose (3-9 mm off the axis), the shade band under the brow ridge (``shade_rows`` under the
    sunglasses), the under-faces of the nose / brow / chin and the neck in the shade tone, two
    separate brow strips in the hair tone (``brow_inner_x`` .. ``brow_half_w``, ``brow_rows``
    thick, the outer third one row higher), eye dots."""
    nz, ny, nx = mat.shape
    vs = part.voxel_size
    X = x_coords(nx)
    row = lambda y: int(math.floor((y - part.origin[1]) / vs + 1e-6))
    zidx = lambda z: int(math.floor((z - part.origin[2]) / vs + 1e-6))
    cols_m = lambda x0, x1: _cols(X, x0 / vs, x1 / vs)      # columns whose centre |x| is within [x0, x1]
    skin = mat == M.MAT_SKIN
    filled = mat != M.MAT_AIR
    front = front_face(filled) & skin
    stats: dict[str, int] = {}

    my = row(p.lips_y) - p.mouth_rows // 2                   # the rows straddling the lips' centre
    mouth = front & _rect(mat.shape, _cols(X, 0, p.mouth_half_cols), my, my + p.mouth_rows - 1)
    col[mouth] = C("line")
    stats["mouth"] = int(mouth.sum())
    if p.lower_lip:
        lip = front & _rect(mat.shape, _cols(X, 0, p.mouth_half_cols - 1), my - 1, my - 1)
        col[lip] = C("lip_soft")
        stats["lower_lip"] = int(lip.sum())

    nose_base = p.wing_c[1] - p.wing_r
    nostril = front & _rect(mat.shape, cols_m(0.003, 0.009), row(nose_base) - 1, row(nose_base) - 1)
    col[nostril] = C("line")
    stats["nostril"] = int(nostril.sum())

    # the shade band under the brow ridge (under the sunglasses), the under-faces, the neck
    n_band = p.shade_rows if p.eyes == "shades" else max(1, int(round(p.band_m / vs)))
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
        x_in, x_out = p.brow_inner_x, p.brow_half_w
        x_mid = x_in + (x_out - x_in) * 2.0 / 3.0
        inner = cols_m(x_in, x_mid)
        outer = cols_m(x_mid, x_out) & ~inner
        brow = front & (_rect(mat.shape, inner, b, b + p.brow_rows - 1) | _rect(mat.shape, outer, b + 1, b + p.brow_rows))
        col[brow] = C("hair")
        stats["brow"] = int(brow.sum())
    if p.eyes in ("dots", "shades"):
        ey = row(p.eye_y - vs)
        dots = front & _rect(mat.shape, cols_m(p.eye_x - 0.0075, p.eye_x + 0.0075), ey, ey + 1)
        col[dots] = C("eye_dark")
        stats["eye_dots"] = int(dots.sum())
    return stats


def hair_params(p: SculptParams) -> HairParams:
    """The style's defaults, the sculpt overrides, the hairline at the forehead plane's top, the
    floor quantum at the head's voxel, the tonsure, the quantization, then the spec's ``hair``
    overrides."""
    over = {**HAIR_COMMON, **HAIR.get(p.hair_style, {}), "hairline_y": p.hairline_y, "floor_quantum": p.voxel,
            "tonsure": p.tonsure, "tonsure_z": p.cranium_c[2], "quantize": p.quantize, **p.hair}
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
    chin's bottom near y = 0, centred on x / z, face toward -Z. Parts ``head`` (tissues by
    depth, the ``sdf`` field on the fine grid), ``eye_l`` / ``eye_r`` (3 mm balls) or ``shades``,
    ``hair``, ``beard`` (hair material) and ``scarf`` (cloth) when asked."""
    from .multires_head import backfill_z, crop_part, mesh_stats, resample, seam_stats
    from .refhead import carve_sockets
    p = (params if params is not None else SculptParams()).resolved()
    if p.eyes not in EYES:
        raise ValueError(f"eyes must be one of {EYES}, got {p.eyes!r}")
    if p.hair_style not in HAIR_STYLES:
        raise ValueError(f"hair_style must be one of {HAIR_STYLES}, got {p.hair_style!r}")
    if p.beard and p.beard not in BEARD_STYLES:
        raise ValueError(f"beard must be one of {BEARD_STYLES}, got {p.beard!r}")
    pal = make_palette(ToonHeadParams.preset_params("box", skin_tone=p.skin_tone, hair_tone=p.hair_tone,
                                                    iris_tone=p.iris_tone))
    C = pal.index
    if p.headscarf:
        pal.add("cloth", p.scarf_tones[0])
        pal.add("cloth_hem", p.scarf_tones[1])
    part, form, field, finfo = sculpt_form(p)
    vs = part.voxel_size
    nz, ny, nx = part.shape
    grid = meshvox.VoxelGrid(tuple(part.origin), vs, part.dims)
    X, Y, Z = grid.centres()
    lm = landmarks(p)
    cart = cartilage_mask(part, p)
    stats: dict = {"form": finfo}

    # ---- hair on the scalp, the pocket one voxel deep under it; the scarf over the hair
    hair = None
    hres = None
    scarf = hem = None
    if p.hair_style != "bald":
        hres = build_hair(form, grid, lm, hair_params(p))
        hair = hres.hair
        floor = hair_floor(X, Z, lm, hres.params)
        under = form & dilate(hair) & (Y >= floor) & ~ear_zone(X, Y, Z, lm)
        pocket = under | under[:, :, ::-1]
        form &= ~pocket
        hair |= pocket
        stats["hair"] = dict(hres.stats, pocket=int(pocket.sum()))
        if p.headscarf:
            scarf, hem, sstats = headscarf_mask(form, hair, grid, lm, hres.params)
            stats["scarf"] = sstats
    elif p.headscarf:
        raise ValueError("a headscarf needs a hair style to cover (hair_style != 'bald')")

    # ---- sockets (before the tissues, so their walls are skin)
    balls: dict = {}
    inner = np.zeros(form.shape, bool)
    if p.eyes == "balls":
        form, balls, inner = carve_sockets(form, grid, lm, p, C)

    # ---- the beard geometry (on the final form, before the tissues: nothing of the form changes)
    beard = None
    if p.beard:
        beard, bstats = beard_mask(form, grid, lm, p.beard)
        stats["beard"] = bstats

    # ---- tissues by depth, then the painted face (colour only)
    mat = np.zeros(form.shape, np.uint8)
    col = np.zeros(form.shape, np.uint8)
    layer_tissues(mat, col, form, cart, vs, C)
    stats["face"] = paint_face(mat, col, part, p, C)
    skin = mat == M.MAT_SKIN
    if inner.any():
        col[hole_walls(inner, skin)] = C("line")
    if p.eyes == "balls":
        Xf = x_coords(nx)
        ey = int(math.floor((p.eye_y - vs - part.origin[1]) / vs + 1e-6))
        half = (p.eye_open_cols - 1) / 2
        ex = p.eye_x / vs
        lid = front_face(form) & skin & _rect(form.shape, _cols(Xf, ex - half, ex + half), ey + p.eye_open_rows, ey + p.eye_open_rows)
        col[lid] = C("line")
        stats["face"]["lid"] = int(lid.sum())
    if p.beard == "stubble":
        stub = jaw_mask(form, grid, lm) & skin & (col == C("skin"))
        col[stub] = C("stubble")
        stats["beard"] = {"style": "stubble", "painted": int(stub.sum())}

    model = Model(name or f"sculpt_{p.eyes}_{p.hair_style}", vs, pal,
                  meta={"generator": "sculpthead", "preset": p.preset, "grid": p.grid, "eyes": p.eyes,
                        "hair_style": p.hair_style, "voxel_mm": round(vs * 1000, 2)})
    head = part
    head.material[:] = mat
    head.color[:] = col
    model.add_part(head)
    model.meta["crumbs_removed"] = keep_largest_component(head)
    if field is not None:
        head.sdf = form_sdf(field, head.solid())
        if not sdfield.is_consistent(head.sdf, head.solid()):
            raise RuntimeError("sdf field disagrees with the head solid")
        stats["sdf"] = {"carved": int(((field <= 0.0) & ~head.solid()).sum()),
                        "min": int(head.sdf.min()), "max": int(head.sdf.max())}
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

    # ---- the beard part (hair material; the lower faces deep, as the hair)
    if beard is not None and beard.any():
        bmat = np.where(beard, M.MAT_HAIR, 0).astype(np.uint8)
        bcol = np.where(beard, C("hair"), 0).astype(np.uint8)
        bcol[beard & ~shift(beard | form, 0, 1, 0)] = C("hair_deep")
        model.add_part(crop_part("beard", grid, bmat, bcol, zero, "head", True))

    # ---- the scarf part (cloth over the hair, the hem and the tail's bottom darker)
    if scarf is not None and scarf.any():
        smat = np.where(scarf, M.MAT_CLOTH, 0).astype(np.uint8)
        scol = np.where(scarf, C("cloth"), 0).astype(np.uint8)
        scol[hem & scarf] = C("cloth_hem")
        model.add_part(crop_part("scarf", grid, smat, scol, zero, "head", True))

    model.meta["build"] = stats
    model.meta["seams"] = seam_stats(model)
    model.meta["mesh"] = mesh_stats(model)
    return model
