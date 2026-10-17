"""Designed voxel-art head: placed on a coarse 1 cm grid, never sampled from a mesh.

Why: a realistic mesh sampled into voxels gives random staircases, features smaller than a
voxel and an uncanny-valley read (the owner rejected every such head). A good hand-made voxel
character is *designed* on its grid: every primitive sits on integer coordinates, the steps
are regular (a quantized ellipsoid gives concentric rings), the face is a flat mask with a few
bold features, a handful of flat colours, strict mirror symmetry. So the head is designed at
1 cm (a skull is ~21 design voxels tall), then upsampled x2 (nearest) to the 5 mm game grid:
every step becomes 2 fine voxels, a deliberate step and never noise, while the engine still has
fine voxels to shave, cut and bruise.

Design space (1 cm voxels; integer coordinates are voxel *centres*):
  x: mirrored about 0; the grid is odd so the centre column is x = 0 (nose, chin, mouth centre),
     paired features use +-x (eyes, ears, brows, mouth corners),
  y: 0 = the chin's bottom row, rows grow upward; the neck rows are negative,
  z: 0 = the face plane (brow, eye lids, mouth, chin); the nose is at z < 0, the skull back ~ +17.
Form: superellipsoid cranium (``box_power`` 2 = ellipsoid, 4 = rounded box) clipped to z >= 1,
a rounded face box in front of it (the flat "mask" with the brow ridge at z = 0, the forehead
one step back), a narrower rounded jaw box, a chin block, a cylinder neck. Features relative to
the local front surface: stepped nose wedge, eyes recessed one voxel (white block, iris block
toward the nose, one highlight, a dark lid line above), brow bars (flush paint or protruding
hair), mouth line with raised or lowered corners, optional tooth row, ear blocks with a darker
hole. Hair is a designed mass, not a scalp cap: a thicker superellipsoid shell above a stepped
hairline (forehead - temple - over the ears - nape), optional bangs sheet over the forehead,
two or three protruding clumps; two tones (top-lit faces lighter). Skin: three flat tones (base,
painted shadow under the nose, the chin and the ears, optional ruddy cheeks), no noise.
Interior (fine grid, hair treated as air): BFS depth from the air -> skin (1), fat (2), muscle
(3), bone (>= 4); nose and ears are cartilage below the skin; eye and tooth voxels keep their
material through the depth of their design voxel.

The ``real`` preset (owner feedback on the first sheet: ``toon`` creepy with its anime eyes and
stub chin, ``normal`` flat) keeps the designed 1 cm form but with canon proportions (eyes at
mid-height, one eye width apart, mouth as wide as the pupils are apart, nose base a third of the
way from the eyes to the chin, ears from brow to nose base), a boxier cranium, a jaw one step
behind the face plane with the chin block flush (a real chin, not a stub), a slim tapered nose,
and draws the face at 5 mm with :mod:`voxgen.facedetail` (``fine_details``): human-sized eyes,
one-row brows, a bridge ridge, nostril dots, a mouth line with lips, an ear spot - the same
vocabulary as :mod:`voxgen.hybridhead`, so the two candidates differ only in the form.

The ``box`` preset (owner's reference: a voxel-art character whose head is a *box with chamfered
corners* - flat face plane, flat sides, box jaw, few big planes - and whose face is painted, not
sculpted) is built by :func:`build_box_head`: the form is designed on an *even* 1 cm grid (no
centre column, so the nose is a 2 cm block) as a chamfered cranium box (an SDF rounded box
quantized at the centres: every chamfer is a 45-degree stair of whole design voxels, never a
1-voxel terrace), one flat face plane (no cheek blocks, no socket recess), a narrower jaw box
with the chin flush, a cylinder neck, the nose one 2 x 3 x 1 block one row above the mouth, the
ears 1 x 3 x 2 blocks on the side planes. Everything else is colour on the 5 mm grid
(:func:`paint_box_face`): a 2 x 8 mouth line (+ lower-lip row), two nostril voxels under the
nose block, the shade tone on the row under the brows (a 2-row band under the sunglasses with
``eyes="shades"``), on every under-face of the face (under the nose, the chin) and on the neck,
6 x 2 brows in the hair tone with the outer third one row higher. Eyes: ``balls`` = 3 mm eyeball
parts in carved sockets with a 4 x 2 window and a lid line (:func:`voxgen.refhead.carve_sockets`),
``dots`` = 4 x 2 dark painted voxels, ``shades`` = a sunglasses part (MAT_GLASS) one fine voxel
proud of the face plane spanning both eyes, wrapping the chamfered corners (the dots are
painted under it, for when the glasses come off). Hair = a 5 mm part
from :func:`voxgen.hairstyle.build_hair` on the box scalp (offsets of one design voxel, a box
sweep, the floor snapped to 1 cm rows, the form carved one voxel under it): ``quiff`` (the 45
degree stair), ``short`` (a 1 cm cap) / ``slick`` (receding temples, a 2 cm nape; no comb
ridges, so the silhouette stays whole design voxels) or ``bald``. The model has the
parts ``head``, ``eye_l`` / ``eye_r``, ``shades``, ``hair``; ``meta["mesh"]`` / ``meta["seams"]``
carry the counts as the other multi-part heads.

Decisions (the specs are silent on a designed head): design voxel = 2 fine voxels (fixed);
the highlight is one design voxel (2x2 fine) on the same screen side for both eyes (colour only,
so the material field stays mirror-symmetric); brows are painted (skin) when flush and MAT_HAIR
when they protrude; stubble is a flat tint on a coarse beard mask.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import materials as M
from . import meshvox
from .body import dilate, exposed, shift
from .facedetail import FaceDetailParams, FaceLandmarks, _cols, _rect, apply_face_details, front_face, hole_walls, x_coords
from .hairstyle import HairParams, build_hair, ear_zone, hair_floor
from .head_from_mesh import depth_map, keep_largest_component, upsample
from .model import Model, Part
from .palette import Palette

#: fine voxels per design voxel (1 cm design grid -> 5 mm game grid)
UPSAMPLE = 2

# ---------------------------------------------------------------------------- colours
#: skin_tone -> (base, shadow, ruddy)
SKIN_TONES = {
    "light": ((232, 190, 160), (196, 146, 126), (236, 164, 146)),
    "pale": ((242, 216, 198), (204, 170, 156), (238, 182, 174)),
    "tan": ((206, 158, 118), (162, 114, 86), (214, 136, 112)),
    "dark": ((128, 86, 62), (90, 58, 42), (142, 88, 72)),
}
#: hair_tone -> (hair, hair_light)
HAIR_TONES = {
    "brown": ((62, 42, 30), (92, 66, 48)),
    "black": ((30, 28, 32), (58, 56, 62)),
    "blond": ((196, 158, 92), (228, 198, 130)),
    "red": ((152, 70, 34), (194, 106, 56)),
    "grey": ((148, 146, 142), (190, 188, 184)),
}
IRIS_TONES = {"blue": (66, 118, 186), "brown": (96, 60, 38), "green": (74, 128, 84)}

PRESETS = ("normal", "toon", "real", "box")
BOX_EYES = ("balls", "dots", "shades")
BOX_HAIR_STYLES = ("quiff", "short", "slick", "bald")
#: hairstyle.HairParams overrides of the box preset (metres): one-design-voxel offsets, a box
#: sweep (shape_power 8), floor levels snapped to 1 cm rows, temple / side boundaries on 1 cm
#: columns, no one-voxel comb ridges (the silhouette stays whole design voxels)
BOX_HAIR_COMMON = dict(top=0.01, side=0.01, back=0.01, ledge=0.01, shape_power=8.0, floor_quantum=0.01,
                       temple_x=0.05, temple_drop=0.01, side_in=0.01, ear_clearance=0.02, sideburn_depth=0.02)
BOX_HAIR = {
    "quiff": dict(lit_band=0.04),                   # lit: the teeth and the crest only, the stair stays dark
    "short": dict(ridges="", lit_band=0.02),
    "slick": dict(back=0.02, nape_drop=0.01, temple_drop=-0.01, ridges=""),   # swept back: receding temples, long nape
}
#: ear_size -> (width, height) of the ear block in design voxels (depth is always 2)
EAR_SIZES = {1: (1, 3), 2: (2, 4), 3: (1, 5)}
HAIR_STYLES = ("short", "bangs", "bald")
EXPRESSIONS = ("neutral", "stern", "happy")
#: expression -> (brow_angle: -1 inner ends lower, 0 flat, +1 inner ends higher;
#:                mouth_corners: -1 down, 0 flat, +1 up; teeth row shown)
EXPRESSION_TABLE = {"neutral": (-1, 1, False), "stern": (-1, -1, False), "happy": (0, 1, True)}


@dataclass
class ToonHeadParams:
    """Knobs of :func:`build_toon_head`, all form sizes in design voxels (1 cm).

    ``preset`` only names the defaults (see :meth:`preset_params`); every field can be
    overridden. ``brow_angle`` / ``mouth_corners`` default to the expression's values when None.
    """
    preset: str = "normal"
    # -- skull
    cranium_w: int = 15      # full width (odd), x radius = (w - 1) / 2
    cranium_h: int = 17      # full height of the cranium ellipsoid
    cranium_d: int = 18      # full depth
    box_power: float = 2.5   # 2 = ellipsoid, 4 = rounded box
    head_h: int = 21         # chin bottom to crown, rows 0 .. head_h - 1
    face_w: int = 13         # flat face mask width (odd)
    face_round: int = 2      # rounding radius of the face box
    face_y0: int = 5         # face box bottom row (the nose base); it reaches the brow top
    jaw_w: int = 11          # jaw box width (odd)
    jaw_h: int = 7           # jaw box rows 0 .. jaw_h - 1
    jaw_d: int = 10          # jaw box depth from its front
    jaw_z: int = 0           # front of the jaw box (1 = one step behind the face plane: the chin block then protrudes)
    chin: int = 1            # 0 none, 1 flush block filling the rounded chin, 2 also protrudes 1
    chin_w: int = 5          # chin block width (odd)
    neck_r: float = 3.5
    neck_rows: int = 3       # rows below the chin
    # -- features
    mouth_y: int = 3
    mouth_w: int = 5         # odd; the outermost voxels are the corners
    lips: bool = False       # lip-colour row under the mouth line
    nose_y: int = 6          # nose bottom row (the bridge row ends just under the eyes)
    nose_w: int = 3          # odd: width of the lower rows (the bridge row is one column)
    nose_h: int = 3
    nose_d: int = 2          # protrusion of the lower rows (the top row protrudes 1)
    nose_taper: bool = False # only the bottom row is nose_w wide; the rows above are one column (a slim nose with wings)
    eye_y: int = 9           # eye white bottom row
    eye_x: int = 2           # inner column of the eye white (|x|)
    eye_w: int = 3
    eye_h: int = 2
    iris: int = 2            # iris block size (n x n), placed toward the nose, top aligned
    iris_tone: str = "blue"
    brow_y: int = 14         # row of the outer brow end (one skin row above the lid line)
    brow_w: int = 4
    brow_relief: int = 0     # 0 flush paint, 1 protrudes one voxel (hair)
    brow_angle: int | None = None
    mouth_corners: int | None = None
    ear_y: int = 8           # ear bottom row
    ear_size: int = 1        # EAR_SIZES: 1 -> 1w x 3h, 2 -> 2w x 4h, 3 -> 1w x 5h (all 2 deep)
    ear_z: int = 7           # front row of the ear
    expression: str = "neutral"
    # -- hair
    hair_style: str = "short"
    hair_thick: int = 3      # on top
    hair_side: int = 1       # thickness on the sides
    hair_back: int = 2       # thickness at the back
    hairline_y: int = 18     # forehead hairline row (short); bangs hang below it
    bangs_y: int = 16        # bottom row of the bangs sheet (its centre steps one row lower)
    clumps: bool = True
    # -- colour
    skin_tone: str = "light"
    hair_tone: str = "brown"
    cheeks: bool = False
    stubble: bool = False
    # -- fine details (the ``real`` preset): the coarse eyes/brows/mouth/ear hole are replaced by
    # the 5 mm vocabulary of voxgen.facedetail, placed on the design rows above
    fine_details: bool = False
    fine_eye_x: float = 5.5          # |X| of the eye centres in fine voxels (half-integer)
    fine_mouth_half_w: float = 5.5   # |X| of the mouth corners in fine voxels
    detail: FaceDetailParams = field(default_factory=FaceDetailParams)
    # -- the ``box`` preset (ignored by the others): chamfered box form on an even grid, the face
    # painted at 5 mm, eyes as parts / dots / sunglasses, hair as a voxgen.hairstyle part
    eyes: str = "coarse"             # box: "balls" | "dots" | "shades"
    cranium_y0: int = 6              # bottom row of the cranium box (its chamfer hides behind the jaw)
    cranium_round: float = 2.0       # chamfer radius of the cranium box (design voxels)
    jaw_round: float = 1.0           # chamfer radius of the jaw box
    mouth_half_cols: float = 3.5     # fine |X| of the outer mouth column (3.5 -> 8 columns = 4 cm)
    lower_lip: bool = True           # one lip-tone row under the mouth line
    brow_shade_rows: int = 1         # fine rows of the shade band under the brows
    shade_rows: int = 2              # fine rows of the shade band under the sunglasses
    shades_rows: int = 3             # sunglasses height in design rows, from eye_y - 1 upward
    shades_half_w: float = 6.0       # sunglasses half-width in design voxels
    hair: dict = field(default_factory=dict)   # hairstyle.HairParams overrides
    eye_radius: float = 0.0095       # eyeball parts (see voxgen.refhead.carve_sockets)
    iris_radius: float = 0.005
    pupil_radius: float = 0.0022
    eye_voxel: float = 0.003
    eye_protrude: float = 0.0        # the ball's front pole this far in front of the face plane
    eye_open_cols: int = 4           # socket window in fine columns x rows
    eye_open_rows: int = 2

    @classmethod
    def preset_params(cls, preset: str = "normal", **overrides) -> "ToonHeadParams":
        """The proportion presets: ``normal`` (human proportions, simplified forms), ``toon``
        (bigger cranium, big eyes, small jaw, bigger nose and ears) and ``real`` (canon
        proportions - eyes at mid-height, one eye width apart, mouth as wide as the pupils are
        apart, nose base a third of the way from the eyes to the chin, ears from brow to nose
        base - with a moderate caricature: nose x1.3, heavier brow step, defined jaw and a wide
        chin, a boxier cranium, and the 5 mm fine-detail face instead of the big coarse eyes)
        and ``box`` (the reference: the proportions of ``real`` on a 14 x 23 x 18 cm chamfered
        box, the face painted, eyes ``balls`` and hair ``quiff`` by default; see the module doc)."""
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}, expected one of {PRESETS}")
        p = cls(preset=preset)
        if preset == "toon":
            p = replace(
                p, cranium_w=19, cranium_h=19, cranium_d=20, box_power=2.6, head_h=23,
                face_w=15, face_round=1, face_y0=4, jaw_w=9, jaw_h=6, jaw_d=9, chin=1,
                neck_r=2.5, mouth_y=2, mouth_w=5, nose_y=5, nose_w=3, nose_h=4, nose_d=3,
                eye_y=8, eye_x=2, eye_w=5, eye_h=4, iris=3, brow_y=15, brow_w=5, brow_relief=1,
                ear_y=7, ear_size=2, ear_z=8, hair_style="bangs", hair_thick=3, hair_side=2,
                hair_back=2, hairline_y=19, bangs_y=17, cheeks=True)
        elif preset == "real":
            p = replace(
                p, cranium_w=15, cranium_h=17, cranium_d=19, box_power=3.2, head_h=23,
                face_w=13, face_round=2, face_y0=7, jaw_w=9, jaw_h=7, jaw_d=9, jaw_z=1, chin=1, chin_w=7,
                neck_r=3.5, neck_rows=3, mouth_y=5, mouth_w=7, nose_y=7, nose_w=3, nose_h=4, nose_d=2,
                nose_taper=True,
                eye_y=11, eye_x=2, eye_w=3, eye_h=2, brow_y=13, brow_w=4, ear_y=8, ear_size=3, ear_z=7,
                hair_style="short", hair_thick=2, hair_side=1, hair_back=2, hairline_y=17, clumps=False,
                cheeks=True, fine_details=True)
        elif preset == "box":
            p = replace(
                p, cranium_w=14, cranium_h=17, cranium_d=18, head_h=23, face_w=14, jaw_w=12, jaw_h=10, jaw_d=14,
                jaw_z=0, chin=0, neck_r=4.0, neck_rows=3, mouth_y=5, nose_y=7, nose_w=2, nose_h=3, nose_d=1,
                eye_y=11, eye_x=2, eye_w=2, eye_h=1, brow_y=13, ear_y=9, ear_size=1, ear_z=7,
                hair_style="quiff", hairline_y=17, eyes="balls", hair_tone="black", iris_tone="brown",
                fine_eye_x=6.0, cheeks=False)
        return replace(p, **overrides)

    def resolved_expression(self) -> tuple[int, int, bool]:
        if self.expression not in EXPRESSION_TABLE:
            raise ValueError(f"unknown expression {self.expression!r}, expected one of {EXPRESSIONS}")
        brow, corners, teeth = EXPRESSION_TABLE[self.expression]
        if self.brow_angle is not None:
            brow = int(self.brow_angle)
        if self.mouth_corners is not None:
            corners = int(self.mouth_corners)
        return brow, corners, teeth


# ---------------------------------------------------------------------------- design grid
class Design:
    """Coarse design volume: materials + colour indices on the 1 cm grid, with integer
    centre coordinates ``X`` (mirrored about 0), ``Y`` (0 = chin row), ``Z`` (0 = face plane)."""

    def __init__(self, half_w: int, rows_below: int, rows_above: int, front: int, depth: int, even: bool = False):
        """``even``: no centre column, the columns sit at half-integer x (the box preset)."""
        self.hx, self.y0, self.z0 = int(half_w), int(rows_below), int(front)
        self.even = bool(even)
        nx, ny, nz = 2 * self.hx + (0 if even else 1), self.y0 + int(rows_above), self.z0 + int(depth)
        self.shape = (nz, ny, nx)
        self.X = (np.arange(nx) - self.hx + (0.5 if even else 0.0))[None, None, :]
        self.Y = (np.arange(ny) - self.y0)[None, :, None]
        self.Z = (np.arange(nz) - self.z0)[:, None, None]
        self.mat = np.zeros(self.shape, np.uint8)
        self.col = np.zeros(self.shape, np.uint8)
        self.cartilage = np.zeros(self.shape, bool)
        self.beard = np.zeros(self.shape, bool)

    # -- primitives (bool masks of the full shape) ----------------------------
    def box(self, x0: float, x1: float, y0: int, y1: int, z0: int, z1: int, r: float = 0.0) -> np.ndarray:
        """Axis box with inclusive bounds on voxel centres (half-integers on an even grid),
        edges rounded by ``r`` (an SDF rounded box quantized at the centres: regular steps,
        mirror-exact; r = 1 cuts the edge voxel, r = 2 a two-step 45-degree chamfer)."""
        X, Y, Z = self.X, self.Y, self.Z
        inside = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1) & (Z >= z0) & (Z <= z1)
        if r <= 0:
            return np.broadcast_to(inside, self.shape).copy()
        ex = np.maximum(0, np.abs(X - (x0 + x1) / 2) - ((x1 - x0) / 2 - r))
        ey = np.maximum(0, np.abs(Y - (y0 + y1) / 2) - ((y1 - y0) / 2 - r))
        ez = np.maximum(0, np.abs(Z - (z0 + z1) / 2) - ((z1 - z0) / 2 - r))
        return np.broadcast_to(inside & (ex ** 2 + ey ** 2 + ez ** 2 <= r * r + 1e-6), self.shape).copy()

    def ellipsoid(self, cy: float, cz: float, rx: float, ry: float, rz: float, power: float = 2.0) -> np.ndarray:
        """Superellipsoid centred on x = 0."""
        p = float(power)
        v = (np.abs(self.X) / rx) ** p + (np.abs(self.Y - cy) / ry) ** p + (np.abs(self.Z - cz) / rz) ** p
        return v <= 1.0 + 1e-6

    def cylinder_y(self, cz: float, r: float, y0: int, y1: int) -> np.ndarray:
        v = (self.X ** 2 + (self.Z - cz) ** 2 <= r * r + 1e-6) & (self.Y >= y0) & (self.Y <= y1)
        return np.broadcast_to(v, self.shape).copy()

    def mirror(self, mask: np.ndarray) -> np.ndarray:
        """mask | its x-mirror (paired features are designed on +x and mirrored)."""
        return mask | mask[:, :, ::-1]

    def columns(self, x0: int, x1: int, y0: int, y1: int, z0: int | None = None, z1: int | None = None) -> np.ndarray:
        """Mirrored rectangle |x| in [x0, x1], y in [y0, y1] (all z unless given)."""
        m = self.box(x0, x1, y0, y1, z0 if z0 is not None else -10 ** 6, z1 if z1 is not None else 10 ** 6)
        return self.mirror(m)

    # -- surface ----------------------------------------------------------------
    def solid(self) -> np.ndarray:
        return self.mat != M.MAT_AIR

    def front_face(self, solid: np.ndarray | None = None) -> np.ndarray:
        """Front-most solid voxel of every (y, x) column (looking from -z)."""
        s = self.solid() if solid is None else solid
        behind = np.cumsum(s, axis=0) - s          # solids in front of me along z
        return s & (behind == 0)

    def set(self, mask: np.ndarray, material: int, color: int) -> None:
        self.mat[mask] = np.uint8(material)
        self.col[mask] = np.uint8(color)


# ---------------------------------------------------------------------------- palette
def make_palette(p: ToonHeadParams) -> Palette:
    if p.skin_tone not in SKIN_TONES:
        raise ValueError(f"unknown skin_tone {p.skin_tone!r}, expected one of {tuple(SKIN_TONES)}")
    if p.hair_tone not in HAIR_TONES:
        raise ValueError(f"unknown hair_tone {p.hair_tone!r}, expected one of {tuple(HAIR_TONES)}")
    skin, shadow, ruddy = SKIN_TONES[p.skin_tone]
    hair, hair_light = HAIR_TONES[p.hair_tone]
    pal = Palette()
    pal.add("skin", skin)
    pal.add("skin_shadow", shadow)
    pal.add("skin_ruddy", ruddy)
    pal.add("stubble", tuple(int(round(c * f)) for c, f in zip(skin, (0.86, 0.86, 0.90))))
    pal.add("lips", tuple(int(round(0.58 * a + 0.42 * b)) for a, b in zip(skin, (168, 70, 78))))
    pal.add("line", (88, 54, 52))
    pal.add("hair", hair)
    if p.preset == "box":
        # the reference's tones: lighter top faces, a deep underside, soft lip, dark eye dots / lenses
        hair_light = tuple(int(round(0.7 * h + 0.3 * w)) for h, w in zip(hair, (255, 250, 245)))
    pal.add("hair_light", hair_light)
    if p.preset == "box":
        pal.add("hair_deep", tuple(int(round(c * 0.62)) for c in hair))
        lips = tuple(int(round(0.58 * a + 0.42 * b)) for a, b in zip(skin, (168, 70, 78)))
        pal.add("lip_soft", tuple(int(round(0.5 * a + 0.5 * b)) for a, b in zip(skin, lips)))
        pal.add("eye_dark", (34, 28, 32))
        pal.add("pupil", (24, 22, 30))
        pal.add("shade_dark", (26, 24, 28))
    pal.add("eye_white", (242, 242, 236))
    pal.add("iris", IRIS_TONES.get(p.iris_tone, IRIS_TONES["blue"]))
    pal.add("highlight", (248, 250, 252))
    pal.add("tooth", (244, 240, 224))
    pal.add("fat", (238, 214, 140))
    pal.add("muscle", (150, 40, 45))
    pal.add("bone", (235, 225, 200))
    pal.add("cartilage", (225, 185, 160))
    return pal


# ---------------------------------------------------------------------------- the design
def design_head(p: ToonHeadParams) -> Design:
    """Place every primitive of the head on the 1 cm design grid."""
    pal = make_palette(p)
    C = pal.index
    rx, ry, rz = (p.cranium_w - 1) / 2, p.cranium_h / 2, p.cranium_d / 2
    hair_t = p.hair_thick if p.hair_style != "bald" else 0
    half_w = int(max(rx, (p.face_w - 1) / 2) + hair_t + p.ear_size + 2)
    crown = p.head_h - 1
    d = Design(half_w, p.neck_rows, p.head_h + hair_t + 2, max(p.nose_d, 2) + 2, int(2 * rz) + hair_t + 2)
    brow_angle, corners, teeth = p.resolved_expression()

    # ---- skull form: cranium (clipped behind the forehead plane) + face mask + jaw + chin + neck
    skin = d.ellipsoid(crown - ry, rz, rx, ry, rz, p.box_power) & (d.Z >= 1)
    fw = (p.face_w - 1) // 2
    skin |= d.box(-fw, fw, p.face_y0, p.brow_y + 1, 0, int(rz), p.face_round)
    jw = (p.jaw_w - 1) // 2
    skin |= d.box(-jw, jw, 0, p.jaw_h - 1, p.jaw_z, p.jaw_z + p.jaw_d - 1, 2)
    if p.chin >= 1:
        cw = (p.chin_w - 1) // 2
        skin |= d.box(-cw, cw, 0, 1, 0 if p.chin == 1 else -1, 3)
    skin |= d.cylinder_y(rz, p.neck_r, -p.neck_rows, 4)
    d.set(skin, M.MAT_SKIN, C("skin"))

    # ---- nose: stepped wedge in front of the face plane; the top row (bridge) is one column
    # wide and protrudes 1, the lower rows are nose_w (odd) wide and protrude nose_d
    nxh = max(0, (int(p.nose_w) - 1) // 2)
    nose = np.zeros(d.shape, bool)
    for i in range(p.nose_h):                      # i = 0 is the bottom row
        bridge = i == p.nose_h - 1 and p.nose_h >= 3
        prot = 1 if bridge else max(1, p.nose_d - max(0, i - (p.nose_h - 3)))
        half = 0 if bridge or (p.nose_taper and i > 0) else nxh
        nose |= d.box(-half, half, p.nose_y + i, p.nose_y + i, -prot, -1)
    d.set(nose & ~d.solid(), M.MAT_SKIN, C("skin"))
    d.cartilage |= nose

    # ---- ears: blocks on the skull sides, 1-voxel darker hole on the outer face
    ear_w, ear_h = EAR_SIZES[max(1, min(p.ear_size, max(EAR_SIZES)))]
    ear_rows = (d.Y >= p.ear_y) & (d.Y <= p.ear_y + ear_h - 1) & (d.Z >= p.ear_z) & (d.Z <= p.ear_z + 1)
    side = np.where(d.solid() & ear_rows, np.abs(d.X), -1)
    widest = int(side.max())
    ear = d.mirror(d.box(widest + 1, widest + ear_w, p.ear_y, p.ear_y + ear_h - 1, p.ear_z, p.ear_z + 1))
    d.set(ear & ~d.solid(), M.MAT_SKIN, C("skin"))
    d.cartilage |= ear
    hole = d.mirror(d.box(widest + ear_w, widest + ear_w, p.ear_y + 1, p.ear_y + 1, p.ear_z + 1, p.ear_z + 1))
    if not p.fine_details:
        d.col[hole & ear] = C("skin_shadow")

    mw = (p.mouth_w - 1) // 2
    # ---- coarse features (the ``real`` preset draws them at 5 mm instead, see build_toon_head)
    if not p.fine_details:
        # ---- eyes: recess one voxel, white block, iris toward the nose (top aligned), one highlight
        eye_cols = d.columns(p.eye_x, p.eye_x + p.eye_w - 1, p.eye_y, p.eye_y + p.eye_h - 1)
        front = d.front_face() & eye_cols
        eye = shift(front, 0, 0, 1)
        d.set(front, M.MAT_AIR, 0)
        d.set(eye, M.MAT_EYE, C("eye_white"))
        n = min(p.iris, p.eye_w, p.eye_h)
        iris = eye & d.columns(p.eye_x, p.eye_x + n - 1, p.eye_y + p.eye_h - n, p.eye_y + p.eye_h - 1)
        d.col[iris] = C("iris")
        # highlight: top row, screen-left column of each iris (same side for both: one light source)
        top = p.eye_y + p.eye_h - 1
        hl = eye & (d.Y == top) & ((d.X == -(p.eye_x + n - 1)) | (d.X == p.eye_x))
        d.col[hl] = C("highlight")
        # dark lid line: the face voxel right above the white
        lid = d.front_face() & d.columns(p.eye_x, p.eye_x + p.eye_w - 1, p.eye_y + p.eye_h, p.eye_y + p.eye_h)
        d.col[lid & (d.mat == M.MAT_SKIN)] = C("line")

        # ---- brows: bars; the inner half one row lower (-1) or higher (+1)
        inner = p.brow_w // 2
        brow = np.zeros(d.shape, bool)
        for k in range(p.brow_w):
            x = p.eye_x + k
            y = p.brow_y + (brow_angle if k < inner else 0)
            brow |= d.columns(x, x, y, y)
        brow_face = d.front_face() & brow & (d.mat == M.MAT_SKIN)
        if p.brow_relief >= 1:
            bump = shift(brow_face, 0, 0, -1) & ~d.solid()
            d.set(bump, M.MAT_HAIR, C("hair"))
        else:
            d.col[brow_face] = C("hair")

        # ---- mouth: dark line, corners one row up (+1) / down (-1); lips row; teeth
        mouth = d.box(-(mw - 1), mw - 1, p.mouth_y, p.mouth_y, -10 ** 6, 10 ** 6)
        mouth |= d.columns(mw, mw, p.mouth_y + corners, p.mouth_y + corners)
        mouth_face = d.front_face() & mouth & (d.mat == M.MAT_SKIN)
        d.col[mouth_face] = C("line")
        if p.lips:
            lip = d.front_face() & d.box(-(mw - 2), mw - 2, p.mouth_y - 1, p.mouth_y - 1, -10 ** 6, 10 ** 6)
            d.col[lip & (d.mat == M.MAT_SKIN)] = C("lips")
        if teeth:
            tooth_cols = d.box(-(mw - 2), mw - 2, p.mouth_y + 1, p.mouth_y + 1, -10 ** 6, 10 ** 6)
            front = d.front_face() & tooth_cols & (d.mat == M.MAT_SKIN)
            d.set(front, M.MAT_AIR, 0)
            d.set(shift(front, 0, 0, 1), M.MAT_TOOTH, C("tooth"))

        if p.cheeks:
            cheek = d.front_face() & d.columns(p.eye_x + 1, p.eye_x + 2, p.nose_y, p.nose_y + 1)
            d.col[cheek & (d.col == C("skin"))] = C("skin_ruddy")

    # ---- painted shadow: the neck's top row (the chin's cast shadow)
    d.col[(d.mat == M.MAT_SKIN) & (d.Y == -1) & (d.col == C("skin"))] = C("skin_shadow")

    # ---- beard mask (fine-grid stubble tint): chin, jaw sides, upper lip; not the lips or neck
    chin_zone = (d.Y >= 0) & (d.Y <= p.mouth_y - 2)
    sides = (d.Y >= 0) & (d.Y <= p.nose_y - 1) & (np.abs(d.X) >= mw + 1)
    upper_lip = (d.Y >= p.mouth_y + 1 + (1 if teeth else 0)) & (d.Y <= p.nose_y - 2) & (np.abs(d.X) <= mw)
    d.beard = (chin_zone | sides | upper_lip) & (d.Z <= p.ear_z) & (d.mat == M.MAT_SKIN) & ~ear

    # ---- hair: anisotropic shell (thick on top, thin sides and back) above a stepped
    # hairline: forehead -> temple corner -> sideburn to mid-ear -> over the ear -> nape at the
    # ear's bottom; bangs sheet on the forehead; quiff, over-ear and nape clumps; two tones
    if hair_t > 0:
        shell = d.ellipsoid(crown - ry, rz, rx + p.hair_side, ry + hair_t, rz + p.hair_back, p.box_power)
        shell &= ~d.solid() & (d.Z >= 0)
        floor = np.select(
            [d.Z <= 1, d.Z <= 3, d.Z <= p.ear_z - 1, d.Z <= p.ear_z + 1],
            [p.hairline_y, p.hairline_y - 2, p.ear_y + 1, p.ear_y + ear_h + 1],
            default=p.ear_y + 1)
        # the nape hangs low only at the back centre; the sides stop above the ears
        floor = np.where((d.Z > p.ear_z + 1) & (np.abs(d.X) > max(2, int(rx) - 3)), p.ear_y + ear_h + 1, floor)
        hair = shell & (d.Y >= floor)
        if p.hair_style == "bangs":
            fb = fw - 1
            bangs = d.box(-fb, fb, p.bangs_y, p.hairline_y, 0, 0)
            bangs |= d.box(-(fb - 2), fb - 2, p.bangs_y - 1, p.bangs_y - 1, 0, 0)   # centre steps down
            bangs |= d.box(-(fb - 1), fb - 1, p.bangs_y + 2, p.hairline_y + 1, -1, -1)  # upper part swept forward
            hair |= bangs & ~d.solid()
        if p.clumps:
            # a quiff over the forehead (short cut) and a nape tuft; nothing sticks out past the ears
            clump = np.zeros(d.shape, bool)
            if p.hair_style != "bangs":
                clump |= d.box(-2, 2, p.hairline_y, p.hairline_y + 1, -1, -1)                   # quiff
            back = int(d.Z[np.nonzero(hair.any(axis=(1, 2)))[0], 0, 0].max()) if hair.any() else int(2 * rz)
            clump |= d.box(-1, 1, p.ear_y + 2, p.ear_y + 3, back + 1, back + 1)                 # nape
            hair |= clump
        hair &= ~d.solid()
        d.set(hair, M.MAT_HAIR, C("hair"))
        filled = d.solid()
        lit = (d.mat == M.MAT_HAIR) & ~shift(filled, 0, -1, 0)
        d.col[lit] = C("hair_light")
    return d


# ---------------------------------------------------------------------------- fine grid
def fine_landmarks(p: ToonHeadParams, d: Design) -> FaceLandmarks:
    """Design rows -> 5 mm landmarks for :func:`voxgen.facedetail.apply_face_details`: a design
    row ``Y`` covers the fine rows ``2 (Y + y0)`` and ``+ 1``; the eye centre is the top fine row
    of ``eye_y``, the bridge ridge runs from the eyes down to two rows above the nose base."""
    k = UPSAMPLE
    fy = lambda y: k * (int(y) + d.y0)
    ear_h = EAR_SIZES[max(1, min(p.ear_size, max(EAR_SIZES)))][1]
    return FaceLandmarks(
        eye_x=float(p.fine_eye_x), eye_y=fy(p.eye_y) + 1, brow_y=fy(p.brow_y) + 1,
        nose_top_y=fy(p.eye_y), nose_base_y=fy(p.nose_y), nostril_x=1.5,
        mouth_y=fy(p.mouth_y), mouth_half_w=float(p.fine_mouth_half_w),
        ear_y=fy(p.ear_y) + ear_h, ear_z=k * (p.ear_z + d.z0) + 1, cheek_y=fy(p.nose_y))


def fine_colors(C) -> dict[str, int]:
    return {"skin": C("skin"), "skin_shadow": C("skin_shadow"), "skin_ruddy": C("skin_ruddy"),
            "lips": C("lips"), "line": C("line"), "brow": C("hair"),
            "eye_white": C("eye_white"), "iris": C("iris"), "highlight": C("highlight")}


def fine_detail_params(p: ToonHeadParams) -> FaceDetailParams:
    """The expression drives the brow tilt and the mouth corners; ``cheeks`` the ruddy band."""
    brow_angle, corners, _teeth = p.resolved_expression()
    return replace(p.detail, brow_tilt=-brow_angle, mouth_corners=corners, cheeks=p.cheeks)


def build_toon_head(params: ToonHeadParams | None = None, fine_voxel: float = 0.005,
                    name: str | None = None) -> Model:
    """Design the head on the 1 cm grid, upsample x2 to ``fine_voxel`` and layer the inside.

    Returns a one-part model (``head``, bone ``head``) with the chin's bottom at y = 0, centred on
    x/z, face toward -Z. ``fine_voxel`` must be 5 mm (two fine voxels per design voxel).
    """
    p = params if params is not None else ToonHeadParams()
    if abs(fine_voxel * UPSAMPLE - 0.01) > 1e-9:
        raise ValueError("build_toon_head designs at 1 cm = 2 fine voxels: fine_voxel must be 0.005")
    if p.preset == "box":
        return build_box_head(p, fine_voxel, name)
    d = design_head(p)
    pal = make_palette(p)
    C = pal.index
    k = UPSAMPLE
    mat = upsample(d.mat, k)
    col = upsample(d.col, k)
    cart = upsample(d.cartilage, k)
    beard = upsample(d.beard, k)
    details: dict = {}
    if p.fine_details:
        details = apply_face_details(mat, col, fine_landmarks(p, d), fine_colors(C), fine_detail_params(p))

    # tissues by depth (hair is air for the skin: the scalp under it stays skin)
    tissue = mat != M.MAT_AIR
    depth = depth_map(tissue & (mat != M.MAT_HAIR))
    skin = mat == M.MAT_SKIN
    for lvl, material, colour in ((2, M.MAT_FAT, "fat"), (3, M.MAT_MUSCLE, "muscle")):
        sel = skin & (depth == lvl)
        mat[sel], col[sel] = material, C(colour)
    sel = skin & (depth >= 4)
    mat[sel], col[sel] = M.MAT_BONE, C("bone")
    sel = skin & cart & (depth >= 2)
    mat[sel], col[sel] = M.MAT_CARTILAGE, C("cartilage")
    if p.stubble:
        sel = beard & (mat == M.MAT_SKIN) & (depth == 1) & (col == C("skin"))
        col[sel] = C("stubble")

    nz, ny, nx = mat.shape
    vs = fine_voxel
    part = Part("head", (nx, ny, nz), vs, origin=(-nx * vs / 2, -d.y0 * k * vs, -nz * vs / 2),
                bone="head", symmetric=True)
    part.material[:] = mat
    part.color[:] = col
    model = Model(name or f"styl_{p.preset}", vs, pal,
                  meta={"generator": "toonhead", "preset": p.preset, "design_voxel_m": k * vs})
    model.add_part(part)
    model.meta["crumbs_removed"] = keep_largest_component(part)
    if details:
        model.meta["details"] = details
    return model


# ---------------------------------------------------------------------------- the box preset
def design_box_head(p: ToonHeadParams) -> Design:
    """The ``box`` form on the even 1 cm grid: a chamfered cranium box, one flat face plane, a
    narrower jaw box (chin flush, its chamfer the only step under the face), a cylinder neck,
    the nose one block, the ears blocks flush on the side planes. Nothing is painted here."""
    pal = make_palette(p)
    C = pal.index
    for key in ("cranium_w", "jaw_w", "nose_w"):
        if int(getattr(p, key)) % 2:
            raise ValueError(f"box preset: {key} must be even (the design grid has no centre column)")
    hx = p.cranium_w // 2                                 # columns at +-0.5 .. +-(hx - 0.5)
    crown = p.head_h - 1
    ear_w, ear_h = EAR_SIZES[max(1, min(p.ear_size, max(EAR_SIZES)))]
    d = Design(hx + ear_w + 2, p.neck_rows, p.head_h + 12, max(p.nose_d, 1) + 2, p.cranium_d + 4, even=True)   # 12 rows of hair room
    skin = d.box(-hx + 0.5, hx - 0.5, p.cranium_y0, crown, 0, p.cranium_d - 1, p.cranium_round)
    jx = p.jaw_w // 2
    skin |= d.box(-jx + 0.5, jx - 0.5, 0, p.jaw_h - 1, p.jaw_z, p.jaw_z + p.jaw_d - 1, p.jaw_round)
    skin |= d.cylinder_y((p.cranium_d - 1) / 2, p.neck_r, -p.neck_rows, p.jaw_h - 1)
    d.set(skin, M.MAT_SKIN, C("skin"))
    nxh = p.nose_w / 2
    nose = d.box(-nxh + 0.5, nxh - 0.5, p.nose_y, p.nose_y + p.nose_h - 1, -p.nose_d, -1)
    d.set(nose & ~d.solid(), M.MAT_SKIN, C("skin"))
    d.cartilage |= nose
    ear = d.mirror(d.box(hx + 0.5, hx + ear_w - 0.5, p.ear_y, p.ear_y + ear_h - 1, p.ear_z, p.ear_z + 1))
    d.set(ear & ~d.solid(), M.MAT_SKIN, C("skin"))
    d.cartilage |= ear
    return d


def paint_box_face(mat: np.ndarray, col: np.ndarray, p: ToonHeadParams, d: Design, C) -> dict[str, int]:
    """Colour-only features on the skin of the fine box form (in place): mouth line, lower lip,
    nostrils, the shade band, the under-faces and the neck in the shade tone, brows, eye dots."""
    nz, ny, nx = mat.shape
    k = UPSAMPLE
    X = x_coords(nx)
    fy = lambda y: k * (int(y) + d.y0)                     # lower fine row of a design row
    fz = lambda z: k * (int(z) + d.z0)                     # front fine z index of a design z
    skin = mat == M.MAT_SKIN
    filled = mat != M.MAT_AIR
    front = front_face(filled) & skin
    face_cols = _cols(X, 0, (p.cranium_w // 2) * k)
    stats: dict[str, int] = {}
    ex = float(p.fine_eye_x)
    ey = fy(p.eye_y) + 1                                   # top row of the eye window

    my = fy(p.mouth_y)
    mouth = front & _rect(mat.shape, _cols(X, 0, p.mouth_half_cols), my, my + 1)
    col[mouth] = C("line")
    stats["mouth"] = int(mouth.sum())
    if p.lower_lip:
        lip = front & _rect(mat.shape, _cols(X, 0, p.mouth_half_cols - 1), my - 1, my - 1)
        col[lip] = C("lip_soft")
        stats["lower_lip"] = int(lip.sum())

    nxo = p.nose_w / 2 * k - 0.5                           # the nose block's outer fine columns
    nostril = front & _rect(mat.shape, _cols(X, nxo, nxo), fy(p.nose_y) - 1, fy(p.nose_y) - 1)
    col[nostril] = C("line")
    stats["nostril"] = int(nostril.sum())

    # the shade band under the brows (under the sunglasses), the under-faces of the face, the neck
    n_band = p.shade_rows if p.eyes == "shades" else p.brow_shade_rows
    b0 = (fy(p.eye_y - 1) if p.eyes == "shades" else fy(p.brow_y)) - n_band
    shade = front & face_cols[None, None, :] & _rect(mat.shape, _cols(X, 0, 1e9), b0, b0 + n_band - 1)
    stats["band"] = int(shade.sum())
    zi = np.arange(nz)[:, None, None]
    rows = np.arange(ny)[None, :, None]
    under = skin & ~shift(filled, 0, 1, 0) & (zi < fz(p.ear_z)) & (rows >= fy(0)) & (rows <= fy(p.brow_y))
    shade |= under
    shade |= exposed(filled) & skin & (rows < fy(0))
    shade &= col == C("skin")
    col[shade] = C("skin_shadow")
    stats["shade"] = int(shade.sum())

    if p.eyes != "shades":
        b = fy(p.brow_y)
        inner = _cols(X, ex - 2.5, ex + 0.5)
        outer = _cols(X, ex + 1.5, ex + 2.5)
        brow = front & (_rect(mat.shape, inner, b, b + 1) | _rect(mat.shape, outer, b + 1, b + 2))
        col[brow] = C("hair")
        stats["brow"] = int(brow.sum())
    if p.eyes in ("dots", "shades"):
        dots = front & _rect(mat.shape, _cols(X, ex - 1.5, ex + 1.5), ey - 1, ey)
        col[dots] = C("eye_dark")
        stats["eye_dots"] = int(dots.sum())
    return stats


def box_hair_params(p: ToonHeadParams) -> HairParams:
    """The hair of the box preset: the style's defaults, the box overrides, the hairline on
    its design row, then the spec's ``hair`` overrides."""
    over = {**BOX_HAIR_COMMON, **BOX_HAIR[p.hair_style], "hairline_y": p.hairline_y * UPSAMPLE * 0.005, **p.hair}
    return HairParams.for_style(p.hair_style, **over)


def build_box_head(p: ToonHeadParams, fine_voxel: float = 0.005, name: str | None = None) -> Model:
    """The ``box`` preset: the designed form upsampled x2, the hair part on its scalp (the form
    carved one voxel under it), the eye sockets, the tissues by depth, the painted face, the
    sunglasses part. Parts ``head`` (+ ``eye_l`` / ``eye_r``, ``shades``, ``hair``) in the local
    frame of the other heads: metres, chin at y = 0, centred on x / z, face toward -Z."""
    from .multires_head import backfill_z, crop_part, mesh_stats, resample, seam_stats
    from .refhead import carve_sockets
    if p.eyes not in BOX_EYES:
        raise ValueError(f"box preset: eyes must be one of {BOX_EYES}, got {p.eyes!r}")
    if p.hair_style not in BOX_HAIR_STYLES:
        raise ValueError(f"box preset: hair_style must be one of {BOX_HAIR_STYLES}, got {p.hair_style!r}")
    d = design_box_head(p)
    pal = make_palette(p)
    C = pal.index
    k = UPSAMPLE
    vs = fine_voxel
    mat = upsample(d.mat, k)
    col = upsample(d.col, k)
    cart = upsample(d.cartilage, k)
    nz, ny, nx = mat.shape
    origin = (-nx * vs / 2, -d.y0 * k * vs, -nz * vs / 2)
    grid = meshvox.VoxelGrid(origin, vs, (nx, ny, nz))
    X, Y, Z = grid.centres()
    form = mat != M.MAT_AIR
    fy = lambda y: k * (int(y) + d.y0)
    zm = lambda z: origin[2] + k * (int(z) + d.z0) * vs   # metre z of the front face of a design z
    ear_h = EAR_SIZES[max(1, min(p.ear_size, max(EAR_SIZES)))][1]
    cm = k * vs
    eye_y = origin[1] + (fy(p.eye_y) + 1) * vs
    lm = {"brow_y": p.brow_y * cm, "head_top": p.head_h * cm, "ear_z": (zm(p.ear_z), zm(p.ear_z + 2)),
          "ear_y": (p.ear_y * cm, (p.ear_y + ear_h) * cm), "ear_inner_x": (p.cranium_w // 2) * cm,
          "eye_x": p.fine_eye_x * vs, "eye_l": (-p.fine_eye_x * vs, eye_y, zm(0)), "eye_r": (p.fine_eye_x * vs, eye_y, zm(0))}
    stats: dict = {}

    # ---- hair on the scalp, the pocket one voxel deep under it
    hair = None
    hres = None
    if p.hair_style != "bald":
        hres = build_hair(form, grid, lm, box_hair_params(p))
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
    mat = np.where(form, mat, np.uint8(0)).astype(np.uint8)
    col = np.where(form, col, np.uint8(0)).astype(np.uint8)

    # ---- tissues by depth, then the painted face (colour only)
    depth = depth_map(form)
    skin = mat == M.MAT_SKIN
    for lvl, material, colour in ((2, M.MAT_FAT, "fat"), (3, M.MAT_MUSCLE, "muscle")):
        sel = skin & (depth == lvl)
        mat[sel], col[sel] = material, C(colour)
    sel = skin & (depth >= 4)
    mat[sel], col[sel] = M.MAT_BONE, C("bone")
    sel = skin & cart & (depth >= 2)
    mat[sel], col[sel] = M.MAT_CARTILAGE, C("cartilage")
    stats["face"] = paint_box_face(mat, col, p, d, C)
    skin = mat == M.MAT_SKIN
    if inner.any():
        col[hole_walls(inner, skin)] = C("line")
    if p.eyes == "balls":
        Xf = x_coords(nx)
        ey = fy(p.eye_y) + 1
        half = (p.eye_open_cols - 1) / 2
        lid = front_face(form) & skin & _rect(form.shape, _cols(Xf, p.fine_eye_x - half, p.fine_eye_x + half), ey + 1, ey + 1)
        col[lid] = C("line")
        stats["face"]["lid"] = int(lid.sum())

    model = Model(name or f"styl_{p.preset}", vs, pal,
                  meta={"generator": "toonhead", "preset": p.preset, "design_voxel_m": cm, "eyes": p.eyes,
                        "hair_style": p.hair_style})
    head = Part("head", (nx, ny, nz), vs, origin=origin, bone="head", symmetric=True)
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

    # ---- sunglasses: one fine voxel proud of the front face, eye_y - 1 .. + shades_rows design rows
    if p.eyes == "shades":
        Xf = x_coords(nx)
        y0 = fy(p.eye_y - 1)
        rect = _rect(form.shape, _cols(Xf, 0, p.shades_half_w * k - 0.5), y0, y0 + p.shades_rows * k - 1)
        ff = front_face(form) & rect
        has = ff.any(axis=0)                                   # (ny, nx)
        zfront = np.where(has, np.argmax(ff, axis=0), nz)      # the face's front z per column
        zi = np.arange(nz)[:, None, None]
        # one voxel proud of the flattest part; at the chamfered corners the lenses wrap: filled
        # from that plane back to the local front (so the part stays one piece)
        block = rect & has[None] & (zi >= int(zfront[has].min()) - 1) & (zi < zfront[None]) & ~form
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
