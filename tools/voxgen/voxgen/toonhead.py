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

Decisions (the specs are silent on a designed head): design voxel = 2 fine voxels (fixed);
the highlight is one design voxel (2x2 fine) on the same screen side for both eyes (colour only,
so the material field stays mirror-symmetric); brows are painted (skin) when flush and MAT_HAIR
when they protrude; stubble is a flat tint on a coarse beard mask.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import materials as M
from .body import shift
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

PRESETS = ("normal", "toon")
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
    jaw_d: int = 10          # jaw box depth from the face plane
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
    ear_size: int = 1        # 1 -> 1w x 3h x 2d, 2 -> 2w x 4h x 2d
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

    @classmethod
    def preset_params(cls, preset: str = "normal", **overrides) -> "ToonHeadParams":
        """The two proportion presets: ``normal`` (human proportions, simplified forms) and
        ``toon`` (bigger cranium, big eyes, small jaw, bigger nose and ears)."""
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

    def __init__(self, half_w: int, rows_below: int, rows_above: int, front: int, depth: int):
        self.hx, self.y0, self.z0 = int(half_w), int(rows_below), int(front)
        nx, ny, nz = 2 * self.hx + 1, self.y0 + int(rows_above), self.z0 + int(depth)
        self.shape = (nz, ny, nx)
        self.X = (np.arange(nx) - self.hx)[None, None, :]
        self.Y = (np.arange(ny) - self.y0)[None, :, None]
        self.Z = (np.arange(nz) - self.z0)[:, None, None]
        self.mat = np.zeros(self.shape, np.uint8)
        self.col = np.zeros(self.shape, np.uint8)
        self.cartilage = np.zeros(self.shape, bool)
        self.beard = np.zeros(self.shape, bool)

    # -- primitives (bool masks of the full shape) ----------------------------
    def box(self, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int, r: float = 0.0) -> np.ndarray:
        """Axis box with inclusive integer bounds on voxel centres, edges rounded by ``r``
        (an SDF rounded box quantized at the centres: regular steps, mirror-exact)."""
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
    pal.add("lips", tuple(int(round(0.7 * a + 0.3 * b)) for a, b in zip(skin, (168, 70, 78))))
    pal.add("line", (88, 54, 52))
    pal.add("hair", hair)
    pal.add("hair_light", hair_light)
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
    skin |= d.box(-jw, jw, 0, p.jaw_h - 1, 0, p.jaw_d - 1, 2)
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
        half = 0 if bridge else nxh
        nose |= d.box(-half, half, p.nose_y + i, p.nose_y + i, -prot, -1)
    d.set(nose & ~d.solid(), M.MAT_SKIN, C("skin"))
    d.cartilage |= nose

    # ---- ears: blocks on the skull sides, 1-voxel darker hole on the outer face
    ear_w, ear_h = (1, 3) if p.ear_size <= 1 else (2, 4)
    ear_rows = (d.Y >= p.ear_y) & (d.Y <= p.ear_y + ear_h - 1) & (d.Z >= p.ear_z) & (d.Z <= p.ear_z + 1)
    side = np.where(d.solid() & ear_rows, np.abs(d.X), -1)
    widest = int(side.max())
    ear = d.mirror(d.box(widest + 1, widest + ear_w, p.ear_y, p.ear_y + ear_h - 1, p.ear_z, p.ear_z + 1))
    d.set(ear & ~d.solid(), M.MAT_SKIN, C("skin"))
    d.cartilage |= ear
    hole = d.mirror(d.box(widest + ear_w, widest + ear_w, p.ear_y + 1, p.ear_y + 1, p.ear_z + 1, p.ear_z + 1))
    d.col[hole & ear] = C("skin_shadow")

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
    mw = (p.mouth_w - 1) // 2
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

    # ---- painted shadow: the neck's top row (the chin's cast shadow)
    d.col[(d.mat == M.MAT_SKIN) & (d.Y == -1) & (d.col == C("skin"))] = C("skin_shadow")
    if p.cheeks:
        cheek = d.front_face() & d.columns(p.eye_x + 1, p.eye_x + 2, p.nose_y, p.nose_y + 1)
        d.col[cheek & (d.col == C("skin"))] = C("skin_ruddy")

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
def build_toon_head(params: ToonHeadParams | None = None, fine_voxel: float = 0.005,
                    name: str | None = None) -> Model:
    """Design the head on the 1 cm grid, upsample x2 to ``fine_voxel`` and layer the inside.

    Returns a one-part model (``head``, bone ``head``) with the chin's bottom at y = 0, centred on
    x/z, face toward -Z. ``fine_voxel`` must be 5 mm (two fine voxels per design voxel).
    """
    p = params if params is not None else ToonHeadParams()
    if abs(fine_voxel * UPSAMPLE - 0.01) > 1e-9:
        raise ValueError("build_toon_head designs at 1 cm = 2 fine voxels: fine_voxel must be 0.005")
    d = design_head(p)
    pal = make_palette(p)
    C = pal.index
    k = UPSAMPLE
    mat = upsample(d.mat, k)
    col = upsample(d.col, k)
    cart = upsample(d.cartilage, k)
    beard = upsample(d.beard, k)

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
    return model
