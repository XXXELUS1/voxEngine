"""Designed voxel-art character (the owner's fixed style, 2026-10-17): a box head with a flat
painted face on a coarse design grid, slab hair, and a block body of normal human proportions.

Why: every sculpted / mesh-sampled head read as creepy to the owner; the reference he chose
is standard voxel art (a rounded-box head, a flat face, dark rectangle eyes with a lighter band
on top, thin brow lines, a small nose block, a tiny mouth line, hair as a mass of stacked slabs)
and the body reference is "just cubes, but the body structure is normal" (a voxel-knight game:
plain cubes, ~7 heads tall, block limbs, clothes as layered blocks). So this module *designs*
everything on integer grids and never samples a mesh or an SDF with sub-voxel noise.

Grids
-----
* Head: a coarse design grid of ``coarse`` metres (15 mm) for the proportions, drawn on the fine
  ``coarse / 5`` grid (3 mm) in the *skull frame* (``Frame``): x 0..44 (the centre column 22),
  y 0..54 (the chin's bottom row 0), z 0..49 (the face plane 0, the face toward -z). The skull is
  a box whose every edge is a 45-degree staircase of fine voxels (3 at the front vertical edges,
  4 on the top / chin edges, 8 on the top-back, corner facets) and whose jaw steps in 4 voxels
  per side under the ear line: rounded, never terraced. The face stays one flat plane: eyes are
  white 10x8 blocks (two coarse cells) with the dark lid rows on top, a 4x4 pupil under the lid
  and one gleam; brows are bars 16 wide, 3 or 5 rows, two voxels proud (hair material); the nose
  is two blocks (a 9x8 tip five voxels proud with softened front edges, a 5x6 bridge two proud);
  the mouth an 11x2 line with the corners a voxel up and a row of eight 2x2x2 teeth behind it;
  cheeks two warm tones in an ellipse with a dithered rim; ears 4x10x10 blocks with a 6x6 notch
  two deep. Hair is a shell 4 voxels thick on the skull (``chunky``: sparse 5x5 clumps two proud
  in the lighter tone; ``slabs``: two octagonal slabs and a quiff on the swept side with groove
  lines along the sweep), the hairline and nape ragged in 5-column steps, the temples bare, the
  sides to mid-ear. The beard is the jaw zone (the cheek line rising to the ear, a ragged top
  edge) grown 4 voxels (``short``) or 6 (``full``) with sparse clumps, the moustache a bar under
  the nose with the ends hanging past the mouth corners; stubble is a two-tone dot pattern. The
  beret is a slouching stack of octagonal slabs on a dark band with sparse crumples on top, the
  feather a separate part: a staircase of white blocks with barbs, its quill buried in the puff.
* Body: designed directly in 10 mm voxels (``body_voxel``), proportions of a 7.2-head adult:
  feet / shoes, shins, thighs, a torso box, shoulder joints, upper arms, forearms, hands (thin
  blocks with the thumb block proud at the front: the notch under it), a neck. Body frame: feet
  on y = 0, x mirrored about the plane x = 0 (an even grid), the torso's depth centred on z = 0.
  Clothes are layered blocks: shells one voxel outside the body (coat body + sleeves with the
  front split, lining columns, proud lapels and collar; apron slab; dress skirt) are MAT_CLOTH
  voxels added around the body; "painted" garments (trousers, shirt, tunic) are the body's
  outer voxel layer turned MAT_CLOTH with the skin kept underneath; shoes are MAT_LEATHER blocks.

Interior (both parts): depth from the air by 6-connected peeling -> skin, fat, muscle, bone
(nose and ears: cartilage below the skin; the eyes MAT_EYE two fine voxels deep; the teeth sit
in the fat band behind the mouth line), so the engine can cut, and the checks stay clean (no
exposed fat, one component per part, mirror symmetry).

Palette discipline: three skin tones (base, shadow, cheek) plus the cheek's half tone and two
stubble tones, two hair tones (base, lighter clumps / top slabs), two or three flat tones per
garment, eye white + lid + pupil + gleam, one mouth tone. Everything is mirror-symmetric except
the slab hair's sweep and the beret with its feather. Every pattern (ragged edges, clumps,
crumples) is a fixed arithmetic rule on the voxel index, never a random number.

Decisions (the specs are silent on a designed character): the metre scale follows the head's
coarse cell (15 mm -> skull 165 mm, character ~1.2 m; ``coarse=0.02`` gives a 1.6 m adult with
the same look); brows are hair material two voxels proud (shaveable); hats are separate parts
(straw = MAT_PLANT, beret / cap = MAT_CLOTH, the feather MAT_PLANT) and the hair under them
keeps its lower shell; a hat and the feather must touch what they sit on; the hair never enters
the ear notch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import materials as M
from .body import dilate, exposed
from .model import Model, Part
from .palette import Palette

#: fine voxels per coarse design cell of the head
UP = 5
#: head design cell (m); the fine head voxel is coarse / UP
COARSE = 0.015
#: body voxel (m): the body is designed directly on this grid
BODY_VOXEL = 0.01

HAIR_STYLES = ("slabs", "chunky", "short", "bald")
BEARDS = ("none", "stubble", "short", "full")
HATS = ("none", "beret", "straw", "cap")
OUTFITS = ("coat", "tunic", "apron", "dress")
SLEEVES = ("long", "rolled")

#: skin_tone -> (base, shadow, cheek)
SKIN_TONES = {
    "light": ((228, 184, 150), (196, 148, 118), (234, 172, 148)),
    "pale": ((240, 212, 192), (204, 170, 154), (242, 190, 178)),
    "tan": ((206, 158, 118), (164, 116, 86), (216, 142, 116)),
    "old": ((222, 192, 166), (184, 150, 126), (226, 180, 162)),
    "dark": ((128, 86, 62), (92, 58, 42), (142, 92, 74)),
    "warm": ((230, 166, 112), (184, 120, 76), (226, 150, 104)),
}
#: hair_tone -> (base, light)
HAIR_TONES = {
    "brown": ((88, 58, 40), (128, 90, 62)),
    "chestnut": ((76, 44, 26), (122, 74, 42)),
    "black": ((36, 32, 36), (66, 62, 70)),
    "blond": ((190, 150, 88), (222, 190, 124)),
    "red": ((150, 72, 36), (190, 108, 60)),
    "grey": ((150, 148, 144), (190, 188, 184)),
}
#: fixed tones shared by every character
COMMON_TONES = {
    "line": (70, 46, 40),
    "mouth": (112, 66, 52),
    "eye_lid": (30, 20, 20),
    "eye_white": (236, 226, 212),
    "eye_dark": (46, 30, 26),
    "eye_light": (150, 128, 120),
    "feather": (238, 226, 200),
    "feather_shade": (200, 180, 146),
    "tooth": (244, 240, 224),
    "fat": (238, 214, 140),
    "muscle": (150, 40, 45),
    "bone": (235, 225, 200),
    "cartilage": (225, 185, 160),
}
#: garment tone names per outfit (overridable through ``VoxCharParams.colors``)
GARMENT_TONES = {
    "coat": {"coat": (236, 232, 222), "coat_shade": (206, 198, 184), "lining": (118, 82, 56),
             "shirt": (240, 240, 236)},
    "tunic": {"tunic": (150, 120, 80), "tunic_shade": (118, 92, 60), "belt": (82, 56, 38)},
    "apron": {"tunic": (120, 96, 72), "tunic_shade": (96, 76, 56), "belt": (82, 56, 38),
              "apron": (226, 220, 206), "apron_shade": (196, 188, 172)},
    "dress": {"dress": (96, 72, 92), "dress_shade": (74, 54, 70), "bodice": (120, 94, 110)},
}
COMMON_GARMENT_TONES = {
    "trousers": (116, 150, 190), "trousers_hem": (150, 180, 212),
    "shoe": (236, 234, 230), "sole": (186, 182, 176),
    "straw": (222, 178, 76), "straw_band": (132, 92, 44),
    "cap": (74, 92, 120), "cap_shade": (54, 68, 92),
    "beret": (140, 36, 26), "beret_shade": (96, 22, 18), "beret_light": (170, 54, 40),
}


@dataclass
class VoxCharParams:
    name: str = "voxchar"
    skin_tone: str = "light"
    hair_tone: str = "brown"
    hair: str = "slabs"            # slabs | chunky | short | bald
    fringe_sweep: int = 0          # -1 / +1: the slabs' quiff and third slab on that side (the character's x sign; +x is the viewer's left)
    beard: str = "none"            # none | stubble | short | full
    mustache: bool = False         # a moustache block under the nose (part of the beard part)
    brows_thick: bool = False      # 5 fine rows instead of 3
    hat: str = "none"              # none | beret (with the feather part) | straw | cap
    outfit: str = "coat"           # coat | tunic | apron | dress
    sleeves: str = "long"          # long | rolled (forearms bare; coat sleeves are always long)
    coarse: float = COARSE         # head design cell (m); fine voxel = coarse / UP
    body_voxel: float = BODY_VOXEL
    colors: dict = field(default_factory=dict)   # tone name -> rgb overrides

    def validate(self) -> None:
        for key, known in (("hair", HAIR_STYLES), ("beard", BEARDS), ("hat", HATS), ("outfit", OUTFITS),
                           ("sleeves", SLEEVES)):
            if getattr(self, key) not in known:
                raise ValueError(f"{key} must be one of {known}, got {getattr(self, key)!r}")
        if self.skin_tone not in SKIN_TONES:
            raise ValueError(f"skin_tone must be one of {tuple(SKIN_TONES)}, got {self.skin_tone!r}")
        if self.hair_tone not in HAIR_TONES:
            raise ValueError(f"hair_tone must be one of {tuple(HAIR_TONES)}, got {self.hair_tone!r}")
        if self.fringe_sweep not in (-1, 0, 1):
            raise ValueError("fringe_sweep must be -1, 0 or +1")


# ---------------------------------------------------------------------------- grids
class Cells:
    """Integer design grid with inclusive ranges per axis; masks of shape (nz, ny, nx)."""

    def __init__(self, x: tuple[int, int], y: tuple[int, int], z: tuple[int, int]):
        self.lo = (int(x[0]), int(y[0]), int(z[0]))
        nx, ny, nz = (int(r[1] - r[0] + 1) for r in (x, y, z))
        self.shape = (nz, ny, nx)
        self.X = np.arange(x[0], x[1] + 1)[None, None, :]
        self.Y = np.arange(y[0], y[1] + 1)[None, :, None]
        self.Z = np.arange(z[0], z[1] + 1)[:, None, None]

    def box(self, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int, r: float = 0.0) -> np.ndarray:
        """Cells with centres inside the inclusive box; ``r`` rounds the edges (r = 1 cuts the
        edge cells: a one-cell 45-degree chamfer, mirror-exact)."""
        X, Y, Z = self.X, self.Y, self.Z
        inside = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1) & (Z >= z0) & (Z <= z1)
        if r <= 0:
            return np.broadcast_to(inside, self.shape).copy()
        ex = np.maximum(0, np.abs(X - (x0 + x1) / 2) - ((x1 - x0) / 2 - r))
        ey = np.maximum(0, np.abs(Y - (y0 + y1) / 2) - ((y1 - y0) / 2 - r))
        ez = np.maximum(0, np.abs(Z - (z0 + z1) / 2) - ((z1 - z0) / 2 - r))
        return np.broadcast_to(inside & (ex ** 2 + ey ** 2 + ez ** 2 <= r * r + 1e-6), self.shape).copy()

    def mirror(self, m: np.ndarray) -> np.ndarray:
        """m | its x-mirror (the grid must be symmetric about x = 0: odd with a centre column,
        or even with x = 0 on a cell boundary)."""
        return m | m[:, :, ::-1]

    def empty(self) -> np.ndarray:
        return np.zeros(self.shape, bool)


def mirror_x(m: np.ndarray) -> np.ndarray:
    return m | m[:, :, ::-1]


def front_face(solid: np.ndarray) -> np.ndarray:
    """Front-most solid voxel of every (y, x) column (looking from -z = index 0)."""
    return solid & (np.cumsum(solid, axis=0) == 1)


def under_face(solid: np.ndarray) -> np.ndarray:
    """Solid voxels whose -y neighbour is air (faces pointing down)."""
    below = np.zeros_like(solid)
    below[:, 1:, :] = solid[:, :-1, :]
    return solid & ~below


def depth_map(solid: np.ndarray, max_depth: int) -> np.ndarray:
    """Depth below the surface by 6-connected peeling: 0 = touches air, ..., ``max_depth`` =
    deeper than that; 255 outside the solid."""
    depth = np.full(solid.shape, max_depth, np.uint8)
    cur = solid.copy()
    for d in range(max_depth):
        ex = exposed(cur)
        depth[ex] = d
        cur &= ~ex
        if not cur.any():
            break
    depth[~solid] = 255
    return depth


def layer_tissues(part: Part, solid: np.ndarray, bands, C) -> np.ndarray:
    """Write skin / fat / muscle / bone by depth into ``part`` (bands = list of (material,
    thickness or None)). Returns the depth map."""
    maxd = sum(t for _, t in bands if t is not None)
    depth = depth_map(solid, maxd)
    lo = 0
    for mat, thick in bands:
        hi = maxd if thick is None else lo + thick
        sel = solid & (depth >= lo) & ((depth <= hi) if thick is None else (depth < hi))
        part.material[sel] = np.uint8(mat)
        part.color[sel] = np.uint8(C(M.name_of(mat)))
        lo = hi
    return depth


def crop_part(part: Part, margin: int = 1) -> Part:
    """The part cut down to its solid bounding box plus ``margin`` voxels, origin adjusted."""
    bb = part.bbox()
    if bb is None:
        return part
    (x0, y0, z0), (x1, y1, z1) = bb
    x0, y0, z0 = max(x0 - margin, 0), max(y0 - margin, 0), max(z0 - margin, 0)
    x1, y1, z1 = min(x1 + margin, part.nx - 1), min(y1 + margin, part.ny - 1), min(z1 + margin, part.nz - 1)
    out = Part(part.name, (x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1), part.voxel_size,
               origin=(part.origin[0] + x0 * part.voxel_size, part.origin[1] + y0 * part.voxel_size,
                       part.origin[2] + z0 * part.voxel_size),
               bone=part.bone, symmetric=part.symmetric, budget=part.budget)
    sl = (slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1))
    out.material = np.ascontiguousarray(part.material[sl])
    out.color = np.ascontiguousarray(part.color[sl])
    out.state = np.ascontiguousarray(part.state[sl])
    return out


# ---------------------------------------------------------------------------- palette
def make_palette(p: VoxCharParams) -> Palette:
    base, shadow, cheek = SKIN_TONES[p.skin_tone]
    hair, hair_light = HAIR_TONES[p.hair_tone]
    tones = {"skin": base, "skin_shadow": shadow, "skin_cheek": cheek,
             "skin_cheek_soft": tuple((a + b) // 2 for a, b in zip(base, cheek)),
             "stubble": tuple(int(round(c * f)) for c, f in zip(base, (0.80, 0.80, 0.86))),
             "stubble_soft": tuple(int(round(c * f)) for c, f in zip(base, (0.90, 0.90, 0.93))),
             "hair": hair, "hair_light": hair_light}
    tones.update(COMMON_TONES)
    tones.update(GARMENT_TONES[p.outfit])
    tones.update(COMMON_GARMENT_TONES)
    tones.update(p.colors)
    pal = Palette()
    for name, rgb in tones.items():
        pal.add(name, rgb)
    return pal


# ---------------------------------------------------------------------------- head
#: array index of skull-frame 0 on x, y, z; the array's shape (nz, ny, nx)
FX, FY, FZ = 20, 10, 14
HEAD_SHAPE = (82, 110, 85)
#: skull box in the skull frame: x 0..2*HW, y 0..TOP (the chin's bottom at 0), z 0..BACK (the face plane at 0)
HW, TOP, BACK = 22, 54, 49
#: head tissue bands in fine voxels of depth
HEAD_BANDS = [(M.MAT_SKIN, 2), (M.MAT_FAT, 2), (M.MAT_MUSCLE, 3), (M.MAT_BONE, None)]
#: eye block (fine): x of the right eye, rows; the pupil inside it; the lid rows on top
EYE_X, EYE_Y, PUPIL_X, PUPIL_Y, LID_Y = (29, 38), (33, 40), (32, 35), (35, 38), (39, 40)
#: ragged-edge offsets (fine rows) indexed by |x - HW| % 7 (mirror-exact)
RAG7 = np.array([0, 1, 0, 2, 1, 0, 1])
#: hairline / nape steps per 5-column bin (palindromes: mirror-exact)
PAT9 = np.array([0, 1, -1, 0, 1, 0, -1, 1, 0])
NAPE9 = np.array([0, 2, 1, 0, 1, 0, 1, 2, 0])
#: the beret's puff: a superellipsoid (half extents x, y, z; power) on the band, slouching to -x and back
BERET_PUFF = ((32.0, 27.0, 35.0), 2.5)
#: the feather's spine (x, half width) per 4-row segment from y = 59 (buried in the puff) up
FEATHER_SPINE = [(44, 1), (46, 1), (48, 2), (50, 3), (51, 3), (52, 3), (52, 3), (51, 2), (50, 1)]


class Frame:
    """The fine head grid in the skull frame: x 0..44 (the centre column 22), y 0..54 (the chin's
    bottom row 0), z 0..49 (the face plane 0, -z toward the viewer), with margins for the neck,
    hair, hat and feather. Masks have the array shape (nz, ny, nx)."""

    def __init__(self):
        nz, ny, nx = HEAD_SHAPE
        self.shape = HEAD_SHAPE
        self.X = (np.arange(nx) - FX)[None, None, :]
        self.Y = (np.arange(ny) - FY)[None, :, None]
        self.Z = (np.arange(nz) - FZ)[:, None, None]
        self.AX = np.abs(self.X - HW)

    def full(self, m) -> np.ndarray:
        return np.broadcast_to(m, self.shape).copy()

    def box(self, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int, r: int = 0, rt: int = 0) -> np.ndarray:
        """Inclusive box; ``r`` cuts the vertical edges 45 degrees (an octagonal plan), ``rt``
        the top edges."""
        X, Y, Z = self.X, self.Y, self.Z
        m = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1) & (Z >= z0) & (Z <= z1)
        ex = np.minimum(X - x0, x1 - X)
        ez = np.minimum(Z - z0, z1 - Z)
        if r > 0:
            m &= (ex + ez) >= r
        if rt > 0:
            m &= ((y1 - Y) + np.minimum(ex, ez)) >= rt
        return self.full(m)


def skull_form(f: Frame) -> np.ndarray:
    """The rounded box: 45-degree staircases of fine voxels on every edge (3 at the front vertical
    edges, 5 at the back ones, 4 on the top and chin edges, 8 on the top-back, 6 on the
    bottom-back), corner facets, the jaw 4 voxels narrower per side below the ear line (a 45-degree
    step, never a terrace)."""
    X, Y, Z = f.X, f.Y, f.Z
    hw = HW - np.clip(25 - Y, 0, 4)                  # a 45-degree step in to the jaw under the ear line
    dx = hw - np.abs(X - HW)
    yb, yt, zf, zb = Y, TOP - Y, Z, BACK - Z
    inside = (dx >= 0) & (yb >= 0) & (yt >= 0) & (zf >= 0) & (zb >= 0)
    cut = ((dx + zf < 3) | (dx + zb < 5) | (yt + zf < 4) | (yt + dx < 4) | (yt + zb < 8)
           | (yb + zf < 4) | (yb + dx < 4) | (yb + zb < 6)
           | (dx + yt + zf < 7) | (dx + yt + zb < 10) | (dx + yb + zf < 7) | (dx + yb + zb < 9))
    return f.full(inside & ~cut)


def head_origin(p: VoxCharParams) -> tuple[float, float, float]:
    """Metre origin of the head array: the centre column at x = 0, the chin's bottom at y = 0,
    the skull's depth centred on z = 0."""
    vs = p.coarse / UP
    return (-(FX + HW + 0.5) * vs, -FY * vs, -(FZ + (BACK + 1) / 2) * vs)


def main_component(m: np.ndarray) -> np.ndarray:
    """The largest 6-connected component of ``m`` (drops stray clumps a pattern left loose)."""
    from .checks import label_components
    labels, n = label_components(m)
    if n <= 1:
        return m
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def top_faces(m: np.ndarray) -> np.ndarray:
    above = np.zeros_like(m)
    above[:, :-1, :] = m[:, 1:, :]
    return m & ~above


def hair_mass(p: VoxCharParams, f: Frame, skull: np.ndarray, form: np.ndarray, ears: np.ndarray) -> np.ndarray:
    """The hair: a shell 3 voxels thick on the skull (thinner than the reference), bottoms per
    region (a ragged hairline on the forehead, the temples bare, the sides to mid-ear, the nape),
    then per style: ``chunky`` raises sparse 5x5 clumps 2 voxels (the lighter tone); ``slabs`` stacks an
    octagonal slab, a second one on the swept side and a quiff block there with groove lines
    along the sweep (the lighter tone on the swept slab and the quiff's top)."""
    X, Y, Z = f.X, f.Y, f.Z
    shell = dilate(skull, 3) & ~form & ~dilate(ears)       # the ears stay clear (no hair in the notch)
    front, back = Z < 0, Z > BACK
    bins = (X // 5) % 9
    s = p.fringe_sweep
    if p.hair == "slabs":
        steps = np.array([3, 2, 1, 1, 0, 0, -1, -1, -2])
        hl = 55 + (PAT9 if s == 0 else (steps if s < 0 else steps[::-1]))[bins]
        side = np.where(Z < 10, 46, 37)
        nape = 30 + 0 * X
    else:
        hl = 50 + PAT9[bins]
        side = np.select([Z < 10, Z < 20, Z < 30], [44, 33, 32], 28) + (5 if p.hair == "short" else 0)
        nape = 22 + NAPE9[bins]
    bottom = np.where(front, hl, np.where(back, nape, side))
    region = (Y >= bottom) & (Y <= (56 if p.hat != "none" else 58))
    hair = shell & f.full(region)
    if p.hair == "chunky":
        clumps = ((f.AX // 5) * 7 + (Y // 5) * 11 + (Z // 5) * 5) % 9 < 3          # sparse, mirror-exact
        hair |= dilate(skull, 5) & ~dilate(skull, 3) & ~dilate(ears, 2) & f.full(region & clumps & (Y <= 54))
        hair = main_component(hair)
    elif p.hair == "slabs" and p.hat == "none":
        hair |= f.box(-3, 47, 57, 62, -3, 52, r=8)                 # one slab on the whole shell
        if s < 0:
            hair |= f.box(-3, 26, 63, 66, 0, 46, r=7)              # the swept slab, joined to the quiff
            quiff = f.box(-3, 18, 55, 68, -4, 4)
        elif s > 0:
            hair |= f.box(18, 47, 63, 66, 0, 46, r=7)
            quiff = f.box(26, 47, 55, 68, -4, 4)
        else:
            hair |= f.box(6, 38, 63, 66, 4, 42, r=7)
            quiff = f.box(12, 32, 55, 68, -4, 4)
        quiff &= f.full((68 - Y) + (Z + 4) >= 6)                   # the top-front edge cut 45 degrees
        hair |= quiff
        hair &= ~form
        groove = top_faces(hair) & quiff & f.full((Z + s * (X // 7)) % 4 == 0)   # strand lines on the quiff only
        hair &= ~groove
    return hair


def hat_masks(p: VoxCharParams, f: Frame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(hat, band, feather) masks before the subtraction of the head and hair."""
    X, Z = f.X, f.Z
    none = f.full(False)
    if p.hat == "beret":
        Y = f.Y
        band = f.box(-7, 51, 55, 58, -7, 56, r=8)
        (rx, ry, rz), power = BERET_PUFF
        lift = np.clip(Y - 58, 0, None)
        xx = np.abs(X - (HW - 0.18 * lift)) / rx
        zz = np.abs(Z - ((BACK + 1) / 2 + 0.12 * lift)) / rz
        puff = f.full((xx ** power + zz ** power + (lift / ry) ** power <= 1.0) & (Y >= 59))
        crumple = f.full(((X // 5) * 7 + (Z // 5) * 11) % 9 < 3)                         # sparse 5x5 crumples
        bump = np.zeros_like(puff)
        bump[:, 1:, :] = top_faces(puff)[:, :-1, :]
        hat = band | puff | (bump & crumple)
        feather = none.copy()
        for k, (xc, w) in enumerate(FEATHER_SPINE):
            barb = 1 if (k >= 2 and k % 2) else 0
            feather |= f.box(xc - w, xc + w + barb, 59 + 4 * k, 62 + 4 * k, 6, 8)
        return hat, band, feather
    if p.hat == "straw":
        brim = f.box(-16, 60, 55, 57, -16, 65, r=14)
        crown = f.box(2, 42, 58, 70, 2, 47, r=9, rt=4)
        return brim | crown, crown & f.full(f.Y <= 61), none
    if p.hat == "cap":
        dome = f.box(-6, 50, 55, 66, -6, 55, r=8, rt=5)
        visor = f.box(8, 36, 55, 57, -20, -7)
        return dome | visor, visor, none
    return none, none, none


def build_head(p: VoxCharParams, pal: Palette) -> tuple[list[Part], dict]:
    """The head parts in the head frame: ``head`` (the form with the painted face, the tissues,
    the brows and the teeth), ``hair``, ``beard`` (with the moustache), ``hat``, ``feather``
    (when present). Returns (parts, stats)."""
    C = pal.index
    f = Frame()
    X, Y, Z, AX = f.X, f.Y, f.Z, f.AX
    vs = p.coarse / UP
    stats: dict = {}

    # -- form: the rounded skull, ears with a notch, the neck, the two nose blocks ----------
    skull = skull_form(f)
    notch = mirror_x(f.box(47, 48, 29, 34, 22, 27))
    ear_block = mirror_x(f.box(45, 48, 27, 36, 20, 29))
    ears = ear_block & ~notch
    neck = f.box(10, 34, -10, -1, 15, 39)
    nose = f.box(18, 26, 24, 31, -5, -1) & ~f.box(18, 26, 24, 24, -5, -5) & ~f.box(18, 26, 31, 31, -5, -5)
    nose |= f.box(20, 24, 32, 37, -2, -1)
    form = skull | ears | neck | nose
    head = Part("head", (f.shape[2], f.shape[1], f.shape[0]), vs, origin=head_origin(p), bone="head", symmetric=True)
    depth = layer_tissues(head, form, HEAD_BANDS, C)
    cart = (nose | ears) & (depth >= 2)
    head.material[cart] = M.MAT_CARTILAGE
    head.color[cart] = C("cartilage")
    stats["form"] = int(form.sum())
    skin = head.material == M.MAT_SKIN
    front = front_face(form) & skin

    # -- eyes: a white block with the dark lid rows on top, the pupil under the lid, one gleam --
    eye = mirror_x(f.box(*EYE_X, *EYE_Y, 0, 1)) & form
    head.material[eye] = M.MAT_EYE
    head.color[eye] = C("eye_white")
    head.color[eye & f.full(Y >= LID_Y[0])] = C("eye_lid")
    pupil = mirror_x(f.box(*PUPIL_X, *PUPIL_Y, 0, 1)) & eye
    head.color[pupil] = C("eye_dark")
    head.color[mirror_x(f.box(PUPIL_X[1], PUPIL_X[1], PUPIL_Y[1], PUPIL_Y[1], 0, 0)) & eye] = C("eye_light")
    stats["eye"] = int(eye.sum())

    # -- mouth line with the corners a voxel up; the tooth row behind it -----------------------
    mouth = f.box(17, 27, 17, 18, 0, 0) & ~f.box(17, 17, 17, 17, 0, 0) & ~f.box(27, 27, 17, 17, 0, 0)
    mouth |= f.box(17, 17, 19, 19, 0, 0) | f.box(27, 27, 19, 19, 0, 0)
    mouth &= front
    head.color[mouth] = C("mouth")
    stats["mouth"] = int(mouth.sum())
    teeth = f.full(False)
    for c in (11, 14, 17, 20, 23, 26, 29, 32):
        teeth |= f.box(c, c + 1, 17, 18, 2, 3)
    teeth &= form & ~eye
    head.material[teeth] = M.MAT_TOOTH
    head.color[teeth] = C("tooth")
    stats["teeth"] = int(teeth.sum())

    # -- cheeks: two warm tones in an ellipse under each eye, the rim dithered -----------------
    e = np.sqrt(((AX - 12) / 7.0) ** 2 + ((Y - 28) / 4.0) ** 2)
    base_skin = front & (head.color == C("skin"))
    head.color[base_skin & f.full(e <= 0.5)] = C("skin_cheek")
    head.color[base_skin & f.full((e > 0.5) & (e <= 1.0))] = C("skin_cheek_soft")
    head.color[base_skin & f.full((e > 1.0) & (e <= 1.25) & ((X + Y) % 2 == 0))] = C("skin_cheek_soft")
    stats["cheek"] = int((base_skin & f.full(e <= 1.0)).sum())

    # -- brows: bars 2 voxels proud (hair material: shaveable), the inner end a row lower -------
    y1 = 47 if p.brows_thick else 45
    brow = mirror_x(f.box(25, 40, 43, y1, -2, -1) | f.box(25, 27, 42, 42, -2, -1)) & ~form
    head.material[brow] = M.MAT_HAIR
    head.color[brow] = C("hair")
    stats["brow"] = int(brow.sum())

    # -- hair, hat, feather ------------------------------------------------------------------
    hair = hair_mass(p, f, skull, form, ear_block) if p.hair != "bald" else f.full(False)
    hat, band, feather = hat_masks(p, f)
    hat &= ~form & ~hair
    feather &= ~form & ~hair & ~hat

    # -- shade: the forehead row under the fringe / hat, the neck, the under-faces, ear notches --
    cover = (hair | hat)[FZ - 1]                       # (ny, nx) just in front of the face plane
    has = cover.any(axis=0)
    fb = np.where(has, cover.argmax(axis=0) - FY, 10 ** 6)[None, None, :]
    shade = front & f.full((Y >= fb - 3) & (Y < fb))
    shade |= (neck | (under_face(form) & (skull | nose))) & skin
    shade |= dilate(notch) & ears & exposed(form) & skin
    head.color[shade & (head.material == M.MAT_SKIN)] = C("skin_shadow")
    stats["shade"] = int(shade.sum())

    # -- beard zone: the chin and jaw, the cheek line rising to the ear, a ragged top edge -------
    top = np.minimum(14 + np.maximum(AX - 9, 0), 28) + RAG7[AX % 7] * (AX <= 20)   # ragged on the face only
    zone = f.full((Y <= top) & (Z <= np.where(AX > 18, 20, 30)) & (Y >= -12))   # shorter along the jaw sides
    mouth_keep = f.full((AX <= 8) & (Y >= 15) & (Y <= 18) & (Z < 0))
    parts = [head]
    if p.beard == "stubble":
        sel = zone & exposed(form) & (head.material == M.MAT_SKIN) & ~neck & ~mouth_keep
        pat = (AX + 2 * Y + Z) % 4
        head.color[sel & f.full(pat == 0)] = C("stubble")
        head.color[sel & f.full(pat == 2)] = C("stubble_soft")
        stats["beard"] = int(sel.sum())

    if hair.any():
        hp = Part("hair", head.dims, vs, origin=head.origin, bone="head")
        hp.material[hair] = M.MAT_HAIR
        hp.color[hair] = C("hair")
        if p.hair == "chunky":
            light = hair & ~dilate(skull, 3)                       # the raised clumps
        else:
            light = hair & f.full(Y >= 63)                         # the swept slab and the quiff's top
        hp.color[light] = C("hair_light")
        stats["hair"] = int(hair.sum())
        parts.append(hp)

    beard = f.full(False)
    if p.beard in ("short", "full"):
        thick = 3 if p.beard == "short" else 5
        beard = dilate(form & zone, thick) & ~form & zone & ~neck & ~mouth_keep
    if p.mustache:
        must = f.box(10, 34, 19, 23, -4, -1) | f.box(10, 13, 15, 18, -4, -1) | f.box(31, 34, 15, 18, -4, -1)
        beard |= must & ~form & ~mouth_keep
    beard &= ~hair & ~brow
    if beard.any():
        # sparse 5x5 clumps one voxel proud, in the lighter tone: the chunky look without a checkerboard
        clumps = dilate(beard) & ~beard & ~form & ~hair & ~brow & ~mouth_keep
        clumps &= f.full(((AX // 5) * 7 + (Y // 5) * 11 + (Z // 5) * 5) % 9 < 3)
        beard = main_component(beard | clumps)
        bp = Part("beard", head.dims, vs, origin=head.origin, bone="head", symmetric=True)
        bp.material[beard] = M.MAT_HAIR
        bp.color[beard] = C("hair")
        bp.color[beard & clumps] = C("hair_light")
        stats["beard"] = int(beard.sum())
        parts.append(bp)

    if hat.any():
        tp = Part("hat", head.dims, vs, origin=head.origin, bone="head", symmetric=p.hat != "beret")
        mat = M.MAT_PLANT if p.hat == "straw" else M.MAT_CLOTH
        tone = {"straw": ("straw", "straw_band", "straw"), "cap": ("cap", "cap_shade", "cap"),
                "beret": ("beret", "beret_shade", "beret_light")}[p.hat]
        tp.material[hat] = mat
        tp.color[hat] = C(tone[0])
        tp.color[hat & band] = C(tone[1])
        tp.color[top_faces(hat) & f.full(Y >= (78 if p.hat == "beret" else 64))] = C(tone[2])
        stats["hat"] = int(hat.sum())
        parts.append(tp)
        if feather.any():
            fp = Part("feather", head.dims, vs, origin=head.origin, bone="head")
            fp.material[feather] = M.MAT_PLANT
            fp.color[feather] = C("feather")
            spine_x = np.zeros(f.shape[1], int) + 10 ** 6
            for k, (xc, _) in enumerate(FEATHER_SPINE):
                spine_x[FY + 59 + 4 * k:FY + 63 + 4 * k] = xc
            fp.color[feather & f.full(X < spine_x[None, :, None])] = C("feather_shade")
            stats["feather"] = int(feather.sum())
            parts.append(fp)

    return [crop_part(q) for q in parts], stats


# ---------------------------------------------------------------------------- body
#: body tissue bands in 10 mm voxels of depth
BODY_BANDS = [(M.MAT_SKIN, 1), (M.MAT_FAT, 1), (M.MAT_MUSCLE, 2), (M.MAT_BONE, None)]
#: body rows (10 mm): the blocks' vertical extents
ROWS = {"shoe": (0, 3), "shin": (4, 28), "thigh": (29, 55), "torso": (56, 95), "joint": (91, 95),
        "uarm": (73, 95), "farm": (55, 72), "hand": (43, 54), "thumb": (49, 54), "neck": (96, 98)}


def body_grid() -> Cells:
    return Cells((-26, 25), (0, 104), (-16, 15))


def body_blocks(g: Cells) -> dict[str, np.ndarray]:
    """The figure's blocks (bool masks), mirrored where paired."""
    R = ROWS
    b = {
        "shoe": g.mirror(g.box(1, 10, *R["shoe"], -12, 5)),
        "shin": g.mirror(g.box(2, 9, *R["shin"], -5, 3)),
        "thigh": g.mirror(g.box(1, 11, *R["thigh"], -6, 5)),
        "torso": g.box(-12, 11, *R["torso"], -7, 6),
        "joint": g.mirror(g.box(11, 13, *R["joint"], -4, 3)),
        "uarm": g.mirror(g.box(13, 20, *R["uarm"], -4, 3)),
        "farm": g.mirror(g.box(14, 19, *R["farm"], -3, 2)),
        "hand": g.mirror(g.box(15, 17, *R["hand"], -3, 2) | g.box(15, 17, *R["thumb"], -4, -4)),
        "neck": g.box(-4, 3, *R["neck"], -4, 3),
    }
    return b


def build_body(p: VoxCharParams, pal: Palette) -> tuple[Part, dict]:
    """The ``body`` part (10 mm) in the body frame with its tissues, painted garments, shoes
    and garment shells. Returns (part, stats)."""
    C = pal.index
    g = body_grid()
    vs = p.body_voxel
    b = body_blocks(g)
    figure = np.zeros(g.shape, bool)
    for m in b.values():
        figure |= m
    part = Part("body", (g.shape[2], g.shape[1], g.shape[0]), vs,
                origin=(g.lo[0] * vs, g.lo[1] * vs, g.lo[2] * vs), bone="body", symmetric=True)
    depth = layer_tissues(part, figure, BODY_BANDS, C)
    surface = figure & (depth == 0)
    stats = {"figure": int(figure.sum())}
    R = ROWS

    def paint(region: np.ndarray, tone: str, material: int = M.MAT_CLOTH) -> int:
        """Turn the outer voxel layer inside ``region`` into ``material`` and keep skin under it."""
        sel = surface & region
        under = dilate(sel) & figure & (depth == 1)
        part.material[under] = M.MAT_SKIN
        part.color[under] = C("skin")
        part.material[sel] = np.uint8(material)
        part.color[sel] = C(tone)
        return int(sel.sum())

    def shell(region: np.ndarray, tone: str, material: int = M.MAT_CLOTH) -> int:
        sel = region & (part.material == M.MAT_AIR)
        part.material[sel] = np.uint8(material)
        part.color[sel] = C(tone)
        return int(sel.sum())

    legs = b["shin"] | b["thigh"]
    rows = lambda y0, y1: g.box(-10 ** 6, 10 ** 6, y0, y1, -10 ** 6, 10 ** 6)

    # trousers: painted, the hem a two-voxel band in the lighter tone (none under a dress)
    if p.outfit != "dress":
        stats["trousers"] = paint(legs, "trousers")
        part.color[surface & b["shin"] & rows(R["shin"][0], R["shin"][0] + 1)] = C("trousers_hem")
    # shoes: leather blocks, the bottom row the sole
    part.material[b["shoe"]] = M.MAT_LEATHER
    part.color[b["shoe"]] = C("shoe")
    part.color[b["shoe"] & rows(0, 0)] = C("sole")

    arms = b["uarm"] | b["farm"] | b["joint"]
    torso = b["torso"]
    if p.outfit == "coat":
        stats["shirt"] = paint(torso, "shirt")
        paint(arms, "shirt")
        # the coat body: a shell one voxel outside the torso, down to mid-thigh, split at the front
        body_shell = g.box(-13, 12, 36, R["torso"][1], -8, 7) & ~g.box(-12, 11, 36, R["torso"][1], -7, 6)
        split = g.box(-2, 1, 36, R["torso"][1], -8, -8)
        body_shell &= ~split
        usleeve = g.mirror(g.box(12, 21, *R["uarm"], -5, 4))
        fsleeve = g.mirror(g.box(13, 20, *R["farm"], -4, 3))
        lapel = g.mirror(g.box(2, 5, R["torso"][1] - 3, R["torso"][1], -9, -9) | g.box(2, 3, R["torso"][1] - 7, R["torso"][1] - 4, -9, -9))
        collar = g.box(-5, 4, *R["neck"], -5, 4) & ~b["neck"] & ~g.box(-2, 1, *R["neck"], -5, -5)
        n = shell(body_shell | usleeve | fsleeve | lapel | collar, "coat")
        lining = g.mirror(g.box(2, 3, 36, R["torso"][1], -8, -8))
        part.color[lining & (part.material == M.MAT_CLOTH)] = C("lining")
        cuff = fsleeve & rows(R["farm"][0], R["farm"][0] + 2)
        hem = body_shell & rows(36, 37)
        part.color[(cuff | hem) & (part.material == M.MAT_CLOTH)] = C("coat_shade")
        stats["coat"] = n
    elif p.outfit in ("tunic", "apron"):
        region = torso | b["uarm"] | b["joint"] | (b["farm"] if p.sleeves == "long" else np.zeros(g.shape, bool))
        stats["tunic"] = paint(region, "tunic")
        skirt = g.box(-13, 12, 44, R["torso"][0] - 1, -8, 7) & ~g.box(-12, 11, 44, R["torso"][0] - 1, -7, 6)
        skirt |= g.box(-13, 12, R["torso"][0], R["torso"][0] + 2, -8, 7)
        shell(skirt, "tunic")
        part.color[(skirt | (surface & torso)) & rows(R["torso"][0] + 2, R["torso"][0] + 4) & (part.material == M.MAT_CLOTH)] = C("belt")
        if p.outfit == "apron":
            slab = g.box(-7, 6, 30, R["torso"][1] - 7, -9, -9)
            stats["apron"] = shell(slab, "apron")
            part.color[slab & rows(30, 31) & (part.material == M.MAT_CLOTH)] = C("apron_shade")
            # straps: one-voxel lines up the chest, over the shoulders and down the back
            for x in (-7, 6):
                line = g.box(x, x, R["torso"][1] - 6, R["torso"][1], -7, -7) | g.box(x, x, R["torso"][1], R["torso"][1], -7, 6)
                line |= g.box(x, x, R["torso"][0] + 5, R["torso"][1], 6, 6)
                part.color[line & surface & (part.material == M.MAT_CLOTH)] = C("apron")
    elif p.outfit == "dress":
        stats["bodice"] = paint(torso | b["uarm"] | b["joint"], "bodice")
        if p.sleeves == "long":
            paint(b["farm"], "bodice")
        skirt = g.box(-14, 13, 4, R["torso"][0] + 1, -9, 8) & ~g.box(-13, 12, 4, R["torso"][0] - 1, -8, 7)
        stats["dress"] = shell(skirt, "dress")
        part.color[skirt & rows(4, 5) & (part.material == M.MAT_CLOTH)] = C("dress_shade")
        part.color[skirt & rows(R["torso"][0], R["torso"][0] + 1) & (part.material == M.MAT_CLOTH)] = C("dress_shade")
    return crop_part(part), stats


# ---------------------------------------------------------------------------- character
def build_character(params: VoxCharParams | None = None, head_only: bool = False) -> Model:
    """The whole character (parts ``body``, ``head``, ``hair``, ``beard``, ``hat``) standing on
    y = 0, or with ``head_only`` the head parts alone in the head frame, named ``<name>_head``."""
    p = params or VoxCharParams()
    p.validate()
    pal = make_palette(p)
    head_parts, hstats = build_head(p, pal)
    meta = {"generator": "voxchar", "coarse_mm": round(p.coarse * 1000, 3),
            "head_voxel_mm": round(p.coarse / UP * 1000, 3), "body_voxel_mm": round(p.body_voxel * 1000, 3),
            "outfit": p.outfit, "hair": p.hair, "beard": p.beard, "hat": p.hat, "build": {"head": hstats}}
    if head_only:
        model = Model(f"{p.name}_head", p.coarse / UP, pal, meta=meta)
        for q in head_parts:
            model.add_part(q)
        return model
    body, bstats = build_body(p, pal)
    meta["build"]["body"] = bstats
    # the head's neck stands on the body's neck stub, its skull centred over the torso's depth
    neck_top = (ROWS["neck"][1] + 1) * p.body_voxel
    offset = (0.0, neck_top + 2 * p.coarse, -0.5 * p.body_voxel)
    meta["head_offset"] = [round(v, 4) for v in offset]
    model = Model(p.name, p.body_voxel, pal, meta=meta)
    model.add_part(body)
    for q in head_parts:
        q.origin = tuple(q.origin[i] + offset[i] for i in range(3))
        q._coords = None
        model.add_part(q)
    model.meta["height_m"] = round(model.bounds_m()[1][1], 3)
    return model
