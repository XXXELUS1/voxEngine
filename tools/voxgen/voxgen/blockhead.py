"""Block-built head of the owner's barber reference (``docs/reference/barber-head.md``, the only
source of truth for the head's shape; 2026-10-17).

Everything is designed on the unit grid U = 20 mm with coordinates that are multiples of 1/4 U and
written as 5 mm fine voxels (``FINE`` = 4 per unit). Every shape is an axis-aligned
:class:`Block`. The 45-degree surfaces exist only on the skin and only where the sheet's table
2.1 lists them, sized in whole units so that ``facet`` turns each exact 1:1 fine staircase into
one plane (:func:`edge_chamfer`): the skull is a 10 x 13 x 9 U box with the face plane 8 wide,
its front and back vertical edges cut 1 U, the chin's bottom-front edge 1 U, the under-jaw
cut diagonally (z < 3 + y for y 0..6, so the jaw runs from the ear down to the chin in profile
and the neck shows under it), the forehead sloping back 2 U over y 9..11, the top-back and
top-side edges 1 U under the beret; the neck box has its front vertical edges cut 1 U; the top of
the nose is the staircase of :func:`chamfer_top_front`. The skull never narrows in x. The hair,
beard, beret and feather are pure cubes: separate overlapping blocks of different sizes with one
colour per block (no per-voxel dither, no shells, no rounding, no smoothing). The skull, the
nose and the beard are mirror-symmetric in material; the hair, the beret and the feather may be
asymmetric (the beret slouches to -X, the feather sits at the top-left). Render:
``--part-styles skull=facet,nose=facet`` and the rest ``cubes``.

Frame: x to the viewer's right (+x = the character's left), y up, the face looks toward -z;
the face plane is z = 0 and the skull goes to +z, the chin's bottom skin row is y = 0. Fine
arrays are ``(nz, ny, nx)`` over ``EXTENT`` (units); the model's metre origin puts x = 0 on the
mirror plane, y = 0 at the chin's bottom and z = 0 at the skull's mid-depth (4.5 U behind the
face), so the head turns about its own centre in the lab.

Parts (in write order; each later part is clipped by everything already placed, so the skull
stays whole under shaveable hair): ``skull`` (skin with the inner tissues by depth, eyes, brows,
teeth, ears, neck, the bridge of the nose), ``nose`` (its own part only so the render can facet
it: ``--part-styles nose=facet``), ``beard`` (beard and moustache, MAT_HAIR), ``hair``,
``beret`` (MAT_CLOTH), ``feather`` (MAT_PLANT).

Decisions where the reference sheet is silent or inconsistent:
* The mouth cavity (1 U deep, eight 1-fine-voxel teeth on its back wall, as in voxchar v2) sits
  at y 2.25..2.75 directly behind the moustache bar instead of the sheet's 1.75..2.25, because at
  1.75..2.25 it would coincide with the visible lip strip and show as a dark slit under the
  moustache; the reference shows no slit ("hidden by the moustache"). The lip strip stays at
  1.5..2.25 in the light skin tone.
* The beard wraps the lower jaw as blocks: a slab under the chin (y -0.5..0, z 0..3, around the
  neck) and a block that fills the chin's bottom-front chamfer, so the chamfer is "under the
  beard" as the sheet says; the jaw blocks also fill the front vertical chamfer up to y 3.5.
* The fringe blocks reach back to z 2.5 and are clipped by the skull, so they stay attached to
  the forehead that slopes away behind them; the nape blocks start no further forward than z 5.5
  and hang behind the diagonal jaw.
* The undercoat stops under the beret band (y 11) so the band shows red all round; the beret and
  the hair are clipped by each other, never merged.
* Gaze: the pupils of both eyes shift by ``gaze`` units along x (negative = toward the viewer's
  left, as on the sheet); the eye material stays symmetric, only the colour moves.
* Skin colour by block: the face plane and the top the base tone, the side columns (|x| >= 4) and
  the neck the shadow tone, the front strip of the corner notch and the lip the light tone.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import materials as M
from .model import Model, Part
from .palette import Palette
from .voxchar import COMMON_TONES, HEAD_BANDS, crop_part, layer_tissues

#: one design unit in metres and fine voxels per unit (5 mm voxels)
U_M = 0.02
FINE = 4
VOXEL = U_M / FINE
#: array extent in units per axis (lo inclusive, hi exclusive): room for the beret rim and the feather
EXTENT = ((-10.0, 10.0), (-3.0, 24.0), (-4.0, 14.0))
#: the skull's mid-depth in units: the model's local z = 0
SKULL_MID_Z = 4.5

#: colour tones (name -> rgb) of the reference's clusters (barber-head.md section 2.5)
TONES = {
    "skin": (222, 153, 95), "skin_light": (235, 193, 143), "skin_shadow": (200, 126, 72),
    "hair_base": (96, 47, 21), "hair_shade": (79, 41, 20), "hair_light": (115, 59, 29),
    "beard_dark": (67, 33, 15), "brow": (53, 25, 11),
    "beret_light": (140, 43, 25), "beret_main": (115, 32, 21), "beret_shade": (88, 22, 14),
    "beret_fold": (68, 15, 10),
    "dark": (44, 14, 7), "pupil": (28, 9, 4), "eye_white": (239, 230, 214),
    "feather": (217, 196, 160), "feather_shade": (198, 165, 130),
}
HAIR_TONES = ("hair_base", "hair_shade", "hair_light")
HAIR_WEIGHTS = (0.5, 0.3, 0.2)

#: fine x positions (1/4 U) of the eight teeth inside the mouth cavity x -1.5..1.5 (mirror-exact)
TEETH_X = (-6, -4, -3, -1, 0, 2, 3, 5)
#: beret layers (2.4): (name, x0, x1, y0, y1, z0, z1, corner cut size, shift to -X, tones)
BERET_LAYERS = (
    ("puff1", -8.0, 8.0, 12.0, 14.0, -3.0, 12.0, 2.0, 1.0, ("beret_main", "beret_shade", "beret_fold")),
    ("puff2", -7.0, 7.0, 14.0, 16.0, -2.0, 11.0, 1.5, 2.0, ("beret_main", "beret_light", "beret_shade")),
    ("puff3", -5.0, 5.0, 16.0, 17.0, 0.0, 8.0, 1.0, 2.5, ("beret_light", "beret_main")),
)


@dataclass
class BlockHeadParams:
    name: str = "barber"
    seed: int = 7                 # every ragged edge (hair clumps, beret rim, crumples) comes from this
    gaze: float = -0.5            # pupil shift along x in units for both eyes (-0.5..0.5, steps of 1/4)
    hair_thickness: float = 1.5   # how far the temple strands reach beyond the skull side (units, 1..2)
    hair_bottom: float = 3.0      # where the temple strands end (units above the chin)
    beard: bool = True
    beret: bool = True
    beret_slouch: float = 1.0     # 0..1 scale of the layers' slide to -X (1 = the sheet's 1 / 2 / 2.5 U)
    feather: bool = True

    def validate(self) -> None:
        if abs(self.gaze) > 0.5 or not _on_grid(self.gaze):
            raise ValueError("gaze must be in -0.5..0.5 in steps of 1/4")
        if not 1.0 <= self.hair_thickness <= 2.0 or not _on_grid(self.hair_thickness, 0.5):
            raise ValueError("hair_thickness must be 1..2 in steps of 1/2")
        if not 2.5 <= self.hair_bottom <= 7.0 or not _on_grid(self.hair_bottom, 0.5):
            raise ValueError("hair_bottom must be 2.5..7 in steps of 1/2")
        if not 0.0 <= self.beret_slouch <= 1.0:
            raise ValueError("beret_slouch must be 0..1")


def _on_grid(v: float, step: float = 0.25) -> bool:
    return abs(v / step - round(v / step)) < 1e-9


# ---------------------------------------------------------------------------- primitives
@dataclass(frozen=True)
class Block:
    """An axis-aligned box in units, lo inclusive / hi exclusive, every bound a multiple of 1/4."""
    x0: float
    x1: float
    y0: float
    y1: float
    z0: float
    z1: float
    tone: str = "skin"
    material: int = M.MAT_SKIN

    def __post_init__(self):
        for v in (self.x0, self.x1, self.y0, self.y1, self.z0, self.z1):
            if not _on_grid(v):
                raise ValueError(f"block bound {v} is not a multiple of 1/4 U")
        if not (self.x0 < self.x1 and self.y0 < self.y1 and self.z0 < self.z1):
            raise ValueError(f"empty block {self}")

    def mirror(self) -> "Block":
        return replace(self, x0=-self.x1, x1=-self.x0)

    def shifted(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> "Block":
        return replace(self, x0=self.x0 + dx, x1=self.x1 + dx, y0=self.y0 + dy, y1=self.y1 + dy,
                       z0=self.z0 + dz, z1=self.z1 + dz)

    def with_tone(self, tone: str, material: int | None = None) -> "Block":
        return replace(self, tone=tone, material=self.material if material is None else material)


def both(b: Block) -> list[Block]:
    """The block and its x-mirror."""
    return [b, b.mirror()]


#: the skull box (barber-head.md 2.1): 10 wide, 13 tall, 9 deep; the face plane z = 0 is 8 wide
SKULL = Block(-5, 5, 0, 13, 0, 9)
NECK = Block(-2.25, 2.25, -3, 0, 1, 5.5)


class Frame:
    """The fine grid over ``EXTENT``: cell-centre coordinates in units (broadcastable) and masks."""

    def __init__(self, extent=EXTENT, fine: int = FINE):
        self.extent, self.fine = extent, fine
        self.lo = tuple(lo for lo, _ in extent)
        nx, ny, nz = (int(round((hi - lo) * fine)) for lo, hi in extent)
        self.dims = (nx, ny, nz)
        self.shape = (nz, ny, nx)
        self.X = (self.lo[0] + (np.arange(nx) + 0.5) / fine)[None, None, :]
        self.Y = (self.lo[1] + (np.arange(ny) + 0.5) / fine)[None, :, None]
        self.Z = (self.lo[2] + (np.arange(nz) + 0.5) / fine)[:, None, None]

    @property
    def origin_m(self) -> tuple[float, float, float]:
        return (self.lo[0] * U_M, self.lo[1] * U_M, (self.lo[2] - SKULL_MID_Z) * U_M)

    def full(self, m) -> np.ndarray:
        return np.broadcast_to(m, self.shape).copy()

    def box(self, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> np.ndarray:
        X, Y, Z = self.X, self.Y, self.Z
        return self.full((X >= x0) & (X < x1) & (Y >= y0) & (Y < y1) & (Z >= z0) & (Z < z1))

    def mask(self, b: Block) -> np.ndarray:
        return self.box(b.x0, b.x1, b.y0, b.y1, b.z0, b.z1)

    def new_part(self, name: str, symmetric: bool = False) -> Part:
        return Part(name, self.dims, VOXEL, origin=self.origin_m, bone="head", symmetric=symmetric)

    def index(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        """Fine index (ix, iy, iz) of the cell containing the unit point."""
        return tuple(int(np.floor((v - lo) * self.fine)) for v, lo in zip((x, y, z), self.lo))


def chamfer_top_front(f: Frame, b: Block) -> np.ndarray:
    """The block minus a 1:1 staircase on its top-front edge: fine row k (from the bottom) keeps
    z >= z0 + k / FINE, so a block 1 U tall goes from full depth at the bottom to nothing at the
    top in four steps: one 45-degree plane under ``facet``. The nose is the only user."""
    return f.mask(b) & f.full((f.Z - b.z0) >= (f.Y - b.y0) - 1e-9)


def corner_cuts(b: Block, size: float) -> list[Block]:
    """The four vertical corner columns ``size`` x ``size`` of a block (a 1 U step for the skull
    rows and the band, 2 U for the puffs): cut, never chamfered."""
    if size <= 0:
        return []
    return [Block(b.x0, b.x0 + size, b.y0, b.y1, b.z0, b.z0 + size),
            Block(b.x1 - size, b.x1, b.y0, b.y1, b.z0, b.z0 + size),
            Block(b.x0, b.x0 + size, b.y0, b.y1, b.z1 - size, b.z1),
            Block(b.x1 - size, b.x1, b.y0, b.y1, b.z1 - size, b.z1)]


class Builder:
    """Writes blocks into parts on one frame; a part placed later never overwrites an earlier one
    (``occupied``), while inside a part the later block wins."""

    def __init__(self, f: Frame, pal: Palette):
        self.f, self.pal = f, pal
        self.occupied = np.zeros(f.shape, bool)
        self.parts: list[Part] = []

    def paint(self, part: Part, mask: np.ndarray, tone: str, material: int, clip: bool = True) -> np.ndarray:
        if clip:
            mask = mask & ~self.occupied
        part.material[mask] = np.uint8(material)
        part.color[mask] = np.uint8(self.pal.index(tone))
        return mask

    def place(self, part: Part, b: Block, clip: bool = True) -> np.ndarray:
        return self.paint(part, self.f.mask(b), b.tone, b.material, clip)

    def place_all(self, part: Part, blocks, clip: bool = True) -> np.ndarray:
        out = np.zeros(self.f.shape, bool)
        for b in blocks:
            out |= self.place(part, b, clip)
        return out

    def cut(self, part: Part, b: Block) -> np.ndarray:
        mask = self.f.mask(b) & part.solid()
        part.material[mask] = 0
        part.color[mask] = 0
        return mask

    def recolor(self, part: Part, mask: np.ndarray, tone: str, material: int | None = M.MAT_SKIN) -> None:
        sel = mask & part.solid()
        if material is not None:
            sel &= part.material == material
        part.color[sel] = np.uint8(self.pal.index(tone))

    def finish(self, part: Part) -> Part:
        self.occupied |= part.solid()
        self.parts.append(part)
        return part


def pick(rng: np.random.Generator, choices, weights=None):
    """One of ``choices`` (deterministic through ``rng``)."""
    i = int(rng.choice(len(choices), p=None if weights is None else np.asarray(weights) / np.sum(weights)))
    return choices[i]


# ---------------------------------------------------------------------------- palette
def make_palette() -> Palette:
    pal = Palette()
    for name, rgb in TONES.items():
        pal.add(name, rgb)
    for name in ("fat", "muscle", "bone", "cartilage", "tooth"):
        pal.add(name, COMMON_TONES[name])
    return pal


# ---------------------------------------------------------------------------- skull
def edge_chamfer(a, b, size: float) -> np.ndarray:
    """Cells to remove for a 45-degree chamfer of ``size`` units between two faces that meet at an
    edge: ``a`` and ``b`` are the cell-centre distances inward from the two faces (units). The
    result is an exact 1:1 fine staircase (a 1 U chamfer removes 6 of the 16 cells of the edge
    column), one plane under ``facet``."""
    return (a + b) < size - 1e-9


def skull_form(f: Frame) -> np.ndarray:
    """The skin skull: the box minus the chamfers of table 2.1 (all 45 degrees, whole units)."""
    X, Y, Z = f.X, f.Y, f.Z
    AX = np.abs(X)
    cut = f.full(edge_chamfer(SKULL.x1 - AX, Z - SKULL.z0, 1.0))      # front vertical edges (|x| 4..5, z 0..1)
    cut |= edge_chamfer(SKULL.x1 - AX, SKULL.z1 - Z, 1.0)             # back vertical edges
    cut |= edge_chamfer(Y - SKULL.y0, SKULL.z1 - Z, 6.0)              # under the jaw: z < 3 + y for y 0..6
    cut |= edge_chamfer(Y - SKULL.y0, Z - SKULL.z0, 1.0)              # the chin's bottom-front edge
    cut |= Z < np.minimum(Y - 9.0, 2.0)                               # the forehead: back 2 U over y 9..11
    cut |= edge_chamfer(SKULL.y1 - Y, SKULL.z1 - Z, 1.0)              # top-back edge
    cut |= edge_chamfer(SKULL.y1 - Y, SKULL.x1 - AX, 1.0)             # top-side edges
    return f.mask(SKULL) & ~cut


def neck_form(f: Frame) -> np.ndarray:
    """The neck box with its front vertical edges cut 1 U."""
    cut = edge_chamfer(NECK.x1 - np.abs(f.X), f.Z - NECK.z0, 1.0)
    return f.mask(NECK) & ~f.full(cut)


def skull_back(y: float) -> float:
    """z of the skull's back at height y (units): the diagonal under the jaw, 9 above it."""
    return float(min(SKULL.z1, 3.0 + y))


def build_skull(bd: Builder, p: BlockHeadParams) -> Part:
    f = bd.f
    part = f.new_part("skull", symmetric=True)
    form = skull_form(f)
    neck = neck_form(f)
    ears = np.zeros(f.shape, bool)
    for b in both(Block(5, 5.5, 5, 7, 2, 3.5)):
        ears |= f.mask(b)
    bridge = f.mask(Block(-0.75, 0.75, 5.5, 8, -0.5, 0))
    form |= neck | ears | bridge
    # the mouth: a cavity 1 U deep behind the moustache bar (cut before the tissues, so its walls are skin)
    cavity = f.mask(Block(-1.5, 1.5, 2.25, 2.75, 0, 1))
    form &= ~cavity
    layer_tissues(part, form, HEAD_BANDS, bd.pal.index)
    # skin tones by block: the side columns, the under-jaw and the neck in shadow, the front
    # chamfer planes (the light strip along the face's edge) and the lip light
    AX = np.abs(f.X)
    bd.recolor(part, (f.full(AX >= 4) & ~bridge & ~ears) | neck | f.full(f.Z >= 3 + f.Y), "skin_shadow")
    bd.recolor(part, f.full((AX >= 4) & (f.Z < 1)), "skin_light")
    bd.recolor(part, f.mask(Block(-1.75, 1.75, 1.5, 2.25, 0, 0.25)), "skin_light")
    # eyes: the socket recessed one fine voxel, the frame 1/4 U around a 2 x 1 inside (white + pupil)
    for s in (1, -1):
        sock = Block(1, 3.5, 6, 7.5, 0, 0.25)
        eye = Block(1, 3.5, 6, 7.5, 0.25, 0.75, "dark", M.MAT_EYE)
        inner = Block(1.25, 3.25, 6.25, 7.25, 0.25, 0.75, "eye_white", M.MAT_EYE)
        if s < 0:
            sock, eye, inner = sock.mirror(), eye.mirror(), inner.mirror()
        bd.cut(part, sock)
        bd.place(part, eye, clip=False)
        bd.place(part, inner, clip=False)
        cx = s * 2.25 + p.gaze
        bd.place(part, Block(cx - 0.5, cx + 0.5, 6.25, 7.25, 0.25, 0.75, "pupil", M.MAT_EYE), clip=False)
    # teeth: eight fine-voxel teeth on the cavity's back wall
    for k in TEETH_X:
        bd.place(part, Block(k / 4, k / 4 + 0.25, 2.25, 2.75, 0.75, 1, "tooth", M.MAT_TOOTH), clip=False)
    # brows: straight bars half a unit proud lying on the eye frames, hair material (shaveable)
    bd.place_all(part, both(Block(0.75, 3.75, 7.5, 8.5, -0.5, 0, "brow", M.MAT_HAIR)), clip=False)
    return bd.finish(part)


def build_nose(bd: Builder) -> Part:
    """Two 2 x 1 x 1 U cubes one unit proud; the upper one's top is the 45-degree staircase."""
    f = bd.f
    part = f.new_part("nose", symmetric=True)
    top = Block(-1, 1, 4.5, 5.5, -1, 0)
    bd.paint(part, chamfer_top_front(f, top), "skin", M.MAT_SKIN)
    bd.place(part, Block(-1, 1, 3.5, 4.5, -1, 0))
    for b in both(Block(0.5, 0.75, 3.5, 3.75, -1, -0.75)):
        bd.recolor(part, f.mask(b), "dark")
    return bd.finish(part)


# ---------------------------------------------------------------------------- beard
def beard_blocks() -> list[Block]:
    """Moustache, goatee ring, chin, the jaw blocks (two browns, 1 x 1 and 2 x 1 blocks, the top
    edge rising from the mouth to the face's edge, the cheeks bare) and the chin's underside."""
    H = M.MAT_HAIR
    out = [Block(-2.5, 2.5, 2.25, 3.5, -0.5, 0, "brow", H),                 # moustache bar
           Block(-4.5, 4.5, -0.5, 0, 0, 3, "beard_dark", H),                # the slab under the chin (around the neck)
           Block(-4.5, 4.5, 0, 1, 0, 1, "beard_dark", H),                   # fills the chin's bottom-front chamfer
           Block(-1, 1, -1, 1.5, -0.75, 0, "hair_base", H)]                 # chin, centre to y = -1
    for b in (Block(1.75, 2.5, 1.5, 2.25, -0.5, 0, "beard_dark", H),        # moustache end
              Block(1.75, 2.5, 0.5, 1.5, -0.5, 0, "beard_dark", H),         # goatee ring
              Block(1, 2, -0.5, 1.5, -0.75, 0, "hair_shade", H),            # chin, edge to y = -0.5
              Block(2, 3, -0.5, 2, -0.75, 0, "hair_base", H),               # jaw front, up to y = 2
              Block(3, 4, -0.5, 3, -0.5, 0, "hair_shade", H),               # up to y = 3
              Block(4, 5, -0.5, 3.5, -0.75, 1, "hair_base", H),             # the corner (fills the front chamfer), up to y = 3.5
              Block(5, 5.75, -0.5, 3.5, 0, 3, "hair_shade", H)):            # the jaw's side
        out += both(b)
    return out


def build_beard(bd: Builder) -> Part:
    part = bd.f.new_part("beard", symmetric=True)
    bd.place_all(part, beard_blocks())
    return bd.finish(part)


# ---------------------------------------------------------------------------- hair
def clump_column(side: int, y_top: float, y_bot: float, x_in: float, z_lo: float, z_hi: float,
                 reach, rng: np.random.Generator, heights=(1.0, 1.5, 2.0), shifts=(0.0, 0.5, 1.0),
                 depths=(2.0, 2.5, 3.0), z_starts=(0.0, 0.5, 1.0)) -> list[Block]:
    """Blocks down a column from ``y_top`` to ``y_bot`` on side ``side`` (+1 / -1): each block's
    height, depth, z start and outward shift are drawn from the sets; it spans x from ``x_in``
    outward to ``reach(y) + shift`` (``reach`` = the outer x for a block whose bottom is y,
    clamped by the caller's maximum), all on the 1/2 U grid."""
    blocks = []
    y = y_top
    while y > y_bot + 1e-9:
        h = min(pick(rng, heights), y - y_bot)
        x_out = reach(y - h, pick(rng, shifts))
        z0 = z_lo + pick(rng, z_starts)
        z1 = min(z0 + pick(rng, depths), z_hi)
        x0, x1 = (x_in, x_out) if side > 0 else (-x_out, -x_in)
        blocks.append(Block(x0, x1, y - h, y, z0, z1, pick(rng, HAIR_TONES, HAIR_WEIGHTS), M.MAT_HAIR))
        y -= h
    return blocks


def fringe_blocks(rng: np.random.Generator) -> list[Block]:
    """Clumps along the forehead y 10..11.5, z -1..0: 1-2 U wide, the bottom ragged (down to 9.5
    near the temples), the top under the beret band."""
    blocks = []
    x = -4.5
    while x < 4.5 - 1e-9:
        w = min(pick(rng, (1.0, 1.5, 2.0)), 4.5 - x)
        temple = x < -3 or x + w > 3
        y0 = pick(rng, (9.5, 9.5, 9.75, 10.0) if temple else (9.5, 9.75, 10.0, 10.25))
        z0 = pick(rng, (-1.0, -0.75))
        # back to z 2.5: clipped by the skull, so the clump stays attached to the sloping forehead
        blocks.append(Block(x, x + w, y0, 11.5, z0, 2.5, pick(rng, HAIR_TONES, HAIR_WEIGHTS), M.MAT_HAIR))
        x += w
    return blocks


def occiput_blocks(rng: np.random.Generator) -> list[Block]:
    """Rows of 2 x 1-1.5 U blocks down the back of the head, z from the skull's back to 10.5, the
    bottom row ragged between y 2.5 and 3.5."""
    blocks = []
    y = 11.0
    while y > 3.5 + 1e-9:
        h = min(pick(rng, (1.0, 1.5)), y - 3.5)
        y0 = y - h
        for x in np.arange(-5.0, 5.0, 2.0):
            bottom = pick(rng, (2.5, 3.0, 3.5)) if y0 <= 3.5 + 1e-9 else y0
            z0 = max(skull_back(y0), 6.0) - pick(rng, (0.0, 0.5))      # hangs behind the diagonal jaw
            z1 = min(z0 + pick(rng, (1.5, 2.0, 2.5)), 10.5)
            blocks.append(Block(x, x + 2, bottom, y, z0, z1, pick(rng, HAIR_TONES, HAIR_WEIGHTS), M.MAT_HAIR))
        y = y0
    return blocks


def build_hair(bd: Builder, p: BlockHeadParams) -> Part:
    f = bd.f
    rng = np.random.default_rng(p.seed)
    part = f.new_part("hair")
    H = M.MAT_HAIR
    # undercoat: a half-unit shell on the sides above y 7 and on the back down to y 3 (no holes under the clumps)
    under = both(Block(5, 5.5, 7, 11, 0, 9, "hair_shade", H))
    under += [Block(-5.5, 5.5, 5, 11, 9, 9.5, "hair_shade", H)]
    bd.place_all(part, under)
    bd.place_all(part, both(Block(5, 5.5, 2, 6, 0.5, 2.5, "hair_base", H)))     # sideburns
    bd.place_all(part, occiput_blocks(rng))

    def reach(y: float, shift: float) -> float:
        base = 6.0 if y >= 6 else 6.5
        return min(base + shift, 5.0 + p.hair_thickness)

    for side in (1, -1):
        bd.place_all(part, clump_column(side, 11.0, p.hair_bottom, 4.5, -0.5, 4.0, reach, rng))
    bd.place_all(part, fringe_blocks(rng))
    return bd.finish(part)


# ---------------------------------------------------------------------------- beret and feather
def offset_rim(b: Block, corner: float, rng: np.random.Generator, tones) -> tuple[list[Block], list[Block]]:
    """(blocks pushed out, blocks pushed in) along a layer's four edges in 2 U segments, each
    one unit tall and one unit deep, skipping the cut corners; the ragged rim of the beret."""
    out, cut = [], []
    h = 1.0 if b.y1 - b.y0 <= 1 else pick(rng, (1.0, b.y1 - b.y0))
    for axis in ("x", "z"):
        lo, hi = (b.x0, b.x1) if axis == "x" else (b.z0, b.z1)
        a0, a1 = (b.z0, b.z1) if axis == "x" else (b.x0, b.x1)
        s = a0 + corner
        while s + 2 <= a1 - corner + 1e-9:
            for edge in (lo, hi):
                r = pick(rng, (-1, 0, 1, 1))
                y0 = b.y0 if pick(rng, (0, 1)) else b.y1 - h
                y1 = y0 + h
                if r == 0:
                    continue
                if axis == "x":
                    x0, x1 = (edge - 1, edge) if (edge == lo) == (r > 0) else (edge, edge + 1)
                    blk = Block(x0, x1, y0, y1, s, s + 2, pick(rng, tones), M.MAT_CLOTH)
                else:
                    z0, z1 = (edge - 1, edge) if (edge == lo) == (r > 0) else (edge, edge + 1)
                    blk = Block(s, s + 2, y0, y1, z0, z1, pick(rng, tones), M.MAT_CLOTH)
                (out if r > 0 else cut).append(blk)
            s += 2
    return out, cut


def crumple_top(b: Block, rng: np.random.Generator, tones, count: int = 5) -> list[Block]:
    """``count`` blocks 1.5-2 U wide and 1/2 or 1 U tall scattered on a layer's roof."""
    blocks = []
    for _ in range(count):
        w, d = pick(rng, (1.5, 2.0)), pick(rng, (1.5, 2.0))
        x0 = pick(rng, tuple(np.arange(b.x0 + 1, b.x1 - 1 - w + 0.001, 0.5)))
        z0 = pick(rng, tuple(np.arange(b.z0 + 1, b.z1 - 1 - d + 0.001, 0.5)))
        blocks.append(Block(x0, x0 + w, b.y1, b.y1 + pick(rng, (0.5, 1.0)), z0, z0 + d, pick(rng, tones), M.MAT_CLOTH))
    return blocks


def stack_layers(slouch: float, rng: np.random.Generator) -> list[tuple[Block, list[Block], list[Block]]]:
    """The puffs slid to -X by ``slouch`` x (1 / 2 / 2.5 U), each as (layer, corner and rim cuts,
    rim blocks pushed out)."""
    layers = []
    for _, x0, x1, y0, y1, z0, z1, corner, shift, tones in BERET_LAYERS:
        dx = -round(shift * slouch * 2) / 2
        b = Block(x0 + dx, x1 + dx, y0, y1, z0, z1, tones[0], M.MAT_CLOTH)
        out, cut = offset_rim(b, corner, rng, tones)
        layers.append((b, corner_cuts(b, corner) + cut, out))
    return layers


def build_beret(bd: Builder, p: BlockHeadParams) -> Part:
    f = bd.f
    rng = np.random.default_rng(p.seed + 1)
    part = f.new_part("beret")
    band = Block(-5.5, 5.5, 11, 12, -1, 10, "beret_shade", M.MAT_CLOTH)
    bd.place(part, band)
    for c in corner_cuts(band, 1.0):
        bd.cut(part, c)
    dx = -round(3 * p.beret_slouch * 2) / 2
    bd.place(part, Block(-5.5 + dx, -2.5 + dx, 10, 12, 0, 8, "beret_shade", M.MAT_CLOTH))   # the slouch
    layers = stack_layers(p.beret_slouch, rng)
    for layer, cuts, out in layers:
        bd.place(part, layer)
        for c in cuts:
            bd.cut(part, c)
        bd.place_all(part, out)
    top = layers[-1][0]
    bd.place_all(part, crumple_top(top, rng, ("beret_light", "beret_main")))
    return bd.finish(part)


def feather_blocks(start=(-3.0, 16.5, 4.0), count: int = 6, size: float = 1.5) -> list[Block]:
    """A staircase of ``count`` cubes up and to -X from ``start`` (step (-1, +1, 0)), with barbs
    1 x 1 x 1/2 U alternating to +z / -z, a half-unit vane along the outer (upper-left) edge that
    widens the plume to 2 U, and a dark half-unit quill along the lower-right corner."""
    blocks = []
    x, y, z = start
    for k in range(count):
        bx, by = x - k, y + k
        blocks.append(Block(bx, bx + size, by, by + size, z, z + size, "feather", M.MAT_PLANT))
        barb_z = (z + size, z + size + 0.5) if k % 2 == 0 else (z - 0.5, z)
        blocks.append(Block(bx, bx + 1, by + 0.5, by + 1.5, barb_z[0], barb_z[1], "feather_shade", M.MAT_PLANT))
        blocks.append(Block(bx - 0.5, bx, by + 0.5, by + 1.5, z + 0.25, z + size - 0.25, "feather_shade" if k % 2 else "feather", M.MAT_PLANT))
        blocks.append(Block(bx + size - 0.5, bx + size, by, by + 0.5, z + 0.5, z + 1, "beard_dark", M.MAT_PLANT))
    return blocks


def build_feather(bd: Builder) -> Part:
    part = bd.f.new_part("feather")
    bd.place_all(part, feather_blocks())
    return bd.finish(part)


# ---------------------------------------------------------------------------- model
def build_head(params: BlockHeadParams | None = None) -> Model:
    """The block head as a model of cropped parts (skull, nose, beard, hair, beret, feather)."""
    p = params or BlockHeadParams()
    p.validate()
    pal = make_palette()
    f = Frame()
    bd = Builder(f, pal)
    build_skull(bd, p)
    build_nose(bd)
    if p.beard:
        build_beard(bd)
    build_hair(bd, p)
    if p.beret:
        build_beret(bd, p)
        if p.feather:
            build_feather(bd)
    meta = {"generator": "blockhead", "unit_mm": round(U_M * 1000, 3), "voxel_mm": round(VOXEL * 1000, 3),
            "seed": p.seed, "gaze": p.gaze, "hair_thickness": p.hair_thickness, "beret_slouch": p.beret_slouch,
            "reference": "docs/reference/barber-head.md"}
    model = Model(p.name, VOXEL, pal, meta=meta)
    for q in bd.parts:
        model.add_part(crop_part(q))
    model.meta["build"] = {q.name: q.count_solid() for q in model.parts}
    return model
