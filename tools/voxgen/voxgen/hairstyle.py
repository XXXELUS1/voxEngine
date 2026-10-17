"""Designed voxel hair styles as a part of their own: a solid base mass over the scalp, clump
geometry on top, three flat tones (base, top-lit, deep).

Why: the owner's reference (a voxel-art character with a spiky quiff) reads as hair by its
*silhouette* and two-tone shading: a stepped mass of blocks with a few lighter highlights,
no strand noise. The earlier caps (``head_from_mesh.hair_cap``: a distance shell of the skull)
read as a helmet. This module designs the hair in metres on whatever grid it is given (5 mm
or 3 mm), from the form of the head and the mesh landmarks, so the same style builds at both
sizes and stays comparable.

Styles (``STYLES``):
  ``quiff``  the reference: a short base (12 mm) and a pompadour that is one *stair* leaning
             back (:class:`Quiff`, :func:`add_quiff`): from the top of the hairline ledge every
             step (``step`` 1 cm) is set back ``lean`` x ``step`` from the one below (1.0 = 45
             degrees, never a vertical front wall), the slab is ``depth`` deep so its back edge
             is the same diagonal, its half-width tapers from the full head width to the crest,
             and the crest carries a few teeth (``spikes``: x, half-width, steps) that continue
             the same stair - stepped spikes leaning back, no tapering towers, no one-voxel tip.
             The base mass in front of the diagonal is cut away, so the silhouette is one clean
             diagonal from the ledge to the crest. (The earlier version - a vertical block with
             pyramid peaks - read as a crenellated castle.)
  ``short``  a cap 15 mm thick with the hairline ledge and one-voxel ridge rows along the comb
             direction (front to back) every ``ridge_period`` columns over the top, starting
             ``ridge_from_front`` behind the front face so the ledge stays a clean line.
  ``slick``  swept back: lower volume on top, the back a little longer, long ridges that run
             over the top and continue down the back face as vertical lines.
  ``messy``  irregular small clumps (1-2 voxels tall at 5 mm: "2-step bumps") scattered over the
             top, leaning different ways; everything mirror-symmetric.

Common build (:func:`build_hair`):
  1. the hair floor (:func:`hair_floor`): a piecewise *flat* lower edge in metres - the forehead
     hairline at ``hairline_frac`` of the brow-to-crown distance, dropped by ``temple_drop``
     outward of ``temple_x``, mid-ear in front of the ear, ``ear_clearance`` above the ear over
     it, the ear's bottom (minus ``nape_drop``) behind it; flat pieces, so the voxel outline is a
     few deliberate steps, never a smooth staircase;
  2. the scalp: the form above the floor, the ear box excluded; its exposed voxels are swept by
     a superellipsoid of offsets (``top`` up, ``side`` sideways, ``back`` backward, ``ledge``
     forward, ``shape_power`` 4 = flatter planes than an ellipsoid) -> the base mass, cut at the
     floor, de-spiked (:func:`head_from_mesh.smooth_cap`), mirrored;
  3. clumps (:func:`add_clumps`) stand on the base top (each column filled down to the base so
     a leaning spike never floats) and ridges (:func:`add_ridges`) are one voxel on the exposed
     top (and, for ``slick``, back) faces of pattern columns;
  4. tones (:func:`tone_masks`): ``lit`` = upward faces of the clumps / the quiff stair and of the base's crown
     terrace within ``lit_band`` of the crown (the lower terraces and the ridges stay base, so
     no contour map and no speckle), ``deep`` =
     hair with air below (undersides, the hairline ledge) and the band ``deep_band`` above the
     floor behind the ears (the nape).
The hair touches the scalp by construction (offset (0, 1, 0) is always in the sweep) and is
kept to its largest 6-component. The caller carves the pocket in the form (see
:mod:`voxgen.refhead`); this module never changes the form.

Decisions: clump x positions on 5 mm boundaries, so a tip is 2 x 2 fine = one coarse voxel;
clump lists are given for x >= 0 and mirrored; everything deterministic (no hashes).
``hairline_y`` (metres) overrides the ``hairline_frac`` rule and ``floor_quantum`` snaps every
floor level down to a multiple of it, so a head designed on a 1 cm grid (:mod:`voxgen.toonhead`
``box``) gets its hairline on a coarse row; ``ridge_width`` widens the comb ridges to whole
coarse columns there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from . import meshvox
from .body import exposed, shift
from .checks import label_components
from .head_from_mesh import smooth_cap

STYLES = ("quiff", "short", "slick", "messy")


@dataclass
class Clump:
    """One block or stepped spike (metres): base centre ``x`` (>= 0, mirrored when > 0), base
    centre ``z`` behind the hair's front face at that x, ``height``, base ``width`` (x) and
    ``depth`` (z), ``lean`` = metres backward per metre of rise (negative: forward), and the
    size at the top (``top_width`` / ``top_depth``; the taper is linear, a side never thinner
    than one voxel)."""
    x: float
    z: float
    height: float
    width: float = 0.02
    depth: float = 0.025
    lean: float = 0.6
    top_width: float = 0.01
    top_depth: float = 0.01


@dataclass
class Quiff:
    """The swept-back pompadour (metres): a stair of ``step``-tall steps standing on the top of
    the hairline ledge, every step set back ``lean`` x ``step`` from the one below (1.0 = 45
    degrees), ``depth`` deep along z (so the back edge is the same diagonal), rising ``height``
    above the base mass; the half-width tapers from ``half_width`` (bottom step) to
    ``half_width_top`` (crest), snapped to ``step``; ``spikes`` = (x, half_width, steps) teeth
    on the crest that continue the stair (x > 0 mirrored)."""
    height: float = 0.02
    depth: float = 0.05
    lean: float = 1.0
    step: float = 0.01
    half_width: float = 0.07
    half_width_top: float = 0.05
    spikes: tuple = ((0.0, 0.02, 1), (0.0, 0.01, 3), (0.04, 0.01, 2))


#: the reference quiff: a 45-degree stair to a 10 cm crest, a centre tooth (4 cm wide, then 2 cm
#: two steps higher) and a pair of 2 cm teeth at the crest's ends, one gap column between
QUIFF = Quiff()
MESSY_CLUMPS = [
    Clump(0.010, 0.015, 0.018, 0.020, 0.020, 0.3, 0.010, 0.010),
    Clump(0.040, 0.025, 0.014, 0.018, 0.018, -0.3, 0.010, 0.010),
    Clump(0.025, 0.055, 0.018, 0.020, 0.020, 0.5, 0.010, 0.010),
    Clump(0.055, 0.060, 0.012, 0.015, 0.015, 0.0, 0.010, 0.010),
    Clump(0.000, 0.090, 0.015, 0.025, 0.020, 0.4, 0.010, 0.010),
    Clump(0.035, 0.105, 0.012, 0.018, 0.018, -0.4, 0.010, 0.010),
    Clump(0.010, 0.135, 0.012, 0.020, 0.020, 0.6, 0.010, 0.010),
]


@dataclass
class HairParams:
    """Lengths in metres."""

    style: str = "quiff"
    hairline_frac: float = 0.36       # forehead hairline: fraction of the way from the brow to the crown
    temple_x: float = 0.045           # outward of this |x| the forehead hairline drops by temple_drop
    temple_drop: float = 0.015
    ear_clearance: float = 0.006      # the hair over the ear stops this far above the ear top
    side_in: float = 0.006            # the skull "sides" start this far inside ear_inner_x
    sideburn_depth: float = 0.020     # the sideburn strip reaches this far in front of the ear
    nape_drop: float = 0.0            # the nape edge is this far below the ear's bottom
    top: float = 0.020                # base mass thickness upward
    side: float = 0.008               # sideways
    back: float = 0.015               # backward
    ledge: float = 0.005              # forward (the hairline ledge over the forehead)
    shape_power: float = 4.0          # superellipsoid exponent of the sweep (2 = round, large = box)
    clumps: list[Clump] = field(default_factory=list)
    quiff: Quiff | None = None        # the stair pompadour (the quiff style)
    hairline_y: float | None = None   # forehead hairline height (metres); None -> the hairline_frac rule
    floor_quantum: float = 0.0        # > 0: every floor level snaps down to a multiple of this (metres)
    ridges: str = ""                  # "" | "top" | "top_back"
    ridge_period: int = 3             # one ridge column every this many columns
    ridge_phase: int = 2              # (column index of |x| + phase) % period < ridge_width -> ridge
    ridge_width: int = 1              # ridge width in columns
    ridge_above: float = 0.025        # ridges only this far above the hair floor
    ridge_from_front: float = 0.015   # ridges start this far behind the hair's front face (the ledge stays clean)
    lit_band: float = 0.035           # upward faces within this of the crown are lit
    deep_band: float = 0.012          # the band above the floor behind the ears is deep
    smooth_passes: int = 2

    @classmethod
    def for_style(cls, style: str, **overrides) -> "HairParams":
        if style not in STYLES:
            raise ValueError(f"unknown hair style {style!r}, expected one of {STYLES}")
        base = {
            "quiff": dict(top=0.012, side=0.008, back=0.012, quiff=replace(QUIFF), lit_band=0.10),
            "short": dict(top=0.015, side=0.010, back=0.015, ridges="top", lit_band=0.012, shape_power=3.0),
            "slick": dict(top=0.012, side=0.008, back=0.020, ridges="top_back", nape_drop=0.010, lit_band=0.012,
                          ridge_above=0.0, shape_power=3.0),
            "messy": dict(top=0.018, side=0.010, back=0.015, clumps=list(MESSY_CLUMPS), lit_band=0.030,
                          shape_power=3.0),
        }[style]
        base.update(overrides)
        return cls(style=style, **base)


@dataclass
class HairResult:
    hair: np.ndarray      # bool (nz, ny, nx) on the hair grid
    lit: np.ndarray       # subset of hair: the lighter tone
    deep: np.ndarray      # subset of hair: the darker tone (lit wins where both)
    scalp: np.ndarray     # form voxels under the hair (above the floor, ears excluded)
    floor: np.ndarray     # (nz, ny, nx) metres: the hair floor height broadcast over the grid
    stats: dict
    params: HairParams | None = None


# ---------------------------------------------------------------------------- floor and scalp
def hair_floor(X: np.ndarray, Z: np.ndarray, lm: dict, p: HairParams) -> np.ndarray:
    """Lower edge of the hair as a height (metres) over (x, z), piecewise flat. Front of the head
    (|x| inside the skull sides): the forehead hairline, dropped by ``temple_drop`` outward of
    ``temple_x``. The skull sides (|x| > ``ear_inner_x`` - ``side_in``): ``ear_clearance`` above
    the ear top, except the sideburn strip ``sideburn_depth`` in front of the ear, which reaches
    mid-ear. Behind the ear: the ear's bottom minus ``nape_drop``."""
    brow, top = float(lm["brow_y"]), float(lm["head_top"])
    y_hl = float(p.hairline_y) if p.hairline_y is not None else brow + p.hairline_frac * (top - brow)
    ear_z0, ear_z1 = lm["ear_z"]
    ear_y0, ear_y1 = lm["ear_y"]
    q = float(p.floor_quantum)
    snap = (lambda v: math.floor(v / q + 1e-6) * q) if q > 0 else (lambda v: v)
    ax = np.abs(X)
    forehead = np.where(ax > p.temple_x, snap(y_hl - p.temple_drop), snap(y_hl))
    sideburn = snap(0.5 * (ear_y0 + ear_y1))
    over_ear = snap(ear_y1 + p.ear_clearance)
    nape = snap(ear_y0 - p.nape_drop)
    side = ax > lm["ear_inner_x"] - p.side_in
    behind = Z > ear_z1 + 0.008
    strip = (Z >= ear_z0 - p.sideburn_depth) & (Z < ear_z0)
    front = Z < ear_z0 - p.sideburn_depth
    side_floor = np.where(strip, sideburn, over_ear)
    front_floor = np.where(front, forehead, over_ear)
    return np.where(behind, nape, np.where(side, side_floor, front_floor))


def ear_zone(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, lm: dict, grow: float = 0.004) -> np.ndarray:
    """The ear boxes (both sides), grown by ``grow``: no scalp and no hair there."""
    return (np.abs(X) > lm["ear_inner_x"] - grow) & (Y > lm["ear_y"][0] - grow) & (Y < lm["ear_y"][1] + grow) \
        & (Z > lm["ear_z"][0] - grow) & (Z < lm["ear_z"][1] + grow)


def scalp_mask(form: np.ndarray, grid: meshvox.VoxelGrid, lm: dict, p: HairParams) -> tuple[np.ndarray, np.ndarray]:
    """(scalp, floor): the form voxels above the hair floor outside the ear boxes, and the floor."""
    X, Y, Z = grid.centres()
    floor = hair_floor(X, Z, lm, p)
    scalp = form & (Y >= floor) & ~ear_zone(X, Y, Z, lm)
    return scalp, floor


# ---------------------------------------------------------------------------- geometry
def sweep(scalp_surface: np.ndarray, vs: float, p: HairParams) -> np.ndarray:
    """Minkowski sum of the scalp surface with the anisotropic superellipsoid of offsets
    (``side`` x, ``top`` up / one voxel down, ``back`` / ``ledge`` in z)."""
    rx, ry, rz, rf = (max(0.0, v / vs) for v in (p.side, p.top, p.back, p.ledge))
    out = np.zeros(scalp_surface.shape, bool)
    q = p.shape_power
    for dx in range(-int(math.ceil(rx)), int(math.ceil(rx)) + 1):
        for dy in range(-1, int(math.ceil(ry)) + 1):
            for dz in range(-int(math.ceil(rf)), int(math.ceil(rz)) + 1):
                rzz = rz if dz >= 0 else rf
                r = (abs(dx) / (rx + 0.5)) ** q + (max(dy, 0) / (ry + 0.5)) ** q + (abs(dz) / (rzz + 0.5)) ** q
                if r <= 1.0:
                    out |= shift(scalp_surface, dx, dy, dz)
    return out


def column_tops(mask: np.ndarray) -> np.ndarray:
    """Highest row index of ``mask`` per (z, x) column, -1 where empty."""
    ny = mask.shape[1]
    has = mask.any(axis=1)
    return np.where(has, ny - 1 - np.argmax(mask[:, ::-1, :], axis=1), -1)


def add_clumps(hair: np.ndarray, form: np.ndarray, grid: meshvox.VoxelGrid, clumps: list[Clump]) -> tuple[np.ndarray, int]:
    """Stepped spikes standing on the base mass. Each clump is a square pyramid in metres
    (half-width / half-depth shrink linearly to half a voxel at the tip), whose centre moves
    back ``lean`` per metre of rise; its columns are filled down to the base so nothing floats.
    Clumps with x > 0 are mirrored. Returns (hair, clumps placed)."""
    if not clumps:
        return hair, 0
    vs = grid.voxel_size
    nz, ny, nx = hair.shape
    xs = (grid.origin[0] + (np.arange(nx) + 0.5) * vs)
    ys = (grid.origin[1] + (np.arange(ny) + 0.5) * vs)
    zs = (grid.origin[2] + (np.arange(nz) + 0.5) * vs)
    base_top = column_tops(hair)                                   # (nz, nx)
    has_col = base_top >= 0
    # the base's front z per x column (metres), over the whole hair mass (before any clump)
    front_idx = np.where(has_col.any(axis=0), np.argmax(has_col, axis=0), -1)   # (nx,)
    placed = 0
    rows = np.arange(ny)[None, :, None]
    out = hair.copy()
    for c in clumps:
        for sx in ((1,) if c.x < vs * 0.25 else (1, -1)):
            cx = sx * c.x
            ix = int(np.argmin(np.abs(xs - cx)))
            if front_idx[ix] < 0:
                continue
            cz0 = zs[front_idx[ix]] + c.z
            foot0 = (np.abs(xs[None, :] - cx) <= c.width / 2 + 1e-9) & (np.abs(zs[:, None] - cz0) <= c.depth / 2 + 1e-9)
            under = base_top[foot0 & has_col]
            if under.size == 0:
                continue
            y0 = int(under.max()) + 1
            H = max(1, int(round(c.height / vs)))
            clump = np.zeros(hair.shape, bool)
            for k in range(H):
                row = y0 + k
                if row >= ny:
                    break
                t = k / H
                wx = max(vs * 0.5, (c.width + (c.top_width - c.width) * t) / 2)
                wz = max(vs * 0.5, (c.depth + (c.top_depth - c.depth) * t) / 2)
                czk = cz0 + c.lean * k * vs
                foot = (np.abs(xs[None, :] - cx) <= wx + 1e-9) & (np.abs(zs[:, None] - czk) <= wz + 1e-9)
                clump[:, row, :] |= foot
            # fill every column from the base top up to the clump's highest voxel
            above_or_at = np.maximum.accumulate(clump[:, ::-1, :], axis=1)[:, ::-1, :]
            fill = above_or_at & (rows > base_top[:, None, :]) & has_col[:, None, :]
            out |= (clump | fill) & ~form
            base_top = column_tops(out)
            has_col = base_top >= 0
            placed += 1
    return out, placed


def add_quiff(hair: np.ndarray, form: np.ndarray, grid: meshvox.VoxelGrid, q: Quiff) -> tuple[np.ndarray, np.ndarray, dict]:
    """The stair pompadour on the base mass ``hair``. The ledge is the hair's front-most column
    at x = 0; the stair starts on the row above the rows where the form's front sits right
    behind the ledge (the flat forehead) and every step is a box ``step`` tall, ``depth`` deep,
    set back ``lean`` x ``step`` per step, ``half_width`` tapering to ``half_width_top`` at the
    crest (``height`` above the base's top in the slab's footprint); the spikes continue the
    stair on the crest; a step's front is clamped one voxel in front of the form (so on a round
    skull the stair hugs the scalp until the diagonal clears it). Base hair in front of the
    diagonal is removed. Returns (hair, the stair mask, stats)."""
    vs = grid.voxel_size
    nz, ny, nx = hair.shape
    X, Y, Z = grid.centres()
    xs = X[0, 0, :]
    ix = int(np.argmin(np.abs(xs)))
    col = hair[:, :, ix]                                            # (nz, ny)
    if not col.any():
        return hair, np.zeros(hair.shape, bool), {"steps": 0, "spikes": 0}
    zf = int(np.argmax(col.any(axis=1)))                            # the ledge: front-most hair z
    k = max(1, int(round(q.step / vs)))                             # fine rows per step
    lean_vox = max(0, int(round(q.lean * q.step / vs)))
    depth_vox = max(1, int(round(q.depth / vs)))
    # rows whose form front (centre column) is the flat forehead behind the ledge
    fcol = form[:, :, ix]
    ffront = np.where(fcol.any(axis=0), np.argmax(fcol, axis=0), -1)   # (ny,) front z index of the form
    ledge_rows = np.nonzero(col[zf])[0]
    y_hl = int(ledge_rows.min())
    z_face = int(ffront[y_hl]) if ffront[y_hl] >= 0 else zf + 1
    flat = np.nonzero((ffront == z_face) & (np.arange(ny) >= y_hl))[0]
    y_start = int(flat.max()) + 1 if flat.size else int(ledge_rows.max()) + 1
    # the crest: ``height`` above the base's top under the slab
    base_top = column_tops(hair)
    zi = np.arange(nz)[:, None]
    foot = (np.abs(xs)[None, :] <= q.half_width + 1e-9) & (zi >= zf) & (zi < zf + depth_vox) & (base_top >= 0)
    y_base = int(base_top[foot].max()) if foot.any() else y_start - 1
    y_crest = max(y_start + k - 1, y_base + int(round(q.height / vs)))
    n_steps = max(1, -(-(y_crest - y_start + 1) // k))
    snap = lambda v: max(vs, math.floor(v / q.step + 1e-6) * q.step)
    stair = np.zeros(hair.shape, bool)
    cut = np.zeros(hair.shape, bool)

    def place(s: int, cols: np.ndarray) -> None:
        y0 = y_start + s * k
        if y0 >= ny:
            return
        y1 = min(ny - 1, y0 + k - 1)
        z0 = zf + (s + 1) * lean_vox
        fr = ffront[y0:y1 + 1]
        fr = fr[fr >= 0]
        if fr.size:                                                 # never behind the scalp: a round skull keeps its cover
            z0 = min(z0, int(fr.min()) - 1)
        z1 = min(nz - 1, z0 + depth_vox - 1)
        if z0 >= nz:
            return
        stair[z0:z1 + 1, y0:y1 + 1, cols] = True
        cut[:z0, y0:y1 + 1, cols] = True

    for s in range(n_steps):
        t = s / max(1, n_steps - 1)
        hw = snap(q.half_width + (q.half_width_top - q.half_width) * t)
        place(s, np.abs(xs) <= hw + 1e-9)
    n_spikes = 0
    for x, shw, steps in q.spikes:
        for sx in ((1,) if x < vs * 0.25 else (1, -1)):
            for j in range(int(steps)):
                place(n_steps + j, np.abs(xs - sx * x) <= shw + 1e-9)
            n_spikes += 1
    stair &= ~form
    out = (hair & ~cut) | stair
    out |= out[:, :, ::-1]
    stats = {"steps": n_steps, "spikes": n_spikes, "rows": [y_start, y_start + n_steps * k - 1], "stair": int(stair.sum())}
    return out, stair & out, stats


def add_ridges(hair: np.ndarray, form: np.ndarray, grid: meshvox.VoxelGrid, floor: np.ndarray, lm: dict,
               p: HairParams) -> tuple[np.ndarray, int]:
    """One-voxel ridges on pattern columns: on the upward faces (``top``) and, for ``top_back``,
    on the backward faces behind the ears too (combed-back strands that run down the nape)."""
    if not p.ridges:
        return hair, 0
    X, Y, Z = grid.centres()
    vs = grid.voxel_size
    col = (np.floor(np.abs(X[0, 0, :]) / vs).astype(int) + p.ridge_phase) % p.ridge_period < p.ridge_width   # (nx,)
    cols = np.broadcast_to(col[None, None, :], hair.shape)
    filled = hair | form
    has_col = hair.any(axis=1)                                                    # (nz, nx)
    front_idx = np.where(has_col.any(axis=0), np.argmax(has_col, axis=0), 0)      # (nx,)
    z_front = Z[front_idx, 0, np.arange(hair.shape[2])]                           # (nx,) metres
    behind_front = Z >= z_front[None, None, :] + p.ridge_from_front
    top_faces = hair & ~shift(filled, 0, -1, 0) & (Y >= floor + p.ridge_above) & behind_front
    ridge = shift(top_faces, 0, 1, 0) & ~filled & cols
    if p.ridges == "top_back":
        back_faces = hair & ~shift(filled, 0, 0, -1) & (Z > lm["ear_z"][1]) & (Y >= floor + vs)
        ridge |= shift(back_faces, 0, 0, 1) & ~filled & cols
    return hair | ridge, int(ridge.sum())


def tone_masks(hair: np.ndarray, added: np.ndarray, form: np.ndarray, Y: np.ndarray, Z: np.ndarray, floor: np.ndarray,
               lm: dict, p: HairParams) -> tuple[np.ndarray, np.ndarray]:
    """(lit, deep): upward faces of the clump / ridge geometry and of the base's crown terrace,
    within ``lit_band`` of the crown (the lower terraces of the base stay base, so a domed cap
    never turns into a contour map); undersides and the nape band."""
    filled = hair | form
    up = hair & ~shift(filled, 0, -1, 0)
    crown = float(Y[hair].max()) if hair.any() else 0.0
    base = hair & ~added
    base_crown = float(Y[base].max()) if base.any() else crown
    vs = float(Y[0, 1, 0] - Y[0, 0, 0]) if Y.shape[1] > 1 else 0.0
    lit = up & (added | (Y >= base_crown - vs * 0.5)) & (Y >= crown - p.lit_band)
    deep = hair & (~shift(filled, 0, 1, 0) | ((Y < floor + p.deep_band) & (Z > lm["ear_z"][0])))
    return lit, deep & ~lit


# ---------------------------------------------------------------------------- the builder
def build_hair(form: np.ndarray, grid: meshvox.VoxelGrid, lm: dict, style: str | HairParams = "quiff",
               **overrides) -> HairResult:
    """The hair of ``style`` on ``grid`` (any voxel size) over the head ``form`` (bool on the same
    grid, metre landmarks ``lm`` of :meth:`voxgen.mh.MakeHuman.landmarks`)."""
    p = style if isinstance(style, HairParams) else HairParams.for_style(style)
    if overrides:
        p = replace(p, **overrides)
    vs = grid.voxel_size
    X, Y, Z = grid.centres()
    scalp, floor = scalp_mask(form, grid, lm, p)
    surface = scalp & exposed(form)
    above = Y >= floor
    hair = sweep(surface, vs, p) & ~form & above & ~ear_zone(X, Y, Z, lm)
    hair = smooth_cap(hair, form, p.smooth_passes) & ~form & above
    hair |= hair[:, :, ::-1]
    base_mask = hair
    base = int(hair.sum())
    hair, n_clumps = add_clumps(hair, form, grid, p.clumps)
    hair |= hair[:, :, ::-1]
    added = hair & ~base_mask                                      # clumps only: ridges stay base-toned
    qstats: dict = {}
    if p.quiff is not None:
        hair, stair, qstats = add_quiff(hair, form, grid, p.quiff)
        added |= stair
    hair, n_ridge = add_ridges(hair, form, grid, floor, lm, p)
    hair |= hair[:, :, ::-1]
    labels, n = label_components(hair)
    crumbs = 0
    if n > 1:
        crumbs = int((labels > 1).sum())
        hair = labels == 1
    lit, deep = tone_masks(hair, added & hair, form, Y, Z, floor, lm, p)
    stats = {"style": p.style, "voxel_mm": round(vs * 1000, 2), "base": base, "clumps": n_clumps, "ridge": n_ridge,
             "crumbs_removed": crumbs, "solid": int(hair.sum()), "lit": int(lit.sum()), "deep": int(deep.sum())}
    if qstats:
        stats["quiff"] = qstats
    return HairResult(hair, lit, deep, scalp, floor, stats, p)
