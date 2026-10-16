"""Layered voxel head from the MakeHuman base mesh (replaces the hand-made SDF head).

Pipeline (``build_head``): take the ``body`` mesh above the neck joint, voxelize its
surface together with the eyeball helpers and flood-fill from outside (the cut neck
is sealed) -> one solid; a multi-source BFS from the air gives every voxel its depth,
and the tissues are assigned by depth: skin (>= 5 mm), fat (~4 mm, thicker on the
cheeks), muscle, bone where the depth exceeds 9 mm above the brow line and 16 mm
below it (neck: only the back half = spine). Teeth come from the tooth-row helpers,
eyes from the eyeball helpers (iris + one highlight on the exposed front), ear and
nose cores are cartilage. Hair is a solid cap: the air within ``hair_thickness``
(Euclidean distance) of the skin, cut at a smooth lower edge (``hair_floor``: forehead
hairline -> sideburn to mid-ear -> nape at mouth level), thinned toward that edge,
trimmed around the ears, closed and de-spiked.
Skin tints (colour only): lips with a darker mouth line, eyebrows (inner end thicker),
a uniform stubble tint with fine noise (5 o'clock shadow), two-band cheek blush,
nostrils. Finally the result is mirrored about x = 0 and crumbs are removed.

Caricature: ``HeadParams.morphs`` are MakeHuman modifier weights (:mod:`voxgen.mh_targets`)
applied to the mesh before everything else, so every landmark (eyes, mouth, ears, chin)
comes from the morphed mesh. Stylized look (``HeadParams.stylized()``): the form is cleaned
(3x3x3 closing then opening, thin sheets such as the ears kept) or voxelized ``form_voxel_mult``
times coarser and upsampled (chunky, deliberate steps) with the details carved on the fine
grid; flat tints (no per-voxel noise: a disciplined palette of 3 skin tones, 3 hair tones,
eye white/iris/highlight, lip and one dark line colour), bigger simpler eyes (n x n iris block,
one highlight, a dark upper-lid line), bold brows, a thin lip band with a mouth line, no blush.
Relief (``HeadParams.relief``, meant for 3 mm): the features as geometry, every step exactly one
voxel (a 2-voxel step at 3 mm reads as a blob): the eye opening loses its top row, which becomes
the dark lid line, and the row above it protrudes one voxel as the upper lid (skin); both lip rows
protrude one voxel so the mouth line is a recess between them, with the corner voxels one row up;
the nostrils are two-voxel slots carved two voxels back into the bottom-front edge of the nose
(open to the front and below) and lined dark inside, not around; the ear's silhouette edge (helix) comes one voxel further
out; the brows protrude one voxel as hair material in the brow tone (a razor can take them off).
Lid, lips, nostrils and ear rim are applied to the form *before* the tissues are layered, so the
added voxels are skin and the holes get a skin lining by depth.
``HeadParams.mid(strength)`` is the "middle" caricature the owner asked for between the realistic
mesh (features by colour only) and the old SDF head (features as lumps): the stylized 3 mm head
with relief and ``MID_MORPHS`` (brow ridge, bigger nose, fuller lips, stronger jaw and cheekbones,
bigger ears, hooded lids) scaled by ``strength``.

Local space of the part: metres, Y up, face toward -Z, chin at y ~ 0, centred on x/z.
The hair helper of MakeHuman is a hull around the whole head and face, so the hairline
is a parameter (``hairline_frac`` of the brow-to-crown distance, ``hairline_up`` on top),
not the helper's edge.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from . import materials as M
from . import meshvox, mh
from .body import dilate, erode, exposed, neighbour_count, shift
from .checks import label_components
from .facedetail import front_face, hole_walls, protrusion, recess
from .model import Model, Part
from .palette import Palette

INF = float("inf")


@dataclass
class HeadParams:
    """All lengths in metres unless the name says ``_mm``."""

    neck_keep: float = 0.0               # neck kept below the neck joint (0 = cut at the joint)
    margin: float = 0.02                 # air around the head in the grid
    skin_mm: float = 5.0
    fat_mm: float = 4.0
    cheek_fat_extra_mm: float = 4.0
    bone_depth_above_brow_mm: float = 9.0
    bone_depth_below_brow_mm: float = 16.0
    cartilage: bool = True
    vessels: bool = True
    eye_open: float = 0.004              # half-height of the forced lid opening (0 = mesh lids only)
    iris_radius: float = 0.0065
    hair: bool = True
    hair_thickness: float = 0.015        # cap thickness along the skin normal (12-18 mm = short haircut)
    hairline_frac: float = 0.36          # front hairline: this fraction of the way from the brow to the crown
    hairline_up: float = 0.0             # raise (+) / lower (-) the front hairline on top of that
    hair_taper: float = 0.015            # length over which the cap thins toward its lower edge
    hair_edge: float = 0.6               # thickness fraction at the edge (1 = blunt)
    nape_taper: float = 0.05             # the back thins over this length up from the nape (tapered neckline)
    temple_recess: float = 0.0           # how much higher the hairline sits at the temples
    sideburn_below_ear_top: float = 0.02 # sideburns end this far below the ear top (mid-ear)
    nape_up: float = 0.0                 # nape hairline relative to the mouth level
    ear_trim: float = 0.006              # hair above the ear is trimmed to this far out of the ear root
    fringe: bool = False
    fringe_length: int = 3               # voxels
    eyebrows: bool = True
    lips: bool = True
    stubble: bool = True
    blush: bool = True
    nostrils: bool = True
    nostril_radius: float = 0.0035       # 1 voxel at 5 mm, a small cluster at 3 mm
    mirror: bool = True
    seed: int = 7
    # caricature and the stylized look
    morphs: dict = field(default_factory=dict)  # MakeHuman modifier -> weight (voxgen.mh_targets); |w| > 1 = caricature
    form_voxel_mult: int = 1             # voxelize the form this many times coarser and upsample (chunky steps)
    clean_surface: bool = False          # 3x3x3 closing then opening of the form (thin sheets kept) before layering
    flat_tints: bool = False             # disciplined palette, no per-voxel colour noise, hair shaded in bands
    brow_rows: int = 0                   # rows of brow voxels (0 = as many as 5 mm holds)
    lip_half_height: float = 0.006       # half-height of the lip band
    lid_line: bool = False               # one dark voxel row along the upper lid
    iris_voxels: int = 0                 # 0 = disc of iris_radius; n = n x n block with one highlight voxel
    eye_open_w: float = 0.85             # half-width of the forced lid opening as a fraction of the eyeball radius
    relief: bool = False                 # features as one-voxel geometry: lid ridge, lips, nostril holes, brows

    @classmethod
    def stylized(cls, **overrides) -> "HeadParams":
        """The stylized cubes look (meant for 3 mm): clean form, flat tints, bigger simpler
        eyes, bold brows, thin lips, no blush, uniform stubble. Any field can be overridden."""
        base = dict(clean_surface=True, flat_tints=True, blush=False, brow_rows=2, lip_half_height=0.0035,
                    lid_line=True, iris_voxels=3, eye_open=0.0065, eye_open_w=0.95, nostril_radius=0.0032)
        base.update(overrides)
        return cls(**base)

    @classmethod
    def mid(cls, strength: float = 1.0, **overrides) -> "HeadParams":
        """The stylized 3 mm head with relief and the ``MID_MORPHS`` caricature scaled by
        ``strength`` (soft ~0.65, medium ~1.2, strong ~1.9); stubble off. ``morphs`` in
        ``overrides`` are added on top (``{"nose-scale-vert": 0.3}``), other fields override."""
        morphs = {k: round(w * strength, 3) for k, w in MID_MORPHS.items()}
        for k, w in overrides.pop("morphs", {}).items():
            morphs[k] = round(morphs.get(k, 0.0) + float(w), 3)
        base = dict(relief=True, stubble=False, morphs=morphs,
                    lip_half_height=0.0035 if strength < 1.5 else 0.0068)   # 3 lip rows, 5 when strong
        base.update(overrides)
        return cls.stylized(**base)


# the "middle" caricature at strength 1: MakeHuman modifier weights (voxgen.mh_targets), l-/r- pairs
# expanded. Brow ridge forward, bigger deeper nose, fuller lips, stronger chin and jaw, cheekbones,
# bigger ears, hooded lids and a hint of eye bags; nothing that sinks the eyes.
MID_MORPHS = {
    "eyebrows-trans-forward": 0.5,
    "forehead-trans-forward": 0.2,
    "nose-scale-vert": 0.9,
    "nose-scale-depth": 1.0,
    "nose-scale-horiz": 0.4,
    "nose-point-width": 0.7,
    "nose-nostrils-width": 0.6,
    "mouth-upperlip-height": 0.8,
    "mouth-lowerlip-height": 0.8,
    "mouth-scale-depth": 0.8,
    "mouth-cupidsbow-width": 0.6,
    "mouth-upperlip-volume": 0.6,
    "mouth-lowerlip-volume": 0.6,
    "chin-prominent": 0.8,
    "chin-width": 0.7,
    "chin-jaw-drop": 0.3,
    "cheek-bones": 0.8,
    "ear-scale": 0.6,
    "eye-eyefold-down": 0.6,
    "eye-bag-incr": 0.3,
}


# colour names the builder asks for -> palette entries of the stylized (flat) palette
FLAT_ALIAS = {"skin_dark": "skin_shadow", "skin_ruddy_soft": "skin_ruddy", "stubble": "skin_shadow",
              "stubble_light": "skin_shadow", "stubble_dark": "skin_shadow", "lip_line": "line",
              "nostril": "line", "lid_line": "line", "brow": "hair_deep"}
RICH_ALIAS = {"lid_line": "lip_line", "brow": "hair"}


def make_palette(flat: bool = False) -> Palette:
    pal = Palette()
    if flat:
        # palette discipline: 3 skin tones, lip, one dark line (+ 3 hair, eye white/iris/highlight below)
        pal.add("skin", (230, 184, 152))
        pal.add("skin_shadow", (198, 156, 140))   # cool shadow tone: stubble
        pal.add("skin_ruddy", (230, 162, 140))    # blush, when on
        pal.add("lips", (208, 132, 120))
        pal.add("line", (110, 62, 58))            # mouth line, upper-lid line, nostrils
    else:
        pal.add("skin", (222, 178, 150))
        pal.add("skin_dark", (204, 160, 134))
        pal.add("skin_ruddy", (226, 166, 144))
        pal.add("skin_ruddy_soft", (224, 172, 147))
        pal.add("stubble", (193, 157, 134))       # skin darkened ~12 % and cooled
        pal.add("stubble_light", (201, 163, 139))
        pal.add("stubble_dark", (185, 151, 129))
        pal.add("lips", (210, 151, 131))          # 50 % lip red, 50 % skin
        pal.add("lip_line", (160, 100, 92))
        pal.add("nostril", (104, 66, 56))
    pal.add("fat", (238, 214, 140))
    pal.add("muscle", (150, 40, 45))
    pal.add("bone", (235, 225, 200))
    pal.add("tooth", (245, 240, 220))
    pal.add("tooth_dark", (232, 224, 198))
    pal.add("cartilage", (225, 185, 160))
    pal.add("vessel", (120, 20, 30))
    pal.add("hair", (58, 40, 30))
    pal.add("hair_light", (72, 52, 40))
    pal.add("hair_deep", (46, 32, 24))
    pal.add("eye_white", (240, 240, 235))
    pal.add("iris", (72, 126, 190))
    pal.add("highlight", (246, 248, 250))
    return pal


# ---------------------------------------------------------------------------
# small helpers (each mask is a separate function so the builder stays readable)
# ---------------------------------------------------------------------------

def depth_map(solid: np.ndarray) -> np.ndarray:
    """6-connected BFS depth from the air: 0 for air, 1 for surface voxels, 2 for the
    voxels behind them, ... (the grid border does not count as air)."""
    depth = np.zeros(solid.shape, np.int16)
    cur = exposed(solid)
    remaining = solid & ~cur
    d = 1
    while cur.any():
        depth[cur] = d
        nxt = dilate(cur) & remaining
        remaining &= ~nxt
        cur = nxt
        d += 1
    depth[remaining] = d  # unreachable (never for a connected solid)
    return depth


def box_dilate(mask: np.ndarray) -> np.ndarray:
    """26-connected (3x3x3 box) binary dilation."""
    out = mask
    for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        out = out | shift(out, dx, dy, dz) | shift(out, -dx, -dy, -dz)
    return out


def box_erode(mask: np.ndarray) -> np.ndarray:
    """3x3x3 box erosion (the grid border counts as solid, so nothing is eaten there)."""
    return ~box_dilate(~mask)


def box_neighbours(mask: np.ndarray) -> np.ndarray:
    """Number of solid voxels among the 26 neighbours."""
    n = np.zeros(mask.shape, np.int16)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx or dy or dz:
                    n += shift(mask, dx, dy, dz)
    return n


def clean_form(solid: np.ndarray, keep_neighbours: int = 12) -> np.ndarray:
    """Morphological cleanup of the form: 3x3x3 closing (one-voxel pits and notches filled)
    then opening (one-voxel bumps, ridges and stair-step flicker removed). The opening would
    also delete every sheet thinner than three voxels (the ears), so voxels of the closed
    form with at least ``keep_neighbours`` solid 26-neighbours survive it: a two-voxel slab
    keeps 17, a lone bump on a flat surface has 9."""
    closed = box_erode(box_dilate(solid))
    opened = box_dilate(box_erode(closed))
    return opened | (closed & (box_neighbours(closed) >= keep_neighbours))


def upsample(mask: np.ndarray, k: int) -> np.ndarray:
    """Every voxel becomes a k x k x k block."""
    return mask.repeat(k, axis=0).repeat(k, axis=1).repeat(k, axis=2)


def form_solid(grid: meshvox.VoxelGrid, body, eyes, mult: int, clean: bool) -> tuple[np.ndarray, dict]:
    """The head solid on ``grid``: body surface + eyeballs voxelized on a grid ``mult`` times
    coarser, flood-filled from outside (the neck cut is sealed), optionally cleaned
    (:func:`clean_form`, on the coarse grid), upsampled to ``grid``."""
    if mult > 1:
        coarse = meshvox.VoxelGrid(grid.origin, grid.voxel_size * mult, tuple(d // mult for d in grid.dims))
    else:
        coarse = grid
    surf = meshvox.voxelize_surface(body, coarse)
    for e in eyes:
        meshvox.voxelize_surface(e, coarse, out=surf)
    solid, info = meshvox.fill_interior(surf, sealed_faces=("-y",))
    if info["leak"]:
        raise RuntimeError(f"mesh voxelization leaked: {info}")
    info = dict(info, form_voxel_mult=mult)
    if clean:
        before = int(solid.sum())
        solid = clean_form(solid)
        info["cleaned_delta"] = int(solid.sum()) - before
    if mult > 1:
        solid = upsample(solid, mult)
    return solid, info


def hash01(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic per-voxel pseudo-random in [0, 1)."""
    h = (ix.astype(np.uint32) * np.uint32(73856093)) ^ (iy.astype(np.uint32) * np.uint32(19349663)) \
        ^ (iz.astype(np.uint32) * np.uint32(83492791)) ^ np.uint32((seed * 2654435761) & 0xFFFFFFFF)
    h = (h ^ (h >> np.uint32(13))) * np.uint32(0x5BD1E995)
    h = h ^ (h >> np.uint32(15))
    return h.astype(np.float64) / 4294967296.0


def smoothstep(a: float, b: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - a) / max(b - a, 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def hairline_y(x: np.ndarray, lm: dict, p: HeadParams) -> np.ndarray:
    """Front hairline height as a function of |x|: ``hairline_frac`` of the way from the
    brow to the crown (plus ``hairline_up``), raised by ``temple_recess`` at the temples."""
    base = lm["brow_y"] + p.hairline_frac * (lm["head_top"] - lm["brow_y"]) + p.hairline_up
    return base + p.temple_recess * smoothstep(0.03, 0.06, np.abs(x))


def hair_floor(x: np.ndarray, z: np.ndarray, lm: dict, p: HeadParams) -> np.ndarray:
    """Lower edge of the hair as a height over (x, z): the front hairline on the forehead,
    dropping to the sideburn level (mid-ear) in front of the ear, level past the ear and
    dropping again behind it to the nape (mouth level). Smooth in x and z, so the voxel
    outline is only stair-stepped, never jagged."""
    ear_z0, ear_z1 = lm["ear_z"]
    sideburn = lm["ear_y"][1] - p.sideburn_below_ear_top
    nape = lm["mouth"][1] + p.nape_up
    t_side = smoothstep(ear_z0 - 0.032, ear_z0 - 0.008, z)
    t_back = smoothstep(ear_z1, ear_z1 + 0.035, z)
    floor = hairline_y(x, lm, p) * (1.0 - t_side) + sideburn * t_side
    return floor * (1.0 - t_back) + nape * t_back


def air_distance(solid: np.ndarray, radius: float) -> np.ndarray:
    """Euclidean distance (in voxels) from every air voxel to the nearest exposed solid
    voxel centre, up to ``radius`` (inf beyond, 0 inside the solid). Brute force over
    the ball of offsets, nearest first (~260 shifts at radius 4), no scipy."""
    surf = exposed(solid)
    dist = np.full(solid.shape, np.inf, np.float32)
    r = int(math.ceil(radius))
    offs = [((dx, dy, dz), math.sqrt(dx * dx + dy * dy + dz * dz))
            for dx in range(-r, r + 1) for dy in range(-r, r + 1) for dz in range(-r, r + 1)]
    offs = sorted((o for o in offs if 0 < o[1] <= radius), key=lambda o: o[1])
    for (dx, dy, dz), d in offs:
        m = shift(surf, dx, dy, dz) & np.isinf(dist)
        dist[m] = d
    dist[solid] = 0.0
    return dist


def hair_cap(grid: meshvox.VoxelGrid, solid: np.ndarray, lm: dict, p: HeadParams) -> np.ndarray:
    """Solid hair cap: the air within ``hair_thickness`` (Euclidean, so the outer surface
    is smooth, not a Manhattan checkerboard) of the skin, cut at the smooth
    ``hair_floor``, thinned to ``hair_edge`` of the thickness over ``hair_taper`` above
    that edge, trimmed around the ears; closed (dilate + erode) and de-spiked."""
    X, Y, Z = grid.centres()
    floor = hair_floor(X, Z, lm, p)
    ear_z0, ear_z1 = lm["ear_z"]
    ear_y1 = lm["ear_y"][1]
    ear_zone = (np.abs(X) > lm["ear_inner_x"] - 0.004) & (Y < ear_y1 + 0.004) & (Z > ear_z0 - 0.008) & (Z < ear_z1 + 0.008)
    # above the ear the cap hugs the skull instead of hanging over the ear
    ear_zone |= (np.abs(X) > lm["ear_inner_x"] + p.ear_trim) & (Y < ear_y1 + 0.014) & (Z > ear_z0 - 0.012) & (Z < ear_z1 + 0.012)
    k = p.hair_thickness / grid.voxel_size
    t_back = smoothstep(ear_z1, ear_z1 + 0.035, Z)
    taper = p.hair_taper * (1.0 - t_back) + p.nape_taper * t_back
    w = p.hair_edge + (1.0 - p.hair_edge) * np.clip((Y - floor) / np.maximum(taper, 1e-6), 0.0, 1.0)
    dist = air_distance(solid, k + 0.5)
    hair = ~solid & (dist <= k * w + 0.5) & (Y >= floor) & ~ear_zone
    # closing fills one-voxel notches of the outline; then spikes off, pits filled
    hair = erode(dilate(hair | solid)) & ~solid & ~ear_zone & (Y >= floor - grid.voxel_size)
    return smooth_cap(hair, solid)


def smooth_cap(hair: np.ndarray, solid: np.ndarray, passes: int = 2) -> np.ndarray:
    """Knock off one-voxel spikes (hair with < 3 hair/solid 6-neighbours) and fill
    one-voxel pits (air with >= 5 hair/solid neighbours, next to hair)."""
    hair = hair.copy()
    for _ in range(passes):
        n = neighbour_count(hair | solid)
        hair &= n >= 3
        n = neighbour_count(hair | solid)
        pits = ~solid & ~hair & (n >= 5) & dilate(hair)
        hair |= pits
    return hair


def pick_front_row(cand: np.ndarray, Z: np.ndarray, rows: int = 1) -> np.ndarray:
    """Per x column, take the row of the frontmost (smallest z) candidate voxel, smooth
    that row over neighbouring columns (median of 3) so the line does not stair-step,
    then keep the frontmost candidate in that row and the ``rows - 1`` rows above it (below
    it where the band has nothing above, e.g. a brow sitting on the top of its band)."""
    out = np.zeros(cand.shape, bool)
    zs, ys, xs = np.nonzero(cand)
    cols = np.unique(xs)
    if cols.size == 0:
        return out
    row0 = {}
    for ix in cols:
        sel = xs == ix
        zz, yy = zs[sel], ys[sel]
        order = np.lexsort((yy, Z[zz, yy, ix]))
        row0[int(ix)] = int(yy[order[0]])
    for ix in cols:
        ix = int(ix)
        neigh = [row0[j] for j in (ix - 1, ix, ix + 1) if j in row0]
        y_target = int(np.median(neigh))
        sel = xs == ix
        zz, yy = zs[sel], ys[sel]
        for r in range(rows):
            # nearest candidate row to the smoothed target, frontmost voxel in it
            dy = np.abs(yy - (y_target + r))
            if r and dy.min() > 1:
                dy = np.abs(yy - (y_target - r))
            best = dy.min()
            if best > 1:
                continue
            pick = (dy == best)
            k = np.argmin(np.where(pick, Z[zz, yy, ix], np.inf))
            out[zz[k], yy[k], ix] = True
    return out


def lid_opening(solid: np.ndarray, eyeball: np.ndarray, X, Y, Z, centre, half_w: float, half_h: float,
                reach: int = 2, top_cut: float = 0.0) -> np.ndarray:
    """Voxels to turn into eye so the eyeball shows through a small lid opening: inside
    the x/y ellipse (``half_w``, ``half_h``) around the eye centre, in every column where
    the eyeball lies at most ``reach`` voxels behind the front surface, the surface voxel
    and everything down to the eyeball become eye. Outside the ellipse the mesh lids stay.
    ``top_cut`` flattens the top: rows above ``centre_y + half_h - top_cut`` stay closed."""
    foot = (((X[0] - centre[0]) / half_w) ** 2 + ((Y[0] - centre[1]) / half_h) ** 2) <= 1.0  # (ny, nx)
    if top_cut > 0:
        foot &= Y[0] <= centre[1] + half_h - top_cut
    front = Z < centre[2]
    s = solid & front
    e = eyeball & front
    first = np.argmax(s, axis=0)
    efirst = np.argmax(e, axis=0)
    sel = foot & s.any(axis=0) & e.any(axis=0) & (efirst >= first) & (efirst - first <= reach)
    out = np.zeros(solid.shape, bool)
    iz = np.arange(solid.shape[0])[:, None, None]
    out |= sel[None, :, :] & (iz >= first[None, :, :]) & (iz <= efirst[None, :, :])
    return out


def capsule_mask(X, Y, Z, a, b, r: float) -> np.ndarray:
    """Voxels within ``r`` of the segment a-b (metres); |x| is used, so both sides get it."""
    ax, ay, az = a
    bx, by, bz = b
    dx, dy, dz = bx - ax, by - ay, bz - az
    px, py, pz = np.abs(X) - ax, Y - ay, Z - az
    t = np.clip((px * dx + py * dy + pz * dz) / max(dx * dx + dy * dy + dz * dz, 1e-12), 0.0, 1.0)
    return (px - t * dx) ** 2 + (py - t * dy) ** 2 + (pz - t * dz) ** 2 <= r * r


def nearest_voxel(mask: np.ndarray, X, Y, Z, target) -> tuple | None:
    """Index (z, y, x) of the voxel of ``mask`` closest to the metre point ``target``."""
    zs, ys, xs = np.nonzero(mask)
    if zs.size == 0:
        return None
    d = (X[zs, ys, xs] - target[0]) ** 2 + (Y[zs, ys, xs] - target[1]) ** 2 + (Z[zs, ys, xs] - target[2]) ** 2
    k = int(np.argmin(d))
    return (zs[k], ys[k], xs[k])


def iris_block(front: np.ndarray, X, Y, centre, n: int) -> tuple[np.ndarray, tuple | None]:
    """n x n block of exposed eye voxels (in x/y, whatever their z): centred on the opening
    (its mean column; the row nearest the eye centre), so the gaze is straight even when the
    lids open more to the temple side; plus the highlight voxel: top row, temple side.
    Returns (iris mask, highlight index or None)."""
    zs, ys, xs = np.nonzero(front)
    out = np.zeros(front.shape, bool)
    if zs.size == 0:
        return out, None
    xc = int(np.round(xs.mean()))
    k = int(np.argmin(np.abs(Y[zs, ys, xs] - centre[1]) + 1e-3 * np.abs(xs - xc)))
    yc = int(ys[k])
    h = n // 2
    sel = (np.abs(xs - xc) <= h) & (np.abs(ys - yc) <= h)
    out[zs[sel], ys[sel], xs[sel]] = True
    side = 1 if centre[0] > 0 else -1
    score = (ys[sel] == yc + h) * 4 + (xs[sel] == xc + side * h) * 2 + (ys[sel] * 2 + side * xs[sel]) * 1e-3
    j = int(np.argmax(score))
    return out, (zs[sel][j], ys[sel][j], xs[sel][j])


def mirror_x(part: Part) -> None:
    """Copy the -x half onto the +x half (grid must be x-centred)."""
    nx = part.nx
    half = nx // 2
    for arr in (part.material, part.color, part.state):
        arr[:, :, nx - half:] = arr[:, :, :half][:, :, ::-1]


def keep_largest_component(part: Part) -> int:
    """Remove every solid 6-component but the largest. Returns the voxels removed."""
    labels, n = label_components(part.solid())
    if n <= 1:
        return 0
    crumbs = (labels > 1)
    cnt = int(np.count_nonzero(crumbs))
    part.material[crumbs] = 0
    part.color[crumbs] = 0
    return cnt


def lip_band(Y: np.ndarray, mouth_y: float, half_height: float, vs: float) -> np.ndarray:
    """Rows of the lip band: ``half_height`` below the mouth line, one voxel less above it once
    the band is more than one row per lip (the lower lip is the fuller one), the line row included."""
    d = Y - mouth_y
    above = max(half_height - vs, vs * 1.5)
    return ((d < 0) & (-d <= half_height)) | ((d >= 0) & (d <= above))


def lid_line_rows(eye: np.ndarray, surf: np.ndarray, X, Z, centres) -> np.ndarray:
    """One smooth row of surface voxels right above each eye opening (the surface recedes under
    the brow, so the row is looked for a few voxels back and forth in z)."""
    out = np.zeros(eye.shape, bool)
    exposed_eye = eye & exposed(surf | eye)
    for ec in centres:
        front = exposed_eye & (Z < ec[2]) & (np.abs(X - ec[0]) < 0.02)   # this eye's opening
        top = front & ~shift(eye, 0, -1, 0)
        above = np.zeros(top.shape, bool)
        for dz in range(-3, 4):
            above |= shift(top, 0, 1, dz)
        out |= pick_front_row(surf & above, Z, 1)
    return out


def relief_pass(solid: np.ndarray, eye: np.ndarray, lid_line: np.ndarray, X, Y, Z, lm: dict, p: HeadParams,
                vs: float) -> tuple[np.ndarray, dict]:
    """Feature relief on the form before the tissues are layered: the upper-lid ridge (the row
    above the lid line comes one voxel forward), both lip rows one voxel forward (the mouth line
    stays as a recess), the nostrils carved two voxels deep. Returns the new solid and the masks
    (``ridge``, ``lips``, ``nostril_inner`` = the deeper hole voxels, whose walls get the dark tone,
    ``ear_rim`` = the helix, one voxel further out along the ear's silhouette)."""
    solid = solid.copy()
    surf = exposed(solid) & ~eye
    masks: dict[str, np.ndarray] = {}
    # upper lid: the surface row above the lid line, over the same columns
    above = np.zeros(solid.shape, bool)
    for dz in range(-3, 4):
        above |= shift(lid_line, 0, 1, dz)
    ridge = protrusion(pick_front_row(surf & above, Z, 1), solid)
    solid |= ridge
    masks["ridge"] = ridge
    # lips: the front voxel of every column in the lip rows (not the mouth row) comes forward
    mouth_y = float(lm["mouth"][1])
    band = surf & front_face(solid) & lip_band(Y, mouth_y, p.lip_half_height, vs) & (np.abs(Y - mouth_y) > vs * 0.5) \
        & (np.abs(X) <= lm["mouth_half_width"] - 0.004)
    lips = np.zeros(solid.shape, bool)
    if band.any():
        band &= Z <= Z[band].min() + 0.016
        lips = protrusion(band, solid)
        solid |= lips
    masks["lips"] = lips
    # nostrils: on the bottom-front edge of the nose (nose voxels with air below *and* in front,
    # between the mouth and the eyes), the one at |x| = 9 mm nearest to 8 mm under and 6 mm behind
    # the tip, plus its neighbour toward the centre (a 6 mm slot), carved two voxels back into the
    # nose: a notch open to the front and the underside, so it reads from the front as the nostril
    inner = np.zeros(solid.shape, bool)
    tip = lm["nose_tip"]
    nose = solid & (Y > lm["mouth"][1] + 0.006) & (Y < lm["eye_y"] - 0.012) & (np.abs(X) < 0.02) & (Z < tip[2] + 0.03)
    edge = nose & ~shift(solid, 0, 1, 0) & ~shift(solid, 0, 0, 1)
    for sx in (-1, 1):
        idx = nearest_voxel(edge, X, Y, Z, (sx * 0.009, tip[1] - 0.008, tip[2] + 0.006))
        if idx is None:
            continue
        z0, y0, x0 = (int(v) for v in idx)
        base = np.zeros(solid.shape, bool)
        base[z0, y0, x0] = True
        zs = np.nonzero(edge[:, y0, x0 - sx])[0]        # the edge voxel of the next column in (its own z)
        if zs.size:
            base[int(zs.min()), y0, x0 - sx] = True
        hole = recess(base, solid, depth=2, direction=(0, 0, 1))
        solid &= ~hole
        inner |= hole & ~base
    masks["nostril_inner"] = inner
    # ear rim: the ear's silhouette edge seen from the side (outer-face ear voxels next to a
    # column that drops two or more voxels toward the head) comes one voxel further out: the helix
    ear_z0, ear_z1 = lm["ear_z"]
    box = (np.abs(X) > lm["ear_inner_x"]) & (Y > lm["ear_y"][0]) & (Y < lm["ear_y"][1]) & (Z > ear_z0) & (Z < ear_z1)
    rim = np.zeros(solid.shape, bool)
    for sx in (-1, 1):
        side = solid & box & ((X > 0) if sx > 0 else (X < 0))
        ax = np.where(side, np.abs(X), -np.inf).max(axis=2)                  # outer |x| per (z, y), (nz, ny)
        drop = np.zeros(ax.shape, bool)
        for dz, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = np.roll(np.roll(ax, dz, axis=0), dy, axis=1)
            drop |= np.isfinite(ax) & (nb < ax - 1.5 * vs)
        edge = side & (np.abs(np.abs(X) - ax[:, :, None]) < vs * 0.5) & drop[:, :, None]
        rim |= shift(edge, sx, 0, 0) & ~solid
    solid |= rim
    masks["ear_rim"] = rim
    return solid, masks


# ---------------------------------------------------------------------------
# the builder
# ---------------------------------------------------------------------------

def build_head(voxel_size: float = 0.005, params: HeadParams | None = None, name: str | None = None,
               human: mh.MakeHuman | None = None) -> Model:
    p = params or HeadParams()
    vs = float(voxel_size)
    vs_mm = vs * 1000.0
    human = human or mh.load(morphs=p.morphs)
    lm = human.landmarks()
    if not (0.17 < lm["head_height"] < 0.40) or lm["head_width"] > 0.35:
        raise ValueError(f"morphed head out of range (broken target?): {lm['head_height']:.3f} m high, "
                         f"{lm['head_width']:.3f} m wide")
    body = human.body()
    pal = make_palette(p.flat_tints)
    alias = FLAT_ALIAS if p.flat_tints else RICH_ALIAS

    def C(name: str) -> int:
        return pal.index(alias.get(name, name))

    # ---- grid (MakeHuman frame: feet at y = 0; the part origin is shifted at the end)
    cut_y = float(lm["neck_base"][1]) - p.neck_keep
    hv = body.vertices[body.vertices[:, 1] > cut_y - 0.01]
    hair_room = p.hair_thickness + 0.004 if p.hair else 0.0
    lo = (hv[:, 0].min() - p.margin - hair_room, cut_y, hv[:, 2].min() - p.margin - hair_room)
    hi = (hv[:, 0].max() + p.margin + hair_room, lm["head_top"] + p.margin + hair_room, hv[:, 2].max() + p.margin + hair_room)
    grid = meshvox.VoxelGrid.from_box(lo, hi, vs, centre_x=True)
    mult = max(1, int(p.form_voxel_mult))
    dims = tuple(int(math.ceil(d / mult)) * mult for d in grid.dims)   # multiples of mult, x stays centred
    zc = (hv[:, 2].min() + hv[:, 2].max()) * 0.5          # grid centred on the head in z too
    grid = meshvox.VoxelGrid((-dims[0] * vs / 2, grid.origin[1], zc - dims[2] * vs / 2), vs, dims)
    X, Y, Z = grid.centres()
    model = Model(name or f"head_mh_{round(vs_mm)}mm", voxel_size=vs, palette=pal,
                  meta={"source": "makehuman hm08 base.obj (CC0)", "params": asdict(p)})
    part = Part("head", grid.dims, vs, origin=grid.origin, bone="head", symmetric=True,
                budget=400_000 if vs < 0.004 else 250_000)
    model.add_part(part)

    # ---- solid: body surface + eyeballs, filled from outside (neck cut sealed)
    body_c = meshvox.clip_faces(body, lo=(-INF, cut_y, -INF))
    eyes = human.helper("l_eye"), human.helper("r_eye")
    body_surf = meshvox.voxelize_surface(body_c, grid)       # the fine skin shell (eye masking only)
    solid, info = form_solid(grid, body_c, eyes, mult, p.clean_surface)
    model.meta["fill"] = info

    # ---- eye mask: eyeball voxels keep the mesh lids; a small forced opening guarantees
    # visibility (with relief the opening loses its top row: that row becomes the lid line)
    eye_solid = np.zeros(grid.shape, bool)
    for e in eyes:
        s, _ = meshvox.fill_interior(meshvox.voxelize_surface(e, grid))
        eye_solid |= s
    eye = eye_solid & ~body_surf & solid
    eye_centres = (lm["eye_l"], lm["eye_r"])
    if p.eye_open > 0:
        for ec in eye_centres:
            eye |= lid_opening(solid, eye_solid, X, Y, Z, ec, lm["eye_radius"] * p.eye_open_w, p.eye_open,
                               reach=3 if p.iris_voxels else 2, top_cut=vs if p.relief else 0.0)
    lid_line = lid_line_rows(eye, exposed(solid) & ~eye, X, Z, eye_centres) if p.lid_line else np.zeros(grid.shape, bool)
    relief_masks: dict[str, np.ndarray] = {}
    if p.relief:
        solid, relief_masks = relief_pass(solid, eye, lid_line, X, Y, Z, lm, p, vs)
        model.meta["relief"] = {k: int(v.sum()) for k, v in relief_masks.items()}

    # ---- tissues by depth
    depth = depth_map(solid)
    depth_mm = (depth.astype(np.float32) - 0.5) * vs_mm
    skin_vox = max(1, int(math.ceil(p.skin_mm / vs_mm - 1e-6)))
    fat_vox = max(1, int(round(p.fat_mm / vs_mm)))
    extra_vox = int(round(p.cheek_fat_extra_mm / vs_mm))
    mat = np.zeros(grid.shape, np.uint8)
    col = np.zeros(grid.shape, np.uint8)
    mat[solid] = M.MAT_MUSCLE
    col[solid] = C("muscle")
    cheek = (((np.abs(X) - (lm["eye_x"] + 0.012)) / 0.028) ** 2 + ((Y - (lm["eye_y"] - 0.04)) / 0.03) ** 2
             + ((Z - (lm["nose_tip"][2] + 0.06)) / 0.045) ** 2) <= 1.0
    fat_limit = skin_vox + fat_vox + np.where(cheek, extra_vox, 0)
    fat = solid & (depth > skin_vox) & (depth <= fat_limit)
    mat[fat] = M.MAT_FAT
    col[fat] = C("fat")
    skin = solid & (depth <= skin_vox)
    mat[skin] = M.MAT_SKIN
    col[skin] = C("skin")
    # bone: skull close under the forehead, deeper under the face; spine in the back of the neck
    ramp = np.clip((Y - (lm["brow_y"] - 0.02)) / 0.02, 0.0, 1.0)
    thr = p.bone_depth_below_brow_mm + (p.bone_depth_above_brow_mm - p.bone_depth_below_brow_mm) * ramp
    bone = solid & (depth_mm >= thr) & (depth > skin_vox + 1)
    neck = Y < lm["chin"][1] - 0.005
    bone &= ~(neck & (Z < lm["neck_base"][2]))
    mat[bone] = M.MAT_BONE
    col[bone] = C("bone")
    if p.cartilage:
        ear_z0, ear_z1 = lm["ear_z"]
        ear = (np.abs(X) > lm["ear_inner_x"]) & (Y > lm["ear_y"][0]) & (Y < lm["ear_y"][1]) & (Z > ear_z0) & (Z < ear_z1)
        nose = (np.abs(X) < 0.018) & (Y > lm["nose_base_y"] - 0.004) & (Y < lm["eye_y"] - 0.004) & (Z < lm["nose_tip"][2] + 0.03)
        cart = solid & (depth > skin_vox) & ~bone & (ear | nose)
        mat[cart] = M.MAT_CARTILAGE
        col[cart] = C("cartilage")

    # ---- vessels: temple vein and carotid, only replacing fat/muscle (they stay under the skin)
    if p.vessels:
        ex, ey, ez = lm["eye_x"], lm["eye_y"], lm["eye_z"]
        temple = capsule_mask(X, Y, Z, (ex + 0.036, ey - 0.005, ez + 0.02), (ex + 0.03, ey + 0.05, ez + 0.03), 0.0045)
        carotid = capsule_mask(X, Y, Z, (0.026, cut_y - 0.01, lm["neck_base"][2] - 0.03), (0.03, lm["chin"][1] - 0.004, lm["chin"][2] + 0.035), 0.005)
        vessel = (temple | carotid) & np.isin(mat, (M.MAT_FAT, M.MAT_MUSCLE))
        mat[vessel] = M.MAT_VESSEL
        col[vessel] = C("vessel")

    # ---- teeth: inside the jaw, never replacing skin (a closed mouth hides them fully)
    teeth = np.zeros(grid.shape, bool)
    for h in ("upper_teeth", "lower_teeth"):
        t, _ = meshvox.voxelize_solid(meshvox.close_holes(human.helper(h)), grid)
        teeth |= t
    teeth &= solid & (mat != M.MAT_SKIN)
    mat[teeth] = M.MAT_TOOTH
    tz, ty, tx = np.nonzero(teeth)
    col[tz, ty, tx] = np.where(tx % 2 == 0, C("tooth"), C("tooth_dark")).astype(np.uint8)

    # ---- eyes
    mat[eye] = M.MAT_EYE
    col[eye] = C("eye_white")
    exposed_all = exposed(mat != M.MAT_AIR)
    skin_ex = (mat == M.MAT_SKIN) & exposed_all
    for ec in (lm["eye_l"], lm["eye_r"]):
        front = eye & exposed_all & (Z < ec[2]) & (np.abs(X - ec[0]) < 0.02)   # this eye's opening
        if p.iris_voxels > 0:
            iris, hl = iris_block(front, X, Y, ec, p.iris_voxels)
        else:
            r2 = (X - ec[0]) ** 2 + (Y - ec[1]) ** 2
            iris = front & (r2 <= p.iris_radius ** 2)
            zs, ys, xs = np.nonzero(iris)
            hl = None
            if zs.size:
                k = int(np.argmax(Y[zs, ys, xs] * 2 + np.abs(X[zs, ys, xs])))  # upper, toward the temple
                hl = (zs[k], ys[k], xs[k])
        col[iris] = C("iris")
        if hl is not None:
            col[hl] = C("highlight")
    col[lid_line & skin_ex] = C("lid_line")

    # ---- skin tints (colour only) on the exposed skin, before the hair hides the scalp
    mouth_y = float(lm["mouth"][1])
    if p.eyebrows:
        band = skin_ex & (Y >= lm["eye_y"] + 0.012 - vs * 0.5) & (Y <= lm["eye_y"] + 0.018 + vs * 0.5) \
            & (np.abs(X) >= lm["eye_x"] - 0.017) & (np.abs(X) <= lm["eye_x"] + 0.021)
        if band.any():
            band &= Z <= Z[band].min() + 0.02
            rows = p.brow_rows or max(1, int(round(0.005 / vs)))
            brow = pick_front_row(band, Z, rows)
            # the head of the brow (inner third) is one voxel thicker, downward; the
            # surface recedes under the brow ridge, so look for exposed skin a little back
            inner = brow & (np.abs(X) <= lm["eye_x"] - 0.009)
            under = skin_ex & ~brow & (shift(inner, 0, -1, 0) | shift(inner, 0, -1, 1) | shift(inner, 0, -1, 2))
            brow |= pick_front_row(under, Z, 1)
            col[brow] = C("brow")
            if p.relief:
                # the brow as geometry: one voxel of hair material in front of every brow voxel
                tuft = protrusion(brow, mat != M.MAT_AIR)
                mat[tuft] = M.MAT_HAIR
                col[tuft] = C("brow")
    if p.lips:
        lips = skin_ex & lip_band(Y, mouth_y, p.lip_half_height, vs) & (np.abs(X) <= lm["mouth_half_width"] - 0.004)
        if lips.any():
            lips &= Z <= Z[lips].min() + 0.016
            col[lips] = C("lips")
            line = lips & (np.abs(Y - mouth_y) <= vs * 0.5)                  # the row between the lips
            col[line] = C("lip_line")
            if p.relief and line.any():
                # mouth corners: the outermost line voxel on each side moves one row up
                ax = np.abs(X[line])
                ends = line & (np.abs(X) >= ax.max() - 1e-6)
                col[ends] = C("lips")
                up = np.zeros(line.shape, bool)
                for dz in range(-2, 3):
                    up |= shift(ends, 0, 1, dz)
                col[pick_front_row(skin_ex & up, Z, 1)] = C("lip_line")
    if p.blush:
        # smooth radial falloff on the cheekbone (two bands) and a faint nose tip
        cx, cy, cz = lm["eye_x"] + 0.016, lm["eye_y"] - 0.028, lm["nose_tip"][2] + 0.045
        r = np.sqrt(((np.abs(X) - cx) / 0.018) ** 2 + ((Y - cy) / 0.010) ** 2 + ((Z - cz) / 0.026) ** 2)
        tipc = lm["nose_tip"]
        r_tip = np.sqrt((X - tipc[0]) ** 2 + (Y - tipc[1]) ** 2 + (Z - tipc[2] - 0.006) ** 2) / 0.009
        col[skin_ex & ((r <= 1.0) | (r_tip <= 1.0))] = C("skin_ruddy_soft")
        col[skin_ex & (r <= 0.55)] = C("skin_ruddy")
    if p.stubble:
        # beard mask: chin and jaw (a little under it near the chin), upper lip, cheeks up
        # to a line rising from the mouth corners to the sideburn, below the cheekbone;
        # never the lips or the nose. Uniform tint plus fine +-4 % noise.
        chin_y, chin_z = float(lm["chin"][1]), float(lm["chin"][2])
        ear_z0 = lm["ear_z"][0]
        sideburn = lm["ear_y"][1] - p.sideburn_below_ear_top
        lips_band = (np.abs(Y - mouth_y) <= 0.006) & (np.abs(X) <= lm["mouth_half_width"] + 0.008)
        jaw = (Y < mouth_y + 0.006) & (Y >= cut_y + 0.02)
        jaw &= (Y >= chin_y - 0.008) | ((Y >= chin_y - 0.018) & (Z <= chin_z + 0.03))
        lip_top = (Y >= mouth_y + 0.006 - vs * 0.5) & (Y <= lm["nose_base_y"] - 0.002) \
            & (np.abs(X) <= lm["mouth_half_width"] + 0.006)
        t = smoothstep(float(lm["mouth"][2]), ear_z0 - 0.005, Z)
        cheek_top = (mouth_y + 0.004) * (1.0 - t) + (sideburn - 0.006) * t
        cheek_top = np.minimum(cheek_top, lm["eye_y"] - 0.038 + 0.018 * smoothstep(-0.10, ear_z0, Z))
        cheeks = (Y >= mouth_y - 0.006) & (Y <= cheek_top) & (np.abs(X) >= lm["mouth_half_width"] + 0.008)
        zone = skin_ex & (Z <= lm["ear_centre"][2]) & (jaw | lip_top | cheeks) & ~lips_band
        zone &= ~((np.abs(X) < 0.018) & (Y > lm["nose_base_y"] - 0.004))
        if p.flat_tints:
            # a flat tint shows its edges: stop just under the chin and in front of the ear
            zone &= (Y >= chin_y - 0.012) & (Z <= ear_z0 - 0.004)
        if p.flat_tints:
            col[zone] = C("stubble")
        else:
            iz, iy, ixx = np.indices(grid.shape)
            r = hash01(np.abs(ixx - grid.dims[0] / 2 + 0.5).astype(np.int64), iy, iz, p.seed)
            shade = np.where(r < 0.3, C("stubble_light"), np.where(r < 0.6, C("stubble_dark"), C("stubble"))).astype(np.uint8)
            col[zone] = shade[zone]
    if p.nostrils and p.relief:
        col[hole_walls(relief_masks["nostril_inner"], mat == M.MAT_SKIN)] = C("nostril")
    elif p.nostrils:
        for sx in (-1, 1):
            target = (sx * 0.009, lm["nose_base_y"], lm["nose_tip"][2] + 0.012)
            idx = nearest_voxel(skin_ex, X, Y, Z, target)
            if idx is not None:
                c = (X[idx], Y[idx], Z[idx])
                near = skin_ex & ((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2 <= p.nostril_radius ** 2)
                if p.flat_tints:
                    near &= np.abs(Y - c[1]) < vs * 0.5                 # one row: a short dark dash
                col[near] = C("nostril")

    # ---- hair cap (solid), 3 shades
    if p.hair:
        hair = hair_cap(grid, mat != M.MAT_AIR, lm, p)
        hair &= Y > cut_y
        mat[hair] = M.MAT_HAIR
        if p.flat_tints:
            # painted bands instead of speckle: the top-lit steps of the crown are light (only
            # there: on every terrace it reads as a contour map), the lower edge (sideburns,
            # nape) and the under-faces are deep
            filled = mat != M.MAT_AIR
            up_lit = hair & ~shift(filled, 0, -1, 0) & (Y > lm["head_top"] - 0.022)
            floor = hair_floor(X, Z, lm, p)
            deep = hair & ((Y < floor + p.hair_taper * 0.6) | ~shift(filled, 0, 1, 0))
            col[hair] = C("hair")
            col[deep] = C("hair_deep")
            col[up_lit] = C("hair_light")
        else:
            iz, iy, ixx = np.indices(grid.shape)
            r = hash01(np.abs(ixx - grid.dims[0] / 2 + 0.5).astype(np.int64), iy, iz, p.seed + 1)
            top = smoothstep(lm["ear_y"][1], lm["head_top"] - 0.01, Y)     # lighter toward the crown
            p_light = 0.12 + 0.30 * top
            p_deep = 0.32 - 0.20 * top
            shade = np.where(r < p_light, C("hair_light"),
                             np.where(r < p_light + p_deep, C("hair_deep"), C("hair"))).astype(np.uint8)
            col[hair] = shade[hair]

    part.material[:] = mat
    part.color[:] = col

    if p.hair and p.fringe:
        from .body import grow_hair
        strip = (mat == M.MAT_HAIR) & (Z < lm["ear_z"][0] - 0.02) & (Y < hair_floor(X, Z, lm, p) + 0.012)
        grow_hair(part, strip, p.fringe_length, "hair_deep", direction=(0.0, -1.0, -0.35), jitter=0.15,
                  density=0.7, seed=p.seed, on=(M.MAT_HAIR,))

    if p.mirror:
        mirror_x(part)
    model.meta["crumbs_removed"] = keep_largest_component(part)
    model.meta["morphs"] = dict(p.morphs)

    # ---- local frame: chin at y ~ 0, centred on x/z
    part.origin = (grid.origin[0], grid.origin[1] - float(lm["chin"][1]), -grid.dims[2] * vs / 2)
    model.meta["landmarks_m"] = {k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else v)
                                 for k, v in lm.items() if k in ("eye_y", "eye_x", "hairline_y", "head_height", "head_width", "head_depth")}
    return model
