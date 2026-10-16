"""Body-building helpers on top of SDF fields: layered tissue, hair columns, tints.

All functions take a :class:`voxgen.model.Part` and SDF fields *in voxels* produced by
:mod:`voxgen.sdf` (or bool masks). Layer thicknesses are in voxels; use
``part.m2v(metres)`` to convert (e.g. ``max(1, round(part.m2v(0.005)))`` for 5 mm).
"""

from __future__ import annotations

import numpy as np

from . import materials as M
from .model import Part, as_mask

_NEIGHBOURS6 = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def shift(arr: np.ndarray, dx: int, dy: int, dz: int, fill=0) -> np.ndarray:
    """Return ``arr`` shifted so that ``out[z, y, x] = arr[z-dz, y-dy, x-dx]``
    (out-of-range cells get ``fill``). Shifts are in voxel index units."""
    out = np.full_like(arr, fill)
    nz, ny, nx = arr.shape
    sz = slice(max(dz, 0), nz + min(dz, 0))
    sy = slice(max(dy, 0), ny + min(dy, 0))
    sx = slice(max(dx, 0), nx + min(dx, 0))
    tz = slice(max(-dz, 0), nz + min(-dz, 0))
    ty = slice(max(-dy, 0), ny + min(-dy, 0))
    tx = slice(max(-dx, 0), nx + min(-dx, 0))
    out[sz, sy, sx] = arr[tz, ty, tx]
    return out


def neighbour_count(mask: np.ndarray, connectivity: int = 6) -> np.ndarray:
    """Number of True 6- (or 26-) neighbours of every cell (outside counts as False)."""
    m = mask.astype(np.uint8)
    out = np.zeros(mask.shape, np.uint8)
    if connectivity == 6:
        offs = _NEIGHBOURS6
    else:
        offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                if (dx, dy, dz) != (0, 0, 0)]
    for dx, dy, dz in offs:
        out += shift(m, dx, dy, dz)
    return out


def exposed(mask: np.ndarray, air: np.ndarray | None = None) -> np.ndarray:
    """Cells of ``mask`` that have at least one 6-neighbour in ``air`` (default: ~mask).
    Cells on the grid border are *not* considered exposed by the border itself."""
    if air is None:
        air = ~mask
    a = air.astype(np.uint8)
    touch = np.zeros(mask.shape, np.uint8)
    for dx, dy, dz in _NEIGHBOURS6:
        touch |= shift(a, dx, dy, dz)
    return mask & (touch > 0)


def dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """6-connected binary dilation."""
    out = mask.copy()
    for _ in range(iterations):
        grown = out.copy()
        for dx, dy, dz in _NEIGHBOURS6:
            grown |= shift(out, dx, dy, dz)
        out = grown
    return out


def erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """6-connected binary erosion (the border counts as outside)."""
    out = mask.copy()
    for _ in range(iterations):
        shrunk = out.copy()
        for dx, dy, dz in _NEIGHBOURS6:
            shrunk &= shift(out, dx, dy, dz, fill=False)
        out = shrunk
    return out


def _layer_color(part: Part, material: int, color) -> int:
    """Resolve a layer colour: explicit index, a palette colour name, or the palette
    entry named like the material (``pal.add("skin", ...)`` is picked up for MAT_SKIN)."""
    if color is not None and not isinstance(color, str):
        return int(color)
    pal = part.model.palette if part.model is not None else None
    name = color if isinstance(color, str) else M.name_of(material)
    if pal is not None and pal.get(name) is not None:
        return int(pal.get(name))
    if pal is not None:
        return pal.add(name, M.DEFAULT_COLORS.get(material, (160, 160, 160)))
    raise ValueError(f"no colour for material {M.name_of(material)}: pass an explicit palette index")


def fill_layers(part: Part, surface, layers, core=None, seal: bool = True, protect_outer: bool = True,
                only_air: bool = False) -> dict[int, int]:
    """Fill a body from an outer surface SDF with materials by depth.

    ``layers`` is a list of ``(material, thickness_vox[, color])`` from the outside in;
    the last layer may have thickness ``None`` (fills everything deeper). ``color`` may
    be a palette index, a palette colour *name*, or omitted (then the palette entry
    named like the material is used, created from the default colour if missing).

    ``core`` is one ``(material, sdf[, color])`` or a list of them: regions written
    *after* the layers (bone skull, cartilage), only inside the surface. With
    ``protect_outer`` the outermost layer is never replaced by a core, so bone can't
    poke through the skin.

    ``seal``: any solid voxel that still touches air after layering gets the outer
    material (guards against SDF approximations leaving fat exposed).

    Thickness is measured as depth below the surface in voxels (``-sdf``), so with
    exact SDFs ``(skin, 1)`` is exactly the one-voxel outer shell.

    Returns ``{material: voxels written}``.
    """
    surface = np.asarray(surface, np.float32)
    inside = surface <= 0.0
    if only_air:
        inside &= part.material == M.MAT_AIR
    depth = -surface
    written: dict[int, int] = {}
    outer_mat = outer_col = None
    lo = 0.0
    for i, entry in enumerate(layers):
        mat, thick = entry[0], entry[1]
        col = _layer_color(part, mat, entry[2] if len(entry) > 2 else None)
        if i == 0:
            outer_mat, outer_col = mat, col
        if thick is None or i == len(layers) - 1 and thick is None:
            hi = np.inf
        else:
            hi = lo + float(thick)
        sel = inside & (depth >= lo) & (depth < hi)
        part.material[sel] = np.uint8(mat)
        part.color[sel] = np.uint8(col)
        written[mat] = written.get(mat, 0) + int(np.count_nonzero(sel))
        lo = hi
        if hi == np.inf:
            break
    if lo != np.inf:
        # The listed layers did not reach the centre: fill the rest with the last layer.
        mat, col = layers[-1][0], _layer_color(part, layers[-1][0], layers[-1][2] if len(layers[-1]) > 2 else None)
        sel = inside & (depth >= lo)
        part.material[sel] = np.uint8(mat)
        part.color[sel] = np.uint8(col)
        written[mat] = written.get(mat, 0) + int(np.count_nonzero(sel))

    if seal and outer_mat is not None:
        solid = part.material != M.MAT_AIR
        ex = exposed(solid) & inside & (part.material != outer_mat)
        part.material[ex] = np.uint8(outer_mat)
        part.color[ex] = np.uint8(outer_col)
        written[outer_mat] = written.get(outer_mat, 0) + int(np.count_nonzero(ex))

    if core is not None:
        cores = core if isinstance(core, list) else [core]
        protected = (part.material == outer_mat) & exposed(part.material != M.MAT_AIR) if (protect_outer and outer_mat is not None) else None
        for entry in cores:
            mat, field = entry[0], entry[1]
            col = _layer_color(part, mat, entry[2] if len(entry) > 2 else None)
            sel = as_mask(field, part.shape) & inside
            if protected is not None:
                sel &= ~protected
            part.material[sel] = np.uint8(mat)
            part.color[sel] = np.uint8(col)
            written[mat] = written.get(mat, 0) + int(np.count_nonzero(sel))
    return written


def paint(part: Part, region, material: int, color, only=None) -> int:
    """Write ``material``/``color`` inside ``region`` (SDF or mask). ``only`` limits the
    write to voxels currently made of those materials, e.g. vessels only inside
    fat/muscle: ``paint(part, vein, MAT_VESSEL, VESSEL, only=[MAT_FAT, MAT_MUSCLE])``."""
    return part.paint(region, material, _layer_color(part, material, color), only=only)


def tint_region(part: Part, region, color, materials=(M.MAT_SKIN,)) -> int:
    """Recolour solid voxels inside ``region`` (SDF or mask) with palette index ``color``
    (or a palette colour name). ``materials`` limits which materials are tinted
    (default: skin only; ``None`` = any solid). Returns the number of voxels tinted."""
    mask = as_mask(region, part.shape) & (part.material != M.MAT_AIR)
    if materials is not None:
        mask &= np.isin(part.material, list(materials))
    if isinstance(color, str):
        color = part.model.palette.index(color)
    n = int(np.count_nonzero(mask))
    part.color[mask] = np.uint8(color)
    return n


def surface_normals(solid: np.ndarray) -> np.ndarray:
    """Outward normal estimate per voxel, shape (nz, ny, nx, 3) as (x, y, z) components:
    minus the sum of offsets to solid 26-neighbours (points toward air)."""
    s = solid.astype(np.float32)
    n = np.zeros(solid.shape + (3,), np.float32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) == (0, 0, 0):
                    continue
                w = shift(s, -dx, -dy, -dz)  # value of the neighbour at +offset
                n[..., 0] -= dx * w
                n[..., 1] -= dy * w
                n[..., 2] -= dz * w
    return n


def grow_hair(part: Part, mask, length_vox, color, direction=(0.0, 1.0, 0.0), jitter: float = 0.35,
              sweep=(0.0, 0.0, 0.0), density: float = 1.0, seed: int = 0, coherence: float = 0.02,
              on=(M.MAT_SKIN,), material: int = M.MAT_HAIR) -> int:
    """Grow hair columns outward from skin voxels inside ``mask``.

    * ``mask``: SDF (inside <= 0) or bool mask selecting the scalp region. Roots are
      voxels of a material in ``on`` (default skin) inside the mask that touch air.
    * ``length_vox``: int or ``(min, max)`` column length in voxels. Lengths vary
      smoothly over the scalp (value noise of feature size ``coherence`` metres) plus a
      per-root random part, so neighbouring columns form strands.
    * ``color``: palette index or colour name for the hair voxels.
    * ``direction``: a fixed ``(x, y, z)`` vector (default straight up) or ``"normal"``
      (outward surface normal per root; the right choice for a scalp). ``sweep`` is
      added to the direction before normalising (e.g. ``(0, -0.3, 0.4)`` combs the hair
      back and down).
    * ``jitter``: random tilt magnitude (0 = perfectly straight columns).
    * ``density``: fraction of eligible roots that grow (0..1).

    Columns are 6-connected staircases (a 3D DDA), start in the air voxel adjacent to
    the root and stop at any non-air, non-hair voxel or at the grid border, so hair
    never overwrites skin (``trim_hair_capsule`` in the C++ side measures hair height
    along exactly such 6-connected chains). Returns the number of hair voxels placed.
    """
    rng = np.random.default_rng(seed)
    col = _layer_color(part, material, color)
    solid = part.material != M.MAT_AIR
    region = as_mask(mask, part.shape)
    roots_mask = region & np.isin(part.material, list(on)) & exposed(solid)
    rz, ry, rx = np.nonzero(roots_mask)
    n = rz.size
    if n == 0:
        return 0
    if density < 1.0:
        keep = rng.random(n) < density
        rz, ry, rx = rz[keep], ry[keep], rx[keep]
        n = rz.size
        if n == 0:
            return 0

    # direction per root
    if isinstance(direction, str) and direction == "normal":
        normals = surface_normals(solid)[rz, ry, rx]
    else:
        normals = np.tile(np.asarray(direction, np.float32), (n, 1))
    d = normals + np.asarray(sweep, np.float32)[None, :]
    if jitter > 0:
        d = d + rng.normal(0.0, jitter, size=d.shape).astype(np.float32)
    norm = np.linalg.norm(d, axis=1)
    bad = norm < 1e-6
    d[bad] = (0.0, 1.0, 0.0)
    norm[bad] = 1.0
    d = d / norm[:, None]

    # length per root: coherent noise + per-root random
    if np.isscalar(length_vox):
        lmin = lmax = int(length_vox)
    else:
        lmin, lmax = int(length_vox[0]), int(length_vox[1])
    if lmax > lmin:
        from .sdf import value_noise  # local import to keep module deps simple
        X, Y, Z = part.coords()
        inv = 1.0 / max(coherence, 1e-6)
        coh = value_noise(X[rz, ry, rx] * inv, Y[rz, ry, rx] * inv, Z[rz, ry, rx] * inv, seed + 7) * 0.5 + 0.5
        u = 0.65 * coh + 0.35 * rng.random(n)
        L = np.rint(lmin + (lmax - lmin) * u).astype(np.int32)
    else:
        L = np.full(n, lmin, np.int32)
    L = np.clip(L, 0, None)

    # vectorised 3D DDA from the root voxel centre
    pos = np.stack([rx, ry, rz], axis=1).astype(np.int32)
    step = np.where(d >= 0, 1, -1).astype(np.int32)
    ad = np.abs(d)
    with np.errstate(divide="ignore"):
        t_delta = np.where(ad > 1e-9, 1.0 / np.maximum(ad, 1e-9), np.inf).astype(np.float32)
    t_max = (0.5 * t_delta).astype(np.float32)
    active = np.arange(n)
    before = int(np.count_nonzero(part.material == material))
    dims = np.array([part.nx, part.ny, part.nz], np.int32)
    for k in range(int(L.max()) if n else 0):
        active = active[L[active] > k]
        if active.size == 0:
            break
        tm = t_max[active]
        axis = np.argmin(tm, axis=1)
        rows = np.arange(active.size)
        pos[active, axis] += step[active, axis]
        t_max[active, axis] += t_delta[active, axis]
        p = pos[active]
        inb = np.all((p >= 0) & (p < dims), axis=1)
        ok = np.zeros(active.size, bool)
        if inb.any():
            pi = p[inb]
            m = part.material[pi[:, 2], pi[:, 1], pi[:, 0]]
            free = (m == M.MAT_AIR) | (m == material)
            ok[inb] = free
            w = pi[free]
            part.material[w[:, 2], w[:, 1], w[:, 0]] = np.uint8(material)
            part.color[w[:, 2], w[:, 1], w[:, 0]] = np.uint8(col)
        active = active[ok]
    return int(np.count_nonzero(part.material == material)) - before


def hair_height(part: Part) -> np.ndarray:
    """Height of every hair voxel above the nearest non-hair solid (6-connected steps,
    the same measure the C++ ``trim_hair_capsule`` uses); -1 for non-hair voxels and
    for hair that is not attached at all."""
    hair = part.material == M.MAT_HAIR
    base = (part.material != M.MAT_AIR) & ~hair
    height = np.full(part.shape, -1, np.int32)
    height[base] = 0
    frontier = base
    h = 0
    while True:
        grown = dilate(frontier) & hair & (height < 0)
        if not grown.any():
            break
        h += 1
        height[grown] = h
        frontier = grown
    return height
