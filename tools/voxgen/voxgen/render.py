"""Preview renderer: orthographic voxel ray-caster on numpy (the agent's eyes).

* 6 axis views + 2 isometric views (``VIEW_NAMES``), extra ``slice_x/y/z`` cross-sections.
* Shading: one warm key light + cool fill + ambient (camera-relative, so every view is
  lit the same way), lambert on the face normal, ambient occlusion from the solid
  neighbours around the visible face, a subtle depth cue, a 1 px dark outline where
  depth jumps, and the damage-state tint of docs/spec/voxbody-api.md.
* Fully vectorised: a 3D DDA marches every pixel ray at once (no per-pixel loops).
  Parts of different voxel sizes are ray-cast separately in their own grids and
  composited by depth, so multi-part models render correctly.

Entry points: :func:`render_views` (dict of PIL images), :func:`contact_sheet` (4x2 sheet
with labels and an info strip), :func:`tint_state` (shared colour rule).
"""

from __future__ import annotations

import math
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import materials as M
from .model import Model, Part
from .palette import Palette

VIEW_NAMES = ["front", "back", "left", "right", "top", "bottom", "iso", "iso2"]
SLICE_VIEWS = ["slice_x", "slice_y", "slice_z"]

BG = (52, 52, 56)
OUTLINE = (14, 14, 16)

# Camera-relative light rig (components along right u, up v, forward f).
KEY_DIR = (-0.45, 0.65, -0.60)
KEY_COLOR = (1.00, 0.92, 0.80)
KEY_INTENSITY = 1.15
FILL_DIR = (0.70, -0.30, -0.65)
FILL_COLOR = (0.70, 0.78, 1.00)
FILL_INTENSITY = 0.35
AMBIENT = (0.22, 0.23, 0.26)
AO_STRENGTH = 0.6
DEPTH_CUE = 0.12
GAMMA = 2.2


# ---------------------------------------------------------------------------
# state tint (mirror of VoxelMesher)
# ---------------------------------------------------------------------------

_BRUISE_RGB = np.array([[0, 0, 0], [217, 89, 89], [115, 64, 140], [153, 166, 77]], np.float32)
_BRUISE_W = np.array([0.0, 0.45, 0.55, 0.45], np.float32)
_BURN_RGB = np.array([[0, 0, 0], [204, 77, 51], [230, 191, 153], [26, 20, 18]], np.float32)
_BURN_W = np.array([0.0, 0.5, 0.6, 0.9], np.float32)
_BLOOD_RGB = np.array([115, 8, 8], np.float32)
_BLOOD_W = 0.65
_CUT_MUL = 0.85


def tint_state(rgb: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Apply the damage-state tint to sRGB colours (float, 0..255, shape (..., 3)).

    Order: bruise level (bits 0-1), burn level (bits 2-3), then BLOOD (0x20) on top;
    CUT (0x10) without BLOOD darkens to 85 %. WET changes nothing (shader only).
    """
    rgb = np.asarray(rgb, np.float32)
    state = np.asarray(state).astype(np.uint8)
    out = rgb.copy()
    b = (state & 0x03).astype(np.intp)
    w = _BRUISE_W[b][..., None]
    out = out * (1 - w) + _BRUISE_RGB[b] * w
    u = ((state >> 2) & 0x03).astype(np.intp)
    w = _BURN_W[u][..., None]
    out = out * (1 - w) + _BURN_RGB[u] * w
    blood = (state & 0x20) != 0
    cut = ((state & 0x10) != 0) & ~blood
    out[blood] = out[blood] * (1 - _BLOOD_W) + _BLOOD_RGB * _BLOOD_W
    out[cut] = out[cut] * _CUT_MUL
    return np.clip(out, 0, 255)


# ---------------------------------------------------------------------------
# cameras
# ---------------------------------------------------------------------------

def _unit(v):
    v = np.asarray(v, np.float64)
    return v / max(np.linalg.norm(v), 1e-12)


def view_basis(name: str):
    """(forward f, up v, right u) unit vectors in model space (Godot axes: Y up, the
    character faces -Z). "front" looks at the face, like a photograph (the character's
    +X side appears on the image's left); "top" has the face toward the top of the image.
    """
    if name == "front":
        f, v = (0, 0, 1), (0, 1, 0)
    elif name == "back":
        f, v = (0, 0, -1), (0, 1, 0)
    elif name == "left":      # camera on the character's left (-X) side
        f, v = (1, 0, 0), (0, 1, 0)
    elif name == "right":
        f, v = (-1, 0, 0), (0, 1, 0)
    elif name == "top":
        f, v = (0, -1, 0), (0, 0, -1)
    elif name == "bottom":
        f, v = (0, 1, 0), (0, 0, -1)
    elif name == "iso":       # from front-left-above
        f, v = (1, -0.75, 1), (0, 1, 0)
    elif name == "iso2":      # from back-right-above
        f, v = (-1, -0.75, -1), (0, 1, 0)
    else:
        raise ValueError(f"unknown view {name!r}; known: {VIEW_NAMES + SLICE_VIEWS}")
    f = _unit(f)
    v = np.asarray(v, np.float64)
    v = _unit(v - f * np.dot(v, f))
    u = _unit(np.cross(f, v))
    return f, v, u


# ---------------------------------------------------------------------------
# ray casting
# ---------------------------------------------------------------------------

def _empty_hits():
    return (np.zeros(0, np.int64), np.zeros(0), np.zeros((0, 3), np.int64),
            np.zeros(0, np.int64), np.zeros(0, np.int64))


def _raycast_part(part: Part, origins_m: np.ndarray, f: np.ndarray):
    """Cast all rays (origins (N,3) metres, one shared unit direction f) through one part.

    Returns ``(ray_idx, t_m, cell (K,3) as [x,y,z], normal_axis (K,), normal_sign (K,))``
    for the K rays that hit a solid voxel. Axis-aligned directions use an exact
    argmax projection; other directions a vectorised 3D DDA over the solid bounding box.
    """
    bb = part.bbox()
    if bb is None:
        return _empty_hits()
    vs = part.voxel_size
    nx, ny, nz = part.nx, part.ny, part.nz
    lo = np.asarray(bb[0], np.float64)
    hi = np.asarray(bb[1], np.float64) + 1.0
    o = (origins_m - np.asarray(part.origin)[None, :]) / vs  # voxel units
    fx = f.astype(np.float64)
    solid = part.material != M.MAT_AIR

    nonzero = np.nonzero(np.abs(fx) > 1e-9)[0]
    if nonzero.size == 1:
        return _project_axis(solid, (nx, ny, nz), lo, hi, o, fx, int(nonzero[0]), vs)

    # --- slab test against the solid bounding box -------------------------------
    N = o.shape[0]
    tmin = np.full(N, -np.inf)
    tmax = np.full(N, np.inf)
    entry_axis = np.zeros(N, np.int64)
    for a in range(3):
        if abs(fx[a]) < 1e-9:
            inside = (o[:, a] >= lo[a]) & (o[:, a] <= hi[a])
            tmax = np.where(inside, tmax, -np.inf)
            continue
        t1 = (lo[a] - o[:, a]) / fx[a]
        t2 = (hi[a] - o[:, a]) / fx[a]
        l, h = np.minimum(t1, t2), np.maximum(t1, t2)
        upd = l > tmin
        entry_axis = np.where(upd, a, entry_axis)
        tmin = np.maximum(tmin, l)
        tmax = np.minimum(tmax, h)
    rays = np.nonzero((tmax > tmin) & (tmax > 0))[0]
    if rays.size == 0:
        return _empty_hits()
    t0 = np.maximum(tmin[rays], 0.0) + 1e-6
    hits = _dda(solid, (nx, ny, nz), o[rays], fx, rays, t0, entry_axis[rays], lo, hi)
    ray_idx, t, cell, naxis, nsign = hits
    return ray_idx, t * vs, cell, naxis, nsign


def _project_axis(solid, dims, lo, hi, o, fx, a, vs):
    """Exact hit lookup for rays parallel to axis ``a`` (0=x,1=y,2=z)."""
    nx, ny, nz = dims
    s = 1 if fx[a] > 0 else -1
    others = [b for b in range(3) if b != a]
    b, c = others
    ib = np.floor(o[:, b]).astype(np.int64)
    ic = np.floor(o[:, c]).astype(np.int64)
    inb = (ib >= 0) & (ib < dims[b]) & (ic >= 0) & (ic < dims[c])
    np_axis = 2 - a
    if s > 0:
        first = np.argmax(solid, axis=np_axis)
    else:
        first = dims[a] - 1 - np.argmax(np.flip(solid, axis=np_axis), axis=np_axis)
    anyhit = solid.any(axis=np_axis)
    rays = np.nonzero(inb)[0]
    if rays.size == 0:
        return _empty_hits()
    ib, ic = ib[rays], ic[rays]
    ok = anyhit[ic, ib]
    rays, ib, ic = rays[ok], ib[ok], ic[ok]
    ia = first[ic, ib]
    cell = np.empty((rays.size, 3), np.int64)
    cell[:, a], cell[:, b], cell[:, c] = ia, ib, ic
    face = ia + (0 if s > 0 else 1)
    t = (face - o[rays, a]) / fx[a]
    return rays, t * vs, cell, np.full(rays.size, a, np.int64), np.full(rays.size, -s, np.int64)


def _dda(solid, dims, o, fx, ids, t0, entry_axis, lo, hi):
    """Vectorised 3D DDA for rays with a shared direction. ``o`` (N,3) voxel-space origins."""
    nx, ny, nz = dims
    dims_arr = np.array(dims, np.int64)
    flat_solid = solid.ravel()
    p = o + t0[:, None] * fx[None, :]
    cell = np.clip(np.floor(p).astype(np.int64), 0, dims_arr - 1)
    step = np.where(fx >= 0, 1, -1).astype(np.int64)
    with np.errstate(divide="ignore"):
        inv = np.where(np.abs(fx) > 1e-9, 1.0 / np.maximum(np.abs(fx), 1e-12), np.inf)
    tmax = np.empty((ids.size, 3))
    for a in range(3):
        if abs(fx[a]) < 1e-9:
            tmax[:, a] = np.inf
        else:
            tmax[:, a] = (cell[:, a] + (1 if step[a] > 0 else 0) - o[:, a]) / fx[a]
    out_ids, out_t, out_cell, out_axis, out_sign = [], [], [], [], []

    flat = cell[:, 0] + nx * (cell[:, 1] + ny * cell[:, 2])
    occ = flat_solid[flat]
    if occ.any():
        sel = occ
        out_ids.append(ids[sel]); out_t.append(t0[sel]); out_cell.append(cell[sel])
        ax = entry_axis[sel]
        out_axis.append(ax); out_sign.append(-step[ax])
    keep = ~occ
    cell, tmax, ids = cell[keep], tmax[keep], ids[keep]
    # bounds of the traversal = solid bbox (cells outside it are air anyway)
    blo = lo.astype(np.int64)
    bhi = (hi - 1).astype(np.int64)
    max_iter = int((bhi - blo + 1).sum()) + 3
    for _ in range(max_iter):
        n = ids.size
        if n == 0:
            break
        axis = tmax.argmin(axis=1)
        rows = np.arange(n)
        t = tmax[rows, axis]
        cell[rows, axis] += step[axis]
        tmax[rows, axis] += inv[axis]
        cx, cy, cz = cell[:, 0], cell[:, 1], cell[:, 2]
        inb = ((cx >= blo[0]) & (cx <= bhi[0]) & (cy >= blo[1]) & (cy <= bhi[1])
               & (cz >= blo[2]) & (cz <= bhi[2]))
        flat = np.where(inb, cx + nx * (cy + ny * cz), 0)
        occ = flat_solid[flat] & inb
        if occ.any():
            out_ids.append(ids[occ]); out_t.append(t[occ]); out_cell.append(cell[occ])
            ax = axis[occ]
            out_axis.append(ax); out_sign.append(-step[ax])
        keep = inb & ~occ
        if not keep.all():
            cell, tmax, ids = cell[keep], tmax[keep], ids[keep]
    if not out_ids:
        return _empty_hits()
    return (np.concatenate(out_ids), np.concatenate(out_t), np.concatenate(out_cell),
            np.concatenate(out_axis), np.concatenate(out_sign))


def _face_ao(part: Part, cell: np.ndarray, naxis: np.ndarray, nsign: np.ndarray) -> np.ndarray:
    """Occlusion 0..1 for visible faces: fraction of solid cells among the 8 cells
    surrounding the face in the outside layer (the layer the face looks into)."""
    flat_solid = (part.material != M.MAT_AIR).ravel()
    nx, ny, nz = part.nx, part.ny, part.nz
    rows = np.arange(cell.shape[0])
    base = cell.copy()
    base[rows, naxis] += nsign
    others = np.array([[1, 2], [0, 2], [0, 1]], np.int64)
    b_ax = others[naxis, 0]
    c_ax = others[naxis, 1]
    count = np.zeros(cell.shape[0], np.float32)
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            if i == 0 and j == 0:
                continue
            q = base.copy()
            q[rows, b_ax] += i
            q[rows, c_ax] += j
            inb = ((q[:, 0] >= 0) & (q[:, 0] < nx) & (q[:, 1] >= 0) & (q[:, 1] < ny)
                   & (q[:, 2] >= 0) & (q[:, 2] < nz))
            flat = np.where(inb, q[:, 0] + nx * (q[:, 1] + ny * q[:, 2]), 0)
            count += flat_solid[flat] & inb
    return count / 8.0


def _palette_array(palette) -> np.ndarray:
    if palette is None:
        return None
    if isinstance(palette, Palette):
        return palette.as_array().astype(np.float32)
    arr = np.zeros((256, 3), np.float32)
    p = np.asarray(palette, np.float32)
    arr[: p.shape[0]] = p[:, :3]
    return arr


def _default_palette_for(parts) -> np.ndarray:
    """Palette for parts without a model: colour index == material id, default colours."""
    arr = np.zeros((256, 3), np.float32)
    for mid, rgb in M.DEFAULT_COLORS.items():
        arr[mid] = rgb
    return arr


def _collect(obj, palette):
    if isinstance(obj, Model):
        parts = list(obj.parts)
        pal = _palette_array(palette if palette is not None else obj.palette)
        use_material_colors = False
    elif isinstance(obj, Part):
        parts = [obj]
        if palette is not None:
            pal, use_material_colors = _palette_array(palette), False
        elif obj.model is not None:
            pal, use_material_colors = _palette_array(obj.model.palette), False
        else:
            pal, use_material_colors = _default_palette_for(parts), True
    else:
        raise TypeError("render expects a Model or a Part")
    return parts, pal, use_material_colors


def _frame(parts, f, v, u, size, margin=0.06):
    """Camera centre, metres-per-pixel scale and ray start distance for a view."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for p in parts:
        bb = p.bbox()
        if bb is None:
            continue
        o = np.asarray(p.origin)
        lo = np.minimum(lo, o + np.asarray(bb[0]) * p.voxel_size)
        hi = np.maximum(hi, o + (np.asarray(bb[1]) + 1) * p.voxel_size)
    if not np.isfinite(lo).all():
        return None
    centre = (lo + hi) / 2
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    rel = corners - centre
    ext_u = np.abs(rel @ u).max()
    ext_v = np.abs(rel @ v).max()
    ext_f = np.abs(rel @ f).max()
    extent = max(ext_u, ext_v) * 2 * (1 + 2 * margin)
    scale = extent / size  # metres per pixel
    return centre, scale, ext_f * 2 + 1e-3


def render_view(obj, view: str, size: int = 256, palette=None, bg=BG, outline: bool = True,
                ao: bool = True, depth_cue: bool = True) -> np.ndarray:
    """Render one view to an (size, size, 3) uint8 RGB array."""
    parts, pal, mat_colors = _collect(obj, palette)
    if view in SLICE_VIEWS:
        return render_slice(obj, view[-1], size=size, palette=palette, bg=bg)
    f, v, u = view_basis(view)
    img = np.empty((size, size, 3), np.float32)
    img[:] = np.asarray(bg, np.float32)
    fr = _frame(parts, f, v, u, size)
    if fr is None:
        return img.astype(np.uint8)
    centre, scale, far = fr
    px = (np.arange(size) + 0.5 - size / 2) * scale
    py = (size / 2 - np.arange(size) - 0.5) * scale
    PX, PY = np.meshgrid(px, py)  # (H, W)
    origins = (centre[None, :] - f[None, :] * far + PX.reshape(-1, 1) * u[None, :]
               + PY.reshape(-1, 1) * v[None, :])
    N = size * size
    depth = np.full(N, np.inf)
    color = np.zeros((N, 3), np.float32)
    normal = np.zeros((N, 3), np.float32)
    occl = np.zeros(N, np.float32)
    finest = min(p.voxel_size for p in parts) if parts else 0.005
    for part in parts:
        hit, t, cell, naxis, nsign = _raycast_part(part, origins, f)
        if hit.size == 0:
            continue
        nearer = t < depth[hit]
        hit, t, cell, naxis, nsign = hit[nearer], t[nearer], cell[nearer], naxis[nearer], nsign[nearer]
        if hit.size == 0:
            continue
        depth[hit] = t
        ci = (cell[:, 2], cell[:, 1], cell[:, 0])
        if mat_colors:
            base = pal[part.material[ci].astype(np.intp)]
        else:
            base = pal[part.color[ci].astype(np.intp)]
        color[hit] = tint_state(base, part.state[ci])
        n = np.zeros((hit.size, 3), np.float32)
        n[np.arange(hit.size), naxis] = nsign
        normal[hit] = n
        occl[hit] = _face_ao(part, cell, naxis, nsign) if ao else 0.0

    hitmask = np.isfinite(depth)
    if hitmask.any():
        h = np.nonzero(hitmask)[0]
        alb = (color[h] / 255.0) ** GAMMA
        kd = _unit(KEY_DIR[0] * u + KEY_DIR[1] * v + KEY_DIR[2] * f)
        fd = _unit(FILL_DIR[0] * u + FILL_DIR[1] * v + FILL_DIR[2] * f)
        nk = np.clip(normal[h] @ kd, 0, None)[:, None]
        nf = np.clip(normal[h] @ fd, 0, None)[:, None]
        light = (np.asarray(AMBIENT, np.float32)[None, :]
                 + np.asarray(KEY_COLOR, np.float32)[None, :] * KEY_INTENSITY * nk
                 + np.asarray(FILL_COLOR, np.float32)[None, :] * FILL_INTENSITY * nf)
        shade = alb * light
        if ao:
            shade *= (1.0 - AO_STRENGTH * occl[h] ** 0.75)[:, None]
        if depth_cue:
            d = depth[h]
            dmin, dmax = d.min(), d.max()
            cue = 1.0 - DEPTH_CUE * np.clip((d - dmin) / max(dmax - dmin, 1e-9), 0, 1)
            shade *= cue[:, None]
        rgb = np.clip(shade, 0, 1) ** (1.0 / GAMMA) * 255.0
        img.reshape(-1, 3)[h] = rgb
    if outline and hitmask.any():
        D = depth.reshape(size, size)
        H = hitmask.reshape(size, size)
        flat = img.reshape(size, size, 3)
        thr = 2.5 * finest
        # background pixels next to a hit -> outline colour
        nb_hit = np.zeros_like(H)
        nb_hit[1:, :] |= H[:-1, :]
        nb_hit[:-1, :] |= H[1:, :]
        nb_hit[:, 1:] |= H[:, :-1]
        nb_hit[:, :-1] |= H[:, 1:]
        sil = nb_hit & ~H
        flat[sil] = np.asarray(OUTLINE, np.float32)
        # hit pixels that are much farther than a neighbour -> darken (internal edges)
        Dz = np.where(H, D, np.inf)
        behind = np.zeros_like(H)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nbd = np.full_like(Dz, np.inf)
            src = Dz[max(-dy, 0): size - max(dy, 0), max(-dx, 0): size - max(dx, 0)]
            nbd[max(dy, 0): size - max(-dy, 0), max(dx, 0): size - max(-dx, 0)] = src
            with np.errstate(invalid="ignore"):
                behind |= H & np.isfinite(nbd) & (Dz - nbd > thr)
        flat[behind] *= 0.45
    return np.clip(img, 0, 255).astype(np.uint8)


def render_slice(obj, axis: str = "z", index: int | None = None, size: int = 256, palette=None,
                 bg=BG) -> np.ndarray:
    """Cross-section of the first part: materials coloured by the part's palette colours
    (x-slice looks from +X, y-slice from above with the face at the top, z-slice from
    the front). ``index`` defaults to the middle of the solid bounding box."""
    parts, pal, mat_colors = _collect(obj, palette)
    part = parts[0]
    bb = part.bbox()
    img = np.empty((size, size, 3), np.float32)
    img[:] = np.asarray(bg, np.float32)
    if bb is None:
        return img.astype(np.uint8)
    col = pal[part.material.astype(np.intp)] if mat_colors else pal[part.color.astype(np.intp)]
    a = {"x": 0, "y": 1, "z": 2}[axis]
    if index is None:
        index = (bb[0][a] + bb[1][a]) // 2
    if axis == "x":
        sl = col[:, :, index]            # (nz, ny)
        m = part.material[:, :, index]
        sl, m = sl.transpose(1, 0, 2)[::-1, ::-1], m.T[::-1, ::-1]   # rows = y (top up), cols = -z (face on the right)
    elif axis == "y":
        sl = col[:, index, :]            # (nz, nx)
        m = part.material[:, index, :]   # rows = z: face (-z) at the top
    else:
        sl = col[index, :, :]            # (ny, nx)
        m = part.material[index, :, :]
        sl, m = sl[::-1, ::-1], m[::-1, ::-1]   # top up, +X on the left (like "front")
    h, w = m.shape
    sl = sl.copy()
    sl[m == 0] = np.asarray(bg, np.float32)
    # darken alternate material boundaries slightly for readability
    s = max(1, int(size // max(h, w)))
    big = np.repeat(np.repeat(sl, s, axis=0), s, axis=1)
    oy, ox = (size - big.shape[0]) // 2, (size - big.shape[1]) // 2
    if big.shape[0] > size or big.shape[1] > size:
        im = Image.fromarray(np.clip(big, 0, 255).astype(np.uint8)).resize((min(size, big.shape[1]), min(size, big.shape[0])), Image.NEAREST)
        big = np.asarray(im, np.float32)
        oy, ox = (size - big.shape[0]) // 2, (size - big.shape[1]) // 2
    img[oy: oy + big.shape[0], ox: ox + big.shape[1]] = big
    return np.clip(img, 0, 255).astype(np.uint8)


def render_views(obj, views=None, size: int = 256, palette=None, **opts) -> dict[str, Image.Image]:
    """Render several views of a Model or Part. Returns ``{view_name: PIL.Image}``.

    ``views`` defaults to the 8 standard views (``VIEW_NAMES``); ``slice_x/y/z`` give
    material cross-sections. ``palette`` overrides the model palette (a Palette or an
    (N,3) array); a bare Part without a model is coloured by material defaults.
    """
    views = list(views) if views else list(VIEW_NAMES)
    return {name: Image.fromarray(render_view(obj, name, size=size, palette=palette, **opts)) for name in views}


# ---------------------------------------------------------------------------
# contact sheet
# ---------------------------------------------------------------------------

def _font(size_px: int):
    try:
        return ImageFont.load_default(size=size_px)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def describe(obj) -> list[str]:
    """Text lines for the info strip: name, parts/dims, voxel count, voxel size, materials."""
    if isinstance(obj, Model):
        parts, name, pal_n = obj.parts, obj.name, len(obj.palette)
    else:
        parts, name, pal_n = [obj], obj.name, (len(obj.model.palette) if obj.model else 0)
    total = sum(p.count_solid() for p in parts)
    lines = []
    part_desc = "  ".join(f"{p.name}: {p.nx}x{p.ny}x{p.nz} @ {p.voxel_size * 1000:g} mm" for p in parts)
    lines.append(f"{name}   |   {part_desc}   |   solid voxels: {total:,}   |   palette: {pal_n} colours")
    hist: dict[int, int] = {}
    for p in parts:
        for mid, c in p.material_histogram().items():
            hist[mid] = hist.get(mid, 0) + c
    items = sorted(hist.items(), key=lambda kv: -kv[1])
    lines.append("materials: " + "   ".join(f"{M.name_of(mid)} {c:,}" for mid, c in items))
    return lines


def contact_sheet(obj, size: int = 256, views=None, palette=None, notes=None, **opts) -> Image.Image:
    """4x2 contact sheet with a label per view and an info strip (name, dims, voxel count,
    voxel size, material histogram, plus any extra ``notes`` lines)."""
    views = list(views) if views else list(VIEW_NAMES)
    t0 = time.perf_counter()
    imgs = render_views(obj, views, size=size, palette=palette, **opts)
    dt = time.perf_counter() - t0
    cols = 4 if len(views) > 4 else max(1, len(views))
    rows = int(math.ceil(len(views) / cols))
    margin, label_h, line_h = 8, 18, 16
    info_lines = describe(obj) + list(notes or [])
    info_lines.append(f"render: {len(views)} views x {size}px in {dt * 1000:.0f} ms")
    W = margin + cols * (size + margin)
    H = margin + rows * (size + label_h + margin) + len(info_lines) * line_h + margin
    sheet = Image.new("RGB", (W, H), (30, 30, 33))
    draw = ImageDraw.Draw(sheet)
    font = _font(13)
    for i, name in enumerate(views):
        r, c = divmod(i, cols)
        x = margin + c * (size + margin)
        y = margin + r * (size + label_h + margin)
        draw.text((x + 4, y + 2), name, fill=(220, 220, 220), font=font)
        sheet.paste(imgs[name], (x, y + label_h))
    y = margin + rows * (size + label_h + margin)
    for line in info_lines:
        draw.text((margin + 4, y), line, fill=(200, 200, 205), font=font)
        y += line_h
    return sheet
