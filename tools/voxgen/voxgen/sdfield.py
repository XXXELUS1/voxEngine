"""Signed distance fields for binary volumes and the ``sdf`` part field (docs/spec/vxm-format.md).

The ``sdf`` field stores, per voxel, the signed distance from the voxel centre to the part's
surface: ``int8``, ``value = clamp(round(distance / voxel_size * SDF_SCALE), -127, 127)``,
negative inside the solid. It is consistent with ``material`` by construction: a solid voxel is
``sdf <= -1`` and an air voxel ``sdf >= 0``, so ``(sdf < 0) == (material != air)`` always holds
(:func:`quantize` enforces it; the zero crossing moves by at most 1/16 voxel). The range is
+-127/16 = +-7.9 voxels; farther voxels saturate, which the mesher never reads.

Two sources of the field:

* an exact SDF (``voxgen.sculpthead`` writes its ``form_field``): the mesher recovers the smooth
  surface the voxels were sampled from (``mesh.surface_nets(hermite=True)``), so a 5 mm grid shows
  no staircase;
* :func:`distance_field` for any binary volume (``voxgen sdf``, ``voxgen build --sdf``): an exact
  Euclidean distance transform (two-pass separable squared EDT, pure numpy, brute force per line:
  O(N * n) which is cheap for parts up to ~100^3), signed, surface at the voxel faces
  (+-0.5 voxel), then a 3x3x3 box blur of the signed field within 2 voxels of the surface so the
  one-voxel stairs become slopes instead of cliffs.

:func:`update_sdf` keeps a field valid after a carve or an add on the binary mask (the Python
analogue of what ``VoxelVolume`` must do in C++: carve = ``max(sdf, -d_shape)``, add =
``min(sdf, d_shape)``, then the consistency clamp).
"""
from __future__ import annotations

import numpy as np

SDF_SCALE = 16          # int8 units per voxel
SDF_MAX = 127
SDF_FIELD = "sdf"


# ----------------------------------------------------------------------------- distance transform
def _edt1d_sq(f: np.ndarray, axis: int, chunk_floats: int = 1 << 24) -> np.ndarray:
    """One separable pass of the squared Euclidean distance transform along ``axis``:
    ``out[i] = min_j f[j] + (i - j)^2`` per line, brute force on (lines, n, n) blocks."""
    g = np.ascontiguousarray(np.moveaxis(f, axis, -1), dtype=np.float32)
    n = g.shape[-1]
    idx = np.arange(n, dtype=np.float32)
    sq = (idx[:, None] - idx[None, :]) ** 2  # sq[i, j] = (i - j)^2
    flat = g.reshape(-1, n)
    out = np.empty_like(flat)
    step = max(1, chunk_floats // (n * n))
    for s in range(0, flat.shape[0], step):
        blk = flat[s:s + step]
        out[s:s + step] = (blk[:, None, :] + sq[None, :, :]).min(-1)
    return np.moveaxis(out.reshape(g.shape), -1, axis)


def edt_sq(mask: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance (voxels, centre to centre) from every voxel to the nearest True
    voxel of ``mask``; 0 on the mask, ``inf`` when the mask is empty."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return np.full(mask.shape, np.inf, np.float32)
    big = np.float32(4.0 * sum(d * d for d in mask.shape) + 4.0)
    f = np.where(mask, np.float32(0.0), big).astype(np.float32)
    for axis in range(mask.ndim):
        f = _edt1d_sq(f, axis)
    return f


def signed_distance_vox(solid: np.ndarray) -> np.ndarray:
    """Signed distance in voxels of a binary volume, negative inside: the surface sits on the
    voxel faces (a surface voxel reads -0.5, the air voxel next to it +0.5). ``inf`` magnitudes
    (no air or no solid at all) are clamped to +-``SDF_MAX / SDF_SCALE``."""
    solid = np.asarray(solid, bool)
    d_out = np.sqrt(edt_sq(solid))
    d_in = np.sqrt(edt_sq(~solid))
    d = np.where(solid, -(d_in - 0.5), d_out - 0.5).astype(np.float32)
    lim = SDF_MAX / SDF_SCALE
    return np.clip(np.nan_to_num(d, posinf=lim, neginf=-lim), -lim, lim).astype(np.float32)


def box_blur3(field: np.ndarray) -> np.ndarray:
    """3x3x3 box mean with edge replication."""
    p = np.pad(np.asarray(field, np.float32), 1, mode="edge")
    acc = np.zeros(field.shape, np.float32)
    nz, ny, nx = field.shape
    for dz in range(3):
        for dy in range(3):
            for dx in range(3):
                acc += p[dz:dz + nz, dy:dy + ny, dx:dx + nx]
    return acc / np.float32(27.0)


def distance_field(solid: np.ndarray, voxel_size: float, blur: bool = True, blur_band: float = 2.0) -> np.ndarray:
    """Signed distance in **metres** (float32, negative inside) of a binary volume, for parts
    without an exact SDF: :func:`signed_distance_vox`, then (``blur``) the 3x3x3 box blur of the
    signed field where ``|d| < blur_band`` voxels, which turns the one-voxel stairs of a sampled
    curve into slopes. The sign is not guaranteed to match ``solid`` after the blur (thin
    features): :func:`quantize` restores that."""
    d = signed_distance_vox(solid)
    if blur:
        near = np.abs(d) < np.float32(blur_band)
        d = np.where(near, box_blur3(d), d).astype(np.float32)
    return (d * np.float32(voxel_size)).astype(np.float32)


# ----------------------------------------------------------------------------- the int8 field
def quantize(d_vox: np.ndarray, solid: np.ndarray) -> np.ndarray:
    """``int8`` field from a signed distance in voxels, consistent with ``solid``: solid voxels
    are forced to <= -1 and air voxels to >= 0 (so ``(sdf < 0) == solid`` exactly), the rest is
    ``clamp(round(d * SDF_SCALE), -127, 127)``."""
    solid = np.asarray(solid, bool)
    if solid.shape != np.shape(d_vox):
        raise ValueError(f"solid {solid.shape} and field {np.shape(d_vox)} differ")
    q = np.rint(np.asarray(d_vox, np.float32) * np.float32(SDF_SCALE))
    q = np.nan_to_num(q, nan=0.0, posinf=SDF_MAX, neginf=-SDF_MAX)
    q = np.where(solid, np.minimum(q, -1.0), np.maximum(q, 0.0))
    return np.clip(q, -SDF_MAX, SDF_MAX).astype(np.int8)


def quantize_metres(d_m: np.ndarray, voxel_size: float, solid: np.ndarray) -> np.ndarray:
    """:func:`quantize` of a field in metres."""
    return quantize(np.asarray(d_m, np.float32) / np.float32(voxel_size), solid)


def dequantize(q: np.ndarray) -> np.ndarray:
    """``int8`` field -> float32 voxels."""
    return (np.asarray(q).astype(np.float32) / np.float32(SDF_SCALE)).astype(np.float32)


def is_consistent(q: np.ndarray, solid: np.ndarray) -> bool:
    """True when ``(q < 0) == solid`` everywhere."""
    return bool(np.array_equal(np.asarray(q) < 0, np.asarray(solid, bool)))


def update_sdf(q: np.ndarray, solid_before: np.ndarray, solid_after: np.ndarray) -> np.ndarray:
    """The field after the binary mask changed from ``solid_before`` to ``solid_after``: the
    removed voxels are carved (``max(sdf, -d_removed)``) and the added ones unioned
    (``min(sdf, d_added)``), ``d_*`` the unblurred signed distance of that voxel set (its faces
    become the new surface), then the consistency clamp. Returns ``q`` itself when nothing changed."""
    before, after = np.asarray(solid_before, bool), np.asarray(solid_after, bool)
    removed, added = before & ~after, after & ~before
    if not removed.any() and not added.any():
        return q
    d = dequantize(q)
    if removed.any():
        d = np.maximum(d, -signed_distance_vox(removed))
    if added.any():
        d = np.minimum(d, signed_distance_vox(added))
    return quantize(d, after)


def part_sdf_vox(part) -> np.ndarray | None:
    """The part's ``sdf`` field in voxels (float32), or None when the part has none."""
    q = getattr(part, SDF_FIELD, None)
    return None if q is None else dequantize(q)


def add_distance_field(part, force: bool = False, blur: bool = True) -> bool:
    """Give ``part`` the ``sdf`` field from its binary solid (:func:`distance_field`) unless it has
    one already (``force`` recomputes). Returns True when the field was written."""
    if part.sdf is not None and not force:
        return False
    solid = part.solid()
    part.sdf = quantize_metres(distance_field(solid, part.voxel_size, blur=blur), part.voxel_size, solid)
    return True
