"""Signed distance fields on the voxel grid of a Part.

Conventions
-----------
* Every primitive takes a *grid* first (a :class:`voxgen.model.Part` or a
  :class:`voxgen.model.Grid` returned by a transform), then geometry in **metres**
  (part local space: Y up, character faces -Z).
* Every primitive returns a ``float32`` array of shape ``(nz, ny, nx)`` holding the
  signed distance **in voxels** (negative inside). Fields of the same grid can be
  combined freely with the ops below; ``field <= 0`` is the solid mask.
* Ops (``union``, ``smooth_union`` ...) take fields, so their size parameters are in
  voxels too: write ``k=part.m2v(0.01)`` for a 1 cm blend.
* Transforms (``translate``, ``rotate``, ``scale``, ``mirror_x``) act on the *grid*
  and return a new Grid: evaluate a primitive on the transformed grid and the shape
  appears transformed. ``mirror_x`` also accepts a field.

Example::

    head = sdf.ellipsoid(part, (0, 0.12, 0), (0.075, 0.085, 0.09))
    nose = sdf.capsule(part, (0, 0.11, -0.085), (0, 0.07, -0.1), 0.012)
    skin = sdf.smooth_union(head, nose, k=part.m2v(0.01))
    ears = sdf.ellipsoid(sdf.mirror_x(part), (0.078, 0.11, 0.01), (0.008, 0.03, 0.018))
    skin = sdf.union(skin, ears)
"""

from __future__ import annotations

import math

import numpy as np

from .model import Grid

Vec3 = tuple[float, float, float]
INF = np.float32(1e9)


# ---------------------------------------------------------------------------
# grid helpers
# ---------------------------------------------------------------------------

def _xyz(grid):
    X, Y, Z = grid.coords()
    return X, Y, Z, np.float32(grid.voxel_size)


def _rel(grid, center: Vec3):
    """Coordinates relative to ``center`` (metres)."""
    X, Y, Z, vs = _xyz(grid)
    cx, cy, cz = (np.float32(v) for v in center)
    return X - cx, Y - cy, Z - cz, vs


def full(grid, value: float = INF) -> np.ndarray:
    """A field holding one constant (``INF`` = "nothing", the identity for union)."""
    return np.full(grid.shape, np.float32(value), np.float32)


def empty(grid) -> np.ndarray:
    """Alias of :func:`full` with +INF (no solid anywhere)."""
    return full(grid, INF)


# ---------------------------------------------------------------------------
# primitives (metres in, voxels out)
# ---------------------------------------------------------------------------

def sphere(grid, center: Vec3, r: float) -> np.ndarray:
    """Sphere of radius ``r`` (m) at ``center``."""
    X, Y, Z, vs = _rel(grid, center)
    return (np.sqrt(X * X + Y * Y + Z * Z) - np.float32(r)) / vs


def ellipsoid(grid, center: Vec3, radii: Vec3) -> np.ndarray:
    """Axis-aligned ellipsoid with semi-axes ``radii`` = (rx, ry, rz) in metres.

    Uses the standard bound approximation (exact on the axes, slightly
    underestimated elsewhere), good enough for layering.
    """
    X, Y, Z, vs = _rel(grid, center)
    rx, ry, rz = (np.float32(max(v, 1e-6)) for v in radii)
    k0 = np.sqrt((X / rx) ** 2 + (Y / ry) ** 2 + (Z / rz) ** 2)
    k1 = np.sqrt((X / rx ** 2) ** 2 + (Y / ry ** 2) ** 2 + (Z / rz ** 2) ** 2)
    k1 = np.maximum(k1, 1e-9)
    return (k0 * (k0 - 1.0) / k1) / vs


def capsule(grid, a: Vec3, b: Vec3, r: float) -> np.ndarray:
    """Capsule: segment a-b (metres) with radius ``r`` (round caps)."""
    X, Y, Z, vs = _rel(grid, a)
    bx, by, bz = (np.float32(b[i] - a[i]) for i in range(3))
    L2 = bx * bx + by * by + bz * bz
    if L2 < 1e-12:
        return sphere(grid, a, r)
    h = np.clip((X * bx + Y * by + Z * bz) / L2, 0.0, 1.0).astype(np.float32)
    dx, dy, dz = X - bx * h, Y - by * h, Z - bz * h
    return (np.sqrt(dx * dx + dy * dy + dz * dz) - np.float32(r)) / vs


def box(grid, center: Vec3, size: Vec3, round: float = 0.0) -> np.ndarray:
    """Box of full ``size`` = (sx, sy, sz) metres at ``center``; ``round`` (m) rounds edges."""
    X, Y, Z, vs = _rel(grid, center)
    hx, hy, hz = (np.float32(max(size[i] / 2 - round, 0.0)) for i in range(3))
    qx, qy, qz = np.abs(X) - hx, np.abs(Y) - hy, np.abs(Z) - hz
    outside = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2 + np.maximum(qz, 0) ** 2)
    inside = np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0)
    return (outside + inside - np.float32(round)) / vs


def cylinder(grid, a: Vec3, b: Vec3, r: float, round: float = 0.0) -> np.ndarray:
    """Capped cylinder from ``a`` to ``b`` (metres, flat caps) with radius ``r``."""
    X, Y, Z, vs = _rel(grid, a)
    bx, by, bz = (np.float32(b[i] - a[i]) for i in range(3))
    L = math.sqrt(float(bx * bx + by * by + bz * bz))
    if L < 1e-9:
        return sphere(grid, a, r)
    ux, uy, uz = bx / L, by / L, bz / L
    h = X * ux + Y * uy + Z * uz                       # along the axis
    px, py, pz = X - ux * h, Y - uy * h, Z - uz * h     # radial component
    rad = np.sqrt(px * px + py * py + pz * pz)
    dr = rad - np.float32(r - round)
    dh = np.abs(h - np.float32(L / 2)) - np.float32(L / 2 - round)
    outside = np.sqrt(np.maximum(dr, 0) ** 2 + np.maximum(dh, 0) ** 2)
    inside = np.minimum(np.maximum(dr, dh), 0)
    return (outside + inside - np.float32(round)) / vs


def cone(grid, a: Vec3, b: Vec3, ra: float, rb: float = 0.0) -> np.ndarray:
    """Capped cone / frustum from ``a`` (radius ``ra``) to ``b`` (radius ``rb``), metres.

    Exact SDF of the 2D trapezoid revolved around the a-b axis.
    """
    X, Y, Z, vs = _rel(grid, a)
    bx, by, bz = (np.float32(b[i] - a[i]) for i in range(3))
    L = math.sqrt(float(bx * bx + by * by + bz * bz))
    if L < 1e-9:
        return sphere(grid, a, max(ra, rb))
    ux, uy, uz = bx / L, by / L, bz / L
    h = X * ux + Y * uy + Z * uz
    px, py, pz = X - ux * h, Y - uy * h, Z - uz * h
    q = np.sqrt(px * px + py * py + pz * pz)             # radial coordinate
    # 2D problem in (q, qy) with the cone centred on the axis: qy in [-hh, hh]
    ra, rb = np.float32(ra), np.float32(rb)
    hh = np.float32(L / 2)
    qy = h - hh
    k1 = (rb, hh)
    k2 = (rb - ra, 2.0 * hh)
    cax = q - np.minimum(q, np.where(qy < 0.0, ra, rb))
    cay = np.abs(qy) - hh
    k2n = k2[0] * k2[0] + k2[1] * k2[1]
    t = np.clip(((k1[0] - q) * k2[0] + (k1[1] - qy) * k2[1]) / k2n, 0.0, 1.0)
    cbx = q - k1[0] + k2[0] * t
    cby = qy - k1[1] + k2[1] * t
    s = np.where((cbx < 0.0) & (cay < 0.0), -1.0, 1.0).astype(np.float32)
    d = s * np.sqrt(np.minimum(cax * cax + cay * cay, cbx * cbx + cby * cby))
    return d / vs


def torus(grid, center: Vec3, R: float, r: float, axis: str = "y") -> np.ndarray:
    """Torus with major radius ``R`` and tube radius ``r`` (metres); ``axis`` is the
    symmetry axis ("x", "y" or "z")."""
    X, Y, Z, vs = _rel(grid, center)
    if axis == "y":
        p, q = np.sqrt(X * X + Z * Z), Y
    elif axis == "x":
        p, q = np.sqrt(Y * Y + Z * Z), X
    elif axis == "z":
        p, q = np.sqrt(X * X + Y * Y), Z
    else:
        raise ValueError("axis must be x, y or z")
    return (np.sqrt((p - np.float32(R)) ** 2 + q * q) - np.float32(r)) / vs


def half_space(grid, point: Vec3, normal: Vec3) -> np.ndarray:
    """Everything on the side of the plane through ``point`` opposite to ``normal``
    (inside where ``(p - point) . normal <= 0``). Handy for cutting shapes."""
    X, Y, Z, vs = _rel(grid, point)
    n = np.asarray(normal, np.float32)
    n = n / max(float(np.linalg.norm(n)), 1e-9)
    return (X * n[0] + Y * n[1] + Z * n[2]) / vs


# ---------------------------------------------------------------------------
# boolean ops (fields in voxels)
# ---------------------------------------------------------------------------

def union(*fields) -> np.ndarray:
    """Union (min) of any number of fields."""
    out = fields[0]
    for f in fields[1:]:
        out = np.minimum(out, f)
    return out.astype(np.float32, copy=False)


def intersect(*fields) -> np.ndarray:
    """Intersection (max)."""
    out = fields[0]
    for f in fields[1:]:
        out = np.maximum(out, f)
    return out.astype(np.float32, copy=False)


def subtract(a, b) -> np.ndarray:
    """``a`` minus ``b`` (max(a, -b))."""
    return np.maximum(a, -b).astype(np.float32, copy=False)


def smooth_union(a, b, k: float) -> np.ndarray:
    """Polynomial smooth min with blend radius ``k`` in voxels (``part.m2v(metres)``)."""
    if k <= 0:
        return union(a, b)
    k = np.float32(k)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0).astype(np.float32)
    return (b * (1.0 - h) + a * h - k * h * (1.0 - h)).astype(np.float32)


def smooth_subtract(a, b, k: float) -> np.ndarray:
    """``a`` minus ``b`` with a smooth (rounded) seam of radius ``k`` voxels."""
    if k <= 0:
        return subtract(a, b)
    k = np.float32(k)
    h = np.clip(0.5 - 0.5 * (a + b) / k, 0.0, 1.0).astype(np.float32)
    return (a * (1.0 - h) - b * h + k * h * (1.0 - h)).astype(np.float32)


def smooth_intersect(a, b, k: float) -> np.ndarray:
    if k <= 0:
        return intersect(a, b)
    k = np.float32(k)
    h = np.clip(0.5 - 0.5 * (b - a) / k, 0.0, 1.0).astype(np.float32)
    return (b * (1.0 - h) + a * h + k * h * (1.0 - h)).astype(np.float32)


def offset(field, d_vox: float) -> np.ndarray:
    """Grow (d>0) or shrink (d<0) a shape by ``d_vox`` voxels."""
    return (field - np.float32(d_vox)).astype(np.float32)


def shell(field, thickness_vox: float) -> np.ndarray:
    """Hollow shell of the given thickness (voxels) around the surface of ``field``."""
    return (np.abs(field) - np.float32(thickness_vox)).astype(np.float32)


def displace(field, amount) -> np.ndarray:
    """Add a displacement field (voxels), e.g. from :func:`noise`, to a surface."""
    return (field + amount).astype(np.float32)


# ---------------------------------------------------------------------------
# transforms (act on the grid; evaluate primitives on the result)
# ---------------------------------------------------------------------------

def translate(grid, offset: Vec3) -> Grid:
    """Shapes evaluated on the result appear moved by ``offset`` (metres)."""
    X, Y, Z = grid.coords()
    ox, oy, oz = (np.float32(v) for v in offset)
    return Grid(X - ox, Y - oy, Z - oz, grid.voxel_size)


def rotation_matrix(euler_deg: Vec3) -> np.ndarray:
    """3x3 rotation for Euler angles in degrees, applied X then Y then Z (extrinsic,
    i.e. R = Rz @ Ry @ Rx). Right-handed, positive = counter-clockwise looking
    down the axis toward the origin."""
    ax, ay, az = (math.radians(v) for v in euler_deg)
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (Rz @ Ry @ Rx).astype(np.float32)


def rotate(grid, euler_deg: Vec3, pivot: Vec3 = (0.0, 0.0, 0.0)) -> Grid:
    """Shapes evaluated on the result appear rotated by ``euler_deg`` (degrees, X,Y,Z
    order) around ``pivot`` (metres). Example: an ear tilted back 15 degrees around Y
    is ``ellipsoid(rotate(part, (0, 15, 0), pivot=ear_center), ear_center, radii)``."""
    X, Y, Z = grid.coords()
    R = rotation_matrix(euler_deg)
    Ri = R.T  # inverse rotation applied to the sample points
    px, py, pz = (np.float32(v) for v in pivot)
    x, y, z = X - px, Y - py, Z - pz
    nx = Ri[0, 0] * x + Ri[0, 1] * y + Ri[0, 2] * z + px
    ny = Ri[1, 0] * x + Ri[1, 1] * y + Ri[1, 2] * z + py
    nz = Ri[2, 0] * x + Ri[2, 1] * y + Ri[2, 2] * z + pz
    return Grid(nx.astype(np.float32), ny.astype(np.float32), nz.astype(np.float32), grid.voxel_size)


def scale(grid, factors, pivot: Vec3 = (0.0, 0.0, 0.0)) -> Grid:
    """Shapes evaluated on the result appear scaled by ``factors`` (scalar or (sx,sy,sz))
    around ``pivot``. Distances stay roughly correct for uniform scale (the returned
    grid's voxel_size is adjusted); non-uniform scale gives an approximate SDF."""
    if np.isscalar(factors):
        factors = (float(factors),) * 3
    sx, sy, sz = (float(v) for v in factors)
    X, Y, Z = grid.coords()
    px, py, pz = (np.float32(v) for v in pivot)
    g = Grid(((X - px) / np.float32(sx) + px).astype(np.float32),
             ((Y - py) / np.float32(sy) + py).astype(np.float32),
             ((Z - pz) / np.float32(sz) + pz).astype(np.float32),
             grid.voxel_size / (sx * sy * sz) ** (1.0 / 3.0))
    return g


def mirror_x(obj, side: int = +1):
    """Mirror symmetry about local x = 0.

    * Given a grid (Part/Grid): returns a grid where ``X`` is replaced by ``+|X|``
      (``side=+1``) or ``-|X|`` (``side=-1``). Author a shape on that side and it
      appears on both sides, exactly symmetric.
    * Given a field (array): returns the union of the field and its x-flipped copy.
      The flip is about the grid's centre plane, which is local x=0 when the part is
      x-centred (``origin.x == -nx/2 * voxel_size``).
    """
    if isinstance(obj, np.ndarray):
        return np.minimum(obj, obj[:, :, ::-1]).astype(np.float32)
    X, Y, Z = obj.coords()
    ax = np.abs(X) if side >= 0 else -np.abs(X)
    return Grid(ax.astype(np.float32), Y, Z, obj.voxel_size)


# ---------------------------------------------------------------------------
# noise
# ---------------------------------------------------------------------------

def _hash01(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic lattice hash -> float32 in [0, 1)."""
    h = (ix.astype(np.uint32) * np.uint32(0x8DA6B343)) ^ (iy.astype(np.uint32) * np.uint32(0xD8163841)) \
        ^ (iz.astype(np.uint32) * np.uint32(0xCB1AB31F)) ^ np.uint32((seed * 0x9E3779B1) & 0xFFFFFFFF)
    h ^= h >> np.uint32(15)
    h *= np.uint32(0x2C1B3C6D)
    h ^= h >> np.uint32(12)
    h *= np.uint32(0x297A2D39)
    h ^= h >> np.uint32(15)
    return (h & np.uint32(0xFFFFFF)).astype(np.float32) / np.float32(0x1000000)


def value_noise(X, Y, Z, seed: int = 0) -> np.ndarray:
    """3D value noise in [-1, 1] sampled at lattice-unit coordinates (arrays)."""
    fx, fy, fz = np.floor(X), np.floor(Y), np.floor(Z)
    ix, iy, iz = fx.astype(np.int64), fy.astype(np.int64), fz.astype(np.int64)
    tx, ty, tz = X - fx, Y - fy, Z - fz
    tx = tx * tx * (3 - 2 * tx)
    ty = ty * ty * (3 - 2 * ty)
    tz = tz * tz * (3 - 2 * tz)

    def corner(dx, dy, dz):
        return _hash01(ix + dx, iy + dy, iz + dz, seed)

    c00 = corner(0, 0, 0) * (1 - tx) + corner(1, 0, 0) * tx
    c10 = corner(0, 1, 0) * (1 - tx) + corner(1, 1, 0) * tx
    c01 = corner(0, 0, 1) * (1 - tx) + corner(1, 0, 1) * tx
    c11 = corner(0, 1, 1) * (1 - tx) + corner(1, 1, 1) * tx
    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty
    v = c0 * (1 - tz) + c1 * tz
    return (v * 2.0 - 1.0).astype(np.float32)


def noise(grid, scale: float = 0.03, amplitude: float = 0.002, seed: int = 0, octaves: int = 2,
          persistence: float = 0.5) -> np.ndarray:
    """Value-noise displacement field in **voxels**.

    ``scale`` is the feature size in metres, ``amplitude`` the peak displacement in
    metres. Add it to a field: ``skin = skin + noise(part, 0.03, 0.002)`` (or
    :func:`displace`). Evaluate on ``mirror_x(part)`` for a symmetric displacement.
    """
    X, Y, Z = grid.coords()
    inv = np.float32(1.0 / max(scale, 1e-6))
    out = np.zeros(grid.shape, np.float32)
    amp, freq, total = 1.0, 1.0, 0.0
    for o in range(max(1, octaves)):
        out += amp * value_noise(X * inv * freq + 17.3 * o, Y * inv * freq + 7.1 * o,
                                 Z * inv * freq + 3.7 * o, seed + 101 * o)
        total += amp
        amp *= persistence
        freq *= 2.0
    out /= total
    return (out * np.float32(amplitude / grid.voxel_size)).astype(np.float32)


def noise01(grid, scale: float = 0.03, seed: int = 0, octaves: int = 1) -> np.ndarray:
    """Value noise remapped to [0, 1] (unitless), e.g. for density/length modulation."""
    n = noise(grid, scale=scale, amplitude=grid.voxel_size, seed=seed, octaves=octaves)
    return np.clip(n * 0.5 + 0.5, 0.0, 1.0).astype(np.float32)


def gradient(field, grid=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Central-difference gradient (gx, gy, gz) of a field, per voxel (unit-ish
    vectors pointing outward for an SDF)."""
    gz, gy, gx = np.gradient(field.astype(np.float32))
    return gx.astype(np.float32), gy.astype(np.float32), gz.astype(np.float32)
