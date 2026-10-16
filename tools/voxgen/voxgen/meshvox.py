"""Triangle mesh -> voxels (numpy only): the generic mesh importer.

* :func:`voxelize_surface` samples every triangle on a barycentric lattice whose spacing
  is at most ``voxel_size * spacing`` (default half a voxel) and marks the voxel of every
  sample, so a closed mesh gives a 6-connectivity-tight shell.
* :func:`fill_interior` floods the air from the grid boundary (6-connectivity) and
  returns everything the flood did not reach as solid: the shell plus its interior
  plus any enclosed cavity. ``sealed_faces`` lets a cut (an open neck) act as a wall.
  The result reports a *leak* when closing the shell by one voxel would have
  enclosed far more volume (a small hole let the flood in); :func:`voxelize_solid`
  also flags a mesh with open edges whose shell enclosed nothing.
* :class:`VoxelGrid` describes the grid (origin, voxel size, dims) and converts metres
  to indices; :func:`crop` cuts a mask to a metre box.
* Transform helpers for the usual import chores: unit scale, Y-rotation by 180 deg
  (model facing +Z -> Godot's -Z, handedness preserved), Z-flip (mirror), translation.

Arrays follow the package convention: masks are bool ``(nz, ny, nx)``, index ``[z, y, x]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .objmesh import ObjMesh

Vec3 = tuple[float, float, float]


# ---------------------------------------------------------------------------
# grid description
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VoxelGrid:
    """Axis-aligned voxel grid: ``origin`` is the metre corner of voxel (0,0,0)."""

    origin: Vec3
    voxel_size: float
    dims: tuple[int, int, int]  # (nx, ny, nz)

    @classmethod
    def from_box(cls, lo, hi, voxel_size: float, centre_x: bool = True) -> "VoxelGrid":
        """Grid covering the metre box [lo, hi]. With ``centre_x`` the x extent is made
        symmetric about x = 0 (so mirroring voxel columns is exact)."""
        lo = [float(v) for v in lo]
        hi = [float(v) for v in hi]
        if centre_x:
            half = max(abs(lo[0]), abs(hi[0]))
            lo[0], hi[0] = -half, half
        dims = [max(1, int(math.ceil((h - l) / voxel_size - 1e-6))) for l, h in zip(lo, hi)]
        if centre_x:
            lo[0] = -dims[0] * voxel_size / 2
        return cls(tuple(lo), float(voxel_size), tuple(dims))

    @property
    def shape(self) -> tuple[int, int, int]:
        nx, ny, nz = self.dims
        return (nz, ny, nx)

    def to_index(self, points: np.ndarray) -> np.ndarray:
        """Metre points (n, 3) -> integer voxel indices (n, 3) as (x, y, z) (may be out of range)."""
        p = (np.asarray(points, np.float64) - np.asarray(self.origin)) / self.voxel_size
        return np.floor(p).astype(np.int64)

    def in_range(self, idx: np.ndarray) -> np.ndarray:
        d = np.asarray(self.dims)
        return np.all((idx >= 0) & (idx < d), axis=1)

    def centres(self):
        """(X, Y, Z) float32 metre coordinates of voxel centres, shape (nz, ny, nx)."""
        nx, ny, nz = self.dims
        vs = self.voxel_size
        x = (self.origin[0] + (np.arange(nx) + 0.5) * vs).astype(np.float32)
        y = (self.origin[1] + (np.arange(ny) + 0.5) * vs).astype(np.float32)
        z = (self.origin[2] + (np.arange(nz) + 0.5) * vs).astype(np.float32)
        Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
        return X, Y, Z

    def mark(self, points: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """Set the voxels containing ``points`` (metres) in a bool mask (points outside are ignored)."""
        if out is None:
            out = np.zeros(self.shape, bool)
        idx = self.to_index(points)
        idx = idx[self.in_range(idx)]
        out[idx[:, 2], idx[:, 1], idx[:, 0]] = True
        return out


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------

def scale(vertices: np.ndarray, factor: float) -> np.ndarray:
    return np.asarray(vertices, np.float64) * float(factor)


def rotate_y180(vertices: np.ndarray) -> np.ndarray:
    """Turn the model around: (x, y, z) -> (-x, y, -z). A model that faces +Z now faces
    -Z, left stays left (no mirroring)."""
    v = np.asarray(vertices, np.float64).copy()
    v[:, 0] *= -1
    v[:, 2] *= -1
    return v


def flip_z(vertices: np.ndarray) -> np.ndarray:
    """Mirror z -> -z (changes handedness: flip the face winding too)."""
    v = np.asarray(vertices, np.float64).copy()
    v[:, 2] *= -1
    return v


def translate(vertices: np.ndarray, offset) -> np.ndarray:
    return np.asarray(vertices, np.float64) + np.asarray(offset, np.float64)


# ---------------------------------------------------------------------------
# surface sampling
# ---------------------------------------------------------------------------

def sample_triangles(tris: np.ndarray, spacing: float, max_samples: int = 20_000_000) -> np.ndarray:
    """Points on the barycentric lattice of every triangle, lattice step <= ``spacing``
    along each edge. ``tris`` is (m, 3, 3). Returns (k, 3)."""
    tris = np.asarray(tris, np.float64)
    if tris.size == 0:
        return np.zeros((0, 3))
    e = np.stack([np.linalg.norm(tris[:, 1] - tris[:, 0], axis=1),
                  np.linalg.norm(tris[:, 2] - tris[:, 1], axis=1),
                  np.linalg.norm(tris[:, 0] - tris[:, 2], axis=1)], axis=1).max(axis=1)
    n = np.maximum(1, np.ceil(e / spacing)).astype(np.int64)
    out = []
    total = 0
    for k in np.unique(n):
        sel = tris[n == k]
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        keep = (i + j) <= k
        u = (i[keep] / k).astype(np.float64)
        v = (j[keep] / k).astype(np.float64)
        w = 1.0 - u - v
        # (m, s, 3) = w*a + u*b + v*c
        pts = (w[None, :, None] * sel[:, None, 0] + u[None, :, None] * sel[:, None, 1]
               + v[None, :, None] * sel[:, None, 2])
        pts = pts.reshape(-1, 3)
        total += len(pts)
        if total > max_samples:
            raise MemoryError(f"sample_triangles: more than {max_samples} samples; raise spacing")
        out.append(pts)
    return np.concatenate(out)


def voxelize_surface(mesh_or_tris, grid: VoxelGrid, spacing: float = 0.5, out: np.ndarray | None = None) -> np.ndarray:
    """Mark every voxel touched by the triangles (an :class:`ObjMesh` or an (m,3,3) array).
    ``spacing`` is the sample step in voxels (<= 0.5 keeps the shell 6-tight).
    Triangles entirely outside the grid are skipped before sampling."""
    tris = mesh_or_tris.triangles() if isinstance(mesh_or_tris, ObjMesh) else np.asarray(mesh_or_tris, np.float64)
    if out is None:
        out = np.zeros(grid.shape, bool)
    if tris.size == 0:
        return out
    lo = np.asarray(grid.origin) - grid.voxel_size
    hi = np.asarray(grid.origin) + np.asarray(grid.dims) * grid.voxel_size + grid.voxel_size
    tmin, tmax = tris.min(axis=1), tris.max(axis=1)
    inside = np.all(tmax >= lo, axis=1) & np.all(tmin <= hi, axis=1)
    tris = tris[inside]
    if tris.size == 0:
        return out
    pts = sample_triangles(tris, spacing * grid.voxel_size)
    return grid.mark(pts, out)


def voxelize_offset_shell(mesh: ObjMesh, grid: VoxelGrid, t_from: float, t_to: float, step: float | None = None,
                          spacing: float = 0.5) -> np.ndarray:
    """Sweep the mesh along its vertex normals from offset ``t_from`` to ``t_to`` (metres,
    positive = along the normal) and mark every voxel touched: a thick shell. ``step``
    between offset copies defaults to half a voxel."""
    if step is None:
        step = 0.5 * grid.voxel_size
    vn = mesh.vertex_normals()
    out = np.zeros(grid.shape, bool)
    n = max(1, int(math.ceil(abs(t_to - t_from) / step)))
    for i in range(n + 1):
        t = t_from + (t_to - t_from) * i / n
        v = mesh.vertices + vn * t
        voxelize_surface(v[mesh.faces], grid, spacing=spacing, out=out)
    return out


# ---------------------------------------------------------------------------
# flood fill
# ---------------------------------------------------------------------------

def _sweep_axis(reach: np.ndarray, free: np.ndarray, axis: int) -> np.ndarray:
    """Propagate ``reach`` along ``axis`` through runs of ``free`` cells: every run that
    contains a reached cell becomes fully reached. Fully vectorised (segment ids)."""
    r = np.moveaxis(reach, axis, -1)
    f = np.moveaxis(free, axis, -1)
    shape = r.shape
    r2 = r.reshape(-1, shape[-1])
    f2 = f.reshape(-1, shape[-1])
    # segment id increments at every blocked cell; combine with the row id
    seg = np.cumsum(~f2, axis=1)
    rows = np.arange(r2.shape[0])[:, None]
    key = rows * (shape[-1] + 1) + seg
    hit = np.zeros(r2.shape[0] * (shape[-1] + 1), bool)
    hit[key[r2 & f2]] = True
    out2 = hit[key] & f2
    out = np.moveaxis(out2.reshape(shape), -1, axis)
    return out | reach


def flood_from_boundary(free: np.ndarray, sealed_faces=(), max_iter: int = 10_000) -> np.ndarray:
    """6-connected flood through ``free`` cells, seeded from every free cell on the grid
    boundary except the faces named in ``sealed_faces`` (any of ``"-x" "+x" "-y" "+y"
    "-z" "+z"``). Returns the reached mask. Runs axis sweeps until stable."""
    free = np.asarray(free, bool)
    reach = np.zeros_like(free)
    faces = {
        "-x": (slice(None), slice(None), 0), "+x": (slice(None), slice(None), -1),
        "-y": (slice(None), 0, slice(None)), "+y": (slice(None), -1, slice(None)),
        "-z": (0, slice(None), slice(None)), "+z": (-1, slice(None), slice(None)),
    }
    for name, sl in faces.items():
        if name not in sealed_faces:
            reach[sl] = free[sl]
    for _ in range(max_iter):
        before = int(np.count_nonzero(reach))
        for axis in (2, 1, 0):
            reach = _sweep_axis(reach, free, axis)
        if int(np.count_nonzero(reach)) == before:
            break
    return reach


def _dilate6(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[:, :, 1:] |= mask[:, :, :-1]
    out[:, :, :-1] |= mask[:, :, 1:]
    return out


def fill_interior(surface: np.ndarray, sealed_faces=(), leak_ratio: float = 1.5) -> tuple[np.ndarray, dict]:
    """Solid = everything the boundary flood through non-surface cells does not reach.

    Returns ``(solid, info)`` with ``info = {"surface", "interior", "outside", "leak",
    "closed_interior"}`` (voxel counts, and ``leak``: True when a 1-voxel dilated
    surface encloses more than ``leak_ratio`` times the interior found with the raw
    shell, i.e. the flood got in through a hole)."""
    surface = np.asarray(surface, bool)
    outside = flood_from_boundary(~surface, sealed_faces)
    solid = ~outside
    interior = int(np.count_nonzero(solid & ~surface))
    # leak probe: close the shell by one voxel and compare the enclosed volume
    thick = _dilate6(surface)
    outside2 = flood_from_boundary(~thick, sealed_faces)
    closed_interior = int(np.count_nonzero(~outside2 & ~thick))
    leak = closed_interior > leak_ratio * max(interior, 1) + 8
    info = {"surface": int(np.count_nonzero(surface)), "interior": interior,
            "outside": int(np.count_nonzero(outside)), "closed_interior": closed_interior, "leak": bool(leak)}
    return solid, info


def voxelize_solid(mesh_or_tris, grid: VoxelGrid, sealed_faces=(), spacing: float = 0.5) -> tuple[np.ndarray, dict]:
    """Shell + interior of a closed mesh in one call (see :func:`fill_interior`). For an
    :class:`ObjMesh` the open-edge count is reported too, and a mesh with open edges
    whose shell enclosed nothing counts as a leak (a hole bigger than one voxel)."""
    surf = voxelize_surface(mesh_or_tris, grid, spacing=spacing)
    solid, info = fill_interior(surf, sealed_faces)
    if isinstance(mesh_or_tris, ObjMesh):
        info["open_edges"] = int(len(mesh_or_tris.open_edges()))
        if info["open_edges"] and info["interior"] == 0:
            info["leak"] = True
    return solid, info


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------

def crop(mask: np.ndarray, grid: VoxelGrid, lo, hi) -> np.ndarray:
    """Keep only the voxels whose centre lies inside the metre box [lo, hi]."""
    X, Y, Z = grid.centres()
    lo = np.asarray(lo, np.float32)
    hi = np.asarray(hi, np.float32)
    box = (X >= lo[0]) & (X <= hi[0]) & (Y >= lo[1]) & (Y <= hi[1]) & (Z >= lo[2]) & (Z <= hi[2])
    return mask & box


def clip_faces(mesh: ObjMesh, lo=None, hi=None) -> ObjMesh:
    """Sub-mesh of the faces that have at least one corner inside the metre box
    (``lo``/``hi`` may be None or contain ±inf per axis)."""
    v = mesh.vertices
    ok = np.ones(len(v), bool)
    if lo is not None:
        ok &= np.all(v >= np.asarray(lo, np.float64), axis=1)
    if hi is not None:
        ok &= np.all(v <= np.asarray(hi, np.float64), axis=1)
    keep = ok[mesh.faces].any(axis=1)
    out = ObjMesh(mesh.vertices, mesh.faces[keep].copy(), mesh.face_group[keep].copy(), list(mesh.groups))
    if mesh.uvs is not None and mesh.face_uvs is not None:
        out.uvs, out.face_uvs = mesh.uvs, mesh.face_uvs[keep].copy()
    return out


def close_holes(mesh: ObjMesh, group: str | None = None) -> ObjMesh:
    """Close every boundary loop with a triangle fan to the loop's centroid (good enough
    to make a tooth row or a cap watertight for flood filling). New faces get ``group``
    (default: the group of the first face)."""
    f = mesh.faces
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    es = np.sort(e, axis=1)
    _, inv, cnt = np.unique(es, axis=0, return_inverse=True, return_counts=True)
    open_e = e[cnt[inv] == 1]  # keep the original orientation
    if len(open_e) == 0:
        return mesh
    # union-find the loops
    parent = np.arange(len(mesh.vertices))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in open_e:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(a) for a in open_e[:, 0]])
    verts = [mesh.vertices]
    new_faces = []
    gid = mesh.group_id(group) if group else int(mesh.face_group[0])
    nv = len(mesh.vertices)
    for r in np.unique(roots):
        loop = open_e[roots == r]
        c = mesh.vertices[np.unique(loop)].mean(axis=0)
        verts.append(c[None, :])
        # boundary edge (a, b) of triangle (.., a, b, ..) -> cap triangle (b, a, c) faces outward
        new_faces.append(np.stack([loop[:, 1], loop[:, 0], np.full(len(loop), nv)], axis=1))
        nv += 1
    faces = np.concatenate([mesh.faces] + new_faces).astype(np.int32)
    fg = np.concatenate([mesh.face_group, np.full(len(faces) - len(mesh.faces), gid, np.int32)])
    return ObjMesh(np.concatenate(verts), faces, fg, list(mesh.groups))
