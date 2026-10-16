"""Model and Part: the in-memory form of a .vxm file.

Array layout (the one decision everything else depends on)
----------------------------------------------------------
Every voxel field of a Part is a numpy ``uint8`` array of shape ``(nz, ny, nx)``,
indexed as ``arr[z, y, x]``. With C order the last axis (x) is the fastest, so
``arr.tobytes()`` is exactly the raw layout the .vxm format wants
(``index = x + nx*(y + ny*z)``). ``Part.dims`` is still reported as ``(nx, ny, nz)``
to match the file header and the C++ side. All helpers in this package
(sdf, body, render, checks, io) follow the ``[z, y, x]`` convention.

Units
-----
Specs think in metres (Godot local space of the part's bone: Y up, the character
faces -Z, X to the viewer's right). The voxel with index (x, y, z) has its centre at
``origin + (x+0.5, y+0.5, z+0.5) * voxel_size``. SDF fields are in *voxels*
(distance / voxel_size), so layer thicknesses are given in voxels.
"""

from __future__ import annotations

import math

import numpy as np

from . import materials as M
from .palette import Palette

Vec3 = tuple[float, float, float]


class Grid:
    """Sampling grid for SDF evaluation: metre coordinates of every voxel centre.

    A Part is itself a Grid (it has ``coords()`` and ``voxel_size``); the transforms in
    :mod:`voxgen.sdf` return new Grid objects with warped coordinates.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, voxel_size: float):
        self.X, self.Y, self.Z = X, Y, Z
        self.voxel_size = float(voxel_size)

    def coords(self):
        """Three float32 arrays (X, Y, Z) of shape (nz, ny, nx), metres."""
        return self.X, self.Y, self.Z

    @property
    def shape(self):
        return self.X.shape


class Part:
    """One dense voxel block bound to a bone.

    Parameters (all in metres except ``dims``):
      name        part name ("head", "torso", ...), unique within a model
      dims        (nx, ny, nz) number of voxels along x, y, z
      voxel_size  edge length of one voxel in metres (0.005 = 5 mm)
      origin      local-space position of the corner of voxel (0,0,0)
      bone        skeleton bone the part follows ("" for props)
      symmetric   if True, checks.symmetry_x measures mirror symmetry about local x=0
      budget      voxel budget for checks.budget (None -> checks default)

    Fields (uint8, shape (nz, ny, nx), index ``[z, y, x]``): ``material``, ``color``
    (palette index), ``state`` (damage bits, normally 0 in generated models).
    """

    def __init__(self, name: str, dims, voxel_size: float, origin: Vec3 = (0.0, 0.0, 0.0),
                 bone: str = "", symmetric: bool = False, budget: int | None = None):
        nx, ny, nz = (int(v) for v in dims)
        if min(nx, ny, nz) <= 0:
            raise ValueError(f"dims must be positive, got {dims}")
        self.name = str(name)
        self.nx, self.ny, self.nz = nx, ny, nz
        self.voxel_size = float(voxel_size)
        self.origin = tuple(float(v) for v in origin)
        self.bone = str(bone)
        self.symmetric = bool(symmetric)
        self.budget = budget
        self.allow_exposed_bone = False
        self.material = np.zeros((nz, ny, nx), np.uint8)
        self.color = np.zeros((nz, ny, nx), np.uint8)
        self.state = np.zeros((nz, ny, nx), np.uint8)
        self.model: "Model | None" = None
        self._coords = None

    # -- construction helpers ---------------------------------------------
    @classmethod
    def from_extent(cls, name: str, lo: Vec3, hi: Vec3, voxel_size: float, bone: str = "",
                    **kw) -> "Part":
        """Part covering the metre box [lo, hi] (origin = lo, dims = ceil((hi-lo)/voxel_size)).

        Example: a prop standing on the floor, centred on x/z, 0.4 m wide and 0.5 m tall::

            Part.from_extent("stool", (-0.2, 0, -0.2), (0.2, 0.5, 0.2), 0.015)
        """
        dims = tuple(max(1, int(math.ceil((h - l) / voxel_size - 1e-6))) for l, h in zip(lo, hi))
        return cls(name, dims, voxel_size, origin=tuple(lo), bone=bone, **kw)

    @classmethod
    def centered(cls, name: str, size: Vec3, voxel_size: float, bone: str = "", floor: bool = True,
                 **kw) -> "Part":
        """Part of metre size ``size`` centred on x/z. ``floor=True`` puts y=0 at the bottom,
        otherwise the part is centred on y as well."""
        sx, sy, sz = size
        lo_y = 0.0 if floor else -sy / 2
        return cls.from_extent(name, (-sx / 2, lo_y, -sz / 2), (sx / 2, lo_y + sy, sz / 2),
                               voxel_size, bone=bone, **kw)

    # -- geometry ------------------------------------------------------------
    @property
    def dims(self) -> tuple[int, int, int]:
        """(nx, ny, nz) as stored in the .vxm header."""
        return (self.nx, self.ny, self.nz)

    @property
    def shape(self) -> tuple[int, int, int]:
        """numpy shape of the fields: (nz, ny, nx)."""
        return (self.nz, self.ny, self.nx)

    @property
    def size_m(self) -> Vec3:
        """Metre extent of the block (nx*vs, ny*vs, nz*vs)."""
        return (self.nx * self.voxel_size, self.ny * self.voxel_size, self.nz * self.voxel_size)

    def m2v(self, metres: float) -> float:
        """Metres -> voxels (float). Use for SDF-op parameters that take voxels."""
        return float(metres) / self.voxel_size

    def v2m(self, voxels: float) -> float:
        return float(voxels) * self.voxel_size

    def to_voxel(self, p: Vec3) -> Vec3:
        """Metre point (local space) -> continuous voxel coordinate (corner of voxel 0 is 0)."""
        return tuple((float(p[i]) - self.origin[i]) / self.voxel_size for i in range(3))

    def to_metres(self, v: Vec3) -> Vec3:
        return tuple(self.origin[i] + float(v[i]) * self.voxel_size for i in range(3))

    def coords(self):
        """(X, Y, Z) float32 arrays of shape (nz, ny, nx): metre coordinates of voxel centres."""
        if self._coords is None:
            # computed in float64 and rounded once, so that mirrored positions about a
            # centred origin are bit-identical in float32 (exact mirror symmetry)
            vs = self.voxel_size
            x = (self.origin[0] + (np.arange(self.nx, dtype=np.float64) + 0.5) * vs).astype(np.float32)
            y = (self.origin[1] + (np.arange(self.ny, dtype=np.float64) + 0.5) * vs).astype(np.float32)
            z = (self.origin[2] + (np.arange(self.nz, dtype=np.float64) + 0.5) * vs).astype(np.float32)
            Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
            self._coords = (np.ascontiguousarray(X), np.ascontiguousarray(Y), np.ascontiguousarray(Z))
        return self._coords

    # -- voxel access ----------------------------------------------------------
    def solid(self) -> np.ndarray:
        """bool mask (nz, ny, nx) of non-air voxels."""
        return self.material != M.MAT_AIR

    def count_solid(self) -> int:
        return int(np.count_nonzero(self.material))

    def bbox(self):
        """((x0,y0,z0), (x1,y1,z1)) inclusive voxel index bounds of solids, or None if empty."""
        idx = np.nonzero(self.material)
        if idx[0].size == 0:
            return None
        z, y, x = idx
        return ((int(x.min()), int(y.min()), int(z.min())), (int(x.max()), int(y.max()), int(z.max())))

    def paint(self, region, material: int, color: int, only=None) -> int:
        """Set material+colour inside ``region`` (bool mask or SDF field, see :func:`as_mask`).

        ``only`` restricts the write to voxels whose current material is in that
        collection (e.g. ``only=[MAT_AIR]`` to add without overwriting). Returns the
        number of voxels written.
        """
        mask = as_mask(region, self.shape)
        if only is not None:
            mask = mask & np.isin(self.material, list(only))
        n = int(np.count_nonzero(mask))
        if n:
            self.material[mask] = np.uint8(material)
            self.color[mask] = np.uint8(color)
            if material == M.MAT_AIR:
                self.state[mask] = 0
        return n

    def clear(self, region) -> int:
        """Set voxels inside ``region`` to air. Returns the number of voxels removed."""
        mask = as_mask(region, self.shape) & self.solid()
        n = int(np.count_nonzero(mask))
        self.material[mask] = 0
        self.color[mask] = 0
        self.state[mask] = 0
        return n

    def material_histogram(self) -> dict[int, int]:
        """{material_id: voxel count} for non-air materials."""
        ids, counts = np.unique(self.material, return_counts=True)
        return {int(i): int(c) for i, c in zip(ids, counts) if i != 0}

    def copy(self) -> "Part":
        p = Part(self.name, self.dims, self.voxel_size, self.origin, self.bone, self.symmetric, self.budget)
        p.allow_exposed_bone = self.allow_exposed_bone
        p.model = self.model  # keeps palette lookups working; the copy is not registered in the model
        p.material = self.material.copy()
        p.color = self.color.copy()
        p.state = self.state.copy()
        return p

    def __repr__(self) -> str:
        return (f"Part({self.name!r}, dims={self.dims}, voxel_size={self.voxel_size}, "
                f"origin={tuple(round(v, 4) for v in self.origin)}, bone={self.bone!r}, "
                f"solid={self.count_solid()})")


class Model:
    """A named set of parts sharing one palette (= one .vxm file).

    ``voxel_size`` is the default voxel size for parts that do not set their own.
    """

    def __init__(self, name: str, voxel_size: float, palette: Palette | None = None, meta: dict | None = None):
        self.name = str(name)
        self.voxel_size = float(voxel_size)
        self.palette = palette if palette is not None else Palette()
        self.parts: list[Part] = []
        self.meta: dict = dict(meta or {})

    def add_part(self, part: Part) -> Part:
        if any(p.name == part.name for p in self.parts):
            raise ValueError(f"model {self.name!r} already has a part named {part.name!r}")
        part.model = self
        self.parts.append(part)
        return part

    def part(self, name: str) -> Part:
        for p in self.parts:
            if p.name == name:
                return p
        raise KeyError(f"no part {name!r} in model {self.name!r}")

    def get_part(self, name: str) -> Part | None:
        return next((p for p in self.parts if p.name == name), None)

    def used_materials(self) -> dict[int, str]:
        """{id: name} of every material present in any part (for the .vxm header)."""
        ids = set()
        for p in self.parts:
            ids.update(p.material_histogram().keys())
        return {i: M.name_of(i) for i in sorted(ids)}

    def count_solid(self) -> int:
        return sum(p.count_solid() for p in self.parts)

    def bounds_m(self):
        """Metre AABB ((x0,y0,z0),(x1,y1,z1)) of all solids across parts, or None."""
        lo = [math.inf] * 3
        hi = [-math.inf] * 3
        for p in self.parts:
            bb = p.bbox()
            if bb is None:
                continue
            for i in range(3):
                lo[i] = min(lo[i], p.origin[i] + bb[0][i] * p.voxel_size)
                hi[i] = max(hi[i], p.origin[i] + (bb[1][i] + 1) * p.voxel_size)
        if lo[0] == math.inf:
            return None
        return (tuple(lo), tuple(hi))

    def __repr__(self) -> str:
        return f"Model({self.name!r}, voxel_size={self.voxel_size}, parts={[p.name for p in self.parts]})"


def as_mask(region, shape=None) -> np.ndarray:
    """Normalise a region argument to a bool array.

    * bool array  -> as is,
    * float array (an SDF field) -> ``region <= 0`` (inside the surface),
    * None -> everything (needs ``shape``).
    """
    if region is None:
        if shape is None:
            raise ValueError("as_mask(None) needs a shape")
        return np.ones(shape, bool)
    region = np.asarray(region)
    if region.dtype == bool:
        return region
    return region <= 0.0
