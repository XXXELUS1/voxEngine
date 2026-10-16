"""Minimal Wavefront OBJ loader with per-face group names (no external deps).

Only what the mesh importer needs: ``v``, ``vt``, ``g`` and ``f`` records. Polygons are
fan-triangulated, every triangle remembers the ``g`` group it came from, texture
coordinates are kept when present (``-1`` where a face has none). Normals in the file
are ignored; :meth:`ObjMesh.vertex_normals` recomputes area-weighted ones.

Coordinates are returned exactly as stored in the file; unit / axis conversion is the
caller's job (see :mod:`voxgen.meshvox` for the transform helpers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


@dataclass
class ObjMesh:
    """Triangle mesh with a group name per face.

    ``vertices`` (n, 3) float64; ``faces`` (m, 3) int32 vertex indices (0-based);
    ``face_group`` (m,) int32 index into ``groups``; ``uvs`` (k, 2) float64 or None;
    ``face_uvs`` (m, 3) int32 indices into ``uvs`` (-1 = none) or None.
    """

    vertices: np.ndarray
    faces: np.ndarray
    face_group: np.ndarray
    groups: list[str] = field(default_factory=list)
    uvs: np.ndarray | None = None
    face_uvs: np.ndarray | None = None

    # -- queries --------------------------------------------------------------
    def group_id(self, name: str) -> int:
        return self.groups.index(name)

    def has_group(self, name: str) -> bool:
        return name in self.groups

    def face_counts(self) -> dict[str, int]:
        """{group name: number of triangles}."""
        counts = np.bincount(self.face_group, minlength=len(self.groups))
        return {g: int(c) for g, c in zip(self.groups, counts) if c}

    def face_mask(self, groups: str | Iterable[str] | Callable[[str], bool]) -> np.ndarray:
        """bool (m,) mask of faces whose group is in ``groups`` (a name, a list of names or
        a predicate on the name, e.g. ``lambda g: g.startswith("joint-")``)."""
        if callable(groups):
            ids = [i for i, g in enumerate(self.groups) if groups(g)]
        else:
            names = [groups] if isinstance(groups, str) else list(groups)
            ids = [self.groups.index(n) for n in names if n in self.groups]
        return np.isin(self.face_group, ids)

    def select(self, groups) -> "ObjMesh":
        """Sub-mesh of the faces of ``groups`` (see :meth:`face_mask`); vertices compacted."""
        keep = self.face_mask(groups)
        faces = self.faces[keep]
        used, inv = np.unique(faces, return_inverse=True)
        out = ObjMesh(self.vertices[used].copy(), inv.reshape(faces.shape).astype(np.int32),
                      self.face_group[keep].copy(), list(self.groups))
        if self.uvs is not None and self.face_uvs is not None:
            out.uvs = self.uvs
            out.face_uvs = self.face_uvs[keep].copy()
        return out

    def group_vertices(self, groups) -> np.ndarray:
        """(k, 3) unique vertices used by the faces of ``groups``."""
        faces = self.faces[self.face_mask(groups)]
        return self.vertices[np.unique(faces)]

    def group_centre(self, groups) -> np.ndarray:
        """Centre of the bounding box of the vertices of ``groups``."""
        v = self.group_vertices(groups)
        if v.size == 0:
            raise KeyError(f"no faces in groups {groups!r}")
        return (v.min(axis=0) + v.max(axis=0)) * 0.5

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(lo, hi) of all vertices referenced by faces."""
        v = self.vertices[np.unique(self.faces)] if self.faces.size else self.vertices
        return v.min(axis=0), v.max(axis=0)

    # -- geometry -------------------------------------------------------------
    def triangles(self) -> np.ndarray:
        """(m, 3, 3) triangle corner positions."""
        return self.vertices[self.faces]

    def face_normals(self, normalise: bool = True) -> np.ndarray:
        """(m, 3) face normals (area-weighted when ``normalise=False``); winding as in the file."""
        t = self.triangles()
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        if normalise:
            ln = np.linalg.norm(n, axis=1)
            n = n / np.maximum(ln, 1e-12)[:, None]
        return n

    def vertex_normals(self) -> np.ndarray:
        """(n, 3) unit vertex normals, area-weighted average of the adjacent face normals."""
        fn = self.face_normals(normalise=False)
        vn = np.zeros_like(self.vertices)
        for k in range(3):
            np.add.at(vn, self.faces[:, k], fn)
        ln = np.linalg.norm(vn, axis=1)
        return vn / np.maximum(ln, 1e-12)[:, None]

    def transformed(self, fn: Callable[[np.ndarray], np.ndarray], flip_winding: bool = False) -> "ObjMesh":
        """New mesh with ``fn(vertices)`` as vertices. ``flip_winding`` reverses the face
        orientation (use it when ``fn`` is a mirror transform, so normals stay outward)."""
        faces = self.faces[:, ::-1].copy() if flip_winding else self.faces.copy()
        out = ObjMesh(np.asarray(fn(self.vertices), np.float64), faces, self.face_group.copy(), list(self.groups))
        if self.uvs is not None and self.face_uvs is not None:
            out.uvs = self.uvs
            out.face_uvs = self.face_uvs[:, ::-1].copy() if flip_winding else self.face_uvs.copy()
        return out

    def open_edges(self) -> np.ndarray:
        """(k, 2) edges used by exactly one triangle (empty for a watertight mesh)."""
        f = self.faces
        if f.size == 0:
            return np.zeros((0, 2), np.int32)
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        e.sort(axis=1)
        _, inv, cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
        return e[cnt[inv] == 1]

    def edge_boundary_vertices(self) -> np.ndarray:
        """Indices of vertices on open edges."""
        return np.unique(self.open_edges())

    def __repr__(self) -> str:
        return f"ObjMesh({len(self.vertices)} verts, {len(self.faces)} tris, {len(self.groups)} groups)"


def _parse_index(token: str, count: int) -> int:
    i = int(token)
    return i - 1 if i > 0 else count + i


def load_obj(path, triangulate: bool = True) -> ObjMesh:
    """Load an OBJ file. Polygons with more than three corners are fan-triangulated.
    Faces before any ``g`` record belong to the group ``"default"``."""
    path = Path(path)
    verts: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    faces: list[tuple[int, int, int]] = []
    face_uv: list[tuple[int, int, int]] = []
    face_group: list[int] = []
    groups: list[str] = []
    group_ids: dict[str, int] = {}
    cur = -1
    any_uv = False
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line or line[0] in "#\n\r":
                continue
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "v":
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif tag == "vt":
                uvs.append((float(parts[1]), float(parts[2]) if len(parts) > 2 else 0.0))
            elif tag == "g" or tag == "o":
                name = " ".join(parts[1:]) if len(parts) > 1 else "default"
                if name not in group_ids:
                    group_ids[name] = len(groups)
                    groups.append(name)
                cur = group_ids[name]
            elif tag == "f":
                if cur < 0:
                    group_ids["default"] = len(groups)
                    groups.append("default")
                    cur = group_ids["default"]
                vi: list[int] = []
                ti: list[int] = []
                nv, nt = len(verts), len(uvs)
                for tok in parts[1:]:
                    sub = tok.split("/")
                    vi.append(_parse_index(sub[0], nv))
                    if len(sub) > 1 and sub[1]:
                        ti.append(_parse_index(sub[1], nt))
                        any_uv = True
                    else:
                        ti.append(-1)
                if len(vi) < 3:
                    continue
                if triangulate:
                    for k in range(1, len(vi) - 1):
                        faces.append((vi[0], vi[k], vi[k + 1]))
                        face_uv.append((ti[0], ti[k], ti[k + 1]))
                        face_group.append(cur)
                else:
                    raise ValueError("load_obj only supports triangulate=True")
    mesh = ObjMesh(np.asarray(verts, np.float64).reshape(-1, 3),
                   np.asarray(faces, np.int32).reshape(-1, 3),
                   np.asarray(face_group, np.int32), groups)
    if any_uv:
        mesh.uvs = np.asarray(uvs, np.float64).reshape(-1, 2)
        mesh.face_uvs = np.asarray(face_uv, np.int32).reshape(-1, 3)
    return mesh
