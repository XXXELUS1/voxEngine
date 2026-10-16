"""Numpy meshers for .vxm parts: the geometry behind the Blender renders (``voxgen.blender``).

Pure numpy, importable without Blender. Two builders, both returning a :class:`Mesh`:

* :func:`culled_quads` - one quad per exposed voxel face, vertices welded (so a Bevel modifier
  works), per-face palette colour with a deterministic per-voxel value jitter, per-corner AO
  (classic "0fps" 3-neighbour count).
* :func:`surface_nets` - naive surface nets on the occupancy with cell-clamped Laplacian
  smoothing; faces keep the colour of the voxel that owns them (crisp eyes, lips), vertices get
  a "cavity" AO. Materials in ``keep_cubic_materials`` (teeth, hair) are excluded from the smooth
  surface and appended as culled cubes (culled against the full occupancy).

Coordinates: positions are metres in the part's local space (Godot axes, Y up, -Z forward):
``origin + corner * voxel_size`` where voxel (x, y, z) occupies ``[x, x+1] x [y, y+1] x [z, z+1]``.
Quads are counter-clockwise seen from outside (right-hand normal points out of the solid).
Colours are sRGB bytes (palette entry * jitter); AO is 1.0 = fully open, lower = occluded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import materials as M

CUBIC_MATERIALS = (M.MAT_TOOTH, M.MAT_HAIR)
_AO_LEVELS = np.array([1.0, 0.78, 0.6, 0.45], np.float32)  # 0fps corner count -> brightness


@dataclass
class Mesh:
    positions: np.ndarray       # (N, 3) float32 metres, part local space (Godot axes)
    quads: np.ndarray           # (F, 4) int32 vertex indices, CCW seen from outside
    face_colors: np.ndarray     # (F, 4) uint8: sRGB r, g, b of the owning voxel (+ jitter), mean AO * 255
    face_material: np.ndarray   # (F,) uint8 material id of the owning voxel
    corner_ao: np.ndarray       # (F, 4) float32 AO per quad corner in (0, 1]
    face_smooth: np.ndarray     # (F,) bool: True for surface-nets faces, False for cubes
    vertex_colors: np.ndarray | None = None  # (N, 4) uint8 rgb + ao per vertex (surface nets only)

    @property
    def n_faces(self) -> int:
        return int(len(self.quads))

    def is_closed(self) -> bool:
        """True when every edge is shared by exactly two quads (a watertight 2-manifold)."""
        if len(self.quads) == 0:
            return False
        q = self.quads
        e = np.stack([np.concatenate([q[:, i] for i in range(4)]),
                      np.concatenate([q[:, (i + 1) % 4] for i in range(4)])], axis=1)
        e.sort(axis=1)
        _, counts = np.unique(e, axis=0, return_counts=True)
        return bool((counts == 2).all())

    def signed_volume(self) -> float:
        """Signed volume in m^3 (positive when the quads are CCW seen from outside)."""
        q = self.quads
        tris = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
        p = self.positions.astype(np.float64)
        a, b, c = p[tris[:, 0]], p[tris[:, 1]], p[tris[:, 2]]
        return float((np.cross(a, b) * c).sum() / 6.0)


# ----------------------------------------------------------------------------- helpers
def _palette_u8(palette) -> np.ndarray:
    """(256, 3) uint8 from a Palette or a list of rgb triples."""
    colors = palette.to_list() if hasattr(palette, "to_list") else list(palette)
    pal = np.zeros((256, 3), np.uint8)
    if colors:
        arr = np.asarray(colors, np.uint8).reshape(-1, 3)
        pal[: len(arr)] = arr
    return pal


def voxel_hash01(x, y, z) -> np.ndarray:
    """Deterministic per-voxel pseudo random in [0, 1) from integer coordinates."""
    with np.errstate(over="ignore"):
        h = (np.asarray(x).astype(np.uint32) * np.uint32(73856093)) \
            ^ (np.asarray(y).astype(np.uint32) * np.uint32(19349663)) \
            ^ (np.asarray(z).astype(np.uint32) * np.uint32(83492791))
        h ^= h >> np.uint32(13)
        h *= np.uint32(0x5BD1E995)
        h ^= h >> np.uint32(15)
    return (h & np.uint32(0xFFFFFF)).astype(np.float32) / float(0xFFFFFF)


def _voxel_rgb(pal, color, vox_xyz, jitter) -> np.ndarray:
    """(n, 3) uint8 palette colour of voxels ``vox_xyz`` (n, 3) scaled by 1 +- jitter."""
    ci = color[vox_xyz[:, 2], vox_xyz[:, 1], vox_xyz[:, 0]]
    f = 1.0 + (voxel_hash01(vox_xyz[:, 0], vox_xyz[:, 1], vox_xyz[:, 2]) * 2.0 - 1.0) * jitter
    return np.clip(pal[ci].astype(np.float32) * f[:, None] + 0.5, 0, 255).astype(np.uint8)


def _pack_face_colors(rgb, corner_ao) -> np.ndarray:
    ao8 = np.clip(corner_ao.mean(axis=1) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return np.concatenate([rgb, ao8[:, None]], axis=1)


def _to_metres(part, pos_vox) -> np.ndarray:
    origin = np.asarray(part.origin, np.float32)
    return (origin[None, :] + pos_vox.astype(np.float32) * np.float32(part.voxel_size)).astype(np.float32)


def _empty(part) -> Mesh:
    return Mesh(np.zeros((0, 3), np.float32), np.zeros((0, 4), np.int32), np.zeros((0, 4), np.uint8),
                np.zeros((0,), np.uint8), np.zeros((0, 4), np.float32), np.zeros((0,), bool))


def _concat(a: Mesh, b: Mesh) -> Mesh:
    off = len(a.positions)
    vc = None
    if a.vertex_colors is not None or b.vertex_colors is not None:
        va = a.vertex_colors if a.vertex_colors is not None else np.full((len(a.positions), 4), 255, np.uint8)
        vb = b.vertex_colors if b.vertex_colors is not None else np.full((len(b.positions), 4), 255, np.uint8)
        vc = np.concatenate([va, vb])
    return Mesh(np.concatenate([a.positions, b.positions]), np.concatenate([a.quads, b.quads + off]),
                np.concatenate([a.face_colors, b.face_colors]), np.concatenate([a.face_material, b.face_material]),
                np.concatenate([a.corner_ao, b.corner_ao]), np.concatenate([a.face_smooth, b.face_smooth]), vc)


# ----------------------------------------------------------------------------- culled quads
def _culled_quads(material, color, pal, solid, emit, jitter, ao) -> tuple | None:
    """Core of :func:`culled_quads` on raw (nz, ny, nx) arrays; positions in voxel corner units."""
    nz, ny, nx = material.shape
    P = 2
    occ = np.zeros((nz + 2 * P, ny + 2 * P, nx + 2 * P), bool)
    occ[P:-P, P:-P, P:-P] = solid
    em = np.zeros_like(occ)
    em[P:-P, P:-P, P:-P] = emit & solid
    E = np.eye(3, dtype=np.int64)
    all_pos, all_ao, all_vox = [], [], []
    for a in range(3):  # 0=x 1=y 2=z
        b, c = (a + 1) % 3, (a + 2) % 3
        for s in (1, -1):
            sh = np.roll(occ, -s * E[a][::-1], axis=(0, 1, 2))  # occ[p + s*ea]
            idx = np.argwhere(em & ~sh)
            if len(idx) == 0:
                continue
            p = idx[:, ::-1]  # xyz, padded
            q = p + s * E[a]  # the air neighbour
            order = [(0, 0), (1, 0), (1, 1), (0, 1)]  # CCW seen from +a
            if s < 0:
                order = order[::-1]
            pos4, ao4 = [], []
            for ub, uc in order:
                pos = p.copy()
                pos[:, a] += 1 if s > 0 else 0
                pos[:, b] += ub
                pos[:, c] += uc
                pos4.append(pos)
                if ao:
                    db, dc = 2 * ub - 1, 2 * uc - 1
                    n1, n2, n3 = q + db * E[b], q + dc * E[c], q + db * E[b] + dc * E[c]
                    s1 = occ[n1[:, 2], n1[:, 1], n1[:, 0]]
                    s2 = occ[n2[:, 2], n2[:, 1], n2[:, 0]]
                    s3 = occ[n3[:, 2], n3[:, 1], n3[:, 0]]
                    ao4.append(np.where(s1 & s2, 3, s1.astype(np.int64) + s2 + s3))
                else:
                    ao4.append(np.zeros(len(p), np.int64))
            all_pos.append(np.stack(pos4, axis=1) - P)  # (F, 4, 3) unpadded corner coords
            all_ao.append(_AO_LEVELS[np.stack(ao4, axis=1)])
            all_vox.append(p - P)
    if not all_pos:
        return None
    pos = np.concatenate(all_pos)
    corner_ao = np.concatenate(all_ao).astype(np.float32)
    vox = np.concatenate(all_vox)
    # weld the integer corners so edges are shared
    flat = pos.reshape(-1, 3)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    quads = inv.reshape(-1, 4).astype(np.int32)
    rgb = _voxel_rgb(pal, color, vox, jitter)
    mats = material[vox[:, 2], vox[:, 1], vox[:, 0]].astype(np.uint8)
    return uniq, quads, rgb, mats, corner_ao


def culled_quads(part, palette, jitter=0.06, ao=True, only_materials=None) -> Mesh:
    """Culled cube faces of a part. ``only_materials``: emit faces only for voxels of these material
    ids (they are still culled and occluded against the whole part)."""
    material, color = part.material, part.color
    solid = material != M.MAT_AIR
    emit = solid if only_materials is None else np.isin(material, list(only_materials))
    pal = _palette_u8(palette)
    res = _culled_quads(material, color, pal, solid, emit, jitter, ao)
    if res is None:
        return _empty(part)
    uniq, quads, rgb, mats, corner_ao = res
    return Mesh(_to_metres(part, uniq), quads, _pack_face_colors(rgb, corner_ao), mats, corner_ao,
                np.zeros(len(quads), bool))


# ----------------------------------------------------------------------------- surface nets
def _surface_nets(material, color, pal, solid, smooth_passes, jitter, inset=0.02):
    """Core of :func:`surface_nets`; positions in voxel corner units. Returns None when empty."""
    nz, ny, nx = material.shape
    occ = np.zeros((nz + 2, ny + 2, nx + 2), bool)
    occ[1:-1, 1:-1, 1:-1] = solid
    corner_off = np.array([(ux, uy, uz) for uz in (0, 1) for uy in (0, 1) for ux in (0, 1)], np.int64)
    cz, cy, cx = nz + 1, ny + 1, nx + 1
    corners = np.stack([occ[uz:uz + cz, uy:uy + cy, ux:ux + cx] for (ux, uy, uz) in corner_off], axis=-1)
    nsolid = corners.sum(-1)
    active = (nsolid > 0) & (nsolid < 8)
    cell_idx = np.argwhere(active)
    V = len(cell_idx)
    if V == 0:
        return None
    vert_id = np.full((cz, cy, cx), -1, np.int64)
    vert_id[active] = np.arange(V)
    cmin = cell_idx[:, ::-1].astype(np.float32)  # xyz cell min corner (padded voxel-centre space)
    cc = corners[active]

    # vertex = mean of the crossing edge midpoints of the cell
    edges = np.array([(i, i | (1 << ax)) for i in range(8) for ax in range(3) if not i & (1 << ax)])
    mid = (corner_off[edges[:, 0]] + corner_off[edges[:, 1]]).astype(np.float32) * 0.5
    cross = cc[:, edges[:, 0]] != cc[:, edges[:, 1]]
    acc = (cross[:, :, None] * mid[None, :, :]).sum(1)
    pos = cmin + acc / np.maximum(cross.sum(1)[:, None].astype(np.float32), 1)

    # one quad per crossing edge of the voxel grid
    E = np.eye(3, dtype=np.int64)
    all_quads, all_fvox = [], []
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        sh = np.roll(occ, -E[a][::-1], axis=(0, 1, 2))
        valid = np.ones_like(occ)
        sl = [slice(None)] * 3
        sl[2 - a] = slice(-1, None)
        valid[tuple(sl)] = False
        idx = np.argwhere((occ != sh) & valid)
        if len(idx) == 0:
            continue
        p = idx[:, ::-1]
        sgn = occ[idx[:, 0], idx[:, 1], idx[:, 2]]  # p solid, p+ea air -> normal +a
        cells = []
        for db, dc in [(-1, -1), (0, -1), (0, 0), (-1, 0)]:
            q = p.copy()
            q[:, b] += db
            q[:, c] += dc
            cells.append(vert_id[q[:, 2], q[:, 1], q[:, 0]])
        quad = np.stack(cells, axis=1)
        quad = np.where(sgn[:, None], quad, quad[:, ::-1])
        all_quads.append(quad)
        all_fvox.append(np.where(sgn[:, None], p, p + E[a]) - 1)
    quads = np.concatenate(all_quads).astype(np.int32)
    fvox = np.concatenate(all_fvox)

    # Laplacian smoothing clamped to the cell
    if smooth_passes > 0:
        e0 = np.concatenate([quads[:, i] for i in range(4)])
        e1 = np.concatenate([quads[:, (i + 1) % 4] for i in range(4)])
        ea, eb = np.concatenate([e0, e1]), np.concatenate([e1, e0])
        deg = np.bincount(ea, minlength=V).astype(np.float32)[:, None]
        lo, hi = cmin + inset, cmin + 1.0 - inset
        for _ in range(smooth_passes):
            acc = np.zeros_like(pos)
            np.add.at(acc, ea, pos[eb])
            pos = np.minimum(np.maximum(acc / np.maximum(deg, 1), lo), hi)

    # per-vertex colour: nearest solid corner voxel; cavity AO from the solid corner count
    cpos = cmin[:, None, :] + corner_off[None, :, :].astype(np.float32)
    d = np.where(cc, ((cpos - pos[:, None, :]) ** 2).sum(-1), np.inf)
    vox = np.clip(cmin.astype(np.int64) + corner_off[np.argmin(d, axis=1)] - 1, 0, [nx - 1, ny - 1, nz - 1])
    vrgb = _voxel_rgb(pal, color, vox, jitter)
    vao = 1.0 - np.clip((nsolid[active].astype(np.float32) - 4.0) / 3.0, 0.0, 1.0) * 0.5
    positions = (pos - 0.5).astype(np.float32)  # padded centre space -> unpadded corner space
    return positions, quads, _voxel_rgb(pal, color, fvox, jitter), material[fvox[:, 2], fvox[:, 1], fvox[:, 0]], \
        vrgb, vao.astype(np.float32)


def surface_nets(part, palette, smooth_passes=2, keep_cubic_materials=CUBIC_MATERIALS, jitter=0.04) -> Mesh:
    """Smooth surface of a part (naive surface nets + clamped Laplacian), with the materials in
    ``keep_cubic_materials`` left out of the surface and appended as culled cubes."""
    material, color = part.material, part.color
    pal = _palette_u8(palette)
    cubic = np.isin(material, list(keep_cubic_materials)) if keep_cubic_materials else np.zeros_like(material, bool)
    solid = (material != M.MAT_AIR) & ~cubic
    res = _surface_nets(material, color, pal, solid, smooth_passes, jitter)
    if res is None:
        out = _empty(part)
    else:
        positions, quads, frgb, fmat, vrgb, vao = res
        corner_ao = vao[quads]
        out = Mesh(_to_metres(part, positions), quads, _pack_face_colors(frgb, corner_ao), fmat.astype(np.uint8),
                   corner_ao.astype(np.float32), np.ones(len(quads), bool),
                   np.concatenate([vrgb, np.clip(vao * 255 + 0.5, 0, 255).astype(np.uint8)[:, None]], axis=1))
    if cubic.any():
        cubes = culled_quads(part, palette, jitter=0.06, ao=True, only_materials=keep_cubic_materials)
        if cubes.n_faces:
            out = _concat(out, cubes)
    return out
