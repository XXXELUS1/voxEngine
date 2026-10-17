"""Numpy meshers for .vxm parts: the geometry behind the Blender renders (``voxgen.blender``).

Pure numpy, importable without Blender. Two builders, both returning a :class:`Mesh`:

* :func:`culled_quads` - one quad per exposed voxel face, vertices welded (so a Bevel modifier
  works), per-face palette colour with a deterministic per-voxel value jitter, per-corner AO
  (classic "0fps" 3-neighbour count).
* :func:`surface_nets` - naive surface nets on the occupancy with Laplacian smoothing
  (``smooth_passes``); ``clamp="cell"`` keeps every vertex inside its own cell (the surface never
  leaves the voxel skin, convex edges stay within half a voxel of the cubes), ``clamp="free"`` drops
  the clamp and, after the passes, offsets every vertex along its normal by the one global distance
  that restores the volume of the unsmoothed net (a Taubin lambda/mu pair was measured too: at the
  classic 0.5/-0.53 three passes leave the 2-voxel staircase at 0.6 voxel peak-to-peak, where the
  plain umbrella leaves 0.09, so the shrink is corrected afterwards instead); ``inflate`` then pushes
  the vertices outward along their normal by that many voxels (0.5 max is sensible: it closes the
  groove where a smooth skin part meets a cubic hair part). Faces keep the colour of
  the voxel that owns them (crisp eyes, lips), vertices get a "cavity" AO. Materials in
  ``keep_cubic_materials`` (empty by default, so hair and teeth are part of the smooth surface;
  ``CUBIC_MATERIALS`` = teeth + hair restores the cubic look) are excluded from the smooth surface
  and appended as culled cubes (culled against the full occupancy). **Hermite mode**
  (``hermite=True``, the default when the part carries the ``sdf`` field of
  :mod:`voxgen.sdfield`): every crossing edge of a cell puts its point at the linear zero crossing
  of the two corner distances, the vertex is the mean of its crossings (classic Hermite surface
  nets, the surface the voxels were sampled from is recovered to ~1/16 voxel), the normals are the
  trilinearly sampled Sobel gradient of the field (``Mesh.vertex_normals``), the vertex AO is read
  from the field along the normal (the corner-count AO of the binary path is grid noise on an exact
  surface), and no Laplacian pass runs (``hermite_passes`` adds light cell-clamped ones). A corner pair whose occupancy and field
  disagree (a cubic material cut out of the surface) crosses at the edge midpoint.
* :func:`step_edge_weights` - per-edge chamfer weights for the "facet" look (cubes whose one-voxel
  steps become exact 45 degree slopes while big flat faces stay crisp). Rule, to be ported to C++
  as is: for every mesh edge shared by exactly two quads with different normals (a convex or
  concave 90 degree edge) take each quad's *extent* away from the edge: the number of consecutive
  voxels, starting at the quad's own voxel inclusive, whose face with the same normal is exposed
  (the flat face continues). The edge is a *step* when either extent == 1 (the tread or the riser
  next to it is one voxel deep). Then group the step edges into *runs*: consecutive unit edges on
  the same straight line, between faces of the same two normals; a step edge is kept only when its
  run is at least ``min_run`` voxels long (default 2), so a lone one-voxel bump or pit stays a cube
  instead of becoming a diamond "pimple" (``min_run=1`` = every step). Kept edges get weight 1,
  all others 0; a Bevel of 0.5 voxel, 1 segment, limited by weight, makes the slopes
  (``voxgen.blender.render_styles`` style ``facet``; ``facet_all`` weights every 90 degree edge).

Coordinates: positions are metres in the part's local space (Godot axes, Y up, -Z forward):
``origin + corner * voxel_size`` where voxel (x, y, z) occupies ``[x, x+1] x [y, y+1] x [z, z+1]``.
Quads are counter-clockwise seen from outside (right-hand normal points out of the solid).
Colours are sRGB bytes (palette entry * jitter); AO is 1.0 = fully open, lower = occluded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import materials as M
from .sdfield import SDF_SCALE

CUBIC_MATERIALS = (M.MAT_TOOTH, M.MAT_HAIR)
SMOOTH_CLAMPS = ("cell", "free")
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
    vertex_normals: np.ndarray | None = None  # (N, 3) float32 unit normals (Hermite nets: sdf gradient)
    hermite: bool = False       # True when the smooth faces came from the sdf field (Hermite nets)

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
    vn = None
    if a.vertex_normals is not None or b.vertex_normals is not None:
        na = a.vertex_normals if a.vertex_normals is not None else _vertex_normals(a.positions, a.quads)
        nb = b.vertex_normals if b.vertex_normals is not None else _vertex_normals(b.positions, b.quads)
        vn = np.concatenate([na, nb]).astype(np.float32)
    return Mesh(np.concatenate([a.positions, b.positions]), np.concatenate([a.quads, b.quads + off]),
                np.concatenate([a.face_colors, b.face_colors]), np.concatenate([a.face_material, b.face_material]),
                np.concatenate([a.corner_ao, b.corner_ao]), np.concatenate([a.face_smooth, b.face_smooth]), vc, vn,
                a.hermite or b.hermite)


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
def _surface_nets(material, color, pal, solid, smooth_passes, jitter, inset=0.02, clamp="cell", inflate=0.0,
                  sdf_vox=None, hermite_passes=0):
    """Core of :func:`surface_nets`; positions in voxel corner units. Returns None when empty.
    ``sdf_vox``: (nz, ny, nx) float32 signed distance in voxels (negative inside) -> Hermite mode."""
    nz, ny, nx = material.shape
    occ = np.zeros((nz + 2, ny + 2, nx + 2), bool)
    occ[1:-1, 1:-1, 1:-1] = solid
    hermite = sdf_vox is not None
    if hermite:
        # pad with |edge value|: an air pad next to a solid boundary voxel crosses at the part's face
        spad = np.pad(np.asarray(sdf_vox, np.float32), 1, mode="edge")
        border = np.pad(np.zeros((nz, ny, nx), bool), 1, constant_values=True)
        spad[border] = np.abs(spad[border])
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

    # vertex = mean of the crossing points of the cell's edges: the edge midpoints (binary), or
    # the linear zero crossings of the corner distances (Hermite)
    edges = np.array([(i, i | (1 << ax)) for i in range(8) for ax in range(3) if not i & (1 << ax)])
    cross = cc[:, edges[:, 0]] != cc[:, edges[:, 1]]
    c0, c1 = corner_off[edges[:, 0]].astype(np.float32), corner_off[edges[:, 1]].astype(np.float32)
    if hermite:
        sc = np.stack([spad[uz:uz + cz, uy:uy + cy, ux:ux + cx] for (ux, uy, uz) in corner_off], axis=-1)[active]
        s0, s1 = sc[:, edges[:, 0]], sc[:, edges[:, 1]]
        den = s0 - s1
        ok = ((s0 < 0) != (s1 < 0)) & (np.abs(den) > 1e-6)
        t = np.where(ok, s0 / np.where(ok, den, 1.0), 0.5)
        t = np.clip(t, 0.0, 1.0).astype(np.float32)
        pts = c0[None, :, :] + t[:, :, None] * (c1 - c0)[None, :, :]          # (V, 12, 3)
    else:
        pts = np.broadcast_to(((c0 + c1) * 0.5)[None, :, :], (V, len(edges), 3))
    acc = (cross[:, :, None] * pts).sum(1)
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

    # Laplacian (umbrella) smoothing: "cell" clamps each vertex to its cell, "free" restores the volume
    # (Hermite: only the optional light cell-clamped passes, the positions are already exact)
    if clamp not in SMOOTH_CLAMPS:
        raise ValueError(f"clamp must be one of {SMOOTH_CLAMPS}, not {clamp!r}")
    if hermite:
        smooth_passes, clamp = int(hermite_passes), "cell"
    if smooth_passes > 0:
        e0 = np.concatenate([quads[:, i] for i in range(4)])
        e1 = np.concatenate([quads[:, (i + 1) % 4] for i in range(4)])
        ea, eb = np.concatenate([e0, e1]), np.concatenate([e1, e0])
        deg = np.bincount(ea, minlength=V).astype(np.float32)[:, None]
        lo, hi = cmin + inset, cmin + 1.0 - inset
        v0 = _signed_volume(pos, quads)
        for _ in range(smooth_passes):
            acc = np.zeros_like(pos)
            np.add.at(acc, ea, pos[eb])
            pos = acc / np.maximum(deg, 1)
            if clamp == "cell":
                pos = np.minimum(np.maximum(pos, lo), hi)
        if clamp == "free":
            for _ in range(2):  # first-order volume restore along the normals, twice for accuracy
                area = 0.5 * np.linalg.norm(np.cross(pos[quads[:, 2]] - pos[quads[:, 0]],
                                                     pos[quads[:, 3]] - pos[quads[:, 1]]), axis=1).sum()
                t = (v0 - _signed_volume(pos, quads)) / max(float(area), 1e-12)
                pos = pos + np.float32(t) * _vertex_normals(pos, quads)
    if inflate:
        pos = pos + np.float32(inflate) * _vertex_normals(pos, quads)

    # per-vertex colour: nearest solid corner voxel; cavity AO from the solid corner count
    cpos = cmin[:, None, :] + corner_off[None, :, :].astype(np.float32)
    d = np.where(cc, ((cpos - pos[:, None, :]) ** 2).sum(-1), np.inf)
    vox = np.clip(cmin.astype(np.int64) + corner_off[np.argmin(d, axis=1)] - 1, 0, [nx - 1, ny - 1, nz - 1])
    vrgb = _voxel_rgb(pal, color, vox, jitter)
    if hermite:
        # the corner-count AO alternates 1.0 / 0.5 along a tilted smooth surface (grid grain on the
        # exact geometry), so Hermite nets take their AO from the field itself
        normals = _sdf_normals(spad, cell_idx, corner_off, pos - cmin, pos, quads)
        vao = _sdf_ao(spad, pos, normals)
    else:
        normals = None
        vao = 1.0 - np.clip((nsolid[active].astype(np.float32) - 4.0) / 3.0, 0.0, 1.0) * 0.5
    positions = (pos - 0.5).astype(np.float32)  # padded centre space -> unpadded corner space
    return positions, quads, _voxel_rgb(pal, color, fvox, jitter), material[fvox[:, 2], fvox[:, 1], fvox[:, 0]], \
        vrgb, vao.astype(np.float32), normals


def _sobel_gradient(f: np.ndarray) -> np.ndarray:
    """(3, nz, ny, nx) gradient (x, y, z components) of a (nz, ny, nx) field: central difference
    along the axis, [1 2 1]/4 smoothing along the other two (edges replicated), so the 1/16 voxel
    quantization of the sdf field does not grain the normals."""
    p = np.pad(np.asarray(f, np.float32), 1, mode="edge")

    def diff(a, ax):  # central difference along ax (shrinks a by 2 along ax)
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[ax], hi[ax] = slice(0, -2), slice(2, None)
        return 0.5 * (a[tuple(hi)] - a[tuple(lo)])

    def smooth(a, ax):  # [1 2 1] / 4 along ax (shrinks a by 2 along ax)
        lo, mid, hi = [slice(None)] * 3, [slice(None)] * 3, [slice(None)] * 3
        lo[ax], mid[ax], hi[ax] = slice(0, -2), slice(1, -1), slice(2, None)
        return 0.25 * a[tuple(lo)] + 0.5 * a[tuple(mid)] + 0.25 * a[tuple(hi)]

    out = np.empty((3,) + tuple(f.shape), np.float32)
    for comp, ax in ((0, 2), (1, 1), (2, 0)):  # x -> numpy axis 2, y -> 1, z -> 0
        d = diff(p, ax)
        for a in range(3):
            if a != ax:
                d = smooth(d, a)
        out[comp] = d
    return out


def _trilinear(field: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Trilinear sample of a (nz, ny, nx) field at (N, 3) xyz points in its index space (the value
    of voxel (x, y, z) sits at (x, y, z)); points are clamped to the grid."""
    nz, ny, nx = field.shape
    q = np.clip(pts.astype(np.float32), 0.0, np.array([nx - 1.001, ny - 1.001, nz - 1.001], np.float32))
    i0 = np.floor(q).astype(np.int64)
    fr = q - i0
    out = np.zeros(len(q), np.float32)
    for dx, dy, dz in ((ux, uy, uz) for uz in (0, 1) for uy in (0, 1) for ux in (0, 1)):
        w = (fr[:, 0] if dx else 1 - fr[:, 0]) * (fr[:, 1] if dy else 1 - fr[:, 1]) * (fr[:, 2] if dz else 1 - fr[:, 2])
        out += w * field[i0[:, 2] + dz, i0[:, 1] + dy, i0[:, 0] + dx]
    return out


def _sdf_ao(spad, pos, normals, steps=(1.0, 2.0, 3.0), weights=(0.5, 0.3, 0.2), floor=0.5) -> np.ndarray:
    """(V,) ambient occlusion in [floor, 1] from the field: the distance sampled ``h`` voxels along
    the normal falls short of ``h`` where other surface is near (a crease, a pocket), open surface
    reads 1. Smooth, grid-free."""
    occ = np.zeros(len(pos), np.float32)
    for h, w in zip(steps, weights):
        d = _trilinear(spad, pos + np.float32(h) * normals)
        occ += np.float32(w) * np.clip((np.float32(h) - d) / np.float32(h), 0.0, 1.0)
    return np.clip(1.0 - occ, floor, 1.0).astype(np.float32)


def _sdf_normals(spad, cell_idx, corner_off, frac, pos, quads) -> np.ndarray:
    """(V, 3) unit normals of the Hermite vertices: the Sobel gradient of the padded field at the
    8 cell corners, trilinearly weighted by the vertex's position in its cell; where the gradient
    vanishes (a saturated or flat field) the mesh normal is used instead."""
    g = _sobel_gradient(spad)                                           # (3, cz+1, cy+1, cx+1)
    V = len(cell_idx)
    w = np.ones((V, 8), np.float32)
    fr = np.clip(frac, 0.0, 1.0).astype(np.float32)
    for i, (ux, uy, uz) in enumerate(corner_off):
        for ax, u in ((0, ux), (1, uy), (2, uz)):
            w[:, i] *= fr[:, ax] if u else (1.0 - fr[:, ax])
    acc = np.zeros((V, 3), np.float32)
    for i, (ux, uy, uz) in enumerate(corner_off):
        gi = g[:, cell_idx[:, 0] + uz, cell_idx[:, 1] + uy, cell_idx[:, 2] + ux].T  # (V, 3)
        acc += w[:, i:i + 1] * gi
    n = np.linalg.norm(acc, axis=1, keepdims=True)
    mesh_n = _vertex_normals(pos, quads)
    return np.where(n > 1e-4, acc / np.maximum(n, 1e-12), mesh_n).astype(np.float32)


def _signed_volume(pos, quads) -> float:
    q = quads
    tris = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    p = pos.astype(np.float64)
    return float((np.cross(p[tris[:, 0]], p[tris[:, 1]]) * p[tris[:, 2]]).sum() / 6.0)


def _vertex_normals(pos, quads) -> np.ndarray:
    """(N, 3) unit normals: area-weighted sum of the (outward, CCW) quad normals around each vertex."""
    p = pos[quads]
    n = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])
    vn = np.zeros_like(pos)
    for i in range(4):
        np.add.at(vn, quads[:, i], n)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def surface_nets(part, palette, smooth_passes=2, keep_cubic_materials=(), jitter=0.04, clamp="cell",
                 inflate=0.0, hermite=None, hermite_passes=0) -> Mesh:
    """Smooth surface of a part (naive surface nets + Laplacian smoothing, see the module docstring
    for ``clamp`` and ``inflate``, the latter in voxels), with the materials in ``keep_cubic_materials``
    (none by default) left out of the surface and appended as culled cubes. ``hermite``: None =
    Hermite nets when the part has the ``sdf`` field, else the binary path; True requires the field;
    False forces the binary path. ``hermite_passes``: light cell-clamped Laplacian passes in Hermite
    mode (0 = the exact crossings)."""
    material, color = part.material, part.color
    pal = _palette_u8(palette)
    cubic = np.isin(material, list(keep_cubic_materials)) if keep_cubic_materials else np.zeros_like(material, bool)
    solid = (material != M.MAT_AIR) & ~cubic
    sdf_q = getattr(part, "sdf", None)
    if hermite is None:
        hermite = sdf_q is not None
    elif hermite and sdf_q is None:
        raise ValueError(f"part {part.name!r} has no 'sdf' field: hermite surface nets need one (voxgen sdf)")
    sdf_vox = (sdf_q.astype(np.float32) / np.float32(SDF_SCALE)) if hermite else None
    res = _surface_nets(material, color, pal, solid, smooth_passes, jitter, clamp=clamp, inflate=inflate,
                        sdf_vox=sdf_vox, hermite_passes=hermite_passes)
    if res is None:
        out = _empty(part)
    else:
        positions, quads, frgb, fmat, vrgb, vao, normals = res
        corner_ao = vao[quads]
        out = Mesh(_to_metres(part, positions), quads, _pack_face_colors(frgb, corner_ao), fmat.astype(np.uint8),
                   corner_ao.astype(np.float32), np.ones(len(quads), bool),
                   np.concatenate([vrgb, np.clip(vao * 255 + 0.5, 0, 255).astype(np.uint8)[:, None]], axis=1),
                   normals, bool(hermite))
    if cubic.any():
        cubes = culled_quads(part, palette, jitter=0.06, ao=True, only_materials=keep_cubic_materials)
        if cubes.n_faces:
            out = _concat(out, cubes)
    return out


# ----------------------------------------------------------------------------- step edges (facet)
@dataclass
class EdgeWeights:
    edges: np.ndarray    # (E, 2) int32 vertex ids (min, max) of every edge of the mesh, sorted
    weights: np.ndarray  # (E,) float32: 1 = chamfer this edge, 0 = keep it sharp
    stats: dict          # edges, edges_90, edges_step, edges_nonmanifold

    def keys(self, n_vertices: int) -> np.ndarray:
        """(E,) int64 ``min * n_vertices + max``, sorted ascending (for ``np.searchsorted``)."""
        return self.edges[:, 0].astype(np.int64) * n_vertices + self.edges[:, 1]


def _run_lengths(E: np.ndarray, axis: int, forward: bool) -> np.ndarray:
    """Length of the run of consecutive True values that ends at each index when scanning along
    ``axis`` in the + direction (forward) or the - direction; 0 where E is False."""
    if not forward:
        E = np.flip(E, axis)
    n = E.shape[axis]
    shape = [1] * E.ndim
    shape[axis] = n
    idx = np.arange(n).reshape(shape)
    last_false = np.maximum.accumulate(np.where(E, -1, idx), axis=axis)
    run = np.where(E, idx - last_false, 0)
    return np.flip(run, axis) if not forward else run


def _face_extents(solid: np.ndarray) -> dict:
    """{(normal axis a, sign s, in-plane axis k, edge side sk): (nz, ny, nx) int} - at every voxel the
    extent of its exposed face with normal s*e_a, measured from the edge on the sk*e_k side of the
    face, away from that edge (the voxel itself counts; 0 where the face is not exposed)."""
    occ = np.pad(solid, 1)
    out = {}
    for a in range(3):  # Godot axis 0=x 1=y 2=z -> numpy axis 2-a
        for s in (1, -1):
            nb = np.roll(occ, -s, axis=2 - a)  # occ[v + s*e_a]
            exposed = (occ & ~nb)[1:-1, 1:-1, 1:-1]
            for k in range(3):
                if k == a:
                    continue
                for sk in (1, -1):
                    # edge on the +k side: the face continues toward -k = run ending at v scanning +k
                    out[(a, s, k, sk)] = _run_lengths(exposed, 2 - k, forward=(sk > 0))
    return out


def step_edge_weights(m: Mesh, part, mode: str = "steps", min_run: int = 2) -> EdgeWeights:
    """Chamfer weights for the edges of a :func:`culled_quads` mesh of ``part`` (see the module
    docstring for the rule). ``mode``: ``"steps"`` (the rule, with ``min_run``) or ``"all"`` (every
    90 degree edge)."""
    N = len(m.positions)
    if m.n_faces == 0:
        return EdgeWeights(np.zeros((0, 2), np.int32), np.zeros(0, np.float32),
                           {"edges": 0, "edges_90": 0, "edges_step": 0, "edges_nonmanifold": 0})
    vs = float(part.voxel_size)
    P = np.rint((m.positions - np.asarray(part.origin, np.float32)) / vs).astype(np.int64)  # corner units
    Q = P[m.quads]                                    # (F, 4, 3)
    F = len(Q)
    C = Q.mean(1)                                     # face centre
    a = np.argmin(Q.max(1) - Q.min(1), axis=1)        # normal axis (the flat one)
    nrm = np.cross(Q[:, 1] - Q[:, 0], Q[:, 2] - Q[:, 0])
    s = np.sign(nrm[np.arange(F), a]).astype(np.int64)
    n_vec = np.zeros_like(C)
    n_vec[np.arange(F), a] = s
    vox = np.floor(C - 0.5 * n_vec).astype(np.int64)  # owning voxel (x, y, z)
    # per quad side i (edge Q[i] -> Q[i+1]): extent of the face away from that edge
    ext = np.zeros((F, 4), np.int64)
    if mode == "steps":
        table = _face_extents(part.solid())
        for i in range(4):
            d = (Q[:, i] + Q[:, (i + 1) % 4]) * 0.5 - C   # +-0.5 along one in-plane axis
            k = np.argmax(np.abs(d), axis=1)
            sk = np.sign(d[np.arange(F), k]).astype(np.int64)
            for key, arr in table.items():
                sel = (a == key[0]) & (s == key[1]) & (k == key[2]) & (sk == key[3])
                if sel.any():
                    v = vox[sel]
                    ext[sel, i] = arr[v[:, 2], v[:, 1], v[:, 0]]
    elif mode != "all":
        raise ValueError(f"mode must be 'steps' or 'all', not {mode!r}")
    # group the 4F quad sides into mesh edges
    e0 = m.quads.astype(np.int64)
    e1 = np.roll(e0, -1, axis=1)
    key = (np.minimum(e0, e1) * N + np.maximum(e0, e1)).ravel()  # (4F,)
    fidx = np.repeat(np.arange(F), 4)
    side = np.tile(np.arange(4), F)
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    order = np.argsort(inv, kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    two = counts == 2
    ia, ib = order[starts[two]], order[starts[two] + 1]
    fa, fb = fidx[ia], fidx[ib]
    different = (a[fa] != a[fb]) | (s[fa] != s[fb])
    if mode == "all":
        step = different
    else:
        step = different & ((ext[fa, side[ia]] == 1) | (ext[fb, side[ib]] == 1))
        if min_run > 1 and step.any():
            # runs: consecutive unit edges on one line between faces of the same two normals
            si = np.flatnonzero(step)
            v0, v1 = e0.ravel()[ia[si]], e1.ravel()[ia[si]]
            p0, p1 = P[v0], P[v1]
            d = np.argmax(np.abs(p1 - p0), axis=1)        # edge direction axis
            lo = np.minimum(p0, p1)
            pos = lo[np.arange(len(si)), d]
            perp = np.stack([np.where(d == 0, lo[:, 1], lo[:, 0]), np.where(d == 2, lo[:, 1], lo[:, 2])], 1)
            nid_a = a[fa[si]] * 2 + (s[fa[si]] > 0)      # face normal id 0..5
            nid_b = a[fb[si]] * 2 + (s[fb[si]] > 0)
            kind = np.minimum(nid_a, nid_b) * 8 + np.maximum(nid_a, nid_b)
            o = np.lexsort((pos, perp[:, 1], perp[:, 0], d, kind))
            g = np.stack([kind[o], d[o], perp[o, 0], perp[o, 1]], 1)
            new_run = np.ones(len(o), bool)
            new_run[1:] = (g[1:] != g[:-1]).any(1) | (pos[o][1:] != pos[o][:-1] + 1)
            run_id = np.cumsum(new_run) - 1
            run_len = np.bincount(run_id)[run_id]
            keep = np.empty(len(o), bool)
            keep[o] = run_len >= min_run
            step[si[~keep]] = False
    w = np.zeros(len(uniq), np.float32)
    w[np.flatnonzero(two)[step]] = 1.0
    edges = np.stack([uniq // N, uniq % N], 1).astype(np.int32)
    stats = {"edges": int(len(uniq)), "edges_90": int(different.sum()), "edges_step": int(step.sum()),
             "edges_nonmanifold": int((~two).sum())}
    return EdgeWeights(edges, w, stats)
