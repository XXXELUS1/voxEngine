"""Numpy meshers for .vxm parts: the geometry behind the Blender renders (``voxgen.blender``).

Pure numpy, importable without Blender. Two builders, both returning a :class:`Mesh`:

* :func:`culled_quads` - one quad per exposed voxel face, vertices welded (so a Bevel modifier
  works), per-face palette colour with a deterministic per-voxel value jitter, per-corner AO
  (classic "0fps" 3-neighbour count).
* :func:`surface_nets` - naive surface nets on the occupancy with cell-clamped Laplacian
  smoothing; faces keep the colour of the voxel that owns them (crisp eyes, lips), vertices get
  a "cavity" AO. Materials in ``keep_cubic_materials`` (empty by default, so hair and teeth are
  part of the smooth surface; ``CUBIC_MATERIALS`` = teeth + hair restores the cubic look) are
  excluded from the smooth surface and appended as culled cubes (culled against the full occupancy).
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


def surface_nets(part, palette, smooth_passes=2, keep_cubic_materials=(), jitter=0.04) -> Mesh:
    """Smooth surface of a part (naive surface nets + clamped Laplacian), with the materials in
    ``keep_cubic_materials`` (none by default) left out of the surface and appended as culled cubes."""
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
