"""Slope mesher: cube faces whose regular staircases become planar facets of 26.6, 45 and 63.4
degrees (treads:risers 2:1, 1:1, 1:2 in voxels; with ``step=2`` the same ratios at twice the size:
4:2, 2:2, 2:4), while lone steps and everything irregular stay cubes. The render style ``slopes``
of ``voxgen.blender.render_styles`` (``--part-styles skull=slopes --facet-step N``) uses it; the
``facet`` style (``mesh.step_edge_weights`` + Bevel) only ever makes 45 degree chamfers, so a 2:1
run came out serrated.

Rule set (heightfield slope fitting per face direction; to be ported as is to ``src/voxbody``,
see "C++ port" below):

1. Directions are ranked ``+y, -y, -z, +z, +x, -x`` (:data:`DIRS`). For every direction ``d`` the
   exposed faces facing ``d`` form a heightfield: face of voxel ``v`` at plane coordinate
   ``h = v[axis] + (sign > 0)`` and *outward height* ``H = sign * h``; its cell is ``(a, b)``, the two
   in-plane voxel coordinates. Several faces may share a cell (overhangs); each is its own sample.
2. *Links*: along each in-plane axis and each side the face is linked to at most one neighbour face
   of the same direction in the next cell whose outward height differs by ``k``, ``|k| <= 2 * step``,
   provided the voxels between form a staircase: the riser between them is a solid wall facing the
   lower side and the column of the lower face is air up to the upper height (``_links``).
3. *Profile*: following the links up to ``R = (min_run + 1) * 2 * step`` cells each way gives the
   relative heights ``p[i]`` of the connected surface through the cell (``NONE`` past its end).
   ``L-`` and ``L+`` = 1 + the number of leading ``p == 0`` cells on each side, ``w = L- + L+ - 1`` is
   the width of the cell's own tread.
4. *Classification* along the axis (``_classify``): the tread must have exactly the pattern width
   ``w == step`` (1:1, riser ``step``, slope 1 per cell) or ``w == 2 * step`` (2:1, riser ``step``,
   slope 1/2 per cell); then the next treads on each side must sit at ``+-m * step`` (ascending or
   descending consistently) with the pattern width each (the last tread of the surface may be cut
   short); a step counts per such tread; the cell is sloped when the steps on both sides together
   are ``>= min_run`` (default 2: a single step stays a cube). A 1:2 profile (riser ``2 * step`` per
   ``step`` cells) is never emitted here: seen from the riser direction it is the 2:1 pattern and
   the longer faces own the step (rule 5). Everything else gets slope 0 (flat).
5. *Ownership* of 1:1 steps: a 1:1 run along axis ``a`` seen from ``d`` is the same staircase as
   the 1:1 run along ``d`` seen from the riser direction (``-a`` for an ascending run); both faces
   have the same extent (``step``), so the direction ranked earlier in :data:`DIRS` keeps the slope
   and the other is reset to 0. For 2:1 the tread direction always owns it (rule 4). A plate at
   most ``step`` voxels thick whose sloped sides would pull its two faces into each other (each
   pulls by ``step / 2``) keeps its cubes: both faces' slopes are reset (``_thin_plates``).
6. *Corners*: a sloped face proposes for each of its 4 corners a plane coordinate
   ``h + sign * (slope_a * (a_c - c_a) + slope_b * (b_c - c_b))`` with ``c_a`` the centre of its tread
   along ``a`` (corner units: ``a + (L+ - L- + 1) / 2``), so the edges of every tread lie at
   ``+- step / 2`` of its plane and consecutive treads of a run meet exactly.
7. *Risers*: the faces of the wall between two treads that both carry a nonzero slope (same sign)
   along the axis and whose linked heights differ by exactly ``step`` are dropped and the wall's
   bottom corners are welded to its top corners (union-find on the integer vertices). Dropped faces
   propose nothing. A riser next to a flat tread stays, shortened to the half the slope leaves.
8. *Dual averaging*: the vertices are the welded integer corners (as in ``mesh.culled_quads``);
   each welded group's coordinate along an axis is the mean of the proposals of the faces normal to
   that axis that touch the group (faces of a consistent run agree exactly, so planes stay planar;
   at a run boundary the blend makes the neighbour a mildly non-planar quad); without a proposal
   the integer value stays. Faces normal to other axes simply follow their vertices.
9. *Output*: quads with two welded corners become triangles (the side walls of a sloped block),
   quads with 3+ welded or crossed corners are dropped (a wall face under the end of a slope collapses
   onto its own bottom edge: coincident vertices across a zero-length edge are welded first and the
   zero-area face goes; a feature pinched flat from both sides leaves coincident faces of opposite
   orientation, which cancel), non-planar quads are split along their
   shorter diagonal. Triangles are stored in ``Mesh.quads`` as ``[a, b, c, c]`` (last index
   repeated); ``Mesh.is_closed`` ignores such zero-length edges and the Blender side splits them.

C++ port: ``VoxelMesher`` in ``src/voxbody`` is to build render meshes with the same rules (per
part, per dirty region) so the runtime skin matches the voxgen previews; the inputs are only the
occupancy and the ``step``/``min_run`` parameters, all intermediate data is per face (links,
profile, L-, L+, slope per in-plane axis) plus one union-find over the integer corners, and nothing
depends on floating point until the final averaging, so the port can be tested voxel for voxel
against ``tests/test_mesh_slopes.py``.

Limitations: only the ratios of one scale are fitted (``step=2`` ignores 1-voxel staircases); a
peak or valley tread between an ascending and a descending run stays flat (half risers on both
sides); slopes along the two in-plane axes are fitted independently, so a diagonal (a-and-b)
staircase facet is the sum of the two, not a best fit; with ``step > 1`` a part that is *not*
exactly ``step``-granular (:func:`granular_phase` is None, e.g. the barber skull with its 1-voxel
face relief) is fitted on the fine grid, where the full-voxel pulls on unit wall faces leave a few
invisible zero-area triangles and the odd dropped folded quad (``stats``: ``faces_collapsed``,
``fins_removed``, ``faces_dropped_crossed``); the raw cube mesh's own non-manifold edges (two
voxels touching along an edge) are kept as they are. Per face the mesher is O(step * min_run)
gathers; the barber skull (16.5k faces) takes ~0.2 s.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mesh import _AO_LEVELS, Mesh, _empty, _pack_face_colors, _palette_u8, _to_metres, _voxel_rgb

DIRS = ((1, 1), (1, -1), (2, -1), (2, 1), (0, 1), (0, -1))  # (axis, sign) in ownership rank order
NONE = 127  # sentinel of the int8/int16 link and profile arrays
_E = np.eye(3, dtype=np.int64)


# ----------------------------------------------------------------------------- grid helpers
def _gather(ap: np.ndarray, P: int, pts: np.ndarray) -> np.ndarray:
    """``ap[pts]`` for ``ap`` padded by ``P`` cells of the "nothing" value; points outside the
    padded box read the pad."""
    nz, ny, nx = ap.shape
    x = np.clip(pts[:, 0] + P, 0, nx - 1)
    y = np.clip(pts[:, 1] + P, 0, ny - 1)
    z = np.clip(pts[:, 2] + P, 0, nz - 1)
    return ap[z, y, x]


def _scatter(ap: np.ndarray, P: int, pts: np.ndarray, values) -> None:
    ap[pts[:, 2] + P, pts[:, 1] + P, pts[:, 0] + P] = values


def _rank(axis: int, sign: int) -> int:
    return DIRS.index((axis, sign))


@dataclass
class _Sheet:
    """The exposed faces of one direction."""
    d: int
    ax: int
    sg: int
    inpl: tuple            # the two in-plane axes (ascending)
    vox: np.ndarray        # (n, 3) int64 xyz of the owning voxels
    h: np.ndarray          # (n,) int64 plane coordinate along ax
    corners: np.ndarray    # (n, 4, 3) int64 corner coordinates, CCW seen from outside
    ao: np.ndarray         # (n, 4) float32 corner AO (0fps rule)
    fid0: int = 0          # global id of face 0
    slope: np.ndarray = field(default_factory=lambda: np.zeros(0))   # (n, 2) float32 per in-plane axis
    L: np.ndarray = field(default_factory=lambda: np.zeros(0))       # (n, 2, 2) int64 [axis][minus, plus]
    prof: dict = field(default_factory=dict)                         # (axis idx, side) -> (n, R) int16

    @property
    def n(self) -> int:
        return int(len(self.vox))


def _sheets(sp: np.ndarray, P: int, shape: tuple, ao: bool) -> list:
    """One :class:`_Sheet` per direction of :data:`DIRS` from the padded occupancy ``sp``."""
    nz, ny, nx = shape
    out = []
    fid0 = 0
    for d, (ax, sg) in enumerate(DIRS):
        off = sg * _E[ax]
        nb = sp[P + off[2]:P + off[2] + nz, P + off[1]:P + off[1] + ny, P + off[0]:P + off[0] + nx]
        exposed = sp[P:P + nz, P:P + ny, P:P + nx] & ~nb
        idx = np.argwhere(exposed)
        vox = idx[:, ::-1].astype(np.int64)
        n = len(vox)
        b, c = (ax + 1) % 3, (ax + 2) % 3
        order = [(0, 0), (1, 0), (1, 1), (0, 1)]  # CCW seen from +ax
        if sg < 0:
            order = order[::-1]
        q = vox + off  # the air neighbour
        corners = np.empty((n, 4, 3), np.int64)
        ao4 = np.ones((n, 4), np.float32)
        for i, (ub, uc) in enumerate(order):
            pos = vox.copy()
            pos[:, ax] += 1 if sg > 0 else 0
            pos[:, b] += ub
            pos[:, c] += uc
            corners[:, i] = pos
            if ao and n:
                db, dc = 2 * ub - 1, 2 * uc - 1
                s1 = _gather(sp, P, q + db * _E[b])
                s2 = _gather(sp, P, q + dc * _E[c])
                s3 = _gather(sp, P, q + db * _E[b] + dc * _E[c])
                ao4[:, i] = _AO_LEVELS[np.where(s1 & s2, 3, s1.astype(np.int64) + s2 + s3)]
        inpl = tuple(k for k in range(3) if k != ax)
        out.append(_Sheet(d, ax, sg, inpl, vox, vox[:, ax] + (1 if sg > 0 else 0), corners, ao4, fid0))
        fid0 += n
    return out


# ----------------------------------------------------------------------------- links and profiles
def _links(sheet: _Sheet, sp: np.ndarray, ex: np.ndarray, P: int, a: int, side: int, step: int) -> np.ndarray:
    """(n,) int8: the outward height change ``k`` to the linked neighbour face in the next cell on
    ``side`` of in-plane axis ``a`` (``NONE`` when there is none). ``ex``: padded exposure mask of the
    sheet's direction."""
    ax, sg, v = sheet.ax, sheet.sg, sheet.vox
    ea, ed = side * _E[a], sg * _E[ax]
    k_out = np.full(sheet.n, NONE, np.int8)
    for k in range(-2 * step, 2 * step + 1):
        u = v + ea + k * ed
        ok = _gather(ex, P, u)
        if k > 0:
            for j in range(1, k):
                ok &= _gather(sp, P, u - j * ed)      # riser wall in the upper column
            for j in range(1, k + 1):
                ok &= ~_gather(sp, P, v + j * ed)     # air above the lower tread
        elif k < 0:
            for j in range(1, -k):
                ok &= _gather(sp, P, v - j * ed)      # riser wall in the own column
            for j in range(1, -k + 1):
                ok &= ~_gather(sp, P, u + j * ed)     # air above the lower (neighbour) tread
        k_out[ok] = k
    return k_out


def _profile(sheet: _Sheet, link_pad: np.ndarray, P: int, a: int, side: int, R: int) -> np.ndarray:
    """(n, R) int16 relative outward heights of the linked faces 1..R cells away on ``side`` of
    axis ``a``; ``NONE`` from the end of the connected surface on."""
    ax, sg = sheet.ax, sheet.sg
    ea, ed = side * _E[a], sg * _E[ax]
    pos = sheet.vox.copy()
    rel = np.zeros(sheet.n, np.int64)
    alive = np.ones(sheet.n, bool)
    prof = np.full((sheet.n, R), NONE, np.int16)
    for i in range(R):
        k = _gather(link_pad, P, pos).astype(np.int64)
        alive &= k != NONE
        if not alive.any():
            break
        k = np.where(alive, k, 0)
        pos = pos + ea + k[:, None] * ed[None, :]
        rel += k
        prof[alive, i] = rel[alive]
    return prof


# ----------------------------------------------------------------------------- classification
def _leading_zeros(prof: np.ndarray) -> np.ndarray:
    nz = prof != 0
    return np.where(nz.any(1), nz.argmax(1), prof.shape[1])


def _count_steps(prof: np.ndarray, start: np.ndarray, w: int, r: int, sign: int, max_steps: int) -> np.ndarray:
    """Number (capped at ``max_steps``) of consecutive treads of width ``w`` beyond ``start`` sitting
    at ``sign * m * r``; the last tread of the surface may be cut short (``NONE`` after its first cell)."""
    n, R = prof.shape
    count = np.zeros(n, np.int64)
    alive = np.ones(n, bool)
    rows = np.arange(n)
    for m in range(1, max_steps + 1):
        ok = alive.copy()
        for c in range(w):
            idx = start - 1 + (m - 1) * w + c  # prof[i] is the cell i + 1 away
            val = np.where(idx < R, prof[rows, np.minimum(idx, R - 1)], NONE)
            if c == 0:
                ok &= val == sign * m * r
            else:
                ok &= (val == sign * m * r) | (val == NONE)
        count += ok
        alive &= ok
    return count


def _classify(p_minus: np.ndarray, p_plus: np.ndarray, step: int, min_run: int) -> tuple:
    """Slope per cell (dh per cell, 0 / +-1 / +-1/2) and (L-, L+) from the two half profiles."""
    L_minus = 1 + _leading_zeros(p_minus)
    L_plus = 1 + _leading_zeros(p_plus)
    w = L_minus + L_plus - 1
    slope = np.zeros(len(w), np.float32)
    for wp, mag in ((step, 1.0), (2 * step, 0.5)):
        sel = w == wp
        if not sel.any():
            continue
        for sign in (1, -1):
            n_plus = _count_steps(p_plus, L_plus, wp, step, sign, min_run)
            n_minus = _count_steps(p_minus, L_minus, wp, step, -sign, min_run)
            hit = sel & (n_plus + n_minus >= min_run) & (slope == 0)
            slope[hit] = sign * mag
    return slope, L_minus, L_plus


# ----------------------------------------------------------------------------- fitting (rules 2-5)
def _fit(sheets: list, sp: np.ndarray, P: int, step: int, min_run: int) -> None:
    """Fill ``slope``, ``L`` and ``prof`` of every sheet, then apply the 1:1 ownership rule."""
    R = (min_run + 1) * 2 * step
    for sh in sheets:
        ex = np.zeros(sp.shape, bool)
        _scatter(ex, P, sh.vox, True)
        sh.slope = np.zeros((sh.n, 2), np.float32)
        sh.L = np.ones((sh.n, 2, 2), np.int64)
        for ai, a in enumerate(sh.inpl):
            profs = {}
            for side in (1, -1):
                lp = np.full(sp.shape, NONE, np.int8)
                _scatter(lp, P, sh.vox, _links(sh, sp, ex, P, a, side, step))
                profs[side] = _profile(sh, lp, P, a, side, R)
            slope, l_minus, l_plus = _classify(profs[-1], profs[1], step, min_run)
            sh.slope[:, ai] = slope
            sh.L[:, ai, 0] = l_minus
            sh.L[:, ai, 1] = l_plus
            sh.prof[(ai, 1)] = profs[1]
            sh.prof[(ai, -1)] = profs[-1]
    for sh in sheets:  # rule 5: a 1:1 run belongs to the earlier-ranked of tread / riser direction
        for ai, a in enumerate(sh.inpl):
            s = sh.slope[:, ai]
            for sign in (1, -1):
                if _rank(a, -sign) < sh.d:
                    s[s == sign * 1.0] = 0.0


def _thin_plates(sheets: list, sp: np.ndarray, P: int, step: int) -> int:
    """Rule 5b: a plate at most ``step`` voxels thick whose two sides would pull their corners
    together (a sloped side pulls by step / 2) keeps its cubes: the slopes of both faces are reset.
    Returns the number of plate columns reset."""
    by_dir = {(sh.ax, sh.sg): sh for sh in sheets}
    n_reset = 0
    for ax in range(3):
        top, bot = by_dir[(ax, 1)], by_dir[(ax, -1)]
        if top.n == 0 or bot.n == 0:
            continue
        fid_bot = np.full(sp.shape, -1, np.int32)
        _scatter(fid_bot, P, bot.vox, np.arange(bot.n, dtype=np.int32))
        v = top.vox
        t = np.ones(top.n, np.int64)  # solid thickness below the face, counted up to step + 1
        alive = np.ones(top.n, bool)
        for j in range(1, step + 1):
            alive &= _gather(sp, P, v - j * _E[ax])
            t += alive
        idx = np.flatnonzero(~alive)
        if len(idx) == 0:
            continue
        b = _gather(fid_bot, P, v[idx] - (t[idx] - 1)[:, None] * _E[ax][None, :])
        ok = b >= 0
        idx, b = idx[ok], b[ok]
        pull = ((top.slope[idx] != 0).any(1).astype(np.float64) + (bot.slope[b] != 0).any(1)) * (step / 2.0)
        hit = pull >= t[idx]
        top.slope[idx[hit]] = 0.0
        bot.slope[b[hit]] = 0.0
        n_reset += int(hit.sum())
    return n_reset


def _union(n: int, pairs: np.ndarray) -> np.ndarray:
    """Union-find labels (the smallest vertex id of each group) for ``pairs`` (m, 2)."""
    label = np.arange(n)
    if len(pairs) == 0:
        return label
    a, b = pairs[:, 0], pairs[:, 1]
    for _ in range(64):
        m = np.minimum(label[a], label[b])
        np.minimum.at(label, label[a], m)
        np.minimum.at(label, label[b], m)
        np.minimum.at(label, a, m)
        np.minimum.at(label, b, m)
        label = label[label]
        if (label[a] == label[b]).all() and (label[label] == label).all():
            break
    return label


def _risers(sheets: list, sp: np.ndarray, P: int, step: int, vid, drop: np.ndarray) -> np.ndarray:
    """Rule 7: mark the riser faces between two sloped treads in ``drop`` and return the (m, 2)
    vertex pairs (wall bottom, wall top) to weld."""
    fid = []
    for sh in sheets:
        g = np.full(sp.shape, -1, np.int32)
        _scatter(g, P, sh.vox, np.arange(sh.n, dtype=np.int32) + sh.fid0)
        fid.append(g)
    pairs = []
    for sh in sheets:
        ed = sh.sg * _E[sh.ax]
        for ai, a in enumerate(sh.inpl):
            b = sh.inpl[1 - ai]
            prof_all = {1: sh.prof[(ai, 1)], -1: sh.prof[(ai, -1)]}
            R = prof_all[1].shape[1]
            for sigma in (1, -1):  # the run ascends toward sigma * a
                sel = np.flatnonzero(np.sign(sh.slope[:, ai]) == sigma)
                if len(sel) == 0:
                    continue
                l_side = sh.L[sel, ai, 1 if sigma > 0 else 0]
                prof = prof_all[sigma][sel]
                idx = l_side - 1
                nxt = np.where(idx < R, prof[np.arange(len(sel)), np.minimum(idx, R - 1)], NONE)
                ok = nxt == step
                v = sh.vox[sel]
                base = v + (sigma * l_side)[:, None] * _E[a][None, :]
                up = _gather(fid[sh.d], P, base + step * ed)
                ok &= up >= 0
                up_slope = sh.slope[np.clip(up - sh.fid0, 0, sh.n - 1), ai]
                ok &= np.sign(up_slope) == sigma
                if not ok.any():
                    continue
                base, v, h = base[ok], v[ok], sh.h[sel][ok]
                rd = _rank(a, -sigma)
                for j in range(1, step + 1):
                    rid = _gather(fid[rd], P, base + j * ed)
                    drop[rid[rid >= 0]] = True
                lo = base.copy()
                lo[:, a] += 1 if sigma < 0 else 0
                lo[:, sh.ax] = h
                hi = lo.copy()
                hi[:, sh.ax] = h + sh.sg * step
                for ub in (0, 1):
                    plo, phi = lo.copy(), hi.copy()
                    plo[:, b] += ub
                    phi[:, b] += ub
                    i_lo, ok_lo = vid(plo)
                    i_hi, ok_hi = vid(phi)
                    good = ok_lo & ok_hi
                    pairs.append(np.stack([i_lo[good], i_hi[good]], 1))
    if not pairs:
        return np.zeros((0, 2), np.int64)
    return np.concatenate(pairs)


def _assemble(fq: np.ndarray, pos: np.ndarray, stats: dict) -> tuple:
    """Rule 9: (quads (F', 4) int32 with triangles as [a, b, c, c], source face index (F',),
    corner index (F', 4)) from the group ids ``fq`` (F, 4) of the kept faces."""
    F = len(fq)
    eq = np.stack([fq[:, i] == fq[:, (i + 1) % 4] for i in range(4)], 1)
    n_eq = eq.sum(1)
    crossed = (fq[:, 0] == fq[:, 2]) | (fq[:, 1] == fq[:, 3])
    full = (n_eq == 0) & ~crossed
    tri = (n_eq == 1) & ~crossed
    stats["faces_collapsed"] = int((~full & ~tri & ~crossed).sum())
    stats["faces_dropped_crossed"] = int(crossed.sum())
    out_q, out_src, out_c = [], [], []
    # triangles from quads with one welded edge
    ti = np.flatnonzero(tri)
    if len(ti):
        i0 = eq[ti].argmax(1)
        rot = (i0[:, None] + 1 + np.arange(3)[None, :]) % 4
        c = np.concatenate([rot, rot[:, 2:3]], 1)
        out_q.append(np.take_along_axis(fq[ti], c, 1))
        out_src.append(ti)
        out_c.append(c)
    fi = np.flatnonzero(full)
    if len(fi):
        q = fq[fi]
        p = pos[q]  # (n, 4, 3)
        vol = np.abs(np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])))
        planar = vol < 1e-6
        c4 = np.tile(np.arange(4), (planar.sum(), 1))
        out_q.append(q[planar])
        out_src.append(fi[planar])
        out_c.append(c4)
        np_ = np.flatnonzero(~planar)
        stats["quads_split"] = int(len(np_))
        if len(np_):
            d02 = np.linalg.norm(p[np_, 2] - p[np_, 0], axis=1)
            d13 = np.linalg.norm(p[np_, 3] - p[np_, 1], axis=1)
            first = d02 <= d13
            ca = np.where(first[:, None], np.array([[0, 1, 2, 2]]), np.array([[1, 2, 3, 3]]))
            cb = np.where(first[:, None], np.array([[0, 2, 3, 3]]), np.array([[1, 3, 0, 0]]))
            for c in (ca, cb):
                out_q.append(np.take_along_axis(q[np_], c, 1))
                out_src.append(fi[np_])
                out_c.append(c)
    stats["triangles"] = int(len(ti)) + 2 * stats.get("quads_split", 0)
    q, src, c = np.concatenate(out_q).astype(np.int32), np.concatenate(out_src), np.concatenate(out_c)
    # fins: a feature pinched flat from both sides leaves pairs of coincident faces of opposite
    # orientation (the same vertex set); they cancel
    key = q.copy()
    key[key[:, 2] == key[:, 3], 3] = -1  # a triangle's key is its 3 distinct vertices
    key = np.sort(key, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    fin = cnt[inv.ravel()] > 1
    stats["fins_removed"] = int(fin.sum())
    if fin.any():
        q, src, c = q[~fin], src[~fin], c[~fin]
    return q, src, c


# ----------------------------------------------------------------------------- public
def _run(solid: np.ndarray, step: int, min_run: int, ao: bool, stats: dict) -> tuple | None:
    """Rules 1-9 on the occupancy; (positions (G, 3) float64 corner units, quads (F', 4) int32,
    corner AO (F', 4), owning voxel (F', 3), sheet id (F',)) or None without faces."""
    shape = solid.shape
    P = 2 * step + 2
    sp = np.pad(solid, P)
    sheets = _sheets(sp, P, shape, ao)
    F = sum(sh.n for sh in sheets)
    stats["faces_in"] = F
    if F == 0:
        return None
    _fit(sheets, sp, P, step, min_run)
    stats["thin_plates_reset"] = _thin_plates(sheets, sp, P, step)
    stats["faces_sloped"] = int(sum((sh.slope != 0).any(1).sum() for sh in sheets))
    # welded integer vertices
    corners = np.concatenate([sh.corners for sh in sheets])  # (F, 4, 3)
    K = int(max(shape)) + 2
    flat = corners.reshape(-1, 3)
    keys = flat[:, 0] + K * (flat[:, 1] + K * flat[:, 2])
    uniq, inv = np.unique(keys, return_inverse=True)
    quads = inv.reshape(F, 4)
    N = len(uniq)
    ipos = np.stack([uniq % K, (uniq // K) % K, uniq // (K * K)], 1).astype(np.float64)

    def vid(pts):
        k = pts[:, 0] + K * (pts[:, 1] + K * pts[:, 2])
        i = np.clip(np.searchsorted(uniq, k), 0, N - 1)
        return i, uniq[i] == k

    # rule 6: corner proposals along the face normal axis
    prop = np.zeros((F, 4), np.float64)
    face_axis = np.concatenate([np.full(sh.n, sh.ax, np.int64) for sh in sheets])
    for sh in sheets:
        off = np.zeros((sh.n, 4), np.float64)
        for ai, a in enumerate(sh.inpl):
            c = sh.vox[:, a] + (sh.L[:, ai, 1] - sh.L[:, ai, 0] + 1) / 2.0
            off += sh.slope[:, ai].astype(np.float64)[:, None] * (sh.corners[:, :, a] - c[:, None])
        prop[sh.fid0:sh.fid0 + sh.n] = sh.h[:, None] + sh.sg * off
    # rule 7: risers
    drop = np.zeros(F, bool)
    pairs = _risers(sheets, sp, P, step, vid, drop)
    stats["risers_dropped"] = int(drop.sum())
    label = _union(N, pairs)
    groups, g_of_v = np.unique(label, return_inverse=True)
    G = len(groups)
    stats["vertices_merged"] = int(N - G)
    # rule 8: dual averaging per axis
    pos = ipos[groups].copy()
    keep = ~drop
    g = g_of_v[quads]
    for ax in range(3):
        sel = keep & (face_axis == ax)
        if not sel.any():
            continue
        gg, pp = g[sel].ravel(), prop[sel].ravel()
        cnt = np.bincount(gg, minlength=G)
        sums = np.bincount(gg, weights=pp, minlength=G)
        has = cnt > 0
        pos[has, ax] = sums[has] / cnt[has]
    # a wall face under the end of a slope can collapse (its top pulled down by step / 2 onto its
    # own bottom edge): weld vertices that coincide across a zero-length edge, the collapsed face
    # is dropped in _assemble and its neighbours become triangles
    kept = np.flatnonzero(keep)
    fk = g[kept]
    e0, e1 = fk.ravel(), np.roll(fk, -1, axis=1).ravel()
    same = (e0 != e1) & (np.linalg.norm(pos[e0] - pos[e1], axis=1) < 1e-9)
    stats["vertices_welded"] = 0
    if same.any():
        lab2 = _union(G, np.stack([e0[same], e1[same]], 1))
        groups2, g2 = np.unique(lab2, return_inverse=True)
        stats["vertices_welded"] = int(G - len(groups2))
        pos, fk = pos[groups2], g2[fk]
    # rule 9: faces
    out_q, src, cidx = _assemble(fk, pos, stats)
    src = kept[src]
    vox = np.concatenate([sh.vox for sh in sheets])[src]
    sheet_id = np.concatenate([np.full(sh.n, sh.d, np.int64) for sh in sheets])[src]
    corner_ao = np.take_along_axis(np.concatenate([sh.ao for sh in sheets])[src], cidx, 1).astype(np.float32)
    stats["faces_out"] = int(len(out_q))
    return pos, out_q, corner_ao, vox, sheet_id


def granular_phase(solid: np.ndarray, step: int) -> tuple | None:
    """The (px, py, pz) phase at which ``solid`` is exactly made of ``step``-cubes (every aligned
    block uniformly solid or air), or None. ``step=1`` is always (0, 0, 0)."""
    if step == 1:
        return (0, 0, 0)
    nz, ny, nx = solid.shape
    for pz in range(step):
        for py in range(step):
            for px in range(step):
                pad = np.pad(solid, ((pz, (-nz - pz) % step), (py, (-ny - py) % step), (px, (-nx - px) % step)))
                blocks = pad.reshape(pad.shape[0] // step, step, pad.shape[1] // step, step,
                                     pad.shape[2] // step, step)
                first = blocks[:, :1, :, :1, :, :1]
                if (blocks == first).all():
                    return (px, py, pz)
    return None


def slope_mesh(solid: np.ndarray, color: np.ndarray | None = None, step: int = 1, min_run: int = 2,
               pal: np.ndarray | None = None, material: np.ndarray | None = None, jitter: float = 0.0,
               ao: bool = True, stats: dict | None = None) -> Mesh:
    """Slope mesh of the occupancy ``solid`` (nz, ny, nx) bool; positions in voxel corner units
    (voxel (x, y, z) spans [x, x+1] x ...), see the module docstring for the rules. ``color``: palette
    index per voxel (``pal`` (256, 3) uint8 gives the rgb, grey without), ``material``: per-voxel id
    for ``Mesh.face_material`` (1 where solid without), ``step``: tread depth of the 1:1 pattern,
    ``min_run``: steps a run needs, ``stats``: a dict to fill with counts. A solid that is exactly
    made of ``step``-cubes at some grid phase (:func:`granular_phase`; a model drawn at 1/2 U on a
    1/4 U grid) is fitted at that coarse scale and scaled back, which is the same rule set without
    the unit vertices between the coarse corners (``stats["coarse_phase"]``); any other solid is
    fitted on the fine grid with the scaled patterns."""
    if step < 1 or min_run < 1:
        raise ValueError(f"step and min_run must be >= 1 (got {step}, {min_run})")
    solid = np.asarray(solid, bool)
    stats = stats if stats is not None else {}
    shape = solid.shape
    phase = granular_phase(solid, step)
    stats["coarse_phase"] = phase
    if phase is not None and step > 1:
        px, py, pz = phase
        nz, ny, nx = shape
        pad = np.pad(solid, ((pz, (-nz - pz) % step), (py, (-ny - py) % step), (px, (-nx - px) % step)))
        res = _run(pad[::step, ::step, ::step], 1, min_run, ao, stats)
        if res is not None:
            pos, quads, corner_ao, vox, sheet = res
            pos = pos * step - np.array([px, py, pz], np.float64)
            vox = vox * step - np.array([px, py, pz])
            for d, (ax, sg) in enumerate(DIRS):  # the fine voxel of the block's face layer
                vox[sheet == d, ax] += (step - 1) if sg > 0 else 0
            vox = np.clip(vox, 0, np.array([nx - 1, ny - 1, nz - 1]))
    else:
        res = _run(solid, step, min_run, ao, stats)
        if res is not None:
            pos, quads, corner_ao, vox, sheet = res
    if res is None:
        return Mesh(np.zeros((0, 3), np.float32), np.zeros((0, 4), np.int32), np.zeros((0, 4), np.uint8),
                    np.zeros((0,), np.uint8), np.zeros((0, 4), np.float32), np.zeros((0,), bool))
    if pal is None:
        pal = np.full((256, 3), 200, np.uint8)
    if color is None:
        color = np.zeros(shape, np.uint8)
    rgb = _voxel_rgb(pal, color, vox, jitter)
    if material is None:
        mats = np.ones(len(vox), np.uint8)
    else:
        mats = material[vox[:, 2], vox[:, 1], vox[:, 0]].astype(np.uint8)
    return Mesh(pos.astype(np.float32), quads, _pack_face_colors(rgb, corner_ao), mats, corner_ao,
                np.zeros(len(quads), bool))


def slope_mesh_part(part, palette, step: int = 1, min_run: int = 2, jitter: float = 0.0,
                    stats: dict | None = None) -> Mesh:
    """:func:`slope_mesh` of a ``Part``: positions in metres of its local space."""
    from . import materials as M
    material = part.material
    solid = material != M.MAT_AIR
    m = slope_mesh(solid, part.color, step, min_run, pal=_palette_u8(palette), material=material,
                   jitter=jitter, stats=stats)
    if m.n_faces == 0:
        return _empty(part)
    m.positions = _to_metres(part, m.positions)
    return m
