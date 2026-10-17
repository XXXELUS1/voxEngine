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
   along the axis and whose linked heights differ by exactly ``step`` are dropped and every integer
   vertex of the wall's two end lines is welded into one group (union-find). Dropped faces propose
   nothing, and a tread that an earlier-ranked sheet already dropped keeps no run of its own (on a
   diagonal staircase every -z face is a riser of the +y runs; welding it from both sheets chained
   whole staircases into one vertex). A riser next to a flat tread stays, shortened to the half
   the slope leaves.
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

``step > 1`` (:func:`_hybrid`): the solid is cut into ``step``-blocks at the grid phase with the
fewest *mixed* blocks (partly solid: the 1-voxel face relief of the bald skull); the uniform blocks
are fitted by the same rules at the block scale (1 block = ``step`` voxels, pulls of half a block)
and scaled back, the mixed blocks stay fine cubes. A block face toward a mixed block is cut into
its unit faces (those against solid voxels culled), every block face with a corner on a mixed block
is pinned flat so the cubes attach at integer corners, and a block face whose edge carries such
unit corners is fan-triangulated from its centre through them. (The earlier fine-grid fit of a
non-granular solid moved unit-face corners by ``step / 2`` and left holes and folded faces.)

Chamfer mode (:func:`chamfer_mesh`, style ``chamfer``): the ``facet`` look without Blender's Bevel.
The step edges of ``mesh.step_edge_table`` (a 90 degree edge, convex or concave, where either face
extends exactly ``step`` voxels, in a run of ``min_run`` unit edges; here both faces must extend
``step / 2`` and a face narrower than ``step`` cannot take two chamfers) are cut 45 degrees
``step / 2`` voxels into both faces: on the doubled grid every edge's wedge of ``step`` half-voxels
is carved (convex) or filled (concave) as a 1:1 staircase, two fillets meeting at a corner get the
tetrahedral mitre ``i + j + k <= step - 1``, a fillet cell whose wall or floor a chamfer took away
is dropped, and the leftover checkerboard edges are filled. The staircases are then fitted by the
rules above at ``step = 1``, ``min_run = 1``, only on the tagged staircase faces, with the plane
shifted by half a cell outward (carves) or inward (fills) so it passes through the staircase's
outer corners: a run of 3-voxel steps becomes one 45 degree plane, a 6:3 run alternating planes and
flats of 3; two level faces of different staircases (a ridge, a valley, a fillet meeting a chamfer)
are separate treads. The chamfer quads take the colour of the tread voxel. The mitre tips leave
sub-voxel slivers folded sideways (``tests``: under 0.5 % of the area), never holes or inverted faces.

Limitations: only the ratios of one scale are fitted (``step=2`` ignores 1-voxel staircases); a
peak or valley tread between an ascending and a descending run stays flat (half risers on both
sides); slopes along the two in-plane axes are fitted independently, so a diagonal (a-and-b)
staircase facet is the sum of the two, not a best fit; the raw cube mesh's own non-manifold edges
(two voxels touching along an edge) are kept as they are. Per face the mesher is O(step * min_run)
gathers; the bald skull at step 3 takes ~0.25 s (slopes, 4k faces) and ~1 s (chamfer, 50k faces).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mesh import _AO_LEVELS, Mesh, _empty, _pack_face_colors, _palette_u8, _to_metres, _voxel_rgb, voxel_hash01

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
    shift: np.ndarray = field(default_factory=lambda: np.zeros(0))   # (n, 2) float64 extra plane offset (chamfer cut)
    L: np.ndarray = field(default_factory=lambda: np.zeros(0))       # (n, 2, 2) int64 [axis][minus, plus]
    prof: dict = field(default_factory=dict)                         # (axis idx, side) -> (n, R) int16

    @property
    def n(self) -> int:
        return int(len(self.vox))


def _sheets(sp: np.ndarray, P: int, shape: tuple, ao: bool, masks: list | None = None) -> list:
    """One :class:`_Sheet` per direction of :data:`DIRS` from the padded occupancy ``sp``;
    ``masks`` (one (nz, ny, nx) bool per direction) restricts the owning voxels."""
    nz, ny, nx = shape
    out = []
    fid0 = 0
    for d, (ax, sg) in enumerate(DIRS):
        off = sg * _E[ax]
        nb = sp[P + off[2]:P + off[2] + nz, P + off[1]:P + off[1] + ny, P + off[0]:P + off[0] + nx]
        exposed = sp[P:P + nz, P:P + ny, P:P + nx] & ~nb
        if masks is not None:
            exposed = exposed & masks[d]
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
def _links(sheet: _Sheet, sp: np.ndarray, ex: np.ndarray, P: int, a: int, side: int, step: int,
           cut: np.ndarray | None = None) -> np.ndarray:
    """(n,) int8: the outward height change ``k`` to the linked neighbour face in the next cell on
    ``side`` of in-plane axis ``a`` (``NONE`` when there is none). ``ex``: padded exposure mask of the
    sheet's direction; ``cut``: padded chamfer tags of the sheet's axis, two level faces with
    different tags (a fillet meeting a chamfer at the riser plane) are not one tread."""
    ax, sg, v = sheet.ax, sheet.sg, sheet.vox
    ea, ed = side * _E[a], sg * _E[ax]
    k_out = np.full(sheet.n, NONE, np.int8)
    for k in range(-2 * step, 2 * step + 1):
        u = v + ea + k * ed
        ok = _gather(ex, P, u)
        if k == 0 and cut is not None:
            ok &= _gather(cut, P, u) == _gather(cut, P, v)
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
def _profiles(sheets: list, sp: np.ndarray, P: int, step: int, min_run: int,
              cut: np.ndarray | None = None) -> None:
    """Rules 2-3: fill ``prof`` of every sheet (links followed up to R cells each way) and reset
    ``slope``, ``shift`` and ``L`` (one flat tread per cell). ``cut``: (3, ...) padded chamfer tags
    per face axis (see :func:`_links`)."""
    R = (min_run + 1) * 2 * step
    for sh in sheets:
        ex = np.zeros(sp.shape, bool)
        _scatter(ex, P, sh.vox, True)
        sh.slope = np.zeros((sh.n, 2), np.float32)
        sh.shift = np.zeros((sh.n, 2), np.float64)
        sh.L = np.ones((sh.n, 2, 2), np.int64)
        for ai, a in enumerate(sh.inpl):
            for side in (1, -1):
                lp = np.full(sp.shape, NONE, np.int8)
                _scatter(lp, P, sh.vox, _links(sh, sp, ex, P, a, side, step, None if cut is None else cut[sh.ax]))
                sh.prof[(ai, side)] = _profile(sh, lp, P, a, side, R)


def _fit(sheets: list, sp: np.ndarray, P: int, step: int, min_run: int, cut: np.ndarray | None = None) -> None:
    """Fill ``prof``, ``slope`` and ``L`` of every sheet (rules 2-4), then apply the 1:1 ownership rule."""
    _profiles(sheets, sp, P, step, min_run, cut)
    for sh in sheets:
        for ai in range(2):
            slope, l_minus, l_plus = _classify(sh.prof[(ai, -1)], sh.prof[(ai, 1)], step, min_run)
            sh.slope[:, ai] = slope
            sh.L[:, ai, 0] = l_minus
            sh.L[:, ai, 1] = l_plus
    for sh in sheets:  # rule 5: a 1:1 run belongs to the earlier-ranked of tread / riser direction
        for ai, a in enumerate(sh.inpl):
            s = sh.slope[:, ai]
            for sign in (1, -1):
                if _rank(a, -sign) < sh.d:
                    s[s == sign * 1.0] = 0.0


def _thin_plates(sheets: list, sp: np.ndarray, P: int, step: int) -> int:
    """Rule 5b: a plate at most ``step`` voxels thick whose two sides would pull their corners
    together (a sloped side pulls by step / 2 per sloped in-plane axis) keeps its cubes: the slopes
    of both faces are reset.
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
        pull = ((top.slope[idx] != 0).sum(1) + (bot.slope[b] != 0).sum(1)) * (step / 2.0)  # per sloped axis
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
                # a tread that an earlier-ranked sheet already dropped as its riser (a diagonal
                # staircase: every -z face is a riser of the +y runs) keeps no run of its own
                sel = np.flatnonzero((np.sign(sh.slope[:, ai]) == sigma) & ~drop[sh.fid0:sh.fid0 + sh.n])
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
                ok &= (np.sign(up_slope) == sigma) & ~drop[np.clip(up, 0, len(drop) - 1)]
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
                for ub in (0, 1):  # every integer vertex of the two wall lines joins one group
                    for j in range(step):
                        plo, phi = lo.copy(), lo.copy()
                        plo[:, b] += ub
                        phi[:, b] += ub
                        plo[:, sh.ax] += sh.sg * j
                        phi[:, sh.ax] += sh.sg * (j + 1)
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
def _run(solid: np.ndarray, step: int, min_run: int, ao: bool, stats: dict,
         pins: np.ndarray | None = None, cut: np.ndarray | None = None) -> tuple | None:
    """Rules 1-9 on the occupancy; (positions (G, 3) float64 corner units, quads (F', 4) int32,
    corner AO (F', 4), owning voxel (F', 3), sheet id (F',)) or None without faces. Faces with a
    corner on the ``pins`` lattice ((nz + 1, ny + 1, nx + 1) bool, z y x) stay flat (hybrid mode:
    the corners of the blocks that are meshed as fine cubes); ``cut`` ((3, nz, ny, nx) int8 per
    face axis, chamfer mode) shifts the plane of a face sloped along one axis by ``sign(cut) / 2``
    outward: positive on the faces of a carved wedge's staircase (the plane through its convex
    corners), negative on a filled one's (through the concave corners); level faces with
    different tags are separate treads (see ``_chamfer_grid``)."""
    shape = solid.shape
    P = 2 * step + 2
    sp = np.pad(solid, P)
    sheets = _sheets(sp, P, shape, ao)
    F = sum(sh.n for sh in sheets)
    stats["faces_in"] = F
    if F == 0:
        return None
    cut_pad = None if cut is None else np.stack([np.pad(c, P) for c in cut])
    _fit(sheets, sp, P, step, min_run, cut_pad)
    stats["thin_plates_reset"] = _thin_plates(sheets, sp, P, step)
    if cut is not None:  # chamfer mode: only the tagged staircase faces slope, with the cut shift
        for sh in sheets:
            tag = cut[sh.ax][sh.vox[:, 2], sh.vox[:, 1], sh.vox[:, 0]]
            sh.slope[tag == 0] = 0.0
            one = (sh.slope != 0).sum(1) == 1
            cv = np.sign(tag).astype(np.float64)
            for ai in range(2):
                sh.shift[:, ai] = np.where(one & (sh.slope[:, ai] != 0), 0.5 * cv, 0.0)
    if pins is not None:
        n_pinned = 0
        for sh in sheets:
            c = sh.corners
            hit = pins[c[..., 2], c[..., 1], c[..., 0]].any(1) & (sh.slope != 0).any(1)
            sh.slope[hit] = 0.0
            n_pinned += int(hit.sum())
        stats["faces_pinned"] = n_pinned
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
            off += sh.shift[:, ai][:, None]
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
    if stats.get("debug"):  # internals for diagnostics and tests
        stats["_debug"] = {"ipos": ipos, "quads": quads, "prop": prop, "drop": drop, "pairs": pairs,
                           "g_of_v": g_of_v, "pos": pos, "fk": fk, "kept": kept, "sheets": sheets}
    # rule 9: faces
    out_q, src, cidx = _assemble(fk, pos, stats)
    src = kept[src]
    vox = np.concatenate([sh.vox for sh in sheets])[src]
    sheet_id = np.concatenate([np.full(sh.n, sh.d, np.int64) for sh in sheets])[src]
    corner_ao = np.take_along_axis(np.concatenate([sh.ao for sh in sheets])[src], cidx, 1).astype(np.float32)
    stats["faces_out"] = int(len(out_q))
    stats["face_dir"] = sheet_id
    return pos, out_q, corner_ao, vox, sheet_id


# ----------------------------------------------------------------------------- hybrid (step > 1)
def _block_phase(solid: np.ndarray, step: int) -> tuple:
    """The grid phase (px, py, pz) at which the fewest ``step``-blocks are mixed (partly solid);
    returns (phase, padded solid, solid blocks (Nz, Ny, Nx) bool, mixed blocks)."""
    nz, ny, nx = solid.shape
    best = None
    for pz in range(step):
        for py in range(step):
            for px in range(step):
                pad = np.pad(solid, ((pz, (-nz - pz) % step), (py, (-ny - py) % step), (px, (-nx - px) % step)))
                b = pad.reshape(pad.shape[0] // step, step, pad.shape[1] // step, step, pad.shape[2] // step, step)
                cnt = b.sum(axis=(1, 3, 5))
                full = cnt == step ** 3
                mixed = (cnt > 0) & ~full
                n = int(mixed.sum())
                if best is None or n < best[0]:
                    best = (n, (px, py, pz), pad, full, mixed)
                if n == 0:
                    return best[1:]
    return best[1:]


def _hybrid(solid: np.ndarray, step: int, min_run: int, ao: bool, stats: dict) -> tuple | None:
    """``step > 1``: the solid cut into ``step``-blocks at the phase with the fewest mixed blocks
    (:func:`_block_phase`); the uniform blocks are fitted at the block scale by :func:`_run`
    (the same rules at 1 block = step voxels) and scaled back, the mixed blocks (fine detail:
    the 1-voxel relief of a face) stay fine cubes. A block face toward a mixed block is cut into
    its unit faces (those against solid fine voxels culled), every face with a corner on a mixed
    block is pinned flat (``pins``), so the fine cubes attach at integer corners, and a block face
    whose edge carries such unit corners is fan-triangulated through them. Returns as
    :func:`_run`, in fine corner units."""
    (px, py, pz), pad, full, mixed = _block_phase(solid, step)
    offset = np.array([px, py, pz], np.int64)
    Nz, Ny, Nx = full.shape
    stats["mixed_blocks"] = int(mixed.sum())
    stats["coarse_phase"] = (px, py, pz) if stats["mixed_blocks"] == 0 else None
    stats["phase"] = (px, py, pz)
    dirs = np.array([sg * _E[ax] for ax, sg in DIRS])  # (6, 3) xyz
    # --- coarse mesh of the uniform blocks, mixed block corners pinned
    pins = np.zeros((Nz + 1, Ny + 1, Nx + 1), bool)
    mz, my, mx = np.nonzero(mixed)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                pins[mz + dz, my + dy, mx + dx] = True
    res = _run(full, 1, min_run, ao, stats, pins=pins) if full.any() else None
    parts_pos, parts_q, parts_ao, parts_vox, parts_dir = [], [], [], [], []
    if res is not None:
        pos, quads, corner_ao, vox, sheet = res
        nb = vox + dirs[sheet]
        inside = ((nb >= 0) & (nb < np.array([Nx, Ny, Nz]))).all(1)
        adj = np.zeros(len(vox), bool)
        adj[inside] = mixed[nb[inside, 2], nb[inside, 1], nb[inside, 0]]
        stats["faces_subdivided"] = int(adj.sum())
        keep = ~adj
        pos = pos * step - offset
        quads, corner_ao, vox, sheet = quads[keep], corner_ao[keep], vox[keep], sheet[keep]
        vox = vox * step - offset
        for d, (ax, sg) in enumerate(DIRS):  # the fine voxel of the block's face layer
            vox[sheet == d, ax] += (step - 1) if sg > 0 else 0
        parts_pos.append(pos)
        parts_q.append(quads.astype(np.int64))
        parts_ao.append(corner_ao)
        parts_vox.append(vox)
        parts_dir.append(sheet)
    # --- fine faces: the mixed blocks and the face layers of the uniform blocks toward them
    if mixed.any():
        masks = []
        for d, (ax, sg) in enumerate(DIRS):
            off = dirs[d]
            adjm = np.zeros_like(full)
            src = full[max(0, -off[2]):Nz - max(0, off[2]), max(0, -off[1]):Ny - max(0, off[1]),
                       max(0, -off[0]):Nx - max(0, off[0])]
            dst = mixed[max(0, off[2]):Nz - max(0, -off[2]), max(0, off[1]):Ny - max(0, -off[1]),
                        max(0, off[0]):Nx - max(0, -off[0])]
            adjm[max(0, -off[2]):Nz - max(0, off[2]), max(0, -off[1]):Ny - max(0, off[1]),
                 max(0, -off[0]):Nx - max(0, off[0])] = src & dst
            m = adjm | mixed
            masks.append(np.repeat(np.repeat(np.repeat(m, step, 0), step, 1), step, 2))
        Pf = 2
        spf = np.pad(pad, Pf)
        fine = _sheets(spf, Pf, pad.shape, ao, masks)
        nf = sum(sh.n for sh in fine)
        stats["faces_fine"] = nf
        if nf:
            corners = np.concatenate([sh.corners for sh in fine]).astype(np.float64) - offset
            base = sum(len(p) for p in parts_pos)
            parts_pos.append(corners.reshape(-1, 3))
            parts_q.append(base + np.arange(nf * 4).reshape(nf, 4))
            parts_ao.append(np.concatenate([sh.ao for sh in fine]))
            parts_vox.append(np.concatenate([sh.vox for sh in fine]) - offset)
            parts_dir.append(np.concatenate([np.full(sh.n, sh.d, np.int64) for sh in fine]))
    else:
        stats["faces_fine"] = 0
    if not parts_q:
        return None
    # --- merge vertices by position (the pinned block corners coincide with fine corners)
    pos = np.concatenate(parts_pos)
    keys = np.round(pos * 4.0).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    pos = uniq.astype(np.float64) / 4.0
    quads = inv[np.concatenate(parts_q)]
    corner_ao = np.concatenate(parts_ao).astype(np.float32)
    vox = np.concatenate(parts_vox)
    sheet = np.concatenate(parts_dir)
    # --- T-junctions: a block edge that carries fine corners is split through them
    n_coarse = len(parts_q[0]) if res is not None else 0
    stats["faces_split_tjunction"] = 0
    if res is not None and stats["faces_fine"] and n_coarse:
        fine_keys = set(map(tuple, keys[sum(len(p) for p in parts_pos[:-1]):].tolist()))
        q = quads[:n_coarse]
        ipos = pos[q]  # (n, 4, 3)
        tri = q[:, 2] == q[:, 3]
        polys, changed = [], []
        for f in range(n_coarse):
            nv = 3 if tri[f] else 4
            poly = []
            hit = False
            for i in range(nv):
                a, b = q[f, i], q[f, (i + 1) % nv]
                poly.append(int(a))
                pa, pb = pos[a], pos[b]
                dlt = pb - pa
                axis = int(np.argmax(np.abs(dlt)))
                if abs(abs(dlt[axis]) - step) > 1e-9 or np.abs(np.delete(dlt, axis)).max() > 1e-9:
                    continue
                if np.abs(pa - np.round(pa)).max() > 1e-9 or np.abs(pb - np.round(pb)).max() > 1e-9:
                    continue
                sgn = 1 if dlt[axis] > 0 else -1
                mids = []
                for j in range(1, step):
                    pm = pa.copy()
                    pm[axis] += sgn * j
                    k = tuple(np.round(pm * 4.0).astype(np.int64).tolist())
                    if k not in fine_keys:
                        mids = None
                        break
                    mids.append(k)
                if mids:
                    hit = True
                    for k in mids:
                        poly.append(("k", k))
            if hit:
                polys.append(poly)
                changed.append(f)
        if changed:  # fan from the face centre (a corner fan would leave its own edges unsplit)
            key_to_id = {tuple(k): i for i, k in enumerate(uniq.tolist())}
            new_q, new_src, cents = [], [], []
            for f, poly in zip(changed, polys):
                ids = [key_to_id[p[1]] if isinstance(p, tuple) else p for p in poly]
                c = len(pos) + len(cents)
                cents.append(pos[[i for p_, i in zip(poly, ids) if not isinstance(p_, tuple)]].mean(0))
                for i in range(len(ids)):
                    new_q.append([c, ids[i], ids[(i + 1) % len(ids)], ids[(i + 1) % len(ids)]])
                    new_src.append(f)
            pos = np.concatenate([pos, np.array(cents, np.float64)])
            keep = np.ones(len(quads), bool)
            keep[changed] = False
            new_q = np.array(new_q, np.int64)
            new_src = np.array(new_src)
            quads = np.concatenate([quads[keep], new_q])
            corner_ao = np.concatenate([corner_ao[keep], np.ones((len(new_q), 4), np.float32)])
            vox = np.concatenate([vox[keep], vox[new_src]])
            sheet = np.concatenate([sheet[keep], sheet[new_src]])
            stats["faces_split_tjunction"] = len(changed)
    stats["faces_out"] = int(len(quads))
    stats["face_dir"] = sheet
    return pos, quads.astype(np.int32), corner_ao, vox, sheet


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
    if step > 1:
        res = _hybrid(solid, step, min_run, ao, stats)
    else:
        stats["coarse_phase"] = (0, 0, 0)
        res = _run(solid, 1, min_run, ao, stats)
    if res is not None:
        pos, quads, corner_ao, vox, sheet = res
        vox = np.clip(vox, 0, np.array([shape[2] - 1, shape[1] - 1, shape[0] - 1]))
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


# ----------------------------------------------------------------------------- chamfer
def _chamfer_wedges(solid: np.ndarray, step: int, min_run: int) -> tuple | None:
    """The chamfered unit edges of ``solid`` (:func:`mesh.step_edge_table`, mode "steps") whose
    two faces both extend at least ``step / 2`` voxels, as arrays: edge start corner p (m, 3),
    edge axis k (m,), in-plane axes (a, b) (m, 2) = the normal axes of the two faces, quadrant
    signs (ua, ub) (m, 2) (the solid quadrant of a convex edge, the air quadrant of a concave
    one), convex (m,) bool, owner (m,) 0 = face a owns the chamfer quads, 1 = face b, tread voxel
    (m, 3) of the owner; or None without such edges."""
    from .mesh import _culled_quads, step_edge_table
    res = _culled_quads(solid.astype(np.uint8), np.zeros(solid.shape, np.uint8), np.full((256, 3), 200, np.uint8),
                        solid, solid, 0.0, False)
    if res is None:
        return None
    P, quads = np.asarray(res[0], np.int64), res[1]
    t = step_edge_table(P, quads, solid, "steps", min_run, step)
    sel = np.flatnonzero(t.is_step & (np.minimum(t.ext_a, t.ext_b) * 2 >= step))
    if len(sel) == 0:
        return None
    e = t.edges[np.flatnonzero(t.two)[sel]]
    fa, fb = t.fa[sel], t.fb[sel]
    p0, p1 = P[e[:, 0]], P[e[:, 1]]
    k = np.argmax(np.abs(p1 - p0), axis=1)
    p = np.minimum(p0, p1)
    axa, sa, axb, sb = t.axis[fa], t.sign[fa], t.axis[fb], t.sign[fb]
    va, vb = t.vox[fa], t.vox[fb]
    convex = (va == vb).all(1)
    qv = np.where(convex[:, None], va, va + sa[:, None] * _E[axa])  # solid (convex) / air (concave) voxel
    d = qv - p
    rows = np.arange(len(sel))
    ua = np.where(d[rows, axa] == 0, 1, -1)
    ub = np.where(d[rows, axb] == 0, 1, -1)
    rank_a = np.array([_rank(int(x), int(s_)) for x, s_ in zip(axa, sa)])
    rank_b = np.array([_rank(int(x), int(s_)) for x, s_ in zip(axb, sb)])
    ext_a, ext_b = t.ext_a[sel], t.ext_b[sel]
    # a face narrower than ``step`` cannot take two chamfers (step / 2 each): when its far edge is
    # a chamfer candidate too, both edges stay sharp (the face keeps its cubes)
    keys = {}
    for i in range(len(sel)):
        keys.setdefault((int(p[i, 0]), int(p[i, 1]), int(p[i, 2]), int(k[i]), int(rank_a[i])), []).append(i)
        keys.setdefault((int(p[i, 0]), int(p[i, 1]), int(p[i, 2]), int(k[i]), int(rank_b[i])), []).append(i)
    keep = np.ones(len(sel), bool)
    for i in range(len(sel)):
        for ext, ax_far, u_far, rank in ((ext_a[i], axb[i], ub[i], rank_a[i]), (ext_b[i], axa[i], ua[i], rank_b[i])):
            if ext >= step:
                continue
            p2 = p[i].copy()
            p2[ax_far] += int(ext) * int(u_far)
            for j in keys.get((int(p2[0]), int(p2[1]), int(p2[2]), int(k[i]), int(rank)), []):
                keep[i] = False
                keep[j] = False
    if not keep.any():
        return None
    p, k, axa, sa, axb, sb, va, vb = p[keep], k[keep], axa[keep], sa[keep], axb[keep], sb[keep], va[keep], vb[keep]
    convex, ua, ub, rank_a, rank_b, ext_a, ext_b = convex[keep], ua[keep], ub[keep], rank_a[keep], rank_b[keep], ext_a[keep], ext_b[keep]
    own_a = (ext_a == step) & ((ext_b != step) | (rank_a <= rank_b))
    owner = np.where(own_a, 0, 1)
    tread = np.where(own_a[:, None], va, vb)
    return p, k, np.stack([axa, axb], 1), np.stack([ua, ub], 1), convex, owner, tread, np.stack([sa, sb], 1)


def _fill_checkerboards(solid: np.ndarray, passes: int = 4) -> int:
    """Fill the air cell with the most solid neighbours of every edge whose four cells are two
    diagonal solids and two diagonal airs (a non-manifold edge); returns the number filled."""
    n_filled = 0
    for _ in range(passes):
        nb = np.zeros(solid.shape, np.int64)
        for ax in range(3):
            for sg in (1, -1):
                sh = np.roll(solid, sg, axis=ax)
                idx = [slice(None)] * 3
                idx[ax] = 0 if sg > 0 else -1
                sh[tuple(idx)] = False
                nb += sh
        found = False
        for ax in range(3):
            o1, o2 = [a for a in range(3) if a != ax]
            s = solid
            a = s[tuple(slice(0, -1) if i in (o1, o2) else slice(None) for i in range(3))]
            b = s[tuple(slice(1, None) if i in (o1, o2) else slice(None) for i in range(3))]
            c = s[tuple(slice(1, None) if i == o1 else slice(0, -1) if i == o2 else slice(None) for i in range(3))]
            d = s[tuple(slice(0, -1) if i == o1 else slice(1, None) if i == o2 else slice(None) for i in range(3))]
            for solid_pair, air_pair in (((a, b), (c, d)), ((c, d), (a, b))):
                cb = solid_pair[0] & solid_pair[1] & ~air_pair[0] & ~air_pair[1]
                if not cb.any():
                    continue
                found = True
                pts = np.argwhere(cb)
                for pt in pts:
                    cands = []
                    for which, cell in enumerate(air_pair):
                        q = pt.copy()
                        if cell is c or cell is a:
                            q[o1] += 1 if cell is c else 0
                        if cell is d:
                            q[o2] += 1
                        if cell is b:
                            q[o1] += 1
                            q[o2] += 1
                        cands.append((nb[tuple(q)], tuple(q)))
                    best = max(cands)[1]
                    solid[best] = True
                    n_filled += 1
        if not found:
            break
    return n_filled


def _chamfer_grid(solid: np.ndarray, step: int, min_run: int, color: np.ndarray, material: np.ndarray,
                  stats: dict) -> tuple:
    """The doubled grid of the chamfer mode: every chamfered edge gets its wedge of ``step``
    half-voxels carved (convex) or filled (concave) as a 1:1 staircase; where two concave edges
    of different axes meet, the corner octant gets the tetrahedral mitre of their fillets
    (``i + j + k <= step - 1``). Returns (solid2, color2, material2, cut2) with ``cut2`` (3, ...)
    per face axis: +1 on the staircase faces of the carves, -1 on those of the fills (the plane
    shift and the tread split of :func:`_run`)."""
    rep = lambda a: np.repeat(np.repeat(np.repeat(a, 2, 0), 2, 1), 2, 2)
    solid2, color2, mat2 = rep(solid), rep(color), rep(material)
    cut2 = np.zeros((3,) + solid2.shape, np.int8)
    w = _chamfer_wedges(solid, step, min_run)
    stats["chamfer_edges"] = 0 if w is None else int(len(w[0]))
    stats["chamfer_cushion_cells"] = 0
    if w is None:
        return solid2, color2, mat2, cut2
    p, k, axes, u, convex, owner, tread, signs = w
    m = len(p)
    rows = np.arange(m)
    shape2 = solid2.shape
    lim = np.array([shape2[2], shape2[1], shape2[0]])
    tz, ty, tx = tread[:, 2], tread[:, 1], tread[:, 0]

    def half_index(axis, sign, off):
        """half-voxel index along ``axis``: ``off`` cells from the corner plane into the ``sign`` side."""
        return 2 * p[rows, axis] + np.where(sign > 0, off, -1 - off)

    def paint(c, sel):
        color2[c[:, 2], c[:, 1], c[:, 0]] = color[tz[sel], ty[sel], tx[sel]]
        mat2[c[:, 2], c[:, 1], c[:, 0]] = material[tz[sel], ty[sel], tx[sel]]

    cover = np.zeros((3,) + shape2, bool)  # per edge axis: the corner tetrahedra of the concave edges
    cx, cc = np.flatnonzero(convex), np.flatnonzero(~convex)

    def cells_at(sel, i, j, kk):
        c = np.zeros((len(sel), 3), np.int64)
        r = np.arange(len(sel))
        c[r, axes[sel, 0]] = half_index(axes[:, 0], u[:, 0], i)[sel]
        c[r, axes[sel, 1]] = half_index(axes[:, 1], u[:, 1], j)[sel]
        c[r, k[sel]] = 2 * p[sel, k[sel]] + kk
        return c, ((c >= 0) & (c < lim)).all(1)

    def tag(sel, c, ok, i, j, sign):
        # the staircase faces: of a fill its cells with i + j = step - 1 (the row ends face axis a,
        # the column tops face axis b) and, for the trimmed ends, every fill cell; of a carve the
        # first solid cells behind / under the wedge (i + j = step), where the cell (0, step)
        # carries only a column top and (step, 0) only a row end (their other face continues the
        # flat beyond the chamfer). The tag is sign * (1 + 2 * run axis + (run ascends toward +)):
        # two level faces of different staircases (two chamfers meeting at a ridge or a valley)
        # are not one tread.
        for side_ax, lead in ((0, j), (1, i)):
            if sign < 0 or (i + j == step and lead < step):
                run = axes[sel, 1 - side_ax]
                asc = sign * u[sel, 1 - side_ax] > 0
                code = sign * (1 + 2 * run + asc)
                cut2[axes[sel[ok], side_ax], c[ok, 2], c[ok, 1], c[ok, 0]] = code[ok]

    # carves: the wedge cells of the convex edges go
    for i in range(step + 1):
        for j in range(step + 1 - i):
            for kk in (0, 1):
                c, ok = cells_at(cx, i, j, kk)
                if i + j <= step - 1:
                    solid2[c[ok, 2], c[ok, 1], c[ok, 0]] = False
                elif i + j == step:
                    tag(cx, c, ok, i, j, 1)
                paint(c[ok], cx[ok])
    # fills: a wedge cell of a concave edge is added only while its supports stand, the cell
    # toward the wall (i - 1) and the cell toward the floor (j - 1): where a chamfer of a convex
    # edge has cut the wall or the floor away (a block corner), the fillet is trimmed instead of
    # floating in front of the chamfer
    kept = {}
    for d_ in range(step):
        for i in range(d_ + 1):
            j = d_ - i
            for kk in (0, 1):
                c, ok = cells_at(cc, i, j, kk)
                cw, okw = cells_at(cc, i - 1, j, kk)
                cf, okf = cells_at(cc, i, j - 1, kk)
                ok &= okw & okf
                sup = np.zeros(len(cc), bool)
                sup[ok] = solid2[cw[ok, 2], cw[ok, 1], cw[ok, 0]] & solid2[cf[ok, 2], cf[ok, 1], cf[ok, 0]]
                solid2[c[sup, 2], c[sup, 1], c[sup, 0]] = True
                tag(cc, c, sup, i, j, -1)
                paint(c[sup], cc[sup])
                kept[(i, j, kk)] = sup
    # the tetrahedra past the ends of the fillets (two fillets of different axes meeting at a
    # corner would touch along an edge only)
    for i in range(step):
        for j in range(step - i):
            for kk in range(-(step - i - j), 0):
                for end, base in ((kk, 0), (1 - kk, 1)):
                    sel = cc[kept[(i, j, base)]]
                    if len(sel) == 0:
                        continue
                    c, ok = cells_at(sel, i, j, end)
                    cover[k[sel[ok]], c[ok, 2], c[ok, 1], c[ok, 0]] = True
    cushion = (cover.sum(0) >= 2) & ~solid2
    stats["chamfer_cushion_cells"] = int(cushion.sum())
    solid2 |= cushion
    cut2[:, cushion] = -7  # the mitre: its own staircases
    stats["checkerboards_filled"] = _fill_checkerboards(solid2)
    return solid2, color2, mat2, cut2


def chamfer_mesh(solid: np.ndarray, color: np.ndarray | None = None, step: int = 1, min_run: int = 2,
                 pal: np.ndarray | None = None, material: np.ndarray | None = None, jitter: float = 0.0,
                 ao: bool = True, stats: dict | None = None) -> Mesh:
    """Chamfer mesh of the occupancy ``solid``: cubes whose step edges (the rule of
    ``mesh.step_edge_weights(mode="steps", min_run, step)``: a 90 degree edge, convex or concave,
    where either face extends exactly ``step`` voxels, in a run of ``min_run`` unit edges; here
    also both faces at least ``step / 2`` voxels) are cut at 45 degrees ``step / 2`` voxels into
    each face by one planar quad per unit edge. Built on the doubled grid: each wedge is carved
    or filled as a staircase of ``step`` half-voxels and the owner's staircase cells (the face
    with extent ``step``, the earlier of :data:`DIRS` when both) are fitted as 1:1 treads whose
    plane passes through the wedge's outer corners (the cut shift), so the chamfer quads take the
    tread's colour. Arguments and output as :func:`slope_mesh`."""
    if step < 1 or min_run < 1:
        raise ValueError(f"step and min_run must be >= 1 (got {step}, {min_run})")
    solid = np.asarray(solid, bool)
    stats = stats if stats is not None else {}
    shape = solid.shape
    if color is None:
        color = np.zeros(shape, np.uint8)
    if material is None:
        material = solid.astype(np.uint8)
    solid2, color2, mat2, cut2 = _chamfer_grid(solid, step, min_run, color, material, stats)
    # every half-voxel step of the doubled grid comes from a wedge (the model's own steps are two
    # half-voxels or more), so each one is fitted (min_run 1); ``min_run`` is the edge run rule
    res = _run(solid2, 1, 1, ao, stats, cut=cut2) if solid2.any() else None
    if res is None:
        return Mesh(np.zeros((0, 3), np.float32), np.zeros((0, 4), np.int32), np.zeros((0, 4), np.uint8),
                    np.zeros((0,), np.uint8), np.zeros((0, 4), np.float32), np.zeros((0,), bool))
    pos, quads, corner_ao, vox2, sheet = res
    pos = pos / 2.0
    if pal is None:
        pal = np.full((256, 3), 200, np.uint8)
    ci = color2[vox2[:, 2], vox2[:, 1], vox2[:, 0]]
    fine = vox2 // 2
    f = 1.0 + (voxel_hash01(fine[:, 0], fine[:, 1], fine[:, 2]) * 2.0 - 1.0) * jitter
    rgb = np.clip(pal[ci].astype(np.float32) * f[:, None] + 0.5, 0, 255).astype(np.uint8)
    mats = mat2[vox2[:, 2], vox2[:, 1], vox2[:, 0]].astype(np.uint8)
    return Mesh(pos.astype(np.float32), quads, _pack_face_colors(rgb, corner_ao), mats, corner_ao,
                np.zeros(len(quads), bool))


def chamfer_mesh_part(part, palette, step: int = 1, min_run: int = 2, jitter: float = 0.0,
                      stats: dict | None = None) -> Mesh:
    """:func:`chamfer_mesh` of a ``Part``: positions in metres of its local space."""
    from . import materials as M
    material = part.material
    solid = material != M.MAT_AIR
    m = chamfer_mesh(solid, part.color, step, min_run, pal=_palette_u8(palette), material=material,
                     jitter=jitter, stats=stats)
    if m.n_faces == 0:
        return _empty(part)
    m.positions = _to_metres(part, m.positions)
    return m
