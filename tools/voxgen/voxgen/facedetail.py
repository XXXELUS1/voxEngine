"""Fine-detail pass of a stylized head: human-sized features placed at 5 mm on landmarks.

Why: the owner rejected both the big-eyed ``toon`` head (creepy) and the flat ``normal`` one. The
cure is a *mix of scales*: the form (cranium, jaw, nose wedge, ears, hair mass) stays coarse,
deliberate 1 cm blocks, and the face is drawn on top with single 5 mm voxels, the way a voxel
artist draws eyes and lips with the smallest brush. This module is that brush, shared by
:mod:`voxgen.hybridhead` (form from the MakeHuman mesh) and the ``real`` preset of
:mod:`voxgen.toonhead` (designed form), so both candidates have the same detail vocabulary.

Vocabulary (fine voxels, all mirror-symmetric about x = 0; paired features are designed on +x):
  eyes    recessed one voxel: white 5 w x 3 h, iris 3 x 3 (``iris_in`` columns toward the nose),
          one highlight voxel (top row, screen-left column of both irises: colour only, so the
          material field stays symmetric), a one-voxel dark lid line on the skin right above;
  brows   one voxel tall, ``brow_len`` long, the outer part one row higher (slight angle), painted
          in the dark hair tone;
  nose    the form's nose kept; a one-voxel bridge ridge (the two centre columns protrude one
          voxel from under the eyes down to where the form's own nose starts to protrude) and
          two dark nostril voxels at the base;
  mouth   a dark line one voxel high, corner voxels one row up (+1) / down (-1), one lip row above
          (thinner: narrower) and one below in the lip tone;
  ears    the form's ears plus a 1 x 2 darker spot on the outer face (the hole);
  cheeks  optional flat ruddy band under the eyes (no noise).

Relief (``FaceDetailParams.relief``): the same features as geometry, one voxel deep, never more
(a two-voxel step reads as a blob at 3-5 mm): brows protrude one voxel (hair material in the brow
tone, so a razor can take them off), a skin ridge one row above the lid line overhangs the eye
(the dark line sits under it), both lip rows protrude one voxel so the mouth line is a recess
between them, the nostrils are carved two voxels deep. The mask primitives (:func:`protrusion`,
:func:`recess`) are shared with :mod:`voxgen.head_from_mesh`, which runs its relief on the solid
before the tissues are layered.

Coordinates: arrays are ``(nz, ny, nx)`` with an even ``nx``; ``X = column - (nx - 1) / 2``
(half-integers, exact mirror), rows ``y`` are plain indices, the face looks toward -z (index 0
is the front). Landmarks come from the morphed mesh (hybrid) or the design grid (toonhead).
Decisions: the pass runs *before* the tissues are layered (the eye recess exposes nothing but
eye); it writes only onto skin voxels of the front/side face and never changes hair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import materials as M
from .body import shift


@dataclass
class FaceLandmarks:
    """Fine-grid landmarks (voxel units; ``x`` values are |X| half-integers, see module doc)."""
    eye_x: float          # |X| of the eye centres
    eye_y: int            # row of the eye centre (the white is eye_y - 1 .. eye_y + 1)
    brow_y: int           # row of the brow's inner part
    nose_top_y: int       # first row of the bridge ridge (just under the eyes)
    nose_base_y: int      # row of the nostrils (the nose's bottom row)
    nostril_x: float      # |X| of the nostril voxels
    mouth_y: int          # row of the mouth line
    mouth_half_w: float   # |X| of the mouth corners
    ear_y: int            # row of the ear hole
    ear_z: int            # z index of the ear hole
    cheek_y: int          # bottom row of the ruddy band


@dataclass
class FaceDetailParams:
    iris_in: int = 0          # iris columns shifted toward the nose (0 = centred, 1 = slight)
    highlight_side: int = -1  # screen side of the highlight column (-1 = viewer's left)
    lid_line: bool = True
    brow_len: int = 8         # columns (inner end at eye_x - 3)
    brow_tilt: int = 1        # outer columns this many rows higher (+) / lower (-)
    brow_outer: int = 3       # how many outer columns are tilted
    nose_bridge: bool = True
    nostrils: bool = True
    mouth_corners: int = 1    # +1 corners up, 0 flat, -1 down
    lips: bool = True
    ear_spot: bool = True
    cheeks: bool = False
    relief: bool = False      # features as one-voxel geometry too (brows, lid ridge, lips, nostril holes)


def x_coords(nx: int) -> np.ndarray:
    """``X`` of every column: half-integers mirrored about 0 (``nx`` must be even)."""
    if nx % 2:
        raise ValueError("facedetail needs an even number of columns (mirror between two columns)")
    return np.arange(nx, dtype=np.float64) - (nx - 1) / 2


def _cols(X: np.ndarray, x0: float, x1: float) -> np.ndarray:
    """Mirrored column mask (nx,): x0 <= |X| <= x1."""
    return (np.abs(X) >= x0 - 1e-6) & (np.abs(X) <= x1 + 1e-6)


def front_face(solid: np.ndarray) -> np.ndarray:
    """Front-most solid voxel of every (y, x) column (looking from -z)."""
    behind = np.cumsum(solid, axis=0) - solid
    return solid & (behind == 0)


def _rect(shape, cols: np.ndarray, y0: int, y1: int) -> np.ndarray:
    """Full-shape mask: the columns ``cols`` (nx,) on rows y0 .. y1 (inclusive, clipped)."""
    nz, ny, nx = shape
    rows = np.zeros(ny, bool)
    rows[max(0, y0):min(ny, y1 + 1)] = True
    return np.broadcast_to(rows[None, :, None] & cols[None, None, :], shape).copy()


def protrusion(base: np.ndarray, solid: np.ndarray, dz: int = -1) -> np.ndarray:
    """Voxels to add so every voxel of ``base`` gets one more in front of it (``dz`` = -1 toward
    the viewer; +-1 along z only): the shifted copy, where there is air. One voxel, never a stack."""
    return shift(base, 0, 0, dz) & ~solid


def recess(base: np.ndarray, solid: np.ndarray, depth: int = 1, direction: tuple[int, int, int] = (0, 0, 1)) -> np.ndarray:
    """Voxels to carve: ``base`` and the ``depth - 1`` solid voxels behind each along ``direction``
    (index steps (dx, dy, dz); the default digs away from the viewer). Stops at air."""
    out = base & solid
    cur = out
    for _ in range(max(0, depth - 1)):
        cur = shift(cur, *direction) & solid
        out |= cur
    return out


def hole_walls(inner: np.ndarray, skin: np.ndarray) -> np.ndarray:
    """Skin voxels 6-adjacent to the inner (deeper) voxels of a carved hole: its floor and side
    walls, never the rim around the opening on the outer face."""
    walls = np.zeros(inner.shape, bool)
    for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        walls |= shift(inner, *d)
    return walls & skin


def apply_face_details(mat: np.ndarray, col: np.ndarray, lm: FaceLandmarks, colors: dict[str, int],
                       p: FaceDetailParams | None = None) -> dict[str, int]:
    """Draw the fine features in place on ``mat``/``col`` (uint8, (nz, ny, nx)).

    ``colors`` maps the roles ``eye_white``, ``iris``, ``highlight``, ``line``, ``lips``, ``brow``,
    ``skin_shadow``, ``skin_ruddy`` to palette indices. Returns voxel counts per feature.
    """
    p = p or FaceDetailParams()
    nz, ny, nx = mat.shape
    X = x_coords(nx)
    stats: dict[str, int] = {}

    def skin_front() -> np.ndarray:
        return front_face(mat != M.MAT_AIR) & (mat == M.MAT_SKIN)

    # ---- eyes: carve the front skin voxel, put the white one voxel behind (recess follows
    # the local surface, so the eye is always exactly one voxel deep)
    ex, ey = float(lm.eye_x), int(lm.eye_y)
    white_cols = _cols(X, ex - 2, ex + 2)
    opening = skin_front() & _rect(mat.shape, white_cols, ey - 1, ey + 1)
    eye = shift(opening, 0, 0, 1)
    mat[opening] = M.MAT_AIR
    col[opening] = 0
    mat[eye] = M.MAT_EYE
    col[eye] = colors["eye_white"]
    iris_cols = _cols(X, ex - 1 - p.iris_in, ex + 1 - p.iris_in)
    iris = eye & _rect(mat.shape, iris_cols, ey - 1, ey + 1)
    col[iris] = colors["iris"]
    # highlight: the top iris row, the same screen-side column of both irises (one light source)
    lo, hi = ex - 1 - p.iris_in, ex + 1 - p.iris_in          # |X| range of the iris columns
    if p.highlight_side < 0:
        hl_col = (np.abs(X - lo) < 1e-6) | (np.abs(X + hi) < 1e-6)
    else:
        hl_col = (np.abs(X - hi) < 1e-6) | (np.abs(X + lo) < 1e-6)
    hl = iris & _rect(mat.shape, hl_col, ey + 1, ey + 1)
    col[hl] = colors["highlight"]
    stats["eye"] = int(eye.sum())
    if p.lid_line:
        lid = skin_front() & _rect(mat.shape, white_cols, ey + 2, ey + 2)
        col[lid] = colors["line"]
        stats["lid"] = int(lid.sum())
    if p.relief:
        # the upper lid: a skin ridge one row above the lid line, overhanging the eye
        base = skin_front() & _rect(mat.shape, white_cols, ey + 3, ey + 3)
        ridge = protrusion(base, mat != M.MAT_AIR)
        mat[ridge] = M.MAT_SKIN
        col[ridge] = colors["skin"]
        stats["lid_ridge"] = int(ridge.sum())

    # ---- brows: one row, inner part flat, outer part tilted
    x0 = ex - 3
    inner_n = max(1, p.brow_len - p.brow_outer)
    brow = _rect(mat.shape, _cols(X, x0, x0 + inner_n - 1), lm.brow_y, lm.brow_y)
    brow |= _rect(mat.shape, _cols(X, x0 + inner_n, x0 + p.brow_len - 1), lm.brow_y + p.brow_tilt, lm.brow_y + p.brow_tilt)
    brow &= skin_front()
    col[brow] = colors["brow"]
    stats["brow"] = int(brow.sum())
    if p.relief:
        hair = protrusion(brow, mat != M.MAT_AIR)
        mat[hair] = M.MAT_HAIR
        col[hair] = colors["brow"]
        stats["brow_relief"] = int(hair.sum())

    # ---- nose: bridge ridge (the two centre columns protrude one voxel) and two nostril dots
    if p.nose_bridge:
        # the ridge runs from under the eyes down to the nose but stays on the bridge: rows whose
        # centre columns already protrude more than one voxel beyond the root (the form's own
        # nose tip) get no extra voxel, so the tip never grows into a bar
        ridge_cols = _cols(X, 0, 0.5)
        front = front_face(mat != M.MAT_AIR)
        zrow = np.where(front[:, :, ridge_cols].any(axis=2), np.arange(nz)[:, None], nz).min(axis=0)  # (ny,)
        z_root = int(zrow[min(ny - 1, lm.nose_top_y + 1)])
        flat_rows = np.zeros(ny, bool)
        flat_rows[max(0, lm.nose_base_y + 2):min(ny, lm.nose_top_y + 1)] = True
        flat_rows &= zrow >= z_root - 1
        base = skin_front() & ridge_cols[None, None, :] & flat_rows[None, :, None]
        ridge = shift(base, 0, 0, -1) & (mat == M.MAT_AIR)
        mat[ridge] = M.MAT_SKIN
        col[ridge] = colors["skin"]
        stats["bridge"] = int(ridge.sum())
    if p.nostrils:
        nx_ = float(lm.nostril_x)
        nostril = skin_front() & _rect(mat.shape, _cols(X, nx_, nx_), lm.nose_base_y, lm.nose_base_y)
        if p.relief:
            # two voxels deep into the nose; the hole's floor and walls take the dark tone
            hole = recess(nostril, mat != M.MAT_AIR, depth=2)
            mat[hole] = M.MAT_AIR
            col[hole] = 0
            nostril = hole_walls(hole & ~nostril, mat == M.MAT_SKIN)
        col[nostril] = colors["line"]
        stats["nostril"] = int(nostril.sum())

    # ---- mouth: line, corners, lips
    mw, my = float(lm.mouth_half_w), int(lm.mouth_y)
    line = _rect(mat.shape, _cols(X, 0, mw - 1), my, my)
    line |= _rect(mat.shape, _cols(X, mw, mw), my + p.mouth_corners, my + p.mouth_corners)
    line &= skin_front()
    col[line] = colors["line"]
    stats["mouth"] = int(line.sum())
    if p.lips:
        upper = _rect(mat.shape, _cols(X, 0, mw - 2), my + 1, my + 1)
        lower = _rect(mat.shape, _cols(X, 0, mw - 1), my - 1, my - 1)
        lips = (upper | lower) & skin_front() & (col != colors["line"])
        col[lips] = colors["lips"]
        stats["lips"] = int(lips.sum())
        if p.relief:
            # both lip rows come forward one voxel; the mouth line stays as a recess between them
            out = protrusion(lips, mat != M.MAT_AIR)
            mat[out] = M.MAT_SKIN
            col[out] = colors["lips"]
            stats["lips_relief"] = int(out.sum())

    # ---- ears: a 1 x 2 darker spot on the outer (side) face at the ear hole
    if p.ear_spot:
        solid = mat != M.MAT_AIR
        side_face = solid & ~shift(solid, -1, 0, 0) & (X[None, None, :] > 0)  # +x side: air at x + 1
        side_face |= solid & ~shift(solid, 1, 0, 0) & (X[None, None, :] < 0)
        ez = int(lm.ear_z)
        zone = np.zeros(mat.shape, bool)
        zone[max(0, ez):min(nz, ez + 1), max(0, lm.ear_y - 1):min(ny, lm.ear_y + 1), :] = True
        cand = side_face & zone & (mat == M.MAT_SKIN)
        # outermost column per side
        xs = np.nonzero(cand.any(axis=(0, 1)))[0]
        spot = np.zeros(mat.shape, bool)
        if xs.size:
            outer = int(np.abs(X[xs]).max())
            spot = cand & _rect(mat.shape, _cols(X, outer - 0.5, outer + 0.5), lm.ear_y - 1, lm.ear_y)
        col[spot] = colors["skin_shadow"]
        stats["ear"] = int(spot.sum())

    # ---- cheeks: flat ruddy band under the eyes
    if p.cheeks:
        cheek = skin_front() & _rect(mat.shape, _cols(X, ex - 1, ex + 2), lm.cheek_y, lm.cheek_y + 1)
        cheek &= col == colors["skin"]
        col[cheek] = colors["skin_ruddy"]
        stats["cheek"] = int(cheek.sum())
    return stats
