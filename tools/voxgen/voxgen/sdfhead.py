"""The head FORM as a smooth implicit surface, voxelised on a coarse grid (2026-10-17).

Why: the owner's bald-head turnaround (``docs/reference/bald-head-turnaround.png``) is a smoothly
sculpted head cut into coarse voxels (about 1/15 of the head width) with every step edge chamfered by
half a voxel. Drawing that form as 2-D relief maps (profile x width, ``barber_maps.SKULL``) gave a
box or a terrace; so the form is now built the way the reference was made: smooth primitives
(a superellipsoid cranium, a tapered jaw block with a flat underside, a nose wedge, a rounded neck),
blended, then sampled at the centres of coarse cells (``COARSE`` fine voxels per cell, anchored at
x = 0, y = 0 (the chin bottom) and z = 0 (the face plane)) and written as full cells of fine voxels.
Rendering with ``--part-styles skull=facet --facet-step COARSE`` chamfers the coarse step edges.

Measurements are in the reference's own units ``u`` (the head is 14.6 u wide, 18.5 u tall, 15.2 u
deep without the nose, see ``REF_SILHOUETTE`` for the row-by-row widths read off the sheet with
``scripts``-free numpy thresholding); ``U_PER_u`` converts to the map engine's units (U = 20 mm):
1 u = 0.75 U = 15 mm = 3 fine voxels, which keeps the head 22 cm wide, the size the barber maps assume.

Frame conventions as in ``sheethead``: +x viewer's right, +y up, the face looks toward -z, the cheek
plane is z = 0, the chin's bottom is y = 0.
"""
from __future__ import annotations

import numpy as np

from .blockhead import FINE, Frame

#: reference units per map unit: 1 u = 0.75 U (15 mm)
U_PER_u = 0.75
#: fine voxels per coarse cell (one reference voxel)
COARSE = 3   # 3 = the sheet's 15 mm cells; 1 = the form sampled at the fine 5 mm voxels
import os as _os
COARSE = int(_os.environ.get("VOXGEN_COARSE", COARSE))

#: row-by-row half-widths of the reference silhouettes (u), rows of 1 u from the crown down:
#: front (head without ears where the ears hide it), profile depth (z from the cheek plane, back),
#: used by :func:`compare_silhouette` to score a build.
REF_FRONT_HALF_WIDTH = [5.4, 6.05, 6.1, 7.1, 7.2, 7.3, 7.3, 7.3, 7.3, 7.3, 7.3, 7.3, 7.3, 7.1, 6.8, 6.8, 6.0, 5.95, 4.95]
REF_PROFILE_FRONT_BACK = [(2.5, 12.1), (1.2, 13.2), (0.8, 13.9), (0.2, 14.5), (0.1, 14.6), (0.0, 15.0), (0.0, 15.2),
                          (-0.2, 15.2), (-0.2, 15.2), (-0.1, 15.2), (-0.6, 15.2), (-1.3, 15.1), (-1.3, 15.0),
                          (-1.2, 14.4), (-0.1, 13.7), (0.2, 13.4), (0.1, 12.9), (0.1, 12.9), (0.1, 13.0)]
HEAD_TOP_u = 18.6


def _lerp_table(y, pts):
    xs = np.array([p[0] for p in pts], float)
    vs = np.array([p[1] for p in pts], float)
    return np.interp(y, xs, vs)


def _rounded_rect(px, pz, hx, hz, r):
    """Signed distance of a 2-D rounded rectangle centred at the origin (half sizes hx, hz, corner r)."""
    qx = np.abs(px) - (hx - r)
    qz = np.abs(pz) - (hz - r)
    outside = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qz, 0) ** 2)
    inside = np.minimum(np.maximum(qx, qz), 0)
    return outside + inside - r


def _smin(a, b, k):
    """Polynomial smooth minimum (blend width k)."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b + (a - b) * h - k * h * (1 - h)


DEFAULT = dict(
    cranium=dict(centre=(0.0, 11.5, 7.3), semi=(7.3, 7.6, 7.7), exps=(3.5, 2.4, 2.6)),   # fitted to the sheet rows 0..7
    # the lower face in plan: a narrow front plane (the eyes / mouth / chin) that widens at 45-ish
    # degrees toward the sides over ``jaw_flare`` u of depth (the faceted cheek of the sheet's 3/4
    # views); half-widths by y (u): front plane and sides (the mandible seen from the front)
    jaw_front=[(0.0, 2.5), (1.0, 3.0), (2.5, 3.8), (4.5, 4.5), (6.5, 4.8), (9.0, 4.9)],
    jaw_side=[(0.0, 4.9), (1.0, 5.2), (2.5, 6.0), (4.5, 6.8), (6.5, 7.1), (9.0, 7.3)],
    jaw_flare=4.0,
    jaw_back=[(0.0, 13.0), (3.0, 13.0), (8.0, 15.0)],     # z of the jaw block's back by y: flat underside, no arc
    jaw_top=9.0, blend=1.2,
    nose=dict(bridge_y=10.0, tip_y=6.5, base_y=5.5, tip_z=-1.3, bridge_hw=0.9, base_hw=1.6),
    neck=dict(hx=3.9, z=(5.5, 13.0), y=(-5.0, 1.5), round=1.5),
)


def head_inside(x, y, z, p=DEFAULT) -> np.ndarray:
    """True where the reference-unit point (x, y, z) is inside the head form (skull + neck + nose)."""
    cx, cy, cz = p["cranium"]["centre"]
    a, b, c = p["cranium"]["semi"]
    ex, ey, ez = p["cranium"]["exps"]
    s = (np.abs(x - cx) / a) ** ex + (np.abs(y - cy) / b) ** ey + (np.abs(z - cz) / c) ** ez
    d_cranium = (np.cbrt(s) - 1.0) * 5.0                     # distance-like near the surface
    # jaw block: a narrow front plane flaring to the sides over jaw_flare u of depth, flat bottom
    # at y 0 (no arc into the neck), back by y; the back corners rounded by 1.5 u
    wf = _lerp_table(y, p["jaw_front"])
    ws = _lerp_table(y, p["jaw_side"])
    hw = wf + (ws - wf) * np.clip(z / p["jaw_flare"], 0, 1)
    zb = _lerp_table(y, p["jaw_back"])
    d_x = np.abs(x) - hw
    d_z = np.maximum(-z, z - zb)
    back_corner = np.sqrt(np.maximum(np.abs(x) - (ws - 1.5), 0) ** 2 + np.maximum(z - (zb - 1.5), 0) ** 2) - 1.5
    d_xz = np.maximum(np.maximum(d_x, d_z), back_corner)
    d_y = np.maximum(-y, y - p["jaw_top"])
    d_jaw = np.maximum(d_xz, d_y)
    d = _smin(d_cranium, d_jaw, p["blend"])
    # nose: a wedge from the bridge down to the tip, flat underside at base_y, widening to the base
    n = p["nose"]
    t = np.clip((n["bridge_y"] - y) / (n["bridge_y"] - n["tip_y"]), 0, 1)
    z_front = n["tip_z"] * t
    hwn = n["bridge_hw"] + (n["base_hw"] - n["bridge_hw"]) * np.clip((n["bridge_y"] - y) / (n["bridge_y"] - n["base_y"]), 0, 1)
    nose = (y >= n["base_y"]) & (y <= n["bridge_y"]) & (np.abs(x) <= hwn) & (z >= z_front) & (z <= 1.0)
    # neck: a rounded column
    q = p["neck"]
    d_neck = np.maximum(_rounded_rect(x, z - (q["z"][0] + q["z"][1]) / 2, q["hx"], (q["z"][1] - q["z"][0]) / 2, q["round"]),
                        np.maximum(q["y"][0] - y, y - q["y"][1]))
    return (d <= 0) | nose | (d_neck <= 0)


def voxelize(f: Frame, layer: dict) -> np.ndarray:
    """Fine mask of the form on frame ``f``: the implicit function sampled at the centres of coarse
    cells of ``COARSE`` fine voxels anchored at x = y = z = 0 (so the form is mirror-exact and the
    chin / face plane sit on cell boundaries); every fine voxel of a cell takes the cell's value."""
    params = {**DEFAULT, **layer.get("params", {})}
    cell = COARSE / f.fine                                   # cell size in U
    def centres(v):
        return (np.floor(v / cell) + 0.5) * cell
    X, Y, Z = centres(f.X), centres(f.Y), centres(f.Z)
    inside = head_inside(X / U_PER_u, Y / U_PER_u, Z / U_PER_u, params)
    return np.broadcast_to(inside, f.shape).copy()


def layer_extent(layer: dict):
    """((x0, x1), (y0, y1), (z0, z1)) in U the form can touch (for the frame)."""
    p = {**DEFAULT, **layer.get("params", {})}
    a, b, c = p["cranium"]["semi"]
    cx, cy, cz = p["cranium"]["centre"]
    q = p["neck"]
    x1 = max(a, max(w for _, w in p["jaw_side"]), q["hx"]) * U_PER_u
    y0 = min(q["y"][0], 0.0) * U_PER_u
    y1 = (cy + b) * U_PER_u
    z0 = min(cz - c, p["nose"]["tip_z"]) * U_PER_u
    z1 = max(cz + c, max(zz for _, zz in p["jaw_back"]), q["z"][1]) * U_PER_u
    return (-x1, x1), (y0, y1), (z0, z1)


def compare_silhouette(solid: np.ndarray, f: Frame) -> str:
    """Row-by-row comparison (1 u rows from the crown down) of the built form's front half-width and
    profile front/back against the reference numbers; returns a text table."""
    X = np.broadcast_to(f.X, f.shape)
    Y = np.broadcast_to(f.Y, f.shape)
    Z = np.broadcast_to(f.Z, f.shape)
    rows = []
    for i, (ref_w, (ref_f, ref_b)) in enumerate(zip(REF_FRONT_HALF_WIDTH, REF_PROFILE_FRONT_BACK)):
        y1 = (HEAD_TOP_u - i) * U_PER_u
        y0 = y1 - U_PER_u
        sel = solid & (Y >= y0) & (Y < y1)
        if not sel.any():
            rows.append(f"{i:2d}  ref w {ref_w:4.1f} z {ref_f:5.1f}..{ref_b:4.1f}   ours: empty")
            continue
        w = (np.abs(X[sel]).max() + 0.5 / f.fine) / U_PER_u
        zf = (Z[sel].min() - 0.5 / f.fine) / U_PER_u
        zb = (Z[sel].max() + 0.5 / f.fine) / U_PER_u
        rows.append(f"{i:2d}  ref w {ref_w:4.1f} z {ref_f:5.1f}..{ref_b:4.1f}   ours w {w:4.1f} z {zf:5.1f}..{zb:4.1f}"
                    f"   dw {w - ref_w:+4.1f} dzf {zf - ref_f:+4.1f} dzb {zb - ref_b:+4.1f}")
    return "\n".join(rows)
