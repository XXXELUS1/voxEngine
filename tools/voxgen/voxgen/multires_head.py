"""Multi-resolution head: the skin FORM at 5 mm, the DETAILS as their own 3 mm parts.

Why: the owner wants a mix of voxel sizes (big parts big, small details small), eyeballs that
are real spheres (so they can rotate later), lips and brows as their own things (animatable),
and fewer voxels. The ``.vxm`` format already allows every part its own ``voxel_size`` and
``origin``; the renderer draws every part at its origin and hides nothing, so the form is
*carved* wherever a detail part sits.

Parts (``build_multires_head``):
  ``head``   5 mm. The morphed MakeHuman head (``HeadParams.morphs``, the same as ``mid_v2``)
             voxelized with the eyeball helpers and flood-filled, cleaned (3x3x3 closing then
             opening), mirrored, largest component, quantized by the 26-neighbour count
             (:func:`voxgen.hybridhead.quantize_form`: no one-voxel bump or pit survives), the
             nostrils carved as one-voxel slots two voxels deep (dark walls); tissues by depth
             (:func:`voxgen.head_from_mesh.layer_tissues`), flat skin tint. Carved: eye sockets
             (every form voxel that holds an eyeball voxel centre, except the exposed skin shell
             outside the lid opening, so the lids overhang; plus the lid opening from the
             landmarks, two rows high, 25 mm wide, cut down to the eye's equator), the mouth
             (every form voxel that holds a lips voxel centre), the scalp (every form voxel that
             holds a hair voxel centre). Socket walls behind the lids are tinted dark.
  ``eye_l``, ``eye_r``  3 mm spheres of radius 12 mm, centred ``eye_forward`` (3 mm) in front of
             the eyeball landmarks, so the white reaches the lid opening of the 5 mm-thick lids
             instead of sitting in a dark hole; MAT_EYE; iris = disc of 5.5 mm on the front
             (-Z) with a 2.2 mm pupil and one highlight voxel on the +x side of the top row
             (viewer's left for both eyes: colour only). A short stalk of eye material grows
             from the back of the ball to the socket wall (the optic nerve) so the ball touches
             the form. Bones ``eye_l`` / ``eye_r`` (names only).
  ``lips``   3 mm, MAT_SKIN. The lip band of the 3 mm voxelization of the same mesh (so the lips
             keep the mesh's curvature; ``lip_half_height`` 6.5 mm = two lower-lip rows, one upper):
             both lips one voxel forward, the mouth line a recess between them, corners one row up;
             ``lips_depth`` voxels deep, then grown backward until they touch the carved form.
             Lip tone, line colour on the mouth line.
  ``brow_l``, ``brow_r``  3 mm, MAT_HAIR in the brow tone: one-voxel-thick strips lying on the
             carved form (the voxels right in front of the form's surface in a band that rises
             ``brow_tilt`` toward the temple), inner ends ``brow_gap`` apart; where the form steps
             back the strip is filled to the step so it stays one piece. Two parts, not one:
             ``voxgen check`` would call the second island of a one-part pair floating, and the
             brows will move separately anyway.
  ``ear_l``, ``ear_r``  3 mm, MAT_SKIN: the ears of the 3 mm voxelization (a 5 mm sheet is a flat
             plate glued to the skull), grown through the 3 mm skin toward the skull until they
             touch the form; the form loses the voxels under them.
  ``hair``   3 mm, MAT_HAIR, three flat tones. The short cut of ``head_from_mesh.hair_cap`` on
             the 3 mm voxelization (trimmed above the ears), plus the 3 mm skin under it that
             the 5 mm form does not fill, so the hair sits on the carved scalp without a gap.

Seams: a detail voxel never has its centre inside a form voxel (``meta["seams"][part]["overlap"]``
is 0: the carve rule and the check are the same rule), and every detail part has voxels whose
6-neighbour centres lie in the form (``contact`` > 0). Boxes of the two grids still interpenetrate
by up to half a form voxel where a 3 mm voxel straddles a 5 mm boundary; that is hidden inside
the other part. ``seam_margin`` (metres) widens the footprint before carving for a visible
groove around the lips and under the hair (default 0: embedded, no groove). The lids are
allowed to wrap the eyeball (their boxes overlap the ball's top and bottom): that is what lids
do, and the eyeball's visible part emerges from under them.

Local frame as the other heads: metres, Y up, face toward -Z, chin at y = 0, centred on x / z;
all parts share it (their origins differ). The form grid is x-centred with an even number of
columns and its rows are aligned so that the eye centre lies on a row boundary (the lid
opening is then symmetric about the eye). Everything is mirrored about x = 0 (material and
geometry; the highlight colour is the one exception).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from . import materials as M
from . import mesh as meshlib
from . import meshvox, mh
from .body import dilate, exposed, shift
from .checks import label_components
from .facedetail import front_face, hole_walls, protrusion, recess
from .head_from_mesh import (FLAT_ALIAS, HeadParams, box_dilate, box_erode, form_solid, hair_cap, hair_ear_zone, hair_floor,
                             layer_tissues, lip_band, make_palette, nearest_voxel)
from .hybridhead import quantize_form
from .model import Model, Part

INF = float("inf")


@dataclass
class MultiresParams:
    """Lengths in metres."""

    form_voxel: float = 0.005
    detail_voxel: float = 0.003
    head: HeadParams = field(default_factory=lambda: HeadParams.mid_v2(1.2))   # morphs, hair, tissues, lips
    eye_radius: float = 0.012
    iris_radius: float = 0.0055
    pupil_radius: float = 0.0022
    eye_open_h: float = 0.005          # half-height of the lid opening (one form row above and below the eye centre)
    eye_open_w: float = 0.0125         # half-width of the lid opening
    stalk_cols: int = 2                # the optic-nerve stalk is stalk_cols x stalk_cols columns around the eye axis
    lips_depth: int = 3                # detail voxels from the lips' front to their back
    brow_tilt: float = 0.003           # the brow's outer end is this much higher than the inner end
    brow_half_h: float = 0.003         # half-height of the brow strip
    brow_gap: float = 0.015            # distance between the brows' inner ends
    brow_out: float = 0.021            # outer end: this far outward of the eye centre
    seam_margin: float = 0.0           # footprint dilation (metres) before the form is carved for lips / hair
    hair_fill_reach: int = 3           # detail voxels of skin under the cap that may become hair (gap to the form)
    quantize_passes: int = 6
    bump_max: int = 11
    pit_min: int = 15
    ears_as_parts: bool = True         # the ears are 3 mm parts from the fine voxelization (a 5 mm sheet is a flat plate)
    eye_forward: float = 0.005         # eyeball centre this much in front of the landmark (the white comes out of the 5 mm lid shadow)
    lip_half_height: float = 0.0065    # lip band half-height: two lower-lip rows, one upper (lip_band)
    margin: float = 0.02               # air around the head in the grids


# ---------------------------------------------------------------------------- grid helpers

def contains(mask: np.ndarray, grid: meshvox.VoxelGrid, points: np.ndarray) -> np.ndarray:
    """For metre ``points`` (n, 3): True where the point lies in a True voxel of ``mask`` (on ``grid``)."""
    idx = grid.to_index(points)
    ok = grid.in_range(idx)
    out = np.zeros(len(points), bool)
    sel = idx[ok]
    out[ok] = mask[sel[:, 2], sel[:, 1], sel[:, 0]]
    return out


def resample(mask: np.ndarray, grid_from: meshvox.VoxelGrid, grid_to: meshvox.VoxelGrid) -> np.ndarray:
    """``mask`` seen from ``grid_to``: True where the centre of a ``grid_to`` voxel lies in a True voxel."""
    X, Y, Z = grid_to.centres()
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    return contains(mask, grid_from, pts).reshape(grid_to.shape)


def footprint(mask: np.ndarray, grid_from: meshvox.VoxelGrid, grid_to: meshvox.VoxelGrid) -> np.ndarray:
    """Voxels of ``grid_to`` that hold the centre of at least one True voxel of ``mask``."""
    zs, ys, xs = np.nonzero(mask)
    o = np.asarray(grid_from.origin)
    pts = o + (np.stack([xs, ys, zs], axis=1) + 0.5) * grid_from.voxel_size
    return grid_to.mark(pts)


def head_grid(human: mh.MakeHuman, lm: dict, hp: HeadParams, vs: float, margin: float, align_y: float | None = None,
              align_z: float | None = None) -> meshvox.VoxelGrid:
    """x-centred grid (even columns) around the head above the neck joint with room for the hair.
    ``align_y`` / ``align_z``: a metre coordinate that must fall on a voxel boundary."""
    body = human.body()
    cut_y = float(lm["neck_base"][1])
    hv = body.vertices[body.vertices[:, 1] > cut_y - 0.01]
    room = hp.hair_thickness + 0.004 if hp.hair else 0.0
    half_x = max(abs(hv[:, 0].min()), abs(hv[:, 0].max())) + margin + room
    nx = int(math.ceil(half_x / vs)) * 2
    y0, y1 = cut_y, lm["head_top"] + margin + room
    if align_y is not None:
        y0 = align_y - math.ceil((align_y - y0) / vs) * vs
    ny = int(math.ceil((y1 - y0) / vs))
    z0, z1 = hv[:, 2].min() - margin - room, hv[:, 2].max() + margin + room
    if align_z is not None:
        z0 = align_z - math.ceil((align_z - z0) / vs) * vs
    nz = int(math.ceil((z1 - z0) / vs))
    return meshvox.VoxelGrid((-nx * vs / 2, y0, z0), vs, (nx, ny, nz))


def mirror_union(mask: np.ndarray) -> np.ndarray:
    return mask | mask[:, :, ::-1]


def backfill_z(front: np.ndarray, stop: np.ndarray, reach: int, step=(0, 0, 1)) -> np.ndarray:
    """``front`` plus the voxels behind it (``step``, default +z) column by column until ``stop``
    (at most ``reach``)."""
    out = front.copy()
    cur = front
    for _ in range(reach):
        cur = shift(cur, *step) & ~stop & ~out
        if not cur.any():
            break
        out |= cur
    return out


# ---------------------------------------------------------------------------- the form (5 mm)

def build_form(human: mh.MakeHuman, lm: dict, p: MultiresParams) -> tuple[meshvox.VoxelGrid, np.ndarray, dict]:
    """The head solid on the form grid: voxelized with the eyeballs and filled, cleaned, mirrored,
    largest component, quantized (no one-voxel terraces)."""
    hp = p.head
    grid = head_grid(human, lm, hp, p.form_voxel, p.margin, align_y=float(lm["eye_y"]))
    cut_y = float(lm["neck_base"][1])
    body_c = meshvox.clip_faces(human.body(), lo=(-INF, cut_y, -INF))
    eyes = (human.helper("l_eye"), human.helper("r_eye"))
    solid, info = form_solid(grid, body_c, eyes, 1, True)
    solid = mirror_union(solid)
    labels, n = label_components(solid)
    if n > 1:
        solid = labels == 1
    solid, qinfo = quantize_form(solid, p.quantize_passes, p.bump_max, p.pit_min)
    info = dict(info, components=int(n), **qinfo)
    return grid, solid, info


def carve_nostrils(solid: np.ndarray, grid: meshvox.VoxelGrid, lm: dict) -> tuple[np.ndarray, np.ndarray]:
    """One-voxel nostril slots two voxels deep on the bottom-front edge of the nose (the edge
    voxel nearest to 9 mm off-centre, 8 mm under and 6 mm behind the tip). Returns the carved
    solid and the inner hole voxels (their walls get the dark tone)."""
    X, Y, Z = grid.centres()
    solid = solid.copy()
    tip = lm["nose_tip"]
    nose = solid & (Y > lm["mouth"][1] + 0.006) & (Y < lm["eye_y"] - 0.012) & (np.abs(X) < 0.02) & (Z < tip[2] + 0.03)
    edge = nose & ~shift(solid, 0, 1, 0) & ~shift(solid, 0, 0, 1)
    inner = np.zeros(solid.shape, bool)
    for sx in (-1, 1):
        idx = nearest_voxel(edge, X, Y, Z, (sx * 0.009, tip[1] - 0.008, tip[2] + 0.006))
        if idx is None:
            continue
        base = np.zeros(solid.shape, bool)
        base[idx] = True
        hole = recess(base, solid, depth=2, direction=(0, 0, 1))
        solid &= ~hole
        inner |= hole & ~base
    return solid, inner


# ---------------------------------------------------------------------------- detail parts (3 mm)

def eyeball(centre, p: MultiresParams, C) -> tuple[meshvox.VoxelGrid, np.ndarray, np.ndarray, np.ndarray]:
    """Grid, material and colour of one eyeball; the fourth value is the ball mask."""
    vs = p.detail_voxel
    n = 2 * int(math.ceil(p.eye_radius / vs)) + 2
    grid = meshvox.VoxelGrid(tuple(float(c) - n * vs / 2 for c in centre), vs, (n, n, n))
    X, Y, Z = grid.centres()
    dx, dy, dz = X - centre[0], Y - centre[1], Z - centre[2]
    ball = dx * dx + dy * dy + dz * dz <= p.eye_radius ** 2
    mat = np.where(ball, M.MAT_EYE, M.MAT_AIR).astype(np.uint8)
    col = np.where(ball, C("eye_white"), 0).astype(np.uint8)
    front = front_face(ball) & (dz < 0)
    r2 = dx * dx + dy * dy
    iris = front & (r2 <= p.iris_radius ** 2)
    col[iris] = C("iris")
    col[front & (r2 <= p.pupil_radius ** 2)] = C("pupil")
    zs, ys, xs = np.nonzero(iris)
    if zs.size:
        top = ys == ys.max()
        k = int(np.argmax(np.where(top, xs, -1)))          # top row, +x end: viewer's left
        col[zs[k], ys[k], xs[k]] = C("highlight")
    return grid, mat, col, ball


def lips_shell(fine_solid: np.ndarray, grid: meshvox.VoxelGrid, lm: dict, p: MultiresParams, C) \
        -> tuple[np.ndarray, np.ndarray]:
    """Lips on the fine grid: the band's front voxels, both lip rows one voxel forward, the line
    row left as a recess, ``lips_depth`` deep. Returns (mask, colour)."""
    X, Y, Z = grid.centres()
    vs = grid.voxel_size
    mouth_y = float(lm["mouth"][1])
    band = exposed(fine_solid) & front_face(fine_solid) & lip_band(Y, mouth_y, p.lip_half_height, vs) \
        & (np.abs(X) <= lm["mouth_half_width"] - 0.004)
    if band.any():
        band &= Z <= Z[band].min() + 0.016
    line = band & (np.abs(Y - mouth_y) <= vs * 0.5)
    front = protrusion(band & ~line, fine_solid)
    shell = band | front
    col = np.zeros(shell.shape, np.uint8)
    col[shell] = C("lips")
    col[line] = C("lip_line")
    if line.any():
        ax = np.abs(X[line])
        ends = line & (np.abs(X) >= ax.max() - 1e-6)        # mouth corners: one row up, on the lip row
        col[ends] = C("lips")
        up = np.zeros(line.shape, bool)
        for dz in range(-2, 3):
            up |= shift(ends, 0, 1, dz)
        col[front_face(front & up)] = C("lip_line")
    slab = shell.copy()
    for k in range(1, p.lips_depth):
        slab |= shift(shell, 0, 0, k)
    col[slab & (col == 0)] = C("lips")
    return slab, col


def brow_strips(form_f: np.ndarray, grid: meshvox.VoxelGrid, lm: dict, p: MultiresParams, blocked: np.ndarray) \
        -> np.ndarray:
    """One-voxel-thick strips lying on the (carved) form: the voxels right in front of its surface
    in the brow band; filled to the steps so each brow is one 6-connected piece."""
    X, Y, Z = grid.centres()
    hug = ~form_f & shift(form_f, 0, 0, -1) & ~blocked
    ax = np.abs(X)
    x_in, x_out = p.brow_gap * 0.5, lm["eye_x"] + p.brow_out
    t = np.clip((ax - x_in) / max(x_out - x_in, 1e-6), 0.0, 1.0)
    yc = lm["eye_y"] + 0.015 + p.brow_tilt * t
    band = (ax >= x_in) & (ax <= x_out) & (np.abs(Y - yc) <= p.brow_half_h) & (Z < lm["eye_z"] + 0.02)
    front = front_face(hug & band)
    nz = front.shape[0]
    zf = np.where(front.any(axis=0), np.argmax(front, axis=0), nz)              # (ny, nx) front z per column
    zmin = np.minimum(np.minimum(zf, shift(zf[None], 1, 0, 0, fill=nz)[0]), shift(zf[None], -1, 0, 0, fill=nz)[0])
    zmin = np.where(zf < nz, zmin, nz)
    start = np.zeros(front.shape, bool)
    yy, xx = np.nonzero(zmin < nz)
    start[zmin[yy, xx], yy, xx] = True
    return backfill_z(start, form_f | blocked, 8)


def hair_part_mask(cap: np.ndarray, fine_solid: np.ndarray, form_f: np.ndarray, grid: meshvox.VoxelGrid, lm: dict,
                   p: MultiresParams) -> np.ndarray:
    """The cap plus the fine skin under it (within ``hair_fill_reach`` voxels) that the carved
    form does not fill, above the hair floor and clear of the ears."""
    X, Y, Z = grid.centres()
    hp = p.head
    near = cap
    for _ in range(p.hair_fill_reach):
        near = box_dilate(near)
    ear = hair_ear_zone(X, Y, Z, lm, hp)
    for _ in range(2):
        ear = box_dilate(ear)
    ear |= (np.abs(X) > lm["ear_inner_x"] - 0.01) & (Y < lm["ear_y"][1] + 0.02) \
        & (Z > lm["ear_z"][0] - 0.012) & (Z < lm["ear_z"][1] + 0.012)
    fill = fine_solid & near & (Y >= hair_floor(X, Z, lm, hp)) & ~ear
    return (cap | fill) & ~form_f


def ear_box(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, lm: dict, grow: float) -> np.ndarray:
    """The ear region (both sides): outward of ``ear_inner_x``, the ear's y / z range grown by ``grow``."""
    return (np.abs(X) > lm["ear_inner_x"] - grow) & (Y > lm["ear_y"][0] - grow) & (Y < lm["ear_y"][1] + grow) \
        & (Z > lm["ear_z"][0] - grow) & (Z < lm["ear_z"][1] + grow)


def crop_part(name: str, grid: meshvox.VoxelGrid, mat: np.ndarray, col: np.ndarray, offset, bone: str,
              symmetric: bool, budget: int | None = None, pad: int = 0) -> Part:
    """A part from full-grid fields, cropped to the solid's bounding box (kept x-centred when
    ``symmetric``), origin shifted into the local frame."""
    zs, ys, xs = np.nonzero(mat)
    if zs.size == 0:
        raise ValueError(f"part {name!r} is empty")
    nx = grid.dims[0]
    x0, x1 = max(0, int(xs.min()) - pad), min(nx - 1, int(xs.max()) + pad)
    if symmetric:
        x0 = min(x0, nx - 1 - x1)
        x1 = nx - 1 - x0
    y0, y1 = max(0, int(ys.min()) - pad), min(grid.dims[1] - 1, int(ys.max()) + pad)
    z0, z1 = max(0, int(zs.min()) - pad), min(grid.dims[2] - 1, int(zs.max()) + pad)
    vs = grid.voxel_size
    origin = (grid.origin[0] + x0 * vs - offset[0], grid.origin[1] + y0 * vs - offset[1], grid.origin[2] + z0 * vs - offset[2])
    part = Part(name, (x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1), vs, origin=origin, bone=bone, symmetric=symmetric,
                budget=budget)
    part.material[:] = mat[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
    part.color[:] = col[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
    return part


# ---------------------------------------------------------------------------- stats

def part_grid(part: Part) -> meshvox.VoxelGrid:
    return meshvox.VoxelGrid(tuple(part.origin), part.voxel_size, part.dims)


def seam_stats(model: Model, form_name: str = "head") -> dict:
    """Per detail part: ``overlap`` = its voxels whose centre lies in a form voxel, ``contact`` = its
    voxels with a 6-neighbour centre in a form voxel (parts in the same local frame)."""
    form = model.part(form_name)
    fgrid = part_grid(form)
    fsolid = form.solid()
    out = {}
    for part in model.parts:
        if part is form:
            continue
        zs, ys, xs = np.nonzero(part.solid())
        vs = part.voxel_size
        pts = np.asarray(part.origin) + (np.stack([xs, ys, zs], axis=1) + 0.5) * vs
        overlap = int(contains(fsolid, fgrid, pts).sum())
        touch = np.zeros(len(pts), bool)
        for d in ((vs, 0, 0), (-vs, 0, 0), (0, vs, 0), (0, -vs, 0), (0, 0, vs), (0, 0, -vs)):
            touch |= contains(fsolid, fgrid, pts + np.asarray(d))
        out[part.name] = {"overlap": overlap, "contact": int(touch.sum()), "solid": int(len(pts))}
    return out


def mesh_stats(model: Model) -> dict:
    """Cells, solid voxels and culled cube faces (quads; triangles = 2 x quads) per part and in
    total: the numbers behind the performance question. The runtime greedy mesher merges
    coplanar same-colour faces, so its triangle count is lower."""
    out: dict = {"parts": {}}
    tc = ts = tq = 0
    for part in model.parts:
        cells = part.nx * part.ny * part.nz
        solid = part.count_solid()
        quads = meshlib.culled_quads(part, model.palette, jitter=0.0, ao=False).n_faces if solid else 0
        out["parts"][part.name] = {"voxel_mm": round(part.voxel_size * 1000, 2), "dims": list(part.dims),
                                   "cells": cells, "solid": solid, "quads": quads, "triangles": 2 * quads}
        tc += cells
        ts += solid
        tq += quads
    out.update(cells=tc, solid=ts, quads=tq, triangles=2 * tq)
    return out


# ---------------------------------------------------------------------------- the builder

def build_multires_head(params: MultiresParams | None = None, name: str | None = None,
                        human: mh.MakeHuman | None = None) -> Model:
    p = params or MultiresParams()
    hp = p.head
    human = human or mh.load(morphs=hp.morphs)
    lm = human.landmarks()
    pal = make_palette(True)
    pal.add("pupil", (24, 22, 30))

    def C(key: str) -> int:
        return pal.index(FLAT_ALIAS.get(key, key))

    cut_y = float(lm["neck_base"][1])
    chin_y = float(lm["chin"][1])

    # ---- the two grids and solids
    cgrid, form, finfo = build_form(human, lm, p)
    fgrid = head_grid(human, lm, hp, p.detail_voxel, p.margin)
    body_c = meshvox.clip_faces(human.body(), lo=(-INF, cut_y, -INF))
    eyes = (human.helper("l_eye"), human.helper("r_eye"))
    fine, _ = form_solid(fgrid, body_c, eyes, 1, True)
    fine = mirror_union(fine)
    Xc, Yc, Zc = cgrid.centres()
    Xf, Yf, Zf = fgrid.centres()
    offset = (0.0, chin_y, cgrid.origin[2] + cgrid.dims[2] * cgrid.voxel_size / 2)
    stats: dict = {"form": finfo}

    # ---- detail masks on their grids
    fwd = np.array([0.0, 0.0, -p.eye_forward])
    eye_centres = {"eye_l": lm["eye_l"] + fwd, "eye_r": lm["eye_r"] + fwd}
    balls = {k: eyeball(c, p, C) for k, c in eye_centres.items()}       # grid, mat, col, ball
    lips, lips_col = lips_shell(fine, fgrid, lm, p, C)
    cap = hair_cap(fgrid, fine, lm, hp) & (Yf > cut_y)
    cap = mirror_union(cap)

    # ---- carve the form: sockets, mouth, scalp
    carved = form.copy()
    shell = exposed(form)
    cavity_all = np.zeros(form.shape, bool)
    opening_all = np.zeros(form.shape, bool)
    for key, (egrid, _m, _c, ball) in balls.items():
        c = eye_centres[key]
        cavity = footprint(ball, egrid, cgrid)
        cols = (np.abs(Xc - c[0]) <= p.eye_open_w) & (np.abs(Yc - c[1]) <= p.eye_open_h)
        opening = cols & (Zc <= c[2]) & form
        cavity_all |= cavity & ~shell
        opening_all |= opening
    carved &= ~(cavity_all | opening_all)
    stats["socket"] = {"cavity": int((cavity_all & form).sum()), "opening": int(opening_all.sum())}
    k = int(round(p.seam_margin / p.detail_voxel))
    lips_fp, cap_fp = lips, cap
    for _ in range(k):
        lips_fp, cap_fp = box_dilate(lips_fp), box_dilate(cap_fp)
    mouth = footprint(lips_fp, fgrid, cgrid) & carved
    scalp = footprint(cap_fp, fgrid, cgrid) & carved
    carved &= ~(mouth | scalp)
    stats["carved"] = {"mouth": int(mouth.sum()), "scalp": int(scalp.sum())}
    ears_f = np.zeros(fine.shape, bool)
    if p.ears_as_parts:
        ears_f = fine & ear_box(Xf, Yf, Zf, lm, 0.004)
        ear_c = carved & footprint(ears_f, fgrid, cgrid)
        carved &= ~ear_c
        stats["carved"]["ears"] = int(ear_c.sum())
    carved, nostril_inner = carve_nostrils(carved, cgrid, lm)
    carved &= carved[:, :, ::-1]                                   # carve both sides alike
    labels, n = label_components(carved)
    if n > 1:
        carved = labels == 1
    stats["carved"]["components_dropped"] = int(n - 1)

    # ---- the form's tissues and tints
    mat_c, col_c = layer_tissues(carved, cgrid, lm, hp, human, C, cut_y)
    skin_c = mat_c == M.MAT_SKIN
    col_c[hole_walls(nostril_inner, skin_c)] = C("nostril")
    rim = shell & carved                                             # original surface voxels that survived
    inner = (cavity_all | opening_all) & ~carved & ~dilate(rim)      # socket air not touching the outer skin
    col_c[hole_walls(inner, skin_c)] = C("lid_line")
    model = Model(name or "multi_head", voxel_size=p.form_voxel, palette=pal,
                  meta={"source": "makehuman hm08 base.obj (CC0)", "generator": "multires_head", "params": asdict(p),
                        "morphs": dict(hp.morphs)})
    head = Part("head", cgrid.dims, p.form_voxel, origin=tuple(o - d for o, d in zip(cgrid.origin, offset)),
                bone="head", symmetric=True, budget=250_000)
    head.material[:] = mat_c
    head.color[:] = col_c
    model.add_part(head)

    # ---- detail parts against the carved form
    form_f = resample(carved, cgrid, fgrid)
    vsd = p.detail_voxel
    for key, (egrid, emat, ecol, ball) in balls.items():
        c = eye_centres[key]
        form_e = resample(carved, cgrid, egrid)
        Xe, Ye, Ze = egrid.centres()
        axis = (np.abs(Xe - c[0]) <= p.stalk_cols * vsd / 2) & (np.abs(Ye - c[1]) <= p.stalk_cols * vsd / 2)
        back = ball & ~shift(ball, 0, 0, -1) & axis & (Ze > c[2])            # the ball's back voxels on the axis
        stalk = backfill_z(back, form_e, 6) & ~ball
        emat[stalk] = M.MAT_EYE
        ecol[stalk] = C("eye_white")
        part = crop_part(key, egrid, emat, ecol, offset, key, False)
        model.add_part(part)
        stats[key] = {"ball": int(ball.sum()), "stalk": int(stalk.sum())}

    if p.ears_as_parts:
        ears_f &= ~form_f
        # the root: grown toward the skull through the fine skin (never through air) to the form
        ear_l = backfill_z(ears_f & (Xf < 0), ~fine | form_f | cap, 4, step=(1, 0, 0))
        labels, n = label_components(ear_l)
        if n > 1:
            ear_l = labels == 1
            stats["ear_crumbs_removed"] = int((labels > 1).sum())
        ears_f = ear_l | ear_l[:, :, ::-1]
        for key, side in (("ear_l", Xf < 0), ("ear_r", Xf > 0)):
            em = ears_f & side
            model.add_part(crop_part(key, fgrid, np.where(em, M.MAT_SKIN, 0).astype(np.uint8),
                                     np.where(em, C("skin"), 0).astype(np.uint8), offset, "head", False))
        stats["ears"] = int(ears_f.sum())

    lips_full = backfill_z(lips, form_f, 8) & ~form_f & ~ears_f
    lips_col = np.where(lips_full & (lips_col == 0), C("lips"), lips_col).astype(np.uint8)
    lips_col[~lips_full] = 0
    lips_mat = np.where(lips_full, M.MAT_SKIN, 0).astype(np.uint8)
    model.add_part(crop_part("lips", fgrid, lips_mat, lips_col, offset, "head", True))
    stats["lips"] = {"shell": int(lips.sum()), "backfill": int((lips_full & ~lips).sum())}

    hair = hair_part_mask(cap, fine, form_f, fgrid, lm, p) & ~lips_full & ~ears_f
    blocked = lips_full | hair | ears_f
    brows = brow_strips(form_f, fgrid, lm, p, blocked)
    brows = mirror_union(brows) & ~form_f & ~blocked
    brow_l = brows & (Xf < 0)
    labels, n = label_components(brow_l)
    if n > 1:
        brow_l = labels == 1
        stats["brow_crumbs_removed"] = int((labels > 1).sum())
    brows = brow_l | brow_l[:, :, ::-1]
    for key, side in (("brow_l", Xf < 0), ("brow_r", Xf > 0)):
        bm = brows & side
        model.add_part(crop_part(key, fgrid, np.where(bm, M.MAT_HAIR, 0).astype(np.uint8),
                                 np.where(bm, C("brow"), 0).astype(np.uint8), offset, "head", False))

    hair = mirror_union(hair) & ~form_f & ~brows
    hair_mat = np.where(hair, M.MAT_HAIR, 0).astype(np.uint8)
    hair_col = np.where(hair, C("hair"), 0).astype(np.uint8)
    filled = hair | form_f
    up_lit = hair & ~shift(filled, 0, -1, 0) & (Yf > lm["head_top"] - 0.022)
    floor = hair_floor(Xf, Zf, lm, hp)
    deep = hair & ((Yf < floor + hp.hair_taper * 0.6) | ~shift(filled, 0, 1, 0))
    hair_col[deep] = C("hair_deep")
    hair_col[up_lit] = C("hair_light")
    hair_part = crop_part("hair", fgrid, hair_mat, hair_col, offset, "head", True)
    model.add_part(hair_part)
    labels, n = label_components(hair_part.solid())
    if n > 1:
        crumbs = labels > 1
        hair_part.material[crumbs] = 0
        hair_part.color[crumbs] = 0
        stats["hair_crumbs_removed"] = int(crumbs.sum())

    model.meta["build"] = stats
    model.meta["seams"] = seam_stats(model)
    model.meta["mesh"] = mesh_stats(model)
    model.meta["landmarks_m"] = {k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else v)
                                 for k, v in lm.items() if k in ("eye_y", "eye_x", "head_height", "head_width", "head_depth")}
    return model
