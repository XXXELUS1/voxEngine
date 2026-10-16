"""Layered voxel head from the MakeHuman base mesh (replaces the hand-made SDF head).

Pipeline (``build_head``): take the ``body`` mesh above the neck joint, voxelize its
surface together with the eyeball helpers and flood-fill from outside (the cut neck
is sealed) -> one solid; a multi-source BFS from the air gives every voxel its depth,
and the tissues are assigned by depth: skin (>= 5 mm), fat (~4 mm, thicker on the
cheeks), muscle, bone where the depth exceeds 9 mm above the brow line and 16 mm
below it (neck: only the back half = spine). Teeth come from the tooth-row helpers,
eyes from the eyeball helpers (iris + one highlight on the exposed front), ear and
nose cores are cartilage. Hair is a solid cap: the scalp patch of the body mesh is
swept outward along its normals by ``hair_thickness`` (tapered at the hairline) and
filled down to the skin. Skin tints (colour only): lips, eyebrows, stubble, blush,
nostrils. Finally the result is mirrored about x = 0 and crumbs are removed.

Local space of the part: metres, Y up, face toward -Z, chin at y ~ 0, centred on x/z.
The hair helper of MakeHuman is a hull around the whole head and face, so the hairline
is a parameter (``HeadParams.hairline_up`` shifts the ``landmarks()["hairline_y"]``
convention), not the helper's edge.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from . import materials as M
from . import meshvox, mh
from .body import dilate, exposed, neighbour_count
from .checks import label_components
from .model import Model, Part
from .objmesh import ObjMesh
from .palette import Palette

INF = float("inf")


@dataclass
class HeadParams:
    """All lengths in metres unless the name says ``_mm``."""

    neck_keep: float = 0.0               # neck kept below the neck joint (0 = cut at the joint)
    margin: float = 0.02                 # air around the head in the grid
    skin_mm: float = 5.0
    fat_mm: float = 4.0
    cheek_fat_extra_mm: float = 4.0
    bone_depth_above_brow_mm: float = 9.0
    bone_depth_below_brow_mm: float = 16.0
    cartilage: bool = True
    vessels: bool = True
    eye_open: float = 0.004              # half-height of the forced lid opening (0 = mesh lids only)
    iris_radius: float = 0.0065
    hair: bool = True
    hair_thickness: float = 0.012
    hairline_up: float = 0.0             # raise (+) / lower (-) the front hairline
    hair_taper: float = 0.02             # length over which the cap thins toward the hairline
    temple_recess: float = 0.010         # how much higher the hairline sits at the temples
    fringe: bool = False
    fringe_length: int = 3               # voxels
    eyebrows: bool = True
    lips: bool = True
    stubble: bool = True
    stubble_mix: float = 0.6
    blush: bool = True
    nostrils: bool = True
    nostril_radius: float = 0.0035       # 1 voxel at 5 mm, a small cluster at 3 mm
    mirror: bool = True
    seed: int = 7


def make_palette() -> Palette:
    pal = Palette()
    pal.add("skin", (222, 178, 150))
    pal.add("skin_dark", (204, 160, 134))
    pal.add("skin_ruddy", (226, 166, 144))
    pal.add("stubble", (170, 136, 122))
    pal.add("lips", (198, 124, 112))
    pal.add("nostril", (104, 66, 56))
    pal.add("fat", (238, 214, 140))
    pal.add("muscle", (150, 40, 45))
    pal.add("bone", (235, 225, 200))
    pal.add("tooth", (245, 240, 220))
    pal.add("tooth_dark", (232, 224, 198))
    pal.add("cartilage", (225, 185, 160))
    pal.add("vessel", (120, 20, 30))
    pal.add("hair", (58, 40, 30))
    pal.add("hair_light", (66, 47, 36))
    pal.add("hair_deep", (50, 35, 26))
    pal.add("eye_white", (240, 240, 235))
    pal.add("iris", (72, 126, 190))
    pal.add("highlight", (246, 248, 250))
    return pal


# ---------------------------------------------------------------------------
# small helpers (each mask is a separate function so the builder stays readable)
# ---------------------------------------------------------------------------

def depth_map(solid: np.ndarray) -> np.ndarray:
    """6-connected BFS depth from the air: 0 for air, 1 for surface voxels, 2 for the
    voxels behind them, ... (the grid border does not count as air)."""
    depth = np.zeros(solid.shape, np.int16)
    cur = exposed(solid)
    remaining = solid & ~cur
    d = 1
    while cur.any():
        depth[cur] = d
        nxt = dilate(cur) & remaining
        remaining &= ~nxt
        cur = nxt
        d += 1
    depth[remaining] = d  # unreachable (never for a connected solid)
    return depth


def hash01(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic per-voxel pseudo-random in [0, 1)."""
    h = (ix.astype(np.uint32) * np.uint32(73856093)) ^ (iy.astype(np.uint32) * np.uint32(19349663)) \
        ^ (iz.astype(np.uint32) * np.uint32(83492791)) ^ np.uint32((seed * 2654435761) & 0xFFFFFFFF)
    h = (h ^ (h >> np.uint32(13))) * np.uint32(0x5BD1E995)
    h = h ^ (h >> np.uint32(15))
    return h.astype(np.float64) / 4294967296.0


def smoothstep(a: float, b: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - a) / max(b - a, 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def hairline_y(x: np.ndarray, lm: dict, p: HeadParams) -> np.ndarray:
    """Front hairline height as a function of |x|: the convention height, raised at the temples."""
    return lm["hairline_y"] + p.hairline_up + p.temple_recess * smoothstep(0.028, 0.055, np.abs(x))


def scalp_vertex_mask(v: np.ndarray, lm: dict, p: HeadParams) -> np.ndarray:
    """Which body vertices carry hair: forehead/temples above the hairline, the sides
    above the ear (sideburn in front of it), the back down to the nape; never the ears."""
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    ear_z0, ear_z1 = lm["ear_z"]
    ear_y0, ear_y1 = lm["ear_y"]
    front = z < ear_z0 - 0.02
    side = ~front & (z <= ear_z1 + 0.01)
    back = z > ear_z1 + 0.01
    nape_y = lm["skull_base"][1] - 0.025
    ok = (front & (y >= hairline_y(x, lm, p)))
    ok |= side & (y >= lm["eye_y"] - 0.005) & ((np.abs(x) >= 0.055) | (y >= ear_y1 + 0.01))
    ok |= back & (y >= nape_y)
    on_ear = (np.abs(x) > lm["ear_inner_x"]) & (y < ear_y1 + 0.004) & (z > ear_z0 - 0.006) & (z < ear_z1 + 0.006)
    return ok & ~on_ear


def hair_cap(body: ObjMesh, grid: meshvox.VoxelGrid, solid: np.ndarray, lm: dict, p: HeadParams) -> np.ndarray:
    """Solid hair cap: the scalp patch swept outward along its normals, thinned toward
    the hairline, filled down to the skin, minus the ears."""
    v = body.vertices
    ok = scalp_vertex_mask(v, lm, p)
    keep = ok[body.faces].all(axis=1)
    if not keep.any():
        return np.zeros(grid.shape, bool)
    scalp = ObjMesh(v, body.faces[keep], body.face_group[keep], list(body.groups))
    # per-vertex thickness: taper from 35 % at the hairline to 100 % ``hair_taper`` above it
    w = np.ones(len(v))
    front = v[:, 2] < lm["ear_z"][0] - 0.02
    w[front] = 0.35 + 0.65 * np.clip((v[front, 1] - hairline_y(v[front, 0], lm, p)) / p.hair_taper, 0.0, 1.0)
    vn = body.vertex_normals()
    out = np.zeros(grid.shape, bool)
    step = 0.5 * grid.voxel_size
    t_from, t_to = -0.006, p.hair_thickness
    n = max(1, int(math.ceil((t_to - t_from) / step)))
    for i in range(n + 1):
        t = t_from + (t_to - t_from) * i / n
        pos = v + vn * (t * (w[:, None] if t > 0 else 1.0))
        meshvox.voxelize_surface(pos[scalp.faces], grid, out=out)
    X, Y, Z = grid.centres()
    ear_z0, ear_z1 = lm["ear_z"]
    ear_zone = (np.abs(X) > lm["ear_inner_x"] - 0.004) & (Y < lm["ear_y"][1] + 0.004) & (Z > ear_z0 - 0.008) & (Z < ear_z1 + 0.008)
    return smooth_cap(out & ~solid & ~ear_zone, solid)


def smooth_cap(hair: np.ndarray, solid: np.ndarray, passes: int = 2) -> np.ndarray:
    """Knock off one-voxel spikes (hair with < 3 hair/solid 6-neighbours) and fill
    one-voxel pits (air with >= 5 hair/solid neighbours, next to hair)."""
    hair = hair.copy()
    for _ in range(passes):
        n = neighbour_count(hair | solid)
        hair &= n >= 3
        n = neighbour_count(hair | solid)
        pits = ~solid & ~hair & (n >= 5) & dilate(hair)
        hair |= pits
    return hair


def pick_front_row(cand: np.ndarray, Z: np.ndarray, rows: int = 1) -> np.ndarray:
    """Per x column, take the row of the frontmost (smallest z) candidate voxel, smooth
    that row over neighbouring columns (median of 3) so the line does not stair-step,
    then keep the frontmost candidate in that row and the ``rows - 1`` rows above it."""
    out = np.zeros(cand.shape, bool)
    zs, ys, xs = np.nonzero(cand)
    cols = np.unique(xs)
    if cols.size == 0:
        return out
    row0 = {}
    for ix in cols:
        sel = xs == ix
        zz, yy = zs[sel], ys[sel]
        order = np.lexsort((yy, Z[zz, yy, ix]))
        row0[int(ix)] = int(yy[order[0]])
    for ix in cols:
        ix = int(ix)
        neigh = [row0[j] for j in (ix - 1, ix, ix + 1) if j in row0]
        y_target = int(np.median(neigh))
        sel = xs == ix
        zz, yy = zs[sel], ys[sel]
        for r in range(rows):
            # nearest candidate row to the smoothed target, frontmost voxel in it
            dy = np.abs(yy - (y_target + r))
            best = dy.min()
            if best > 1:
                continue
            pick = (dy == best)
            k = np.argmin(np.where(pick, Z[zz, yy, ix], np.inf))
            out[zz[k], yy[k], ix] = True
    return out


def lid_opening(solid: np.ndarray, eyeball: np.ndarray, X, Y, Z, centre, half_w: float, half_h: float,
                reach: int = 2) -> np.ndarray:
    """Voxels to turn into eye so the eyeball shows through a small lid opening: inside
    the x/y ellipse (``half_w``, ``half_h``) around the eye centre, in every column where
    the eyeball lies at most ``reach`` voxels behind the front surface, the surface voxel
    and everything down to the eyeball become eye. Outside the ellipse the mesh lids stay."""
    foot = (((X[0] - centre[0]) / half_w) ** 2 + ((Y[0] - centre[1]) / half_h) ** 2) <= 1.0  # (ny, nx)
    front = Z < centre[2]
    s = solid & front
    e = eyeball & front
    first = np.argmax(s, axis=0)
    efirst = np.argmax(e, axis=0)
    sel = foot & s.any(axis=0) & e.any(axis=0) & (efirst >= first) & (efirst - first <= reach)
    out = np.zeros(solid.shape, bool)
    iz = np.arange(solid.shape[0])[:, None, None]
    out |= sel[None, :, :] & (iz >= first[None, :, :]) & (iz <= efirst[None, :, :])
    return out


def capsule_mask(X, Y, Z, a, b, r: float) -> np.ndarray:
    """Voxels within ``r`` of the segment a-b (metres); |x| is used, so both sides get it."""
    ax, ay, az = a
    bx, by, bz = b
    dx, dy, dz = bx - ax, by - ay, bz - az
    px, py, pz = np.abs(X) - ax, Y - ay, Z - az
    t = np.clip((px * dx + py * dy + pz * dz) / max(dx * dx + dy * dy + dz * dz, 1e-12), 0.0, 1.0)
    return (px - t * dx) ** 2 + (py - t * dy) ** 2 + (pz - t * dz) ** 2 <= r * r


def nearest_voxel(mask: np.ndarray, X, Y, Z, target) -> tuple | None:
    """Index (z, y, x) of the voxel of ``mask`` closest to the metre point ``target``."""
    zs, ys, xs = np.nonzero(mask)
    if zs.size == 0:
        return None
    d = (X[zs, ys, xs] - target[0]) ** 2 + (Y[zs, ys, xs] - target[1]) ** 2 + (Z[zs, ys, xs] - target[2]) ** 2
    k = int(np.argmin(d))
    return (zs[k], ys[k], xs[k])


def mirror_x(part: Part) -> None:
    """Copy the -x half onto the +x half (grid must be x-centred)."""
    nx = part.nx
    half = nx // 2
    for arr in (part.material, part.color, part.state):
        arr[:, :, nx - half:] = arr[:, :, :half][:, :, ::-1]


def keep_largest_component(part: Part) -> int:
    """Remove every solid 6-component but the largest. Returns the voxels removed."""
    labels, n = label_components(part.solid())
    if n <= 1:
        return 0
    crumbs = (labels > 1)
    cnt = int(np.count_nonzero(crumbs))
    part.material[crumbs] = 0
    part.color[crumbs] = 0
    return cnt


# ---------------------------------------------------------------------------
# the builder
# ---------------------------------------------------------------------------

def build_head(voxel_size: float = 0.005, params: HeadParams | None = None, name: str | None = None,
               human: mh.MakeHuman | None = None) -> Model:
    p = params or HeadParams()
    vs = float(voxel_size)
    vs_mm = vs * 1000.0
    human = human or mh.load()
    lm = human.landmarks()
    body = human.body()
    pal = make_palette()
    C = pal.index

    # ---- grid (MakeHuman frame: feet at y = 0; the part origin is shifted at the end)
    cut_y = float(lm["neck_base"][1]) - p.neck_keep
    hv = body.vertices[body.vertices[:, 1] > cut_y - 0.01]
    hair_room = p.hair_thickness + 0.004 if p.hair else 0.0
    lo = (hv[:, 0].min() - p.margin - hair_room, cut_y, hv[:, 2].min() - p.margin - hair_room)
    hi = (hv[:, 0].max() + p.margin + hair_room, lm["head_top"] + p.margin + hair_room, hv[:, 2].max() + p.margin + hair_room)
    grid = meshvox.VoxelGrid.from_box(lo, hi, vs, centre_x=True)
    zc = (hv[:, 2].min() + hv[:, 2].max()) * 0.5          # grid centred on the head in z too
    grid = meshvox.VoxelGrid((grid.origin[0], grid.origin[1], zc - grid.dims[2] * vs / 2), vs, grid.dims)
    X, Y, Z = grid.centres()
    model = Model(name or f"head_mh_{round(vs_mm)}mm", voxel_size=vs, palette=pal,
                  meta={"source": "makehuman hm08 base.obj (CC0)", "params": asdict(p)})
    part = Part("head", grid.dims, vs, origin=grid.origin, bone="head", symmetric=True,
                budget=400_000 if vs < 0.004 else 250_000)
    model.add_part(part)

    # ---- solid: body surface + eyeballs, filled from outside (neck cut sealed)
    body_c = meshvox.clip_faces(body, lo=(-INF, cut_y, -INF))
    body_surf = meshvox.voxelize_surface(body_c, grid)
    eyes = human.helper("l_eye"), human.helper("r_eye")
    eye_surf = np.zeros(grid.shape, bool)
    for e in eyes:
        meshvox.voxelize_surface(e, grid, out=eye_surf)
    solid, info = meshvox.fill_interior(body_surf | eye_surf, sealed_faces=("-y",))
    model.meta["fill"] = info
    if info["leak"]:
        raise RuntimeError(f"mesh voxelization leaked: {info}")

    # ---- tissues by depth
    depth = depth_map(solid)
    depth_mm = (depth.astype(np.float32) - 0.5) * vs_mm
    skin_vox = max(1, int(math.ceil(p.skin_mm / vs_mm - 1e-6)))
    fat_vox = max(1, int(round(p.fat_mm / vs_mm)))
    extra_vox = int(round(p.cheek_fat_extra_mm / vs_mm))
    mat = np.zeros(grid.shape, np.uint8)
    col = np.zeros(grid.shape, np.uint8)
    mat[solid] = M.MAT_MUSCLE
    col[solid] = C("muscle")
    cheek = (((np.abs(X) - (lm["eye_x"] + 0.012)) / 0.028) ** 2 + ((Y - (lm["eye_y"] - 0.04)) / 0.03) ** 2
             + ((Z - (lm["nose_tip"][2] + 0.06)) / 0.045) ** 2) <= 1.0
    fat_limit = skin_vox + fat_vox + np.where(cheek, extra_vox, 0)
    fat = solid & (depth > skin_vox) & (depth <= fat_limit)
    mat[fat] = M.MAT_FAT
    col[fat] = C("fat")
    skin = solid & (depth <= skin_vox)
    mat[skin] = M.MAT_SKIN
    col[skin] = C("skin")
    # bone: skull close under the forehead, deeper under the face; spine in the back of the neck
    ramp = np.clip((Y - (lm["brow_y"] - 0.02)) / 0.02, 0.0, 1.0)
    thr = p.bone_depth_below_brow_mm + (p.bone_depth_above_brow_mm - p.bone_depth_below_brow_mm) * ramp
    bone = solid & (depth_mm >= thr) & (depth > skin_vox + 1)
    neck = Y < lm["chin"][1] - 0.005
    bone &= ~(neck & (Z < lm["neck_base"][2]))
    mat[bone] = M.MAT_BONE
    col[bone] = C("bone")
    if p.cartilage:
        ear_z0, ear_z1 = lm["ear_z"]
        ear = (np.abs(X) > lm["ear_inner_x"]) & (Y > lm["ear_y"][0]) & (Y < lm["ear_y"][1]) & (Z > ear_z0) & (Z < ear_z1)
        nose = (np.abs(X) < 0.018) & (Y > lm["nose_base_y"] - 0.004) & (Y < lm["eye_y"] - 0.004) & (Z < lm["nose_tip"][2] + 0.03)
        cart = solid & (depth > skin_vox) & ~bone & (ear | nose)
        mat[cart] = M.MAT_CARTILAGE
        col[cart] = C("cartilage")

    # ---- vessels: temple vein and carotid, only replacing fat/muscle (they stay under the skin)
    if p.vessels:
        ex, ey, ez = lm["eye_x"], lm["eye_y"], lm["eye_z"]
        temple = capsule_mask(X, Y, Z, (ex + 0.036, ey - 0.005, ez + 0.02), (ex + 0.03, ey + 0.05, ez + 0.03), 0.0045)
        carotid = capsule_mask(X, Y, Z, (0.026, cut_y - 0.01, lm["neck_base"][2] - 0.03), (0.03, lm["chin"][1] - 0.004, lm["chin"][2] + 0.035), 0.005)
        vessel = (temple | carotid) & np.isin(mat, (M.MAT_FAT, M.MAT_MUSCLE))
        mat[vessel] = M.MAT_VESSEL
        col[vessel] = C("vessel")

    # ---- teeth: inside the jaw, never replacing skin (a closed mouth hides them fully)
    teeth = np.zeros(grid.shape, bool)
    for h in ("upper_teeth", "lower_teeth"):
        t, _ = meshvox.voxelize_solid(meshvox.close_holes(human.helper(h)), grid)
        teeth |= t
    teeth &= solid & (mat != M.MAT_SKIN)
    mat[teeth] = M.MAT_TOOTH
    tz, ty, tx = np.nonzero(teeth)
    col[tz, ty, tx] = np.where(tx % 2 == 0, C("tooth"), C("tooth_dark")).astype(np.uint8)

    # ---- eyes: eyeball voxels keep the mesh lids; a small forced opening guarantees visibility
    eye_solid = np.zeros(grid.shape, bool)
    for e in eyes:
        s, _ = meshvox.fill_interior(meshvox.voxelize_surface(e, grid))
        eye_solid |= s
    eye = eye_solid & ~body_surf
    if p.eye_open > 0:
        for ec in (lm["eye_l"], lm["eye_r"]):
            eye |= lid_opening(mat != M.MAT_AIR, eye_solid, X, Y, Z, ec, lm["eye_radius"] * 0.85, p.eye_open)
    mat[eye] = M.MAT_EYE
    col[eye] = C("eye_white")
    exposed_all = exposed(mat != M.MAT_AIR)
    for ec in (lm["eye_l"], lm["eye_r"]):
        front = eye & exposed_all & (Z < ec[2])
        r2 = (X - ec[0]) ** 2 + (Y - ec[1]) ** 2
        iris = front & (r2 <= p.iris_radius ** 2)
        col[iris] = C("iris")
        zs, ys, xs = np.nonzero(iris)
        if zs.size:
            k = int(np.argmax(Y[zs, ys, xs] * 2 + np.abs(X[zs, ys, xs])))  # upper, toward the temple
            col[zs[k], ys[k], xs[k]] = C("highlight")

    # ---- skin tints (colour only) on the exposed skin, before the hair hides the scalp
    skin_ex = (mat == M.MAT_SKIN) & exposed_all
    if p.eyebrows:
        band = skin_ex & (Y >= lm["eye_y"] + 0.012 - vs * 0.5) & (Y <= lm["eye_y"] + 0.018 + vs * 0.5) \
            & (np.abs(X) >= lm["eye_x"] - 0.017) & (np.abs(X) <= lm["eye_x"] + 0.021)
        if band.any():
            band &= Z <= Z[band].min() + 0.02
            rows = max(1, int(round(0.005 / vs)))
            col[pick_front_row(band, Z, rows)] = C("hair")
    if p.lips:
        lips = skin_ex & (np.abs(Y - lm["mouth"][1]) <= 0.006) & (np.abs(X) <= lm["mouth_half_width"] - 0.004)
        if lips.any():
            lips &= Z <= Z[lips].min() + 0.016
            col[lips] = C("lips")
    if p.blush:
        blush = skin_ex & ((((np.abs(X) - (lm["eye_x"] + 0.014)) / 0.02) ** 2 + ((Y - (lm["eye_y"] - 0.034)) / 0.012) ** 2
                           + ((Z - (lm["nose_tip"][2] + 0.045)) / 0.028) ** 2) <= 1.0)
        tipc = lm["nose_tip"]
        blush |= skin_ex & ((X - tipc[0]) ** 2 + (Y - tipc[1]) ** 2 + (Z - tipc[2] - 0.006) ** 2 <= 0.011 ** 2)
        col[blush] = C("skin_ruddy")
    if p.stubble:
        ear_zc = lm["ear_centre"][2]
        below_mouth = (Y <= lm["mouth"][1] + 0.004) & (Y >= cut_y + 0.02)
        under_jaw_ok = (Y >= lm["chin"][1] - 0.008) | ((Y >= lm["chin"][1] - 0.018) & (Z <= lm["chin"][2] + 0.03))
        cheeks = (Y > lm["mouth"][1]) & (Y <= lm["eye_y"] - 0.038) & (np.abs(X) >= 0.034)
        zone = skin_ex & (Z <= ear_zc) & ((below_mouth & under_jaw_ok) | cheeks)
        zone &= ~(np.abs(Y - lm["mouth"][1]) <= 0.006) | (np.abs(X) > lm["mouth_half_width"])
        zone &= ~((np.abs(X) < 0.016) & (Y > lm["nose_base_y"] - 0.006))
        iz, iy, ixx = np.indices(grid.shape)
        zone &= hash01(np.abs(ixx - grid.dims[0] / 2 + 0.5).astype(np.int64), iy, iz, p.seed) < p.stubble_mix
        col[zone] = C("stubble")
    if p.nostrils:
        for sx in (-1, 1):
            target = (sx * 0.009, lm["nose_base_y"], lm["nose_tip"][2] + 0.012)
            idx = nearest_voxel(skin_ex, X, Y, Z, target)
            if idx is not None:
                c = (X[idx], Y[idx], Z[idx])
                near = skin_ex & ((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2 <= p.nostril_radius ** 2)
                col[near] = C("nostril")

    # ---- hair cap (solid), 3 shades
    if p.hair:
        hair = hair_cap(body_c, grid, mat != M.MAT_AIR, lm, p)
        hair &= Y > cut_y
        mat[hair] = M.MAT_HAIR
        iz, iy, ixx = np.indices(grid.shape)
        r = hash01(np.abs(ixx - grid.dims[0] / 2 + 0.5).astype(np.int64), iy, iz, p.seed + 1)
        shade = np.where(r < 0.18, C("hair_light"), np.where(r < 0.36, C("hair_deep"), C("hair"))).astype(np.uint8)
        col[hair] = shade[hair]

    part.material[:] = mat
    part.color[:] = col

    if p.hair and p.fringe:
        from .body import grow_hair
        strip = (mat == M.MAT_HAIR) & (Z < lm["ear_z"][0] - 0.02) & (Y < hairline_y(X, lm, p) + 0.012)
        grow_hair(part, strip, p.fringe_length, "hair_deep", direction=(0.0, -1.0, -0.35), jitter=0.15,
                  density=0.7, seed=p.seed, on=(M.MAT_HAIR,))

    if p.mirror:
        mirror_x(part)
    model.meta["crumbs_removed"] = keep_largest_component(part)

    # ---- local frame: chin at y ~ 0, centred on x/z
    part.origin = (grid.origin[0], grid.origin[1] - float(lm["chin"][1]), -grid.dims[2] * vs / 2)
    model.meta["landmarks_m"] = {k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else v)
                                 for k, v in lm.items() if k in ("eye_y", "eye_x", "hairline_y", "head_height", "head_width", "head_depth")}
    return model
