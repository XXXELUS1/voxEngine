"""Hybrid head: the FORM from the MakeHuman mesh on a coarse 1 cm grid, the DETAILS designed at 5 mm.

Why: a mesh sampled at 5 mm gives random one-voxel staircases (uncanny), a head designed by hand
at 1 cm gives a flat doll. This builder takes the best of both: the morphed MakeHuman head
(mild caricature through :mod:`voxgen.mh_targets`) is voxelized at 1 cm, so every step of the
form is a deliberate 1 cm block, cleaned (3x3x3 closing + opening), mirror-symmetrized, reduced
to its largest component and *quantized* (every one-voxel bump removed, every one-voxel pit
filled, by the 26-neighbour count, until stable), then upsampled x2 to the 5 mm game grid. The
hair is a designed mass on the same coarse grid: an anisotropic shell (thick on top, thin on the
sides, medium at the back) over the scalp, above a stepped hairline (forehead ledge - temples -
sideburns to mid-ear - over the ears - nape at the ear's bottom); at 5 mm a few single-voxel
strands hang off the hairline and the nape (the only fine hair). The face is drawn at 5 mm by
:mod:`voxgen.facedetail` on landmarks of the *morphed* mesh: human-sized eyes (5 x 3, recessed
one voxel, 3 x 3 iris, one highlight, a lid line), one-row brows, a nose bridge ridge and two
nostril dots on the form's own nose, a mouth line with lips, an ear spot. Interior layers by BFS
depth (skin 1, fat 2, muscle 3, bone >= 4; nose and ears cartilage), hair treated as air.

Palette discipline: 3 skin tones (base, shadow, ruddy), lip tone, 3 hair tones (base, lit crown
only - every lit terrace would read as a contour map -, deep lower edge), eye white / iris /
highlight, one dark line colour; no noise.
Decisions: coarse grid has an odd number of columns (a centre column, exact mirror); the eye
recess follows the local surface (always exactly one voxel deep); strands are deterministic
(every third column, by |x|, so they mirror); the neck is cut at the MakeHuman neck joint.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from . import materials as M
from . import meshvox, mh
from .body import shift
from .checks import label_components
from .facedetail import FaceDetailParams, FaceLandmarks, apply_face_details, x_coords
from .head_from_mesh import clean_form, depth_map, keep_largest_component, smooth_cap, upsample
from .model import Model, Part
from .palette import Palette
from .toonhead import HAIR_TONES, IRIS_TONES, SKIN_TONES

#: mild caricature: a longer, deeper nose, a firmer chin, cheekbones, a heavier brow
DEFAULT_MORPHS = {
    "nose-scale-vert": 0.8,
    "nose-scale-depth": 0.5,
    "chin-prominent": 0.4,
    "cheek-bones": 0.4,
    "eyebrows-trans-forward": 0.6,
}
UPSAMPLE = 2


@dataclass
class HybridHeadParams:
    morphs: dict = field(default_factory=lambda: dict(DEFAULT_MORPHS))
    coarse_voxel: float = 0.01
    margin: float = 0.02            # air around the head (metres), hair room added on top
    quantize_passes: int = 6
    bump_max: int = 11              # 26-neighbour count at or under which a solid voxel is a bump (9 lone, 11 in a 1-wide terrace)
    pit_min: int = 15               # count at or above which an air voxel is a pit (17 lone, 15 in a 1-wide groove)
    # -- hair (coarse voxels)
    hair: bool = True
    hair_top: int = 2
    hair_side: int = 1
    hair_back: int = 2
    hairline_frac: float = 0.34     # forehead hairline: fraction of the way from the brow to the crown
    temple_drop: int = 2            # the hairline drops this many rows at the temples
    strands: bool = True            # single 5 mm strands along the hairline and the nape
    # -- colour
    skin_tone: str = "light"
    hair_tone: str = "brown"
    iris_tone: str = "blue"
    stubble: bool = False
    detail: FaceDetailParams = field(default_factory=lambda: FaceDetailParams(cheeks=True))


def make_palette(p: HybridHeadParams) -> Palette:
    skin, shadow, ruddy = SKIN_TONES[p.skin_tone]
    hair, hair_light = HAIR_TONES[p.hair_tone]
    pal = Palette()
    pal.add("skin", skin)
    pal.add("skin_shadow", shadow)
    pal.add("skin_ruddy", ruddy)
    pal.add("stubble", tuple(int(round(c * f)) for c, f in zip(skin, (0.86, 0.86, 0.90))))
    pal.add("lips", tuple(int(round(0.58 * a + 0.42 * b)) for a, b in zip(skin, (168, 70, 78))))
    pal.add("line", (88, 54, 52))
    pal.add("hair", hair)
    pal.add("hair_light", hair_light)
    pal.add("hair_deep", tuple(int(round(c * 0.72)) for c in hair))
    pal.add("eye_white", (242, 242, 236))
    pal.add("iris", IRIS_TONES.get(p.iris_tone, IRIS_TONES["blue"]))
    pal.add("highlight", (248, 250, 252))
    pal.add("fat", (238, 214, 140))
    pal.add("muscle", (150, 40, 45))
    pal.add("bone", (235, 225, 200))
    pal.add("cartilage", (225, 185, 160))
    return pal


# ---------------------------------------------------------------------------- coarse form
def box_neighbours_edge(mask: np.ndarray) -> np.ndarray:
    """26-neighbour count with the grid border replicated (the neck cut at the bottom of the grid
    is a wall, not air, so its rim is never mistaken for a bump)."""
    nz, ny, nx = mask.shape
    padded = np.pad(mask, 1, mode="edge")
    n = np.zeros(mask.shape, np.int16)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx or dy or dz:
                    n += padded[1 + dz:1 + dz + nz, 1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx]
    return n


def quantize_form(solid: np.ndarray, passes: int = 6, bump_max: int = 11, pit_min: int = 15) -> tuple[np.ndarray, dict]:
    """Remove bumps (solid with <= ``bump_max`` of 26 neighbours: a lone voxel on a flat face has
    9, a voxel of a one-wide terrace 11, a corner of a two-wide step 12) and fill pits (air with
    >= ``pit_min``: a lone hole has 17, a one-wide groove 15), mirror-symmetric, until stable: so
    every remaining step is at least two voxels long wherever the form allows it."""
    removed = filled = 0
    for _ in range(passes):
        n = box_neighbours_edge(solid)
        bumps = solid & (n <= bump_max)
        pits = ~solid & (n >= pit_min)
        bumps = (bumps | bumps[:, :, ::-1]) & solid
        pits = (pits | pits[:, :, ::-1]) & ~solid
        if not bumps.any() and not pits.any():
            break
        removed += int(bumps.sum())
        filled += int(pits.sum())
        solid = (solid & ~bumps) | pits
    return solid, {"bumps_removed": removed, "pits_filled": filled}


def coarse_form(human: mh.MakeHuman, p: HybridHeadParams) -> tuple[meshvox.VoxelGrid, np.ndarray, dict]:
    """Voxelize the morphed head (body above the neck joint + eyeballs) at ``coarse_voxel`` on an
    x-centred grid with an odd number of columns, fill, clean, symmetrize, keep the largest
    component, quantize."""
    lm = human.landmarks()
    body = human.body()
    vs = p.coarse_voxel
    cut_y = float(lm["neck_base"][1])
    hv = body.vertices[body.vertices[:, 1] > cut_y - 0.01]
    room = (p.hair_top + 1) * vs if p.hair else 0.0
    side = (p.hair_side + 1) * vs if p.hair else 0.0
    back = (p.hair_back + 1) * vs if p.hair else 0.0
    half_x = max(abs(hv[:, 0].min()), abs(hv[:, 0].max())) + p.margin + side
    nx = int(math.ceil(2 * half_x / vs))
    nx += 1 - nx % 2                                   # odd: a centre column at x = 0
    z0, z1 = hv[:, 2].min() - p.margin - side, hv[:, 2].max() + p.margin + back
    nz = int(math.ceil((z1 - z0) / vs))
    zc = (z0 + z1) / 2
    ny = int(math.ceil((lm["head_top"] + p.margin + room - cut_y) / vs))
    grid = meshvox.VoxelGrid((-nx * vs / 2, cut_y, zc - nz * vs / 2), vs, (nx, ny, nz))
    body_c = meshvox.clip_faces(body, lo=(-math.inf, cut_y, -math.inf))
    surf = meshvox.voxelize_surface(body_c, grid)
    for e in ("l_eye", "r_eye"):
        meshvox.voxelize_surface(human.helper(e), grid, out=surf)
    solid, info = meshvox.fill_interior(surf, sealed_faces=("-y",))
    if info["leak"]:
        raise RuntimeError(f"mesh voxelization leaked: {info}")
    raw = int(solid.sum())
    solid = clean_form(solid)
    solid |= solid[:, :, ::-1]
    labels, n = label_components(solid)
    if n > 1:
        solid = labels == 1
    solid, qinfo = quantize_form(solid, p.quantize_passes, p.bump_max, p.pit_min)
    info = dict(info, raw=raw, cleaned=int(solid.sum()), components=int(n), **qinfo)
    return grid, solid, info


# ---------------------------------------------------------------------------- hair (coarse)
def _coarse_landmarks(grid: meshvox.VoxelGrid, lm: dict) -> dict:
    """Landmarks as coarse grid indices (rows/z) and |X| in coarse voxels."""
    vs = grid.voxel_size
    oy, oz = grid.origin[1], grid.origin[2]
    row = lambda y: int(math.floor((y - oy) / vs))
    zi = lambda z: int(math.floor((z - oz) / vs))
    return {
        "brow": row(lm["brow_y"]), "top": row(lm["head_top"]),
        "ear_y0": row(lm["ear_y"][0]), "ear_y1": row(lm["ear_y"][1]),
        "ear_z0": zi(lm["ear_z"][0]), "ear_z1": zi(lm["ear_z"][1]),
        "ear_x": lm["ear_inner_x"] / vs,
    }


def hair_mass(grid: meshvox.VoxelGrid, solid: np.ndarray, lm: dict, p: HybridHeadParams) -> tuple[np.ndarray, np.ndarray]:
    """Designed hair on the coarse grid. Returns (hair mask, floor rows (nz, nx))."""
    nz, ny, nx = solid.shape
    c = _coarse_landmarks(grid, lm)
    X = np.arange(nx) - (nx - 1) / 2
    hl = int(round(c["brow"] + p.hairline_frac * (c["top"] - c["brow"])))
    # the forehead's front z at the hairline row (centre columns): the ledge hangs one in front
    centre = np.abs(X) <= 2
    rows = solid[:, hl, :] & centre[None, :]
    zf = int(np.nonzero(rows.any(axis=1))[0].min()) if rows.any() else 0
    Zc = np.arange(nz)[:, None]
    Xc = X[None, :]
    floor = np.select(
        [Zc <= zf + 1, Zc <= zf + 3, Zc <= c["ear_z0"] - 1, Zc <= c["ear_z1"] + 1],
        [hl, hl - p.temple_drop, (c["ear_y0"] + c["ear_y1"] + 1) // 2, c["ear_y1"] + 1],
        default=c["ear_y0"] + 1)
    floor = np.where((Zc > c["ear_z1"] + 1) & (np.abs(Xc) > c["ear_x"] - 3), c["ear_y1"] + 1, floor)
    floor = np.broadcast_to(floor, (nz, nx))
    # anisotropic shell: the scalp (ears excluded) swept by an ellipsoid of offsets that grows
    # up, sideways, back and one voxel forward (the hairline ledge)
    scalp = solid & (np.abs(X)[None, None, :] < c["ear_x"] - 0.5)
    hair = np.zeros(solid.shape, bool)
    rx, ry, rz = p.hair_side + 0.5, p.hair_top + 0.5, p.hair_back + 0.5
    for dx in range(-p.hair_side, p.hair_side + 1):
        for dy in range(0, p.hair_top + 1):
            for dz in range(-1, p.hair_back + 1):
                if (dx / rx) ** 2 + (dy / ry) ** 2 + (dz / rz) ** 2 <= 1.0:
                    hair |= shift(scalp, dx, dy, dz)
    above = np.arange(ny)[None, :, None] >= floor[:, None, :]
    hair &= ~solid & above
    hair = smooth_cap(hair, solid) & ~solid & above
    hair |= hair[:, :, ::-1]
    return hair, floor


# ---------------------------------------------------------------------------- builder
def fine_landmarks(grid: meshvox.VoxelGrid, lm: dict, mat: np.ndarray, vs: float) -> FaceLandmarks:
    """Landmarks of the morphed mesh as 5 mm indices. The nose base is read off the form itself
    (the lowest row above the mouth where the centre columns protrude), so the nostrils sit at
    the bottom of the form's nose whatever the morph did."""
    oy, oz = grid.origin[1], grid.origin[2]
    nz, ny, nx = mat.shape
    row = lambda y: int(math.floor((y - oy) / vs))
    half = lambda x: math.floor(x / vs) + 0.5
    eye_y = row(lm["eye_y"])
    mouth_y = row(lm["mouth"][1])
    X = x_coords(nx)
    solid = mat != M.MAT_AIR
    centre = np.abs(X) <= 0.5
    front = np.where(solid[:, :, centre].any(axis=2), np.arange(nz)[:, None], nz).min(axis=0)   # (ny,)
    nose_base = row(lm["nose_tip"][1] - 0.012)
    for y in range(mouth_y + 2, eye_y - 2):
        if front[y] <= front[mouth_y] - 2:
            nose_base = y
            break
    ez = int(math.floor((lm["ear_centre"][2] - oz) / vs))
    return FaceLandmarks(
        eye_x=half(lm["eye_x"]), eye_y=eye_y, brow_y=eye_y + 4, nose_top_y=eye_y - 1,
        nose_base_y=nose_base, nostril_x=1.5, mouth_y=mouth_y,
        mouth_half_w=math.floor(lm["mouth_half_width"] / vs - 0.5) + 0.5,
        ear_y=row(lm["ear_centre"][1]), ear_z=ez, cheek_y=nose_base)


def build_hybrid_head(params: HybridHeadParams | None = None, fine_voxel: float = 0.005,
                      name: str | None = None, human: mh.MakeHuman | None = None) -> Model:
    """Coarse form from the morphed mesh, designed hair, fine details, layers. One part ``head``
    (bone ``head``), chin at y = 0, centred on x/z, face toward -Z."""
    p = params or HybridHeadParams()
    if abs(fine_voxel * UPSAMPLE - p.coarse_voxel) > 1e-9:
        raise ValueError("build_hybrid_head: fine_voxel must be half the coarse voxel (5 mm for 1 cm)")
    human = human or mh.load(morphs=p.morphs)
    lm = human.landmarks()
    pal = make_palette(p)
    C = pal.index
    grid, solid, info = coarse_form(human, p)
    hair_c = np.zeros(solid.shape, bool)
    if p.hair:
        hair_c, _ = hair_mass(grid, solid, lm, p)

    k = UPSAMPLE
    vs = fine_voxel
    skin = upsample(solid, k)
    hair = upsample(hair_c, k)
    mat = np.zeros(skin.shape, np.uint8)
    col = np.zeros(skin.shape, np.uint8)
    mat[skin], col[skin] = M.MAT_SKIN, C("skin")

    # ---- fine details on the bare form (hair is added after, so it never hides a feature)
    flm = fine_landmarks(grid, lm, mat, vs)
    colors = {"skin": C("skin"), "skin_shadow": C("skin_shadow"), "skin_ruddy": C("skin_ruddy"),
              "lips": C("lips"), "line": C("line"), "brow": C("hair_deep"),
              "eye_white": C("eye_white"), "iris": C("iris"), "highlight": C("highlight")}
    stats = apply_face_details(mat, col, flm, colors, p.detail)

    # ---- hair: coarse mass, lit tops, deep lower edge, fine strands at the hairline and nape
    hair &= mat == M.MAT_AIR
    mat[hair], col[hair] = M.MAT_HAIR, C("hair")
    nz, ny, nx = mat.shape
    if p.hair and p.strands:
        X = x_coords(nx)
        filled = mat != M.MAT_AIR
        edge = hair & ~shift(filled, 0, 1, 0)                  # hair with air below
        ear_z0 = int(math.floor((lm["ear_z"][0] - grid.origin[2]) / vs))
        ear_z1 = int(math.floor((lm["ear_z"][1] - grid.origin[2]) / vs))
        zone = (np.arange(nz) < ear_z0 - 3) | (np.arange(nz) > ear_z1 + 3)
        pattern = (np.floor(np.abs(X)).astype(int) % 3) == 1
        strand = shift(edge & zone[:, None, None] & pattern[None, None, :], 0, -1, 0) & ~filled
        mat[strand], col[strand] = M.MAT_HAIR, C("hair_deep")
        stats["strands"] = int(strand.sum())
    filled = mat != M.MAT_AIR
    is_hair = mat == M.MAT_HAIR
    col[is_hair & ~shift(filled, 0, 1, 0)] = C("hair_deep")   # under-faces / lower edge
    crown = int(np.nonzero(is_hair.any(axis=(0, 2)))[0].max()) if is_hair.any() else 0
    lit = is_hair & ~shift(filled, 0, -1, 0) & (np.arange(ny)[None, :, None] >= crown - 5)
    col[lit] = C("hair_light")                                 # top-lit, crown only (no contour rings)

    # ---- tissues by depth (hair is air for the skin)
    depth = depth_map(filled & ~is_hair)
    skin = mat == M.MAT_SKIN
    for lvl, material, colour in ((2, M.MAT_FAT, "fat"), (3, M.MAT_MUSCLE, "muscle")):
        sel = skin & (depth == lvl)
        mat[sel], col[sel] = material, C(colour)
    sel = skin & (depth >= 4)
    mat[sel], col[sel] = M.MAT_BONE, C("bone")
    Xf = x_coords(nx)[None, None, :]
    rows = np.arange(ny)[None, :, None]
    zs = np.arange(nz)[:, None, None]
    nose = (np.abs(Xf) <= 3.5) & (rows >= flm.nose_base_y - 1) & (rows <= flm.eye_y - 1) & (zs <= int(depth.shape[0] * 0.5))
    ear = np.abs(Xf) >= lm["ear_inner_x"] / vs
    sel = skin & (depth >= 2) & (mat != M.MAT_BONE) & (nose | ear)
    mat[sel], col[sel] = M.MAT_CARTILAGE, C("cartilage")
    if p.stubble:
        chin_row = int(math.floor((float(lm["chin"][1]) - grid.origin[1]) / vs))
        beard = (rows <= flm.mouth_y - 2) | ((rows <= flm.nose_base_y - 2) & (np.abs(Xf) >= flm.mouth_half_w + 1))
        beard |= (rows >= flm.mouth_y + 2) & (rows <= flm.nose_base_y - 1) & (np.abs(Xf) <= flm.mouth_half_w)
        beard = beard & (rows >= chin_row - 2) & (zs <= flm.ear_z - 2) & ~ear
        sel = beard & (mat == M.MAT_SKIN) & (depth == 1) & (col == C("skin"))
        col[sel] = C("stubble")

    part = Part("head", (nx, ny, nz), vs,
                origin=(-nx * vs / 2, grid.origin[1] - float(lm["chin"][1]), -nz * vs / 2),
                bone="head", symmetric=True)
    part.material[:] = mat
    part.color[:] = col
    model = Model(name or "hybrid", vs, pal,
                  meta={"generator": "hybridhead", "coarse_voxel_m": p.coarse_voxel, "morphs": dict(p.morphs),
                        "form": info, "details": stats, "landmarks_fine": asdict(flm)})
    model.add_part(part)
    model.meta["crumbs_removed"] = keep_largest_component(part)
    return model
