"""Body voxelized from the MakeHuman hm08 base body (CC0, :mod:`voxgen.mh`) at 10 mm, headless,
with clothes as colour and one-voxel shells: the town characters' bodies.

Why: the owner wants whole characters in one style, not heads on a stand. The body is the
part where a 10 mm voxel is the right size (the reference's body is blocky: hands are mitts,
boots are blocks) and where MakeHuman's proportions are already right, so it is *sampled*, not
designed: the base body (A-pose as it comes, forearms forward, feet on y = 0, facing -Z),
optionally morphed by MakeHuman modifiers (``BUILDS``: ``lean`` / ``broad`` / ``plump`` /
``female`` are modifier sets of :mod:`voxgen.mh_targets`, the macro gender / weight targets
are not available), voxelized and flood-filled, cleaned the hybrid way (3x3x3 closing then
opening, mirror union, 26-neighbour quantization until no one-wide step is left, the largest
component), the head cut off ``neck_cut`` above the neck joint (the stub's top becomes skin:
the sculpted head's own neck stands on it, see :mod:`voxgen.character`). Tissues by depth in
metres (skin ``skin_m``, fat to ``fat_m``, muscle, bone where deeper than ``bone_depth_m``:
a coarse core in the torso and thighs, none in the forearms).

Clothes (:func:`clothes`): every garment is 2-3 flat tones, no pattern smaller than a voxel.
  paint   the body's skin voxels recoloured: ``tunic`` (torso and sleeves to the elbow or the
          wrist, a painted belt), trousers (legs under the tunic's hem), boots (``boot_top``),
          a few 3 x 3 patches for a patched tunic, apron straps;
  shells  a ``clothes`` part (cloth / leather) of voxels one 10 mm voxel proud of the body (a
          26-neighbour dilation, so the shell is one 6-connected piece round every corner):
          ``apron`` (the front of the torso, chest to the knee, a narrower bib), ``robe`` (shoulders
          to the ankles over torso and arms, a skirt filled between the legs, a rope belt),
          ``gambeson`` (torso and arms to the hip, horizontal quilting rows, a leather belt),
          ``dress`` (collarbone to the ankles with the skirt, sleeves to the elbow, a waist band).
  tones   ``base``, ``shade`` (the hem and the sleeves' undersides, the quilting rows) and
          ``trim`` (belt / band / straps); hands always skin, the head's neck never covered.
Landmarks come from the MakeHuman joints (metres, after the morphs), so a plump body keeps
its belt at its waist.

Decisions (the specs are silent on bodies): one 10 mm part ``body`` (bone ``body``) for now -
limb parts and 5 mm hands come with the skeleton; the clothes shell is a separate part so the
razor and the engine can treat cloth apart from skin; nothing in the body is asymmetric.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import materials as M
from . import meshvox, mh
from .body import exposed, shift
from .checks import label_components
from .head_from_mesh import box_dilate, clean_form, depth_map
from .hybridhead import quantize_form
from .model import Part
from .multires_head import crop_part
from .palette import Palette

Vec3 = tuple[float, float, float]
RGB = tuple[int, int, int]
BUILDS = ("normal", "lean", "broad", "plump", "female")
OUTFITS = ("tunic", "apron", "robe", "gambeson", "dress")
#: MakeHuman modifier sets of the builds (``voxgen.mh_targets`` names; l-/r- pairs expand)
BUILD_MORPHS: dict[str, dict[str, float]] = {
    "normal": {},
    "lean": {"torso-scale-horiz": -0.35, "torso-scale-depth": -0.2, "stomach-tone": 0.5, "upperarm-fat": -0.3,
             "upperleg-fat": -0.3, "hip-scale-horiz": -0.2},
    "broad": {"torso-scale-horiz": 0.5, "torso-vshape": 0.6, "torso-muscle-pectoral": 0.5, "torso-scale-depth": 0.2,
              "upperarm-muscle": 0.5, "upperarm-shoulder-muscle": 0.6, "lowerarm-muscle": 0.3, "neck-scale-horiz": 0.4,
              "upperleg-muscle": 0.3},
    "plump": {"torso-scale-horiz": 0.3, "torso-scale-depth": 0.4, "stomach-pregnant": 0.6, "stomach-tone": -0.5,
              "hip-scale-horiz": 0.4, "buttocks-volume": 0.4, "upperarm-fat": 0.4, "upperleg-fat": 0.4,
              "neck-double": 0.5, "neck-scale-horiz": 0.3},
    "female": {"torso-scale-horiz": -0.25, "torso-vshape": -0.3, "hip-scale-horiz": 0.3, "buttocks-volume": 0.3,
               "upperarm-scale-horiz": -0.2, "neck-scale-horiz": -0.2, "upperarm-fat": -0.2},
}
#: default garment tones (sRGB) per outfit: base, shade, trim; trousers, boots and patches are shared
GARMENT_TONES: dict[str, dict[str, RGB]] = {
    "tunic": {"base": (120, 92, 68), "shade": (92, 68, 50), "trim": (74, 48, 30)},
    "apron": {"base": (226, 214, 190), "shade": (186, 172, 150), "trim": (186, 172, 150)},
    "robe": {"base": (96, 72, 52), "shade": (70, 52, 38), "trim": (178, 158, 112)},
    "gambeson": {"base": (138, 110, 70), "shade": (104, 82, 52), "trim": (74, 48, 30)},
    "dress": {"base": (52, 54, 66), "shade": (36, 38, 48), "trim": (120, 116, 128)},
}
COMMON_TONES: dict[str, RGB] = {"trousers": (62, 58, 56), "boots": (46, 36, 30), "patches": (84, 98, 76),
                                "under": (84, 104, 128)}


@dataclass
class BodyParams:
    """Lengths in metres."""
    voxel: float = 0.010
    build: str = "normal"            # one of BUILDS
    morphs: dict = field(default_factory=dict)   # extra MakeHuman modifiers on top of the build's
    skin_tone: str = "light"         # used only when build_body makes its own palette
    neck_cut: float = 0.015          # the stub's top this far above the neck joint
    clean: bool = True               # closing / opening + quantization
    skin_m: float = 0.010
    fat_m: float = 0.020
    bone_depth_m: float = 0.040      # bone where deeper than this
    margin: float = 0.02             # grid margin around the mesh
    boot_top: float = 0.17           # painted boots up to here (0 = barefoot)


@dataclass
class Outfit:
    """A garment set: ``kind`` (one of OUTFITS), its tones (``base`` / ``shade`` / ``trim``,
    defaults from GARMENT_TONES), the painted extras (``trousers``, ``boots``, ``under`` = the
    tunic under an apron, ``patches``), sleeves of the painted tunic (``"elbow"`` / ``"wrist"``
    / ``"none"``), ``patches`` for a patched tunic, ``belt`` for a painted / leather belt."""
    kind: str = "tunic"
    tones: dict[str, RGB] = field(default_factory=dict)
    sleeves: str = "elbow"
    patches: bool = False
    belt: bool = True

    def tone(self, key: str) -> RGB:
        if key in self.tones:
            return self.tones[key]
        if key in GARMENT_TONES.get(self.kind, {}):
            return GARMENT_TONES[self.kind][key]
        return COMMON_TONES[key]


@dataclass
class BodyResult:
    part: Part
    grid: meshvox.VoxelGrid
    solid: np.ndarray
    landmarks: dict
    stats: dict
    params: BodyParams
    human: mh.MakeHuman


# ---------------------------------------------------------------------------- the form
def build_morphs(p: BodyParams) -> dict[str, float]:
    if p.build not in BUILDS:
        raise ValueError(f"build must be one of {BUILDS}, got {p.build!r}")
    out = dict(BUILD_MORPHS[p.build])
    for k, w in p.morphs.items():
        out[k] = out.get(k, 0.0) + float(w)
    return {k: w for k, w in out.items() if w != 0.0}


def body_grid(human: mh.MakeHuman, p: BodyParams) -> meshvox.VoxelGrid:
    """The 10 mm grid over the body's bounds: y from the ground, x symmetric with an *even*
    column count (the mirror plane on a voxel boundary, as the head's)."""
    lo, hi = human.body().bounds()
    vs = p.voxel
    half = max(abs(lo[0]), abs(hi[0])) + p.margin
    nx = 2 * int(math.ceil(half / vs - 1e-6))
    ny = int(math.ceil((hi[1] + p.margin) / vs - 1e-6))
    z0 = lo[2] - p.margin
    nz = int(math.ceil((hi[2] + p.margin - z0) / vs - 1e-6))
    return meshvox.VoxelGrid((-nx * vs / 2, 0.0, float(z0)), vs, (nx, ny, nz))


def body_landmarks(human: mh.MakeHuman, p: BodyParams) -> dict:
    """Metre landmarks of the body from the joints: the neck cut (``neck_top``), the torso and
    the limb levels the clothes are cut at."""
    j = human.joint
    neck = j("neck")
    lm = {
        "neck_y": float(neck[1]), "neck_z": float(neck[2]),
        "neck_top": float(neck[1] + p.neck_cut),
        "shoulder_y": float(j("l-shoulder")[1]), "shoulder_x": float(abs(j("l-shoulder")[0])),
        "chest_y": float(j("spine-1")[1]), "waist_y": float(j("spine-2")[1]) - 0.02,
        "hip_y": float(j("pelvis")[1]), "crotch_y": float(j("l-upper-leg")[1]) - 0.06,
        "knee_y": float(j("l-knee")[1]), "ankle_y": float(j("l-ankle")[1]),
        "elbow_y": float(j("l-elbow")[1]), "wrist_y": float(j("l-hand")[1]) + 0.025,
        "torso_z": float(j("spine-3")[2]),
    }
    lm["torso_half_w"] = lm["shoulder_x"] + 0.02
    return lm


def voxelize_body(human: mh.MakeHuman, grid: meshvox.VoxelGrid) -> tuple[np.ndarray, dict]:
    """Shell + flood-filled interior of the body mesh (the ground face sealed)."""
    solid, info = meshvox.voxelize_solid(human.body(), grid, sealed_faces=("-y",))
    if info["leak"]:
        raise RuntimeError(f"the body mesh leaked while flood filling: {info}")
    return solid, info


def clean_body(solid: np.ndarray) -> tuple[np.ndarray, dict]:
    """Closing then opening (thin sheets kept), mirror union, 26-neighbour quantization, the
    largest component."""
    out = clean_form(solid)
    out |= out[:, :, ::-1]
    out, qinfo = quantize_form(out)
    labels, n = label_components(out)
    crumbs = 0
    if n > 1:
        crumbs = int((labels > 1).sum())
        out = labels == 1
    return out, {**qinfo, "components": int(n), "crumbs_removed": crumbs}


def cut_neck(solid: np.ndarray, grid: meshvox.VoxelGrid, y_cut: float) -> np.ndarray:
    """Everything with its centre above ``y_cut`` removed (the head and the top of the neck)."""
    X, Y, Z = grid.centres()
    return solid & (Y < y_cut)


def layer_body(mat: np.ndarray, col: np.ndarray, solid: np.ndarray, p: BodyParams, C) -> dict[str, int]:
    """Skin / fat / muscle / bone by depth in metres (``skin_m``, ``fat_m``, ``bone_depth_m``),
    in place. The cut face counts as surface, so the stub's top is skin."""
    vs = p.voxel
    depth = depth_map(solid)
    d_skin = int(math.ceil(p.skin_m / vs - 1e-6))
    d_fat = int(math.ceil(p.fat_m / vs - 1e-6))
    d_bone = int(math.ceil(p.bone_depth_m / vs - 1e-6))
    mat[solid] = M.MAT_SKIN
    col[solid] = C("skin")
    out = {}
    for material, sel, colour in ((M.MAT_FAT, (depth > d_skin) & (depth <= d_fat), "fat"),
                                  (M.MAT_MUSCLE, (depth > d_fat) & (depth <= d_bone), "muscle"),
                                  (M.MAT_BONE, depth > d_bone, "bone")):
        sel &= solid
        mat[sel], col[sel] = material, C(colour)
        out[colour] = int(sel.sum())
    out["skin"] = int((mat == M.MAT_SKIN).sum())
    return out


def _ensure_palette(palette: Palette | None, p: BodyParams) -> Palette:
    if palette is not None:
        return palette
    from .toonhead import ToonHeadParams, make_palette
    return make_palette(ToonHeadParams.preset_params("box", skin_tone=p.skin_tone))


def build_body(params: BodyParams | None = None, palette: Palette | None = None, human: mh.MakeHuman | None = None) -> BodyResult:
    """The headless 10 mm body part (bone ``body``, x-centred, feet on y = 0, facing -Z) with
    its grid, solid mask, landmarks and stats. ``palette`` (the character's) must name ``skin``,
    ``skin_shadow``, ``fat``, ``muscle``, ``bone``; without one the box palette of ``skin_tone``
    is made."""
    p = params if params is not None else BodyParams()
    pal = _ensure_palette(palette, p)
    C = pal.index
    human = human if human is not None else mh.load(morphs=build_morphs(p))
    grid = body_grid(human, p)
    lm = body_landmarks(human, p)
    solid, vinfo = voxelize_body(human, grid)
    stats: dict = {"voxelize": vinfo, "morphs": build_morphs(p)}
    if p.clean:
        solid, cinfo = clean_body(solid)
        stats["clean"] = cinfo
    solid = cut_neck(solid, grid, lm["neck_top"])
    solid &= solid[:, :, ::-1]
    labels, n = label_components(solid)
    if n > 1:
        stats["crumbs_after_cut"] = int((labels > 1).sum())
        solid = labels == 1
    mat = np.zeros(solid.shape, np.uint8)
    col = np.zeros(solid.shape, np.uint8)
    stats["tissues"] = layer_body(mat, col, solid, p, C)
    # the shade tone where the body shadows itself: faces with solid above and air below (the
    # chin of the stub is the head's job; here: under the arms, the crotch, under the belly)
    under = (mat == M.MAT_SKIN) & ~shift(solid, 0, 1, 0) & shift(solid, 0, -1, 0)
    col[under] = C("skin_shadow")
    part = crop_part("body", grid, mat, col, (0.0, 0.0, 0.0), "body", True)
    pgrid = meshvox.VoxelGrid(tuple(part.origin), part.voxel_size, part.dims)
    stats["solid"] = int(solid.sum())
    return BodyResult(part, pgrid, part.solid(), lm, stats, p, human)


# ---------------------------------------------------------------------------- clothes
def _regions(body: BodyResult) -> dict[str, np.ndarray]:
    """Bool masks on the body part's grid: ``torso`` (|x| within the torso width, above the
    crotch), ``arms`` (outside it, above the crotch), ``hands`` (arms below the wrist), ``legs``."""
    X, Y, Z = body.grid.centres()
    lm = body.landmarks
    solid = body.solid
    ax = np.abs(X)
    above = Y >= lm["crotch_y"]
    torso = solid & above & (ax <= lm["torso_half_w"])
    arms = solid & above & (ax > lm["torso_half_w"])
    hands = arms & (Y < lm["wrist_y"])
    legs = solid & ~above
    return {"torso": torso, "arms": arms, "hands": hands, "legs": legs, "X": X, "Y": Y, "Z": Z}


def skirt_fill(cover: np.ndarray, solid: np.ndarray, Y: np.ndarray, y0: float, y1: float) -> np.ndarray:
    """Cloth between the legs: on every (y, z) line of rows ``y0 <= y < y1`` that has cloth or
    body on both sides of x = 0, every air voxel between the outermost ones (an A-line skirt)."""
    nz, ny, nx = cover.shape
    occ = cover | solid
    rows = (Y[0, :, 0] >= y0) & (Y[0, :, 0] < y1)
    has = occ.any(axis=2)                                       # (nz, ny)
    first = np.argmax(occ, axis=2)                              # leftmost occupied x
    last = nx - 1 - np.argmax(occ[:, :, ::-1], axis=2)          # rightmost
    xi = np.arange(nx)[None, None, :]
    left_side = (occ & (xi < nx // 2)).any(axis=2)
    right_side = (occ & (xi >= nx // 2)).any(axis=2)
    ok = has & left_side & right_side & rows[None, :]
    fill = ok[:, :, None] & (xi >= first[:, :, None]) & (xi <= last[:, :, None]) & ~occ
    return fill


def clothes(body: BodyResult, outfit: Outfit, palette: Palette) -> tuple[Part | None, dict]:
    """Paint the garments on the body part (in place) and return the shell part ``clothes``
    (None for a painted-only outfit) with stats. See the module doc for the garments."""
    if outfit.kind not in OUTFITS:
        raise ValueError(f"outfit must be one of {OUTFITS}, got {outfit.kind!r}")
    part = body.part
    lm = body.landmarks
    vs = part.voxel_size
    reg = _regions(body)
    X, Y, Z = reg["X"], reg["Y"], reg["Z"]
    solid = body.solid
    skin = part.material == M.MAT_SKIN
    surface = exposed(solid) & skin
    col = part.color
    kind = outfit.kind
    name = lambda key: f"{kind}_{key}"
    C = lambda key: palette.add(name(key), outfit.tone(key))
    stats: dict = {"kind": kind}
    tunic_hem = lm["crotch_y"] - 0.05
    sleeve_y = {"elbow": lm["elbow_y"], "wrist": lm["wrist_y"], "none": lm["neck_top"]}[outfit.sleeves]

    # ---- painted garments on the skin: trousers, boots, the tunic (always under an apron)
    boot_top = body.params.boot_top
    legs_or_low = solid & (Y < tunic_hem)
    col[surface & legs_or_low & (Y >= boot_top)] = palette.add("trousers", outfit.tone("trousers"))
    boots = surface & (Y < boot_top)
    col[boots] = palette.add("boots", outfit.tone("boots"))
    stats["boots"] = int(boots.sum())
    if kind in ("tunic", "apron"):
        tone_key = "under" if kind == "apron" else "base"
        tunic = surface & ((reg["torso"] & (Y >= tunic_hem)) | (reg["arms"] & (Y >= sleeve_y)))
        col[tunic] = palette.add("tunic_base" if kind == "tunic" else "apron_under", outfit.tone(tone_key))
        shade_rgb = outfit.tone("shade") if kind == "tunic" else _darker(outfit.tone("under"))
        col[tunic & ~shift(solid, 0, 1, 0)] = palette.add("tunic_shade", shade_rgb)
        stats["tunic"] = int(tunic.sum())
        if outfit.patches:
            patches = np.zeros(solid.shape, bool)
            for px, py in ((0.09, lm["waist_y"] + 0.10), (-0.05, lm["waist_y"] - 0.01), (0.14, lm["elbow_y"] + 0.04)):
                patches |= (np.abs(X - px) <= 1.5 * vs) & (np.abs(Y - py) <= 1.5 * vs)
            patches = tunic & (patches | patches[:, :, ::-1]) & (Z < lm["torso_z"])
            col[patches] = palette.add("patches", outfit.tone("patches"))
            stats["patches"] = int(patches.sum())
        if outfit.belt and kind == "tunic":
            belt = tunic & (np.abs(Y - lm["waist_y"]) <= vs)
            col[belt] = C("trim")
            stats["belt"] = int(belt.sum())

    # ---- shells
    shell = np.zeros(solid.shape, bool)
    mat_shell = np.full(solid.shape, M.MAT_CLOTH, np.uint8)
    around = box_dilate(solid) & ~solid            # 26-dilation: the shell stays 6-connected round corners
    not_hands = ~box_dilate(reg["hands"]) | box_dilate(reg["torso"] | reg["legs"])
    if kind == "apron":
        bib = (np.abs(X) <= 0.10) & (Y >= lm["waist_y"] + 0.04)
        skirt = (np.abs(X) <= 0.15) & (Y < lm["waist_y"] + 0.04)
        front = box_dilate(reg["torso"]) & (Z < lm["torso_z"] - 0.01)
        shell = around & front & (bib | skirt) & (Y <= lm["chest_y"] + 0.06) & (Y >= lm["knee_y"] + 0.03)
        straps = surface & reg["torso"] & (np.abs(np.abs(X) - 0.085) <= vs * 0.5) & (Y > lm["chest_y"] + 0.06) \
            & (Z < lm["torso_z"] + 0.02)
        col[straps] = C("base")
        stats["straps"] = int(straps.sum())
    elif kind == "robe":
        shell = around & (Y < lm["neck_top"]) & (Y >= lm["ankle_y"] + 0.03) & not_hands
        shell |= skirt_fill(shell, solid, Y, lm["ankle_y"] + 0.03, lm["crotch_y"] + 0.02)
    elif kind == "gambeson":
        shell = around & (Y < lm["neck_top"]) & (Y >= lm["hip_y"] - 0.03) & not_hands
    elif kind == "dress":
        top = lm["shoulder_y"] + 0.07                   # a high dress: the shoulders covered, the neck stub bare
        shell = around & (Y < top) & (Y >= lm["ankle_y"] + 0.03) & not_hands
        shell &= ~(box_dilate(reg["arms"]) & ~box_dilate(reg["torso"]) & (Y < lm["elbow_y"]))
        shell |= skirt_fill(shell, solid, Y, lm["ankle_y"] + 0.03, lm["crotch_y"] + 0.02)
    shell |= shell[:, :, ::-1]
    shell &= ~solid
    if not shell.any():
        return None, stats
    labels, n = label_components(shell)
    if n > 1:
        stats["shell_crumbs"] = int((labels > 1).sum())
        shell = labels == 1
    scol = np.where(shell, C("base"), 0).astype(np.uint8)
    # the shade tone: the hem (the lowest 2.5 cm of the garment) and the sleeves' undersides -
    # not every terrace of the skirt, which would turn the robe into a contour map
    under = shell & ~shift(shell | solid, 0, 1, 0)
    y_min = float(Y[shell].min())
    shade = under & ((Y < y_min + 0.025) | box_dilate(reg["arms"]))
    scol[shade] = C("shade")
    if kind == "gambeson":
        rows = np.arange(shell.shape[1])
        quilt = shell & ((rows[None, :, None] - int(round((lm["waist_y"] - part.origin[1]) / vs))) % 5 == 0)
        scol[quilt & ~shade] = C("shade")
        stats["quilt"] = int(quilt.sum())
    if outfit.belt and kind in ("robe", "gambeson", "dress"):
        belt = shell & (np.abs(Y - lm["waist_y"]) <= vs) & (np.abs(X) <= lm["torso_half_w"] + vs)
        scol[belt] = C("trim")
        if kind == "gambeson":
            mat_shell[belt] = M.MAT_LEATHER
        stats["belt"] = int(belt.sum())
    smat = np.where(shell, mat_shell, 0).astype(np.uint8)
    stats["shell"] = int(shell.sum())
    cpart = crop_part("clothes", body.grid, smat, scol, (0.0, 0.0, 0.0), "body", True)
    return cpart, stats


def _darker(rgb: RGB, f: float = 0.74) -> RGB:
    return tuple(int(round(c * f)) for c in rgb)
