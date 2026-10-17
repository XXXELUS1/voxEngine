"""Reference-style head: a minimal painted face on a coarse form, a designed hair part on top.

Why: the owner's reference is a voxel-art character whose face is *minimal on a coarse grid* -
big flat planes, the nose one step, the mouth a tiny dark line, two skin tones plus a painted
shade band, no relief, no noise - and whose hair reads by silhouette: a spiky quiff of blocks
with a few lighter highlights. He rejected voxelized realistic relief (``mid``, ``multires``)
and toon heads, and liked the 1 cm form of :mod:`voxgen.hybridhead` and facet smoothing.

Parts (``build_ref_head``):
  ``head``  5 mm. The morphed MakeHuman head voxelized at 1 cm (:func:`multires_head.build_form`:
            closing/opening, mirror, largest component, 26-neighbour quantization - no one-voxel
            terrace), the nose morphed so it reads as one or two clean steps
            (``nose-scale-depth`` 0.8, ``nose-scale-vert`` 0.6), a modest chin; upsampled x2;
            tissues by depth (:func:`head_from_mesh.layer_tissues`). The ears are the form's
            own, cleaned to blocks. Features are *painted*, never sculpted (:func:`paint_face`):
            mouth = a dark line ``mouth_rows`` (2) fine rows tall and 8 columns (4 cm) wide,
            optional lower-lip row 50 % toward the lip tone; nostrils = two dark voxels;
            shade band = the darker skin tone on the row under the eyes, on every under-face of
            the face (under the brow ridge, the nose, the chin) and on the neck below the chin;
            brows = bars 6 x 2 fine (3 x 1 coarse) in the hair tone, one coarse voxel above the
            eye, the outer third one fine row higher. Two skin tones + the shade tone, no
            stubble, no blush.
  eyes      ``eyes="dots"``: dark 4 x 2 fine (2 x 1 coarse) voxels painted on the form (classic
            voxel-art eyes). ``eyes="balls"``: ``eye_l`` / ``eye_r`` 3 mm eyeball parts
            (:func:`multires_head.eyeball`: radius ``eye_radius`` 9.5 mm, iris, pupil, one
            highlight) in carved sockets - the ball's front pole ``eye_protrude`` in front of the
            form's surface at the eye, the opening 3 columns x 2 rows, the socket walls dark, a
            one-row dark lid line above the opening, a stalk to the socket back.
  ``hair``  ``hair_voxel`` (5 mm default, 3 mm option), :func:`hairstyle.build_hair` on its own
            grid from the form resampled there; the form is carved one voxel deep under the hair
            (``pocket``, the surface voxels 6-adjacent to hair) and the hair fills the pocket, so
            the hair is embedded and 6-adjacent to the scalp. Tones: base, lit, deep.

Local frame as the other heads: metres, Y up, face toward -Z, chin at y = 0, centred on x / z;
``model.meta["mesh"]`` and ``model.meta["seams"]`` carry the cell / voxel / quad counts per part
and the overlap / contact counts (:func:`multires_head.mesh_stats`, :func:`seam_stats`).
Decisions: the coarse grid has an even number of columns (the nose is a 2 cm block, not a 1 cm
fin) and the eye height on a row boundary; eye and brow columns are centred on the nearest 5 mm
boundary, so every painted feature is whole coarse voxels wide; paint happens after the tissues
are layered (colour only) and the socket carve before (so socket walls are skin).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from . import materials as M
from . import meshvox, mh
from .body import dilate, exposed, shift
from .checks import label_components
from .facedetail import _cols, _rect, front_face, hole_walls, x_coords
from .hairstyle import STYLES, HairParams, build_hair, ear_zone
from .head_from_mesh import HeadParams, layer_tissues, upsample
from .hybridhead import fine_landmarks
from .model import Model, Part
from .multires_head import (MultiresParams, backfill_z, build_form, crop_part, eyeball, footprint, head_grid,
                            mesh_stats, resample, seam_stats)
from .palette import Palette
from .toonhead import HAIR_TONES, IRIS_TONES, SKIN_TONES

INF = float("inf")
EYE_OPTIONS = ("dots", "balls")
UPSAMPLE = 2

#: the nose as one or two clean steps at 1 cm, a modest chin, no brow ridge (big calm planes)
DEFAULT_MORPHS = {
    "nose-scale-depth": 0.8,
    "nose-scale-vert": 0.6,
    "chin-prominent": 0.15,
    "chin-width": 0.3,
}


@dataclass
class RefHeadParams:
    """Lengths in metres."""

    morphs: dict = field(default_factory=lambda: dict(DEFAULT_MORPHS))
    coarse_voxel: float = 0.01
    eyes: str = "dots"                  # "dots" | "balls"
    hair_style: str = "quiff"           # hairstyle.STYLES
    hair_voxel: float = 0.005
    hair: dict = field(default_factory=dict)   # HairParams overrides
    pocket: bool = True                 # carve the form one voxel deep under the hair
    margin: float = 0.02
    hair_room: float = 0.065            # grid room above / around the head for the hair
    quantize_passes: int = 6
    bump_max: int = 11
    pit_min: int = 15
    # -- painted face (fine voxels unless metres)
    mouth_rows: int = 2
    mouth_half_cols: float = 3.5        # |X| of the outer mouth column (3.5 -> 8 columns = 4 cm)
    lower_lip: bool = True              # one row under the line, 50 % toward the lip tone
    shade_eyes: bool = True             # the row under the eyes in the shade tone
    shade_underfaces: bool = True       # every under-face of the face in the shade tone
    shade_neck: bool = True             # the neck below the chin in the shade tone
    brow_gap_rows: int = 2              # fine rows between the eye and the brow (one coarse voxel)
    brow_tilt_cols: int = 2             # outer columns of the brow one row higher
    # -- eyeballs
    eye_radius: float = 0.0095
    iris_radius: float = 0.0045
    pupil_radius: float = 0.0018
    eye_voxel: float = 0.003
    eye_protrude: float = 0.0           # the ball's front pole this far in front of the form's surface (0 = flush)
    eye_open_cols: int = 4              # opening width in form columns
    eye_open_rows: int = 2              # opening height in form rows
    # -- colour
    skin_tone: str = "light"
    hair_tone: str = "black"
    iris_tone: str = "brown"
    tissues: HeadParams = field(default_factory=lambda: HeadParams(hair=False, stubble=False, blush=False))

    def __post_init__(self):
        if self.eyes not in EYE_OPTIONS:
            raise ValueError(f"eyes must be one of {EYE_OPTIONS}, got {self.eyes!r}")
        if self.hair_style not in STYLES:
            raise ValueError(f"hair_style must be one of {STYLES}, got {self.hair_style!r}")


def _mix(a, b, t: float) -> tuple[int, int, int]:
    return tuple(int(round((1 - t) * x + t * y)) for x, y in zip(a, b))


def make_palette(p: RefHeadParams) -> Palette:
    skin, shadow, _ruddy = SKIN_TONES[p.skin_tone]
    hair, _ = HAIR_TONES[p.hair_tone]
    lips = _mix(skin, (168, 70, 78), 0.42)
    pal = Palette()
    pal.add("skin", skin)
    pal.add("skin_shadow", shadow)
    pal.add("lip_soft", _mix(skin, lips, 0.5))
    pal.add("line", (70, 42, 44))
    pal.add("eye_dark", (34, 28, 32))
    pal.add("hair", hair)
    pal.add("hair_light", _mix(hair, (255, 250, 245), 0.3))
    pal.add("hair_deep", tuple(int(round(c * 0.62)) for c in hair))
    pal.add("eye_white", (242, 242, 236))
    pal.add("iris", IRIS_TONES.get(p.iris_tone, IRIS_TONES["brown"]))
    pal.add("pupil", (24, 22, 30))
    pal.add("highlight", (248, 250, 252))
    pal.add("fat", (238, 214, 140))
    pal.add("muscle", (150, 40, 45))
    pal.add("bone", (235, 225, 200))
    pal.add("tooth", (245, 240, 220))
    pal.add("tooth_dark", (232, 224, 198))
    pal.add("cartilage", (225, 185, 160))
    pal.add("vessel", (120, 20, 30))
    return pal


# ---------------------------------------------------------------------------- painted face
def paint_face(mat: np.ndarray, col: np.ndarray, flm, lm: dict, grid: meshvox.VoxelGrid, C, p: RefHeadParams,
               hair_fp: np.ndarray | None = None) -> dict[str, int]:
    """Colour-only features on the skin of the fine form (``mat`` / ``col`` in place): mouth line,
    lower lip, nostrils, shade band, brows and - for ``eyes="dots"`` - the eye dots. ``hair_fp``
    (bool, same grid) keeps paint from under the hair."""
    nz, ny, nx = mat.shape
    vs = grid.voxel_size
    X = x_coords(nx)
    stats: dict[str, int] = {}
    skin = mat == M.MAT_SKIN
    filled = mat != M.MAT_AIR
    front = front_face(filled) & skin
    if hair_fp is not None:
        front &= ~hair_fp
    ex = int(round(float(lm["eye_x"]) / vs))                    # eye centre on a column boundary
    ey = int(flm.eye_y)

    # ---- mouth: a dark line, optional lower-lip row
    my = int(flm.mouth_y)
    mouth = front & _rect(mat.shape, _cols(X, 0, p.mouth_half_cols), my - p.mouth_rows + 1, my)
    col[mouth] = C("line")
    stats["mouth"] = int(mouth.sum())
    if p.lower_lip:
        lip = front & _rect(mat.shape, _cols(X, 0, p.mouth_half_cols - 1), my - p.mouth_rows, my - p.mouth_rows)
        col[lip] = C("lip_soft")
        stats["lower_lip"] = int(lip.sum())

    # ---- nostrils: two dark voxels at the nose base
    nostril = front & _rect(mat.shape, _cols(X, flm.nostril_x, flm.nostril_x), flm.nose_base_y, flm.nose_base_y)
    col[nostril] = C("line")
    stats["nostril"] = int(nostril.sum())

    # ---- shade band: under the eyes, the under-faces of the face, the neck
    shade = np.zeros(mat.shape, bool)
    if p.shade_eyes:
        shade |= front & _rect(mat.shape, _cols(X, ex - 2.5, ex + 2.5), ey - 2, ey - 2)
    if p.shade_underfaces:
        under = skin & ~shift(filled, 0, 1, 0)                   # air below
        zs = np.arange(nz)[:, None, None]
        rows = np.arange(ny)[None, :, None]
        chin_row = int(math.floor((float(lm["chin"][1]) - grid.origin[1]) / vs))
        face_z = int(math.floor((lm["ear_z"][0] - grid.origin[2]) / vs))
        under &= (zs < face_z) & (rows >= chin_row - 1) & (rows <= ey + 8)
        shade |= under
    if p.shade_neck:
        chin_row = int(math.floor((float(lm["chin"][1]) - grid.origin[1]) / vs))
        rows = np.arange(ny)[None, :, None]
        shade |= exposed(filled) & skin & (rows < chin_row - 1)
    shade &= col == C("skin")
    col[shade] = C("skin_shadow")
    stats["shade"] = int(shade.sum())

    # ---- brows: 6 x 2 bars in the hair tone, the outer columns one row higher
    b0 = ey + p.brow_gap_rows + 1                               # the eye's top row is ey
    inner = _cols(X, ex - 1.5, ex + 3.5 - p.brow_tilt_cols)
    outer = _cols(X, ex + 3.5 - p.brow_tilt_cols + 1, ex + 3.5)
    brow = front & (_rect(mat.shape, inner, b0, b0 + 1) | _rect(mat.shape, outer, b0 + 1, b0 + 2))
    col[brow] = C("hair")
    stats["brow"] = int(brow.sum())

    # ---- eyes as dots: 4 x 2 dark voxels
    if p.eyes == "dots":
        dots = front & _rect(mat.shape, _cols(X, ex - 1.5, ex + 1.5), ey - 1, ey)
        col[dots] = C("eye_dark")
        stats["eye_dots"] = int(dots.sum())
    return stats


# ---------------------------------------------------------------------------- eyeballs
def carve_sockets(form: np.ndarray, grid: meshvox.VoxelGrid, lm: dict, p: RefHeadParams, C) -> tuple[np.ndarray, dict, np.ndarray]:
    """Carve the eye sockets of the fine form for the 3 mm eyeballs. Returns (carved form,
    {name: (egrid, mat, col, ball, centre)}, inner socket air for the dark walls)."""
    X, Y, Z = grid.centres()
    vs = grid.voxel_size
    mp = MultiresParams(detail_voxel=p.eye_voxel, eye_radius=p.eye_radius, iris_radius=p.iris_radius,
                        pupil_radius=p.pupil_radius)
    shell = exposed(form)
    front = front_face(form)
    carved = form.copy()
    balls = {}
    cavity_all = np.zeros(form.shape, bool)
    opening_all = np.zeros(form.shape, bool)
    half_w = p.eye_open_cols * vs / 2
    half_h = p.eye_open_rows * vs / 2
    for key, lmk in (("eye_l", "eye_l"), ("eye_r", "eye_r")):
        c0 = np.asarray(lm[lmk], np.float64)
        cx = (math.floor(c0[0] / vs) + 0.5) * vs if p.eye_open_cols % 2 else round(c0[0] / vs) * vs
        cy = (math.floor(c0[1] / vs) + 0.5) * vs if p.eye_open_rows % 2 else round(c0[1] / vs) * vs
        cols = (np.abs(X - cx) <= half_w - 1e-6) & (np.abs(Y - cy) <= half_h - 1e-6)
        zf = float(Z[front & cols].min())                        # the form's front surface at the eye
        cz = zf - vs / 2 - p.eye_protrude + p.eye_radius
        centre = np.array([cx, cy, cz])
        egrid, emat, ecol, ball = eyeball(centre, mp, C)
        cavity = footprint(ball, egrid, grid)
        opening = cols & (Z <= cz) & form
        cavity_all |= cavity & ~shell
        opening_all |= opening
        balls[key] = (egrid, emat, ecol, ball, centre)
    carved &= ~(cavity_all | opening_all)
    carved &= carved[:, :, ::-1]
    rim = shell & carved
    inner = (cavity_all | opening_all) & ~carved & ~dilate(rim)
    return carved, balls, inner


# ---------------------------------------------------------------------------- the builder
def build_ref_head(params: RefHeadParams | None = None, name: str | None = None,
                   human: mh.MakeHuman | None = None) -> Model:
    p = params or RefHeadParams()
    human = human or mh.load(morphs=p.morphs)
    lm = human.landmarks()
    pal = make_palette(p)
    C = pal.index
    cut_y = float(lm["neck_base"][1])
    chin_y = float(lm["chin"][1])
    stats: dict = {}

    # ---- the coarse form (1 cm), upsampled to the fine grid (5 mm)
    room = HeadParams(hair=True, hair_thickness=p.hair_room - 0.004)
    mp = MultiresParams(form_voxel=p.coarse_voxel, head=room, margin=p.margin, quantize_passes=p.quantize_passes,
                        bump_max=p.bump_max, pit_min=p.pit_min)
    cgrid, form_c, finfo = build_form(human, lm, mp)
    k = UPSAMPLE
    vs = p.coarse_voxel / k
    fgrid = meshvox.VoxelGrid(cgrid.origin, vs, tuple(d * k for d in cgrid.dims))
    form = upsample(form_c, k)
    stats["form"] = finfo
    offset = (0.0, chin_y, fgrid.origin[2] + fgrid.dims[2] * vs / 2)

    # ---- hair on its own grid; the pocket under it
    hgrid = fgrid if abs(p.hair_voxel - vs) < 1e-9 else head_grid(human, lm, room, p.hair_voxel, p.margin)
    form_h = form if hgrid is fgrid else resample(form, fgrid, hgrid)
    hres = build_hair(form_h, hgrid, lm, p.hair_style, **p.hair)
    hair_h = hres.hair
    hair_fp = footprint(hair_h, hgrid, fgrid) if hgrid is not fgrid else hair_h
    pocket = np.zeros(form.shape, bool)
    if p.pocket:
        Xf, Yf, Zf = fgrid.centres()
        under = form & dilate(hair_fp) & ~ear_zone(Xf, Yf, Zf, lm)     # surface voxels 6-adjacent to hair
        pocket = under | under[:, :, ::-1]
        form = form & ~pocket
        if hgrid is fgrid:
            hair_h = hair_h | pocket
        else:
            hair_h = hair_h | (resample(pocket, fgrid, hgrid) & ~resample(form, fgrid, hgrid))
        hair_fp = hair_fp | pocket
    stats["hair"] = dict(hres.stats, pocket=int(pocket.sum()))

    # ---- sockets (before the tissues, so their walls are skin)
    balls: dict = {}
    inner = np.zeros(form.shape, bool)
    if p.eyes == "balls":
        form, balls, inner = carve_sockets(form, fgrid, lm, p, C)
    labels, n = label_components(form)
    if n > 1:
        form = labels == 1

    # ---- tissues, then the painted face
    mat, col = layer_tissues(form, fgrid, lm, p.tissues, human, C, cut_y)
    skin_mask = mat == M.MAT_SKIN
    probe = np.where(form, M.MAT_SKIN, 0).astype(np.uint8)
    flm = fine_landmarks(fgrid, lm, probe, vs)
    stats["face"] = paint_face(mat, col, flm, lm, fgrid, C, p, hair_fp=hair_fp)
    if inner.any():
        col[hole_walls(inner, skin_mask)] = C("line")
    if p.eyes == "balls":
        # one-row dark lid line on the front face right above each opening
        X = x_coords(form.shape[2])
        xc = math.floor(float(lm["eye_x"]) / vs) + 0.5           # the opening's centre column
        ey = int(flm.eye_y)
        lid = front_face(form) & skin_mask & _rect(form.shape, _cols(X, xc - 1, xc + 1), ey + 1, ey + 1)
        col[lid] = C("line")
        stats["face"]["lid"] = int(lid.sum())

    model = Model(name or "ref_head", voxel_size=vs, palette=pal,
                  meta={"source": "makehuman hm08 base.obj (CC0)", "generator": "refhead",
                        "params": asdict(p),
                        "morphs": dict(p.morphs), "landmarks_fine": asdict(flm)})
    head = Part("head", fgrid.dims, vs, origin=tuple(o - d for o, d in zip(fgrid.origin, offset)),
                bone="head", symmetric=True, budget=250_000)
    head.material[:] = mat
    head.color[:] = col
    model.add_part(head)

    # ---- eyeballs with their stalks
    for key, (egrid, emat, ecol, ball, centre) in balls.items():
        form_e = resample(form, fgrid, egrid)
        Xe, Ye, Ze = egrid.centres()
        vse = egrid.voxel_size
        axis = (np.abs(Xe - centre[0]) <= vse) & (np.abs(Ye - centre[1]) <= vse)
        back = ball & ~shift(ball, 0, 0, -1) & axis & (Ze > centre[2])
        stalk = backfill_z(back, form_e, 8) & ~ball
        emat[stalk] = M.MAT_EYE
        ecol[stalk] = C("eye_white")
        model.add_part(crop_part(key, egrid, emat, ecol, offset, key, False))
        stats[key] = {"ball": int(ball.sum()), "stalk": int(stalk.sum())}

    # ---- the hair part
    hmat = np.where(hair_h, M.MAT_HAIR, 0).astype(np.uint8)
    hcol = np.where(hair_h, C("hair"), 0).astype(np.uint8)
    hcol[hres.deep & hair_h] = C("hair_deep")
    hcol[hres.lit & hair_h] = C("hair_light")
    model.add_part(crop_part("hair", hgrid, hmat, hcol, offset, "head", True))

    model.meta["build"] = stats
    model.meta["seams"] = seam_stats(model)
    model.meta["mesh"] = mesh_stats(model)
    return model
