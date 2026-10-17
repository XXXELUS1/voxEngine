"""Head built from hand-drawn relief maps: the maps are the model (``specs/chars/vox/barber_maps.py``,
editable like pixel art), this module is the engine (2026-10-17).

The unit grid (U = 20 mm, ``FINE`` = 4 fine voxels of 5 mm per unit), the ``Block`` / ``Frame`` /
``Builder`` primitives and the tissue palette come from :mod:`voxgen.blockhead`; nothing of its
head form is used any more. The frame's extent is computed from the maps (the union of every
layer's box plus ``FRAME_MARGIN``, rounded outward to whole units) and the model's metre origin
puts x = 0 on the mirror plane, y = 0 at the chin's bottom skin row and z = 0 at the skull's
mid-depth (``SKULL_MID_Z``: halfway between the maps' cheek plane z = 0 and the back of the skull
z = 13), so the head turns about its own centre in the lab.

A map layer (see the maps module's docstring for the keys) is a grid of characters seen from the
front (columns = x, depth along z), from the side (columns = z, depth along x) or as a *profile*
(columns = z, rows = y, the char giving the half-width in x: the cell is filled for |x| < width).
For every cell holding a fill char (``depth`` for front / side maps, ``width`` for profile maps)
the engine fills the cell's box into the layer's part, clipped by every voxel already solid in
any part (so a later layer never overwrites an earlier one and the skull stays whole under
everything shaveable); a ``shell`` char fills instead, over the cell's whole depth, the
dilation by its thickness of everything placed before the layer (:func:`dilate`: the six axis
directions, one voxel at a time, so a shell rounds convex corners at 45 degrees; the layer's
own earlier chars do not count, so every char's thickness is measured from the same base)
where nothing is solid yet, so a beard hugs whatever jaw is under it; ``mirror`` maps give the
x <= 0 half
and are mirrored about x = 0. Added
voxels are coloured by ``paint[char]`` if present, else by the layer's ``colour``, else by the
seeded lattice of ``palette`` / ``block``: one colour per lattice cell of ``block`` units,
anchored at the frame origin, drawn once per part, so the two hair maps (front and side) agree.
A paint-only char recolours the front-most solid voxel of each fine column of the target part
inside the cell's footprint (the first solid voxel from z = -inf toward +z; for a side or profile
map from x = -inf toward +x), after ``recess`` has cut that many units off those columns: the
voxel the cut exposes takes the material the old surface had (a philtrum cut into skin stays
skin), unless ``materials`` names one for the char; a paint char without a recess keeps the
material of the voxel it recolours. ``materials`` overrides the material per char.

The skull: the first layer must be the profile map named ``skull`` (its fill is the skin form of
the skull and neck). The engine then adds the ears (``EARS``, 1 U proud) with a notch cut into
each outer face (``EAR_NOTCHES``, its floor and walls painted ``NOTCH_TONE``), cuts the mouth
cavity (``CAVITY_X`` x ``CAVITY_Y``, ``CAVITY_DEPTH`` deep from the face surface it finds there)
behind closed lips: the cavity's front ``LIP_THICK`` stays skin and only its ``LIP_SLIT_Y`` fine
row is coloured ``LIP_TONE`` across the mouth's width, so a clean-shaven head shows a thin dark
mouth line and the teeth stay hidden until the lips are cut (the sealed cavity is an enclosed
air pocket: ``checks`` warns about it, by design). It layers the whole form by depth (skin
2 fine, fat, muscle, bone: ``voxchar.HEAD_BANDS``), stands the teeth (``TEETH_X``: seven 2-fine
teeth with 1-fine gaps fill the 5 U back wall exactly, mirror-exact, the middle tooth on the
axis - eight such teeth would need 5.75 U) against the cavity's back wall and shades the neck
(y < 0) in ``NECK_TONE``. A layer that recesses re-skins the part afterwards (:func:`reskin`:
every fat / muscle / bone voxel the cut left within the skin band of the new surface becomes
skin, so a socket's walls never show fat). After the ``face`` layer the two nostrils
(``NOSTRILS``: one fine voxel wide at x = +-0.5, 2 fine deep) are painted dark on the nose
cube's underside.

Routing: the layer named ``face`` writes into the ``skull`` part, except its ``m`` cells (the
moustache, shaveable) that go to ``beard``; every other layer writes a part of its own name. The
parts come out in order of first use: skull, beard, beret, hair, feather. Everything the maps
leave unsaid is decided here: the paint acts on the target part's own solid (never on another
part's voxels), the chars of a layer are applied in order of first appearance with the fill
chars before the paint-only ones, and a cell whose box is clipped away entirely is reported
(``meta["layers"]``), not an error.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np

from . import materials as M
from .blockhead import FINE, U_M, VOXEL, Block, Builder, Frame, both, make_palette
from .model import Model, Part
from .voxchar import HEAD_BANDS, crop_part, depth_map, layer_tissues

#: the barber's maps (the default model of :func:`build_head`)
DEFAULT_MAPS = Path(__file__).resolve().parents[1] / "specs" / "chars" / "vox" / "barber_maps.py"
#: layer name -> the part that receives its voxels (a layer not listed writes a part of its own name)
LAYER_PART = {"face": "skull"}
#: (layer name, char) -> part, for single chars that go elsewhere (the moustache is shaveable)
CHAR_PART = {("face", "m"): "beard"}
#: parts whose material must be mirror-exact (the symmetry check)
SYMMETRIC_PARTS = ("skull", "beard")
#: the layer that gives the skull form (a profile map, first in the write order)
SKULL_LAYER = "skull"
VIEWS = ("front", "side", "profile")
#: free space (units) around the maps' bounds; the frame is then rounded outward to whole units
FRAME_MARGIN = 0.5

#: the skull's mid-depth in units (the maps put the cheek plane at z = 0, the back at z = 13):
#: the model's local z = 0
SKULL_MID_Z = 6.5
#: the ears: skin blocks 1 U proud on the skull's sides
EARS = both(Block(5, 6, 4, 7, 7, 10))
#: the ear's notch: a 1 x 1 U pit cut 0.5 U into the ear's outer face; its floor and walls are
#: painted in ``NOTCH_TONE``
EAR_NOTCHES = both(Block(5.5, 6, 5, 6, 8, 9))
NOTCH_TONE = "skin_shadow"
#: the mouth cavity: x and y ranges and the depth (units) cut from the face surface backwards
CAVITY_X, CAVITY_Y, CAVITY_DEPTH = (-2.5, 2.5), (3.25, 3.75), 1.0
#: the lips: the cavity's front wall (units) stays skin; the fine row starting at ``LIP_SLIT_Y``
#: is coloured ``LIP_TONE`` across the mouth's width (the mouth line)
LIP_THICK, LIP_SLIT_Y, LIP_TONE = 0.25, 3.5, "skin_shadow"
#: fine x of each tooth's left edge (x = k / FINE) and the tooth width in fine voxels: seven
#: 2-fine teeth with 1-fine gaps across the 5 U back wall (mirror-exact)
TEETH_X, TOOTH_W = (-10, -7, -4, -1, 2, 5, 8), 2
#: the teeth stand this deep (units) against the cavity's back wall
TOOTH_DEPTH = 0.25
#: the nostrils: the bottom-front fine voxels of the nose cube's underside (y = 4.5)
NOSTRILS = both(Block(0.5, 0.75, 4.5, 4.75, -1.5, -1.0))
NOSTRIL_COLOUR = "eye_frame"
NECK_TONE = "skin_shadow"


# ---------------------------------------------------------------------------- helpers
def load_maps(path) -> ModuleType:
    """Import a maps module from a file path (specs are not a package)."""
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("voxgen_maps_" + path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for key in ("LAYERS", "COLOURS"):
        if not hasattr(module, key):
            raise AttributeError(f"{path}: maps module has no {key}")
    return module


def hex_rgb(text: str) -> tuple[int, int, int]:
    """``"#de995f"`` -> (222, 153, 95)."""
    s = text.lstrip("#")
    if len(s) != 6:
        raise ValueError(f"colour {text!r} is not #rrggbb")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def material_id(value) -> int:
    """``"MAT_HAIR"``, ``"hair"`` or an id -> material id."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    name = str(value)
    if name.upper().startswith("MAT_"):
        mid = getattr(M, name.upper(), None)
        if not isinstance(mid, int):
            raise ValueError(f"unknown material {value!r}")
        return mid
    return M.id_of(name)


def is_front(layer: dict) -> bool:
    """True for a front map (columns = x); False for a side or profile map (columns = z)."""
    view = layer["view"]
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r} (one of {VIEWS})")
    return view == "front"


def fills(layer: dict) -> dict[str, tuple[float, float]]:
    """char -> (lo, hi) of the box along the view's normal: ``depth`` for a front or side map,
    (-w, w) from ``width`` for a profile map."""
    if layer["view"] == "profile":
        if "depth" in layer:
            raise ValueError("a profile map takes width, not depth")
        return {ch: (-float(w), float(w)) for ch, w in layer.get("width", {}).items()}
    if "width" in layer:
        raise ValueError(f"width is defined for profile maps only, not {layer['view']!r}")
    return {ch: (float(lo), float(hi)) for ch, (lo, hi) in layer.get("depth", {}).items()}


def cell_box(layer: dict, r: int, c: int, char: str) -> Block:
    """The box (units) of map cell row ``r``, column ``c`` for the fill char ``char``: for a front
    map x from the column, y from the row, z from ``depth``; for a side map z from the column and x
    from ``depth``; for a profile map z from the column and x = +-``width``. Mirror with
    ``.mirror()``: the mirror of column c covers [-hi, -lo)."""
    cell = float(layer["cell"])
    o, y0 = layer["origin"]
    lo, hi = fills(layer)[char]
    y_lo, y_hi = y0 - (r + 1) * cell, y0 - r * cell
    a_lo, a_hi = o + c * cell, o + (c + 1) * cell
    if is_front(layer):
        return Block(a_lo, a_hi, y_lo, y_hi, lo, hi)
    return Block(lo, hi, y_lo, y_hi, a_lo, a_hi)


def char_grid(layer: dict) -> np.ndarray:
    """The rows as a (nrows, ncols) array of single characters."""
    return np.array([list(row) for row in layer["rows"]], dtype="<U1")


def layer_bounds(layer: dict) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float] | None]:
    """((x0, x1), (y0, y1), (z0, z1)) in units that the map's grid and its fill boxes can touch
    (a mirror map counts both halves); the depth axis is None for a paint-only layer."""
    cell = float(layer["cell"])
    o, y0 = layer["origin"]
    nrows, ncols = char_grid(layer).shape
    ys = (y0 - nrows * cell, y0)
    along = (o, o + ncols * cell)
    fl = fills(layer).values()
    normal = (min(lo for lo, _ in fl), max(hi for _, hi in fl)) if fl else None
    if is_front(layer):
        if layer.get("mirror"):
            along = (min(along[0], -along[1]), max(along[1], -along[0]))
        return along, ys, normal
    return normal, ys, along


def frame_extent(layers, margin: float = FRAME_MARGIN) -> tuple[tuple[float, float], ...]:
    """The frame extent ((x0, x1), (y0, y1), (z0, z1)) in whole units around every layer of
    ``layers`` (a list of (name, layer)) and the engine's own blocks, with ``margin`` units of
    air; symmetric in x."""
    lo, hi = [np.inf] * 3, [-np.inf] * 3
    for _, layer in layers:
        for i, rng in enumerate(layer_bounds(layer)):
            if rng is not None:
                lo[i], hi[i] = min(lo[i], rng[0]), max(hi[i], rng[1])
    for b in EARS + NOSTRILS:
        for i, (a, c) in enumerate(((b.x0, b.x1), (b.y0, b.y1), (b.z0, b.z1))):
            lo[i], hi[i] = min(lo[i], a), max(hi[i], c)
    half_x = max(-lo[0], hi[0])
    lo[0], hi[0] = -half_x, half_x
    return tuple((float(np.floor(a - margin)), float(np.ceil(c + margin))) for a, c in zip(lo, hi))


class SheetFrame(Frame):
    """blockhead's fine grid over an extent computed from the maps; local z = 0 at ``mid_z``."""

    def __init__(self, extent, mid_z: float = SKULL_MID_Z):
        super().__init__(extent)
        self.mid_z = float(mid_z)

    @property
    def origin_m(self) -> tuple[float, float, float]:
        return (self.lo[0] * U_M, self.lo[1] * U_M, (self.lo[2] - self.mid_z) * U_M)


def make_frame(layers) -> SheetFrame:
    """The frame of a layer list: extent from :func:`frame_extent`, local z = 0 at ``SKULL_MID_Z``."""
    skull = dict(layers).get(SKULL_LAYER)
    if skull is None or skull["view"] != "profile":
        raise ValueError(f"maps need a profile layer named {SKULL_LAYER!r}")
    return SheetFrame(frame_extent(layers))


def _along(f: Frame, layer: dict) -> np.ndarray:
    """The frame's column coordinates of the map: x for a front map, z otherwise (flat)."""
    return (f.X if is_front(layer) else f.Z).reshape(-1)


def _cell_index(layer: dict, along: np.ndarray, ys: np.ndarray, nrows: int, ncols: int):
    """Row / column index of the map cell holding each fine centre (and whether it is inside the map)."""
    cell = float(layer["cell"])
    o, y0 = layer["origin"]
    r = np.floor((y0 - ys) / cell).astype(int)
    c = np.floor((along - o) / cell).astype(int)
    return r, (r >= 0) & (r < nrows), c, (c >= 0) & (c < ncols)


def footprint(f: Frame, layer: dict, char: str) -> np.ndarray:
    """Bool mask of the fine cells (by centre) that lie in map cells holding ``char``: shape
    (ny, nx) for a front map, (ny, nz) for a side or profile map. A ``mirror`` map also marks
    the x-mirror."""
    grid = char_grid(layer)
    nrows, ncols = grid.shape
    hit = grid == char
    ys = f.Y.reshape(-1)
    along = _along(f, layer)
    out = np.zeros((ys.size, along.size), bool)
    if layer.get("mirror") and not is_front(layer):
        raise ValueError("mirror is defined for front maps only")
    for a in ([along, -along] if layer.get("mirror") else [along]):
        r, rok, c, cok = _cell_index(layer, a, ys, nrows, ncols)
        rc = np.clip(r, 0, nrows - 1)
        cc = np.clip(c, 0, ncols - 1)
        out |= hit[rc[:, None], cc[None, :]] & rok[:, None] & cok[None, :]
    return out


def _cell_counts(f: Frame, layer: dict, plane_counts: np.ndarray) -> np.ndarray:
    """Sum a per-column count (ny, n_along) into the map's (nrows, ncols) cells (mirror halves add up)."""
    grid_shape = char_grid(layer).shape
    nrows, ncols = grid_shape
    ys = f.Y.reshape(-1)
    along = _along(f, layer)
    counts = np.zeros(grid_shape, np.int64)
    for a in ([along, -along] if layer.get("mirror") else [along]):
        r, rok, c, cok = _cell_index(layer, a, ys, nrows, ncols)
        rr, cc = r[rok], c[cok]
        np.add.at(counts, (np.repeat(rr, cc.size), np.tile(cc, rr.size)), plane_counts[np.ix_(rok, cok)].ravel())
    return counts


def extrude(f: Frame, layer: dict, plane: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """The footprint pushed along the view's normal over [lo, hi) units: a (nz, ny, nx) mask."""
    if is_front(layer):
        return f.full(plane[None, :, :] & (f.Z >= lo) & (f.Z < hi))
    return f.full(plane.T[:, :, None] & (f.X >= lo) & (f.X < hi))


def _depth_first(arr: np.ndarray, layer: dict) -> np.ndarray:
    """View of a (nz, ny, nx) array with the view's depth axis first and the footprint axes after:
    (nz, ny, nx) for a front map, (nx, nz, ny) for a side or profile map (so a side footprint is
    transposed)."""
    return arr if is_front(layer) else np.moveaxis(arr, 2, 0)


def lattice_colours(bd: Builder, pname: str, layer: dict, rng: np.random.Generator) -> np.ndarray:
    """Palette index of every fine cell from the part's colour lattice (``block`` units per cell,
    anchored at the frame origin), drawn once per (part, block, palette) through ``rng``."""
    f = bd.f
    block = float(layer["block"])
    bf = block * FINE
    if abs(bf - round(bf)) > 1e-9:
        raise ValueError(f"block {block} U is not a whole number of fine voxels")
    bf = int(round(bf))
    palette = dict(layer["palette"])
    key = (pname, bf, tuple(sorted(palette.items())))
    cache = bd.__dict__.setdefault("lattices", {})
    if key not in cache:
        names = list(palette)
        w = np.asarray([palette[n] for n in names], float)
        shape = tuple(-(-n // bf) for n in (f.shape[0], f.shape[1], f.shape[2]))
        idx = rng.choice(len(names), size=shape, p=w / w.sum())
        lut = np.asarray([bd.pal.index(n) for n in names], np.uint8)
        cache[key] = lut[idx]
    lat = cache[key]
    zi, yi, xi = (np.arange(n) // bf for n in f.shape)
    return lat[zi[:, None, None], yi[None, :, None], xi[None, None, :]]


def _get_part(bd: Builder, parts: dict[str, Part], pname: str) -> Part:
    if pname not in parts:
        parts[pname] = bd.f.new_part(pname, symmetric=pname in SYMMETRIC_PARTS)
    return parts[pname]


def _chars_in_order(layer: dict) -> list[str]:
    """The layer's chars in order of first appearance, the fill chars (depth, width, shell) first."""
    seen: list[str] = []
    for row in layer["rows"]:
        for ch in row:
            if ch != "." and ch not in seen:
                seen.append(ch)
    fl = set(fills(layer)) | set(layer.get("shell", {}))
    return [ch for ch in seen if ch in fl] + [ch for ch in seen if ch not in fl]


def dilate(mask: np.ndarray, n: int) -> np.ndarray:
    """``mask`` grown ``n`` times by one voxel along the six axis directions (no wrap-around): the
    L1 ball, so a shell grown this way meets a convex corner with a 45 degree chamfer."""
    out = mask.copy()
    for _ in range(int(n)):
        grown = out.copy()
        for ax in range(out.ndim):
            head = [slice(None)] * out.ndim
            tail = [slice(None)] * out.ndim
            head[ax], tail[ax] = slice(1, None), slice(None, -1)
            grown[tuple(head)] |= out[tuple(tail)]
            grown[tuple(tail)] |= out[tuple(head)]
        out = grown
    return out


# ---------------------------------------------------------------------------- layers
def apply_layer(bd: Builder, parts: dict[str, Part], name: str, layer: dict, rng: np.random.Generator) -> dict:
    """Write one map layer into ``parts`` (created on demand); returns a report
    ``{"part": ..., "added": n, "painted": n, "cut": n, "empty_cells": [(char, row, col), ...]}``
    where ``empty_cells`` lists the fill cells whose box was clipped away entirely."""
    f = bd.f
    fl = fills(layer)
    shell = layer.get("shell", {})
    paint = layer.get("paint", {})
    recess = layer.get("recess", {})
    mats = layer.get("materials", {})
    base_mat = material_id(layer.get("material", "MAT_SKIN"))
    target = LAYER_PART.get(name, name)
    report = {"part": target, "added": 0, "painted": 0, "cut": 0, "empty_cells": []}
    grid = char_grid(layer)
    base = bd.occupied.copy() if shell else None       # what the shell chars grow from: placed before this layer
    for ch in _chars_in_order(layer):
        pname = CHAR_PART.get((name, ch), target)
        part = _get_part(bd, parts, pname)
        mat = material_id(mats[ch]) if ch in mats else base_mat
        plane = footprint(f, layer, ch)
        if ch in fl and ch in shell:
            raise ValueError(f"layer {name!r}: char {ch!r} is both a depth and a shell char")
        if ch in fl or ch in shell:
            if ch in fl:
                lo, hi = fl[ch]
                mask = extrude(f, layer, plane, lo, hi) & ~bd.occupied
            else:
                n = float(shell[ch]) * FINE
                if abs(n - round(n)) > 1e-9:
                    raise ValueError(f"layer {name!r}: shell {shell[ch]} U is not a whole number of fine voxels")
                mask = extrude(f, layer, plane, -np.inf, np.inf) & dilate(base, int(round(n))) & ~bd.occupied
            part.material[mask] = np.uint8(mat)
            if ch in paint:
                part.color[mask] = np.uint8(bd.pal.index(paint[ch]))
            elif "colour" in layer:
                part.color[mask] = np.uint8(bd.pal.index(layer["colour"]))
            elif "palette" in layer:
                part.color[mask] = lattice_colours(bd, pname, layer, rng)[mask]
            else:
                raise ValueError(f"layer {name!r}: no colour for char {ch!r} (no paint entry, colour or palette)")
            bd.occupied |= mask
            report["added"] += int(mask.sum())
            per_col = mask.sum(axis=0) if is_front(layer) else mask.sum(axis=2).T
            counts = _cell_counts(f, layer, per_col)
            for r, c in zip(*np.nonzero((grid == ch) & (counts == 0))):
                report["empty_cells"].append((ch, int(r), int(c)))
        elif ch in paint:
            plane_d = plane if is_front(layer) else plane.T
            md, cd, od = (_depth_first(a, layer) for a in (part.material, part.color, bd.occupied))
            n_rec = int(round(float(recess.get(ch, 0.0)) * FINE))
            di = np.arange(md.shape[0])[:, None, None]
            solid = md != 0
            has = solid.any(axis=0) & plane_d
            first = np.argmax(solid, axis=0)
            new_mat = None
            if n_rec:
                old_surf = has[None] & (di == first[None])
                old_mat = np.where(old_surf, md, 0).max(axis=0)       # the material of the surface cut away
                cut = has[None] & (di >= first[None]) & (di < first[None] + n_rec) & solid
                md[cut] = 0
                cd[cut] = 0
                od[cut] = False
                report["cut"] += int(cut.sum())
                solid = md != 0
                has = solid.any(axis=0) & plane_d
                first = np.argmax(solid, axis=0)
                new_mat = np.broadcast_to(old_mat[None], md.shape)
            surf = has[None] & (di == first[None])
            cd[surf] = np.uint8(bd.pal.index(paint[ch]))
            if ch in mats:
                md[surf] = np.uint8(mat)
            elif new_mat is not None:
                md[surf] = new_mat[surf]
            report["painted"] += int(surf.sum())
        else:
            raise ValueError(f"layer {name!r}: char {ch!r} has neither a fill nor a paint entry")
    return report


# ---------------------------------------------------------------------------- skull
def cavity_block(f: Frame, form: np.ndarray) -> Block:
    """The mouth cavity: ``CAVITY_X`` x ``CAVITY_Y``, ``CAVITY_DEPTH`` deep from the front-most
    solid voxel of ``form`` inside that footprint."""
    (x0, x1), (y0, y1) = CAVITY_X, CAVITY_Y
    columns = form & f.full((f.X >= x0) & (f.X < x1) & (f.Y >= y0) & (f.Y < y1))
    if not columns.any():
        raise ValueError("no skull under the mouth cavity")
    z_face = f.lo[2] + int(np.argmax(columns.any(axis=(1, 2)))) / f.fine
    return Block(x0, x1, y0, y1, z_face, z_face + CAVITY_DEPTH)


def finish_skull(bd: Builder, part: Part) -> Block:
    """Turn the skull layer's skin fill into the skull: the ears with their notches, the mouth
    cavity with its teeth behind closed lips, the tissues by depth, the neck, notch and lip tones.
    Returns the cavity block (lips included)."""
    f = bd.f
    form = part.solid()
    for b in EARS:
        form |= f.mask(b)
    cavity = cavity_block(f, form)
    hollow = f.mask(Block(cavity.x0, cavity.x1, cavity.y0, cavity.y1, cavity.z0 + LIP_THICK, cavity.z1))
    slit = f.mask(Block(cavity.x0, cavity.x1, LIP_SLIT_Y, LIP_SLIT_Y + 1 / FINE, cavity.z0, cavity.z0 + LIP_THICK))
    notches = np.zeros(f.shape, bool)
    for b in EAR_NOTCHES:
        notches |= f.mask(b)
    hollow |= notches
    form &= ~hollow
    part.material[~form] = 0
    part.color[~form] = 0
    layer_tissues(part, form, HEAD_BANDS, bd.pal.index)
    bd.recolor(part, f.full(f.Y < 0), NECK_TONE)
    bd.recolor(part, dilate(notches, 1) & ~notches, NOTCH_TONE)     # the notch's floor and walls
    bd.recolor(part, slit, LIP_TONE)                                  # the mouth line on the closed lips
    for k in TEETH_X:
        bd.place(part, Block(k / FINE, (k + TOOTH_W) / FINE, cavity.y0, cavity.y1, cavity.z1 - TOOTH_DEPTH,
                             cavity.z1, "tooth", M.MAT_TOOTH), clip=False)
    bd.occupied &= ~hollow
    bd.occupied |= part.solid()
    return cavity


def reskin(bd: Builder, part: Part) -> int:
    """Re-skin the tissue a recess exposed: every fat / muscle / bone voxel of ``part`` within the
    skin band (``HEAD_BANDS[0]``) of the current surface becomes skin in the skin colour; the
    painted surface (eyes, frames, tones) is left alone. Returns the number of voxels changed."""
    thick = int(HEAD_BANDS[0][1])
    depth = depth_map(part.solid(), thick)
    tissue = np.isin(part.material, (M.MAT_FAT, M.MAT_MUSCLE, M.MAT_BONE))
    sel = tissue & (depth < thick)
    part.material[sel] = np.uint8(M.MAT_SKIN)
    part.color[sel] = np.uint8(bd.pal.index(M.name_of(M.MAT_SKIN)))
    return int(sel.sum())


def paint_nostrils(bd: Builder, skull: Part) -> None:
    for b in NOSTRILS:
        bd.recolor(skull, bd.f.mask(b), NOSTRIL_COLOUR)


# ---------------------------------------------------------------------------- diagnostics
def islands(bd: Builder, parts: dict[str, Part]) -> list[dict]:
    """Per-part components other than the largest (what ``checks.floating`` reports and what
    ``detach_loose`` would drop in the game), with their bounds in units and the other parts they
    rest on (6-adjacency): an island with no supporter floats in space."""
    from .checks import component_sizes, label_components
    f = bd.f
    out = []
    for name, part in parts.items():
        labels, count = label_components(part.solid())
        sizes = component_sizes(labels, count)
        for lab in range(2, count + 1):
            m = labels == lab
            grown = m.copy()
            for ax in (0, 1, 2):
                for s in (1, -1):
                    grown |= np.roll(m, s, axis=ax)
            zz, yy, xx = np.nonzero(m)
            lo = tuple(float(f.lo[i] + v.min() / f.fine) for i, v in enumerate((xx, yy, zz)))
            hi = tuple(float(f.lo[i] + (v.max() + 1) / f.fine) for i, v in enumerate((xx, yy, zz)))
            out.append({"part": name, "voxels": int(sizes[lab]), "lo": lo, "hi": hi,
                        "resting_on": [q for q, p in parts.items() if q != name and (grown & p.solid()).any()]})
    return out


# ---------------------------------------------------------------------------- model
def make_sheet_palette(colours: dict[str, str]):
    """blockhead's palette (tissue tones included) plus the maps' colours by name."""
    pal = make_palette()
    for cname, hx in colours.items():
        pal.add(cname, hex_rgb(hx))
    return pal


def build_parts(maps, seed: int = 1, layers=None) -> tuple[Builder, dict[str, Part], dict]:
    """The uncropped parts on the frame (write order preserved) and the per-layer reports;
    ``layers`` (a list of (name, layer)) replaces ``maps.LAYERS``."""
    layers = list(maps.LAYERS if layers is None else layers)
    if not layers or layers[0][0] != SKULL_LAYER:
        raise ValueError(f"the first layer must be {SKULL_LAYER!r}")
    pal = make_sheet_palette(maps.COLOURS)
    bd = Builder(make_frame(layers), pal)
    parts: dict[str, Part] = {}
    rng = np.random.default_rng(seed)
    reports: dict[str, list[dict]] = {}
    for lname, layer in layers:
        rep = apply_layer(bd, parts, lname, layer, rng)
        reports.setdefault(lname, []).append(rep)
        if lname == SKULL_LAYER:
            finish_skull(bd, parts[SKULL_LAYER])
        if rep["cut"]:
            rep["reskinned"] = reskin(bd, parts[rep["part"]])
        if lname == "face":
            paint_nostrils(bd, parts[SKULL_LAYER])
    for q in parts.values():
        bd.finish(q)
    return bd, parts, reports


def build_head(seed: int = 1, maps=None, name: str = "barber", layers=None) -> Model:
    """The relief-map head as a model of cropped parts; ``maps`` is a module (or namespace) with
    ``LAYERS`` and ``COLOURS``, by default the barber's; ``layers`` (a list of (name, layer))
    replaces ``maps.LAYERS`` (a bald head: the skull and face layers only)."""
    maps = maps if maps is not None else load_maps(DEFAULT_MAPS)
    bd, parts, reports = build_parts(maps, seed, layers)
    meta = {"generator": "sheethead", "unit_mm": round(U_M * 1000, 3), "voxel_mm": round(VOXEL * 1000, 3),
            "seed": seed, "maps": str(getattr(maps, "__file__", "")), "reference": "docs/reference/barber-head.md",
            "frame": [list(r) for r in bd.f.extent], "mid_z": bd.f.mid_z}
    model = Model(name, VOXEL, bd.pal, meta=meta)
    for q in parts.values():
        model.add_part(crop_part(q))
    model.meta["build"] = {q.name: q.count_solid() for q in model.parts}
    model.meta["layers"] = {k: [{kk: vv for kk, vv in r.items() if kk != "empty_cells"} | {"empty_cells": len(r["empty_cells"])}
                                for r in v] for k, v in reports.items()}
    model.meta["islands"] = islands(bd, parts)
    return model
