"""Head built from hand-drawn relief maps: the maps are the model (``specs/chars/vox/barber_maps.py``,
editable like pixel art), this module is the engine (2026-10-17).

The frame, the unit grid (U = 20 mm, ``FINE`` = 4 fine voxels of 5 mm per unit), the palette and
the skull form come from :mod:`voxgen.blockhead`: the skull is its 10 x 13 x 9 U box with the
table-2.1 chamfers (:func:`blockhead.skull_form`) plus the neck, the ears and the hidden mouth
cavity with its eight teeth, layered inside by depth (skin / fat / muscle / bone) - but without
blockhead's own eyes, brows, nose and beard: every feature comes from the maps.

A map layer (see the maps module's docstring for the keys) is a grid of characters seen from the
front (columns = x, depth along z) or from the side (columns = z, depth along x). For every cell
holding a ``depth`` char the engine fills the cell's box into the layer's part, clipped by every
voxel already solid in any part (so a later layer never overwrites an earlier one and the skull
stays whole under everything shaveable); ``mirror`` maps give the x <= 0 half and are mirrored
about x = 0. A ``paint`` char recolours the front-most solid voxel of each fine column of the
target part inside the cell's footprint (the first solid voxel from z = -inf toward +z; for a side
map from x = -inf toward +x), after ``recess`` has cut that many units off those columns; a char
with both ``depth`` and ``paint`` fills its box and colours the added voxels with the paint
colour; ``materials`` overrides the material per char (for a paint-only char it sets the material
of the painted surface voxels). Layers with ``palette`` / ``block`` colour their added voxels by
a seeded lattice: one colour per lattice cell of ``block`` units, anchored at the frame origin,
drawn once per part, so the two hair maps (front and side) agree.

Routing: the layer named ``face`` writes into the ``skull`` part, except its ``m`` cells (the
moustache, shaveable) that go to ``beard``; every other layer writes a part of its own name. The
parts come out in order of first use: skull, beard, beret, hair, feather. After the face layer the
two nostrils are painted dark on the nose cube's underside (its bottom-front fine voxels at
x = +-0.5..0.75, as in blockhead). Everything the maps leave unsaid is decided here: the paint
acts on the target part's own solid (never on another part's voxels), the chars of a layer are
applied in order of first appearance with the depth chars before the paint-only ones, and a cell
whose box is clipped away entirely is reported (``meta["layers"]``), not an error.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np

from . import materials as M
from .blockhead import (FINE, TEETH_X, U_M, VOXEL, Block, Builder, Frame, both, make_palette, neck_form,
                        skull_form)
from .model import Model, Part
from .voxchar import HEAD_BANDS, crop_part, layer_tissues

#: the barber's maps (the default model of :func:`build_head`)
DEFAULT_MAPS = Path(__file__).resolve().parents[1] / "specs" / "chars" / "vox" / "barber_maps.py"
#: layer name -> the part that receives its voxels (a layer not listed writes a part of its own name)
LAYER_PART = {"face": "skull"}
#: (layer name, char) -> part, for single chars that go elsewhere (the moustache is shaveable)
CHAR_PART = {("face", "m"): "beard"}
#: parts whose material must be mirror-exact (the symmetry check)
SYMMETRIC_PARTS = ("skull", "beard")

EARS = both(Block(5, 5.5, 5, 7, 2, 3.5))
#: the mouth cavity, 1 U deep behind the moustache rows (cut before the tissues, so its walls are skin)
CAVITY = Block(-1.5, 1.5, 2.25, 2.75, 0, 1)
NOSTRILS = both(Block(0.5, 0.75, 3.5, 3.75, -1, -0.75))
NOSTRIL_COLOUR = "eye_frame"


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


def cell_box(layer: dict, r: int, c: int, char: str) -> Block:
    """The box (units) of map cell row ``r``, column ``c`` for the depth char ``char``: for a front
    map x from the column, y from the row, z from ``depth``; for a side map z from the column and x
    from ``depth``. Mirror with ``.mirror()``: the mirror of column c covers [-hi, -lo)."""
    cell = float(layer["cell"])
    o, y0 = layer["origin"]
    lo, hi = layer["depth"][char]
    y_lo, y_hi = y0 - (r + 1) * cell, y0 - r * cell
    a_lo, a_hi = o + c * cell, o + (c + 1) * cell
    if layer["view"] == "front":
        return Block(a_lo, a_hi, y_lo, y_hi, lo, hi)
    if layer["view"] == "side":
        return Block(lo, hi, y_lo, y_hi, a_lo, a_hi)
    raise ValueError(f"unknown view {layer['view']!r}")


def char_grid(layer: dict) -> np.ndarray:
    """The rows as a (nrows, ncols) array of single characters."""
    return np.array([list(row) for row in layer["rows"]], dtype="<U1")


def _cell_index(layer: dict, along: np.ndarray, ys: np.ndarray, nrows: int, ncols: int):
    """Row / column index of the map cell holding each fine centre (and whether it is inside the map)."""
    cell = float(layer["cell"])
    o, y0 = layer["origin"]
    r = np.floor((y0 - ys) / cell).astype(int)
    c = np.floor((along - o) / cell).astype(int)
    return r, (r >= 0) & (r < nrows), c, (c >= 0) & (c < ncols)


def footprint(f: Frame, layer: dict, char: str) -> np.ndarray:
    """Bool mask of the fine cells (by centre) that lie in map cells holding ``char``: shape
    (ny, nx) for a front map, (ny, nz) for a side map. A ``mirror`` map also marks the x-mirror."""
    grid = char_grid(layer)
    nrows, ncols = grid.shape
    hit = grid == char
    ys = f.Y.reshape(-1)
    along = (f.X if layer["view"] == "front" else f.Z).reshape(-1)
    out = np.zeros((ys.size, along.size), bool)
    halves = [along, -along] if layer.get("mirror") else [along]
    if layer.get("mirror") and layer["view"] != "front":
        raise ValueError("mirror is defined for front maps only")
    for a in halves:
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
    along = (f.X if layer["view"] == "front" else f.Z).reshape(-1)
    counts = np.zeros(grid_shape, np.int64)
    for a in ([along, -along] if layer.get("mirror") else [along]):
        r, rok, c, cok = _cell_index(layer, a, ys, nrows, ncols)
        rr, cc = r[rok], c[cok]
        np.add.at(counts, (np.repeat(rr, cc.size), np.tile(cc, rr.size)), plane_counts[np.ix_(rok, cok)].ravel())
    return counts


def extrude(f: Frame, layer: dict, plane: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """The footprint pushed along the view's normal over [lo, hi) units: a (nz, ny, nx) mask."""
    if layer["view"] == "front":
        return f.full(plane[None, :, :] & (f.Z >= lo) & (f.Z < hi))
    return f.full(plane.T[:, :, None] & (f.X >= lo) & (f.X < hi))


def _depth_first(arr: np.ndarray, layer: dict) -> np.ndarray:
    """View of a (nz, ny, nx) array with the view's depth axis first and the footprint axes after:
    (nz, ny, nx) for a front map, (nx, nz, ny) for a side map (so a side footprint is transposed)."""
    return arr if layer["view"] == "front" else np.moveaxis(arr, 2, 0)


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
    seen: list[str] = []
    for row in layer["rows"]:
        for ch in row:
            if ch != "." and ch not in seen:
                seen.append(ch)
    depth = layer.get("depth", {})
    return [ch for ch in seen if ch in depth] + [ch for ch in seen if ch not in depth]


# ---------------------------------------------------------------------------- layers
def apply_layer(bd: Builder, parts: dict[str, Part], name: str, layer: dict, rng: np.random.Generator) -> dict:
    """Write one map layer into ``parts`` (created on demand); returns a report
    ``{"part": ..., "added": n, "painted": n, "cut": n, "empty_cells": [(char, row, col), ...]}``
    where ``empty_cells`` lists the depth cells whose box was clipped away entirely."""
    f = bd.f
    depth = layer.get("depth", {})
    paint = layer.get("paint", {})
    recess = layer.get("recess", {})
    mats = layer.get("materials", {})
    base_mat = material_id(layer.get("material", "MAT_SKIN"))
    target = LAYER_PART.get(name, name)
    report = {"part": target, "added": 0, "painted": 0, "cut": 0, "empty_cells": []}
    grid = char_grid(layer)
    for ch in _chars_in_order(layer):
        pname = CHAR_PART.get((name, ch), target)
        part = _get_part(bd, parts, pname)
        mat = material_id(mats[ch]) if ch in mats else base_mat
        plane = footprint(f, layer, ch)
        if ch in depth:
            lo, hi = depth[ch]
            mask = extrude(f, layer, plane, lo, hi) & ~bd.occupied
            part.material[mask] = np.uint8(mat)
            if ch in paint:
                part.color[mask] = np.uint8(bd.pal.index(paint[ch]))
            elif "palette" in layer:
                part.color[mask] = lattice_colours(bd, pname, layer, rng)[mask]
            else:
                raise ValueError(f"layer {name!r}: no colour for char {ch!r} (no paint entry and no palette)")
            bd.occupied |= mask
            report["added"] += int(mask.sum())
            per_col = mask.sum(axis=0) if layer["view"] == "front" else mask.sum(axis=2).T
            counts = _cell_counts(f, layer, per_col)
            for r, c in zip(*np.nonzero((grid == ch) & (counts == 0))):
                report["empty_cells"].append((ch, int(r), int(c)))
        elif ch in paint:
            plane_d = plane if layer["view"] == "front" else plane.T
            md, cd, od = (_depth_first(a, layer) for a in (part.material, part.color, bd.occupied))
            n_rec = int(round(float(recess.get(ch, 0.0)) * FINE))
            di = np.arange(md.shape[0])[:, None, None]
            solid = md != 0
            has = solid.any(axis=0) & plane_d
            first = np.argmax(solid, axis=0)
            if n_rec:
                cut = has[None] & (di >= first[None]) & (di < first[None] + n_rec) & solid
                md[cut] = 0
                cd[cut] = 0
                od[cut] = False
                report["cut"] += int(cut.sum())
                solid = md != 0
                has = solid.any(axis=0) & plane_d
                first = np.argmax(solid, axis=0)
            surf = has[None] & (di == first[None])
            cd[surf] = np.uint8(bd.pal.index(paint[ch]))
            if ch in mats:
                md[surf] = np.uint8(mat)
            report["painted"] += int(surf.sum())
        else:
            raise ValueError(f"layer {name!r}: char {ch!r} has neither depth nor paint")
    return report


# ---------------------------------------------------------------------------- skull
def build_skull(bd: Builder, parts: dict[str, Part]) -> Part:
    """The skin skull of blockhead (box + chamfers) with the neck, the ears and the mouth cavity
    with its teeth, layered inside by depth; the neck in the shadow tone. No face features."""
    f = bd.f
    part = _get_part(bd, parts, "skull")
    neck = neck_form(f)
    form = skull_form(f) | neck
    for b in EARS:
        form |= f.mask(b)
    form &= ~f.mask(CAVITY)
    layer_tissues(part, form, HEAD_BANDS, bd.pal.index)
    bd.recolor(part, neck, "skin_shadow")
    for k in TEETH_X:
        bd.place(part, Block(k / 4, k / 4 + 0.25, 2.25, 2.75, 0.75, 1, "tooth", M.MAT_TOOTH), clip=False)
    bd.occupied |= part.solid()
    return part


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


def build_parts(maps, seed: int = 1) -> tuple[Builder, dict[str, Part], dict]:
    """The uncropped parts on the frame (write order preserved) and the per-layer reports."""
    pal = make_sheet_palette(maps.COLOURS)
    bd = Builder(Frame(), pal)
    parts: dict[str, Part] = {}
    skull = build_skull(bd, parts)
    rng = np.random.default_rng(seed)
    reports: dict[str, list[dict]] = {}
    for lname, layer in maps.LAYERS:
        reports.setdefault(lname, []).append(apply_layer(bd, parts, lname, layer, rng))
        if lname == "face":
            paint_nostrils(bd, skull)
    for q in parts.values():
        bd.finish(q)
    return bd, parts, reports


def build_head(seed: int = 1, maps=None, name: str = "barber") -> Model:
    """The relief-map head as a model of cropped parts; ``maps`` is a module (or namespace) with
    ``LAYERS`` and ``COLOURS``, by default the barber's."""
    maps = maps if maps is not None else load_maps(DEFAULT_MAPS)
    bd, parts, reports = build_parts(maps, seed)
    meta = {"generator": "sheethead", "unit_mm": round(U_M * 1000, 3), "voxel_mm": round(VOXEL * 1000, 3),
            "seed": seed, "maps": str(getattr(maps, "__file__", "")), "reference": "docs/reference/barber-head.md"}
    model = Model(name, VOXEL, bd.pal, meta=meta)
    for q in parts.values():
        model.add_part(crop_part(q))
    model.meta["build"] = {q.name: q.count_solid() for q in model.parts}
    model.meta["layers"] = {k: [{kk: vv for kk, vv in r.items() if kk != "empty_cells"} | {"empty_cells": len(r["empty_cells"])}
                                for r in v] for k, v in reports.items()}
    model.meta["islands"] = islands(bd, parts)
    return model
