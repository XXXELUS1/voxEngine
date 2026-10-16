"""MagicaVoxel ``.vox`` (version 150) export/import.

Export writes one ``SIZE``/``XYZI`` model per part, a scene graph
(``nTRN`` root -> ``nGRP`` -> ``nTRN``+``nSHP`` per part, named after the part) and the
``RGBA`` palette. Because .vox stores only a colour per voxel, every distinct
``(palette index, material)`` pair of the model gets its own .vox colour slot, and the
side-car ``<name>.vox.json`` maps slot -> (palette index, material) so that
``import_vox`` restores materials losslessly (as long as <= 255 pairs are used).

Axes: Godot is Y-up with the character facing -Z; MagicaVoxel is Z-up. The mapping is
``vox = (x, nz-1-z, y)`` (so the character faces +Y in MagicaVoxel) and ``SIZE = (nx, nz, ny)``.
The scene-graph translation places each part by its origin in units of the *part's*
voxel size; parts with different voxel sizes therefore show at the wrong relative scale
in MagicaVoxel (the format has no per-model scale), but import back is still exact.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from . import materials as M
from .model import Model, Part
from .palette import Palette


# ---------------------------------------------------------------------------
# binary helpers
# ---------------------------------------------------------------------------

def _string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<i", len(b)) + b


def _dict(d: dict) -> bytes:
    out = struct.pack("<i", len(d))
    for k, v in d.items():
        out += _string(str(k)) + _string(str(v))
    return out


def _chunk(cid: bytes, content: bytes, children: bytes = b"") -> bytes:
    return cid + struct.pack("<ii", len(content), len(children)) + content + children


def _read_string(data: bytes, pos: int):
    (n,) = struct.unpack_from("<i", data, pos)
    return data[pos + 4: pos + 4 + n].decode("utf-8", "replace"), pos + 4 + n


def _read_dict(data: bytes, pos: int):
    (n,) = struct.unpack_from("<i", data, pos)
    pos += 4
    d = {}
    for _ in range(n):
        k, pos = _read_string(data, pos)
        v, pos = _read_string(data, pos)
        d[k] = v
    return d, pos


# ---------------------------------------------------------------------------
# colour slot mapping
# ---------------------------------------------------------------------------

def build_slots(model: Model):
    """Assign .vox colour slots (1..255) to every (palette index, material) pair in use.
    Returns ``(slots: dict[(pal_idx, mat)] -> slot, rgba (256,4) uint8)``."""
    pairs = []
    seen = set()
    for p in model.parts:
        solid = p.solid()
        if not solid.any():
            continue
        key = p.color[solid].astype(np.int32) * 256 + p.material[solid].astype(np.int32)
        for k in np.unique(key):
            pair = (int(k) // 256, int(k) % 256)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    if len(pairs) > 255:
        raise ValueError(f"{len(pairs)} (colour, material) pairs in use; .vox can hold 255")
    pal = model.palette.as_array()
    rgba = np.zeros((256, 4), np.uint8)
    slots = {}
    for i, (ci, mat) in enumerate(pairs, start=1):
        slots[(ci, mat)] = i
        rgba[i - 1, :3] = pal[ci] if ci < len(model.palette) else (255, 0, 255)
        rgba[i - 1, 3] = 255
    return slots, rgba


def _part_xyzi(part: Part, slots: dict) -> np.ndarray:
    """(n, 4) uint8 rows of x, y_vox, z_vox, slot."""
    zz, yy, xx = np.nonzero(part.material)
    if zz.size == 0:
        return np.zeros((0, 4), np.uint8)
    key = part.color[zz, yy, xx].astype(np.int32) * 256 + part.material[zz, yy, xx].astype(np.int32)
    lut = np.zeros(256 * 256, np.uint8)
    for (ci, mat), slot in slots.items():
        lut[ci * 256 + mat] = slot
    out = np.empty((zz.size, 4), np.uint8)
    out[:, 0] = xx
    out[:, 1] = part.nz - 1 - zz
    out[:, 2] = yy
    out[:, 3] = lut[key]
    return out


def part_translation(part: Part) -> tuple[int, int, int]:
    """nTRN ``_t`` for a part: its bounding box centre in .vox voxel units (see module doc)."""
    vs = part.voxel_size
    ox, oy, oz = (round(v / vs) for v in part.origin)
    nx, ny, nz = part.nx, part.ny, part.nz
    return (int(ox + nx // 2), int(-oz - nz + nz // 2), int(oy + ny // 2))


def to_bytes(model: Model) -> tuple[bytes, dict]:
    """Serialise the model. Returns ``(vox_bytes, side_car_json_dict)``."""
    parts = [p for p in model.parts if p.nx <= 256 and p.ny <= 256 and p.nz <= 256]
    skipped = [p.name for p in model.parts if p not in parts]
    slots, rgba = build_slots(model)
    children = b""
    for p in parts:
        children += _chunk(b"SIZE", struct.pack("<iii", p.nx, p.nz, p.ny))
        xyzi = _part_xyzi(p, slots)
        children += _chunk(b"XYZI", struct.pack("<i", xyzi.shape[0]) + xyzi.tobytes())
    # scene graph: 0 = root nTRN, 1 = nGRP, then (nTRN, nSHP) pairs
    root_child = 1
    children += _chunk(b"nTRN", struct.pack("<i", 0) + _dict({}) + struct.pack("<iiii", root_child, -1, -1, 1) + _dict({}))
    grp = struct.pack("<i", 1) + _dict({}) + struct.pack("<i", len(parts))
    for i in range(len(parts)):
        grp += struct.pack("<i", 2 + 2 * i)
    children += _chunk(b"nGRP", grp)
    for i, p in enumerate(parts):
        trn_id, shp_id = 2 + 2 * i, 3 + 2 * i
        t = part_translation(p)
        children += _chunk(b"nTRN", struct.pack("<i", trn_id) + _dict({"_name": p.name})
                           + struct.pack("<iiii", shp_id, -1, 0, 1) + _dict({"_t": f"{t[0]} {t[1]} {t[2]}"}))
        children += _chunk(b"nSHP", struct.pack("<i", shp_id) + _dict({}) + struct.pack("<i", 1)
                           + struct.pack("<i", i) + _dict({}))
    children += _chunk(b"RGBA", rgba.tobytes())
    data = b"VOX " + struct.pack("<i", 150) + _chunk(b"MAIN", b"", children)
    side = {
        "format": 1,
        "name": model.name,
        "voxel_size": model.voxel_size,
        "axes": "vox = (x, nz-1-z, y)",
        "palette": model.palette.to_list(),
        "slots": {str(slot): {"color": ci, "material": mat, "material_name": M.name_of(mat)}
                  for (ci, mat), slot in slots.items()},
        "parts": [{"name": p.name, "bone": p.bone, "voxel_size": p.voxel_size, "origin": list(p.origin),
                   "dims": [p.nx, p.ny, p.nz], "model_index": i, "symmetric": p.symmetric, "budget": p.budget}
                  for i, p in enumerate(parts)],
        "skipped_parts": skipped,
        "meta": dict(model.meta),
    }
    return data, side


def write(model: Model, path, json_path=None) -> tuple[Path, Path]:
    """Write ``<path>`` (.vox) and the side-car ``<path>.json`` (default: ``name.vox.json``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data, side = to_bytes(model)
    path.write_bytes(data)
    jp = Path(json_path) if json_path else path.with_suffix(path.suffix + ".json")
    jp.write_text(json.dumps(side, indent=1), encoding="utf-8")
    return path, jp


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def read(path) -> dict:
    """Parse a .vox file. Returns
    ``{"version", "models": [{"size": (sx,sy,sz), "xyzi": (n,4) uint8}], "palette": (256,4) uint8,
    "nodes": {id: {...}}, "shapes": {model_index: {"name", "t": (x,y,z)}}}``.
    """
    data = Path(path).read_bytes()
    if data[:4] != b"VOX ":
        raise ValueError("not a .vox file")
    (version,) = struct.unpack_from("<i", data, 4)
    out = {"version": version, "models": [], "palette": _default_vox_palette(), "nodes": {}, "shapes": {}}
    pos = 8
    sizes = []

    def parse(pos, end):
        while pos < end:
            cid = data[pos:pos + 4]
            csize, chsize = struct.unpack_from("<ii", data, pos + 4)
            body = data[pos + 12: pos + 12 + csize]
            if cid == b"SIZE":
                sizes.append(struct.unpack("<iii", body[:12]))
            elif cid == b"XYZI":
                (n,) = struct.unpack_from("<i", body, 0)
                arr = np.frombuffer(body[4:4 + 4 * n], np.uint8).reshape(n, 4).copy()
                out["models"].append({"size": sizes[len(out["models"])] if len(sizes) > len(out["models"]) else None, "xyzi": arr})
            elif cid == b"RGBA":
                out["palette"] = np.frombuffer(body[:1024], np.uint8).reshape(256, 4).copy()
            elif cid == b"nTRN":
                p = 0
                (nid,) = struct.unpack_from("<i", body, p); p += 4
                attrs, p = _read_dict(body, p)
                child, _res, layer, nframes = struct.unpack_from("<iiii", body, p); p += 16
                frames = []
                for _ in range(nframes):
                    fd, p = _read_dict(body, p)
                    frames.append(fd)
                out["nodes"][nid] = {"type": "nTRN", "attrs": attrs, "child": child, "layer": layer, "frames": frames}
            elif cid == b"nGRP":
                p = 0
                (nid,) = struct.unpack_from("<i", body, p); p += 4
                attrs, p = _read_dict(body, p)
                (n,) = struct.unpack_from("<i", body, p); p += 4
                kids = list(struct.unpack_from(f"<{n}i", body, p))
                out["nodes"][nid] = {"type": "nGRP", "attrs": attrs, "children": kids}
            elif cid == b"nSHP":
                p = 0
                (nid,) = struct.unpack_from("<i", body, p); p += 4
                attrs, p = _read_dict(body, p)
                (n,) = struct.unpack_from("<i", body, p); p += 4
                models = []
                for _ in range(n):
                    (mid,) = struct.unpack_from("<i", body, p); p += 4
                    md, p = _read_dict(body, p)
                    models.append((mid, md))
                out["nodes"][nid] = {"type": "nSHP", "attrs": attrs, "models": models}
            if chsize:
                parse(pos + 12 + csize, pos + 12 + csize + chsize)
            pos += 12 + csize + chsize
        return pos

    parse(pos, len(data))
    # resolve shape names/translations through their parent nTRN
    for nid, node in out["nodes"].items():
        if node["type"] == "nTRN" and node["child"] in out["nodes"] and out["nodes"][node["child"]]["type"] == "nSHP":
            shp = out["nodes"][node["child"]]
            t = (0, 0, 0)
            if node["frames"] and "_t" in node["frames"][0]:
                t = tuple(int(v) for v in node["frames"][0]["_t"].split())
            for mid, _ in shp["models"]:
                out["shapes"][mid] = {"name": node["attrs"].get("_name", f"model{mid}"), "t": t}
    return out


def _default_vox_palette() -> np.ndarray:
    """MagicaVoxel's default palette (approximation: a 6x6x6 colour cube then greys),
    used only when a file has no RGBA chunk."""
    pal = np.zeros((256, 4), np.uint8)
    i = 0
    for r in (255, 204, 153, 102, 51, 0):
        for g in (255, 204, 153, 102, 51, 0):
            for b in (255, 204, 153, 102, 51, 0):
                if i < 255:
                    pal[i] = (r, g, b, 255)
                    i += 1
    return pal


def to_model(vox: dict, side: dict | None, name: str | None = None) -> Model:
    """Rebuild a Model from parsed .vox data and the side-car json (may be None: then every
    voxel becomes MAT_GENERIC with the .vox colours as palette)."""
    if side is not None:
        palette = Palette.from_list(side["palette"])
        model = Model(name or side.get("name", "import"), float(side.get("voxel_size", 0.005)), palette,
                      meta=dict(side.get("meta", {})))
        slot_map = {int(k): (int(v["color"]), int(v["material"])) for k, v in side["slots"].items()}
        parts_meta = side["parts"]
    else:
        palette = Palette()
        for i in range(255):
            palette._append(tuple(int(v) for v in vox["palette"][i][:3]), f"vox{i + 1}")
        model = Model(name or "import", 0.005, palette)
        slot_map = None
        parts_meta = [{"name": vox["shapes"].get(i, {}).get("name", f"model{i}"), "bone": "", "voxel_size": 0.005,
                       "model_index": i, "dims": None, "origin": None}
                      for i in range(len(vox["models"]))]
    lut_c = np.zeros(256, np.uint8)
    lut_m = np.zeros(256, np.uint8)
    if slot_map is not None:
        for slot, (ci, mat) in slot_map.items():
            lut_c[slot], lut_m[slot] = ci, mat
    else:
        lut_c[1:] = np.arange(1, 256)
        lut_m[1:] = M.MAT_GENERIC
    for pm in parts_meta:
        mi = pm["model_index"]
        if mi >= len(vox["models"]):
            raise ValueError(f"model index {mi} for part {pm['name']!r} not in .vox")
        vm = vox["models"][mi]
        sx, sy, sz = vm["size"]
        nx, ny, nz = (pm["dims"] if pm.get("dims") else (sx, sz, sy))
        if (sx, sy, sz) != (nx, nz, ny):
            raise ValueError(f"part {pm['name']!r}: .vox size {vm['size']} does not match dims {(nx, ny, nz)}")
        origin = tuple(pm["origin"]) if pm.get("origin") else (-nx / 2 * pm["voxel_size"], 0.0, -nz / 2 * pm["voxel_size"])
        part = Part(pm["name"], (nx, ny, nz), float(pm["voxel_size"]), origin=origin, bone=pm.get("bone", ""),
                    symmetric=bool(pm.get("symmetric", False)), budget=pm.get("budget"))
        xyzi = vm["xyzi"]
        if xyzi.shape[0]:
            x = xyzi[:, 0].astype(np.int64)
            z = nz - 1 - xyzi[:, 1].astype(np.int64)
            y = xyzi[:, 2].astype(np.int64)
            slot = xyzi[:, 3]
            ok = (x < nx) & (y < ny) & (z >= 0) & (z < nz)
            part.material[z[ok], y[ok], x[ok]] = lut_m[slot[ok]]
            part.color[z[ok], y[ok], x[ok]] = lut_c[slot[ok]]
        model.add_part(part)
    return model


def import_vox(vox_path, json_path=None, name: str | None = None) -> Model:
    """Read ``.vox`` (+ optional side-car json) into a Model."""
    vox = read(vox_path)
    side = None
    if json_path is None:
        cand = Path(vox_path).with_suffix(".vox.json")
        if cand.exists():
            json_path = cand
    if json_path is not None:
        side = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return to_model(vox, side, name=name)
