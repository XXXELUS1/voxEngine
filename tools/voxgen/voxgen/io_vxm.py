"""Read/write the .vxm container (docs/spec/vxm-format.md).

Layout: ``"VXM1"`` | uint32 LE header length | UTF-8 JSON header | raw fields.
For every part in order, for every field in ``part.fields`` order, ``nx*ny*nz`` bytes with
``index = x + nx*(y + ny*z)``; our arrays are ``(nz, ny, nx)`` C-order so that is
``array.tobytes()``. The standard fields (``material``, ``color``, ``state``) are uint8; the
optional ``sdf`` field is int8 (:mod:`voxgen.sdfield`); any other field name is kept as raw
uint8 in ``Part.extra`` so it survives a read-write round trip (written after the standard three;
the C++ loader skips it).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from . import __version__
from .model import Model, Part
from .palette import Palette

MAGIC = b"VXM1"
FIELDS = Part.FIELDS
SIGNED_FIELDS = ("sdf",)


def header_dict(model: Model, spec: str | None = None) -> dict:
    parts = []
    for p in model.parts:
        parts.append({
            "name": p.name,
            "bone": p.bone,
            "voxel_size": p.voxel_size,
            "origin": [float(v) for v in p.origin],
            "dims": [p.nx, p.ny, p.nz],
            "fields": p.fields,
        })
    meta = {"generator": f"voxgen {__version__}"}
    if spec:
        meta["spec"] = spec
    meta.update(model.meta)
    return {
        "format": 1,
        "name": model.name,
        "voxel_size": model.voxel_size,
        "palette": model.palette.to_list(),
        "materials": {str(k): v for k, v in model.used_materials().items()},
        "parts": parts,
        "meta": meta,
    }


def to_bytes(model: Model, spec: str | None = None) -> bytes:
    header = json.dumps(header_dict(model, spec), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    chunks = [MAGIC, struct.pack("<I", len(header)), header]
    for p in model.parts:
        for f in p.fields:
            arr = p.get_field(f)
            assert arr.shape == (p.nz, p.ny, p.nx) and arr.dtype.itemsize == 1, f
            chunks.append(np.ascontiguousarray(arr).tobytes())
    return b"".join(chunks)


def write(model: Model, path, spec: str | None = None) -> Path:
    """Write ``model`` to ``path`` (.vxm). ``spec`` is recorded in ``meta.spec``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(to_bytes(model, spec))
    return path


def from_bytes(data: bytes) -> Model:
    if data[:4] != MAGIC:
        raise ValueError("not a VXM1 file")
    (hlen,) = struct.unpack("<I", data[4:8])
    header = json.loads(data[8:8 + hlen].decode("utf-8"))
    if header.get("format") != 1:
        raise ValueError(f"unsupported vxm format {header.get('format')!r}")
    palette = Palette.from_list(header.get("palette", [[0, 0, 0]]))
    model = Model(header.get("name", "model"), float(header.get("voxel_size", 0.005)), palette,
                  meta=dict(header.get("meta", {})))
    pos = 8 + hlen
    for ph in header.get("parts", []):
        nx, ny, nz = (int(v) for v in ph["dims"])
        part = Part(ph["name"], (nx, ny, nz), float(ph.get("voxel_size", model.voxel_size)),
                    origin=tuple(float(v) for v in ph.get("origin", (0, 0, 0))), bone=ph.get("bone", ""))
        n = nx * ny * nz
        for field in ph.get("fields", ["material"]):
            raw = data[pos:pos + n]
            if len(raw) != n:
                raise ValueError(f"truncated field {field!r} of part {part.name!r}")
            pos += n
            dtype = np.int8 if field in SIGNED_FIELDS else np.uint8
            part.set_field(field, np.frombuffer(raw, dtype).reshape(nz, ny, nx).copy())
        model.add_part(part)
    return model


def read(path) -> Model:
    """Read a .vxm file into a Model."""
    return from_bytes(Path(path).read_bytes())
