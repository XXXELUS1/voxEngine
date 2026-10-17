import json
import struct

import numpy as np

from helpers import layered_ball
from voxgen import io_vox, io_vxm, sdfield
from voxgen.materials import MAT_HAIR, MAT_SKIN


def test_vxm_roundtrip_bytes_equal(tmp_path):
    model, part = layered_ball(hair=True)
    part.state[part.material == MAT_SKIN] = 0x21  # some state bits survive too
    path = tmp_path / "ball.vxm"
    io_vxm.write(model, path, spec="tests")
    raw = path.read_bytes()
    assert raw[:4] == b"VXM1"
    (hlen,) = struct.unpack("<I", raw[4:8])
    header = json.loads(raw[8:8 + hlen])
    assert header["format"] == 1 and header["name"] == "ball"
    assert header["parts"][0]["dims"] == [32, 32, 32]
    assert header["parts"][0]["fields"] == ["material", "color", "state"]
    assert "8" in header["materials"] and header["materials"]["8"] == "hair"
    # raw bytes are x-fastest: the first nx bytes are material[z=0, y=0, :]
    n = 32 * 32 * 32
    body = raw[8 + hlen:]
    assert len(body) == 3 * n
    assert np.frombuffer(body[:32], np.uint8).tolist() == part.material[0, 0, :].tolist()
    # a known voxel lands at x + nx*(y + ny*z)
    z, y, x = 16, 16, 3
    assert body[x + 32 * (y + 32 * z)] == part.material[z, y, x]

    back = io_vxm.read(path)
    assert io_vxm.to_bytes(back, spec="tests") == raw
    assert back.parts[0].origin == part.origin
    assert back.palette.to_list() == model.palette.to_list()
    assert np.array_equal(back.parts[0].state, part.state)


def test_vxm_unknown_field_is_skipped(tmp_path):
    model, part = layered_ball(n=8)
    raw = bytearray(io_vxm.to_bytes(model))
    (hlen,) = struct.unpack("<I", raw[4:8])
    header = json.loads(raw[8:8 + hlen])
    header["parts"][0]["fields"] = ["material", "extra", "color", "state"]
    hb = json.dumps(header).encode()
    n = 8 * 8 * 8
    fields = bytes(raw[8 + hlen:])
    new = b"VXM1" + struct.pack("<I", len(hb)) + hb + fields[:n] + bytes(n) + fields[n:]
    back = io_vxm.from_bytes(new)
    assert np.array_equal(back.parts[0].material, part.material)
    assert np.array_equal(back.parts[0].color, part.color)
    # the unknown field is kept as raw bytes (written after the standard ones), so nothing is lost
    assert back.parts[0].fields == ["material", "color", "state", "extra"]
    assert back.parts[0].extra["extra"].dtype == np.uint8 and not back.parts[0].extra["extra"].any()
    assert io_vxm.to_bytes(back)[-4 * n:] == fields + bytes(n)


def test_vxm_sdf_field_roundtrip(tmp_path):
    model, part = layered_ball(n=12, r=0.025)
    assert part.sdf is None and part.fields == ["material", "color", "state"]
    sdfield.add_distance_field(part)
    assert part.fields == ["material", "color", "state", "sdf"] and part.sdf.dtype == np.int8
    assert sdfield.is_consistent(part.sdf, part.solid())
    path = tmp_path / "ball.vxm"
    io_vxm.write(model, path)
    raw = path.read_bytes()
    (hlen,) = struct.unpack("<I", raw[4:8])
    header = json.loads(raw[8:8 + hlen])
    assert header["parts"][0]["fields"] == ["material", "color", "state", "sdf"]
    n = 12 ** 3
    assert len(raw) - 8 - hlen == 4 * n
    assert np.frombuffer(raw[-n:], np.int8).reshape(12, 12, 12).tolist() == part.sdf.tolist()
    back = io_vxm.read(path)
    assert back.parts[0].sdf.dtype == np.int8 and np.array_equal(back.parts[0].sdf, part.sdf)
    assert io_vxm.to_bytes(back) == raw
    # paint / clear keep the field consistent; copy keeps it; dropping it works
    c = part.copy()
    assert np.array_equal(c.sdf, part.sdf)
    c.clear(np.abs(c.coords()[1]) < 0.004)
    assert sdfield.is_consistent(c.sdf, c.solid()) and c.count_solid() < part.count_solid()
    c.paint(np.abs(c.coords()[1]) < 0.004, MAT_SKIN, 1)
    assert sdfield.is_consistent(c.sdf, c.solid())
    c.sdf = None
    assert c.fields == ["material", "color", "state"]


def test_vox_roundtrip_colours_and_materials(tmp_path):
    model, part = layered_ball(hair=True)
    vox_path, json_path = io_vox.write(model, tmp_path / "ball.vox")
    assert json_path.name == "ball.vox.json"
    parsed = io_vox.read(vox_path)
    assert parsed["version"] == 150
    assert parsed["models"][0]["size"] == (32, 32, 32)
    assert parsed["models"][0]["xyzi"].shape[0] == part.count_solid()
    assert parsed["shapes"][0]["name"] == "ball"
    # the RGBA chunk carries our palette colours (slot k -> rgba[k-1])
    side = json.loads(json_path.read_text())
    for slot, info in side["slots"].items():
        rgb = tuple(int(v) for v in parsed["palette"][int(slot) - 1][:3])
        assert rgb == tuple(model.palette[info["color"]])
    back = io_vox.import_vox(vox_path, json_path)
    assert np.array_equal(back.parts[0].material, part.material)
    assert np.array_equal(back.parts[0].color, part.color)
    assert back.parts[0].origin == part.origin
    assert back.palette.to_list() == model.palette.to_list()
    assert (back.parts[0].material == MAT_HAIR).sum() == (part.material == MAT_HAIR).sum()


def test_vox_import_without_json_uses_vox_colours(tmp_path):
    model, part = layered_ball(n=12)
    vox_path, json_path = io_vox.write(model, tmp_path / "b.vox")
    json_path.unlink()
    back = io_vox.import_vox(vox_path)
    assert back.parts[0].count_solid() == part.count_solid()
    assert back.parts[0].material.max() == 63  # generic
