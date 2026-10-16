"""Contact sheet of the renders in a directory: tiles with labels (style, view/light, engine,
samples, seconds, faces) read from the ``.json`` side-cars that ``render_styles`` writes.
Pillow only; ``voxgen render-sheet <dir>`` is the CLI wrapper."""
from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SHEET_NAME = "sheet.png"


def _font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def collect(dir_path) -> list[tuple[Path, dict]]:
    """All renders in ``dir_path`` (PNG + json side-car), ordered by the json ``index`` then name."""
    tiles = []
    for png in sorted(Path(dir_path).glob("*.png")):
        if png.name == SHEET_NAME:
            continue
        js = png.with_suffix(".json")
        info = json.loads(js.read_text()) if js.exists() else {}
        tiles.append((png, info))
    tiles.sort(key=lambda t: (t[1].get("index", 1 << 30), t[0].name))
    return tiles


def _fit_font(d, text: str, width: int, path: str, sizes=(16, 15, 14, 13, 12, 11, 10)):
    """Largest font size whose rendered ``text`` fits ``width`` px."""
    for size in sizes:
        f = _font(path, size)
        if d.textlength(text, font=f) <= width:
            return f
    return _font(path, sizes[-1])


def make_sheet(dir_path, out=None, tile: int = 640, cols: int = 3, pad: int = 12, label_h: int = 58) -> Path:
    """Write the contact sheet of every render in ``dir_path``; returns the sheet path."""
    dir_path = Path(dir_path)
    out = Path(out) if out else dir_path / SHEET_NAME
    tiles = collect(dir_path)
    if not tiles:
        raise FileNotFoundError(f"no renders (*.png) in {dir_path}")
    cols = max(1, min(cols, len(tiles)))
    rows = (len(tiles) + cols - 1) // cols
    W = cols * tile + (cols + 1) * pad
    H = rows * (tile + label_h) + (rows + 1) * pad + 40
    sheet = Image.new("RGB", (W, H), (24, 24, 26))
    d = ImageDraw.Draw(sheet)
    f_big, f_foot = _font(FONT_B, 22), _font(FONT, 15)
    for i, (png, info) in enumerate(tiles):
        r, c = divmod(i, cols)
        x = pad + c * (tile + pad)
        y = pad + r * (tile + label_h + pad)
        im = Image.open(png).convert("RGB")
        if im.size != (tile, tile):
            im = im.resize((tile, tile), Image.LANCZOS)
        sheet.paste(im, (x, y))
        style = info.get("style", png.stem)
        d.text((x + 10, y + tile + 4), style, font=f_big, fill=(235, 235, 235))
        faces = info.get("faces")
        txt = (f"{info.get('view', '?')} / {info.get('light', '?')}  |  {info.get('engine', '?')} "
               f"{info.get('samples', '?')} spl  |  {info.get('render_seconds', '?')} s  |  "
               f"{faces:,} faces" if isinstance(faces, int) else "? faces")
        d.text((x + 10, y + tile + 34), txt, font=_fit_font(d, txt, tile - 20, FONT), fill=(170, 170, 175))
    info0 = tiles[0][1]
    foot = (f"{info0.get('model', dir_path.name)}  {info0.get('dims', '')} @ {info0.get('voxel_mm', '?')} mm   -   "
            f"Blender {info0.get('blender', '?')} bpy, {info0.get('size', tile)} px, bg {info0.get('bg', '?')}, "
            f"AgX, OIDN")
    d.text((pad, H - 30), foot, font=f_foot, fill=(140, 140, 145))
    os.makedirs(out.parent, exist_ok=True)
    sheet.save(out)
    return out
