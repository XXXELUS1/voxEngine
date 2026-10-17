"""Blender (bpy) beauty renders of .vxm models.

``render_styles`` needs the bpy venv (``scripts/install_blender.sh``) and is run as a subprocess
by ``voxgen render``; ``sheet`` only needs Pillow. The names and the part-style resolution below
are shared by the CLI and the renderer so the CLI can validate arguments without importing bpy.

Styles are per model, ``mix`` is per part: every part whose exposed surface is mostly skin
(``SKIN_MATERIALS``: skin + cartilage) is rendered ``smooth`` (surface nets, shade smooth, per-face
voxel colours) and every other part (hair, eyes, shades, teeth) ``facet``. ``--part-styles
"head=smooth,hair=facet"`` overrides the mapping of any style (part name -> one of ``PART_STYLES``).
A smooth part with the ``sdf`` field is meshed by Hermite surface nets (the exact surface, label
"nets hermite"); without it the Laplacian-smoothed binary nets ("nets cell x2") are used. ``slopes``
(``voxgen.mesh_slopes``) is the low-poly facet look: regular staircases become 26.6 / 45 / 63.4
degree planes, lone steps stay cubes, no Bevel (``--part-styles skull=slopes --facet-step 2``).
"""
from __future__ import annotations

import numpy as np

from voxgen import materials as M

STYLES = ("cubes", "bevel", "facet", "facet_all", "slopes", "lowpoly", "smooth", "clay", "toon", "mix")
PART_STYLES = ("cubes", "bevel", "facet", "facet_all", "slopes", "lowpoly", "smooth", "clay")  # toon is model-wide (EEVEE)
LIGHTS = ("studio", "candle")
BACKGROUNDS = ("dark", "grey", "light")
SMOOTH_CLAMPS = ("cell", "free")
SKIN_MATERIALS = (M.MAT_SKIN, M.MAT_CARTILAGE)
MIX_STYLES = {"skin": "smooth", "other": "facet"}
# view name -> (azimuth deg, elevation deg, lens mm, frame margin)
VIEWS = {
    "three_quarter": (33.0, 11.0, 50.0, 1.0),
    "front": (0.0, 4.0, 60.0, 1.0),
    "profile": (90.0, 4.0, 60.0, 1.0),
    "back": (180.0, 4.0, 60.0, 1.0),
    "full": (30.0, 18.0, 35.0, 1.25),
}


def parse_part_styles(text: str) -> dict[str, str]:
    """``"head=smooth,hair=facet"`` -> {"head": "smooth", "hair": "facet"} (validated against PART_STYLES)."""
    out: dict[str, str] = {}
    for item in (v.strip() for v in (text or "").split(",") if v.strip()):
        if "=" not in item:
            raise ValueError(f"--part-styles expects part=style items, got {item!r}")
        name, style = (v.strip() for v in item.split("=", 1))
        if style not in PART_STYLES:
            raise ValueError(f"unknown part style {style!r} for {name!r}; known: {','.join(PART_STYLES)}")
        out[name] = style
    return out


def exposed_face_materials(part) -> dict[int, int]:
    """{material id: number of exposed voxel faces} of a part (what the renders actually show)."""
    mat = np.asarray(part.material)
    solid = mat != M.MAT_AIR
    occ = np.pad(solid, 1)
    counts: dict[int, int] = {}
    for ax in range(3):
        for s in (1, -1):
            nb = np.roll(occ, -s, axis=ax)[1:-1, 1:-1, 1:-1]
            ids, n = np.unique(mat[solid & ~nb], return_counts=True)
            for i, c in zip(ids, n):
                counts[int(i)] = counts.get(int(i), 0) + int(c)
    return counts


def is_skin_part(part) -> bool:
    """True when skin (+ cartilage) owns at least as many exposed faces as any other material."""
    counts = exposed_face_materials(part)
    skin = sum(counts.get(m, 0) for m in SKIN_MATERIALS)
    other = max((c for m, c in counts.items() if m not in SKIN_MATERIALS), default=0)
    return skin > 0 and skin >= other


def resolve_part_styles(model, style: str, overrides: dict[str, str] | None = None) -> dict[str, str]:
    """{part name: style} for every part of ``model``: ``style`` for all (``mix``: smooth for the skin
    parts, facet for the rest), then ``overrides`` on top; an override naming an unknown part raises."""
    names = [p.name or f"part{i}" for i, p in enumerate(model.parts)]
    if style == "mix":
        mapping = {n: MIX_STYLES["skin" if is_skin_part(p) else "other"] for n, p in zip(names, model.parts)}
    else:
        mapping = {n: style for n in names}
    for name, st in (overrides or {}).items():
        if name not in mapping:
            raise ValueError(f"--part-styles: no part {name!r} in {model.name!r} (parts: {', '.join(names)})")
        mapping[name] = st
    return mapping
