"""Whole characters: a sculpted head (:mod:`voxgen.sculpthead`, 3 mm) standing on a MakeHuman
body (:mod:`voxgen.body_from_mesh`, 10 mm) in clothes, as one multi-part model.

Why: the generator must be a *universal* character tool in one style, not a copier of one
reference - five townspeople (barber, knight, monk, drunk, widow) built from the same two
constructions with per-character knobs, tones and garments, every part faceted at its own
voxel size. The head keeps its local frame (chin's bottom at y = 0, face toward -Z); the body
is cut ``neck_cut`` above the neck joint and the head's parts are shifted so the head's neck
(its lowest row, ``extent_y[0]``) stands on the stub and its neck axis sits over the body's
neck joint - no overlap between the head and the body. Character-level paint on the head:
``scar`` (a one-voxel line on the viewer's-left cheek, the only asymmetric thing in a
character, colour only) and ``nose_tint`` (the nose tip and wings in the ruddy skin tone).

Parts: ``body`` (10 mm), ``clothes`` (10 mm, when the outfit has a shell), ``head``, ``eye_l``,
``eye_r``, ``hair``, ``beard``, ``scarf`` (3 mm) - the heads' names, so the lab scene finds
them. One palette: the head's box palette plus the garment tones. ``meta["mesh"]`` counts
cells / solid / quads per part, ``meta["build"]`` the head's, body's and clothes' build stats,
``meta["seams"]`` the head's detail seams.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import materials as M
from .body import exposed
from .body_from_mesh import BodyParams, Outfit, build_body, clothes
from .model import Model, Part
from .multires_head import mesh_stats
from .sculpthead import SculptParams, build_sculpt_head


@dataclass
class CharacterParams:
    name: str = "character"
    head: SculptParams = field(default_factory=lambda: SculptParams(voxel=0.003, hair_style="short"))
    body: BodyParams = field(default_factory=BodyParams)
    outfit: Outfit = field(default_factory=Outfit)
    scar: bool = False               # a painted scar line on the viewer's-left cheek
    nose_tint: bool = False          # the nose tip and wings in the ruddy skin tone
    scar_from: tuple[float, float] = (-0.044, 0.120)   # (x, y) metres in the head frame
    scar_to: tuple[float, float] = (-0.031, 0.088)


def paint_scar(head: Part, a: tuple[float, float], b: tuple[float, float], C) -> int:
    """Exposed skin voxels of the face within half a voxel of the segment ``a``-``b`` (x, y
    metres, the face side z < -0.02) in the ``scar`` tone. Colour only. Returns the count."""
    X, Y, Z = head.coords()
    vs = head.voxel_size
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = np.clip(((X - ax) * dx + (Y - ay) * dy) / max(L2, 1e-12), 0.0, 1.0)
    d = np.sqrt((X - (ax + t * dx)) ** 2 + (Y - (ay + t * dy)) ** 2)
    skin = head.material == M.MAT_SKIN
    sel = exposed(head.solid()) & skin & (d <= 0.55 * vs) & (Z < -0.02) & (head.color == C("skin"))
    head.color[sel] = C("scar")
    return int(sel.sum())


def paint_nose_tint(head: Part, p: SculptParams, C) -> int:
    """The nose tip sphere and the wing spheres (plus half a voxel) of the exposed skin in the
    ruddy tone; the nostril dots stay. Returns the count."""
    X, Y, Z = head.coords()
    vs = head.voxel_size
    tip = np.sqrt((X - p.nose_tip[0]) ** 2 + (Y - p.nose_tip[1]) ** 2 + (Z - p.nose_tip[2]) ** 2) <= p.nose_tip_r + 0.5 * vs
    wings = np.sqrt((np.abs(X) - p.wing_c[0]) ** 2 + (Y - p.wing_c[1]) ** 2 + (Z - p.wing_c[2]) ** 2) <= p.wing_r + 0.5 * vs
    sel = exposed(head.solid()) & (head.material == M.MAT_SKIN) & (tip | wings) & (head.color == C("skin"))
    head.color[sel] = C("skin_ruddy")
    return int(sel.sum())


def shift_parts(model: Model, offset: tuple[float, float, float]) -> None:
    """Move every part of ``model`` by ``offset`` (metres) - the head onto the body."""
    for part in model.parts:
        part.origin = tuple(part.origin[i] + offset[i] for i in range(3))
        part._coords = None


def build_character(params: CharacterParams, head_only: bool = False) -> Model:
    """The character model (see the module doc); with ``head_only`` the painted head alone, in
    its own frame, named ``<name>_head`` (for the head close-ups)."""
    p = params
    hp = p.head.resolved()
    head_model = build_sculpt_head(hp, name=f"{p.name}_head" if head_only else p.name)
    pal = head_model.palette
    skin = pal[pal.index("skin")]
    line = pal[pal.index("line")]
    pal.add("scar", tuple(int(round(0.5 * s + 0.5 * l)) for s, l in zip(skin, line)))
    C = pal.index
    head = head_model.part("head")
    hstats = dict(head_model.meta["build"])
    if p.scar:
        hstats["scar"] = paint_scar(head, p.scar_from, p.scar_to, C)
    if p.nose_tint:
        hstats["nose_tint"] = paint_nose_tint(head, hp, C)
    head_model.meta["build"] = hstats
    if head_only:
        head_model.meta["character"] = p.name
        return head_model

    body = build_body(replace(p.body, skin_tone=hp.skin_tone), palette=pal)
    cpart, cstats = clothes(body, p.outfit, pal)
    lm = body.landmarks
    offset = (0.0, lm["neck_top"] - hp.extent_y[0], lm["neck_z"] - hp.neck_z)
    shift_parts(head_model, offset)

    model = Model(p.name, p.body.voxel, pal,
                  meta={"generator": "character", "head_voxel_mm": round(hp.voxel * 1000, 2),
                        "body_voxel_mm": round(p.body.voxel * 1000, 2), "outfit": p.outfit.kind,
                        "build": {"head": hstats, "body": body.stats, "clothes": cstats},
                        "head_offset": [round(v, 4) for v in offset], "seams": head_model.meta["seams"]})
    model.add_part(body.part)
    if cpart is not None:
        model.add_part(cpart)
    for part in list(head_model.parts):
        model.add_part(part)
    model.meta["mesh"] = mesh_stats(model)
    return model
