"""MakeHuman hm08 base mesh (CC0, see assets/makehuman/README.md) in Godot space.

``load()`` reads ``assets/makehuman/base.obj`` once (cached) and returns a
:class:`MakeHuman` whose mesh is already converted: decimetres -> metres, turned 180
degrees about Y so the body faces -Z (handedness kept: the character's left stays
its left, which is -X in Godot), feet (``joint-ground``) at y = 0.

Groups in the file: ``body`` (the skin), ``helper-*`` proxies (``l-eye``/``r-eye``
eyeball spheres, ``upper-teeth``/``lower-teeth`` open tooth-row shells, ``tongue``,
``hair`` — a hull around the whole head *and face* used by MakeHuman to fit hair
meshes, not a scalp cap — plus tights/skirt/genital/eyelashes) and 125 ``joint-*``
groups: tiny cubes marking skeleton joint positions (a free skeleton for later).

``landmarks()`` turns helpers and joints into head measurements (metres): eye centres,
mouth centre (gap between the tooth rows), nose tip, chin, ears (from the body mesh;
hm08 has no ear joints), head top, neck base (``joint-neck``), skull base
(``joint-head``), jaw and mouth joints. Hairline: the hair helper gives no usable
edge (it wraps the face), so ``hairline_y`` is a convention: ``eye_y + 0.070``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import meshvox
from .objmesh import ObjMesh, load_obj

ASSET = Path(__file__).resolve().parent.parent / "assets" / "makehuman" / "base.obj"
SOURCE_URL = "https://raw.githubusercontent.com/makehumancommunity/makehuman/master/makehuman/data/3dobjs/base.obj"
MH_UNIT_M = 0.1  # MakeHuman works in decimetres

HELPERS = {
    "l_eye": "helper-l-eye", "r_eye": "helper-r-eye",
    "upper_teeth": "helper-upper-teeth", "lower_teeth": "helper-lower-teeth",
    "tongue": "helper-tongue", "hair": "helper-hair",
}


def to_godot(mesh: ObjMesh, ground_y_dm: float) -> ObjMesh:
    """dm -> m, rotate 180 deg about Y (face +Z -> -Z), feet at y = 0."""
    off = np.array([0.0, -ground_y_dm * MH_UNIT_M, 0.0])
    return mesh.transformed(lambda v: meshvox.rotate_y180(meshvox.scale(v, MH_UNIT_M)) + off)


@dataclass
class MakeHuman:
    """The converted mesh plus cached sub-meshes and landmarks."""

    mesh: ObjMesh
    _cache: dict = field(default_factory=dict, repr=False)

    # -- sub-meshes ------------------------------------------------------------
    def body(self) -> ObjMesh:
        return self._sub("body")

    def helper(self, name: str) -> ObjMesh:
        """Helper mesh by short name (``"l_eye"``, ``"upper_teeth"``, ``"hair"``, ...)
        or by its full group name."""
        return self._sub(HELPERS.get(name, name))

    def _sub(self, group: str) -> ObjMesh:
        if group not in self._cache:
            if not self.mesh.has_group(group):
                raise KeyError(f"no group {group!r} in the MakeHuman mesh")
            self._cache[group] = self.mesh.select(group)
        return self._cache[group]

    # -- joints ----------------------------------------------------------------
    def joint(self, name: str) -> np.ndarray:
        """Centre of the joint cube ``joint-<name>`` (metres)."""
        return self.mesh.group_centre(name if name.startswith("joint-") else f"joint-{name}")

    def joints(self) -> dict[str, np.ndarray]:
        """{name without the "joint-" prefix: centre} for all joint cubes."""
        return {g[6:]: self.mesh.group_centre(g) for g in self.mesh.groups if g.startswith("joint-")}

    # -- landmarks -------------------------------------------------------------
    def landmarks(self) -> dict:
        """Head measurements in metres (see module docstring). Cached."""
        if "landmarks" in self._cache:
            return self._cache["landmarks"]
        lm: dict = {}
        neck = self.joint("neck")
        lm["neck_base"] = neck
        lm["skull_base"] = self.joint("head")
        lm["head_joint_2"] = self.joint("head-2")
        lm["jaw_joint"] = self.joint("jaw")
        lm["mouth_joint"] = self.joint("mouth")
        eye_l = self.helper("l_eye").bounds()
        eye_r = self.helper("r_eye").bounds()
        lm["eye_l"] = (eye_l[0] + eye_l[1]) * 0.5
        lm["eye_r"] = (eye_r[0] + eye_r[1]) * 0.5
        lm["eye_radius"] = float((eye_l[1] - eye_l[0]).max() * 0.5)
        lm["eye_y"] = float((lm["eye_l"][1] + lm["eye_r"][1]) * 0.5)
        lm["eye_x"] = float(abs(lm["eye_l"][0] - lm["eye_r"][0]) * 0.5)
        lm["eye_z"] = float((lm["eye_l"][2] + lm["eye_r"][2]) * 0.5)
        ut = self.helper("upper_teeth").bounds()
        lt = self.helper("lower_teeth").bounds()
        lm["mouth"] = np.array([0.0, (ut[0][1] + lt[1][1]) * 0.5, (ut[0][2] + ut[1][2] + lt[0][2] + lt[1][2]) * 0.25])
        lm["mouth_half_width"] = float(max(ut[1][0], lt[1][0]))
        lm["teeth_front_z"] = float(min(ut[0][2], lt[0][2]))

        v = self.body().vertices
        head = v[v[:, 1] > neck[1]]
        lm["head_top"] = float(head[:, 1].max())
        tip = head[head[:, 2].argmin()]
        lm["nose_tip"] = tip.copy()
        lm["nose_base_y"] = float(tip[1] - 0.015)
        front = head[head[:, 2] < tip[2] + 0.06]
        lm["chin"] = front[front[:, 1].argmin()].copy()
        ear_x = float(head[:, 0].max())
        ear = head[np.abs(head[:, 0]) > ear_x - 0.012]
        lm["ear_x"] = ear_x
        lm["ear_inner_x"] = ear_x - 0.014
        lm["ear_y"] = (float(ear[:, 1].min()), float(ear[:, 1].max()))
        lm["ear_z"] = (float(ear[:, 2].min()), float(ear[:, 2].max()))
        lm["ear_centre"] = np.array([ear_x, ear[:, 1].mean(), ear[:, 2].mean()])
        lm["brow_y"] = lm["eye_y"] + 0.015
        lm["hairline_y"] = lm["eye_y"] + 0.070
        lm["head_width"] = float(head[:, 0].max() - head[:, 0].min())
        lm["head_depth"] = float(head[:, 2].max() - head[:, 2].min())
        lm["head_height"] = float(lm["head_top"] - lm["chin"][1])
        self._cache["landmarks"] = lm
        return lm


@lru_cache(maxsize=4)
def _load_cached(path: str) -> MakeHuman:
    raw = load_obj(path)
    ground = raw.group_centre("joint-ground")[1]
    return MakeHuman(to_godot(raw, ground))


def load(path=ASSET) -> MakeHuman:
    """The converted base mesh (cached per path). Raises FileNotFoundError when the
    asset is missing; nothing is downloaded."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MakeHuman base mesh not found: {path} (source: {SOURCE_URL})")
    return _load_cached(str(path.resolve()))
