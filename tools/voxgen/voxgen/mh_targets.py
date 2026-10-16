"""MakeHuman morph targets (CC0, like the bundled base mesh) applied to base.obj vertices.

A *target* is a text file of lines ``vertex_index dx dy dz`` (decimetres, indices into the
base.obj vertex order, 19 158 vertices). MakeHuman's modelling sliders are *modifiers*
(``assets/makehuman/modeling_modifiers.json``): a two-sided modifier such as
``nose-scale-vert`` (``decr``/``incr``) applies ``nose-scale-vert-decr.target`` for negative
weights and ``nose-scale-vert-incr.target`` for positive ones; a one-sided one
(``head-oval``) is a single file. Several modifiers share a target stem and differ only
in their side labels (``nose-trans`` down|up / backward|forward / in|out), so the
canonical name is ``<target>-<min>|<max>`` and the bare stem is accepted only when it is
unique. ``<target>-<side>`` (``nose-trans-up``) also works and fixes the direction.
Weights beyond +-1 are allowed: that is how a face becomes a caricature. Macro
modifiers (Gender/Age/Weight/...) are combinatorial and not supported here.

Files: ``~/.cache/voxengine/makehuman/targets/<group>/<name>.target`` (downloaded from
the MakeHuman repository on first use), falling back to the copies committed under
``assets/makehuman/targets/`` for the targets the specs in the repo use.
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "makehuman"
CATALOGUE = ASSETS / "modeling_modifiers.json"
ASSET_TARGETS = ASSETS / "targets"
CACHE_DIR = Path(os.environ.get("VOXENGINE_CACHE", "~/.cache/voxengine")).expanduser() / "makehuman" / "targets"
SOURCE_URL = "https://raw.githubusercontent.com/makehumancommunity/makehuman/master/makehuman/data/targets"
N_VERTICES = 19158
FACE_GROUPS = ("head", "forehead", "eyebrows", "neck", "eyes", "nose", "mouth", "ears", "chin", "cheek")
MACRO_GROUPS = ("macrodetails", "macrodetails-universal", "macrodetails-height", "macrodetails-proportions")


@dataclass(frozen=True)
class Modifier:
    group: str
    target: str
    min: str | None = None
    max: str | None = None

    @property
    def two_sided(self) -> bool:
        return self.min is not None

    @property
    def name(self) -> str:
        return f"{self.target}-{self.min}|{self.max}" if self.two_sided else self.target

    def files(self) -> tuple[str | None, str]:
        """(file for negative weights or None, file for positive weights), relative to the
        targets root: ``<group>/<name>.target``."""
        if self.two_sided:
            return (f"{self.group}/{self.target}-{self.min}.target", f"{self.group}/{self.target}-{self.max}.target")
        return (None, f"{self.group}/{self.target}.target")


@lru_cache(maxsize=2)
def catalogue(path: str | None = None) -> dict[str, Modifier]:
    """{name: Modifier} from modeling_modifiers.json: canonical names, plus the bare target
    stem when it is unique, plus ``<target>-<side>`` aliases. Macro modifiers are skipped."""
    data = json.loads(Path(path or CATALOGUE).read_text(encoding="utf-8"))
    mods: list[Modifier] = []
    for group in data:
        if group["group"] in MACRO_GROUPS:
            continue
        for m in group["modifiers"]:
            if "target" not in m or "macrovar" in m:
                continue
            mods.append(Modifier(group["group"], m["target"], m.get("min"), m.get("max")))
    out: dict[str, Modifier] = {}
    stems: dict[str, int] = {}
    for m in mods:
        out[m.name] = m
        stems[m.target] = stems.get(m.target, 0) + 1
    for m in mods:
        if stems[m.target] == 1:
            out.setdefault(m.target, m)
        if m.two_sided:
            out.setdefault(f"{m.target}-{m.min}", m)
            out.setdefault(f"{m.target}-{m.max}", m)
    return out


def resolve(name: str) -> tuple[Modifier, int]:
    """Modifier for ``name`` and a direction: +1 normally, -1 when ``name`` is the
    ``<target>-<min>`` alias (then a positive weight means that side)."""
    cat = catalogue()
    if name not in cat:
        raise KeyError(f"unknown MakeHuman modifier {name!r}")
    m = cat[name]
    if m.two_sided and name == f"{m.target}-{m.min}" and name != m.name:
        return m, -1
    return m, 1


def expand_pairs(morphs: dict[str, float]) -> dict[str, float]:
    """``{"ear-scale": w}`` -> ``{"l-ear-scale": w, "r-ear-scale": w}`` for names that are
    not modifiers themselves but exist as an l-/r- pair (so a face stays symmetric)."""
    cat = catalogue()
    out: dict[str, float] = {}
    for name, w in morphs.items():
        if name in cat:
            out[name] = out.get(name, 0.0) + float(w)
        elif f"l-{name}" in cat and f"r-{name}" in cat:
            for side in ("l", "r"):
                out[f"{side}-{name}"] = out.get(f"{side}-{name}", 0.0) + float(w)
        else:
            raise KeyError(f"unknown MakeHuman modifier {name!r} (no l-/r- pair either)")
    return out


def _download(rel: str, dest: Path) -> None:
    url = f"{SOURCE_URL}/{rel}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            tmp.write_bytes(r.read())
    except Exception:
        proc = subprocess.run(["curl", "-sS", "-f", "-L", "-o", str(tmp), url], capture_output=True, text=True)
        if proc.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise FileNotFoundError(f"cannot download {url}: {proc.stderr.strip()}")
    tmp.replace(dest)


def target_path(rel: str, download: bool = True) -> Path:
    """Local path of ``<group>/<name>.target``: the cache, else the committed asset, else
    download into the cache (``download=False`` raises instead)."""
    cached = CACHE_DIR / rel
    if cached.is_file() and cached.stat().st_size > 0:
        return cached
    bundled = ASSET_TARGETS / rel
    if bundled.is_file():
        return bundled
    if not download:
        raise FileNotFoundError(f"target {rel} not in {CACHE_DIR} nor {ASSET_TARGETS}")
    _download(rel, cached)
    return cached


def parse_target(text: str) -> tuple[np.ndarray, np.ndarray]:
    """(indices int64 (n,), offsets float64 (n, 3) in decimetres) of a target file's text."""
    rows = [ln.split() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not rows:
        return np.zeros(0, np.int64), np.zeros((0, 3), np.float64)
    a = np.array(rows, np.float64)
    if a.ndim != 2 or a.shape[1] != 4:
        raise ValueError("target lines must be 'index dx dy dz'")
    idx = a[:, 0].astype(np.int64)
    if (idx < 0).any() or (idx >= N_VERTICES).any() or not np.all(a[:, 0] == idx):
        raise ValueError(f"target vertex indices must be integers in [0, {N_VERTICES})")
    return idx, a[:, 1:4].copy()


@lru_cache(maxsize=256)
def load_target(rel: str) -> tuple[np.ndarray, np.ndarray]:
    return parse_target(target_path(rel).read_text(encoding="utf-8"))


def files_for(morphs: dict[str, float]) -> list[tuple[str, float]]:
    """[(relative target file, weight)] for a {modifier: weight} dict (pairs expanded,
    sides chosen by the sign; zero weights dropped)."""
    out: list[tuple[str, float]] = []
    for name, w in expand_pairs(morphs).items():
        if w == 0.0:
            continue
        m, direction = resolve(name)
        w *= direction
        neg, pos = m.files()
        if w < 0 and neg is not None:
            out.append((neg, -w))
        else:
            out.append((pos, w))        # one-sided: a negative weight simply inverts the target
    return out


def apply_morphs(vertices_dm: np.ndarray, morphs: dict[str, float]) -> np.ndarray:
    """base.obj vertices (19158, 3) in decimetres -> morphed copy. ``morphs`` is
    {modifier name: weight}; weights add linearly, any magnitude."""
    v = np.array(vertices_dm, np.float64, copy=True)
    if v.shape != (N_VERTICES, 3):
        raise ValueError(f"expected the {N_VERTICES} base.obj vertices, got {v.shape}")
    for rel, w in files_for(morphs):
        idx, off = load_target(rel)
        np.add.at(v, idx, off * w)
    return v


def prefetch(groups=FACE_GROUPS) -> list[Path]:
    """Download every target of ``groups`` into the cache (skips files already there)."""
    paths = []
    for m in set(catalogue().values()):
        if m.group in groups:
            for rel in m.files():
                if rel is not None:
                    paths.append(target_path(rel))
    return paths
