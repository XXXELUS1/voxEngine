"""Palette: up to 256 sRGB colours, index 0 reserved for air (black)."""

from __future__ import annotations

import numpy as np

RGB = tuple[int, int, int]


class Palette:
    """Ordered list of named sRGB colours.

    * index 0 is always air = (0, 0, 0) and cannot be changed,
    * ``add(name, rgb)`` returns the index (an identical rgb already present is reused),
    * at most 256 entries; ``add`` raises ValueError when the palette is full.

    Typical spec usage::

        pal = Palette()
        SKIN = pal.add("skin", (224, 180, 150))
        HAIR = pal.add("hair", (60, 40, 30))
    """

    MAX = 256

    def __init__(self, colors: list[RGB] | None = None):
        self.colors: list[RGB] = [(0, 0, 0)]
        self.names: dict[str, int] = {"air": 0}
        if colors:
            for i, c in enumerate(colors):
                if i == 0:
                    continue
                self._append(c, f"c{i}")

    # -- building -----------------------------------------------------------
    def _append(self, rgb, name: str) -> int:
        rgb = _clean(rgb)
        if len(self.colors) >= self.MAX:
            raise ValueError(f"palette full ({self.MAX} colours), cannot add {name!r}")
        self.colors.append(rgb)
        idx = len(self.colors) - 1
        self.names.setdefault(name, idx)
        return idx

    def add(self, name: str, rgb) -> int:
        """Add a colour and return its index. Same rgb -> same index (deduplicated).

        If ``name`` already exists with a different rgb, a new entry is added and the
        name is re-pointed to it.
        """
        rgb = _clean(rgb)
        if rgb == (0, 0, 0):
            # Black is reserved for air; nudge so that solids never get colour index 0.
            rgb = (1, 1, 1)
        for i, c in enumerate(self.colors):
            if i and c == rgb:
                self.names[name] = i
                return i
        idx = self._append(rgb, name)
        self.names[name] = idx
        return idx

    def get(self, name: str, default=None):
        """Index of a named colour or ``default``."""
        return self.names.get(name, default)

    def index(self, name: str) -> int:
        """Index of a named colour (KeyError if missing)."""
        return self.names[name]

    def nearest(self, rgb) -> int:
        """Index of the closest colour (Euclidean in sRGB), never 0."""
        if len(self.colors) < 2:
            raise ValueError("palette has no colours")
        arr = self.as_array()[1:].astype(np.float32)
        d = ((arr - np.asarray(_clean(rgb), np.float32)) ** 2).sum(axis=1)
        return int(np.argmin(d)) + 1

    # -- access --------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.colors)

    def __getitem__(self, idx: int) -> RGB:
        return self.colors[idx]

    def __iter__(self):
        return iter(self.colors)

    def name_of(self, idx: int) -> str:
        for n, i in self.names.items():
            if i == idx:
                return n
        return f"c{idx}"

    def as_array(self, size: int = 256) -> np.ndarray:
        """(size, 3) uint8 array, unused slots black."""
        out = np.zeros((size, 3), np.uint8)
        n = min(size, len(self.colors))
        out[:n] = np.asarray(self.colors[:n], np.uint8)
        return out

    def to_list(self) -> list[list[int]]:
        """JSON-friendly ``[[r,g,b], ...]`` (what the .vxm header stores)."""
        return [list(c) for c in self.colors]

    @classmethod
    def from_list(cls, colors) -> "Palette":
        p = cls()
        for i, c in enumerate(colors):
            if i == 0:
                continue
            p._append(c, f"c{i}")
        return p

    def __repr__(self) -> str:
        return f"Palette({len(self.colors)} colours)"


def _clean(rgb) -> RGB:
    r, g, b = (int(round(float(v))) for v in rgb[:3])
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
