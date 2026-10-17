"""The bald head of docs/reference/bald-head-turnaround.png, built the way that sheet was made: a
smooth head form (voxgen.sdfhead: superellipsoid cranium, tapered jaw with a flat underside, nose
wedge, rounded neck) cut into coarse 15 mm voxels, plus the painted face (brows as blocks, eyes as
painted cells, the mouth as a slit) from a relief map. Render: ``--part-styles skull=facet
--facet-step 3`` (half-voxel chamfers on every coarse step), the brows/eyes/ears cubic."""
import importlib.util
from pathlib import Path

from voxgen.sheethead import build_head

_maps = importlib.util.spec_from_file_location("barber_maps", Path(__file__).with_name("barber_maps.py"))
maps = importlib.util.module_from_spec(_maps)
_maps.loader.exec_module(maps)

# features in map units U (1 u of the reference = 0.75 U): ears 1.5 u proud, 4 u tall, 3.5 u deep,
# at mid-depth; the mouth slit at y 4 u, 6 u wide; nostrils under the nose tip
SKULL = dict(view="sdf", material="MAT_SKIN", colour="skin",
             features=dict(ears=(5.5, 6.25, 5.0, 7.75, 5.25, 7.75), notches=(5.75, 6.25, 5.75, 7.25, 6.25, 7.0),
                           cavity_x=(-2.25, 2.25), cavity_y=(2.75, 3.25), lip_slit_y=3.0,
                           nostril=(0.25, 0.75, 4.0, 4.25, -1.0, -0.5)))

FACE = dict(
    view="front", cell=0.25, origin=(-5.5, 9.0), mirror=True, material="MAT_SKIN",
    depth={"b": (-0.5, 0.0)},
    paint={"K": "eye_frame", "W": "eye_white", "P": "pupil", "s": "skin_light", "b": "hair_black"},
    recess={"K": 0.25, "W": 0.25, "P": 0.25},
    materials={"b": "MAT_HAIR", "K": "MAT_EYE", "W": "MAT_EYE", "P": "MAT_EYE"},
    rows=[
        "......................",
        "......................",
        "......................",
        "......bbbbbbbbbbbbb...",
        "......bbbbbbbbbbbbb...",
        "......bbbbbbbbbbbbb...",
        "......................",
        "......................",
        "......................",
        "......KKKKKKKKK.......",
        "......WWWPPPWWW.......",
        "......WWWPPPWWW.......",
        "......WWWPPPWWW.......",
        "......................",
        "......................",
        "......................",
        "......................",
        "......................",
        "......................",
        "......................",
        "......................",
        "......................",
        "......................",
        "......................",
        "......................",
        "......................",
        "......................",
        "......................",
        "......................",
        "......................",
        "......................",
        "......................",
        "......................",
        "......................",
        "......................",
        "......................",
    ],
)

LAYERS = [("skull", SKULL), ("face", FACE)]


def build():
    return build_head(maps=maps, name="bald_sdf", layers=LAYERS)
