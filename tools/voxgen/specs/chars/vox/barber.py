"""The barber of the owner's concept sheet (``docs/reference/barber-head.md``, profile and back on
``docs/reference/barber-head-sheet.png``), built from the hand-drawn relief maps of
``barber_maps.py`` by ``voxgen.sheethead``: the skull and neck are the ``SKULL`` profile map
(1/2 U cells on the 20 mm unit grid written in 5 mm voxels, each char the half-width in x: sloped
forehead, 45-degree nose wedge to a 2 U cube, proud chin, jaw to the ear, rounded back; render
``skull`` with ``facet --facet-step 2``, the rest ``cubes``), to which the engine adds the ears,
the hidden mouth with its teeth, the tissue layers and the neck tone; everything on it comes from
the other maps - the face relief (brows, the recessed framed eyes, the philtrum, the moustache,
the light cheeks and lip), the beard, the red beret (front and side maps), the hair (front and
side maps) and the feather - laid down in that order, a later layer never overwriting an earlier
one. The maps are the art: edit them, not this file. Head only for now: the body comes in the
next pass, after the owner approves the head."""
from pathlib import Path

from voxgen.sheethead import build_head, load_maps

#: the maps module, imported by path (specs are not a package)
MAPS = load_maps(Path(__file__).with_name("barber_maps.py"))
SEED = 1


def build():
    return build_head(seed=SEED, maps=MAPS, name="barber")
