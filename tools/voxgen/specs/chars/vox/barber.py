"""The barber of the owner's concept sheet (``docs/reference/barber-head.md``), built from the
hand-drawn relief maps of ``barber_maps.py`` by ``voxgen.sheethead``: the skull and neck are the
blockhead form (the 10 x 13 x 9 U box on the 20 mm unit grid written in 5 mm voxels, with the
table-2.1 45-degree chamfers in whole units; render ``skull`` with ``facet``, the rest ``cubes``),
and everything on it comes from the maps - the face relief (brows, the recessed framed eyes, the
nose wedge and cube, the moustache, the light cheeks and lip), the beard, the red beret, the hair
(front and side maps) and the feather - laid down in that order, a later layer never overwriting
an earlier one. The maps are the art: edit them, not this file. Head only for now: the body comes
in the next pass, after the owner approves the head."""
from pathlib import Path

from voxgen.sheethead import build_head, load_maps

#: the maps module, imported by path (specs are not a package)
MAPS = load_maps(Path(__file__).with_name("barber_maps.py"))
SEED = 1


def build():
    return build_head(seed=SEED, maps=MAPS, name="barber")
