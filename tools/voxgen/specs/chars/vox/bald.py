"""The barber's bare skull, to compare the head's form with the owner's bald turnaround
(``docs/reference/bald-head-turnaround.png``): the same maps as ``barber.py`` but only the
``SKULL`` profile map and the ``FACE`` relief (brows, recessed eyes, philtrum, cheek and lip
tones) - no beard, beret, hair or feather, and the moustache chars of the face map are dropped so
the head is clean-shaven and the mouth shows as a dark line. Render ``skull`` with
``facet --facet-step 2``. Edit the maps, not this file."""
from pathlib import Path

from voxgen.sheethead import build_head, load_maps

#: the maps module, imported by path (specs are not a package)
MAPS = load_maps(Path(__file__).with_name("barber_maps.py"))
SEED = 1
#: the face map without its moustache cells ('m' -> nothing): clean-shaven
FACE_BARE = {**MAPS.FACE, "rows": [row.replace("m", ".") for row in MAPS.FACE["rows"]]}
LAYERS = [("skull", MAPS.SKULL), ("face", FACE_BARE)]


def build():
    return build_head(seed=SEED, maps=MAPS, name="bald", layers=LAYERS)
