"""The barber of the owner's concept sheet, built block by block after
``docs/reference/barber-head.md`` (``voxgen.blockhead``): a 10 x 13 x 9 U skull on the 20 mm unit
grid written in 5 mm voxels, its skin edges cut 45 degrees in whole units (front and back
vertical edges, the diagonal under the jaw, the forehead, the chin; render ``skull`` and ``nose``
with ``facet``, the rest ``cubes``), the flat painted face (recessed framed eyes looking to the
viewer's left, straight brows, the two-cube nose with its top sloped, the hidden mouth with its
teeth), the two-brown beard and moustache as blocks, chunky hair clumps at the temples, the
fringe and the nape, the red beret as slid ragged layers slouching to -X with the cream feather
staircase at the top-left. Head only for now: the body comes in the next pass, after the owner
approves the head."""
from voxgen.blockhead import BlockHeadParams, build_head

PARAMS = BlockHeadParams(name="barber")


def build():
    return build_head(PARAMS)
