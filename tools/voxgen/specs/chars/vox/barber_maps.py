"""Relief maps of the barber's head, hand-read from ``docs/reference/barber-concept.png``
(see ``docs/reference/barber-head.md`` for the measurements). This file IS the model: the owner may
edit these maps like pixel art.

Format (consumed by ``voxgen.sheethead``): every layer is a dict with
  view     "front" (columns = x, rows = y, depth along z) or "side" (columns = z, rows = y, depth along x);
  cell     size of one character in units U (1 U = 20 mm; 0.25 or 0.5);
  origin   (x0, y0) for front maps / (z0, y0) for side maps: the top-left corner of row 0, column 0,
           in U; row r covers y in [y0 - (r+1)*cell, y0 - r*cell), column c covers [o + c*cell, o + (c+1)*cell);
  rows     list of equal-length strings, top row first; "." = nothing;
  depth    char -> (lo, hi): the box filled along the depth axis (z for front maps, x for side maps), in U;
  mirror   True: the map gives the x <= 0 half only; the engine mirrors it about x = 0 (front maps);
  paint    char -> colour name: the char recolours the surface voxels of the already placed solid
           instead of adding anything (used for skin shades and the eye recess);
  recess   char -> depth in U cut into the existing solid before painting (eyes);
  palette  {colour name: weight} for the seeded per-block colouring of the added voxels;
  block    edge of the colour lattice in U (one colour per lattice cell);
  material voxgen material name for the added voxels.
Coordinates follow the skull frame of voxgen.blockhead: +x is the viewer's right (the character's
left), +y up, the face plane is z = 0 and looks toward -z, the chin's bottom skin row is y = 0.
"""

COLOURS = {
    "skin": "#de995f", "skin_light": "#ebc18f", "skin_shadow": "#c87e48",
    "hair": "#602f15", "hair_dark": "#4f2914", "hair_light": "#733b1d", "hair_black": "#35190b",
    "eye_frame": "#2c0e07", "pupil": "#1c0904", "eye_white": "#d8c9b0",
    "red": "#732015", "red_light": "#8c2b19", "red_dark": "#58160e", "red_crease": "#440f0a",
    "feather": "#d9c4a0", "feather_shadow": "#c6a582",
}

# Face relief on the flat face plane (z = 0), 1/4 U cells, left half (x -4..0), rows y 10 -> 0.
# digits 1..4: skin pushed out by n/4 U (the nose: the wedge rows 1,2,3,4 then the cube 4);
# b brow, m moustache (hair, +1/2 U); K eye frame, W white, P pupil (cut 1/4 U into the skull and
# painted); s light skin (paint only).
FACE = dict(
    view="front", cell=0.25, origin=(-4.0, 10.0), mirror=True, material="MAT_SKIN",
    depth={"1": (-0.25, 0.0), "2": (-0.5, 0.0), "3": (-0.75, 0.0), "4": (-1.0, 0.0),
           "b": (-0.5, 0.0), "m": (-0.5, 0.0)},
    paint={"K": "eye_frame", "W": "eye_white", "P": "pupil", "s": "skin_light", "d": "skin_shadow",
           "1": "skin", "2": "skin", "3": "skin", "4": "skin", "b": "hair_black", "m": "hair_black"},
    recess={"K": 0.25, "W": 0.25, "P": 0.25},
    materials={"b": "MAT_HAIR", "m": "MAT_HAIR", "K": "MAT_SKIN", "W": "MAT_EYE", "P": "MAT_EYE"},
    rows=[
        "................",  # y 9.75-10  forehead
        "................",
        "................",
        "................",  # y 9-9.25
        "..bbbbbbbbbbbb..",  # y 8.75-9   brow x -3.5..-0.5
        "..bbbbbbbbbbbb..",
        "..bbbbbbbbbbbb..",
        "..bbbbbbbbbbbb..",  # y 8-8.25
        "..KKKKKKKKKK.222",  # y 7.75-8   eye frame x -3.5..-1, nose bridge x -0.75..0
        "..KKKKKKKKKK.222",  # y 7.5-7.75 upper lid: the eye is half closed as on the sheet
        "..KWWPPPPWWK.222",  # y 7.25-7.5 white 1/2, pupil 1 U, white 1/2
        "..KWWPPPPWWK.222",
        "..KWWPPPPWWK.222",  # y 6.75-7
        "..KKKKKKKKKK.222",  # y 6.5-6.75
        "..ssss.......222",  # y 6.25-6.5 cheek light 1x1 U
        "..ssss.......222",
        "..ssss.......222",
        "..ssss.......222",  # y 5.5-5.75
        "............1111",  # y 5.25-5.5 nose wedge: 45 deg from the bridge down to the tip
        "............2222",
        "............3333",
        "............4444",  # y 4.5-4.75
        "............4444",  # y 4.25-4.5 nose cube, 1 U proud, flat bottom
        "............4444",
        "............4444",
        "............4444",  # y 3.5-3.75
        "......mmmmmmmmmm",  # y 3.25-3.5 moustache x -2.5..2.5
        "......mmmmmmmmmm",
        "......mmmmmmmmmm",
        "......mmmmmmmmmm",
        "......mmmmmmmmmm",  # y 2.25-2.5
        "......mmmsssssss",  # y 2-2.25   moustache ends drop, lip strip light
        "......mmmsssssss",
        "......mmmsssssss",  # y 1.5-1.75
        "................",  # y 1.25-1.5 (beard layer covers)
        "................",
        "................",
        "................",
        "................",
        "................",  # y 0-0.25
    ],
)

# Beard: 1/2 U cells, left half (x -5.5..0), rows y 4 -> -1.
# '#' on the face plane (z -0.75..0); '>' side strip / chamfer filler (z -0.75..3, only outside the skull);
# '=' under the chin (z -0.75..2).
BEARD = dict(
    view="front", cell=0.5, origin=(-5.5, 4.0), mirror=True, material="MAT_HAIR",
    depth={"#": (-0.75, 0.0), ">": (-0.75, 3.0), "=": (-0.75, 2.0)},
    palette={"hair_dark": 5, "hair_black": 3, "hair": 2}, block=1.0,
    rows=[
        "...#.......",  # y 3.5-4   beard top at the face edge
        "..>##......",  # y 3-3.5
        ">>>##......",  # y 2.5-3
        ">>>###.....",  # y 2-2.5
        ">>>####....",  # y 1.5-2   (lip strip x >= -1.5 stays bare)
        ".>>########",  # y 1-1.5   chin block + goatee ring
        "..>########",  # y 0.5-1
        "...########",  # y 0-0.5
        "....=======",  # y -0.5-0  hangs under the chin
        ".......====",  # y -1-(-0.5) centre tuft
    ],
)

# Beret: 1/2 U cells, full width (x -9..9), rows y 18 -> 10. Letters give the z extent (mixed
# along a row so the plan outline is ragged, narrower letters at the ends so the plan is rounded):
# A -2..11.5, B -1.5..11, C -1..10 (band), D 0..9, E 1..7 (crumples on top). Slouch to -x.
BERET = dict(
    view="front", cell=0.5, origin=(-9.0, 18.0), mirror=False, material="MAT_CLOTH",
    depth={"A": (-2.0, 11.5), "B": (-1.5, 11.0), "C": (-1.0, 10.0), "D": (0.0, 9.0), "E": (1.0, 7.0)},
    palette={"red": 45, "red_light": 25, "red_dark": 20, "red_crease": 10}, block=1.5,
    rows=[
        "................EEE...EE............",  # y 17.5-18 crumples (both rest on the row below)
        "..........EEDDDDDDDDDDDEE...........",  # y 17-17.5
        "........DDDDDDDDDDDDDDDDDD..........",  # y 16.5-17
        "......DDCCCCCCCDDDCCCCCCCCDD........",  # y 16-16.5
        "....CCCCCCBBBBBBCCCCCBBBBBBBCC......",  # y 15.5-16
        "...BBBBCCCBBBBBBBBBBBBBBBBBBCCC.....",  # y 15-15.5
        "..CBBAAAABBBAAAAAAABBBAAAAABBC......",  # y 14.5-15
        ".DCBBAAAAAAAAAAAAAAAAAAAAAAAABBC....",  # y 14-14.5
        ".CBBAAAAAAAAAAAAAAAAAAAAAAAAAABBC...",  # y 13.5-14 widest
        "DCBBAAAAAAAAAAAAAAAAAAAAAAAAAABBC...",  # y 13-13.5
        "DCBBBBBBBBBBBBBBBBBBBBBBBBBBBBCD....",  # y 12.5-13
        ".DDDDDCCCCCCCCCCCCCCCCCCCCCCCC......",  # y 12-12.5 band x -5.5..5.5, droop x -8.5..-6
        ".DDDDDDCCCCCCCCCCCCCCCCCCCCCC.......",  # y 11.5-12
        "..DDDDD.............................",  # y 11-11.5 droop over the left temple
        "..DDDD..............................",  # y 10.5-11
        "...DD...............................",  # y 10-10.5
    ],
)

# Hair, front map: 1/2 U cells, full width (x -7..7), rows y 12 -> 2.5. Letters give the z extent:
# fringe rows follow the sloping forehead: d 0.5..2.5, c 0..2, b -0.5..1; side locks 1 -0.5..1.5,
# 2 0..2.5, 3 0.5..4, 6 1..4.5.
HAIR_FRONT = dict(
    view="front", cell=0.5, origin=(-7.0, 12.0), mirror=False, material="MAT_HAIR",
    depth={"b": (-0.5, 1.0), "c": (0.0, 2.0), "d": (0.5, 2.5),
           "1": (-0.5, 1.5), "2": (0.0, 2.5), "3": (0.5, 4.0), "6": (1.0, 4.5)},
    palette={"hair": 5, "hair_dark": 3, "hair_light": 2}, block=1.0,
    rows=[
        "............................",  # y 11.5-12 (band)
        "..111dddddddddddddddddd111..",  # y 11-11.5 fringe under the band, locks x 4.5..6
        ".2111ccccccccccccccccc1112..",  # y 10.5-11
        "..21bbbbb.bbb.bbbbbbb.bb12..",  # y 10-10.5 ragged lower edge
        ".321bb....b......b....bb123.",  # y 9.5-10 tufts over the temples
        ".321b..................b123.",  # y 9-9.5
        ".321....................123.",  # y 8.5-9
        ".3321..................1233.",  # y 8-8.5
        "3321....................1233",  # y 7.5-8
        "3322....................2233",  # y 7-7.5
        ".332....................233.",  # y 6.5-7
        "3332....................2333",  # y 6-6.5
        "3362....................2633",  # y 5.5-6
        ".662....................266.",  # y 5-5.5
        ".662....................266.",  # y 4.5-5
        "..62....................26..",  # y 4-4.5
        "..66....................66..",  # y 3.5-4
        "...6....................6...",  # y 3-3.5
        "...6....................6...",  # y 2.5-3
    ],
)

# Hair, side map (profile): 1/2 U cells, columns z -1.5..11, rows y 12 -> 2.5. Letters give the
# x extent: C -5.5..5.5 (a 1/2 U shell on the skull's sides and back), B -6..6 (wider nape clumps).
HAIR_SIDE = dict(
    view="side", cell=0.5, origin=(-1.5, 12.0), mirror=False, material="MAT_HAIR",
    depth={"C": (-5.5, 5.5), "B": (-6.0, 6.0)},
    palette={"hair": 5, "hair_dark": 3, "hair_light": 2}, block=1.0,
    rows=[
        ".........................",  # y 11.5-12 (band)
        ".....CCCCCCCCCCCCCCCCCCC.",  # y 11-11.5 z 1..10.5
        ".....CCCCCCCCCCCCCCCCCCC.",  # y 10.5-11
        ".....CCCCCCCCCCCCCCCCCCCB",  # y 10-10.5
        ".....CCCCCCCCCCCCCCCCCCBB",  # y 9.5-10
        "......CCCCCCCCCCCCCCCCCCC",  # y 9-9.5
        "......CCCCCCCCCCCCCCCCCC.",  # y 8.5-9
        "......CCCCCCCCCCCCCCCCCCB",  # y 8-8.5
        "......CCCCCCCCCCCCCCCCCBB",  # y 7.5-8
        ".......CCCCCCCCCCCCCCCCC.",  # y 7-7.5
        ".............CCCCCCCCCCC.",  # y 6.5-7
        ".............CCCCCCCCCCCB",  # y 6-6.5
        ".............CCCCCCCCCCC.",  # y 5.5-6
        "..............CCCCCCCCCC.",  # y 5-5.5
        "..............CCCCCCCCCCB",  # y 4.5-5
        "...............CCCCCCCCC.",  # y 4-4.5
        "................CCCCCCCC.",  # y 3.5-4
        ".................CCCCCC..",  # y 3-3.5
        "..................CCCC...",  # y 2.5-3 nape ends above the collar
    ],
)

# Feather: 1/2 U cells, x -6..2, rows y 23 -> 15; a 1 U staircase rising to the upper left from the
# beret's top (anchored inside puff 3 at y 15..16.5). a z 3..4.5, b z 2.5..5 (the wider vane).
FEATHER = dict(
    view="front", cell=0.5, origin=(-6.0, 23.0), mirror=False, material="MAT_PLANT",
    depth={"a": (3.0, 4.5), "b": (2.5, 5.0)},
    palette={"feather": 7, "feather_shadow": 3}, block=1.0,
    rows=[
        ".aa.............",  # y 22.5-23
        ".aaa............",
        "..aaa...........",
        "..aaa...........",
        "...aaa..........",
        "...aaaa.........",
        "....aaaa........",
        "....baaa........",
        ".....bbaa.......",
        ".....bbbaa......",
        "......bbba......",
        ".......bbb......",
        ".......bbbb.....",
        "........bbb.....",
        "........bb......",
        ".........b......",  # y 15-15.5 base inside the beret
    ],
)

# Write order: the skull and neck (voxgen.blockhead), then these; a later layer never overwrites
# an earlier one, so the skull stays whole under everything shaveable.
LAYERS = [("face", FACE), ("beard", BEARD), ("beret", BERET), ("hair", HAIR_FRONT), ("hair", HAIR_SIDE),
          ("feather", FEATHER)]

for _name, _layer in LAYERS:
    _w = {len(r) for r in _layer["rows"]}
    assert len(_w) == 1, f"{_name}: ragged rows {_w}"
    _chars = set("".join(_layer["rows"])) - {"."}
    _known = set(_layer.get("depth", {})) | set(_layer.get("paint", {}))
    assert _chars <= _known, f"{_name}: unknown chars {_chars - _known}"
