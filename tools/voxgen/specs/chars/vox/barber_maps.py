"""Relief maps of the barber's head, hand-read from ``docs/reference/barber-concept.png``
(see ``docs/reference/barber-head.md`` for the measurements). This file IS the model: the owner may
edit these maps like pixel art.

Format (consumed by ``voxgen.sheethead``): every layer is a dict with
  view     "front" (columns = x, rows = y, depth along z), "side" (columns = z, rows = y, depth along x)
           or "profile" (columns = z, rows = y; the char gives the HALF-WIDTH in x: the cell is filled
           for |x| < width[char] — the skull is built this way, so its profile is drawn, not coded);
  cell     size of one character in units U (1 U = 20 mm; 0.25 or 0.5);
  origin   (x0, y0) for front maps / (z0, y0) for side maps: the top-left corner of row 0, column 0,
           in U; row r covers y in [y0 - (r+1)*cell, y0 - r*cell), column c covers [o + c*cell, o + (c+1)*cell);
  rows     list of equal-length strings, top row first; "." = nothing;
  depth    char -> (lo, hi): the box filled along the depth axis (z for front maps, x for side maps), in U;
  width    profile maps: char -> half-width in U (symmetric about x = 0);
  mirror   True: the map gives the x <= 0 half only; the engine mirrors it about x = 0 (front maps);
  paint    char -> colour name: the char recolours the surface voxels of the already placed solid
           instead of adding anything (used for skin shades and the eye recess);
  recess   char -> depth in U cut into the existing solid before painting (eyes);
  shell    char -> thickness in U: the cell is filled with the dilation of the solid already placed
           (the skull) by that thickness, only where nothing is solid yet, within the cell's footprint
           (all depths): a layer that hugs whatever shape is under it (beard);
  palette  {colour name: weight} for the seeded per-block colouring of the added voxels;
  block    edge of the colour lattice in U (one colour per lattice cell);
  material voxgen material name for the added voxels.
Coordinates: +x is the viewer's right (the character's left), +y up, the face looks toward -z; the
cheek plane is z = 0 (the nose tip is at z = -2, the back of the skull at z = 13, the hair behind
it to z = 16.5, the beret from z = -3 to 17), the chin's bottom skin row is y = 0, the neck goes
down to y = -3. Profile measured on docs/reference/barber-head-sheet.png (18 wide, 24 tall, 20 deep).
"""

COLOURS = {
    "skin": "#de995f", "skin_light": "#ebc18f", "skin_shadow": "#c87e48",
    "hair": "#602f15", "hair_dark": "#4f2914", "hair_light": "#733b1d", "hair_black": "#35190b",
    "eye_frame": "#2c0e07", "pupil": "#1c0904", "eye_white": "#d8c9b0",
    "red": "#732015", "red_light": "#8c2b19", "red_dark": "#58160e", "red_crease": "#440f0a",
    "feather": "#d9c4a0", "feather_shadow": "#c6a582",
}

# Skull + neck as a PROFILE width map (1/2 U cells): columns z -2.5..14, rows y 13.5 -> -3.
# Half-width codes: 1=1 A=1.5 2=2 B=2.5 3=3 C=3.5 4=4 D=4.5 5=5. After docs/reference/bald-head-
# turnaround.png: the face is 8 wide (4) at the cheek plane z 0 and the sides 10 wide (5) from the
# brow down to the cheekbones; the crown is a dome (rows narrow in z and x above y 11); the back is
# rounded in 1/2 U steps and curves into the neck (occiput); the forehead slopes back from the brow
# to the hairline; the nose (x +-1) is a 45-degree wedge from the bridge down to a 2 U cube 2 U
# proud; below the cheekbones the jaw tapers (mouth rows 8 wide, chin 5..6 wide, 1/2 U proud) and
# its bottom runs from the chin back and UP to the jaw angle under the ear; the neck (5 wide,
# z 5.5..11.5) stands under the ear. Ears: |x| 5..6, z 7..10, y 4..7 with a 1 U notch (engine).
SKULL = dict(
    view="profile", cell=0.5, origin=(-2.5, 13.5), material="MAT_SKIN", colour="skin",
    width={"1": 1.0, "A": 1.5, "2": 2.0, "B": 2.5, "3": 3.0, "C": 3.5, "4": 4.0, "D": 4.5, "5": 5.0},
    rows=[
        ".............2222222222..........",  # y 13..13.5
        "...........3CCCCCCCCCC3B2A.......",  # y 12.5..13
        "..........C4444444444444C3B2.....",  # y 12..12.5
        ".........4DDDDDDDDDDDDDDD4C3B....",  # y 11.5..12
        "........4D5555555555555555D4C3...",  # y 11..11.5
        ".......4D555555555555555555D4C3..",  # y 10.5..11
        ".......4D555555555555555555D4C3..",  # y 10..10.5
        "......4D5555555555555555555D4C3..",  # y 9.5..10
        "......4D5555555555555555555D4C3..",  # y 9..9.5
        ".....4D55555555555555555555D4C3..",  # y 8.5..9
        ".....4D55555555555555555555D4C3..",  # y 8..8.5
        "....14D55555555555555555555D4C3..",  # y 7.5..8
        "....14D55555555555555555555D4C3..",  # y 7..7.5
        "...114D55555555555555555555D4C3..",  # y 6.5..7
        "..1114D55555555555555555555D4C3..",  # y 6..6.5
        ".11114D55555555555555555555D4C3..",  # y 5.5..6
        ".11114D55555555555555555555D4C3..",  # y 5..5.5
        ".11114D55555555555555555555D4C3..",  # y 4.5..5
        ".....4D55555555555555555555D4C3..",  # y 4..4.5
        ".....4DDDDDDDDDDDDDDDDDDDDD4C3B..",  # y 3.5..4
        ".....4DDDDDDDDDDDDDDDDDDDD4C3B...",  # y 3..3.5
        ".....C4DDDDDDDDDDDDDDDDDD4C3B....",  # y 2.5..3
        "....B3C444444444444444444444.....",  # y 2..2.5
        "....B3CCCCCCCCCCCCCCCCCCCBBB.....",  # y 1.5..2
        "....BB3333333333333333BBBBBB.....",  # y 1..1.5
        ".....B3333333333333BBBBBBBBB.....",  # y 0.5..1
        ".....2BBBBBBBBBBBBBBBBBBBBBB.....",  # y 0..0.5
        "................BBBBBBBBBBBB.....",  # y -0.5..0
        "................BBBBBBBBBBBB.....",  # y -1..-0.5
        "................BBBBBBBBBBBB.....",  # y -1.5..-1
        "................BBBBBBBBBBBB.....",  # y -2..-1.5
        "................BBBBBBBBBBBB.....",  # y -2.5..-2
        "................BBBBBBBBBBBB.....",  # y -3..-2.5
    ],
)

# Face relief on the cheek plane (z = 0), 1/4 U cells, left half (x -4..0), rows y 10 -> 0.
# b brow, m moustache (hair, +1/2 U); K eye frame, W white, P pupil (cut 1/2 U into the face and
# painted); _ philtrum (cut 1/2 U, skin); s light skin (paint only).
FACE = dict(
    view="front", cell=0.25, origin=(-4.0, 10.0), mirror=True, material="MAT_SKIN",
    depth={"b": (-0.5, 0.0), "m": (-0.5, 0.0)},
    paint={"K": "eye_frame", "W": "eye_white", "P": "pupil", "s": "skin_light", "d": "skin_shadow",
           "_": "skin", "b": "hair_black", "m": "hair_black"},
    recess={"K": 0.5, "W": 0.5, "P": 0.5, "_": 0.5},
    materials={"b": "MAT_HAIR", "m": "MAT_HAIR", "K": "MAT_SKIN", "W": "MAT_EYE", "P": "MAT_EYE"},
    rows=[
        "................",  # y 9.75-10  forehead
        "................",
        "................",
        "................",  # y 9-9.25
        "..bbbbbbbbbbbb..",  # y 8.75-9   brow x -3.5..-0.5, 1/2 U proud
        "..bbbbbbbbbbbb..",
        "..bbbbbbbbbbbb..",
        "..bbbbbbbbbbbb..",  # y 8-8.25
        "..KKKKKKKKKK....",  # y 7.75-8   eye frame x -3.5..-1, set 1/2 U into the face
        "..KKKKKKKKKK....",  # y 7.5-7.75 upper lid: the eye is half closed
        "..KWWPPPPWWK....",  # y 7.25-7.5 white 1/2, pupil 1 U, white 1/2
        "..KWWPPPPWWK....",
        "..KWWPPPPWWK....",  # y 6.75-7
        "..KKKKKKKKKK....",  # y 6.5-6.75
        "..ssss..........",  # y 6.25-6.5 cheek light 1x1 U
        "..ssss..........",
        "..ssss..........",
        "..ssss..........",  # y 5.5-5.75
        "................",  # y 5.25-5.5 (the nose comes from the SKULL profile map)
        "................",
        "................",
        "................",  # y 4.5-4.75
        "............____",  # y 4.25-4.5 philtrum: cut 1/2 U under the nose
        "............____",  # y 4-4.25
        "......mmmmmmmmmm",  # y 3.75-4   moustache x -2.5..2.5, 1/2 U proud
        "......mmmmmmmmmm",
        "......mmmmmmmmmm",
        "......mmmmmmmmmm",  # y 3-3.25
        "......mmmsssssss",  # y 2.75-3   moustache ends drop, lip strip light
        "......mmmsssssss",  # y 2.5-2.75
        "................",  # y 2.25-2.5 (beard layer covers)
        "................",
        "................",
        "................",
        "................",
        "................",
        "................",
        "................",
        "................",
        "................",  # y 0-0.25
    ],
)

# Beard: 1/2 U cells, left half (x -5.5..0), rows y 4 -> -1. Shell codes: the cell is filled with
# a shell grown from the skull already placed (dilation by the given thickness, only outside it),
# so the beard hugs the tapered jaw: '#' 0.75 U (chin and cheeks), '>' 0.5 U (sides), '=' 0.75 U
# under the chin (grows down from the jaw bottom).
BEARD = dict(
    view="front", cell=0.5, origin=(-5.5, 4.0), mirror=True, material="MAT_HAIR",
    shell={"#": 0.75, ">": 0.5, "=": 0.75},
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

# Beret, front map: 1/2 U cells, full width (x -9..9), rows y 18 -> 10. Letters give the z extent
# (mixed along a row so the plan outline is ragged, narrower letters toward the ends and toward the
# top so the plan is rounded and the cap is a dome, not a slab): A -2.5..17, B -2..16, C -1..14.5
# (band), D 0..13, E 1..11, F 2.5..9.5, G 4..8 (crown crumples). Slouch to -x.
BERET = dict(
    view="front", cell=0.5, origin=(-9.0, 18.0), mirror=False, material="MAT_CLOTH",
    depth={"A": (-2.5, 17.0), "B": (-2.0, 16.0), "C": (-1.0, 14.5), "D": (0.0, 13.0), "E": (1.0, 11.0),
           "F": (2.5, 9.5), "G": (4.0, 8.0)},
    palette={"red": 45, "red_light": 25, "red_dark": 20, "red_crease": 10}, block=1.5,
    rows=[
        "................GGG...GG............",  # y 17.5-18 crumples on the crown
        "..........FFFFFFFFFFFFFFF...........",  # y 17-17.5 crown
        "........FFEEEEEEEEEEEEEEFF..........",  # y 16.5-17
        "......EEDDDDDDDDDDDDDDDDEEE.........",  # y 16-16.5
        "....EDDDCCCCCCCCCCCCCCCCDDDE........",  # y 15.5-16
        "...DDCCCCBBBBBBBBBBBBBBBCCCDD.......",  # y 15-15.5
        "..DCCBBBBAAAAAAAAAAAAAABBBCCD.......",  # y 14.5-15
        ".DCBBAAAAAAAAAAAAAAAAAAAAAABBCD.....",  # y 14-14.5
        ".CBBAAAAAAAAAAAAAAAAAAAAAAAABBC.....",  # y 13.5-14 widest
        "DCBBAAAAAAAAAAAAAAAAAAAAAAAAABBC....",  # y 13-13.5
        "DCBBBBBBBBBBBBBBBBBBBBBBBBBBBBCD....",  # y 12.5-13
        ".DDDDDCCCCCCCCCCCCCCCCCCCCCCCC......",  # y 12-12.5 band x -5.5..5.5, droop x -8.5..-6
        ".DDDDDDCCCCCCCCCCCCCCCCCCCCCC.......",  # y 11.5-12
        "..DDDDD.............................",  # y 11-11.5 droop over the left temple
        "..DDDD..............................",  # y 10.5-11
        "...DD...............................",  # y 10-10.5
    ],
)

# Beret, side map: the back of the beret hangs lower than the band (profile sheet): columns
# z -2.5..17 (39 cols), rows y 11.5 -> 10; W = x -7..7.
BERET_SIDE = dict(
    view="side", cell=0.5, origin=(-2.5, 11.5), mirror=False, material="MAT_CLOTH",
    depth={"W": (-7.0, 7.0)},
    palette={"red": 45, "red_light": 25, "red_dark": 20, "red_crease": 10}, block=1.5,
    rows=[
        "..............................WWWWWWWWW",  # y 11-11.5 back slouch z 12.5..17
        "...............................WWWWWWWW",  # y 10.5-11
        ".................................WWWWW.",  # y 10-10.5
    ],
)

# Hair, front map: 1/2 U cells, full width (x -7..7), rows y 12 -> 2.5. Letters give the z extent:
# fringe rows follow the sloping forehead: d 0.5..2.5, c 0..2, b -0.5..1; side locks (in front of
# the ear) 1 -0.5..1.5, 2 0..2.5, 3 0.5..4, 6 1..4.5.
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

# Hair, side map (profile): 1/2 U cells, columns z -1.5..17 (37 cols), rows y 12 -> 1.5. Letters give
# the x extent: C -5.5..5.5 (a 1/2 U shell on the skull's sides and the mass behind it, z 13..16.5),
# B -6..6 (wider nape clumps). The ear (z 7..10, y 4..7) and the cheek side in front of it stay bare.
HAIR_SIDE = dict(
    view="side", cell=0.5, origin=(-1.5, 12.0), mirror=False, material="MAT_HAIR",
    depth={"C": (-5.5, 5.5), "B": (-6.0, 6.0)},
    palette={"hair": 5, "hair_dark": 3, "hair_light": 2}, block=1.0,
    rows=[
        ".....................................",  # y 11.5..12
        "......CCCCCCCCCCCCCCCCCCCCCCCCCCB....",  # y 11..11.5
        "......CCCCCCCCCCCCCCCCCCCCCCCCCCC....",  # y 10.5..11
        "......CCCCCCCCCCCCCCCCCCCCCCCCCCCB...",  # y 10..10.5
        ".......CCCCCCCCCCCCCCCCCCCCCCCCCCC...",  # y 9.5..10
        ".......CCCCCCCCCCCCCCCCCCCCCCCCCCCB..",  # y 9..9.5
        ".............CCCCCCCCCCCCCCCCCCCCCC..",  # y 8.5..9
        ".............CCCCCCCCCCCCCCCCCCCCCCB.",  # y 8..8.5
        ".............CCCCCCCCCCCCCCCCCCCCCCC.",  # y 7.5..8
        "..............CCCCCCCCCCCCCCCCCCCCCC.",  # y 7..7.5
        ".......................CCCCCCCCCCCCCB",  # y 6.5..7  (ear z 7..10 bare)
        ".......................CCCCCCCCCCCC..",  # y 6..6.5
        ".......................CCCCCCCCCCCCB.",  # y 5.5..6
        ".......................CCCCCCCCCCC...",  # y 5..5.5
        ".......................CCCCCCCCCCC...",  # y 4.5..5
        ".......................CCCCCCCCCCB...",  # y 4..4.5
        ".....................CCCCCCCCCCCC....",  # y 3.5..4
        ".......................CCCCCCCCC.....",  # y 3..3.5
        ".........................CCCCCCCB....",  # y 2.5..3
        "..........................CCCCC......",  # y 2..2.5
        "...........................CCC.......",  # y 1.5..2
    ],
)

# Feather: 1/2 U cells, x -6..2, rows y 23 -> 15; a 1 U staircase rising to the upper left from the
# beret's top (anchored inside puff 3 at y 15..16.5), standing at z 7..9. a z 7..8.5, b z 6.5..9.
FEATHER = dict(
    view="front", cell=0.5, origin=(-6.0, 23.0), mirror=False, material="MAT_PLANT",
    depth={"a": (7.0, 8.5), "b": (6.5, 9.0)},
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

# Write order: the skull map first (skin, with the engine's ears, mouth cavity and teeth), then
# these; a later layer never overwrites an earlier one, so the skull stays whole under everything
# shaveable.
LAYERS = [("skull", SKULL), ("face", FACE), ("beard", BEARD), ("beret", BERET), ("beret", BERET_SIDE),
          ("hair", HAIR_FRONT), ("hair", HAIR_SIDE), ("feather", FEATHER)]

for _name, _layer in LAYERS:
    _w = {len(r) for r in _layer["rows"]}
    assert len(_w) == 1, f"{_name}: ragged rows {_w}"
    _chars = set("".join(_layer["rows"])) - {"."}
    _known = (set(_layer.get("depth", {})) | set(_layer.get("paint", {})) | set(_layer.get("width", {}))
              | set(_layer.get("shell", {})))
    assert _chars <= _known, f"{_name}: unknown chars {_chars - _known}"
