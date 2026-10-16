# MakeHuman base mesh (hm08)

`base.obj` is the MakeHuman 1.x base mesh ("basemesh hm08"), copied unchanged from

    https://raw.githubusercontent.com/makehumancommunity/makehuman/master/makehuman/data/3dobjs/base.obj

License: **CC0** (public domain dedication). The file header says: "This asset was
explicitly released as CC0 in september 2020"; copyright holders at that point were
Data Collection AB, Joel Palmius and Jonas Hauquier (see the header).

Contents: 19 158 vertices, 18 486 quads; units are decimetres, Y up, the body faces +Z.
Groups: `body` (skin), `helper-*` proxies (eyes, teeth, tongue, hair hull, tights, skirt,
genital, eyelashes) and 125 `joint-*` cubes marking skeleton joints.

voxgen reads it through `voxgen.mh.load()` (metres, facing -Z, feet at y = 0); the
generator never downloads anything.
