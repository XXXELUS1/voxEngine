# MakeHuman base mesh (hm08) and morph targets

`base.obj` is the MakeHuman 1.x base mesh ("basemesh hm08"), copied unchanged from

    https://raw.githubusercontent.com/makehumancommunity/makehuman/master/makehuman/data/3dobjs/base.obj

License: **CC0** (public domain dedication). The file header says: "This asset was
explicitly released as CC0 in september 2020"; copyright holders at that point were
Data Collection AB, Joel Palmius and Jonas Hauquier (see the header).

Contents: 19 158 vertices, 18 486 quads; units are decimetres, Y up, the body faces +Z.
Groups: `body` (skin), `helper-*` proxies (eyes, teeth, tongue, hair hull, tights, skirt,
genital, eyelashes) and 125 `joint-*` cubes marking skeleton joints.

voxgen reads it through `voxgen.mh.load()` (metres, facing -Z, feet at y = 0); the
generator never downloads the base mesh.

## Morph targets (`targets/`, `modeling_modifiers.json`)

`modeling_modifiers.json` is MakeHuman's modelling-slider catalogue, copied unchanged from
`makehuman/data/modifiers/modeling_modifiers.json` of the same repository. `targets/<group>/<name>.target`
are the morph targets (text lines `vertex_index dx dy dz`, decimetres, indices into the
`base.obj` vertex order) copied unchanged from `makehuman/data/targets/<group>/<name>.target`.
Same CC0 release and copyright holders as the base mesh (every file carries the header).

Only the targets used by the committed specs are bundled (`specs/chars/*.py`,
`specs/calibration/head_mh_3mm.py`). Any other modifier is downloaded on first use by
`voxgen.mh_targets` into `~/.cache/voxengine/makehuman/targets/` (`mh_targets.prefetch()` fetches
all face groups); the bundled copies are the offline fallback.
