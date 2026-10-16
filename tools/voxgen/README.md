# voxgen — генератор воксельных моделей

Оффлайн-половина пайплайна voxEngine: Python-спеки на SDF → `.vxm` (для Godot/voxbody),
`.vox` (MagicaVoxel) + `.vox.json`, контактный лист превью (глаза агента) и отчёт проверок.
Контракт CLI: `docs/spec/voxgen-cli.md`, формат файла: `docs/spec/vxm-format.md`.

## Установка и запуск

```bash
pip install -e tools/voxgen            # или PYTHONPATH=tools/voxgen
python3 -m voxgen build-all tools/voxgen/specs --out godot/assets/models --preview build/previews --vox build/vox
python3 -m voxgen build tools/voxgen/specs/head_test.py
python3 -m voxgen preview godot/assets/models/head_test.vxm --views front,iso,slice_x --size 320
python3 -m voxgen check   godot/assets/models/head_test.vxm
python3 -m voxgen info    godot/assets/models/head_test.vxm
python3 -m voxgen import-vox build/vox/head_test.vox --spec-json build/vox/head_test.vox.json --out /tmp/h.vxm
cd tools/voxgen && python3 -m pytest -q
```

`build`/`build-all` пишут `<name>.vxm`, `<name>.vox` + `<name>.vox.json`, `<name>.png`
(контактный лист 4×2: front/back/left/right/top/bottom/iso/iso2 + инфо-полоса) и
`build/reports/<name>.json` (`{"errors","warnings","stats"}`); код возврата 1, если есть ошибки.
Спеки, начинающиеся с `_`, в `build-all` пропускаются (это библиотечные модули, например `_human_head.py`).

Дополнительные виды для `preview --views`: `slice_x`, `slice_y`, `slice_z` — срезы по материалам,
чтобы увидеть слои (кожа/жир/мышца/кость).

## Как написать спек

Спек — модуль с функцией `build() -> voxgen.Model`. Всё задаётся **в метрах** в локальном
пространстве части (Godot: Y вверх, персонаж смотрит в −Z). Библиотека сама переводит в воксели:
SDF-поля хранятся в вокселях (`field <= 0` — внутри), толщины слоёв тоже в вокселях
(`part.m2v(0.005)` переводит метры в воксели).

```python
from voxgen import Model, Part, Palette, sdf, body
from voxgen.materials import *

def build():
    pal = Palette()                               # индекс 0 = воздух, до 256 цветов
    pal.add("skin", (224, 180, 150))              # имя = имя материала -> подхватится fill_layers
    pal.add("fat", (238, 214, 140)); pal.add("muscle", (150, 40, 45)); pal.add("bone", (235, 225, 200))
    HAIR = pal.add("hair", (60, 40, 30))

    m = Model("blob", voxel_size=0.005, palette=pal)
    p = m.add_part(Part.from_extent("head", (-0.1, -0.02, -0.1), (0.1, 0.2, 0.1), 0.005,
                                    bone="head", symmetric=True))
    gm = sdf.mirror_x(p)                          # парные детали: рисуем на +X, получаем обе стороны
    skin = sdf.ellipsoid(p, (0, 0.09, 0), (0.07, 0.09, 0.08))
    skin = sdf.smooth_union(skin, sdf.sphere(gm, (0.075, 0.09, 0), 0.02), k=p.m2v(0.01))
    skin = skin + sdf.noise(gm, scale=0.03, amplitude=0.002, seed=1)   # органичный силуэт, симметрично
    body.fill_layers(p, skin, [(MAT_SKIN, 1), (MAT_FAT, 1), (MAT_MUSCLE, None)],
                     core=(MAT_BONE, sdf.ellipsoid(p, (0, 0.09, 0), (0.055, 0.075, 0.065))))
    body.grow_hair(p, sdf.half_space(p, (0, 0.14, 0), (0, -1, 0)), (3, 8), HAIR,
                   direction="normal", sweep=(0, -0.3, 0.4), jitter=0.3)
    body.tint_region(p, sdf.sphere(gm, (0.045, 0.07, -0.06), 0.025), pal.add("ruddy", (229, 166, 142)))
    return m
```

Что есть в библиотеке:

- `Model(name, voxel_size, palette)`, `Part(name, dims, voxel_size, origin, bone, symmetric, budget)`,
  `Part.from_extent(name, lo, hi, voxel_size)`, `Part.centered(name, size, voxel_size)` (пропсы: стоит на y=0,
  центр по x/z). Поля `material/color/state` — `uint8` формы `(nz, ny, nx)`, индекс `[z, y, x]`
  (запись в `.vxm` идёт x-fastest через `tobytes()`).
- `sdf`: `sphere, ellipsoid, capsule, box(round=), cylinder, cone, torus, half_space`;
  `union, intersect, subtract, smooth_union, smooth_subtract, offset, shell, displace`;
  трансформы сетки `translate, rotate(euler_deg, pivot), scale, mirror_x`; шум `noise(scale_m, amplitude_m, seed)`.
- `body`: `fill_layers(part, surface, layers, core=)` — материал по глубине, последний слой с толщиной `None`
  заполняет остальное, кости/хрящи — отдельными SDF, внешний слой защищён; `grow_hair(part, mask, length_vox,
  color, direction, sweep, jitter, density, seed)` — 6-связные столбики MAT_HAIR от кожи наружу;
  `tint_region(part, region, color, materials=)`; `paint(part, region, material, color, only=)`.
- `checks.run(model)`: `floating` (6-связность), `budget`, `palette_size`, `symmetry_x` (если `symmetric=True`,
  волосы не учитываются), `layer_order` (волосы прилегают к коже, кость/жир/мышца не торчат наружу),
  `hollow` (замкнутые полости).
- `render.render_views(model_or_part, views, size) -> {name: PIL.Image}`, `render.contact_sheet(...)`,
  `render.tint_state(rgb, state)` — та же тонировка состояний, что в мешере voxbody.
- `io_vxm.write/read`, `io_vox.write/read/import_vox`.
- Импорт мешей: `objmesh.load_obj` (OBJ по группам), `meshvox` (`VoxelGrid`, `voxelize_surface`, `fill_interior`,
  `voxelize_solid`, `voxelize_offset_shell`, `close_holes`, трансформы), `mh.load()` (меш MakeHuman в пространстве Godot,
  хелперы, суставы, `landmarks()`), `head_from_mesh.build_head(voxel_size, HeadParams())` — голова из меша.

Материалы — `voxgen/materials.py` (те же id, что в `docs/spec/vxm-format.md` и `voxel_volume.h`).

## Спеки в репозитории

- `specs/_human_head.py` — параметрическая голова `build_head(voxel_size_m, dims_hint=None, variant="default")`:
  слои кожа/жир/мышца, череп + челюсть, зубы, хрящи ушей и носа, глаза, сосуды, волосы, щетина, брови.
  Варианты: `default`, `bald`, `shaved`.
- `specs/head_mh.py` — голова из меша MakeHuman hm08 (`voxgen.head_from_mesh.build_head(0.005, HeadParams())`,
  меш в `assets/makehuman/base.obj`, CC0); `specs/calibration/head_mh_3mm.py` — та же при 3 мм.
- `specs/head_test.py` — голова 5 мм, которую грузит лабораторная сцена Godot (`godot/assets/models/head_test.vxm`);
  с 2026-10-16 это голова из меша (как `head_mh`), SDF-голова осталась в `_human_head.py` и `calibration/head_{3,5,8}mm.py`.
- `specs/calibration/head_{3,5,8}mm.py` — та же голова для сравнения размера вокселя.
- `specs/props/{stool,mug,razor}.py` — табурет (15 мм), кружка (10 мм), бритва (4 мм).

## Соглашения рендера

Ортографические виды. `front` — как фотография: камера перед лицом (+X персонажа слева на картинке),
`left` — камера со стороны −X (левое ухо), `top` — лицо вверху картинки. Освещение привязано к камере
(тёплый ключевой + холодный заполняющий + ambient), AO по 8 соседям вокруг видимой грани, 1px контур на скачках глубины,
лёгкое затемнение по глубине.
