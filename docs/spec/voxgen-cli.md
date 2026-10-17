# voxgen — генератор вокселей (Python) — контракт CLI и структуры

Пакет: `tools/voxgen/voxgen/`. Запуск из корня репозитория: `python3 -m voxgen ...` после
`pip install -e tools/voxgen` (или `PYTHONPATH=tools/voxgen`). Зависимости: numpy, pillow. Никакого scipy.

## Команды

```
voxgen build <spec.py> [--out godot/assets/models] [--preview build/previews] [--vox build/vox]
                       [--report build/reports] [--no-check] [--size 256] [--sdf]
    Импортирует спек-модуль, вызывает build() -> Model, пишет <name>.vxm, <name>.vox (+ .vox.json),
    контактный лист превью <name>.png (--size — пикселей на вид), отчёт проверок <report>/<name>.json.
    Код возврата 1, если проверки нашли ошибки уровня error. `--sdf` — частям без поля `sdf` добавить
    его из бинарного объёма (`sdfield.distance_field`: точный EDT, знак, размытие 3³ у поверхности).
voxgen sdf <file.vxm> [--out <file.vxm>] [--force] [--no-blur]
    Добавить поле `sdf` (int8, docs/spec/vxm-format.md) всем частям модели, у которых его нет
    (`--force` — пересчитать и имеющиеся); по умолчанию перезаписывает входной файл.
voxgen build-all [specs-dir]           все specs/**/*.py, кроме начинающихся с "_" (те же опции)
voxgen preview <file.vxm|.vox|spec.py> [--out x.png] [--size 256] [--part NAME]
               [--views front,back,left,right,top,bottom,iso,iso2,slice_x,slice_y,slice_z]
    slice_* — срез материалов через центр части (slice_* рисует первую часть модели); один вид
    пишется как есть, несколько — контактным листом.
voxgen check <file.vxm>                 печатает отчёт проверок (json) и возвращает 1 при ошибках
voxgen import-vox <file.vox> --spec-json <file.vox.json> --out <file.vxm>
voxgen info <file.vxm>                  части, размеры, число вокселей, материалы
voxgen render <model.vxm|spec.py> [--styles cubes,bevel,facet,facet_all,smooth,clay,toon,mix]
              [--light studio|candle] [--views three_quarter,front,profile,back,full] [--size 640] [--samples 96]
              [--bg dark|grey|light] [--part-styles head=smooth,hair=facet] [--smooth-passes 2]
              [--smooth-clamp cell|free] [--smooth-inflate 0] [--no-hermite] [--hermite-passes 0]
              [--out build/renders/<name>/] [--cubic-materials 5,8] [--jitter 0] [--facet-step 1] [--save-blend]
    Красивые рендеры в Blender (bpy как python-модуль, venv из scripts/install_blender.sh:
    $VOXENGINE_BPY или ~/.cache/voxengine/venv-bpy; без него — понятная ошибка с командой установки).
    Для спека .py сначала собирает .vxm в godot/assets/models (без проверок). На каждую пару
    стиль×вид запускает подпроцесс venv-python `voxgen.blender.render_styles`; пишет
    <style>_<light>_<view>.png + .json (движок, сэмплы, секунды, грани) + .log, затем sheet.png
    с подписями; печатает время. Все части модели — в одной сцене, каждая в своём origin;
    плоскость пола на y = min(0, низ модели), поэтому пропсы на y=0 стоят на полу.
    Стили: cubes (culled quads + AO), bevel (кубы + Bevel), facet (кубы, у которых одновоксельные
    ступеньки скошены под 45°: `mesh.step_edge_weights`, правило в докстринге модуля, `min_run=2` —
    одиночный бугорок остаётся кубом; один плоский материал на цвет палитры, чтобы фаски не смешивали
    цвета; `--jitter` по умолчанию 0; `--facet-step N` — ступенькой считается уступ глубиной N вокселей,
    фаска 0.5·N воксела: 2 для моделей, нарисованных шагом 1/2 U на сетке 1/4 U), facet_all (скошено каждое ребро 90°), smooth (surface nets,
    цвет по вокселю), clay (smooth + подповерхностное + «отпечатки»), toon (EEVEE, 3-ступенчатая рампа
    + Freestyle); фон `--bg`: dark, grey, light (тёплый светло-серый мир и пол, экспозиция −0.7);
    в smooth-стилях волосы и зубы по умолчанию — часть гладкой поверхности; `--cubic-materials 5,8`
    (id или имена материалов: tooth,hair) оставляет их кубиками. Стиль `mix` — по частям: часть, у
    которой большинство открытых граней — кожа (skin + cartilage), рендерится как smooth (surface nets,
    shade smooth, цвет по вокселю на грань, кожа: roughness 0.6, без бампа, слабый SSS 0.1), остальные
    (волосы, глаза, очки, зубы) — facet; `--part-styles head=smooth,hair=facet` переопределяет карту
    часть→стиль для любого стиля (стили частей: cubes,bevel,facet,facet_all,smooth,clay). Гладкие части:
    `--smooth-passes N` (проходов Лапласа, по умолчанию 2), `--smooth-clamp cell|free` (cell — вершины не
    выходят из своей ячейки; free — без зажима, объём восстанавливается общим сдвигом по нормали),
    `--smooth-inflate V` (сдвиг наружу в вокселях, 0..0.5, закрывает щель у кубических волос). Гладкая
    часть с полем `sdf` строится Hermite surface nets: вершины в точном ноль-пересечении поля, нормали —
    его градиент (custom split normals), без проходов Лапласа; `--no-hermite` — бинарный путь и для неё,
    `--hermite-passes N` — лёгкие проходы поверх точных вершин. В .json пишется `smooth.hermite`, подпись
    листа — «nets hermite». Карта частей и параметры nets пишутся в .json и в подпись листа. Код возврата
    1, если хоть один рендер упал.
voxgen render-sheet <dir> [--out <dir>/sheet.png] [--tile 640]   пересобрать контактный лист из рендеров
```

Мешеры `voxgen.mesh` (чистый numpy, без Blender): `culled_quads(part, palette, jitter, ao)`,
`step_edge_weights(mesh, part, mode='steps'|'all', min_run=2, step=1)` (веса фасок для `facet`) и
`surface_nets(part, palette, smooth_passes, keep_cubic_materials=(), clamp='cell'|'free', inflate=0,
hermite=None, hermite_passes=0)` (`hermite=None` — Hermite-режим, если у части есть поле `sdf`) →
`Mesh` (positions float32 (N,3) в
метрах local space части, оси Godot; quads int32 (F,4) CCW снаружи; face_colors uint8 (F,4) rgb+ao;
face_material; corner_ao; face_smooth; vertex_colors для surface nets; vertex_normals и флаг `hermite`
в Hermite-режиме). Поле расстояний: `voxgen.sdfield` — `distance_field(solid, voxel_size)` (метры, точный
EDT без scipy), `quantize(d_vox, solid)` → int8, `update_sdf` (срез/добавление), `add_distance_field(part)`.

## Спек-файл

Python-модуль с функцией `build() -> voxgen.Model`. Пример: `tools/voxgen/specs/head_test.py`.
Спек — исходник истины модели, он коммитится. Выходы генерируются.

## Что должно быть в библиотеке (минимум)

- `Model(name, voxel_size)`, `Model.add_part(Part)`, `Part(name, dims, voxel_size, origin, bone)` с numpy-полями `material`, `color`, `state` (uint8, shape (nz, ny, nx) в numpy или (nx,ny,nz) — главное, чтобы запись в .vxm шла x-fastest; задокументировать выбор); дополнительные именованные поля `Part.extra` (байт на воксель, `part.sdf` — int8-поле расстояний, `paint`/`clear` держат его согласованным с маской).
- Палитра: `Palette` с `add(name, rgb) -> index`, поиск ближайшего, лимит 256, индекс 0 = воздух.
- Материалы: `materials.py` — те же id, что в `docs/spec/vxm-format.md`.
- SDF-примитивы на numpy-сетке (возвращают поле расстояний в вокселях): sphere, ellipsoid, capsule, box (rounded), cylinder, cone, torus; операции union/subtract/intersect/smooth_union; трансформы translate/rotate (Euler)/scale; `mirror_x` (симметрия); шум (value noise 3D) для смещения поверхности.
- Слои тела: `fill_layers(part, sdf, layers=[(skin, 1), (fat, 2), (muscle, ...)], core=(bone, sdf_bone))` — материал по глубине от поверхности (значение SDF), кость по отдельному SDF.
- Волосы: `grow_hair(part, mask_sdf, length_vox, color, direction=(0,1,0), jitter=...)` — столбики MAT_HAIR от кожи наружу (длина варьируется).
- Шум цвета на уровне данных НЕ делаем (его делает шейдер). Но можно `tint_region(part, sdf, color_index)` для румянца.
- Рендер превью: ортографический raymarcher на numpy, 6 сторон + 2 изометрии, ламберт + AO по соседям + контур; палитра применяется; состояние тонируется как в мешере (см. `docs/spec/voxbody-api.md`). Контактный лист 4×2 с подписями видов и инфо (имя, dims, voxel count). Выходной PNG — это «глаза» агента.
- Проверки (`checks.py`): `floating` (компоненты, не связанные с самой большой, 6-связность; error если > N вокселей, warning иначе), `budget` (лимит вокселей на часть), `palette_size`, `symmetry_x` (процент несимметричных вокселей, если часть помечена symmetric=True), `layer_order` (кожа не внутри кости и т.п. — hair только снаружи skin), `hollow` (воздух внутри тела — warning). Отчёт: `{"errors": [...], "warnings": [...], "stats": {...}}`.
- IO: `io_vxm.write(model, path)`, `io_vxm.read(path) -> Model`, `io_vox.write(model, path)`, `io_vox.read(path)`.
- Импорт мешей (`objmesh.py`, `meshvox.py`): `load_obj(path) -> ObjMesh` (вершины, треугольники, имя группы
  на грань, `vt` при наличии; `select(groups)`, `group_centre`, `vertex_normals`, `open_edges`, `transformed(fn)`);
  `VoxelGrid.from_box(lo, hi, vs, centre_x=True)`, `voxelize_surface(mesh, grid)` (выборка каждого треугольника с шагом
  ≤ ½ вокселя), `fill_interior(surface, sealed_faces=("-y",)) -> (solid, info)` (заливка 6-связным потоком от границы
  сетки, запечатанные грани — для среза шеи; `info["leak"]` — дырка в оболочке), `voxelize_solid`, `voxelize_offset_shell`
  (смещение по нормалям — толстая оболочка), `close_holes` (закрыть открытые контуры веером), `clip_faces`, `crop`,
  трансформы `scale`/`rotate_y180`/`flip_z`/`translate` (MakeHuman: дм → м, лицо +Z → −Z).
- MakeHuman (`mh.py`): `load()` — базовый меш hm08 (CC0, `tools/voxgen/assets/makehuman/base.obj`, сеть не нужна)
  в пространстве Godot (метры, лицо в −Z, ступни на y=0); `body()`, `helper("l_eye"|"upper_teeth"|"hair"|…)`,
  `joint(name)`/`joints()` (125 кубиков-суставов — готовый скелет), `landmarks()` (центры глаз, рот по зубам, кончик носа,
  подбородок, уши, макушка, основание шеи, линия бровей, конвенция линии роста волос `eye_y + 0.07`).
- Голова из меша (`head_from_mesh.py`): `build_head(voxel_size, HeadParams(), name) -> Model` — срез по суставу шеи,
  вокселизация + заливка, слои по глубине (BFS от воздуха): кожа ≥ 5 мм, жир ~4 мм (щёки толще), мышца, кость с
  глубины 9 мм над бровями / 16 мм под ними (шея — только задняя половина), зубы и глаза из хелперов (радужка + блик
  на открытой части), хрящ в ушах и носу, волосы — сплошная шапка (скальп меша, выдавленный по нормали на
  `hair_thickness`, тоньше к линии роста), тинты кожи (губы, брови, щетина 60 %, румянец, ноздри), зеркало по X.

## Тесты

`tools/voxgen/tests/` на pytest: roundtrip vxm, roundtrip vox, SDF-примитивы (объём сферы ±), fill_layers порядок слоёв, checks на синтетике, рендер даёт непустое изображение, CLI build на `specs/head_test.py` в tmp. Мешеры: знаковый объём culled quads сферы = число вокселей × voxel_size³; surface nets непустой и замкнутый (каждое ребро у двух квадов).
