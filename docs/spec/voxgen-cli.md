# voxgen — генератор вокселей (Python) — контракт CLI и структуры

Пакет: `tools/voxgen/voxgen/`. Запуск из корня репозитория: `python3 -m voxgen ...` после
`pip install -e tools/voxgen` (или `PYTHONPATH=tools/voxgen`). Зависимости: numpy, pillow. Никакого scipy.

## Команды

```
voxgen build <spec.py> [--out godot/assets/models] [--preview build/previews] [--vox build/vox]
                       [--report build/reports] [--no-check] [--size 256]
    Импортирует спек-модуль, вызывает build() -> Model, пишет <name>.vxm, <name>.vox (+ .vox.json),
    контактный лист превью <name>.png (--size — пикселей на вид), отчёт проверок <report>/<name>.json.
    Код возврата 1, если проверки нашли ошибки уровня error.
voxgen build-all [specs-dir]           все specs/**/*.py, кроме начинающихся с "_" (те же опции)
voxgen preview <file.vxm|.vox|spec.py> [--out x.png] [--size 256] [--part NAME]
               [--views front,back,left,right,top,bottom,iso,iso2,slice_x,slice_y,slice_z]
    slice_* — срез материалов через центр части (slice_* рисует первую часть модели); один вид
    пишется как есть, несколько — контактным листом.
voxgen check <file.vxm>                 печатает отчёт проверок (json) и возвращает 1 при ошибках
voxgen import-vox <file.vox> --spec-json <file.vox.json> --out <file.vxm>
voxgen info <file.vxm>                  части, размеры, число вокселей, материалы
```

## Спек-файл

Python-модуль с функцией `build() -> voxgen.Model`. Пример: `tools/voxgen/specs/head_test.py`.
Спек — исходник истины модели, он коммитится. Выходы генерируются.

## Что должно быть в библиотеке (минимум)

- `Model(name, voxel_size)`, `Model.add_part(Part)`, `Part(name, dims, voxel_size, origin, bone)` с numpy-полями `material`, `color`, `state` (uint8, shape (nz, ny, nx) в numpy или (nx,ny,nz) — главное, чтобы запись в .vxm шла x-fastest; задокументировать выбор).
- Палитра: `Palette` с `add(name, rgb) -> index`, поиск ближайшего, лимит 256, индекс 0 = воздух.
- Материалы: `materials.py` — те же id, что в `docs/spec/vxm-format.md`.
- SDF-примитивы на numpy-сетке (возвращают поле расстояний в вокселях): sphere, ellipsoid, capsule, box (rounded), cylinder, cone, torus; операции union/subtract/intersect/smooth_union; трансформы translate/rotate (Euler)/scale; `mirror_x` (симметрия); шум (value noise 3D) для смещения поверхности.
- Слои тела: `fill_layers(part, sdf, layers=[(skin, 1), (fat, 2), (muscle, ...)], core=(bone, sdf_bone))` — материал по глубине от поверхности (значение SDF), кость по отдельному SDF.
- Волосы: `grow_hair(part, mask_sdf, length_vox, color, direction=(0,1,0), jitter=...)` — столбики MAT_HAIR от кожи наружу (длина варьируется).
- Шум цвета на уровне данных НЕ делаем (его делает шейдер). Но можно `tint_region(part, sdf, color_index)` для румянца.
- Рендер превью: ортографический raymarcher на numpy, 6 сторон + 2 изометрии, ламберт + AO по соседям + контур; палитра применяется; состояние тонируется как в мешере (см. `docs/spec/voxbody-api.md`). Контактный лист 4×2 с подписями видов и инфо (имя, dims, voxel count). Выходной PNG — это «глаза» агента.
- Проверки (`checks.py`): `floating` (компоненты, не связанные с самой большой, 6-связность; error если > N вокселей, warning иначе), `budget` (лимит вокселей на часть), `palette_size`, `symmetry_x` (процент несимметричных вокселей, если часть помечена symmetric=True), `layer_order` (кожа не внутри кости и т.п. — hair только снаружи skin), `hollow` (воздух внутри тела — warning). Отчёт: `{"errors": [...], "warnings": [...], "stats": {...}}`.
- IO: `io_vxm.write(model, path)`, `io_vxm.read(path) -> Model`, `io_vox.write(model, path)`, `io_vox.read(path)`.

## Тесты

`tools/voxgen/tests/` на pytest: roundtrip vxm, roundtrip vox, SDF-примитивы (объём сферы ±), fill_layers порядок слоёв, checks на синтетике, рендер даёт непустое изображение, CLI build на `specs/head_test.py` в tmp.
