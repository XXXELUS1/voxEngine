# Стенд стилей

Одни данные (воксели) — разные рендеры. Владелец выбирает стиль глазами, после этого мешер и материал
становятся «нашими», остальные остаются как опции. Шарики не делаем.

## Стили

| id | меш | материал/шейдер | суть |
|---|---|---|---|
| `cubes` | greedy | `voxel.gdshader` (v1) | честные кубики, AO, пер-воксельный шум цвета |
| `bevel` | greedy | `voxel_bevel.gdshader` | те же кубики, но нормаль в полосе `bevel` (0.05–0.3 вокселя) у ребра плавно наклоняется к соседней грани → скруглённые рёбра без геометрии; вогнутые рёбра — лёгкая тёмная складка |
| `smooth` | surface nets | `voxel_smooth.gdshader` | гладкая поверхность поверх вокселей; нормали из сглаженного меша; цвет и AO из вокселей; порез вскрывает «кубики внутри» (внутренние грани раны можно показывать greedy-мешем раны — опция `cut_faces_cubic`) |
| `clay` | surface nets | `voxel_clay.gdshader` | пластилин: wrap-диффуз (SSS-подобно, `wrap 0.4`, тёплый подтон в полутени), roughness 0.55, specular 0.3, процедурные «отпечатки пальцев» — шумовая нормаль частотой ~2 вокселя, амплитуда 0.15; слегка приглушённая палитра |
| `toon` | любой (параметр `toon_base`) | `voxel_toon.gdshader` + контур | 3-ступенчатая рампа света, контур инвертированной оболочкой (второй MeshInstance, рост по нормали 0.6 вокселя, чёрный, cull front) |
| `raymarch` | box per chunk | `voxel_raymarch.gdshader` + `Texture3D` | DDA по 3D-текстуре в фрагментном шейдере, uniform `smoothing 0..1` (0 — кубики, 1 — трилинейная поверхность); повреждение = обновление региона текстуры, без ремеша. **Опционально**, делается последним; на llvmpipe не проверяем |

Все шейдеры: контракт вершинных данных как в `docs/spec/voxbody-api.md` (COLOR.rgb sRGB, COLOR.a AO,
UV = material/state), компилируются на forward_plus и gl_compatibility. Материалы лежат в
`godot/addons/voxbody/styles/<id>.tres`; `VoxelSkinnedBody.style`/`VoxelPart.style` подставляют нужный.

## Surface nets (`VoxelMesher`, mode SURFACE_NETS)

- Ячейки между центрами вокселей; ячейка «активная», если среди 8 углов есть и solid, и air.
- Вершина — среднее позиций средних точек активных рёбер (naive surface nets), затем 2 итерации
  лапласовского сглаживания с ограничением в пределах ячейки (без схлопывания тонких деталей: волос-столбиков,
  зубов — для вокселей `MAT_HAIR`/`MAT_TOOTH` сглаживание выключено, они остаются «кубиками»).
- Квад на каждое активное ребро (4 ячейки вокруг), ориентация по знаку перехода.
- Цвет/материал/состояние вершины: от solid-угла, ближайшего к вершине; при равенстве — большинство.
- Нормали: взвешенные по площади нормали треугольников (после сглаживания).
- Веса скиннинга: от того же solid-угла.
- Граница чанка: ячейки на границе дублируются в обоих чанках (перекрытие 1), координаты вершин считаются из
  одних данных → швов нет.

## Скриншоты стенда (Godot)

`godot/scenes/style_bench.tscn` + `godot/tests/style_shots.gd`: загружает модель (аргумент `--model`,
по умолчанию `res://assets/models/human_ref.vxm`, запасной вариант `head_test.vxm`), для каждого стиля из
`--styles` выставляет материал/мешер, ставит камеру по `header.landmarks` (лицо крупно и в полный рост),
снимает кадр и пишет `build/styles/<model>/<style>_<view>.png`; затем `voxgen sheet` собирает контактный лист
`build/styles/<model>/sheet.png`. Освещение: как в лаборатории (тёплая свеча + холодный контровой) и второй
пресет «день» (`--light day`).

## Blender (красивые рендеры)

- Окружение: `scripts/install_blender.sh` → venv `~/.cache/voxengine/venv-bpy` (python3.11, `bpy==4.5.14`,
  + `pip install -e tools/voxgen`). Проверено: Cycles CPU 512 px / 64 сэмплов ≈ 14 с на 4 ядрах, EEVEE работает без GPU.
- `scripts/blender_render.sh <model.vxm> [--styles cubes,bevel,smooth,clay,toon] [--size 768] [--samples 96] [--views face,full]`
  → `tools/voxgen/voxgen/blender/render_styles.py` (запуск `venv/bin/python`): строит меш через Python-мешер
  voxgen (greedy / surface nets, см. ниже), вершинные цвета → Principled; стили: `cubes` (flat), `bevel`
  (модификатор Bevel 0.25 вокселя, 2 сегмента), `smooth` (surface nets + Smooth by Angle), `clay` (SSS 0.3,
  roughness 0.55, нормаль-шум), `toon` (Shader-to-RGB рампа + Freestyle контур). Студия: тёплый ключевой,
  холодный контровой, мягкий заполняющий, серый градиентный мир, AgX. Выход `build/renders/<model>/<style>_<view>.png` + контактный лист.

## Python-мешеры (`tools/voxgen/voxgen/mesh.py`)

`greedy_mesh(part) -> Mesh`, `surface_nets(part, smooth_iters=2) -> Mesh`; `Mesh` = numpy-массивы
`positions (N,3) float32 (метры, rest/local space), normals, colors (N,4) uint8 (rgb + AO), uv (материал, состояние),
bones (N,2) uint8, weights (N,1) uint8, indices (M,3) uint32`. Та же логика, что в C++ (сверять тестом: для
одного объёма число вершин/индексов greedy-меша Python == C++).

## Веб-просмотрщик

`voxgen web <model.vxm> [--out build/web/<name>.html]` → один самодостаточный HTML: three.js (cdnjs, версия
зафиксирована), OrbitControls, меши `greedy` и `surface_nets` встроены как base64 (positions/normals/colors/
indices), переключатели стилей (`cubes` flat + AO, `smooth`, `clay` (MeshPhysicalMaterial sheen/roughness),
`toon` (MeshToonMaterial + outline)), экспозиция, пресеты света «свеча/день», кнопка «срез» (clipping plane по X).
Лимит 16 МБ. Публикуется как артефакт claude.ai по запросу владельца.
