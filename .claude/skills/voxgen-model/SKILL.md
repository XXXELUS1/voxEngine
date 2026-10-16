---
name: voxgen-model
description: Use when creating or changing a voxel model (a head, a body part, a prop, a hair style, a calibration or proportion variant) in tools/voxgen/specs — writing or editing the spec, building it with voxgen, looking at the preview PNG, running the checks, exporting .vox for the owner or re-importing his MagicaVoxel edits.
---

# voxgen-model — делаем и правим модели

Контракт: `docs/spec/voxgen-cli.md`, формат: `docs/spec/vxm-format.md`, библиотека: `tools/voxgen/README.md`.
Запуск из корня репозитория; voxgen уже установлен хуком (`pip install -e tools/voxgen`), иначе `PYTHONPATH=tools/voxgen`.

## Порядок работы

1. **Стиль-гайд**: `docs/style-guide.md` — разделы 2 (размеры), 3 (волосы), 7 (палитра), 9 («наше»), 11 (журнал).
   Замечание владельца → сначала строка в «Журнал правок», потом правка.
2. **Спек**: `tools/voxgen/specs/<name>.py` с `build() -> Model`. Пропсы — `specs/props/`, калибровка —
   `specs/calibration/`, общие модули — с префиксом `_` (`build-all` их пропускает). Голова — через параметры
   `build_head(voxel_size_m, dims_hint, variant, name)` в `specs/_human_head.py`, а не копией файла.
3. **Сборка**: `python3 -m voxgen build tools/voxgen/specs/<name>.py` → `godot/assets/models/<name>.vxm`,
   `build/previews/<name>.png`, `build/vox/<name>.vox` + `.vox.json`, `build/reports/<name>.json`. Код 1 = ошибки.
4. **Смотреть**: `Read build/previews/<name>.png` (контактный лист 4×2: front/back/left/right/top/bottom/iso/iso2
   + инфо-полоса). Слои: `python3 -m voxgen preview godot/assets/models/<name>.vxm --views front,slice_x,slice_z
   --out build/previews/<name>_slices.png`. Крупнее: `--size 512`; одна часть: `--part head`.
5. **Оценить** по скиллу `review-model-preview` — по пунктам, письменно, до отчёта.
6. **Проверки**: `python3 -m voxgen check <vxm>` (json: errors/warnings/stats), `python3 -m voxgen info <vxm>`.
   Предупреждения тоже устраняем (одиночные висящие воксели, exposed fat).
7. **Итерация**: перед пересборкой скопировать старое превью в scratchpad, чтобы сравнить «до/после».
8. **Финал**: `make assets` (все спеки собираются), `cd tools/voxgen && python3 -m pytest -q`;
   если модель грузится в лабораторию (`head_test`, `stool`) — `make screenshot` и посмотреть `build/lab.png`.

## Шпаргалка API (`from voxgen import Model, Part, Palette, sdf, body`; `from voxgen.materials import *`)

- Всё в **метрах** в local space части (Y вверх, лицо в −Z, центр по x/z). SDF-поля — в **вокселях**
  (`field <= 0` — внутри); толщины в вокселях: `part.m2v(0.005)`.
- `Palette()`: `idx = pal.add("skin", (224,180,150))` — имя = имя материала, тогда `fill_layers` берёт его сам;
  оттенки `skin_dark`, `skin_ruddy`. Индекс 0 — воздух. ≤ 256, обычно 12–24.
- `Model(name, voxel_size, palette, meta)`; `m.add_part(Part)`.
- `Part(name, dims, voxel_size, origin, bone, symmetric, budget)`; удобнее `Part.from_extent(name, lo, hi, vs,
  bone=, symmetric=True)` (origin = lo) и `Part.centered(name, size, vs)` (пропс на y=0, центр x/z).
  Поля `material/color/state` — uint8 `(nz, ny, nx)`, индекс `[z, y, x]`. Симметрия точна только если сетка
  x-центрирована (`origin.x == -nx*vs/2`).
- Сетки: `g = part` (обычная), `gm = sdf.mirror_x(part)` — парные детали рисуем на +X, получаем обе стороны;
  `sdf.translate/rotate(grid, euler_deg, pivot)/scale`.
- Примитивы: `sphere(g, c, r)`, `ellipsoid(g, c, radii)`, `capsule(g, a, b, r)`, `box(g, c, size, round=)`,
  `cylinder(g, a, b, r)`, `cone(g, a, b, ra, rb)`, `torus(g, c, R, r, axis)`, `half_space(g, point, normal)`.
- Операции: `union, intersect, subtract, smooth_union(a, b, k_vox), smooth_subtract, offset, shell, displace`;
  шум формы `field + sdf.noise(gm, scale=0.03, amplitude=0.002, seed=1)`.
- Ткани: `body.fill_layers(part, skin_sdf, [(MAT_SKIN, 1), (MAT_FAT, 1), (MAT_MUSCLE, None)],
  core=[(MAT_BONE, skull_sdf), (MAT_CARTILAGE, ear_sdf)])` — слои снаружи внутрь, последний `None` = остальное;
  `protect_outer` не даёт кости вылезти; `seal` закрывает щели кожей.
- Краска: `part.paint(sdf, MAT_METAL, idx, only=[0])` (`only` — писать только поверх этих материалов);
  `body.tint_region(part, sdf, "skin_ruddy")` — перекрасить кожу (румянец), материал не меняется.
- Волосы: `body.grow_hair(part, mask_sdf, (3, 8), HAIR, direction="normal", sweep=(0,-0.3,0.4), jitter=0.3,
  density=1.0, seed=0)` — столбики `MAT_HAIR` от открытой кожи наружу; щетина: длина 1, `density≈0.6`.
- Проверки: `checks.run(model)` — `floating`, `budget`, `palette_size`, `symmetry_x`, `layer_order`, `hollow`.
- IO: `io_vxm.write/read`, `io_vox.write/read/import_vox`; рендер: `render.contact_sheet(model, size)`.

## Рендер в Blender (красивые картинки для владельца)

- Один раз: `scripts/install_blender.sh` (venv `~/.cache/voxengine/venv-bpy`, python3.11 + bpy 4.5 LTS, ~300 МБ).
- `python3 -m voxgen render godot/assets/models/<name>.vxm --styles cubes,bevel,smooth,clay,toon
  --light studio|candle --views three_quarter,front,profile,full --size 640 --samples 96` (или
  `scripts/blender_render.sh ...`) → `build/renders/<name>/<style>_<light>_<view>.png` + `sheet.png`.
  Можно дать спек `.py` — соберётся сам. Cycles CPU 640 px / 96 spl ≈ 35–50 с на стиль; для быстрой
  проверки `--size 384 --samples 32`. Смотреть `sheet.png` (Read) — подписи: стиль, вид/свет, движок, секунды.
- Пересобрать лист после ручной чистки папки: `python3 -m voxgen render-sheet build/renders/<name>`.
- Стили — только для оценки вида (стиль-гайд, раздел 8), в данные модели шум/сглаживание не пишем.

## Экспорт для владельца и обратный импорт правок из MagicaVoxel

- `build` уже пишет `build/vox/<name>.vox` (формат 150, `nTRN/nSHP` на часть) и `<name>.vox.json`
  (палитра → материал, origin, voxel_size). Отдать владельцу оба файла (или артефакт CI).
- Обратно: `python3 -m voxgen import-vox <edited>.vox --spec-json <name>.vox.json --out /tmp/x.vxm`,
  затем `voxgen preview` и `voxgen check` на результате. В `.vox` хранится только цвет — материал
  восстанавливается по `.vox.json`; новые цвета владельца получат ближайший материал/цвет палитры — проверить `info`.
- Чтобы правка не потерялась при следующей генерации, она должна стать источником истины: кладём
  `tools/voxgen/specs/vox/<name>.vox` + `.vox.json` в git и делаем спек-обёртку
  `def build(): return io_vox.import_vox(Path(__file__).parent / "vox" / "<name>.vox")`. Решение из ADR-0004.
- Гибрид «SDF + ручная правка» не делаем: либо спек, либо `.vox`; иначе непонятно, что истина.

## Частые ошибки

- **Висящие воксели**: тонкие детали (кончик носа, мочка) при малом `k` отрываются; увеличить радиус/`k`,
  добавить `seal`, проверить `floating` в отчёте. Крошки на силуэте видно в превью — убирать.
- **Волосы внутри кожи / не на коже**: `grow_hair` только после `fill_layers`; маска должна захватывать
  открытую кожу; `layer_order` ругается — это ошибка спека, а не проверки.
- **Модель висит в воздухе в Godot**: неверный `origin`. Персонаж — подбородок у y≈0 (`HEAD_LO.y` ≈ −0.04),
  пропс — `Part.centered` (стоит на y=0). Лаборатория ставит `Head` на y=0.05.
- **Забыли `mirror_x`**: глаза/уши только с одной стороны или `symmetry_x` > 0.5 %. Парные детали — на `gm`.
- **Ось Z перепутана**: лицо должно смотреть в −Z; в превью `front` нос виден, `left` — левое ухо.
- **Запас сетки**: над макушкой оставить место под волосы (`length_vox` max), иначе столбики обрежутся границей.
- **Цвет 0 у твёрдого вокселя** (`palette_size` error): использовать индексы из `pal.add`, не литералы.
- **Шум цвета в данных**: не делаем — это шейдер (стиль-гайд, раздел 8).
- **Бюджет**: голова 5 мм ≈ 35 000 твёрдых; 3 мм ≈ 160 000; лимит `part.budget` (250 000 / 400 000).
