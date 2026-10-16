---
name: godot-lab
description: Use when working inside the Godot project godot/ — the lab scene, GDScript tools and camera, the voxel shader/material, headless tests or the xvfb screenshot; also when Godot reports missing VoxelPart/VoxelModelNode classes, when a scene must be checked without the editor, or when adding a new barber tool to tools.gd.
---

# godot-lab — лабораторный проект Godot 4.7

Проект: `godot/` (`project.godot`: main scene `res://scenes/lab.tscn`, Forward+, Jolt, фичи `4.7`).
Подробности управления — `godot/README.md`; API расширения — `docs/spec/voxbody-api.md`.

## Карта сцены и файлов

```
Lab (Node3D, scripts/lab.gd — VoxelLab: HUD, R новая голова, F каркас, F12 скриншот в user://screenshots/)
├─ WorldEnvironment (AgX, SSAO)        ├─ CandleLight (OmniLight3D, тёплый)   ├─ RimLight (DirectionalLight3D, холодный)
├─ Camera (scripts/orbit_camera.gd — OrbitCamera: ПКМ орбита, колесо зум, СКМ панорама; target_node = Head)
├─ Head (VoxelModelNode, model_path = res://assets/models/head_test.vxm, стоит на y = 0.05)
├─ Stool (VoxelModelNode; lab.gd грузит stool.vxm, если он есть)   ├─ Debris (Node3D — родитель обломков)
├─ Ground (MeshInstance3D) + GroundBody (StaticBody3D)
├─ Tools (scripts/tools.gd — VoxelToolController: бритва/дубина/свеча/ножницы/ложка, рейкаст по частям, обломки)
└─ HUD (CanvasLayer: InfoLabel, MissingLabel «Нет модели…»)
```
- `scripts/voxel_debris.gd` — `VoxelDebris` (RigidBody3D + VoxelPart) для отвалившихся кусков.
- `addons/voxbody/`: `voxbody.gdextension`, `bin/` (не в git), `voxel.gdshader`, `voxel_material.tres`.
- `tests/run_tests.gd` (раннер, `extends SceneTree`), `tests/voxbody_tests.gd` (кейсы), `tests/screenshot.gd`.
- Слои физики: 1 world, 2 debris, 3 body, 4 tools. Input-действия: `tool_razor..tool_scoop` (1–5), `reload_model` (R),
  `screenshot` (F12), `toggle_wireframe` (F).

## Команды

```
godot --headless --audio-driver Dummy --path godot --import                      # импорт + кэш классов (после новых файлов)
godot --headless --audio-driver Dummy --path godot -s res://tests/run_tests.gd   # headless-тесты, код 1 при провале
godot --headless --audio-driver Dummy --path godot --check-only --script res://scripts/tools.gd   # парсинг одного скрипта
godot --headless --audio-driver Dummy --path godot --quit                        # прогнать _ready главной сцены без экрана
make screenshot        # = scripts/screenshot.sh build/lab.png: xvfb-run -a godot --rendering-driver opengl3 … -s res://tests/screenshot.gd
make test              # pytest + --import + run_tests.gd
```
`--audio-driver Dummy` убирает ошибки ALSA. Скриншот идёт через Compatibility (opengl3, llvmpipe): свет и AO
приблизительны, но шейдер обязан компилироваться и там; модели должны быть собраны (`make assets`).
Редактор на машине агента недоступен — всё через headless; владелец открывает `godot/` сам.

## Как добавить инструмент (`scripts/tools.gd`)

1. `enum Tool { …, NEW }`; строки в `TOOL_NAMES` (русское имя для HUD), `TOOL_KEYS`, `TOOL_ACTIONS`
   (`"tool_new": Tool.NEW`); `@export_group("New")` с параметрами в вокселях (`*_vox`) и секундах (`*_s`).
2. `func _apply_new(part: VoxelPart, hit: Dictionary, vs: float, …) -> int` — вызывает `part.cut / carve_sphere_local /
   bruise / burn / trim_hair` в **метрах** (радиус = `radius_vox * vs`), возвращает число изменённых вокселей.
3. Ветка `Tool.NEW:` в `match tool` внутри `_apply_tool`; `removes_voxels = true`, если удаляет воксели
   (тогда отвалившееся станет `VoxelDebris`). `hit` содержит `local_position`, `local_normal`, `voxel`, `material`.
4. `project.godot` `[input]`: действие `tool_new` (`physical_keycode` для «6» = 54).
5. Подсказка в `lab.gd::_refresh_hud`, таблица в `godot/README.md`; при новой операции ядра — сначала спека и
   `voxbody-dev`. Проверка: `--check-only`, `--import`, тесты, `make screenshot`.

## Соглашения

- GDScript со **статической типизацией** (`untyped_declaration` — предупреждение в project.godot): `var x: int`,
  `-> void`, `for part: VoxelPart in …`, типизированные словари/массивы где можно. Табы (`.editorconfig`).
- `class_name` у переиспользуемых скриптов, `##`-докстринг в шапке с решениями «где задание молчит», константы
  вместо магических чисел, `@export` с единицами в имени, `@onready` для узлов, сигналы вверх / вызовы вниз.
- Сцены — текстовые `.tscn`, имена узлов PascalCase, `.uid`-файлы коммитятся. Один скрипт — одна ответственность
  (lab = HUD/реакция, tools = инструменты, camera = камера, debris = обломки).
- Параметры инструментов — в вокселях и секундах, пересчёт в метры через `part.volume.voxel_size`.
- Ничего не хранить в `godot/assets/models/` руками — только выход voxgen.

## Godot 4.7: особенности

- **Jolt** — физика по умолчанию проекта (`3d/physics_engine="Jolt Physics"`); RigidBody обломков, слои/маски как выше.
- **Forward+** в игре (AgX, SSAO, MSAA ×2); Compatibility только для скриншота — в шейдере только GLSL ES 3.0:
  без `textureQueryLod`, SSS, `uint`-хэши ок. Шейдер объявляет `voxel_size` uniform — `VoxelPart` выставляет его.
- `RenderingServer.set_debug_generate_wireframes(true)` нужно вызвать до создания мешей (lab.gd делает это в `_init`).
- Скрипт `extends SceneTree` (тесты): узлы добавлять к `root` после первого `process_frame`, иначе нет `_ready`.
- `OS.get_cmdline_user_args()` — аргументы после `--` (так screenshot.gd получает путь).

## Если классов расширения нет

Симптомы: «Could not find type VoxelPart», узлы `VoxelModelNode` стали placeholder-ами, `run_tests.gd` печатает
«the voxbody GDExtension is not loaded». Причина — нет библиотеки в `godot/addons/voxbody/bin/` (не в git) или она старее
заголовков. Лечение: `pgrep -fa scons` (не мешать фоновой сборке хука, лог `build/scons.log`) → `make ext` →
`--import` ещё раз → тесты. `godot/.godot/extension_list.cfg` должен содержать `res://addons/voxbody/voxbody.gdextension`.
Если `.so` есть, но не грузится — имя файла ↔ `[libraries]` в `.gdextension` (`libvoxbody.linux.template_debug.x86_64.so`).
