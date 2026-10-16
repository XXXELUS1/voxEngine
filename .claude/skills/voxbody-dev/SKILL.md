---
name: voxbody-dev
description: Use when working on the C++ GDExtension in src/voxbody (VoxelVolume, VoxelModel, VoxelMesher, VoxelPart, VoxelModelNode) — adding or changing a method, property, signal or material, fixing a voxel operation, building debug/release, running the headless Godot tests, or hunting godot-cpp 10 / Godot 4.7 build errors.
---

# voxbody-dev — работа над C++-расширением

Контракт: `docs/spec/voxbody-api.md` (менять спеку до кода). Формат данных: `docs/spec/vxm-format.md`.

## Карта файлов (`src/voxbody/`)

- `src/register_types.cpp|h` — `GDREGISTER_CLASS` всех пяти классов на уровне SCENE, entry `voxbody_library_init`.
- `src/voxel_volume.h|cpp` — три поля `PackedByteArray` (x-fastest), enum `Material`/`State`, операции
  (`carve_*`, `bruise_sphere`, `burn_sphere`, `stamp_state_sphere`, `trim_hair_capsule`), `raycast` (DDA),
  `detach_disconnected` (flood fill 6-связность). Ядро детерминизма.
- `src/voxel_model.h|cpp` — `.vxm` туда-обратно, палитра 256, `materials`, `meta`, части.
- `src/voxel_mesher.h|cpp` — greedy meshing, AO по вершинам, тинт состояний, `tint_color`, `ao_factor`.
- `src/voxel_part.h|cpp` — Node3D с внутренним `MeshInstance3D`, local ↔ voxel, обёртки операций в метрах, материал.
- `src/voxel_model_node.h|cpp` — собирает `VoxelPart` по частям модели.
- `SConstruct` — `SConscript("godot-cpp/SConstruct", {"api_version": "4.7"})`, выход в `godot/addons/voxbody/bin/`.
- `godot/addons/voxbody/voxbody.gdextension` (`compatibility_minimum 4.7`, `reloadable`), `voxel.gdshader`,
  `voxel_material.tres` — потребители меша; тесты — `godot/tests/voxbody_tests.gd`.

## Сборка

```
pgrep -fa scons                      # хук мог запустить фоновую сборку: дождаться (tail -f build/scons.log)
make ext                             # = scripts/build_ext.sh debug   → libvoxbody.linux.template_debug.x86_64.so
make ext-release                     # = scripts/build_ext.sh release → libvoxbody.linux.template_release.x86_64.so
scripts/build_ext.sh debug arch=arm64        # лишние аргументы уходят в scons
cd src/voxbody && scons -j4 target=template_debug    # напрямую; первая сборка godot-cpp ~10 мин
```
Windows: сам не собираем — CI (`.github/workflows/ci.yml`, job `ext-windows`, MSVC) кладёт `.dll` в артефакт
`voxbody-windows`; владелец копирует их в `godot/addons/voxbody/bin/`. Перед сдачей: собираются **обе** цели.
Ошибка «extension not loaded» в тестах = нет `.so` в `bin/` или он старее заголовков.

## Как добавить метод / свойство / сигнал

1. Спека: строка в `docs/spec/voxbody-api.md` (имя, аргументы, единицы, что возвращает).
2. Заголовок: объявление в `public:`; операции над вокселями — в `VoxelVolume` (voxel space), обёртка в
   метрах — в `VoxelPart` (`local_to_voxel`, радиус / `voxel_size`).
3. `_bind_methods()`:
   ```cpp
   ClassDB::bind_method(D_METHOD("fill_sphere", "center", "radius", "material", "color"), &VoxelVolume::fill_sphere, DEFVAL(1));
   ClassDB::bind_static_method("VoxelMesher", D_METHOD("build_mesh", "volume", "palette", "options"), &VoxelMesher::build_mesh, DEFVAL(Dictionary()));
   ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "voxel_size"), "set_voxel_size", "get_voxel_size");   // setter+getter уже привязаны
   ADD_SIGNAL(MethodInfo("pieces_detached", PropertyInfo(Variant::ARRAY, "pieces")));
   BIND_ENUM_CONSTANT(MAT_NEW);   // + VARIANT_ENUM_CAST(godot::Class::Enum) вне namespace в заголовке
   ```
   Имена в `D_METHOD` — по числу аргументов; `DEFVAL` перечисляются для последних аргументов справа налево.
4. Кейс в `godot/tests/voxbody_tests.gd`, пересборка, тесты, `godot/README.md`/lab при необходимости.

## Godot-cpp 10 / Godot 4.7: грабли

- `api_version "4.7"` в SConstruct обязателен, иначе биндинги генерируются под другую версию и классы 4.7 «не существуют».
- `Ref<T> r; r.instantiate();` для Resource/RefCounted (не `new`, не `memnew` в `Ref`); `memnew(MeshInstance3D)` для узлов,
  `queue_free()`/`memdelete` для удаления; узел-ребёнок, который могут удалить извне — хранить `instance_id` и проверять.
- `PackedByteArray`: `resize(n)` → `ptrw()` для записи, `ptr()` для чтения; указатель невалиден после любого
  `resize`/копии (COW). Возврат по значению дёшев. Для меша — `PackedVector3Array`/`PackedInt32Array` в `Array` по `Mesh::ARRAY_*`.
- **Исключений нет** (`-fno-exceptions`): `ERR_FAIL_COND_V(cond, ret)`, `ERR_FAIL_INDEX_V`, `ERR_FAIL_NULL_V`,
  `UtilityFunctions::push_error/print`. Никаких `throw`, `std::stoi`, `std::regex`.
- `TypedArray<VoxelVolume>` → в GDScript `Array[VoxelVolume]`; `Dictionary` ключи — `Variant` (`String`), чтение `d["hit"]`.
- `String` — UTF-32: в C-строку через `.utf8().get_data()`; `StringName` для имён узлов/методов.
- Виртуалы `_process(double)`, `_ready()`, `_notification(int)` просто переопределяются (без биндинга); если
  `_process` не вызывается — `set_process(true)` в `NOTIFICATION_READY`. В редакторе узел тоже живёт:
  `Engine::get_singleton()->is_editor_hint()` для рантайм-только работы.
- Свойства из `.tscn` восстанавливаются в порядке `ADD_PROPERTY`: размерности раньше данных (как `dims` в `VoxelVolume`).
- `reloadable = true`: редактор перегружает `.so`, но живые объекты пересоздаются — при странностях перезапустить
  редактор; headless-запуски всегда грузят свежую библиотеку.
- Внутренние дети — `add_child(n, false, Node::INTERNAL_MODE_BACK)`, чтобы не ломать `get_child_count()` у GDScript.

## Тесты

```
godot --headless --audio-driver Dummy --path godot --import                     # один раз (и после новых скриптов)
godot --headless --audio-driver Dummy --path godot -s res://tests/run_tests.gd  # код 1 при провале
make test                                                                       # pytest + то же самое
```
Новый кейс: в `godot/tests/voxbody_tests.gd` функция `func test_<name>() -> void:` с `begin("…")` и
`check(cond, "описание")` / `check_approx(a, b, "…")`, затем добавить её в список в `run()`. Помощники:
`filled_volume(size, material, color)`, `fill_box(...)`, `test_palette()`; узлы — через `_root.add_child`, потом `free()`.
Кейсы на `head_test.vxm` пропускаются, если модель не собрана (`make assets`). Тест на детерминизм (ADR-0005):
одна операция на `duplicate_volume()` → `get_material_data() == …` и `get_state_data() == …`.

## Правила

- **Детерминизм** (ADR-0005): операции — чистые функции от полей и аргументов, целочисленные индексы, сравнения через
  квадраты расстояний, без RNG/времени/статики; случайность — только через `seed`-аргумент. Никаких `float`
  из физики внутри операций: квантование — на стороне GDScript перед событием. Не включать `-ffast-math`.
- **Производительность**: перестраивать только грязную часть (`mark_dirty` → один `rebuild`); циклы по
  ограничивающему боксу операции, не по всему объёму; `std::vector::reserve`; никаких Variant-вызовов в горячем
  цикле; бюджет 48³ ≤ ~5 мс в release — мерить `Time.get_ticks_usec()` в лаборатории (HUD показывает мс операции).
- **Материалы синхронно**: `voxel_volume.h` enum ↔ `tools/voxgen/voxgen/materials.py` ↔ таблица в
  `docs/spec/vxm-format.md`; `material_bleeds`/`material_bruises` ↔ `BLEEDS`/`BRUISES`; тинт состояний ↔ `render.tint_state`.
- Стиль Godot: табы, `snake_case`, `p_`-параметры в объявлениях, комментарии на английском, решения «где спека
  молчит» — в шапке заголовка.
