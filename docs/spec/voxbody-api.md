# API расширения `voxbody` (GDExtension, C++) — контракт для GDScript

Все классы регистрируются на уровне SCENE. Код: `src/voxbody/src/`. Библиотека: `godot/addons/voxbody/`.

## Пространства координат

- **voxel space** — координаты в вокселях части: (0,0,0) — угол вокселя 0; центр вокселя (x,y,z) = (x+0.5, y+0.5, z+0.5). Все операции `VoxelVolume` работают здесь. Целочисленно-детерминированы.
- **local space** — метры в системе узла `VoxelPart`: `local = origin + voxel * voxel_size`. `VoxelPart` конвертирует и вызывает операции `VoxelVolume`.

## `VoxelVolume` extends Resource

Реализован (см. `voxel_volume.h`). Кратко:

- `create(dims: Vector3i, voxel_size: float)`, `duplicate_volume() -> VoxelVolume`
- `get_dims() -> Vector3i`, `voxel_size`, `origin`, `part_name`, `bone` (свойства)
- `get_material(v: Vector3i) -> int`, `get_color`, `get_state`, `set_voxel(v, material, color)`, `set_state(v, state)`
- `get_material_data() -> PackedByteArray` и `set_*_data` для трёх полей
- `count_solid() -> int`, `get_solid_min() -> Vector3i`, `get_solid_max() -> Vector3i`
- `carve_capsule(a: Vector3, b: Vector3, radius: float, mark_wound := true) -> int` (сколько удалено)
- `carve_sphere(center, radius, mark_wound := true) -> int`
- `bruise_sphere(center, radius, level: int) -> int`, `burn_sphere(center, radius, level) -> int`
- `stamp_state_sphere(center, radius, set_bits, clear_bits) -> int`
- `trim_hair_capsule(a, b, radius, keep_length: int) -> int`
- `raycast(origin: Vector3, dir: Vector3, max_dist: float) -> Dictionary` `{hit, voxel: Vector3i, normal: Vector3i, distance, material, position}`
- `detach_disconnected(anchor: Vector3i) -> Array[VoxelVolume]` — отрезает всё, что не связано с якорем (6-связность), возвращает куски со своим `origin` в том же local space.
- enum `Material` (MAT_SKIN…), enum `State` (STATE_CUT…)

## `VoxelModel` extends Resource

Загрузчик `.vxm` (см. `docs/spec/vxm-format.md`).

- `static load_from_file(path: String) -> VoxelModel` (null + push_error при ошибке). Путь `res://`, `user://` или абсолютный.
- `load_from_buffer(bytes: PackedByteArray) -> bool`
- `save_to_file(path: String) -> bool` (пишет все три поля)
- `model_name: String`, `palette: PackedColorArray` (256 цветов, sRGB, alpha 1), `materials: Dictionary`
- `get_part_count() -> int`, `get_part(i: int) -> VoxelVolume`, `find_part(name: String) -> VoxelVolume` (null если нет), `add_part(v: VoxelVolume)`

## `VoxelMesher` extends RefCounted

- `static build_mesh(volume: VoxelVolume, palette: PackedColorArray, options: Dictionary = {}) -> ArrayMesh`
  - greedy meshing по 6 направлениям, один surface, PRIMITIVE_TRIANGLES, индексы.
  - `ARRAY_VERTEX`: local space (`origin + voxel*voxel_size`), т.е. меш уже стоит на месте; `VoxelPart` ставит MeshInstance3D в identity.
  - `ARRAY_NORMAL`: нормаль грани.
  - `ARRAY_COLOR`: `rgb` = sRGB-альбедо вокселя **после тинта состояния** (см. ниже), `a` = AO-фактор вершины 0.55..1.0 (0.55 = три соседа закрывают, 1.0 = открыто).
  - `ARRAY_TEX_UV`: `(material / 255.0, state / 255.0)` — для шейдера (блеск крови, мокрое).
  - Слияние квадов только при равных (color, material, state, ao0..ao3). Диагональ квада переворачивается при `ao0 + ao3 > ao1 + ao2` (анизотропия AO).
  - options: `ao: bool = true`, `ao_strength: float = 0.45` (1.0 - 0.45 = минимум).
- Тинт состояния (линейная интерполяция базового цвета к целевому с весом w):
  - bruise 1: (217, 89, 89) w=0.45; 2: (115, 64, 140) w=0.55; 3: (153, 166, 77) w=0.45
  - burn 1: (204, 77, 51) w=0.5; 2: (230, 191, 153) w=0.6; 3: (26, 20, 18) w=0.9
  - BLOOD: (115, 8, 8) w=0.65 (поверх остального); CUT без BLOOD: базовый цвет × 0.85
  - WET: без изменения цвета (только шейдер).
- Производительность: 48³ полностью заполненный — не более ~5 мс в release. Перестройка только изменённой части.

## `VoxelPart` extends Node3D

Одна часть на сцене: держит `VoxelVolume`, создаёт внутренний `MeshInstance3D` (internal child), перестраивает меш, когда грязный.

Свойства: `volume: VoxelVolume`, `palette: PackedColorArray`, `material_override: Material` (null → загрузить `res://addons/voxbody/voxel_material.tres`; если и его нет — StandardMaterial3D с vertex color), `auto_rebuild: bool = true` (перестройка в `_process`, иначе вручную `rebuild()`), `ao_enabled: bool = true`.

Методы:
- `rebuild()` — немедленно построить меш.
- `mark_dirty()`
- `get_mesh_instance() -> MeshInstance3D`
- `local_to_voxel(p: Vector3) -> Vector3`, `voxel_to_local(v: Vector3) -> Vector3`
- `cut(a_local: Vector3, b_local: Vector3, radius_m: float) -> int` — capsule-срез в метрах; отмечает рану.
- `carve_sphere_local(center_local, radius_m) -> int`
- `bruise(center_local, radius_m, level) -> int`, `burn(center_local, radius_m, level) -> int`
- `trim_hair(a_local, b_local, radius_m, keep_length_vox: int) -> int`
- `raycast_local(origin_local: Vector3, dir_local: Vector3, max_dist_m: float) -> Dictionary` — как у volume, но `position` и `distance` в метрах, `normal: Vector3`.
- `raycast_global(origin: Vector3, dir: Vector3, max_dist_m) -> Dictionary` — то же, вход/выход в мировых координатах (учитывая `global_transform`).
- `detach_loose(anchor_voxel: Vector3i = Vector3i(-1,-1,-1)) -> Array[VoxelVolume]` — вызывает `detach_disconnected`, помечает грязным, испускает `pieces_detached(pieces)`. Создание узлов для кусков — дело GDScript (`VoxelPart.new()` + RigidBody3D).
- `get_solid_count() -> int`

Сигналы: `mesh_rebuilt`, `pieces_detached(pieces: Array)`.

## `VoxelModelNode` extends Node3D

Собирает `VoxelPart` из `VoxelModel`.

Свойства: `model_path: String` (`res://assets/models/head_test.vxm`), при установке в дереве — загружает. `model: VoxelModel`.

Методы: `load_model(path) -> bool`, `set_model(model)`, `get_part_node(name: String) -> VoxelPart`, `get_part_nodes() -> Array[VoxelPart]`, `rebuild_all()`.

Дочерние `VoxelPart` именуются по `part.name`; `bone` пока не используется (привязка к Skeleton3D — следующая итерация: `VoxelPart` будет переезжать под `BoneAttachment3D`).

## Шейдер `godot/addons/voxbody/voxel.gdshader`

Контракт с мешером:
- `COLOR.rgb` — sRGB-альбедо; перевести в linear (`pow(c, 2.2)` или точная формула).
- `COLOR.a` — AO, умножить на ALBEDO (или `AO = COLOR.a`, `AO_LIGHT_AFFECT = 0.6`).
- `UV.x * 255` = material, `UV.y * 255` = state: `state & 0x20` (кровь) → ROUGHNESS 0.25, `state & 0x40` (мокрое) → ROUGHNESS 0.3; иначе ROUGHNESS из uniform `roughness` (0.85).
- Пер-воксельный шум цвета: в vertex shader передать `VERTEX` (local) во varying; во fragment: `ivec3 cell = ivec3(floor((v_local - 0.0001 * NORMAL_local) / voxel_size))`, хэш → ±`noise_strength` (uniform, по умолчанию 0.06) по яркости и ±0.02 по оттенку. Uniform `voxel_size` (по умолчанию 0.005) выставляет `VoxelPart` при создании материала (дублирует материал per part через `duplicate()` только если voxel_size частей отличается).
- Материал-ресурс `godot/addons/voxbody/voxel_material.tres` (ShaderMaterial на этом шейдере) — то, что `VoxelPart` грузит по умолчанию.
