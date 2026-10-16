# Рантайм v2: `VoxelSkinnedBody`, чанки, скиннинг, попадания

Цель: загрузить `.vxm` v2 с частями `space: "rest"`, показать тело, которое анимируется скелетом Godot
(ретаргет Mixamo, рэгдолл на капсулах), при этом бритва/клещи/удар режут и красят те же воксели.

## Узлы (C++, `src/voxbody`)

### `VoxelSkinnedBody` extends Node3D

Создаёт и держит:

```
VoxelSkinnedBody
├── Skeleton3D "Skeleton"            ← кости из header.skeleton, rest из rest_pos/rest_rot
│   ├── PhysicalBone3D "<bone>"      ← капсула из bone.physics (слой "body_hit"), по одной на кость с physics
│   └── ...
├── VoxelChunkMesh "<part>/<cx>_<cy>_<cz>"  (MeshInstance3D со skeleton и Skin)   ← один на чанк части
└── (части "bone:<Name>" — как v1: VoxelPart под BoneAttachment3D "<Name>")
```

Свойства: `model_path`, `model: VoxelModel`, `mesh_mode: int` (GREEDY | SURFACE_NETS), `chunk_size: int = 32`,
`material_override`, `style: StringName` (имя стиля из `styles.md`, выбирает материал по таблице `res://addons/voxbody/styles/<style>.tres`).

Методы:
- `load_model(path) -> bool`, `get_skeleton() -> Skeleton3D`, `get_part_volume(name) -> VoxelVolume`,
  `rebuild_all()`, `rebuild_dirty()`.
- `raycast_global(origin, dir, max_dist) -> Dictionary` `{hit, part, voxel: Vector3i, bone: int, position (global), normal (global), material}` — см. «Попадания».
- `cut_global(a, b, radius_m) -> int`, `carve_sphere_global(c, r) -> int`, `bruise_global(c, r, level) -> int`,
  `burn_global(c, r, level) -> int`, `trim_hair_global(a, b, r, keep) -> int` — вход в мировых координатах;
  узел переводит в пространство покоя (см. ниже), вызывает `VoxelVolume` часть(ей), помечает грязные чанки.
- `detach_loose() -> Array[Dictionary]` `[{volume: VoxelVolume, transform: Transform3D (global)}]`, сигнал `pieces_detached(pieces)`.
- `set_ragdoll(enabled: bool)` — `physical_bones_start_simulation()` / stop.

### `VoxelChunkMesh` extends MeshInstance3D

Внутренний; хранит `part`, `region: AABB (voxels)`, `dirty`. `skin` — общий `Skin` на модель (bind = инверсия глобального rest каждой кости).

## Чанки и мешер

- Часть делится на чанки `chunk_size³` по вокселям (последний ряд — остаток). Один `ArrayMesh` на чанк.
- `VoxelMesher.build_mesh(volume, palette, options)` получает `options.region = {min: Vector3i, size: Vector3i}`;
  соседей для AO/граней читает из **всего** объёма (чанк — только диапазон выдаваемых граней/ячеек).
- Если у объёма есть поля весов, мешер пишет `ARRAY_BONES` (4 × int, лишние = 0) и `ARRAY_WEIGHTS` (4 × float):
  для greedy — веса вокселя, которому принадлежит грань (квад сливается только при равных `bone0,bone1,w0`;
  это дробит квады на суставах — приемлемо); для surface nets — веса solid-угла с наибольшим вкладом в вершину.
- Шов между частями разного разрешения (голова 5 мм / тело 10 мм): грань части не выдаётся, если центр
  соседнего «воздушного» вокселя лежит внутри solid-вокселя другой `rest`-части той же модели
  (проверка по второй части через `local → voxel` с её `voxel_size`). Так шея не светит внутренними гранями.
- Перестройка: `rebuild_dirty()` в `_process` (как `auto_rebuild` в v1). Операция повреждения помечает только
  чанки, пересекающие её AABB (+1 воксель для AO).

## Пространство покоя ↔ мир

Скиннинг на GPU деформирует вершины; воксели живут в rest-пространстве. Для попаданий:

1. Луч в мире → `PhysicsDirectSpaceState3D.intersect_ray` по слою `body_hit` → `PhysicalBone3D` → индекс кости `b`.
2. `M_b = global_bone_pose(b) * inverse(global_rest_pose(b))` — текущая деформация для вокселей, у которых
   главная кость `b`. Переводим луч: `o' = inverse(M_b) * o`, `d' = inverse(M_b).basis * d`, затем DDA по каждой
   `rest`-части (`VoxelVolume.raycast` в её voxel space), берём ближайший hit.
3. Уточнение (одна итерация): у найденного вокселя читаем `bone0, bone1, w0`; если `w0 < 255`, строим
   смешанный трансформ `M = w0·M_bone0 + (1−w0)·M_bone1` (линейная смесь матриц, достаточно для суставов),
   пересчитываем луч и повторяем DDA один раз. Результат: воксель + точка в rest-пространстве.
4. Операции (`cut_global` и т.д.) переводят входные точки тем же `inverse(M)` по кости, найденной лучом
   (для сегмента a–b — по кости хита `a`), и работают в rest-пространстве. Этого достаточно для бритья:
   лезвие ведётся по одной области за кадр.

Капсулы `PhysicalBone3D`: из `bone.physics`; если в файле нет — радиус по максимальному расстоянию вокселей
кости от её оси (считает загрузчик), высота = `length`.

## Куски

`detach_loose()`: для каждого куска считает главную кость большинства вокселей `b` и `transform = M_b` (в мир);
спавн `RigidBody3D` — на стороне GDScript (`godot/scripts/voxel_debris.gd`, как в v1), меш куска — через
`VoxelPart` в режиме `world`.

## Что остаётся из v1

`VoxelPart`/`VoxelModelNode` работают как раньше (пропсы, жёсткие части, лаборатория v1). `VoxelSkinnedBody`
переиспользует `VoxelVolume`, `VoxelModel`, `VoxelMesher`.

## Тесты (`godot/tests`)

Загрузка эталонного человека; число костей = профилю; Skin bind-позы = инверсия rest; попадание лучом в кисть
в T-позе даёт кость `LeftHand`; после поворота `LeftUpperArm` на 90° луч в новую позицию кисти находит те же
воксели; `cut_global` по предплечью удаляет > 0 вокселей и перестраивает только чанки предплечья;
`detach_loose` после отрезания кисти возвращает кусок с главной костью `LeftHand` и трансформом у кисти.
