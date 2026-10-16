# voxEngine — памятка для Claude Code

Читать первой в каждой сессии. Владелец (Константин) — архитектор и арт-директор, код не правит:
он редактирует сцены/модели (MagicaVoxel или словами) и даёт отзывы текстом и скриншотами.
Единственный исполнитель — Claude. Качество важнее скорости: это фундамент всей игры.

## Что это
Воксельный движок тел и повреждений для **Barberians** (Godot 4.7, кооп-цирюльник от первого лица:
бреем, стрижём, рвём зубы воксельным головам). Голова — плотный массив вокселей (материал + цвет +
состояние) → меш; бритва удаляет воксели, синяки и ожоги перекрашивают, отвалившееся падает телом.

## Пять опор (docs/design/barberians.md)
1. **Ощущение рук** — бритва, клещи, удар, бросок: тактильно, со звуком, с мгновенным видимым результатом.
2. **Повреждения, которые остаются** — раны, синяки, шрамы, отрастающие волосы.
3. **Реакции NPC** — на боль, кровь, крики и твой вид.
4. **Город, который помнит** — постоянные жители, отношения, слухи.
5. **Рост** — инструменты, комнаты, улицы, гильдия, король в финале.

Движок закрывает опоры 1–2; всё остальное — контент поверх него.

## Карта репозитория
| Путь | Что там |
|---|---|
| `docs/spec/` | **Контракты — источник истины**: `vxm-format.md`, `voxbody-api.md`, `voxgen-cli.md` |
| `docs/design/barberians.md` | Диздок владельца (не править без просьбы); разделы «Визуал и звук», «Техника», «Этапы» |
| `docs/adr/`, `docs/style-guide.md`, `docs/workflow.md`, `docs/roadmap.md` | Решения, визуальный стиль + журнал правок, порядок работы, этапы |
| `tools/voxgen/` | Python-генератор: `specs/` (спеки моделей, коммитятся), `voxgen/` (SDF, слои, волосы, рендер, проверки), `tests/` |
| `src/voxbody/` | C++ GDExtension: `src/*.cpp,*.h`, `SConstruct`, сабмодуль `godot-cpp` 10.0.0-stable (api 4.7) |
| `godot/` | Проект Godot: `addons/voxbody/` (`.gdextension`, `bin/`, шейдер, материал), `scenes/lab.tscn`, `scripts/`, `tests/`, `assets/models/*.vxm` (генерируется) |
| `scripts/`, `Makefile` | `build_ext.sh`, `build_assets.sh`, `test_all.sh`, `screenshot.sh`, `install_godot.sh` и `make`-обёртки |
| `build/` | Всё генерируемое, не в git: `previews/*.png`, `vox/`, `reports/`, логи хука, `lab.png` |
| `.claude/` | `settings.json`, `hooks/session-start.sh` (зависимости, Godot, фоновая сборка), `skills/` |

## Архитектура: два слоя
```
tools/voxgen (Python, оффлайн)  ──.vxm──▶  src/voxbody (C++, рантайм)  ◀──  godot/ (GDScript: логика, сцены)
спек .py → SDF → слои → волосы → проверки → PNG    загрузка → greedy mesh → carve/bruise/burn/trim → detach
```
- **voxgen** делает всё медленное и творческое: моделирование, ткани, волосы, превью-рендер, проверки.
  Numpy, без scipy. Выход: `.vxm` + `.vox` (MagicaVoxel) + PNG-превью + JSON-отчёт.
- **voxbody** делает только быстрое и детерминированное: хранит три байтовых поля на часть, режет и
  красит в voxel space, строит меш только для грязной части. Геометрию в рантайме не генерирует.
- `.vxm` — единственный контракт между слоями (`docs/spec/vxm-format.md`). Почему так: `docs/adr/0004-two-layer-pipeline.md`.

## Команды (из корня репозитория)
```
make ext            # debug-сборка расширения → godot/addons/voxbody/bin/ (первая ~10 мин: godot-cpp)
make ext-release    # release-сборка; перед сдачей C++ должны собираться обе цели
make assets         # все спеки → godot/assets/models/*.vxm, build/previews/*.png, build/vox/, build/reports/
make test           # pytest tools/voxgen, затем headless-тесты Godot (godot/tests/run_tests.gd)
make screenshot     # сцена lab через Xvfb + llvmpipe → build/lab.png (GPU не нужен)
python3 -m voxgen build tools/voxgen/specs/head_test.py                 # одна модель + превью + .vox + отчёт
python3 -m voxgen preview godot/assets/models/head_test.vxm --views front,iso,slice_x --out build/previews/x.png
python3 -m voxgen check godot/assets/models/head_test.vxm                # отчёт проверок; info — состав модели
godot --headless --audio-driver Dummy --path godot --import              # после правок сцен/скриптов/class_name
godot --headless --audio-driver Dummy --path godot -s res://tests/run_tests.gd
```
Перед своим `scons` — `pgrep -fa scons`: хук сессии мог запустить сборку в фоне (лог `build/scons.log`).

## Соглашения
- **Оси** Godot: Y вверх, −Z вперёд (лицо персонажа смотрит в −Z), X вправо от зрителя. Единицы — метры.
- **voxel space** — координаты в вокселях части, центр вокселя (x,y,z) = (x+0.5, y+0.5, z+0.5);
  **local space** — метры узла `VoxelPart`: `local = origin + voxel * voxel_size`. `VoxelVolume` работает
  только в voxel space, `VoxelPart` конвертирует. Индекс поля `x + nx*(y + ny*z)` (x быстрее всех);
  в numpy это shape `(nz, ny, nx)`, индекс `[z, y, x]`.
- **Материалы живут в двух местах кода**: `src/voxbody/src/voxel_volume.h` (enum `Material`) и
  `tools/voxgen/voxgen/materials.py`; истина — таблица в `docs/spec/vxm-format.md`. Менять все три вместе.
- **Спеки — источник истины.** Сначала правим `docs/spec/*.md`, потом код. Где спека молчит — принимаем
  решение и фиксируем его в шапке файла (англ.) или в ADR.
- **Генерируемое не коммитится**: `*.vxm`, `*.vox`, `build/`, `godot/.godot/`, `addons/voxbody/bin/*`.
  Коммитятся спеки `.py`, исходники, сцены, `.uid`, тесты.
- **GDScript**: статическая типизация везде (`var x: int`, `-> void`, типизированный `for`), табы
  (`.editorconfig`), `class_name` у переиспользуемых скриптов, `##`-докстринги, сигналы вверх / вызовы вниз.
- **C++**: стиль Godot (табы, `snake_case`, `p_`-параметры как в godot-cpp), без исключений и RTTI,
  `Ref<>` для ресурсов, `PackedByteArray` для полей, `ERR_FAIL_*` вместо падений, комментарии на английском.
- **Python**: PEP 8, 4 пробела, type hints, numpy-векторизация вместо циклов по вокселям, без scipy.
- Проза в `docs/` — по-русски; идентификаторы, команды, комментарии в коде — по-английски.

## Рабочее соглашение с владельцем
1. Работаем в ветках `claude/<topic>`, пушим, в конце — короткое резюме по-русски: что сделано, куда
   смотреть (путь к PNG/скриншоту), что решить. Коммиты — по-английски с префиксом области (`docs/workflow.md`).
2. **Каждая визуальная правка от владельца** («нос крупнее», «кожа бледнее») сначала записывается в
   `docs/style-guide.md` → «Журнал правок», и только потом вносится в спек. Повторяемое — становится правилом гайда.
3. **После сборки модели — обязательно открыть превью** `build/previews/<name>.png` (инструментом Read)
   и оценить по чеклисту скилла `review-model-preview` до того, как отчитываться.
4. **После любой правки в `godot/` или `src/voxbody/`** — `make test` (минимум headless-тесты);
   после правки сцены — попытаться `make screenshot` и посмотреть `build/lab.png`.
5. **Никогда не коммитить `.vxm`, `.vox` и бинарники.** Не трогать `docs/design/` без просьбы владельца.
6. Не оставлять TODO-заглушек там, где спека требует реализации. Невозможное — честно в отчёт.

## Где читать дальше
- Контракты: `docs/spec/vxm-format.md`, `docs/spec/voxbody-api.md`, `docs/spec/voxgen-cli.md`.
- Почему так: `docs/adr/` (движок, своё ядро, godot-cpp, два слоя, детерминизм, плотные массивы).
- Как выглядит «наше»: `docs/style-guide.md`. Как работаем: `docs/workflow.md`. Индекс: `docs/README.md`.
- Скиллы `.claude/skills/`: `voxgen-model` (модели), `voxbody-dev` (C++), `godot-lab` (сцена и тесты),
  `review-model-preview` (честная оценка превью и отчёт владельцу).

## Текущий этап
**Этап 0 «База»** закрывается этой сессией (спеки, voxgen, voxbody, лаборатория, скрипты, CI, документация).
Дальше — **Этап 1 «Прототип ядра»**: весело ли брить и резать воксельную голову. Список задач, статус и
компоненты — в `docs/roadmap.md`; при смене этапа обновлять этот блок и роадмап.
