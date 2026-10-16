# Как мы работаем: Константин + Claude Code

Роли: **Константин** — архитектор и арт-директор. Ставит задачи словами и скриншотами, редактирует сцены
в редакторе Godot и модели в MagicaVoxel, принимает результат глазами. Код не правит. **Claude** —
единственный исполнитель: код, спеки, модели, тесты, документация, ветки и пуши.

## 1. Старт сессии

Хук `.claude/hooks/session-start.sh` (зарегистрирован в `.claude/settings.json`) при каждом старте
облачной сессии: добавляет `~/.local/bin` в PATH, доставляет недостающие Python-зависимости
(`pip install -e tools/voxgen`, pytest, scons), ставит Godot 4.7.2, инициализирует сабмодуль godot-cpp
и **в фоне** запускает сборку расширения (`build/scons.log`) и генерацию моделей (`build/voxgen.log`),
если они устарели. Последняя строка хука — статус.

Первые действия Claude: прочитать `CLAUDE.md`; `pgrep -fa scons` и `tail build/scons.log` — не мешать
фоновой сборке; `ls godot/addons/voxbody/bin godot/assets/models` — что уже есть. Запускать свой `scons`
только когда фоновый закончился.

## 2. Приём задачи (слова + скриншоты)

1. Пересказать задачу в одну-две строки своими словами (что изменится для игрока/владельца) и найти её
   место в `docs/roadmap.md`; если её там нет — добавить строку.
2. Если задача визуальная (скриншот, «нос больше», «слишком красное») — сначала строка в
   `docs/style-guide.md` → «Журнал правок», потом правка.
3. Определить компоненты (`voxgen` / `voxbody` / `godot` / `docs` / `infra`) и открыть нужный скилл
   (`.claude/skills/`). Если задача требует изменения контракта — сначала правка `docs/spec/*.md`.
4. Вопросы владельцу — только блокирующие, не более трёх, с вариантами и рекомендацией. Всё, что можно
   решить самому и переделать за минуты, решаем сами и фиксируем решение в коде/ADR.

## 3. Ветки, коммиты, пуш

- Ветка на задачу: `claude/<topic>` (`claude/razor-depth`, `claude/head-caricature`). Коммитит и пушит
  Claude (кроме агентных сессий, где это делает оркестратор).
- Сообщение коммита — **по-английски, повелительное наклонение, префикс области**:
  `voxbody:`, `voxgen:`, `godot:`, `docs:`, `infra:` (несколько — через запятую: `voxbody, godot:`).
  Первая строка ≤ 72 символов; тело — зачем и что проверено. Примеры:
  `voxgen: add caricature head variant (nose +40%, ears +30%)`,
  `voxbody: clamp carve radius to part bounds; add headless test`,
  `docs: log owner's note on paler skin in style guide`.
- Никогда не коммитятся: `*.vxm`, `*.vox` (кроме осознанно сохранённых правок владельца рядом со спеком,
  см. раздел 7), `build/`, `godot/.godot/`, бинарники из `addons/voxbody/bin/`.
- Резюме в конце работы — по-русски, 3–7 строк: что сделано, куда смотреть (пути к PNG/скриншоту,
  ветка), что решить владельцу.

## 4. Критерии готовности (definition of done)

| Вид задачи | Готово, когда |
|---|---|
| **Модель** (спек в `tools/voxgen/specs`) | `voxgen build` с кодом 0, `errors: []`, предупреждения устранены или объяснены; **превью PNG открыт и оценён** по чеклисту гайда; для тел посмотрен срез; `make test` зелёный (pytest); если модель грузится в lab — скриншот сделан |
| **Движок** (`src/voxbody`) | собираются **обе цели** (`make ext` и `make ext-release`); headless-тесты зелёные (`make test`), на новое поведение добавлен кейс в `godot/tests/voxbody_tests.gd`; `docs/spec/voxbody-api.md` обновлена, если менялось API; таблица материалов синхронна в трёх местах |
| **Сцена / GDScript** (`godot/`) | `godot --headless --path godot --import` без ошибок; `--check-only` по изменённым скриптам; headless-тесты зелёные; **скриншот попытались сделать** (`make screenshot` → `build/lab.png`) и посмотрели; `godot/README.md` обновлён, если менялось управление |
| **Генератор** (`tools/voxgen/voxgen`) | pytest зелёный, новые функции покрыты тестом; `make assets` проходит для всех спеков; `tools/voxgen/README.md` обновлён |
| **Документация** | ссылки существуют; команды из текста реально запускались в этой сессии |
| **Инфраструктура** (`scripts/`, `Makefile`, CI, хук) | скрипт запущен из другого cwd; CI-воркфлоу синтаксически валиден; хук идемпотентен |

## 5. Как владелец смотрит результат

1. Получает ветку и резюме. На Windows: `git pull`, берёт `.dll` из артефакта CI `voxbody-windows`
   (кладёт в `godot/addons/voxbody/bin/`), генерирует модели (`python -m voxgen build-all ...`, см.
   корневой `README.md`).
2. Открывает `godot/` в редакторе Godot 4.7, запускает `scenes/lab.tscn` (F5): 1–5 инструменты, ЛКМ —
   применить, ПКМ — орбита, R — новая голова, F — каркас, F12 — скриншот в `user://screenshots/`.
3. Смотрит `build/previews/*.png` (или артефакт CI `previews`) — то же, что видел Claude.
4. Отвечает словами и скриншотом. Каждое замечание → «Журнал правок» → правка → новая ветка/коммит.
5. Правки сцен, сделанные владельцем в редакторе, коммитит сам владелец (текстовые `.tscn`/`.tres`);
   Claude после этого делает `--import` и тесты, чтобы ничего не сломалось.

## 6. Куда что попадает

| Что | Куда | В git? |
|---|---|---|
| Спеки моделей | `tools/voxgen/specs/**/*.py` (библиотечные модули — с `_`) | да |
| Модели `.vxm` | `godot/assets/models/` | нет (`make assets`) |
| Превью, `.vox` + `.vox.json`, отчёты проверок | `build/previews/`, `build/vox/`, `build/reports/` | нет |
| Библиотеки расширения | `godot/addons/voxbody/bin/` | нет (`make ext`, CI) |
| Скриншот лаборатории | `build/lab.png` (`make screenshot`); из игры — `user://screenshots/` | нет |
| Логи хука и сборок | `build/*.log` | нет |
| Временные файлы агента | scratchpad сессии (`/tmp/claude-*/…/scratchpad/<area>/`) | нет |

## 7. Как добавить материал

Материал — число 0…63 (1–31 тело, 32–62 предметы, 63 generic). Три места меняются **одним коммитом**:

1. `docs/spec/vxm-format.md` — строка в таблице материалов (id, имя, свойства) — источник истины.
2. `src/voxbody/src/voxel_volume.h` — `enum Material` (`MAT_<NAME> = id`) + `BIND_ENUM_CONSTANT` в
   `voxel_volume.cpp`; если материал кровоточит / принимает синяки / твёрдый — `material_bleeds` /
   `material_bruises`; блестящие — список в `voxel.gdshader` (`gloss_roughness`).
3. `tools/voxgen/voxgen/materials.py` — константа, `NAMES`, `DEFAULT_COLORS`, множества `BLEEDS` /
   `BRUISES` / `HARD`; при необходимости — правила `checks.layer_order` и тинт в `render.py`.

Потом: базовый цвет в `docs/style-guide.md` (раздел 7), тест в pytest (имя ↔ id) и кейс в
`godot/tests/voxbody_tests.gd`, если у материала особое поведение в операциях.

## 8. Как добавить инструмент в лабораторию

1. `godot/scripts/tools.gd`: значение в `enum Tool`, строки в `TOOL_NAMES` (русское имя для HUD),
   `TOOL_KEYS`, `TOOL_ACTIONS` (`"tool_<name>": Tool.<NAME>`), `@export`-группа параметров, функция
   `_apply_<name>(part, hit, vs, ...) -> int` и ветка в `match tool` внутри `_apply_tool` (поставить
   `removes_voxels = true`, если инструмент удаляет воксели — тогда отвалившееся станет обломками).
2. `godot/project.godot`, секция `[input]`: действие `tool_<name>` с клавишей (`physical_keycode`:
   цифры 1…9 = 49…57).
3. Если нужна новая операция над вокселями — сначала `docs/spec/voxbody-api.md`, потом `VoxelVolume`
   (voxel space) и обёртка в `VoxelPart` (метры), тест в `voxbody_tests.gd`.
4. HUD-подсказка в `godot/scripts/lab.gd` (`_refresh_hud`), таблица в `godot/README.md`.
5. Проверка: `--import`, `--check-only --script res://scripts/tools.gd`, headless-тесты, `make screenshot`.
