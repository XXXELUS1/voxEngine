# voxEngine

Воксельный движок тел и повреждений для игры **Barberians** (Godot 4.7, кооперативный
«цирюльник от первого лица»: бреем, стрижём и рвём зубы воксельным головам). Голова — плотный
3D-массив вокселей (материал + цвет + состояние), из него строится меш; бритва удаляет воксели,
синяки и ожоги перекрашивают, отвалившиеся куски становятся отдельными телами. Это фундамент:
генератор моделей на Python (`voxgen`), C++-расширение для Godot (`voxbody`) и лабораторная сцена.

## Карта репозитория

| Путь | Что там |
|------|---------|
| `docs/` | Диздок (`docs/design/barberians.md`) и контракты: формат `.vxm`, API `voxbody`, CLI `voxgen` (`docs/spec/`) |
| `godot/` | Проект Godot 4.7: аддон `addons/voxbody` (GDExtension + шейдер), лабораторная сцена, `tests/` (headless-тесты), `assets/models/` (генерируемые `.vxm`) |
| `src/voxbody/` | C++ исходники расширения (`src/`), `SConstruct`, сабмодуль `godot-cpp` |
| `tools/voxgen/` | Python-генератор: спеки моделей (`specs/`), библиотека SDF/слоёв/волос, рендер превью, проверки, тесты |
| `scripts/` | `build_ext.sh`, `build_assets.sh`, `test_all.sh`, `screenshot.sh`, `install_godot.sh` — всё, что умеет `Makefile` |
| `build/` | Генерируемое и не коммитится: превью `previews/`, `.vox`, отчёты проверок, логи, `lab.png` |
| `.github/workflows/` | CI: тесты voxgen + превью, сборка расширения под Linux и Windows (артефакты с `.so`/`.dll`), headless-тесты Godot |
| `.claude/` | Хук старта облачной сессии Claude Code (`hooks/session-start.sh`) и скиллы |

## Быстрый старт для владельца (Windows)

1. Установить **Godot 4.7.2** (обычная сборка, не .NET): <https://godotengine.org/download>.
2. Клонировать вместе с сабмодулем:
   ```
   git clone --recurse-submodules <url репозитория> voxEngine
   ```
   (если уже клонировано без него: `git submodule update --init --recursive`).
3. Получить библиотеку расширения — один из двух способов:
   - **Из CI** (без компилятора): на GitHub во вкладке *Actions* открыть последний зелёный прогон
     нужной ветки, скачать артефакт **`voxbody-windows`** и распаковать оба `.dll` в
     `godot/addons/voxbody/bin/`.
   - **Собрать самому**: поставить *Visual Studio 2022 Build Tools* (компонент «Desktop development
     with C++»), Python 3 и `pip install scons`, затем в консоли:
     ```
     cd src\voxbody
     scons -j4 target=template_debug
     scons -j4 target=template_release
     ```
     Первая сборка компилирует godot-cpp (10–20 минут), дальше — секунды. Файлы попадут в
     `godot\addons\voxbody\bin\libvoxbody.windows.<target>.x86_64.dll`.
4. Сгенерировать модели (`.vxm` не хранятся в git):
   ```
   pip install -e tools\voxgen
   python -m voxgen build-all tools\voxgen\specs --out godot\assets\models --preview build\previews --vox build\vox
   ```
   Превью-листы появятся в `build\previews\*.png`.
5. Открыть папку `godot/` в Godot (Import → выбрать `godot/project.godot`) и запустить лабораторную сцену.

## Быстрый старт для Claude (облачная сессия)

Хук `.claude/hooks/session-start.sh` при старте сессии сам ставит зависимости, скачивает Godot 4.7.2,
в фоне собирает расширение (`build/scons.log`) и генерирует модели (`build/voxgen.log`). Дальше:

```
make ext && make assets && make test     # сборка расширения, модели, pytest + headless-тесты Godot
make screenshot                          # рендер лабораторной сцены без GPU -> build/lab.png
make ext-release                         # release-сборка; make clean / make distclean — уборка
```

Те же действия без `make`: `scripts/build_ext.sh [release]`, `scripts/build_assets.sh`,
`scripts/test_all.sh`, `scripts/screenshot.sh [out.png]` (запускаются из любой директории).

## Дальше

- Правила работы и соглашения: [`CLAUDE.md`](CLAUDE.md)
- Контракты: [`docs/spec/vxm-format.md`](docs/spec/vxm-format.md),
  [`docs/spec/voxbody-api.md`](docs/spec/voxbody-api.md), [`docs/spec/voxgen-cli.md`](docs/spec/voxgen-cli.md)
- Диздок: [`docs/design/barberians.md`](docs/design/barberians.md)
