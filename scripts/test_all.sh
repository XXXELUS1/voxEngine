#!/usr/bin/env bash
# Usage: scripts/test_all.sh [--no-godot]
#
# Runs everything that can be tested on this machine and exits non-zero on any failure:
#   1. pytest for tools/voxgen (python3 -m pytest, so the same interpreter that has numpy/pillow)
#   2. if a built voxbody library exists in godot/addons/voxbody/bin and `godot` is on PATH:
#      imports the Godot project once (godot --headless --import, creates godot/.godot/ and the
#      extension list) and runs the headless suite  godot --headless --path godot -s res://tests/run_tests.gd
#      The Godot part is skipped with a notice when the extension or Godot is missing;
#      set VOXENGINE_REQUIRE_GODOT=1 to turn that skip into a failure (CI does its own thing).
# Works from any cwd.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

run_godot=1
if [ "${1:-}" = "--no-godot" ]; then run_godot=0; fi

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then PY=python; fi
status=0

echo "== pytest (tools/voxgen)"
if (cd tools/voxgen && "$PY" -m pytest -q); then
  echo "== pytest: OK"
else
  echo "== pytest: FAILED" >&2
  status=1
fi

godot_skip() {
  echo "== godot tests: SKIPPED - $1"
  if [ "${VOXENGINE_REQUIRE_GODOT:-0}" = "1" ]; then
    echo "   (VOXENGINE_REQUIRE_GODOT=1 makes this a failure)" >&2
    status=1
  fi
}

if [ "$run_godot" = 1 ]; then
  shopt -s nullglob
  libs=(godot/addons/voxbody/bin/libvoxbody.*.so godot/addons/voxbody/bin/libvoxbody.*.dll godot/addons/voxbody/bin/libvoxbody.*.framework)
  shopt -u nullglob
  if [ ${#libs[@]} -eq 0 ]; then
    godot_skip "extension not built (run scripts/build_ext.sh or make ext)"
  elif ! command -v godot >/dev/null 2>&1; then
    godot_skip "godot not on PATH (run scripts/install_godot.sh)"
  elif [ ! -f godot/tests/run_tests.gd ]; then
    godot_skip "godot/tests/run_tests.gd does not exist"
  else
    echo "== godot --headless --import"
    if ! godot --headless --audio-driver Dummy --path godot --import; then
      echo "== godot import: FAILED" >&2
      status=1
    fi
    echo "== godot --headless -s res://tests/run_tests.gd"
    if godot --headless --audio-driver Dummy --path godot -s res://tests/run_tests.gd; then
      echo "== godot tests: OK"
    else
      echo "== godot tests: FAILED" >&2
      status=1
    fi
  fi
fi

if [ "$status" = 0 ]; then
  echo "== all tests passed"
else
  echo "== some tests FAILED" >&2
fi
exit "$status"
