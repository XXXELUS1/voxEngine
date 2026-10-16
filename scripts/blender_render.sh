#!/usr/bin/env bash
# Usage: scripts/blender_render.sh <model.vxm|spec.py> [voxgen render options...]
#
# Thin wrapper for `python3 -m voxgen render`: Blender beauty renders of a model
# (styles cubes/bevel/smooth/clay/toon, lights studio/candle, camera views) into
# build/renders/<name>/ with a labelled sheet.png. Needs the bpy venv from
# scripts/install_blender.sh ($VOXENGINE_BPY or ~/.cache/voxengine/venv-bpy). Works from any cwd.
#   scripts/blender_render.sh godot/assets/models/head_test.vxm --styles cubes,clay --views three_quarter,front
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then PY=python; fi
export PYTHONPATH="$ROOT/tools/voxgen${PYTHONPATH:+:$PYTHONPATH}"

if [ $# -lt 1 ]; then
  sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
  exit 2
fi
VENV="${VOXENGINE_BPY:-$HOME/.cache/voxengine/venv-bpy}"
if [ ! -x "$VENV/bin/python" ] && [ ! -x "$VENV/Scripts/python.exe" ]; then
  echo "blender_render.sh: bpy venv missing at $VENV - run: scripts/install_blender.sh" >&2
  exit 1
fi
exec "$PY" -m voxgen render "$@"
