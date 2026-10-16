#!/usr/bin/env bash
# Usage: scripts/build_assets.sh [extra `voxgen build-all` args...]
#
# Generates every model from tools/voxgen/specs (specs starting with "_" are library modules
# and are skipped):
#   .vxm      -> godot/assets/models/   (loaded by the voxbody extension)
#   .png      -> build/previews/        (contact sheets: the agent's eyes)
#   .vox+json -> build/vox/             (MagicaVoxel)
#   reports   -> build/reports/<name>.json
# Exit code is voxgen's: 1 when the checks report errors. Works from any cwd.
# Needs numpy + pillow; voxgen is imported from tools/voxgen even without `pip install -e`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then PY=python; fi
export PYTHONPATH="$ROOT/tools/voxgen${PYTHONPATH:+:$PYTHONPATH}"

if ! "$PY" -c "import numpy, PIL, voxgen" >/dev/null 2>&1; then
  echo "build_assets.sh: voxgen deps missing - run: $PY -m pip install -e tools/voxgen" >&2
  exit 1
fi

mkdir -p build godot/assets/models
echo "build_assets.sh: $PY -m voxgen build-all tools/voxgen/specs $*"
"$PY" -m voxgen build-all tools/voxgen/specs \
  --out godot/assets/models --preview build/previews --vox build/vox "$@"
