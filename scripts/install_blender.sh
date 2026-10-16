#!/usr/bin/env bash
# Usage: scripts/install_blender.sh
#
# Creates the Blender-as-a-python-module venv that `voxgen render` (scripts/blender_render.sh) runs:
#   $HOME/.cache/voxengine/venv-bpy   (python3.11 -m venv; bpy==4.5.14 LTS, numpy, pillow, voxgen -e)
# Idempotent: when the venv already imports bpy of that version (and voxgen) nothing is installed.
# bpy wheels exist only for python 3.11; the download is ~300 MB (needs network / the agent proxy).
#
# Environment overrides: VOXENGINE_BPY (venv dir), BPY_VERSION (4.5.14), BPY_PYTHON (python3.11).
# Exit codes: 0 ready (prints the bpy version), 1 install failed, 2 no python3.11 on the host.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VOXENGINE_BPY:-$HOME/.cache/voxengine/venv-bpy}"
BPY_VERSION="${BPY_VERSION:-4.5.14}"
PY="${BPY_PYTHON:-python3.11}"
VPY="$VENV/bin/python"

log() { echo "[install_blender] $*"; }

# True when the venv python imports bpy of the wanted version plus the other deps.
venv_ok() {
  [ -x "$VPY" ] && "$VPY" -c "
import sys, bpy, numpy, PIL, voxgen
sys.exit(0 if bpy.app.version_string.startswith('$BPY_VERSION') else 1)" >/dev/null 2>&1
}

if venv_ok; then
  log "already installed: $VENV (bpy $("$VPY" -c 'import bpy; print(bpy.app.version_string)' 2>/dev/null))"
  exit 0
fi

if [ ! -x "$VPY" ]; then
  if ! command -v "$PY" >/dev/null 2>&1; then
    log "$PY not found: bpy $BPY_VERSION wheels need python 3.11 (apt install python3.11 python3.11-venv)"
    exit 2
  fi
  log "creating venv $VENV with $PY"
  mkdir -p "$(dirname "$VENV")"
  "$PY" -m venv "$VENV"
fi

if ! "$VPY" -c "import bpy, numpy, PIL" >/dev/null 2>&1 \
   || ! "$VPY" -c "import bpy, sys; sys.exit(0 if bpy.app.version_string.startswith('$BPY_VERSION') else 1)" >/dev/null 2>&1; then
  log "pip install bpy==$BPY_VERSION numpy pillow (large download)"
  "$VPY" -m pip install -q --upgrade pip
  "$VPY" -m pip install -q "bpy==$BPY_VERSION" numpy pillow || { log "pip install failed"; exit 1; }
fi
if ! "$VPY" -c "import voxgen" >/dev/null 2>&1; then
  log "pip install -e $ROOT/tools/voxgen"
  "$VPY" -m pip install -q -e "$ROOT/tools/voxgen" || { log "voxgen install failed"; exit 1; }
fi

if venv_ok; then
  log "ready: $VENV (bpy $("$VPY" -c 'import bpy; print(bpy.app.version_string)' 2>/dev/null))"
else
  log "venv exists but bpy $BPY_VERSION / voxgen still do not import, see: $VPY -c 'import bpy, voxgen'"
  exit 1
fi
