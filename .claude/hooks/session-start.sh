#!/usr/bin/env bash
# SessionStart hook for Claude Code cloud sessions of voxEngine.
# Registered in .claude/settings.json; runs at every session start/resume. Idempotent and
# fast (well under a second) once everything is in place. Steps:
#   1. PATH gets $HOME/.local/bin (persisted for the session through $CLAUDE_ENV_FILE)
#   2. Python deps: pip install -e tools/voxgen (numpy, pillow) + pytest + scons - only what is missing
#   3. Godot 4.7.2 linux binary via scripts/install_godot.sh (skipped when `godot --version` says 4.7.2)
#   4. git submodule update --init --recursive when src/voxbody/godot-cpp is empty
#   5. background: scons target=template_debug in src/voxbody (log build/scons.log) when there is no
#      linux debug .so or the C++ sources are newer than it (cheap: scons hashes content, so an
#      up-to-date tree only relinks nothing)
#   6. background: voxgen build-all (log build/voxgen.log) when models are missing or specs are newer
#   7. one-line status
# Network problems never fail the session: every step only warns and the hook exits 0.
# Only runs in cloud sessions (CLAUDE_CODE_REMOTE=true); VOXENGINE_HOOK_FORCE=1 runs it anywhere.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ] && [ "${VOXENGINE_HOOK_FORCE:-}" != "1" ]; then
  exit 0
fi

ROOT="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"
mkdir -p build

NWARN=0
log()  { echo "[voxengine] $*"; }
warn() { echo "[voxengine] WARNING: $*" >&2; NWARN=$((NWARN + 1)); }

# --- 1. PATH ---------------------------------------------------------------------------------
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *)
    export PATH="$HOME/.local/bin:$PATH"
    if [ -n "${CLAUDE_ENV_FILE:-}" ] && ! grep -qsF '.local/bin' "$CLAUDE_ENV_FILE"; then
      echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$CLAUDE_ENV_FILE"
    fi ;;
esac

# --- 2. Python deps --------------------------------------------------------------------------
PY=python3
voxgen_editable_ok() {
  "$PY" -c "import os, sys, numpy, PIL, voxgen; sys.exit(0 if os.path.realpath(os.path.dirname(voxgen.__file__)) == os.path.realpath(sys.argv[1]) else 1)" \
    "$ROOT/tools/voxgen/voxgen" >/dev/null 2>&1
}
pkgs=()
if ! voxgen_editable_ok; then pkgs+=(-e "$ROOT/tools/voxgen"); fi
if ! "$PY" -c "import pytest" >/dev/null 2>&1; then pkgs+=(pytest); fi
if ! command -v scons >/dev/null 2>&1; then pkgs+=(scons); fi
if [ ${#pkgs[@]} -gt 0 ]; then
  log "pip install ${pkgs[*]}"
  if PIP_BREAK_SYSTEM_PACKAGES=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
     "$PY" -m pip install -q "${pkgs[@]}" > build/pip.log 2>&1; then
    hash -r
  else
    warn "pip install failed (offline?), see build/pip.log"
  fi
fi
PY_STATUS="ok"
if ! voxgen_editable_ok; then PY_STATUS="voxgen MISSING"; fi
if ! "$PY" -c "import pytest" >/dev/null 2>&1; then PY_STATUS="$PY_STATUS, pytest MISSING"; fi
if ! command -v scons >/dev/null 2>&1; then PY_STATUS="$PY_STATUS, scons MISSING"; fi

# --- 3. Godot 4.7.2 --------------------------------------------------------------------------
GODOT_STATUS="godot MISSING"
if command -v godot >/dev/null 2>&1 && godot --version 2>/dev/null | grep -q '^4\.7\.2\.'; then
  GODOT_STATUS="godot $(godot --version 2>/dev/null | cut -d. -f1-4)"
else
  log "installing Godot 4.7.2 (scripts/install_godot.sh)"
  if bash "$ROOT/scripts/install_godot.sh" > build/install_godot.log 2>&1; then
    hash -r
    GODOT_STATUS="godot $(godot --version 2>/dev/null | cut -d. -f1-4)"
  else
    warn "Godot download failed (offline?), see build/install_godot.log"
  fi
fi

# --- 4. godot-cpp submodule ------------------------------------------------------------------
if [ ! -f src/voxbody/godot-cpp/SConstruct ]; then
  log "godot-cpp submodule is empty: git submodule update --init --recursive"
  if ! git submodule update --init --recursive > build/submodule.log 2>&1; then
    warn "submodule checkout failed (offline?), see build/submodule.log"
  fi
fi

# --- 5. extension build (background) ---------------------------------------------------------
ext_needed() {
  local lib
  shopt -s nullglob
  local libs=(godot/addons/voxbody/bin/libvoxbody.linux.template_debug.*.so)
  shopt -u nullglob
  if [ ${#libs[@]} -eq 0 ]; then return 0; fi
  for lib in "${libs[@]}"; do
    if [ -n "$(find src/voxbody/src src/voxbody/SConstruct -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.hpp' -o -name SConstruct \) -newer "$lib" -print -quit)" ]; then
      return 0
    fi
  done
  return 1
}
if pgrep -f 'scons.*target=template_debug' >/dev/null 2>&1; then
  EXT_STATUS="building (already running, build/scons.log)"
elif ! ext_needed; then
  EXT_STATUS="ok"
elif [ ! -f src/voxbody/godot-cpp/SConstruct ]; then
  EXT_STATUS="SKIPPED (godot-cpp missing)"
  warn "cannot build the extension without the godot-cpp submodule"
elif ! command -v scons >/dev/null 2>&1; then
  EXT_STATUS="SKIPPED (scons missing)"
else
  log "building the extension in the background: scons target=template_debug (build/scons.log)"
  ( cd src/voxbody && setsid nohup scons -j"$(nproc)" target=template_debug \
      > "$ROOT/build/scons.log" 2>&1 < /dev/null & )
  EXT_STATUS="building (build/scons.log)"
fi

# --- 6. assets (background) ------------------------------------------------------------------
assets_needed() {
  local oldest
  oldest="$(ls -t godot/assets/models/*.vxm 2>/dev/null | tail -n 1 || true)"
  if [ -z "$oldest" ]; then return 0; fi
  if [ -n "$(find tools/voxgen/specs tools/voxgen/voxgen -type f -name '*.py' -newer "$oldest" -print -quit)" ]; then
    return 0
  fi
  return 1
}
if pgrep -f '(-m |/)voxgen build-all' >/dev/null 2>&1; then
  ASSETS_STATUS="building (already running, build/voxgen.log)"
elif ! assets_needed; then
  ASSETS_STATUS="ok"
elif ! "$PY" -c "import numpy, PIL" >/dev/null 2>&1; then
  ASSETS_STATUS="SKIPPED (numpy/pillow missing)"
else
  log "generating models in the background: voxgen build-all (build/voxgen.log)"
  ( PYTHONPATH="$ROOT/tools/voxgen${PYTHONPATH:+:$PYTHONPATH}" setsid nohup "$PY" -m voxgen build-all tools/voxgen/specs \
      --out godot/assets/models --preview build/previews --vox build/vox \
      > "$ROOT/build/voxgen.log" 2>&1 < /dev/null & )
  ASSETS_STATUS="building (build/voxgen.log)"
fi

# --- 7. status -------------------------------------------------------------------------------
log "ready: $GODOT_STATUS | python: $PY_STATUS | ext: $EXT_STATUS | assets: $ASSETS_STATUS | warnings: $NWARN"
exit 0
