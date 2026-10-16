#!/usr/bin/env bash
# Usage: scripts/screenshot.sh [out.png] [extra godot args...]
#
# Renders the lab scene off-screen and writes a PNG (default build/lab.png):
#   xvfb-run -a godot --path godot --rendering-driver opengl3 --resolution 1280x720 \
#       -s res://tests/screenshot.gd -- <absolute out.png>
# No GPU is needed: Xvfb + Mesa llvmpipe drive the Compatibility renderer (opengl3).
# The output path is made absolute and handed to the script as OS.get_cmdline_user_args()[0]
# (Godot changes its cwd to the project dir, so a relative path would land in godot/).
# When DISPLAY is already set (a desktop session) Godot runs without xvfb.
# Extra arguments go to godot before -s (e.g. --resolution 1920x1080). Works from any cwd.
# Needs: godot on PATH, a built extension, xvfb-run (apt install xvfb) unless DISPLAY is set.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

out="${1:-build/lab.png}"
if [ $# -gt 0 ]; then shift; fi
case "$out" in
  /*) ;;
  *) out="$ROOT/$out" ;;
esac
mkdir -p "$(dirname "$out")"

if ! command -v godot >/dev/null 2>&1; then
  echo "screenshot.sh: godot not on PATH (run scripts/install_godot.sh)" >&2
  exit 1
fi
if [ ! -f godot/tests/screenshot.gd ]; then
  echo "screenshot.sh: godot/tests/screenshot.gd does not exist" >&2
  exit 1
fi

cmd=(godot --path "$ROOT/godot" --rendering-driver opengl3 --resolution 1280x720 --audio-driver Dummy
     "$@" -s res://tests/screenshot.gd -- "$out")
rm -f "$out"
if [ -n "${DISPLAY:-}" ]; then
  echo "screenshot.sh: DISPLAY=$DISPLAY, running without xvfb"
  "${cmd[@]}"
else
  if ! command -v xvfb-run >/dev/null 2>&1; then
    echo "screenshot.sh: xvfb-run not found (apt install xvfb) and no DISPLAY" >&2
    exit 1
  fi
  xvfb-run -a "${cmd[@]}"
fi

if [ -s "$out" ]; then
  echo "$out"
else
  echo "screenshot.sh: godot exited but $out was not written" >&2
  exit 1
fi
