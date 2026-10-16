#!/usr/bin/env bash
# Usage: scripts/build_ext.sh [debug|release] [extra scons args...]
#
# Builds the voxbody GDExtension (src/voxbody) with scons -j<nproc>.
#   debug (default) -> target=template_debug   (what the Godot editor loads)
#   release         -> target=template_release (what exported games load)
# Extra arguments go to scons verbatim, e.g. `platform=windows use_mingw=yes` or `arch=arm64`.
# Output lands in godot/addons/voxbody/bin/ as libvoxbody.<platform>.<target>.<arch>.<ext>
# (see src/voxbody/SConstruct); the path of the built library is printed on the last line.
# Environment: VOXENGINE_JOBS overrides the job count.
# Works from any cwd. The first build also compiles godot-cpp (~10 min on 4 cores); later builds
# are incremental.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mode="${1:-debug}"
if [ $# -gt 0 ]; then shift; fi
case "$mode" in
  debug | template_debug) target=template_debug ;;
  release | template_release) target=template_release ;;
  -h | --help | help)
    sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
    exit 0 ;;
  *)
    echo "build_ext.sh: unknown mode '$mode' (expected debug or release)" >&2
    exit 2 ;;
esac

if [ ! -f src/voxbody/godot-cpp/SConstruct ]; then
  echo "build_ext.sh: src/voxbody/godot-cpp is empty - run: git submodule update --init --recursive" >&2
  exit 1
fi
if ! command -v scons >/dev/null 2>&1; then
  echo "build_ext.sh: scons not found - run: python3 -m pip install scons" >&2
  exit 1
fi

# Host platform/arch in godot-cpp naming, overridable by platform=/arch= in the extra args.
case "$(uname -s)" in
  Linux) platform=linux ;;
  Darwin) platform=macos ;;
  MINGW* | MSYS* | CYGWIN* | Windows_NT) platform=windows ;;
  *) platform=linux ;;
esac
case "$(uname -m)" in
  x86_64 | amd64) arch=x86_64 ;;
  aarch64 | arm64) arch=arm64 ;;
  i?86) arch=x86_32 ;;
  *) arch="$(uname -m)" ;;
esac
for a in "$@"; do
  case "$a" in
    platform=*) platform="${a#platform=}" ;;
    arch=*) arch="${a#arch=}" ;;
  esac
done

jobs="${VOXENGINE_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
echo "build_ext.sh: scons -j$jobs target=$target $*  (platform=$platform arch=$arch)"
(cd src/voxbody && scons -j"$jobs" target="$target" "$@")

bin_dir="godot/addons/voxbody/bin"
case "$platform" in
  linux) out="$bin_dir/libvoxbody.linux.$target.$arch.so" ;;
  windows) out="$bin_dir/libvoxbody.windows.$target.$arch.dll" ;;
  macos) out="$bin_dir/libvoxbody.macos.$target.framework/libvoxbody.macos.$target" ;;
  *) out="" ;;
esac
if [ -n "$out" ] && [ -f "$out" ]; then
  echo "$ROOT/$out"
else
  echo "build_ext.sh: build finished but expected library not found: ${out:-?}" >&2
  ls -la "$bin_dir" >&2 || true
  exit 1
fi
