#!/usr/bin/env bash
# Usage: scripts/install_godot.sh [--force]
#
# Installs the Godot 4.7.2 editor binary for Linux (x86_64 or arm64) the way the
# cloud-session hook (.claude/hooks/session-start.sh) and CI expect it:
#   download  https://github.com/godotengine/godot/releases/download/4.7.2-stable/Godot_v4.7.2-stable_linux.<arch>.zip
#   verify    against SHA512-SUMS.txt of the same release (skipped with a warning if the sums file is unreachable)
#   unpack    to   $HOME/.cache/voxengine/godot-4.7.2
#   symlink   $HOME/.local/bin/godot -> that binary
# Idempotent: when `godot --version` (or the cached binary) already reports 4.7.2 nothing is
# downloaded and only the symlink is (re)created. `--force` re-downloads anyway.
#
# Environment overrides: GODOT_VERSION (4.7.2), VOXENGINE_CACHE_DIR ($HOME/.cache/voxengine),
# GODOT_BIN_DIR ($HOME/.local/bin).
# Exit codes: 0 installed or already present, 1 download/verification failed, 2 unsupported host.
set -euo pipefail

GODOT_VERSION="${GODOT_VERSION:-4.7.2}"
CACHE_DIR="${VOXENGINE_CACHE_DIR:-$HOME/.cache/voxengine}"
BIN_DIR="${GODOT_BIN_DIR:-$HOME/.local/bin}"
TARGET="$CACHE_DIR/godot-$GODOT_VERSION"
LINK="$BIN_DIR/godot"
FORCE=0
if [ "${1:-}" = "--force" ]; then FORCE=1; fi

log() { echo "[install_godot] $*"; }

case "$(uname -s)" in
  Linux) ;;
  *)
    log "only Linux is handled here; on Windows/macOS install Godot $GODOT_VERSION from https://godotengine.org/download"
    exit 2 ;;
esac
case "$(uname -m)" in
  x86_64 | amd64) ARCH=x86_64 ;;
  aarch64 | arm64) ARCH=arm64 ;;
  *) log "unsupported architecture: $(uname -m)"; exit 2 ;;
esac

TAG="${GODOT_VERSION}-stable"
ASSET="Godot_v${TAG}_linux.${ARCH}"
ZIP="${ASSET}.zip"
BASE_URL="https://github.com/godotengine/godot/releases/download/${TAG}"

# True when $1 is an executable that prints the wanted version.
reports_version() {
  [ -x "$1" ] && "$1" --version 2>/dev/null | grep -q "^${GODOT_VERSION}\."
}

make_link() {
  mkdir -p "$BIN_DIR"
  if [ -e "$LINK" ] && [ ! -L "$LINK" ]; then
    log "replacing non-symlink $LINK"
  fi
  ln -sfn "$TARGET" "$LINK"
  case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) log "note: $BIN_DIR is not on PATH" ;;
  esac
  log "godot $GODOT_VERSION -> $LINK"
}

if [ "$FORCE" = 0 ]; then
  if found="$(command -v godot 2>/dev/null)" && reports_version "$found"; then
    log "already installed: $found ($("$found" --version))"
    exit 0
  fi
  if reports_version "$TARGET"; then
    log "cached binary ok: $TARGET"
    make_link
    exit 0
  fi
fi

# --- download into a fresh temporary directory, move into place only at the end ---
mkdir -p "$CACHE_DIR"
TMP="$(mktemp -d "$CACHE_DIR/download.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

fetch() { # fetch <url> <out>
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --retry 3 --retry-delay 2 -o "$2" "$1"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$2" "$1"
  else
    log "neither curl nor wget is available"
    return 1
  fi
}

log "downloading $BASE_URL/$ZIP"
if ! fetch "$BASE_URL/$ZIP" "$TMP/$ZIP"; then
  log "download failed (no network?)"
  exit 1
fi

if fetch "$BASE_URL/SHA512-SUMS.txt" "$TMP/SHA512-SUMS.txt"; then
  if grep -F " $ZIP" "$TMP/SHA512-SUMS.txt" | grep -v '/' > "$TMP/expected.sha512" && [ -s "$TMP/expected.sha512" ]; then
    if (cd "$TMP" && sha512sum --quiet -c expected.sha512); then
      log "sha512 ok"
    else
      log "sha512 MISMATCH for $ZIP - refusing to install"
      exit 1
    fi
  else
    log "warning: $ZIP not listed in SHA512-SUMS.txt, skipping verification"
  fi
else
  log "warning: could not fetch SHA512-SUMS.txt, skipping verification"
fi

mkdir -p "$TMP/unpacked"
if command -v unzip >/dev/null 2>&1; then
  unzip -q -o "$TMP/$ZIP" -d "$TMP/unpacked"
else
  python3 -I -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "$TMP/$ZIP" "$TMP/unpacked"
fi

BIN="$TMP/unpacked/$ASSET"
if [ ! -f "$BIN" ]; then
  BIN="$(find "$TMP/unpacked" -maxdepth 2 -type f -name 'Godot_v*' | head -n 1 || true)"
fi
if [ -z "$BIN" ] || [ ! -f "$BIN" ]; then
  log "no Godot binary found inside $ZIP"
  exit 1
fi
chmod +x "$BIN"
if ! reports_version "$BIN"; then
  log "downloaded binary does not report version $GODOT_VERSION: $("$BIN" --version 2>&1 | head -n 1)"
  exit 1
fi
mv -f "$BIN" "$TARGET"
log "installed $TARGET"
make_link
