#!/usr/bin/env bash
set -euo pipefail

# synth_entrypoint.sh
#
# This entrypoint runs BlenderProc with our render script.
#   docker compose run --rm synth
#
# Override via env:
#   N_IMAGES=400 WIDTH=640 HEIGHT=480 docker compose run --rm synth
#   USE_CC_TEXTURES=1 CC_TEXTURES_DIR=/work/resources/cctextures docker compose run --rm synth

ASSETS_ROOT="${ASSETS_ROOT:-/work/models/ycb}"
OUT_DIR="${OUT_DIR:-/work/data/synth_coco}"

N_IMAGES="${N_IMAGES:-3}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"

# Flags (0/1)
FORCE_TEXTURE="${FORCE_TEXTURE:-1}"
DEBUG_PRINT_PATHS="${DEBUG_PRINT_PATHS:-1}"

# cc_textures (PBR) background
USE_CC_TEXTURES="${USE_CC_TEXTURES:-1}"
CC_TEXTURES_DIR="${CC_TEXTURES_DIR:-/work/resources/cctextures}"

# Optional extra args
EXTRA_ARGS="${EXTRA_ARGS:-}"

args=(
  run /work/scripts/render_ycb2_coco.py
  --assets_root "${ASSETS_ROOT}"
  --out_dir "${OUT_DIR}"
  --n_images "${N_IMAGES}"
  --width "${WIDTH}"
  --height "${HEIGHT}"
)

if [[ "${FORCE_TEXTURE}" == "1" ]]; then
  args+=( --force_texture )
fi

if [[ "${DEBUG_PRINT_PATHS}" == "1" ]]; then
  args+=( --debug_print_paths )
fi

if [[ "${USE_CC_TEXTURES}" == "1" ]]; then
  if [[ ! -d "${CC_TEXTURES_DIR}" ]]; then
    echo "[synth_entrypoint] ERROR: CC_TEXTURES_DIR not found: ${CC_TEXTURES_DIR}" >&2
    echo "[synth_entrypoint] Hint: run 'docker compose run --rm textures' first (or download cc_textures)." >&2
    exit 2
  fi
  # empty dir check (non-fatal but useful)
  if [[ -z "$(ls -A "${CC_TEXTURES_DIR}" 2>/dev/null || true)" ]]; then
    echo "[synth_entrypoint] WARNING: CC_TEXTURES_DIR is empty: ${CC_TEXTURES_DIR}" >&2
    echo "[synth_entrypoint] Hint: run 'docker compose run --rm textures' to download cc_textures." >&2
  fi
  args+=( --use_cc_textures --cc_textures_dir "${CC_TEXTURES_DIR}" )
fi

# Show what we will run (useful for logs)
echo "[synth_entrypoint] blenderproc ${args[*]} ${EXTRA_ARGS}"

# Exec BlenderProc
# NOTE: EXTRA_ARGS may contain multiple tokens, so keep it unquoted at the end.
exec blenderproc "${args[@]}" ${EXTRA_ARGS}
