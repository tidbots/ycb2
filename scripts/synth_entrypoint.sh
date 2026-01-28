#!/usr/bin/env bash
set -euo pipefail

# synth_entrypoint.sh
#
# docker compose run --rm synth
# Env overrides:
#   N_IMAGES=400 WIDTH=640 HEIGHT=480 SEED=0 \
#   USE_CC_TEXTURES=1 CC_TEXTURES_DIR=/work/assets/cc_textures \
#   CC_USED_ASSETS="wood,tiles,concrete,metal" \
#   PREFER_OPTIX=1 FORCE_TEXTURE=1 DEBUG_PRINT_PATHS=0 \
#   docker compose run --rm synth

ASSETS_ROOT="${ASSETS_ROOT:-/work/models/ycb}"
OUT_DIR="${OUT_DIR:-/work/data/synth_coco}"

N_IMAGES="${N_IMAGES:-3}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
SEED="${SEED:-0}"

# Flags (0/1)
FORCE_TEXTURE="${FORCE_TEXTURE:-1}"
DEBUG_PRINT_PATHS="${DEBUG_PRINT_PATHS:-0}"
PREFER_OPTIX="${PREFER_OPTIX:-1}"

# cc_textures (PBR)
USE_CC_TEXTURES="${USE_CC_TEXTURES:-1}"
CC_TEXTURES_DIR="${CC_TEXTURES_DIR:-/work/assets/cc_textures}"
CC_USED_ASSETS="${CC_USED_ASSETS:-}"   # comma-separated, e.g. "wood,tiles,concrete"
USE_ALL_CC_MATERIALS="${USE_ALL_CC_MATERIALS:-0}"  # 1: load everything (heavy)

# Optional extra args
EXTRA_ARGS="${EXTRA_ARGS:-}"

args=(
  run /work/scripts/render_ycb2_coco.py
  --assets_root "${ASSETS_ROOT}"
  --out_dir "${OUT_DIR}"
  --n_images "${N_IMAGES}"
  --width "${WIDTH}"
  --height "${HEIGHT}"
  --seed "${SEED}"
)

if [[ "${PREFER_OPTIX}" == "1" ]]; then
  args+=( --prefer_optix )
fi

if [[ "${FORCE_TEXTURE}" == "1" ]]; then
  args+=( --force_texture )
fi

if [[ "${DEBUG_PRINT_PATHS}" == "1" ]]; then
  args+=( --debug_print_paths )
fi

if [[ "${USE_CC_TEXTURES}" == "1" ]]; then
  # cc_textures は dir を渡すだけで有効化
if [[ -n "${CC_TEXTURES_DIR:-}" ]]; then
  args+=( --cc_textures_dir "${CC_TEXTURES_DIR}" )
fi
  if [[ -n "${CC_USED_ASSETS}" ]]; then
    args+=( --cc_used_assets "${CC_USED_ASSETS}" )
  fi
  if [[ "${USE_ALL_CC_MATERIALS}" == "1" ]]; then
    args+=( --use_all_cc_materials )
  fi
fi

echo "[synth_entrypoint] blenderproc ${args[*]}"
exec blenderproc "${args[@]}" ${EXTRA_ARGS}
