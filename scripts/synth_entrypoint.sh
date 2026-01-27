#!/usr/bin/env bash
set -euo pipefail

# synth_entrypoint.sh
#
# This entrypoint runs BlenderProc with our render script.
# It is designed so you can simply do:
#   docker compose run --rm synth
#
# You can override parameters via environment variables:
#   N_IMAGES=400 WIDTH=640 HEIGHT=480 FORCE_TEXTURE=1 DEBUG_PRINT_PATHS=1 docker compose run --rm synth

ASSETS_ROOT="${ASSETS_ROOT:-/work/models/ycb}"
OUT_DIR="${OUT_DIR:-/work/data/synth_coco}"

N_IMAGES="${N_IMAGES:-3}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"

# Flags (0/1)
FORCE_TEXTURE="${FORCE_TEXTURE:-1}"
DEBUG_PRINT_PATHS="${DEBUG_PRINT_PATHS:-1}"

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

# Show what we will run (useful for logs)
echo "[synth_entrypoint] blenderproc ${args[*]}"

# Exec BlenderProc
exec blenderproc "${args[@]}" ${EXTRA_ARGS}

