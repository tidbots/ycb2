#!/bin/bash
set -euo pipefail

BASE_CONFIG="${BASE_CONFIG:-/work/configs/synth_config.json}"
DR_CONFIG="${DR_CONFIG:-/work/configs/synth_dr_config.json}"
PATHS_CONFIG="${PATHS_CONFIG:-/work/configs/synth_dr_paths.json}"
BG_ROOT="${BG_ROOT:-/work/assets/backgrounds}"

SPLIT="${SPLIT:-val}"
NUM_IMAGES="${NUM_IMAGES:-30}"
SEED="${SEED:-1}"
OUT_DIR="${OUT_DIR:-/work/out/coco_bg_${SPLIT}}"

echo "[INFO] BASE_CONFIG=${BASE_CONFIG}"
echo "[INFO] DR_CONFIG=${DR_CONFIG}"
echo "[INFO] PATHS_CONFIG=${PATHS_CONFIG}"
echo "[INFO] BG_ROOT=${BG_ROOT}"
echo "[INFO] SPLIT=${SPLIT} NUM_IMAGES=${NUM_IMAGES} SEED=${SEED}"
echo "[INFO] OUT_DIR=${OUT_DIR}"

test -f "${BASE_CONFIG}"  || { echo "[FATAL] missing BASE_CONFIG: ${BASE_CONFIG}"; exit 2; }
test -f "${DR_CONFIG}"    || { echo "[FATAL] missing DR_CONFIG: ${DR_CONFIG}"; exit 2; }
test -f "${PATHS_CONFIG}" || { echo "[FATAL] missing PATHS_CONFIG: ${PATHS_CONFIG}"; exit 2; }

mkdir -p "${OUT_DIR}"

blenderproc run /work/scripts/render_ycb_coco_pbr_modes_bg.py -- \
  --base_config "${BASE_CONFIG}" \
  --dr_config "${DR_CONFIG}" \
  --paths_config "${PATHS_CONFIG}" \
  --bg_root "${BG_ROOT}" \
  --split "${SPLIT}" \
  --num_images "${NUM_IMAGES}" \
  --output_dir "${OUT_DIR}" \
  --seed "${SEED}" \
  --overwrite

