#!/usr/bin/env bash
# Generate guest electron-density grids with the released ED-VAE checkpoint.
#
# Usage:
#   bash scripts/04_infer_guest_ed.sh [case ...]
#   INFERENCE_SEED=42 bash scripts/04_infer_guest_ed.sh bezcoa levpuz qupyik
#
# Environment variables:
#   INPUT_DIR     host ED directory (default examples/host_ed)
#   OUTPUT_DIR    output directory (default runs/examples/guest_ed)
#   CHECKPOINT    ED-VAE checkpoint (default models/checkpoints/guestgen_step_5000.pt)
#   INFERENCE_SEED          optional RNG seed
#   INFERENCE_DETERMINISTIC 1 (default when a seed is set) requests deterministic kernels
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

PYTHON="$(resolve_python EDVAE_PYTHON guestgen)"

INPUT_DIR="${INPUT_DIR:-${PROJECT_ROOT}/examples/host_ed}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/examples/guest_ed}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/models/checkpoints/guestgen_step_5000.pt}"
INFERENCE_SEED="${INFERENCE_SEED-}"
INFERENCE_DETERMINISTIC="${INFERENCE_DETERMINISTIC:-1}"
require_file "${CHECKPOINT}"

SEED_ARGS=()
if [ -n "${INFERENCE_SEED}" ]; then
  SEED_ARGS=(--seed "${INFERENCE_SEED}")
  if [ "${INFERENCE_DETERMINISTIC}" = "1" ]; then
    SEED_ARGS+=(--deterministic)
  fi
fi

TARGET_ARGS=()
if [ $# -gt 0 ]; then
  TARGET_ARGS=(--targets "$@")
fi

cd "${PROJECT_ROOT}/src/edvae"
echo "== guest ED inference: ${INPUT_DIR} -> ${OUTPUT_DIR}"
"${PYTHON}" infer.py \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  "${SEED_ARGS[@]}" \
  "${TARGET_ARGS[@]}"
echo "== done"
