#!/usr/bin/env bash
# Training-data preparation: structural augmentation, xTB/Multiwfn electron
# density and LMDB building.
#
# Usage:
#   bash scripts/01_prepare_training_data.sh [augment|ed|lmdb|all]
#
# Defaults process the bundled example pairs (4 host/guest complexes) and a
# single xTB/Multiwfn pair. Environment variables:
#   TRAINING_SOURCE_DIR  input pairs (default examples/training_source/experimental)
#   RUN_ROOT             output root (default runs/examples/preprocessing)
#   PAIR_LIMIT           number of base pairs (default 4)
#   NUM_AUG              augmentations per pair (default 5)
#   ED_PAIR_LIMIT        pairs sent through xTB/Multiwfn (default 1)
#   LMDB_LIMIT           maximum valid LMDB entries (default 32)
#   XTB_BIN, MULTIWFN_BIN, MULTIWFN_SETTINGS, OBABEL_BIN  external programs
#   DRY_RUN=1            plan the ED stage without executing xTB/Multiwfn
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

PYTHON="$(resolve_python EDVAE_PYTHON guestgen)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

STAGE="${1:-all}"
case "${STAGE}" in
  augment|ed|lmdb|all) ;;
  -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 0 ;;
  *) echo "usage: $0 [augment|ed|lmdb|all]" >&2; exit 2 ;;
esac

TRAINING_SOURCE_DIR="${TRAINING_SOURCE_DIR:-${PROJECT_ROOT}/examples/training_source/experimental}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/examples/preprocessing}"
PAIR_LIMIT="${PAIR_LIMIT:-4}"
NUM_AUG="${NUM_AUG:-5}"
ED_PAIR_LIMIT="${ED_PAIR_LIMIT:-1}"
LMDB_LIMIT="${LMDB_LIMIT:-32}"
DRY_RUN="${DRY_RUN:-0}"

AUGMENT_OUTPUT="${AUGMENT_OUTPUT:-${RUN_ROOT}/01_augmented}"
ED_OUTPUT="${ED_OUTPUT:-${RUN_ROOT}/02_electron_density}"
PROCESSED_PKL_DIR="${PROCESSED_PKL_DIR:-${ED_OUTPUT}/processed_48}"
LMDB_OUTPUT="${LMDB_OUTPUT:-${RUN_ROOT}/03_lmdb_48}"

run_augment() {
  echo "== augmentation: ${TRAINING_SOURCE_DIR} -> ${AUGMENT_OUTPUT}"
  "${PYTHON}" -m preprocessing.augmentation \
    --input-dir "${TRAINING_SOURCE_DIR}" \
    --output "${AUGMENT_OUTPUT}" \
    --limit "${PAIR_LIMIT}" \
    --num-aug "${NUM_AUG}" \
    --rotation-step-size 5 \
    --translation-step-size 20 \
    --seed 123
}

run_ed() {
  echo "== electron density: ${AUGMENT_OUTPUT}/molecules -> ${ED_OUTPUT}"
  local ed_args=(
    --layout training
    --input "${AUGMENT_OUTPUT}/molecules"
    --output "${ED_OUTPUT}"
    --limit "${ED_PAIR_LIMIT}"
    --xtb "${XTB_BIN:-xtb}"
    --multiwfn "${MULTIWFN_BIN:-Multiwfn}"
    --obabel "${OBABEL_BIN:-obabel}"
    --xtb-input-format xyz
    --xtb-iterations 999
    --grid-spacing 0.5
    --extension 6
    --num-points 48
    --step-size 0.35
    --interpolation-backend pyvista
  )
  if [ -n "${MULTIWFN_SETTINGS:-}" ]; then
    ed_args+=(--multiwfn-settings "${MULTIWFN_SETTINGS}")
  fi
  if [ "${DRY_RUN}" = "1" ]; then
    ed_args+=(--dry-run)
  fi
  "${PYTHON}" -m preprocessing.electron_density "${ed_args[@]}"
}

run_lmdb() {
  if [ "${DRY_RUN}" = "1" ]; then
    echo "DRY_RUN=1: LMDB stage skipped because no aligned PKLs were calculated."
    return
  fi
  echo "== LMDB: ${PROCESSED_PKL_DIR} -> ${LMDB_OUTPUT}"
  "${PYTHON}" -m preprocessing.lmdb_builder \
    --input "${PROCESSED_PKL_DIR}" \
    --output "${LMDB_OUTPUT}" \
    --limit "${LMDB_LIMIT}" \
    --num-points 48 \
    --high-value-policy zero
}

case "${STAGE}" in
  augment) run_augment ;;
  ed)      run_ed ;;
  lmdb)    run_lmdb ;;
  all)     run_augment; run_ed; run_lmdb ;;
esac

echo "== done: ${STAGE}"
