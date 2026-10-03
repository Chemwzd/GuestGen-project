#!/usr/bin/env bash
# Prepare host electron-density grids with xTB + Multiwfn.
#
# Usage:
#   bash scripts/03_prepare_host_ed.sh [case ...]
#   DRY_RUN=1 bash scripts/03_prepare_host_ed.sh bezcoa
#
# Environment variables:
#   HOST_MOL_DIR    host MOL directory (default examples/host_mol)
#   HOST_ED_OUTPUT  output directory (default runs/examples/host_ed)
#   XTB_BIN, MULTIWFN_BIN, MULTIWFN_SETTINGS, OBABEL_BIN  external programs
#   DRY_RUN=1       plan the run without executing xTB/Multiwfn
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

PYTHON="$(resolve_python EDVAE_PYTHON guestgen)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

CASES=("$@")
if [ "${#CASES[@]}" -eq 0 ]; then
  CASES=(bezcoa levpuz qupyik)
fi

HOST_MOL_DIR="${HOST_MOL_DIR:-${PROJECT_ROOT}/examples/host_mol}"
HOST_ED_OUTPUT="${HOST_ED_OUTPUT:-${PROJECT_ROOT}/runs/examples/host_ed}"
DRY_RUN="${DRY_RUN:-0}"

ed_args=(
  --layout host
  --input "${HOST_MOL_DIR}"
  --output "${HOST_ED_OUTPUT}"
  --xtb "${XTB_BIN:-xtb}"
  --multiwfn "${MULTIWFN_BIN:-Multiwfn}"
  --obabel "${OBABEL_BIN:-obabel}"
  --xtb-input-format mol
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
for case_name in "${CASES[@]}"; do
  ed_args+=(--name "${case_name}")
done
if [ "${DRY_RUN}" = "1" ]; then
  ed_args+=(--dry-run)
fi

echo "== host ED: ${CASES[*]} -> ${HOST_ED_OUTPUT}"
"${PYTHON}" -m preprocessing.electron_density "${ed_args[@]}"
echo "== done"
