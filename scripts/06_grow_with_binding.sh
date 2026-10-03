#!/usr/bin/env bash
# Fragment growth with GFN2-xTB binding-energy screening.
#
# Usage:
#   bash scripts/06_grow_with_binding.sh [target] [--execute] [run_case options]
#
# Requires an executable xTB (set XTB_BIN or put xtb on PATH) when --execute
# is used. Like script 05, the defaults are the lightweight example
# configuration; set EXAMPLE=0 for the full protocol defaults. Overridable
# variables: INITIAL_CORE, ITERATION, RETAIN_MOLS_NUM, OUTPUT_MOLS_NUM,
# BUCKET_SIZE, NUM_CPU, BATCH_SIZE, BINDING_WORKERS, OUTPUT_DIR, SOLVENT.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

PYTHON="$(resolve_python FRAG_PYTHON e2m)"
TARGET="${1:-bezcoa}"
if [ $# -gt 0 ]; then shift; fi
EXAMPLE="${EXAMPLE:-1}"

args=(--project-root "${PROJECT_ROOT}" --target "${TARGET}" --mode binding)
if [ "${EXAMPLE}" = "1" ]; then
  args+=(
    --initial-core "${INITIAL_CORE:-${PROJECT_ROOT}/examples/initial_core.sdf}"
    --iteration "${ITERATION:-2}"
    --retain-mols-num "${RETAIN_MOLS_NUM:-2}"
    --output-mols-num "${OUTPUT_MOLS_NUM:-5}"
    --bucket-size "${BUCKET_SIZE:-2}"
    --num-cpu "${NUM_CPU:-2}"
    --batch-size "${BATCH_SIZE:-4}"
    --binding-workers "${BINDING_WORKERS:-2}"
    --output-dir "${OUTPUT_DIR:-${PROJECT_ROOT}/runs/examples/fragment_growth/with_binding/${TARGET}}"
  )
fi

"${PYTHON}" "${PROJECT_ROOT}/src/fragment_growth/run_case.py" "${args[@]}" "$@"
