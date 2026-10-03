#!/usr/bin/env bash
# Fragment growth without binding-energy screening.
#
# Usage:
#   bash scripts/05_grow_without_binding.sh [target] [--execute] [run_case options]
#
# By default the script runs the lightweight example configuration (initial
# core from examples/, 2 iterations, small pools, output under
# runs/examples/fragment_growth/without_binding/<target>). Set EXAMPLE=0 to
# use the full protocol defaults of run_case.py instead, or override the
# individual variables below:
#   INITIAL_CORE, ITERATION, RETAIN_MOLS_NUM, OUTPUT_MOLS_NUM, BUCKET_SIZE,
#   NUM_CPU, BATCH_SIZE, OUTPUT_DIR
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

PYTHON="$(resolve_python FRAG_PYTHON e2m)"
TARGET="${1:-bezcoa}"
if [ $# -gt 0 ]; then shift; fi
EXAMPLE="${EXAMPLE:-1}"

args=(--project-root "${PROJECT_ROOT}" --target "${TARGET}" --mode nonbinding)
if [ "${EXAMPLE}" = "1" ]; then
  args+=(
    --initial-core "${INITIAL_CORE:-${PROJECT_ROOT}/examples/initial_core.sdf}"
    --iteration "${ITERATION:-2}"
    --retain-mols-num "${RETAIN_MOLS_NUM:-4}"
    --output-mols-num "${OUTPUT_MOLS_NUM:-10}"
    --bucket-size "${BUCKET_SIZE:-4}"
    --num-cpu "${NUM_CPU:-2}"
    --batch-size "${BATCH_SIZE:-8}"
    --output-dir "${OUTPUT_DIR:-${PROJECT_ROOT}/runs/examples/fragment_growth/without_binding/${TARGET}}"
  )
fi

"${PYTHON}" "${PROJECT_ROOT}/src/fragment_growth/run_case.py" "${args[@]}" "$@"
