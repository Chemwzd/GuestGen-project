#!/usr/bin/env bash
# Summarize a fragment-growth run (validity, uniqueness, diversity, property
# distributions, drug-like fractions).
#
# Usage:
#   bash scripts/07_analyze_growth.sh [run_dir] [analyze_growth.py options]
#
# Default run directory: runs/examples/fragment_growth/without_binding/bezcoa
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

PYTHON="$(resolve_python FRAG_PYTHON e2m)"
RUN_DIR="${1:-${PROJECT_ROOT}/runs/examples/fragment_growth/without_binding/bezcoa}"
if [ $# -gt 0 ]; then shift; fi

"${PYTHON}" "${PROJECT_ROOT}/tools/analyze_growth.py" "${RUN_DIR}" "$@"
