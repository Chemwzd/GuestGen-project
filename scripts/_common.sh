#!/usr/bin/env bash
# Shared helpers for the GuestGen pipeline scripts. Source this file from a
# script inside scripts/ (do not execute it directly).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export PROJECT_ROOT

# resolve_python VAR_NAME ENV_NAME
#   Return the Python interpreter to use. Order of precedence:
#   1. the environment variable VAR_NAME (e.g. EDVAE_PYTHON),
#   2. an existing conda environment named ENV_NAME (guestgen / e2m),
#   3. `python` from PATH (activate the intended environment yourself).
resolve_python() {
  local var_name="$1"
  local env_name="$2"
  local value="${!var_name:-}"
  if [ -n "${value}" ]; then
    printf '%s' "${value}"
    return 0
  fi
  local candidate
  for candidate in \
    "${HOME}/anaconda3/envs/${env_name}/bin/python" \
    "${HOME}/miniconda3/envs/${env_name}/bin/python" \
    "${HOME}/miniforge3/envs/${env_name}/bin/python"; do
    if [ -x "${candidate}" ]; then
      printf '%s' "${candidate}"
      return 0
    fi
  done
  if command -v conda >/dev/null 2>&1; then
    local resolved
    resolved="$(conda run -n "${env_name}" python -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
    if [ -n "${resolved}" ]; then
      printf '%s' "${resolved}"
      return 0
    fi
  fi
  printf '%s' "python"
}

require_file() {
  if [ ! -e "$1" ]; then
    echo "error: required file not found: $1" >&2
    exit 1
  fi
}
