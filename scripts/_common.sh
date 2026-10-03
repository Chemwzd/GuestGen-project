#!/usr/bin/env bash
# Shared helpers for the GuestGen pipeline scripts. Source this file from a
# script inside scripts/ (do not execute it directly).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export PROJECT_ROOT

_fail() {
  echo "error: $*" >&2
  exit 1
}

# resolve_python VAR_NAME ENV_NAME
#   Return the Python interpreter to use. Order of precedence:
#   1. the environment variable VAR_NAME (e.g. EDVAE_PYTHON), validated;
#   2. an existing conda environment named ENV_NAME (guestgen / e2m);
#   3. `python3` or `python` from PATH (activate the environment yourself).
resolve_python() {
  local var_name="$1"
  local env_name="$2"
  local value="${!var_name:-}"
  local candidate
  local resolved
  if [ -n "${value}" ]; then
    if [ -x "${value}" ] && [ -f "${value}" ]; then
      printf '%s' "${value}"
      return 0
    fi
    if resolved="$(command -v "${value}" 2>/dev/null)" && [ -n "${resolved}" ]; then
      printf '%s' "${resolved}"
      return 0
    fi
    _fail "${var_name}=${value} is neither an executable file nor a command on PATH"
  fi
  for candidate in \
    "${HOME}/anaconda3/envs/${env_name}/bin/python" \
    "${HOME}/miniconda3/envs/${env_name}/bin/python" \
    "${HOME}/miniforge3/envs/${env_name}/bin/python"; do
    if [ -x "${candidate}" ] && [ -f "${candidate}" ]; then
      printf '%s' "${candidate}"
      return 0
    fi
  done
  if command -v conda >/dev/null 2>&1; then
    if resolved="$(conda run -n "${env_name}" python -c 'import sys; print(sys.executable)' 2>/dev/null)" \
      && [ -n "${resolved}" ]; then
      printf '%s' "${resolved}"
      return 0
    fi
    echo "warning: conda could not resolve environment '${env_name}' (check 'conda env list'); falling back to PATH python" >&2
  fi
  if resolved="$(command -v python3 2>/dev/null)" && [ -n "${resolved}" ]; then
    printf '%s' "${resolved}"
    return 0
  fi
  if resolved="$(command -v python 2>/dev/null)" && [ -n "${resolved}" ]; then
    printf '%s' "${resolved}"
    return 0
  fi
  _fail "cannot find a Python interpreter: set ${var_name} or activate the '${env_name}' environment"
}

require_file() {
  if [ ! -f "$1" ]; then
    echo "error: required file not found: $1" >&2
    exit 1
  fi
}

# require_option_value OPTION REMAINING_ARG_COUNT
require_option_value() {
  if [ "$2" -lt 2 ]; then
    echo "error: option $1 requires a value" >&2
    exit 2
  fi
}

# require_safe_name VALUE LABEL
require_safe_name() {
  case "$1" in
    ""|.|..|-*|*/*|*\\*)
      echo "error: $2 must be a plain name without path separators: '$1'" >&2
      exit 2
      ;;
  esac
}
