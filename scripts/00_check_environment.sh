#!/usr/bin/env bash
# Verify the two Python environments, the external programs and the released
# model/data assets. Non-fatal checks print [WARN]; only missing core Python
# dependencies abort the script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

EDVAE_PYTHON="$(resolve_python EDVAE_PYTHON guestgen)"
FRAG_PYTHON="$(resolve_python FRAG_PYTHON e2m)"

echo "Project: ${PROJECT_ROOT}"
echo "ED-VAE python:   ${EDVAE_PYTHON}"
echo "fragment python: ${FRAG_PYTHON}"
echo "Host: $(hostname)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
fi

echo
echo "-- ED-VAE environment (guestgen) --"
"${EDVAE_PYTHON}" - <<'PY'
import sys
import torch, numpy, scipy, rdkit, lmdb
print(f"python {sys.version.split()[0]}, torch {torch.__version__.split('+')[0]}, "
      f"numpy {numpy.__version__}, scipy {scipy.__version__}, rdkit {rdkit.__version__}, "
      f"lmdb {lmdb.__version__}, cuda available: {torch.cuda.is_available()}")
try:
    import pyvista
    print(f"pyvista {pyvista.__version__}")
except ImportError:
    print("[WARN] pyvista not installed (required for ED interpolation)")
try:
    from sugar.molecule import HostMolecule  # noqa: F401
    print("sugar: available")
except ImportError:
    print("[WARN] sugar not installed (use --recenter-backend native as fallback)")
PY

echo
echo "-- fragment-growth environment (e2m) --"
"${FRAG_PYTHON}" - <<'PY'
import sys
import torch, numpy, scipy, rdkit
import torch_geometric, sklearn
print(f"python {sys.version.split()[0]}, torch {torch.__version__.split('+')[0]} "
      f"(cuda {torch.version.cuda}), numpy {numpy.__version__}, scipy {scipy.__version__}, "
      f"rdkit {rdkit.__version__}, torch_geometric {torch_geometric.__version__}, "
      f"scikit-learn {sklearn.__version__}, cuda available: {torch.cuda.is_available()}")
PY

echo
echo "-- external programs --"
check_program() {
  local label="$1" env_name="$2" candidates="$3"
  local value="${!env_name:-}"
  if [ -n "${value}" ]; then
    if [ -x "${value}" ] && [ -f "${value}" ]; then
      echo "[OK] ${label}: ${value}"
    else
      echo "[WARN] ${label}: ${env_name}=${value} is not an executable file"
    fi
    return
  fi
  local command_name
  for command_name in ${candidates}; do
    if command -v "${command_name}" >/dev/null 2>&1; then
      echo "[OK] ${label}: $(command -v "${command_name}")"
      return
    fi
  done
  echo "[WARN] ${label} not found; set ${env_name} if you run steps that need it"
}
check_program "xTB" "XTB_BIN" "xtb"
check_program "Multiwfn" "MULTIWFN_BIN" "Multiwfn multiwfn"
check_program "Open Babel" "OBABEL_BIN" "obabel"

echo
echo "-- released assets --"
check_sha() {
  local relative="$1" expected="$2"
  local path="${PROJECT_ROOT}/${relative}"
  if [ ! -f "${path}" ]; then
    echo "[MISSING] ${relative}"
    return 0
  fi
  if ! command -v sha256sum >/dev/null 2>&1; then
    echo "[WARN] sha256sum is unavailable; cannot verify ${relative}"
    return 0
  fi
  local actual
  if ! actual="$(sha256sum "${path}" 2>/dev/null | awk '{print $1}')" || [ -z "${actual}" ]; then
    echo "[WARN] could not hash ${relative}"
    return 0
  fi
  if [ "${actual}" = "${expected}" ]; then
    echo "[OK] ${relative}"
  else
    echo "[WARN] ${relative}: sha256 ${actual} != released ${expected}"
  fi
}
check_sha "models/checkpoints/guestgen_step_5000.pt" "2579ce1e4b06ef84701a0c787995ba672c695e5f4bbf3f1726f996bc3310eb81"
check_sha "src/fragment_growth/weights/GPPM.pth" "d8d2a31c1005b8d4b003d4cd3572942fb6588d3344d905cee6235abcaf7d7293"
check_sha "src/fragment_growth/weights/TAPM.pth" "a1a670e3de8bd8723f6a3d1068a7c1b69c6a6ff84552f7fc3b80364536a03163"
check_sha "src/fragment_growth/frag_lib/Cores.sdf" "7412e8c85b63a010ea23070c8f5b713340110b1040353cacd2fcf33a49782fe4"
check_sha "src/fragment_growth/frag_lib/Frags.sdf" "d381d1a757642f56026194f117dcef06cade98fec48b116673f26afccae3f573"

for required in \
  "examples/training_lmdb_subset/data.mdb" \
  "examples/initial_core.sdf" \
  "examples/host_mol/bezcoa.mol" \
  "examples/host_ed/bezcoa/property_aligned_scale_0_1.pkl" \
  "examples/guest_ed/bezcoa.npy"; do
  if [ -f "${PROJECT_ROOT}/${required}" ]; then
    echo "[OK] ${required}"
  else
    echo "[WARN] missing example asset: ${required}"
  fi
done

echo
echo "Environment check finished."
