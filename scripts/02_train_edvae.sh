#!/usr/bin/env bash
# ED-VAE training.
#
# Usage:
#   bash scripts/02_train_edvae.sh [options] [-- extra train.py arguments]
#
# Two presets are provided:
#   --smoke (default)  5 steps on the bundled example LMDB subset
#   --full             the full training protocol (5,000 steps, batch 32, 32 CPU
#                      workers, warm-up 200, cosine T_max 3800, val/log 50/10);
#                      combine with --data-dir pointing at the full LMDB
#
# Options: --data-dir DIR, --max-steps N, --batch-size N, --num-cpus N,
#          --num-warmup-steps N, --t-max N, --val-interval N, --log-interval N,
#          --name NAME, --monitor-backend shell|swanlab
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

PYTHON="$(resolve_python EDVAE_PYTHON guestgen)"

MODE="smoke"
DATA_DIR=""; MAX_STEPS=""; BATCH_SIZE=""; NUM_CPUS=""; WARMUP=""; T_MAX=""
VAL_INTERVAL=""; LOG_INTERVAL=""; NAME=""; MONITOR="shell"; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --full) MODE="full"; shift ;;
    --smoke) MODE="smoke"; shift ;;
    --data-dir) require_option_value "$1" $#; DATA_DIR="$2"; shift 2 ;;
    --max-steps) require_option_value "$1" $#; MAX_STEPS="$2"; shift 2 ;;
    --batch-size) require_option_value "$1" $#; BATCH_SIZE="$2"; shift 2 ;;
    --num-cpus) require_option_value "$1" $#; NUM_CPUS="$2"; shift 2 ;;
    --num-warmup-steps) require_option_value "$1" $#; WARMUP="$2"; shift 2 ;;
    --t-max) require_option_value "$1" $#; T_MAX="$2"; shift 2 ;;
    --val-interval) require_option_value "$1" $#; VAL_INTERVAL="$2"; shift 2 ;;
    --log-interval) require_option_value "$1" $#; LOG_INTERVAL="$2"; shift 2 ;;
    --name) require_option_value "$1" $#; NAME="$2"; shift 2 ;;
    --monitor-backend) require_option_value "$1" $#; MONITOR="$2"; shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

if [ "${MODE}" = "full" ]; then
  MAX_STEPS="${MAX_STEPS:-5000}"
  BATCH_SIZE="${BATCH_SIZE:-32}"
  NUM_CPUS="${NUM_CPUS:-32}"
  WARMUP="${WARMUP:-200}"
  T_MAX="${T_MAX:-3800}"
  VAL_INTERVAL="${VAL_INTERVAL:-50}"
  LOG_INTERVAL="${LOG_INTERVAL:-10}"
  NAME="${NAME:-train_full}"
else
  MAX_STEPS="${MAX_STEPS:-5}"
  BATCH_SIZE="${BATCH_SIZE:-2}"
  NUM_CPUS="${NUM_CPUS:-2}"
  WARMUP="${WARMUP:-2}"
  T_MAX="${T_MAX:-4}"
  VAL_INTERVAL="${VAL_INTERVAL:-5}"
  LOG_INTERVAL="${LOG_INTERVAL:-1}"
  NAME="${NAME:-train_example}"
fi

check_positive_int() {
  local label="$1" value="$2"
  case "${value}" in
    ""|*[!0-9]*) echo "error: ${label} must be a positive integer (got '${value}')" >&2; exit 2 ;;
  esac
  if [ "${value}" -le 0 ]; then
    echo "error: ${label} must be a positive integer (got '${value}')" >&2
    exit 2
  fi
}
check_nonnegative_int() {
  local label="$1" value="$2"
  case "${value}" in
    ""|*[!0-9]*) echo "error: ${label} must be a non-negative integer (got '${value}')" >&2; exit 2 ;;
  esac
}
check_positive_int "--max-steps" "${MAX_STEPS}"
check_positive_int "--batch-size" "${BATCH_SIZE}"
check_positive_int "--t-max" "${T_MAX}"
check_positive_int "--val-interval" "${VAL_INTERVAL}"
check_positive_int "--log-interval" "${LOG_INTERVAL}"
check_nonnegative_int "--num-cpus" "${NUM_CPUS}"
check_nonnegative_int "--num-warmup-steps" "${WARMUP}"
require_safe_name "${NAME}" "--name"

DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/examples/training_lmdb_subset}"
CKPT_DIR="${PROJECT_ROOT}/runs/training/${NAME}/checkpoints"
LOG_DIR="${PROJECT_ROOT}/runs/training/${NAME}/logs"
require_file "${DATA_DIR}/data.mdb"

echo "== ED-VAE training (${MODE}): ${DATA_DIR}"
echo "   steps=${MAX_STEPS} batch=${BATCH_SIZE} workers=${NUM_CPUS} ckpt=${CKPT_DIR}"

cd "${PROJECT_ROOT}/src/edvae"
"${PYTHON}" train.py \
  --data_dir "${DATA_DIR}" \
  --ckpt_dir "${CKPT_DIR}" \
  --log_dir "${LOG_DIR}" \
  --max_channels 128 \
  --latent_dim 128 \
  --dropout_ratio 0.3 \
  --optimizer Adam \
  --lr 0.0003 \
  --start_lr 0.00003 \
  --loss_beta 0.00001 \
  --random_seed 42 \
  --num_warmup_steps "${WARMUP}" \
  --t_max "${T_MAX}" \
  --max_steps "${MAX_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_cpus "${NUM_CPUS}" \
  --val_interval "${VAL_INTERVAL}" \
  --log_interval "${LOG_INTERVAL}" \
  --monitor_backend "${MONITOR}" \
  "${EXTRA[@]}"

echo "== done: checkpoints in ${CKPT_DIR}"
