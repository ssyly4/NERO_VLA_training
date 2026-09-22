#!/usr/bin/env bash
set -Eeuo pipefail

# 在 .154 服务器现有的 OpenPI 训练容器内运行，禁止覆盖已有训练任务。
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_ROOT=/home/dev/workspace/nero_training
OPENPI_ROOT=/home/dev/workspace/openpi_deploy/repos/openpi
PYTHON="$OPENPI_ROOT/.venv/bin/python"
LAUNCHER="$SCRIPT_DIR/train_right_bottle_pi05_openpi.py"
DATASET="$TRAIN_ROOT/lerobot_v21/local/nero_bottle_into_box_right_60_command_sft_v1"
CACHE=/home/dev/.cache/huggingface/lerobot/local/nero_bottle_into_box_right_60_command_sft_v1
CONFIG=pi05_nero_bottle_box_right60_command_eff4_v1
EXP=lora_opt30000_bottle_box_right60_command_h16_eff4_v1
STATS="$TRAIN_ROOT/assets/$CONFIG/local/nero_bottle_into_box_right_60_command_sft_v1/norm_stats.json"
CHECKPOINT="$TRAIN_ROOT/checkpoints/$CONFIG/$EXP"
LOG="$TRAIN_ROOT/logs/nero_bottle_box_right60_command_eff4_30k.log"
STATUS="$TRAIN_ROOT/logs/nero_bottle_box_right60_command_eff4_30k.status"
LOCK="$TRAIN_ROOT/nero_bottle_box_right60_command_eff4_30k.lock"

mkdir -p "$TRAIN_ROOT/logs"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "Another bottle training run holds $LOCK" >&2
  exit 1
fi
exec >>"$LOG" 2>&1

set_status() { printf '%s %s\n' "$1" "$(date --iso-8601=seconds)" >"$STATUS"; }
fail() {
  set_status FAILED
  echo "$1" >&2
  exit 1
}
failed() {
  code=$?
  set_status "FAILED(exit=$code,line=$1)"
  echo "Pipeline failed at line $1 with exit code $code"
  exit "$code"
}
trap 'failed $LINENO' ERR

echo "=== Bottle-box pi0.5 LoRA pipeline started $(date --iso-8601=seconds) ==="
set_status WAITING_FOR_CONVERSION
for _ in $(seq 1 180); do
  [[ -f "$DATASET/meta/nero_pi05_sft_manifest.json" ]] && break
  pgrep -f '[p]ython convert_right_pico_to_openpi_v21.py' >/dev/null || fail "Conversion stopped before completion"
  sleep 10
done
[[ -f "$DATASET/meta/nero_pi05_sft_manifest.json" ]] || fail "Timed out waiting for v2.1 conversion"
set_status VALIDATING
[[ -d "$DATASET" ]] || fail "Missing v2.1 dataset: $DATASET"
[[ ! -e "$CHECKPOINT" ]] || fail "Refusing existing checkpoint: $CHECKPOINT"
mkdir -p "$(dirname "$CACHE")"
if [[ -e "$CACHE" || -L "$CACHE" ]]; then
  [[ "$(readlink -f "$CACHE")" == "$DATASET" ]] || fail "Cache path already points elsewhere: $CACHE"
else
  ln -s "$DATASET" "$CACHE"
fi

cd "$OPENPI_ROOT"
export PYTHONPATH="$OPENPI_ROOT:$OPENPI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" "$LAUNCHER" --stage check
if [[ ! -s "$STATS" ]]; then
  set_status NORM_STATS
  JAX_PLATFORMS=cpu "$PYTHON" "$LAUNCHER" --stage stats
else
  echo "Using existing stats: $STATS"
fi

set_status TRAINING
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.78
"$PYTHON" "$LAUNCHER" --stage train
[[ -d "$CHECKPOINT/119999/params" ]] || fail "Final checkpoint missing: $CHECKPOINT/119999/params"
set_status COMPLETE
echo "Training complete: $CHECKPOINT/119999"
