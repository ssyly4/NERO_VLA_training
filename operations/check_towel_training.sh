#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/dev/workspace/nero_training
LOG_DIR="$ROOT/logs/fullflow70"

echo "host_time=$(date --iso-8601=seconds)"
echo "master_status=$(cat "$LOG_DIR/master.status" 2>/dev/null || echo NOT_STARTED)"
for variant in next_feedback_event4 command_event4; do
    echo "${variant}_status=$(cat "$LOG_DIR/$variant.status" 2>/dev/null || echo PENDING)"
    checkpoint_root=$(find "$ROOT/checkpoints/pi05_nero_towel_fullflow_70_${variant}_h24_v1" \
        -mindepth 3 -maxdepth 3 -type d -name params 2>/dev/null | sed 's,/params$,,' | sort -V || true)
    if [[ -n "$checkpoint_root" ]]; then
        echo "${variant}_checkpoints:"
        echo "$checkpoint_root"
    fi
done

if [[ -s "$LOG_DIR/final_summary.txt" ]]; then
    echo "--- final summary ---"
    cat "$LOG_DIR/final_summary.txt"
fi
echo "--- active pipeline processes ---"
docker exec cuda12_8_torch_2_9_1_core bash -lc \
    "pgrep -af '[r]un_nero_towel_fullflow70|[c]onvert_nero_bimanual|[t]rain.py' || true"
echo "--- GPU ---"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
echo "--- storage ---"
df -h /home/dev
echo "--- latest master log ---"
tail -30 "$LOG_DIR/master.log" 2>/dev/null || true
