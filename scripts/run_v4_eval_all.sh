#!/usr/bin/env bash
# Sweep eval over all completed v4_full/<fm>/<student>/ runs.
# Idempotent: skips runs whose downstream_report.json already exists.
set -u
cd "$(dirname "$0")/.."

ROOT="${1:-outputs/v4_full}"
CONFIG="configs/multi_source_v1.json"

for fm_dir in "$ROOT"/*/; do
  FM=$(basename "$fm_dir")
  [ "$FM" == "telemetry" ] && continue
  for stu_dir in "$fm_dir"*/; do
    STU_SHORT=$(basename "$stu_dir")
    [ -f "$stu_dir/best.pt" ] || { echo "[wait] $FM/$STU_SHORT not yet complete"; continue; }
    [ -f "$stu_dir/downstream_report.json" ] && { echo "[skip] $FM/$STU_SHORT"; continue; }
    case "$STU_SHORT" in
      vit-tiny)  STU="vit_tiny_patch16_224"  ;;
      vit-small) STU="vit_small_patch16_224" ;;
      vit-base)  STU="vit_base_patch16_224"  ;;
      *)         STU="vit_small_patch16_224" ;;
    esac
    echo ""
    echo "[eval] $FM/$STU_SHORT  student=$STU"
    python3 scripts/evaluate_v4_downstream.py \
      --config "$CONFIG" \
      --student_ckpt "$stu_dir/best.pt" \
      --splits "$stu_dir/splits.npz" \
      --output "$stu_dir" \
      --teacher "$FM" \
      --student "$STU" \
      --batch_size 64 --num_workers 16 \
      2>&1 | tee "$stu_dir/eval.log"
  done
done
echo ""
echo "[v4 eval sweep done] $(date -Iseconds)"
