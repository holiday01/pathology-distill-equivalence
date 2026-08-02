#!/usr/bin/env bash
# v4 H-Optimus-1 add-on: 1 FM × 3 students = 3 runs.
# Brings teacher count from 12 → 13, matching the paper outline (C1).
#
# Prerequisite: HuggingFace gated access to bioptimus/H-optimus-1 must be
# granted before running. Confirm with:
#   huggingface-cli whoami && huggingface-cli download bioptimus/H-optimus-1 \
#       --include "config.json" --local-dir /tmp/hopt1_check
# (a successful config.json download confirms access).
#
# H-Optimus-1 is ~1.13B params (ViT-G/14, same arch as H-Optimus-0).
# Time budget: ~10/13/15 hr per ViT-{Ti,S,B} student on RTX 5090, FP32.
set -u
cd "$(dirname "$0")/.."

OUT_ROOT="${OUT_ROOT:-outputs/v4_full}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-5}"
SEED="${SEED:-42}"
BS="${BS:-24}"
FM="${FM:-h-optimus-1}"

mkdir -p "$OUT_ROOT/$FM"

# Pre-flight: check HF access by attempting to load the teacher (cheap config-only check).
echo "[preflight] verifying HF access to $FM ..."
python3 - <<PY
import os, sys
try:
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id="bioptimus/H-optimus-1", filename="config.json")
except Exception as e:
    print(f"[preflight FAIL] {e}", file=sys.stderr)
    print("    -> request access at https://huggingface.co/bioptimus/H-optimus-1", file=sys.stderr)
    sys.exit(2)
print("[preflight OK]")
PY
[ $? -ne 0 ] && exit 2

STUDENTS=("vit_tiny_patch16_224" "vit_small_patch16_224" "vit_base_patch16_224")

START_ALL=$(date -Iseconds)
echo "[v4 h-optimus-1 add-on] starting $START_ALL"

for STU in "${STUDENTS[@]}"; do
  STU_SHORT=$(echo "$STU" | sed 's/vit_\([a-z]*\)_patch16_224/vit-\1/')
  OUT="$OUT_ROOT/$FM/$STU_SHORT"
  if [ -f "$OUT/best.pt" ]; then
    echo "[skip] $FM $STU_SHORT already done"; continue
  fi
  mkdir -p "$OUT"
  echo ""
  echo "============================================================"
  echo "[run] $FM × $STU_SHORT  bs=$BS  seed=$SEED"
  echo "      started $(date -Iseconds)"
  echo "============================================================"
  python3 scripts/distill_v4.py \
    --config configs/multi_source_v1.json \
    --output "$OUT" \
    --teacher "$FM" \
    --student "$STU" \
    --epochs "$EPOCHS" --patience "$PATIENCE" \
    --batch_size "$BS" --num_workers 6 \
    --seed "$SEED" \
    --lam_cls 1.0 --lam_pat 0.5 \
    2>&1 | tee "$OUT/distill.log"
done

echo ""
echo "[v4 h-optimus-1 add-on] finished $(date -Iseconds) (started $START_ALL)"
echo "  next:"
echo "    bash scripts/run_v4_eval_all.sh outputs/v4_full"
echo "    FMS=h-optimus-1 bash scripts/run_seg_eval.sh"
echo "    add 'h-optimus-1' to TEACHERS in scripts/run_plism_eval.sh and re-run"
echo "    python3 scripts/aggregate_atlas.py"
