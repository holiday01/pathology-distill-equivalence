#!/usr/bin/env bash
# Hibou re-fit with the per-teacher normalisation fix (research/hibou_fix.md).
#
# The seed=42 atlas trained hibou-B/L under ImageNet normalisation, which is
# wrong for hibou and held the students near zero (best_val ~0.02). The fix in
# distill_wsi_model.py (TeacherModel.forward converts to hibou stats for hibou
# teachers only) is now in place. This retrains the 6 hibou runs at seed=42 so
# they are directly comparable to the other 30 atlas runs.
#
# Output: outputs/v4_hiboufix/<fm>/<student>/  (does NOT overwrite the suspect
# seed=42 runs until a human confirms best_val is healthy, i.e. >0.2).
# Re-runnable; skips runs whose best.pt already exists.
#
# Run AFTER the seed2 sweep finishes (one GPU). Acceptance check after:
#   for f in outputs/v4_hiboufix/*/*/training_log.json; do
#     python3 -c "import json,sys;d=json.load(open('$f'));print('$f', d.get('best_val'))"; done
# Healthy = best_val in the 0.1-0.3 band like other teachers (was ~0.002-0.02).
set -u
cd "$(dirname "$0")/.."

OUT_ROOT="${OUT_ROOT:-outputs/v4_hiboufix}"
EPOCHS="${EPOCHS:-30}"; PATIENCE="${PATIENCE:-5}"; SEED="${SEED:-42}"
mkdir -p "$OUT_ROOT"

STUDENTS=("vit_tiny_patch16_224" "vit_small_patch16_224" "vit_base_patch16_224")
FMS_ARR=("hibou-b 96" "hibou-l 96")

echo "[hibou re-fit] start $(date -Iseconds) seed=$SEED  (normalisation fix)"
for line in "${FMS_ARR[@]}"; do
  read -r FM BS <<< "$line"
  for STU in "${STUDENTS[@]}"; do
    STU_SHORT=$(echo "$STU" | sed 's/vit_\([a-z]*\)_patch16_224/vit-\1/')
    OUT="$OUT_ROOT/$FM/$STU_SHORT"
    [ -f "$OUT/best.pt" ] && { echo "[skip] $FM $STU_SHORT done"; continue; }
    mkdir -p "$OUT"
    echo ""; echo "[run] $FM x $STU_SHORT bs=$BS seed=$SEED  $(date -Iseconds)"
    python3 scripts/distill_v4.py \
      --config configs/multi_source_v1.json \
      --output "$OUT" --teacher "$FM" --student "$STU" \
      --epochs "$EPOCHS" --patience "$PATIENCE" \
      --batch_size "$BS" --num_workers 6 --seed "$SEED" \
      --lam_cls 1.0 --lam_pat 0.5 \
      2>&1 | tee "$OUT/distill.log"
  done
done
echo ""; echo "[hibou re-fit] done $(date -Iseconds)"
echo "  next: confirm best_val healthy (>0.2), then copy into outputs/v4_full/,"
echo "        re-run eval_c16_equivalence + downstream, and lift the suspect flag."
