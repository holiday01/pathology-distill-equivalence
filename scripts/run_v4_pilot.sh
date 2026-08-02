#!/usr/bin/env bash
# v4 Stage-1 pilot: 3 teachers × 30 epoch × minimal v4 recipe (L_CLS + L_PAT)
# Goal: measure paired ρ(teacher_auroc, student_auroc) on C16 + PAIP
#       vs v0 hybrid MSE loss for sanity; decide n regime
# Note: H-Optimus-1 gated → substitute with H-Optimus-0 (same 1.1B ViT-G arch)
set -u
cd "$(dirname "$0")/.."

OUT_ROOT="outputs/v4_pilot"
EPOCHS=30
PATIENCE=5
mkdir -p "$OUT_ROOT"

# fm bs (conservative for VRAM)
FMS=(
  "phikon        128"
  "uni           128"
  "h-optimus-0    32"
)

for line in "${FMS[@]}"; do
  read -r FM BS <<< "$line"
  OUT="$OUT_ROOT/$FM"
  if [ -f "$OUT/best.pt" ]; then
    echo "[skip] $FM already done"; continue
  fi
  mkdir -p "$OUT"
  echo ""
  echo "============================================================"
  echo "[pilot] $FM bs=$BS epochs=$EPOCHS patience=$PATIENCE"
  echo "        started $(date -Iseconds)"
  echo "============================================================"
  python3 scripts/distill_v4.py \
    --config configs/multi_source_v1.json \
    --output "$OUT" \
    --teacher "$FM" \
    --epochs "$EPOCHS" --patience "$PATIENCE" \
    --batch_size "$BS" --num_workers 6 \
    --lam_cls 1.0 --lam_pat 0.5 \
    2>&1 | tee "$OUT/distill.log"
done

echo ""
echo "[pilot done] $(date -Iseconds)"
