#!/usr/bin/env bash
# v4 full sweep: 12 FMs × 3 students × 1 seed = 36 distillation runs.
# Minimal v4 recipe (L_CLS cosine L2 + L_PAT cosine); MGD+DINO deferred to Stage 2.
# H-Optimus-1 gated — run after access granted.
set -u
cd "$(dirname "$0")/.."

OUT_ROOT="outputs/v4_full"
EPOCHS=30
PATIENCE=5
SEED=42
mkdir -p "$OUT_ROOT"

# Students
STUDENTS=("vit_tiny_patch16_224" "vit_small_patch16_224" "vit_base_patch16_224")

# FMs sorted small-to-large; batch sizes tuned for ViT-B student + teacher VRAM
# fm bs
FMS=(
  "phikon         96"
  "hibou-b        96"
  "conch          96"
  "phikon-v2      96"
  "uni            96"
  "hibou-l        96"
  "virchow        32"
  "virchow2       32"
  "uni2-h         32"
  "h-optimus-0    24"
  "prov-gigapath  24"
  "midnight       24"
)

START_ALL=$(date -Iseconds)
echo "[v4 full sweep] starting $START_ALL"
echo "  12 FMs × 3 students = 36 runs"

for line in "${FMS[@]}"; do
  read -r FM BS <<< "$line"
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
done

echo ""
echo "[v4 full sweep] finished $(date -Iseconds) (started $START_ALL)"
python3 scripts/make_paper_metrics.py --root "$OUT_ROOT" \
  --telemetry "$OUT_ROOT/../v4_full/telemetry/gpu_timeseries.csv" 2>&1 | tail -3
