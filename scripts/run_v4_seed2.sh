#!/usr/bin/env bash
# v4 seed-2 sweep: top-N teachers × 3 students × seed=2025.
# Addresses reviewer-5 ask: TOST equivalence claims need ≥2 seeds.
#
# Output goes to outputs/v4_seed2/<fm>/<student>/ to keep the seed=42 atlas
# untouched.  Re-runnable; skips runs whose best.pt already exists.
#
# Defaults are conservative — only 5 teachers, since each run is ~3-15 hours.
# Override with: FMS="phikon uni virchow ..." bash scripts/run_v4_seed2.sh
set -u
cd "$(dirname "$0")/.."

OUT_ROOT="${OUT_ROOT:-outputs/v4_seed2}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-5}"
SEED="${SEED:-2025}"
mkdir -p "$OUT_ROOT"

STUDENTS=("vit_tiny_patch16_224" "vit_small_patch16_224" "vit_base_patch16_224")

# Top-5 teachers by primary-eval ranking (curated; revisit after first eval pass).
# Picked to span the param scale: small/mid/large/huge/ultra-huge.
DEFAULT_FMS=(
  "phikon         96"
  "uni            96"
  "hibou-l        96"
  "virchow2       32"
  "h-optimus-0    24"
)

if [ -n "${FMS:-}" ]; then
  # User override: parse "fm1 fm2 fm3" -> auto bs (24 = safe for any teacher × ViT-B)
  FMS_ARR=()
  for fm in $FMS; do FMS_ARR+=("$fm 24"); done
else
  FMS_ARR=("${DEFAULT_FMS[@]}")
fi

START_ALL=$(date -Iseconds)
echo "[v4 seed-2 sweep] starting $START_ALL  seed=$SEED"
echo "  ${#FMS_ARR[@]} FMs × ${#STUDENTS[@]} students = $((${#FMS_ARR[@]} * ${#STUDENTS[@]})) runs"

for line in "${FMS_ARR[@]}"; do
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
echo "[v4 seed-2 sweep] finished $(date -Iseconds) (started $START_ALL)"
echo "  next: re-run aggregate_atlas.py with both seed=42 and seed=2025 atlases"
echo "        and update stats_v4 to compute paired-seed variance for TOST."
