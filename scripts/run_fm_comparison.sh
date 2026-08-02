#!/usr/bin/env bash
# All-FM distillation sweep: 11 Foundation Models × ViT-S student.
# Per-model batch size based on VRAM budget. Early stop (patience=3, epochs=10).
# Small → large ordering so fast results appear first.
set -u
cd "$(dirname "$0")/.."

CONFIG="configs/multi_source_v1.json"
OUT_ROOT="outputs/fm_comparison"
EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-3}"
WORKERS="${WORKERS:-6}"
mkdir -p "$OUT_ROOT"

# Layout: name bs epoch_hint
# bs from peak mem × batch scaling, leaving headroom on 32GB RTX 5090.
FMS=(
  "phikon        128"
  "hibou-b       128"
  "conch         128"
  "phikon-v2     128"
  "uni           128"
  "hibou-l       128"
  "virchow       48"
  "virchow2      48"
  "uni2-h        48"
  "h-optimus-0   32"
  "prov-gigapath 32"
)

# Reuse existing phikon-v2 run if present
if [ -f "outputs/distill_multi_v1/best.pt" ] && [ -f "outputs/eval_multi_v1/evaluation_report.json" ]; then
  mkdir -p "$OUT_ROOT/phikon-v2"
  [ -L "$OUT_ROOT/phikon-v2/distill" ] || ln -s "$(pwd)/outputs/distill_multi_v1" "$OUT_ROOT/phikon-v2/distill"
  [ -L "$OUT_ROOT/phikon-v2/eval"    ] || ln -s "$(pwd)/outputs/eval_multi_v1"    "$OUT_ROOT/phikon-v2/eval"
  echo "[reuse] phikon-v2 → symlinked from existing distill_multi_v1/eval_multi_v1"
fi

for line in "${FMS[@]}"; do
  read -r FM BS <<< "$line"
  FM_DIR="$OUT_ROOT/$FM"
  DISTILL_DIR="$FM_DIR/distill"
  EVAL_DIR="$FM_DIR/eval"

  # Skip if evaluation already done (e.g. reused phikon-v2)
  if [ -f "$EVAL_DIR/evaluation_report.json" ]; then
    echo "[skip] $FM already has evaluation_report.json"
    continue
  fi

  mkdir -p "$FM_DIR"
  echo ""
  echo "============================================================"
  echo "[run] teacher=$FM  bs=$BS  epochs=$EPOCHS  patience=$PATIENCE"
  echo "      started $(date -Iseconds)"
  echo "============================================================"

  python3 scripts/distill_multi.py \
    --config "$CONFIG" \
    --output "$DISTILL_DIR" \
    --teacher "$FM" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --batch_size "$BS" \
    --num_workers "$WORKERS" \
    2>&1 | tee "$FM_DIR/distill.log"
  DISTILL_RC=${PIPESTATUS[0]}

  if [ "$DISTILL_RC" -ne 0 ] || [ ! -f "$DISTILL_DIR/best.pt" ]; then
    echo "[fail] $FM distill rc=$DISTILL_RC; skipping eval" | tee -a "$FM_DIR/distill.log"
    continue
  fi

  echo "[eval] $FM"
  python3 scripts/evaluate_multi.py \
    --config "$CONFIG" \
    --student_ckpt "$DISTILL_DIR/best.pt" \
    --splits "$DISTILL_DIR/splits.npz" \
    --output "$EVAL_DIR" \
    --teacher "$FM" \
    --batch_size "$BS" \
    --num_workers "$WORKERS" \
    2>&1 | tee "$FM_DIR/eval.log"

  # Incremental summary after each FM
  python3 scripts/summarize_fm_comparison.py --root "$OUT_ROOT" \
    2>&1 | tee "$OUT_ROOT/summary.log"
done

echo ""
echo "[done] sweep finished at $(date -Iseconds)"
python3 scripts/summarize_fm_comparison.py --root "$OUT_ROOT"
