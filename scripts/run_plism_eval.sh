#!/usr/bin/env bash
# PLISM robustness eval sweep for all 13 v4 teacher FMs.
#
# For each teacher, extract features once on all 91 (stain × scanner) H5 files,
# then compute cross-scanner / cross-staining / both top-K retrieval.
# Student checkpoints are handled separately (custom extractor wrapper).
#
# Prereqs: plismbench installed (CLI in ~/.local/bin/plismbench), PLISM
# dataset downloaded to $PLISM_DIR (default /path/to/wsi_datasets/plism).
set -u

cd "$(dirname "$0")/.."

PLISM_DIR="${PLISM_DIR:-/path/to/wsi_datasets/plism}"
OUT_ROOT="${OUT_ROOT:-outputs/plism}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WORKERS="${WORKERS:-4}"

# Route torch shared tensors through /dev/shm (file_system) instead of fd
# table — avoids `unable to allocate shared memory(shm) for file </torch_*>`.
# sitecustomize.py is auto-imported by Python's site module on every startup
# (works for `python script.py`, unlike PYTHONSTARTUP which is interactive-only).
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)/_pyhooks${PYTHONPATH:+:$PYTHONPATH}"
# Force CUDA 13 JIT/NVRTC libs ahead of the older .so.12 copies that torch and
# friends pull in. Without LD_PRELOAD the python process loads libnvJitLink.so.12
# first, which can't compile PTX for sm_120 (Blackwell / RTX 5090) — eval then
# dies with CUDA_ERROR_JIT_COMPILER_NOT_FOUND. LD_LIBRARY_PATH covers the
# secondary lookup of libnvrtc-builtins.so.13.0.
CU13_LIB=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib
export LD_PRELOAD="$CU13_LIB/libnvJitLink.so.13 $CU13_LIB/libnvrtc.so.13${LD_PRELOAD:+ $LD_PRELOAD}"
export LD_LIBRARY_PATH="$CU13_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Lift soft fd cap (kernel hard cap is already 1B+).
ulimit -n 1048576 2>/dev/null || true

mkdir -p "$OUT_ROOT/features" "$OUT_ROOT/results"

# v4 teachers → plismbench extractor names (13 total; h-optimus-1 gated)
TEACHERS=(
  "phikon"         # ViT-B   ~86M
  "phikonv2"       # ViT-L  ~303M
  "hibou_base"     # ViT-B
  "hibou_large"    # ViT-L
  "conch"          # ViT-B
  "uni"            # ViT-L
  "uni2h"          # ViT-H
  "virchow"        # ViT-H
  "virchow2"       # ViT-H
  "hoptimus0"      # ViT-G  1.1B
  "provgigapath"   # ViT-G
  "midnight_12k"   # ViT-G
  "gpfm"           # distilled multi-teacher
)

run_extract() {
  local ex="$1"
  local feat_dir="$OUT_ROOT/features"
  if [ -d "$feat_dir/$ex" ] && [ -f "$feat_dir/$ex/.done" ]; then
    echo "[skip] $ex (features already extracted)"
    return 0
  fi
  echo ""
  echo "============================================================"
  echo "[extract] $ex  bs=$BATCH_SIZE  started $(date -Iseconds)"
  echo "============================================================"
  ~/.local/bin/plismbench extract \
    --extractor "$ex" \
    --export-dir "$feat_dir" \
    --download-dir "$PLISM_DIR" \
    --device 0 \
    --batch-size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    2>&1 | tee "$OUT_ROOT/features/${ex}.log"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" = "0" ]; then
    touch "$feat_dir/$ex/.done"
    echo "[done] $ex"
  else
    echo "[fail] $ex (rc=$rc)"
  fi
}

START=$(date -Iseconds)
echo "[plism sweep] start $START  ${#TEACHERS[@]} teachers"
for t in "${TEACHERS[@]}"; do
  run_extract "$t"
done
echo "[plism sweep] extraction done $(date -Iseconds)  (started $START)"

# Per-extractor evaluate (CLI takes one extractor at a time)
echo ""
echo "[evaluate] computing retrieval metrics per extractor"
for t in "${TEACHERS[@]}"; do
  [ -f "$OUT_ROOT/features/$t/.done" ] || continue
  echo "---- evaluate: $t ----"
  ~/.local/bin/plismbench evaluate \
    --extractor "$t" \
    --features-dir "$OUT_ROOT/features" \
    --metrics-dir "$OUT_ROOT/results" \
    --device gpu \
    2>&1 | tee "$OUT_ROOT/results/${t}.log" || true
done

echo ""
echo "[plism sweep] FINISHED $(date -Iseconds)"
echo "Results at: $OUT_ROOT/results/"
