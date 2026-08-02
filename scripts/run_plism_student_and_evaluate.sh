#!/usr/bin/env bash
# Run student-side PLISM extraction and evaluate all 36 student models.
# Run AFTER patch_center_probe.py completes.
# Usage: bash scripts/run_plism_student_and_evaluate.sh
set -u
cd "$(dirname "$0")/.."

PLISM_DIR="${PLISM_DIR:-/path/to/wsi_datasets/plism}"
FEAT_ROOT="outputs/plism/features"
METRICS_ROOT="outputs/plism/results"
V4_ROOT="outputs/v4_full"
LOG_DIR="outputs/plism/student_logs"
mkdir -p "$LOG_DIR"

CU13_LIB=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib
export LD_PRELOAD="$CU13_LIB/libnvJitLink.so.13 $CU13_LIB/libnvrtc.so.13${LD_PRELOAD:+ $LD_PRELOAD}"
export LD_LIBRARY_PATH="$CU13_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[plism-student] start $(date -Iseconds)"

# Extract features for all 36 student runs
python3 -u scripts/plism_student_extract.py \
    --plism-dir "$PLISM_DIR" \
    --out-root "$FEAT_ROOT" \
    --v4-root "$V4_ROOT" \
    --batch-size 128 \
    --workers 4 \
    2>&1 | tee "$LOG_DIR/extract_all.log"

echo "[plism-student] extraction done $(date -Iseconds)"

# Evaluate each student with plismbench evaluate
for fm_dir in "$V4_ROOT"/*/; do
    FM=$(basename "$fm_dir")
    [ "$FM" = "telemetry" ] || [ "$FM" = "figures" ] && continue
    for stu_dir in "$fm_dir"*/; do
        STU=$(basename "$stu_dir")
        [ -d "$stu_dir" ] || continue
        FEAT_DIR="$FEAT_ROOT/$FM/$STU"
        [ -f "$FEAT_DIR/.done" ] || { echo "[skip-eval] $FM/$STU (no .done)"; continue; }
        EXTRACTOR="$FM/$STU"
        METRICS_SUBDIR="$METRICS_ROOT/8139_tiles/$FM/$STU"
        if [ -d "$METRICS_SUBDIR" ]; then
            echo "[skip-eval] $FM/$STU (metrics exist)"
            continue
        fi
        echo "---- evaluate: $FM/$STU ----"
        ~/.local/bin/plismbench evaluate \
            --extractor "$EXTRACTOR" \
            --features-dir "$FEAT_ROOT" \
            --metrics-dir "$METRICS_ROOT" \
            --device gpu \
            2>&1 | tee "$LOG_DIR/eval_${FM}_${STU}.log" || true
    done
done

echo "[plism-student] evaluation done $(date -Iseconds)"

# Build updated tableS3 and figures
echo "[plism-student] regenerating figures and table"
python3 scripts/figure_plism_robustness.py \
    --plism-results outputs/plism/results \
    --out-dir outputs/v4_full/figures \
    2>&1 | tee "$LOG_DIR/figure.log" || true

python3 scripts/build_tableS3_plism.py 2>&1 | tee "$LOG_DIR/tableS3.log" || true

echo "[plism-student] DONE $(date -Iseconds)"
