#!/bin/bash
# =============================================================
#  WSI Foundation Model 蒸餾 — 一鍵自動化執行腳本
#  用法：bash run_all.sh [WSI_DIR]
#  範例：bash run_all.sh /data/slides
#        bash run_all.sh          # 無參數：使用 demo 模式
# =============================================================

set -euo pipefail

# ── 設定 ──────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$BASE_DIR/outputs"
WSI_DIR="${1:-}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/run_${TIMESTAMP}.log"

mkdir -p "$OUTPUT_DIR"

# ── 顏色輸出 ─────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'

log() { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $1" | tee -a "$LOG_FILE"; }
ok()  { echo -e "${GREEN}✓${NC} $1" | tee -a "$LOG_FILE"; }
warn(){ echo -e "${YELLOW}⚠${NC}  $1" | tee -a "$LOG_FILE"; }
err() { echo -e "${RED}✗${NC} $1" | tee -a "$LOG_FILE"; }

# ── 主流程 ───────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  WSI Foundation Model 蒸餾 — 全自動化執行"
echo "  開始時間: $(date)"
echo "  輸出目錄: $OUTPUT_DIR"
echo "================================================================"
echo "" | tee -a "$LOG_FILE"

# Step 0: 環境檢查
log "Step 0: 環境檢查"
python3 -c "import sys; print(f'  Python: {sys.version}')" | tee -a "$LOG_FILE"

if python3 -c "import torch; print(f'  PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null | tee -a "$LOG_FILE"; then
    ok "PyTorch 已安裝"
else
    warn "PyTorch 未安裝，請先安裝：pip install torch torchvision --break-system-packages"
    err "缺少必要依賴，請安裝後重試"
    exit 1
fi

if python3 -c "import timm" 2>/dev/null; then
    ok "timm 已安裝"
else
    warn "timm 未安裝（非必要，未安裝時使用內建 fallback CNN）"
fi

echo ""

# Step 1: Pipeline Benchmark
log "Step 1: WSI Pipeline 時間 Benchmark"
echo "  量測前處理 vs. FM 推論時間分配..."

if [ -n "$WSI_DIR" ] && [ -d "$WSI_DIR" ]; then
    log "  使用真實 WSI: $WSI_DIR"
    python3 "$SCRIPT_DIR/benchmark_pipeline.py" \
        --wsi_dir "$WSI_DIR" \
        --model vit_base \
        --batch_size 64 \
        --output "$OUTPUT_DIR/benchmark/" \
        2>&1 | tee -a "$LOG_FILE"
else
    warn "未指定 WSI 目錄，使用 demo 模式（合成資料）"
    python3 "$SCRIPT_DIR/benchmark_pipeline.py" \
        --demo \
        --n_demo_slides 10 \
        --model vit_base \
        --output "$OUTPUT_DIR/benchmark/" \
        2>&1 | tee -a "$LOG_FILE"
fi

ok "Benchmark 完成，結果存於 $OUTPUT_DIR/benchmark/"
echo ""

# Step 2: 模型速度比較
log "Step 2: 不同模型推論速度比較"
python3 "$SCRIPT_DIR/benchmark_pipeline.py" \
    --demo \
    --n_demo_slides 3 \
    --compare_models \
    --output "$OUTPUT_DIR/benchmark/" \
    2>&1 | tee -a "$LOG_FILE"

ok "速度比較完成"
echo ""

# Step 3: 蒸餾訓練（Feature-based）
log "Step 3: Feature-based 蒸餾訓練（ViT-L → ViT-Small）"
python3 "$SCRIPT_DIR/distill_wsi_model.py" \
    --teacher vit_large_patch16_224 \
    --student vit_small_patch16_224 \
    --method feature \
    --epochs 5 \
    --batch_size 32 \
    --demo \
    --output "$OUTPUT_DIR/distill/feature/" \
    2>&1 | tee -a "$LOG_FILE"

ok "Feature 蒸餾完成"
echo ""

# Step 4: 蒸餾訓練（Relation-based）
log "Step 4: Relation-based 蒸餾訓練（HVisKD 風格，適合 Segmentation）"
python3 "$SCRIPT_DIR/distill_wsi_model.py" \
    --teacher vit_large_patch16_224 \
    --student vit_small_patch16_224 \
    --method relation \
    --epochs 5 \
    --batch_size 32 \
    --demo \
    --output "$OUTPUT_DIR/distill/relation/" \
    2>&1 | tee -a "$LOG_FILE"

ok "Relation 蒸餾完成"
echo ""

# Step 5: 蒸餾訓練（Hybrid，GPFM 風格）
log "Step 5: Hybrid 蒸餾訓練（GPFM 風格，最強效果）"
python3 "$SCRIPT_DIR/distill_wsi_model.py" \
    --teacher vit_large_patch16_224 \
    --student vit_small_patch16_224 \
    --method hybrid \
    --epochs 5 \
    --batch_size 32 \
    --demo \
    --output "$OUTPUT_DIR/distill/hybrid/" \
    2>&1 | tee -a "$LOG_FILE"

ok "Hybrid 蒸餾完成"
echo ""

# Step 6: 結果彙整
log "Step 6: 彙整所有結果"

python3 - << 'PYEOF' 2>&1 | tee -a "$LOG_FILE"
import json, os, glob

output_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs") \
    if "__file__" in dir() else "outputs"

# 讀取 benchmark 結果
bench_file = os.path.join(output_base, "benchmark/benchmark_results.json")
if os.path.exists(bench_file):
    with open(bench_file) as f:
        bench = json.load(f)
    print("\n  === Pipeline Benchmark 摘要 ===")
    for slide in bench[:3]:
        print(f"  Slide: {slide['slide_name']}")
        for step in slide["steps"]:
            pct = step["time_sec"] / slide["total_time"] * 100
            print(f"    {step['name']:<25}: {step['time_sec']:.3f}s ({pct:.1f}%)")

# 讀取蒸餾結果
for method in ["feature", "relation", "hybrid"]:
    log_file = os.path.join(output_base, f"distill/{method}/training_log.json")
    if os.path.exists(log_file):
        with open(log_file) as f:
            log = json.load(f)
        last = log["history"][-1]
        print(f"\n  === {method.upper()} 蒸餾結果 ===")
        print(f"    壓縮比: {log['compression_ratio']:.1f}x")
        print(f"    Teacher 參數: {log['teacher_params']:,}")
        print(f"    Student 參數: {log['student_params']:,}")
        print(f"    最終 Loss:    {last['loss']:.4f}")
PYEOF

# 最終摘要
echo ""
echo "================================================================"
echo "  全流程完成！"
echo "  結束時間: $(date)"
echo ""
echo "  輸出檔案："
find "$OUTPUT_DIR" -name "*.json" | sort | while read f; do
    echo "    $f"
done
echo ""
echo "  查看 benchmark 報告："
echo "    cat $OUTPUT_DIR/benchmark/benchmark_results.json"
echo ""
echo "  下一步建議："
echo "    1. 用真實 WSI 資料替換 demo 資料執行訓練"
echo "    2. See README.md for the full distillation-atlas pipeline"
echo "================================================================"
echo ""
ok "所有任務完成，Log 檔：$LOG_FILE"
