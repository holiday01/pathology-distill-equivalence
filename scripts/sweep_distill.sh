#!/bin/bash
# =============================================================
# WSI FM 蒸餾超參數 sweep
# 跑 10 組對照實驗，目的是找出真正影響 CKA/loss 的參數
# 用法：bash scripts/sweep_distill.sh
# =============================================================

set -u  # 遇到未定義變數就停，但不要 -e 因為單一 run 失敗不該中斷整個 sweep

cd "$(dirname "$0")/.." || exit 1
SWEEP_ID="sweep_$(date +%Y%m%d_%H%M%S)"
ROOT="outputs/${SWEEP_ID}"
DISTILL_DIR="${ROOT}/distill"
EVAL_DIR="${ROOT}/evaluation"
LOG_FILE="${ROOT}/sweep.log"

mkdir -p "${DISTILL_DIR}" "${EVAL_DIR}"

# 所有設定共用
TEACHER="phikon"
STUDENT="vit_small_patch16_224"
EPOCHS=8
SEED=42
# 固定總 patches 數 (40×32=1280)，讓不同 batch_size 每個 epoch 看到同樣多資料
N_SLIDES=40
PATCHES_PER_SLIDE=32

# Best hyperparameters (後半段 batch sweep 會套用這組)
BEST_WD="0.05"
BEST_GRAD_CLIP="1.0"
BEST_WARMUP="0.1"
BEST_PROJ="mlp"

echo "Sweep ID: ${SWEEP_ID}" | tee "${LOG_FILE}"
echo "Teacher: ${TEACHER} | Student: ${STUDENT} | Epochs: ${EPOCHS}" | tee -a "${LOG_FILE}"
echo "Dataset: ${N_SLIDES}×${PATCHES_PER_SLIDE}=${N_SLIDES}*${PATCHES_PER_SLIDE} patches" | tee -a "${LOG_FILE}"
echo "===============================================" | tee -a "${LOG_FILE}"

# run_one: name, method, pretrained, warmup, proj, wd, grad_clip, batch, lr
run_one() {
    local name="$1" method="$2" pretrained="$3" warmup="$4" proj="$5"
    local wd="$6" gc="$7" bs="$8" lr="$9"
    local t_start t_end

    local pretrained_flag=""
    [[ "${pretrained}" == "1" ]] && pretrained_flag="--pretrained_student"

    echo "" | tee -a "${LOG_FILE}"
    echo "[$(date +%H:%M:%S)] === ${name} ===" | tee -a "${LOG_FILE}"
    echo "  method=${method} pretrained=${pretrained} warmup=${warmup} proj=${proj}" | tee -a "${LOG_FILE}"
    echo "  wd=${wd} grad_clip=${gc} batch=${bs} lr=${lr}" | tee -a "${LOG_FILE}"

    t_start=$(date +%s)

    # 1) 蒸餾訓練
    python3 scripts/distill_wsi_model.py \
        --demo \
        --teacher "${TEACHER}" \
        --student "${STUDENT}" \
        --method "${method}" \
        --epochs "${EPOCHS}" \
        --batch_size "${bs}" \
        --lr "${lr}" \
        --seed "${SEED}" \
        --output "${DISTILL_DIR}" \
        --run_name "${name}" \
        --n_slides "${N_SLIDES}" \
        --patches_per_slide "${PATCHES_PER_SLIDE}" \
        --warmup_ratio "${warmup}" \
        --weight_decay "${wd}" \
        --grad_clip "${gc}" \
        --projector "${proj}" \
        ${pretrained_flag} \
        >> "${LOG_FILE}" 2>&1

    local distill_exit=$?
    if [[ ${distill_exit} -ne 0 ]]; then
        echo "  [FAILED] 蒸餾訓練失敗 (exit ${distill_exit})" | tee -a "${LOG_FILE}"
        return 1
    fi

    # 2) 驗證
    local ckpt="${DISTILL_DIR}/${name}/student_${STUDENT}_${method}.pt"
    if [[ ! -f "${ckpt}" ]]; then
        echo "  [FAILED] 找不到 checkpoint: ${ckpt}" | tee -a "${LOG_FILE}"
        return 1
    fi

    python3 scripts/evaluate_distillation.py \
        --teacher "${TEACHER}" \
        --student "${STUDENT}" \
        --student_ckpt "${ckpt}" \
        --batch_size 32 \
        --seed "${SEED}" \
        --output "${EVAL_DIR}/${name}/" \
        --demo \
        >> "${LOG_FILE}" 2>&1

    local eval_exit=$?
    t_end=$(date +%s)
    local elapsed=$((t_end - t_start))

    if [[ ${eval_exit} -ne 0 ]]; then
        echo "  [FAILED] 驗證失敗 (exit ${eval_exit})" | tee -a "${LOG_FILE}"
        return 1
    fi

    # 抓 CKA 結果快速顯示
    local cka="?"
    if [[ -f "${EVAL_DIR}/${name}/evaluation_report.json" ]]; then
        cka=$(python3 -c "import json; d=json.load(open('${EVAL_DIR}/${name}/evaluation_report.json')); print(f\"{d['results']['feature_similarity']['linear_cka']:.4f}\")")
    fi
    echo "  [OK] CKA=${cka} | 耗時 ${elapsed}s" | tee -a "${LOG_FILE}"
}

# ── 實驗組別 ───────────────────────────────────
# name              method   pt warmup proj    wd        gc   bs   lr
run_one "01_baseline"        hybrid   0  0.0    linear  1e-4      0     32   1e-4
run_one "02_pretrained"      hybrid   1  0.0    linear  1e-4      0     32   1e-4
run_one "03_warmup"          hybrid   1  0.1    linear  1e-4      0     32   1e-4
run_one "04_mlp_proj"        hybrid   1  0.1    mlp     1e-4      0     32   1e-4
run_one "05_best"            hybrid   1  "${BEST_WARMUP}" "${BEST_PROJ}" "${BEST_WD}" "${BEST_GRAD_CLIP}" 32   1e-4
run_one "06_best_bs128"      hybrid   1  "${BEST_WARMUP}" "${BEST_PROJ}" "${BEST_WD}" "${BEST_GRAD_CLIP}" 128  4e-4
run_one "07_best_bs256"      hybrid   1  "${BEST_WARMUP}" "${BEST_PROJ}" "${BEST_WD}" "${BEST_GRAD_CLIP}" 256  8e-4
run_one "08_best_bs512"      hybrid   1  "${BEST_WARMUP}" "${BEST_PROJ}" "${BEST_WD}" "${BEST_GRAD_CLIP}" 512  1.6e-3
run_one "09_best_feature"    feature  1  "${BEST_WARMUP}" "${BEST_PROJ}" "${BEST_WD}" "${BEST_GRAD_CLIP}" 128  4e-4
run_one "10_best_relation"   relation 1  "${BEST_WARMUP}" "${BEST_PROJ}" "${BEST_WD}" "${BEST_GRAD_CLIP}" 128  4e-4

# ── 彙整 ───────────────────────────────────
echo "" | tee -a "${LOG_FILE}"
echo "[$(date +%H:%M:%S)] === Sweep 完成，產生 summary ===" | tee -a "${LOG_FILE}"
python3 scripts/summarize_sweep.py --sweep_dir "${ROOT}" 2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "所有結果在：${ROOT}" | tee -a "${LOG_FILE}"
echo "Summary：${ROOT}/summary.md" | tee -a "${LOG_FILE}"
echo "完整 log：${LOG_FILE}" | tee -a "${LOG_FILE}"
