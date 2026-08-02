#!/usr/bin/env bash
# PanNuke segmentation linear-probe eval over all completed v4_full ckpts.
# Per (teacher, student): patch-token probe trained on folds 1+2, tested on fold 3.
# Per-class Dice + mDice + slide-cluster bootstrap + TOST (margin=0.05).
set -u
cd "$(dirname "$0")/.."

ROOT="${ROOT:-outputs/v4_full}"
PANNUKE="${PANNUKE:-data/external/pannuke}"

# Force PyTorch tensor sharing through /dev/shm via sitecustomize.py.
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)/_pyhooks${PYTHONPATH:+:$PYTHONPATH}"
ulimit -n 1048576 2>/dev/null || true

declare -A STUMAP=(
  [vit-tiny]=vit_tiny_patch16_224
  [vit-small]=vit_small_patch16_224
  [vit-base]=vit_base_patch16_224
)

FM_LIST="${FMS:-phikon phikon-v2 hibou-b hibou-l conch uni uni2-h virchow virchow2 h-optimus-0 prov-gigapath midnight}"
STUDENTS="${STUDENTS:-vit-tiny vit-small vit-base}"

START=$(date -Iseconds)
TOTAL=0; RAN=0; SKIP=0; FAIL=0
for fm in $FM_LIST; do
  for stu in $STUDENTS; do
    TOTAL=$((TOTAL+1))
    run_dir="$ROOT/$fm/$stu"
    [ -f "$run_dir/best.pt" ] || { echo "[skip]    $fm/$stu (no best.pt)"; SKIP=$((SKIP+1)); continue; }
    out="$run_dir/pannuke_seg"
    if [ -f "$out/pannuke_seg_report.json" ]; then
      echo "[done]    $fm/$stu"
      SKIP=$((SKIP+1)); continue
    fi
    stu_tim=${STUMAP[$stu]:-}
    [ -z "$stu_tim" ] && { echo "[ERR]     $fm/$stu unknown student"; FAIL=$((FAIL+1)); continue; }
    mkdir -p "$out"
    echo "============================================================"
    echo "[seg eval] $fm × $stu  $(date -Iseconds)"
    echo "============================================================"
    if python3 scripts/evaluate_pannuke_seg.py \
        --teacher "$fm" \
        --student "$stu_tim" \
        --student_ckpt "$run_dir/best.pt" \
        --pannuke_root "$PANNUKE" \
        --train_folds 1,2 --test_fold 3 \
        --output "$out" \
        --batch_size 16 --num_workers 4 \
        2>&1 | tee "$out/seg.log"; then
      RAN=$((RAN+1))
      echo "[OK]      $fm/$stu"
    else
      FAIL=$((FAIL+1))
      echo "[FAIL]    $fm/$stu"
    fi
  done
done

echo ""
echo "[seg eval] FINISHED $(date -Iseconds)  (started $START)"
echo "  total=$TOTAL  new=$RAN  done/skip=$SKIP  failed=$FAIL"

# Aggregate to CSV
python3 - <<'PY'
import json, glob, csv
rows = []
for rep in sorted(glob.glob("outputs/v4_full/*/*/pannuke_seg/pannuke_seg_report.json")):
    try:
        d = json.load(open(rep))
        run = rep.split("outputs/v4_full/")[1].split("/pannuke_seg")[0]
        row = {"run": run, "mDice_fg_T": d["mDice_fg"]["teacher"],
               "mDice_fg_S": d["mDice_fg"]["student"],
               "mDice_fg_delta": d["mDice_fg"]["delta"]}
        for cn, v in d["per_class"].items():
            row[f"{cn}_T"] = v["teacher_dice"]
            row[f"{cn}_S"] = v["student_dice"]
            row[f"{cn}_d"] = v["delta_mean"]
            row[f"{cn}_equiv"] = v["tost_equivalent"]
        rows.append(row)
    except Exception as e:
        print("  skipped", rep, e)
if rows:
    keys = sorted({k for r in rows for k in r})
    keys = ["run"] + [k for k in keys if k != "run"]
    with open("outputs/v4_full/pannuke_seg_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        [w.writerow(r) for r in rows]
    print(f"[summary] {len(rows)} rows → outputs/v4_full/pannuke_seg_summary.csv")
PY
