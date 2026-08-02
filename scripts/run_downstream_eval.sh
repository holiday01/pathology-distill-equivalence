#!/usr/bin/env bash
# Downstream evaluation launcher for the v4_full atlas.
# For every completed (best.pt exists) student ckpt, run
# evaluate_v4_downstream.py to compute:
#   - CKA + cosine to teacher
#   - C16 linear + MLP probe with TOST equivalence + slide-cluster bootstrap
#   - Kather-MSI linear probe
#   - ECE / MCE / AURC / risk@coverage=0.9  (Guo 2017, Geifman 2017)
#   - Medical-center probe (de Jong 2025)
#   - Latency + speedup
#
# Skip logic: skip if downstream_report.json already exists.
# Each eval ~20-40 min on RTX 5090 depending on teacher size.
#
# Usage:
#   bash scripts/run_downstream_eval.sh                    # all completed ckpts
#   FMS="phikon uni" bash scripts/run_downstream_eval.sh   # specific subset
set -u
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/multi_source_v1.json}"
BS="${BS:-128}"
NW="${NW:-4}"
SEED="${SEED:-42}"
ROOT="${ROOT:-outputs/v4_full}"
FMS="${FMS:-}"
STUDENTS="${STUDENTS:-vit-tiny vit-small vit-base}"

# Force PyTorch tensor sharing through /dev/shm via sitecustomize.py
# (avoids `unable to allocate shared memory(shm) for file </torch_*>`).
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)/_pyhooks${PYTHONPATH:+:$PYTHONPATH}"
ulimit -n 1048576 2>/dev/null || true

# Map vit-{tiny,small,base} → timm model id
declare -A STUMAP=(
  [vit-tiny]=vit_tiny_patch16_224
  [vit-small]=vit_small_patch16_224
  [vit-base]=vit_base_patch16_224
)

# FM → teacher registry key (matches distill_wsi_model.FM_REGISTRY)
FM_LIST="${FMS:-phikon phikon-v2 hibou-b hibou-l conch uni uni2-h virchow virchow2 h-optimus-0 prov-gigapath midnight}"

TOTAL=0; DONE_=0; RAN=0; FAILED=0
START=$(date -Iseconds)
echo "[downstream eval] start $START"

for fm in $FM_LIST; do
  for stu in $STUDENTS; do
    TOTAL=$((TOTAL+1))
    run_dir="$ROOT/$fm/$stu"
    [ -f "$run_dir/best.pt" ] || { echo "[skip]   $fm/$stu (no best.pt)"; continue; }
    [ -f "$run_dir/splits.npz" ] || { echo "[skip]   $fm/$stu (no splits.npz)"; continue; }

    out="$run_dir/eval_downstream"
    if [ -f "$out/downstream_report.json" ]; then
      echo "[done]   $fm/$stu (report exists)"
      DONE_=$((DONE_+1))
      continue
    fi

    stu_tim=${STUMAP[$stu]:-}
    [ -z "$stu_tim" ] && { echo "[ERR]    $fm/$stu unknown student"; FAILED=$((FAILED+1)); continue; }

    mkdir -p "$out"
    echo "============================================================"
    echo "[eval]   $fm × $stu  $(date -Iseconds)"
    echo "============================================================"
    if python3 scripts/evaluate_v4_downstream.py \
        --config "$CONFIG" \
        --student_ckpt "$run_dir/best.pt" \
        --splits "$run_dir/splits.npz" \
        --output "$out" \
        --teacher "$fm" \
        --student "$stu_tim" \
        --batch_size "$BS" --num_workers "$NW" \
        --seed "$SEED" \
        --n_boot 1000 --tost_margin 0.03 \
        2>&1 | tee "$out/eval.log"; then
      RAN=$((RAN+1))
      echo "[OK]     $fm/$stu"
    else
      FAILED=$((FAILED+1))
      echo "[FAIL]   $fm/$stu"
    fi
  done
done

echo ""
echo "[downstream eval] FINISHED $(date -Iseconds)  (started $START)"
echo "  total=$TOTAL  new=$RAN  already-done=$DONE_  failed=$FAILED"

# Aggregate all reports
python3 - <<'PY'
import json, glob
rows = []
for rep in glob.glob("outputs/v4_full/*/*/eval_downstream/downstream_report.json"):
    try:
        d = json.load(open(rep))
        run = rep.split("outputs/v4_full/")[1].split("/eval_downstream")[0]
        r = d["results"]
        row = {
            "run": run,
            "cka": r["feature_similarity"]["cka_overall"],
            "cos": r["feature_similarity"]["cosine_overall"],
        }
        for probe in ["c16_linear", "c16_mlp"]:
            pr = r.get("probes", {}).get(probe, {})
            tost = pr.get("tost", {})
            row[f"{probe}_Tacc"] = pr.get("teacher_acc")
            row[f"{probe}_Sacc"] = pr.get("student_acc")
            row[f"{probe}_d"] = tost.get("d_mean")
            row[f"{probe}_equiv"] = tost.get("equivalent")
            row[f"{probe}_kappa"] = pr.get("cohen_kappa")
            cal = pr.get("calibration", {})
            row[f"{probe}_s_ece"] = cal.get("student", {}).get("ece")
            row[f"{probe}_t_ece"] = cal.get("teacher", {}).get("ece")
            row[f"{probe}_s_aurc"] = cal.get("student", {}).get("aurc")
            row[f"{probe}_t_aurc"] = cal.get("teacher", {}).get("aurc")
        cp = r.get("medical_center_probe", {})
        row["center_auc_teacher"] = cp.get("teacher", {}).get("center_probe_auc")
        row["center_auc_student"] = cp.get("student", {}).get("center_probe_auc")
        row["RI_teacher"] = cp.get("teacher", {}).get("robustness_index")
        row["RI_student"] = cp.get("student", {}).get("robustness_index")
        row["speedup"] = r.get("efficiency", {}).get("speedup")
        rows.append(row)
    except Exception as e:
        print("  skipped", rep, e)
if rows:
    import csv
    out = "outputs/v4_full/downstream_summary.csv"
    keys = sorted({k for r in rows for k in r})
    keys = ["run"] + [k for k in keys if k != "run"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        [w.writerow(r) for r in rows]
    print(f"[summary] {len(rows)} rows → {out}")
PY
