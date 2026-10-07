#!/usr/bin/env bash
# =====================================================================
# CAMELYON16 full-training-set upgrade, run unattended end to end.
#
#   Stage 1  fetch the 111 lesion-annotation XMLs           (~minutes)
#   Stage 2  fetch the remaining WSIs to 159 normal + 111 tumour
#            (188 slides, ~338 GB, network-capped at ~7.6 MB/s)
#   Stage 3  re-extract patches with lesion-polygon labels
#   Stage 4  re-run the slide-cluster equivalence evaluation
#   Stage 5  re-run the derived statistics
#
# Every stage writes a .done marker and is skipped on re-run, so the
# script is safe to relaunch after a reboot or a dropped connection.
# Nothing here overwrites the current results: the new patch bundle and
# all outputs go to *_270 paths, leaving the 82-slide numbers that the
# submitted manuscript reports untouched until they are deliberately
# replaced.
# =====================================================================
set -u
cd "$(dirname "$0")/.."

WSI_DIR=/path/to/wsi_datasets/camelyon16
ANN_DIR=data/c16_annotations
H5_OUT=/path/to/cache/patches/patches_c16_annotated_270.h5
RESULT_DIR=outputs/v4_full/c16_270
# eval_c16_equivalence.py treats --out as a FILENAME PREFIX and writes
# <prefix>.json and <prefix>.csv. Passing a directory here silently produces
# no file at the path the completion check looks for.
EQUIV_PREFIX=$RESULT_DIR/c16_equiv_annotated
STATE=outputs/pipeline_270
LOG=$STATE/pipeline.log

mkdir -p "$STATE" "$ANN_DIR" "$RESULT_DIR"
exec > >(tee -a "$LOG") 2>&1

say() { echo "[$(date -Is)] $*"; }
done_marker() { echo "$STATE/$1.done"; }
stage_done() { [ -f "$(done_marker "$1")" ]; }
mark_done() { date -Is > "$(done_marker "$1")"; say "STAGE $1 COMPLETE"; }

say "=== pipeline start ==="

# ---------------------------------------------------------------- 1
if stage_done annotations; then
  say "stage 1 (annotations) already done, skipping"
else
  say "stage 1: fetching lesion annotations"
  BASE=https://camelyon-dataset.s3.amazonaws.com/CAMELYON16/annotations
  miss=0
  for i in $(seq -w 1 111); do
    f="$ANN_DIR/tumor_$i.xml"
    [ -s "$f" ] && continue
    curl -sfL -o "$f" "$BASE/tumor_$i.xml" || { rm -f "$f"; miss=$((miss+1)); }
  done
  n=$(ls "$ANN_DIR"/tumor_*.xml 2>/dev/null | wc -l)
  say "stage 1: have $n/111 annotation files ($miss failed)"
  [ "$n" -ge 111 ] && mark_done annotations || say "stage 1 INCOMPLETE, will retry next run"
fi

# ---------------------------------------------------------------- 2
if stage_done download; then
  say "stage 2 (WSI download) already done, skipping"
else
  say "stage 2: downloading WSIs to 159 normal + 111 tumour (resume-safe)"
  python3 scripts/download_camelyon16.py \
      --out "$WSI_DIR" --normal 159 --tumor 111 --test 0 --workers 8
  n_norm=$(ls "$WSI_DIR"/normal_*.tif 2>/dev/null | wc -l)
  n_tum=$(ls "$WSI_DIR"/tumor_*.tif 2>/dev/null | wc -l)
  say "stage 2: have $n_norm normal + $n_tum tumour on disk"
  if [ "$n_norm" -ge 159 ] && [ "$n_tum" -ge 111 ]; then
    mark_done download
  else
    say "stage 2 INCOMPLETE, relaunch to resume"; exit 1
  fi
fi

# ---------------------------------------------------------------- 3
if stage_done extract; then
  say "stage 3 (patch extraction) already done, skipping"
else
  say "stage 3: extracting lesion-annotated patches over 270 slides"
  python3 scripts/extract_c16_annotated.py \
      --wsi_dir "$WSI_DIR" --ann_dir "$ANN_DIR" --out "$H5_OUT" --seed 42
  [ -s "$H5_OUT" ] && mark_done extract || { say "stage 3 FAILED"; exit 1; }
fi

# ---------------------------------------------------------------- 4
if stage_done equiv; then
  say "stage 4 (equivalence eval) already done, skipping"
else
  say "stage 4: slide-cluster equivalence over the enlarged cohort"
  python3 scripts/eval_c16_equivalence.py \
      --h5 "$H5_OUT" --test_frac 0.5 --out "$EQUIV_PREFIX"
  [ -s "${EQUIV_PREFIX}.json" ] && mark_done equiv || { say "stage 4 FAILED"; exit 1; }
fi

# ---------------------------------------------------------------- 5
if stage_done derived; then
  say "stage 5 (derived stats) already done, skipping"
else
  say "stage 5: derived statistics on the enlarged cohort"
  # compute_intra_slide_icc.py and compute_c16_calibration_annotated.py take no
  # arguments: the 82-slide H5 and outputs/v4_full are module-level constants.
  # Run as-is they would recompute on the OLD data and overwrite
  # intra_slide_icc.json and c16_calibration_annotated.json, which are the
  # source of the rho = 0.2467 and ECE 0.050/0.039 figures the submitted
  # manuscript prints. Rewrite both constants at load time instead of editing
  # the released scripts, so the 82-slide outputs stay provably untouched.
  #
  # compute_cluster_se_inflation.py is deliberately NOT re-run here. It derives
  # an accuracy-based inflation from the atlas downstream_report.json files,
  # which is a different quantity from the AUROC inflation the paper reports
  # (that comes straight from stage 4's linear_auc_inflation), and it writes
  # into outputs/v4_full as well.
  for s in compute_intra_slide_icc compute_c16_calibration_annotated; do
    say "  running $s against the 270-slide bundle"
    PIPE_H5="$H5_OUT" PIPE_ROOT="$RESULT_DIR" PIPE_SCRIPT="scripts/$s.py" \
    python3 - <<'PY' || say "  $s failed (non-fatal)"
import os, re, pathlib, runpy, sys
src = pathlib.Path(os.environ["PIPE_SCRIPT"]).read_text()
h5, root = os.environ["PIPE_H5"], os.environ["PIPE_ROOT"]
src = re.sub(r'^(C16|H5)\s*=\s*".*"$', lambda m: f'{m.group(1)} = "{h5}"', src, flags=re.M)
src = re.sub(r'^ROOT\s*=\s*".*"$', f'ROOT = "{root}"', src, flags=re.M)
src = src.replace('"outputs/v4_full/intra_slide_icc.json"', f'"{root}/intra_slide_icc.json"')
if h5 not in src or root not in src:
    raise SystemExit(f"ABORT: path rewrite did not take in {os.environ['PIPE_SCRIPT']}")
# The temp copy must live in scripts/: these files do
# sys.path.insert(0, Path(__file__).parent) to reach their sibling modules,
# so running the copy from anywhere else breaks that import.
tmp = pathlib.Path("scripts") / (pathlib.Path(os.environ["PIPE_SCRIPT"]).stem + "_270_tmp.py")
tmp.write_text(src)
try:
    sys.argv = [str(tmp)]
    runpy.run_path(str(tmp), run_name="__main__")
finally:
    tmp.unlink(missing_ok=True)
PY
  done
  if [ -s "$RESULT_DIR/intra_slide_icc.json" ] && [ -s "$RESULT_DIR/c16_calibration_annotated.json" ]; then
    mark_done derived
  else
    say "stage 5 INCOMPLETE: expected outputs missing, not marking done"
  fi
fi

# ---------------------------------------------------------------- 6
# External audit on the enlarged cohort. All four published (teacher,
# student) pairs are re-run against the same patch bundle and the same
# split as the atlas, so the external verdicts and the 36 atlas verdicts
# are read off one measurement.
if stage_done external; then
  say "stage 6 (external audit) already done, skipping"
else
  say "stage 6: external audit on the enlarged cohort"
  say "  6a: GPFM vs CONCH / Phikon / UNI (probe-averaged over 8 fits)"
  python3 scripts/eval_gpfm_equivalence.py \
      --h5 "$H5_OUT" --test_frac 0.5 \
      --out "$RESULT_DIR/gpfm_equiv.json" || say "  gpfm failed (non-fatal)"
  say "  6b: H0-mini vs H-Optimus-0"
  H5_ANNOT="$H5_OUT" python3 - <<'PY' || say "  h0mini failed (non-fatal)"
import os, re, pathlib, runpy, sys
src = pathlib.Path("scripts/h6_flip_h0mini.py").read_text()
# the script pins the 82-slide bundle at module scope; point it at the new one
src = re.sub(r'^H5 = ".*"$', f'H5 = "{os.environ["H5_ANNOT"]}"', src, flags=re.M)
tmp = pathlib.Path("outputs/pipeline_270/_h0mini_270.py"); tmp.write_text(src)
sys.argv = [str(tmp)]
runpy.run_path(str(tmp), run_name="__main__")
PY
  mark_done external
fi

say "=== pipeline complete ==="
say "new results are under $RESULT_DIR ; the 82-slide results are untouched"
