#!/usr/bin/env bash
# Detached waiter: block until the PLISM sweep finishes, then render the
# cross-teacher robustness heatmap. Launched with nohup so it survives logout /
# session end. Safe to leave unattended.
set -u
cd /path/to/wsi_hl
export MPLBACKEND=Agg

RESUME_LOG="outputs/plism_eval_resume_20260525_0831.log"
AUTO_LOG="outputs/plism_auto_figure.log"

{
  echo "[auto-fig] waiter started $(date -Iseconds); watching $RESUME_LOG"
  while true; do
    if grep -q "\[plism sweep\] FINISHED" "$RESUME_LOG" 2>/dev/null; then
      echo "[auto-fig] detected sweep FINISHED $(date -Iseconds)"
      break
    fi
    if ! pgrep -f "run_plism_eval.sh" >/dev/null 2>&1; then
      echo "[auto-fig] run_plism_eval.sh gone (no FINISHED marker) $(date -Iseconds);" \
           "proceeding on whatever results exist"
      break
    fi
    sleep 60
  done
  echo "[auto-fig] rendering robustness heatmap $(date -Iseconds)"
  python3 scripts/figure_plism_robustness.py \
      --plism-results outputs/plism/results \
      --out-dir outputs/v4_full/figures
  echo "[auto-fig] DONE $(date -Iseconds)"
} >> "$AUTO_LOG" 2>&1
