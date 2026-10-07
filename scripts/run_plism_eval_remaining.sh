#!/usr/bin/env bash
# Re-score the PLISM student runs whose features were re-extracted but whose
# GPU evaluation hit OOM (GPU memory held by another tenant), using the
# row-chunked top-k of plism_evaluate_chunked.py (identical outputs).
set -u
cd "$(dirname "$0")/.."
CU13_LIB=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib
export LD_PRELOAD="$CU13_LIB/libnvJitLink.so.13 $CU13_LIB/libnvrtc.so.13${LD_PRELOAD:+ $LD_PRELOAD}"
export LD_LIBRARY_PATH="$CU13_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for run in uni/vit-small uni/vit-tiny prov-gigapath/vit-base prov-gigapath/vit-small \
           prov-gigapath/vit-tiny uni2-h/vit-base phikon-v2/vit-tiny; do
    d=outputs/plism/results/8139_tiles/$run
    if [ -f "$d/metrics.csv" ] && [ "$(($(wc -l < $d/metrics.csv)-1))" = 4095 ]; then
        echo "skip $run (complete)"; continue; fi
    rm -rf "$d/pickles"
    echo "==== evaluate $run $(date -Iseconds)"
    python3 scripts/plism_evaluate_chunked.py --extractor "$run" 2>&1 \
        | grep -v -i warn | grep -i "error\|success" | tail -2
    echo "pairs now: $(($(wc -l < $d/metrics.csv 2>/dev/null || echo 0)-1)) $run"
done
echo "[done] $(date -Iseconds)"
