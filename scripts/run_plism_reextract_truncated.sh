#!/usr/bin/env bash
# Re-extract PLISM features for the student runs whose feature files were
# truncated (zero-filled rows after an interrupted extraction), then
# re-score them. Detected 2026-10-07: 158 slide files across 8 runs.
set -u
cd "$(dirname "$0")/.."
CU13_LIB=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib
export LD_PRELOAD="$CU13_LIB/libnvJitLink.so.13 $CU13_LIB/libnvrtc.so.13${LD_PRELOAD:+ $LD_PRELOAD}"
export LD_LIBRARY_PATH="$CU13_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for run in uni/vit-base uni/vit-small uni/vit-tiny prov-gigapath/vit-base \
           prov-gigapath/vit-small prov-gigapath/vit-tiny uni2-h/vit-base phikon-v2/vit-tiny; do
    t=${run%/*}; s=${run#*/}
    echo "==== re-extract $run $(date -Iseconds)"
    python3 -u scripts/plism_student_extract.py --teacher "$t" --student "$s" \
        --batch-size 128 --workers 4 2>&1 | grep -v -i warn | tail -3
    bad=$(python3 -c "
import numpy as np,glob
print(sum(int((abs(np.load(f,mmap_mode='r')[:,:3]).sum(1)==0).any()) for f in glob.glob('outputs/plism/features/$run/*/features.npy')))")
    echo "truncated slide files after re-extraction: $bad"
    [ "$bad" = "0" ] || { echo "STILL TRUNCATED: $run"; continue; }
    d=outputs/plism/results/8139_tiles/$run
    rm -rf "$d/pickles"; mkdir -p "$d/old"
    for f in metrics.csv results.csv metrics.csv.partial results.csv.partial; do
        [ -f "$d/$f" ] && mv "$d/$f" "$d/old/$f"
    done
    ~/.local/bin/plismbench evaluate --extractor "$run" \
        --features-dir outputs/plism/features --metrics-dir outputs/plism/results \
        --device gpu 2>&1 | grep -v -i warn | tail -2
    echo "pairs now: $(($(wc -l < $d/metrics.csv)-1))"
done
echo "[done] $(date -Iseconds)"
