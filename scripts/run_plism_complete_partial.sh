#!/usr/bin/env bash
# Finish the PLISM student evaluations that stopped before all 4,095 slide
# pairs were scored. plismbench keeps one pickle per slide pair, so a rerun
# only computes the missing pairs. Writes logs next to the originals.
set -u
cd "$(dirname "$0")/.."
CU13_LIB=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib
export LD_PRELOAD="$CU13_LIB/libnvJitLink.so.13 $CU13_LIB/libnvrtc.so.13${LD_PRELOAD:+ $LD_PRELOAD}"
export LD_LIBRARY_PATH="$CU13_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for run in uni/vit-base uni/vit-small uni/vit-tiny prov-gigapath/vit-base \
           prov-gigapath/vit-small prov-gigapath/vit-tiny uni2-h/vit-base phikon-v2/vit-tiny; do
    echo "---- complete: $run $(date -Iseconds)"
    d=outputs/plism/results/8139_tiles/$run
    # plismbench skips a run whose metrics.csv exists; set the partial
    # tables aside (pickles stay, so finished pairs are not recomputed)
    for f in metrics.csv results.csv; do
        [ -f "$d/$f" ] && mv "$d/$f" "$d/$f.partial"
    done
    ~/.local/bin/plismbench evaluate --extractor "$run" \
        --features-dir outputs/plism/features --metrics-dir outputs/plism/results \
        --device gpu 2>&1 | grep -v -i warn | tail -5
    echo "pairs now: $(($(wc -l < outputs/plism/results/8139_tiles/$run/metrics.csv)-1))"
done
echo "[done] $(date -Iseconds)"
