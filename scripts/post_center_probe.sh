#!/usr/bin/env bash
# Run after patch_center_probe.py completes.
# 1. Regenerate atlas_summary.json (picks up center probe data)
# 2. Generate center probe figure
# 3. Update aggregate stats
# Usage: bash scripts/post_center_probe.sh
set -u
cd "$(dirname "$0")/.."

echo "[post-center] start $(date -Iseconds)"

# Regenerate atlas_summary
echo "[post-center] regenerating atlas_summary.json"
python3 -u scripts/aggregate_atlas.py 2>&1 | tee /tmp/aggregate_atlas.log

# Generate center probe figure
echo "[post-center] generating center probe figure"
python3 -u scripts/figure_center_probe.py \
    --atlas outputs/v4_full/atlas_summary.json \
    --c16-annotated outputs/v4_full/c16_equiv_annotated.json \
    --out outputs/v4_full/figures/F3_center_probe \
    2>&1 | tee /tmp/figure_center_probe.log

echo "[post-center] DONE $(date -Iseconds)"
echo "Check /tmp/figure_center_probe.log for results."
