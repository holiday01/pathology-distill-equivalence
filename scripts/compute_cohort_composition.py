#!/usr/bin/env python3
"""Record the composition of the lesion-annotated CAMELYON16 extraction.

Methods has to state how many slides and tiles the evaluation rests on and how
much of it is tumour. Those numbers were hand-typed against the earlier 82-slide
extraction and survived the move to the full 270-slide cohort, so they are
derived here instead and consumed through paper/macros.tex.

Reads only `labels` and `slide_ids` from the patch H5 (a few MB of an 11 GB
file) and reproduces the train/test split the evaluation uses, so the reported
composition is the composition the equivalence test actually saw.

  python3 scripts/compute_cohort_composition.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_c16_equivalence import load_c16, stratified_slide_split

H5 = "/path/to/cache/patches/patches_c16_annotated_270.h5"
OUT = Path("outputs/v4_full/c16_270/cohort_composition.json")
TEST_FRAC = 0.5


def half(labels, slide_ids, ii):
    s, y = slide_ids[ii], labels[ii]
    slides = np.unique(s)
    tiles_per = np.array([int((s == x).sum()) for x in slides])
    return {
        "n_slides": int(len(slides)),
        "n_tiles": int(len(ii)),
        "n_tumour_tiles": int(y.sum()),
        "pct_tumour_tiles": round(100 * float(y.mean()), 2),
        # A CAMELYON16 tumour slide is predominantly benign tissue, so a slide
        # counts as tumour-bearing if any tile centre falls inside an annotated
        # polygon. No slide has a tumour *majority* -- see note in main().
        "n_slides_with_tumour": int(sum((y[s == x]).sum() > 0 for x in slides)),
        "mean_tiles_per_slide": round(float(tiles_per.mean()), 1),
        "median_tiles_per_slide": int(np.median(tiles_per)),
    }


def main():
    idx, labels, slide_ids = load_c16(H5, None)
    tr, te, n_test_slides, n_total_slides = stratified_slide_split(
        slide_ids, labels, idx, TEST_FRAC)

    out = {
        "h5": H5,
        "test_frac": TEST_FRAC,
        "n_total_slides": int(n_total_slides),
        "n_test_slides": int(n_test_slides),
        "all": half(labels, slide_ids, idx),
        "train": half(labels, slide_ids, tr),
        "test": half(labels, slide_ids, te),
    }

    # The split stratifies on each slide's majority tile label. On this cohort
    # every slide is majority-normal, so the tumour stratum is empty and the
    # split degenerates to a random half of the slides. Recorded rather than
    # silently relied on, because Methods must not claim stratification that
    # does not happen.
    out["label_stratification_active"] = bool(out["all"]["n_slides_with_tumour"]
                                              and any(
        round(labels[slide_ids == x].mean()) == 1
        for x in np.unique(slide_ids)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"[wrote] {OUT}")
    for k in ("all", "train", "test"):
        d = out[k]
        print(f"  {k:6s} {d['n_slides']:4d} slides  {d['n_tiles']:6d} tiles  "
              f"{d['n_tumour_tiles']:5d} tumour ({d['pct_tumour_tiles']:.2f}%)  "
              f"{d['n_slides_with_tumour']:3d} tumour-bearing slides")
    print(f"  label stratification active: {out['label_stratification_active']}")


if __name__ == "__main__":
    main()
