#!/usr/bin/env python3
"""Slide-count sweep of the cluster/naive SE inflation on the HEADLINE split.

sweep_slide_count_inflation.py drew its own 50/50 slide split and refit its
own probe, so its 135-slide point (8.2x) was not the headline evaluation
(4.9x) and the two could not be compared. This version subsamples the
headline test half itself, using the per-pair tile scores written by
eval_c16_equivalence.py --save_scores, so the full-size point reproduces the
headline inflation by construction. Sampler, bootstrap and pairs are those of
sweep_slide_count_inflation.py: whole slides are drawn (m fixed), the
tumour-bearing/normal slide ratio of the pool is kept, 8 replicates per size.

Usage:
  python3 scripts/sweep_slide_count_headline.py \
      --scores outputs/v4_full/c16_270_rerun/scores \
      --out outputs/v4_full/c16_270/slide_count_sweep_headline.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from sweep_slide_count_inflation import (cluster_and_naive_se, SIZES, REPS,
                                         TEACHERS, STUDENTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="outputs/v4_full/c16_270_rerun/scores")
    ap.add_argument("--out", default="outputs/v4_full/c16_270/slide_count_sweep_headline.json")
    a = ap.parse_args()
    rows = []
    for t in TEACHERS:
        for s in STUDENTS:
            z = np.load(Path(a.scores) / f"{t}__{s}__linear.npz")
            y, ts, ss, g = z["y"], z["t"], z["s"], z["groups"]
            pool = np.unique(g)
            tum = np.array(sorted({x for x in pool if y[g == x].sum() > 0}))
            nor = np.array(sorted(set(pool) - set(tum)))
            frac = len(tum) / len(pool)
            for n in SIZES:
                for rep in range(REPS):
                    rng = np.random.default_rng(1000 * n + rep)
                    kt = min(max(2, round(n * frac)), len(tum))
                    kn = min(len(nor), n - kt)
                    sel = set(np.r_[rng.choice(tum, kt, replace=False),
                                    rng.choice(nor, kn, replace=False)])
                    m = np.isin(g, list(sel))
                    c, nv = cluster_and_naive_se(y[m], ts[m], ss[m], g[m], seed=rep)
                    if np.isfinite(c) and np.isfinite(nv) and nv > 0:
                        rows.append({"teacher": t, "student": s, "n_test_slides": n,
                                     "rep": rep, "n_tiles": int(m.sum()),
                                     "m": float(m.sum() / len(sel)),
                                     "se_cluster": c, "se_naive": nv,
                                     "inflation": c / nv})
            print(f"  [done] {t} x {s}", flush=True)
    Path(a.out).write_text(json.dumps({"sizes": SIZES, "reps": REPS,
                                       "teachers": TEACHERS, "split": "headline",
                                       "rows": rows}, indent=2))
    print(f"[wrote] {a.out}")


if __name__ == "__main__":
    main()
