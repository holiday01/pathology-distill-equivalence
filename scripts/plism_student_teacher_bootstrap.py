#!/usr/bin/env python3
"""Slide-level bootstrap CI for student-minus-teacher PLISM top-1 (all regime).

The 4,095 slide pairs share slides, so pairs are not independent. Each
replicate resamples the 91 slides with replacement and weights pair (i, j) by
n_i * n_j, the number of times both slides were drawn; the statistic is the
weighted mean of the per-pair difference (student - teacher) and the ratio of
weighted means (retention).

Usage:
  python3 scripts/plism_student_teacher_bootstrap.py \
      --out outputs/plism/student_teacher_bootstrap.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from plism_student_retention import STUDENT_DIR, STUDENTS
from paper_cohort import keep_teacher

B = 2000
SEED = 0


def load(path):
    rows = list(csv.DictReader(open(path)))
    return {(r["slide_a"], r["slide_b"]): float(r["top_1_accuracy"]) for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/plism/results/8139_tiles")
    ap.add_argument("--out", default="outputs/plism/student_teacher_bootstrap.json")
    a = ap.parse_args()
    R = Path(a.root)
    out = []
    for t, sd in STUDENT_DIR.items():
        if not keep_teacher(t):
            continue
        tv = load(R / t / "metrics.csv")
        keys = sorted(tv)
        slides = sorted({s for k in keys for s in k})
        ix = {s: i for i, s in enumerate(slides)}
        ia = np.array([ix[k[0]] for k in keys]); ib = np.array([ix[k[1]] for k in keys])
        T = np.array([tv[k] for k in keys])
        rng = np.random.default_rng(SEED)
        W = np.empty((B, len(keys)))
        for b in range(B):
            n = np.bincount(rng.integers(0, len(slides), len(slides)), minlength=len(slides))
            W[b] = n[ia] * n[ib]
        for s in STUDENTS:
            sv = load(R / sd / s / "metrics.csv")
            S = np.array([sv[k] for k in keys])
            diff = (W @ (S - T)) / W.sum(1)
            ret = (W @ S) / (W @ T)
            out.append({"teacher": t, "student": s,
                        "diff": float((S - T).mean()),
                        "diff_ci95": np.percentile(diff, [2.5, 97.5]).tolist(),
                        "retention": float(S.mean() / T.mean()),
                        "retention_ci95": np.percentile(ret, [2.5, 97.5]).tolist(),
                        "frac_pairs_student_higher": float((S > T).mean())})
            o = out[-1]
            print(f"{t:13s} {s:9s} ret={o['retention']:.2f} "
                  f"[{o['retention_ci95'][0]:.2f},{o['retention_ci95'][1]:.2f}] "
                  f"pairs S>T {o['frac_pairs_student_higher']:.2f}", flush=True)
    json.dump({"B": B, "seed": SEED, "rows": out}, open(a.out, "w"), indent=1)
    print(f"[wrote] {a.out}")


if __name__ == "__main__":
    main()
