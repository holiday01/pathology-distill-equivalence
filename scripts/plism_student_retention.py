#!/usr/bin/env python3
"""PLISM top-1 retention of each distilled student relative to its teacher.

Reads the per-pair metrics.csv that `plismbench evaluate` writes for every
teacher ({root}/{teacher}/metrics.csv) and student
({root}/{student_dir}/{vit-*}/metrics.csv), requires all 4,095 slide pairs on
both sides, and averages top-1 accuracy within each regime:
  inter-scanner  same staining, different scanner
  inter-staining same scanner, different staining
  both           both differ
  all            every pair
Retention = student mean / teacher mean, regime by regime.

Usage:
  python3 scripts/plism_student_retention.py \
      --out outputs/plism/student_retention.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from paper_cohort import keep_teacher

N_PAIRS = 4095
# plismbench teacher id -> student run directory
STUDENT_DIR = {"phikon": "phikon", "phikonv2": "phikon-v2", "conch": "conch",
               "uni": "uni", "uni2h": "uni2-h", "virchow": "virchow",
               "virchow2": "virchow2", "hoptimus0": "h-optimus-0",
               "provgigapath": "prov-gigapath", "midnight_12k": "midnight"}
STUDENTS = ("vit-tiny", "vit-small", "vit-base")
REGIMES = ("inter-scanner", "inter-staining", "both", "all")


def regime_means(path, metric="top_1_accuracy"):
    rows = list(csv.DictReader(open(path)))
    if len(rows) != N_PAIRS:
        raise SystemExit(f"{path}: {len(rows)} pairs, expected {N_PAIRS}")
    v = np.array([float(r[metric]) for r in rows])
    st = np.array([r["staining_a"] == r["staining_b"] for r in rows])
    sc = np.array([r["scanner_a"] == r["scanner_b"] for r in rows])
    masks = {"inter-scanner": st & ~sc, "inter-staining": ~st & sc,
             "both": ~st & ~sc, "all": np.ones(len(v), bool)}
    return {k: float(v[masks[k]].mean()) for k in REGIMES}


def summarise(rows):
    """Panel-level statements the manuscript makes about student robustness."""
    from scipy.stats import spearmanr
    by_t = {}
    for r in rows:
        by_t.setdefault(r["teacher"], {})[r["student"]] = r
    ret = {s: [r["retention"]["all"] for r in rows if r["student"] == s] for s in STUDENTS}
    mono = {k: [t for t, d in by_t.items()
                if d["vit-tiny"]["student_top1"][k] < d["vit-small"]["student_top1"][k]
                < d["vit-base"]["student_top1"][k]] for k in REGIMES}
    above = [(r["teacher"], r["student"]) for r in rows
             if r["student_top1"]["all"] > r["teacher_top1"]["all"]]
    sc_gt_st = [(r["teacher"], r["student"]) for r in rows
                if r["retention"]["inter-scanner"] > r["retention"]["inter-staining"]]
    ratio = [r["student_top1"]["inter-scanner"] / r["student_top1"]["inter-staining"]
             for r in rows]
    tb = [(d["vit-base"]["teacher_top1"]["all"], d["vit-base"]["student_top1"]["all"])
          for d in by_t.values()]
    rho, p = spearmanr(*zip(*tb))
    return {
        "median_retention_all": {s: float(np.median(v)) for s, v in ret.items()},
        "range_retention_all": {s: [float(min(v)), float(max(v))] for s, v in ret.items()},
        "monotone_teachers": mono,
        "students_above_teacher_all": above,
        "scanner_retention_gt_staining": sc_gt_st,
        "student_scanner_over_staining_ratio": [float(min(ratio)), float(max(ratio))],
        "teacher_vs_vitb_all_spearman": [float(rho), float(p), len(tb)],
        # retention has the teacher in its denominator, so part of any
        # negative association with teacher top-1 is mechanical
        "teacher_vs_retention_spearman": {
            s: [float(x) for x in spearmanr(
                [r["teacher_top1"]["all"] for r in rows if r["student"] == s],
                [r["retention"]["all"] for r in rows if r["student"] == s])] + [len(by_t)]
            for s in STUDENTS},
        "vitb_parity_or_above_are_lowest_teachers": sorted(
            t for t, d in by_t.items() if d["vit-base"]["retention"]["all"] >= 0.995) == sorted(
            sorted(by_t, key=lambda t: by_t[t]["vit-base"]["teacher_top1"]["all"])[:4]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/plism/results/8139_tiles")
    ap.add_argument("--out", default="outputs/plism/student_retention.json")
    a = ap.parse_args()
    R = Path(a.root)
    out = []
    for t, sd in STUDENT_DIR.items():
        if not keep_teacher(t):
            continue
        tm = regime_means(R / t / "metrics.csv")
        for s in STUDENTS:
            sm = regime_means(R / sd / s / "metrics.csv")
            out.append({"teacher": t, "student": s, "teacher_top1": tm,
                        "student_top1": sm,
                        "retention": {k: sm[k] / tm[k] for k in REGIMES}})
    json.dump({"n_pairs": N_PAIRS, "metric": "top_1_accuracy", "rows": out,
               "summary": summarise(out)}, open(a.out, "w"), indent=1)
    print(f"[wrote] {a.out} ({len(out)} pairs)")


if __name__ == "__main__":
    main()
