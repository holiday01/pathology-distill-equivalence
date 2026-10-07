#!/usr/bin/env python3
"""Rewrite only the teacher/student GFLOPs columns of paper_metrics.csv from
outputs/fm_flops.json (compute_fm_flops_consistent.py), leaving every other
column byte-identical. A full make_paper_metrics.py rerun cannot be used for
this: the h-optimus-0 x vit-small energy was taken from a telemetry log that
is not the one the script integrates.
"""
import csv
import json
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/v4_full/paper_metrics.csv")
flops = json.load(open("outputs/fm_flops.json"))
rows = list(csv.DictReader(open(path)))
fields = list(rows[0].keys())
for r in rows:
    t = flops[r["fm"]]["gflops"]
    s = flops[f"student_{r['student'].split('_')[0]}"]["gflops"]
    r["teacher_gflops"], r["student_gflops"], r["gflops_ratio"] = t, s, t / s
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f"updated {len(rows)} rows in {path}")
