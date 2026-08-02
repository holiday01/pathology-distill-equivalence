#!/usr/bin/env python3
"""Task 1.3 — GPFM-style average-rank across the distillation atlas.

For each student size (vit-tiny / vit-small / vit-base) we have up to 12
teachers. Within that field we rank teachers on each equivalence/quality
metric (higher = better unless noted) and average the ranks, the way GPFM
reports average-rank/N. Lower average rank = better.

Metrics (per (teacher,student) pair, from downstream_report.json):
  cka          feature_similarity.cka_overall            (higher better)
  s_acc        probes.c16_linear.student_acc             (higher better)
  dacc_close   -|probes.c16_linear.tost.d_mean|          (closer to 0 better)
  kappa        probes.c16_linear.cohen_kappa             (higher better)
  speedup      efficiency.speedup                        (higher better)

Outputs:
  outputs/v4_full/average_rank.csv   (per student-size, per teacher)
  outputs/v4_full/average_rank.md    (ranked tables + overall mean rank)
"""
import json, glob, csv
import numpy as np
from collections import defaultdict

ROOT = "outputs/v4_full"
# (key, extractor, higher_is_better)
METRICS = [
    ("cka",        lambda r: r["feature_similarity"]["cka_overall"], True),
    ("s_acc",      lambda r: r["probes"]["c16_linear"]["student_acc"], True),
    ("dacc_close", lambda r: -abs(r["probes"]["c16_linear"]["tost"]["d_mean"]), True),
    ("kappa",      lambda r: r["probes"]["c16_linear"]["cohen_kappa"], True),
    ("speedup",    lambda r: r["efficiency"]["speedup"], True),
]


def load():
    data = defaultdict(dict)  # student -> teacher -> {metric: val}
    for f in sorted(glob.glob(f"{ROOT}/*/*/downstream_report.json")):
        parts = f.split("/")
        teacher, student = parts[-3], parts[-2]
        r = json.load(open(f)).get("results", {})
        vals = {}
        for key, fn, _ in METRICS:
            try:
                v = fn(r)
                vals[key] = float(v) if v is not None and np.isfinite(v) else None
            except (KeyError, TypeError):
                vals[key] = None
        data[student][teacher] = vals
    return data


def rank_within(values, higher_better):
    """Return dict teacher->rank (1=best). Ties share the average rank.
    Teachers with None are dropped from that metric's ranking."""
    items = [(t, v) for t, v in values.items() if v is not None]
    if not items:
        return {}
    arr = np.array([v for _, v in items], dtype=float)
    order = (-arr) if higher_better else arr
    # average-rank for ties
    ranks = order.argsort().argsort().astype(float)
    # convert to competition-friendly average ranks (1-based, tie-averaged)
    sorted_vals = np.sort(order)
    rmap = {}
    for t, v in items:
        pos = np.where(sorted_vals == (-v if higher_better else v))[0]
        rmap[t] = float(pos.mean() + 1)
    return rmap


def main():
    data = load()
    if not data:
        print("no reports yet"); return
    rows = []
    md = ["# Average rank across the distillation atlas (GPFM-style)\n",
          "Lower = better. Rank computed within each student-size field of "
          "teachers; averaged over CKA, student C16-linear accuracy, |Δacc| "
          "closeness, Cohen's κ, and speedup.\n"]
    overall = defaultdict(list)
    for student in sorted(data):
        teachers = data[student]
        per_metric = {}
        for key, _, hb in METRICS:
            per_metric[key] = rank_within({t: v[key] for t, v in teachers.items()}, hb)
        md.append(f"\n## {student}  (field of {len(teachers)} teachers)\n")
        md.append("| teacher | " + " | ".join(k for k, _, _ in METRICS) + " | avg_rank |")
        md.append("|---|" + "|".join("---" for _ in METRICS) + "|---|")
        avg_list = []
        for t in teachers:
            rs = [per_metric[k].get(t) for k, _, _ in METRICS]
            present = [x for x in rs if x is not None]
            avg = float(np.mean(present)) if present else None
            avg_list.append((t, avg, rs))
            if avg is not None:
                overall[t].append(avg)
            rows.append({"student": student, "teacher": t,
                         **{k: (per_metric[k].get(t)) for k, _, _ in METRICS},
                         "avg_rank": avg})
        for t, avg, rs in sorted(avg_list, key=lambda x: (x[1] is None, x[1])):
            cells = " | ".join(f"{x:.1f}" if x is not None else "·" for x in rs)
            md.append(f"| {t} | {cells} | {avg:.2f} |" if avg is not None
                      else f"| {t} | {cells} | · |")

    md.append("\n## Overall mean rank (averaged across student sizes)\n")
    md.append("| teacher | mean_avg_rank | n_sizes |")
    md.append("|---|---|---|")
    for t, lst in sorted(overall.items(), key=lambda x: np.mean(x[1])):
        md.append(f"| {t} | {np.mean(lst):.2f} | {len(lst)} |")

    cols = ["student", "teacher"] + [k for k, _, _ in METRICS] + ["avg_rank"]
    with open(f"{ROOT}/average_rank.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})
    open(f"{ROOT}/average_rank.md", "w").write("\n".join(md) + "\n")
    print(f"[wrote] {ROOT}/average_rank.csv  ({len(rows)} rows)")
    print(f"[wrote] {ROOT}/average_rank.md")
    best = min(overall.items(), key=lambda x: np.mean(x[1]))
    print(f"best overall mean rank: {best[0]} = {np.mean(best[1]):.2f}")


if __name__ == "__main__":
    main()
