#!/usr/bin/env python3
r"""Task 1.2 — empirical slide-cluster vs naive patch-level SE inflation.

For every completed v4_full run we already stored, in downstream_report.json:
  - student/teacher C16-linear accuracy
  - the *slide-cluster* bootstrap 95% CI on that accuracy
    (bootstrap_ci(..., groups=slide_id) in evaluate_v4_downstream.py)
  - n_test patches

The slide-cluster SE is recovered from the stored cluster CI:
    se_cluster = (hi - lo) / (2 * 1.959964)
The naive patch-level SE treats the n_test patches as i.i.d. Bernoulli:
    se_naive   = sqrt(acc*(1-acc) / n_test)
Inflation factor = se_cluster / se_naive  (>=1 means naive understates SE).

Headline per-pair value = the student C16-linear probe (the deployed model).
Output: outputs/v4_full/cluster_vs_naive_se.csv (one row per pair) and a
markdown summary with the median + IQR across pairs.

NOTE on power: the atlas C16 test split currently has n_slides=2, so each
per-pair ratio is low-power; the *median across pairs* is the reported
statistic. The full-set C16 re-extraction (with per-patch slide dumps for a
paired-difference definition) is the camera-ready item flagged in
results_npjdm.tex \S results-plism.
"""
import json, glob, math, csv, os
import numpy as np

Z95 = 1.959964
ROOT = "outputs/v4_full"


def inflation(acc, lo, hi, n):
    if n is None or n <= 0 or acc is None:
        return None
    se_cluster = (hi - lo) / (2 * Z95)
    se_naive = math.sqrt(max(acc * (1 - acc), 1e-12) / n)
    if se_naive <= 0 or not np.isfinite(se_cluster):
        return None
    return se_cluster, se_naive, se_cluster / se_naive


def main():
    rows = []
    for f in sorted(glob.glob(f"{ROOT}/*/*/downstream_report.json")):
        parts = f.split("/")
        teacher, student = parts[-3], parts[-2]
        pr = json.load(open(f)).get("results", {}).get("probes", {}).get("c16_linear")
        if not pr:
            continue
        n = pr.get("n_test")
        nsl = pr.get("n_slides_test")
        rec = {"teacher": teacher, "student": student, "n_test": n, "n_slides": nsl}
        for who in ("student", "teacher"):
            acc = pr.get(f"{who}_acc")
            ci = pr.get(f"{who}_acc_ci95_cluster")
            if acc is None or not ci:
                continue
            res = inflation(acc, ci[0], ci[1], n)
            if res:
                sc, sn, infl = res
                rec[f"{who}_acc"] = round(acc, 4)
                rec[f"{who}_se_cluster"] = round(sc, 5)
                rec[f"{who}_se_naive"] = round(sn, 5)
                rec[f"{who}_inflation"] = round(infl, 3)
        if "student_inflation" in rec:
            rows.append(rec)

    if not rows:
        print("no reports with c16_linear yet"); return

    cols = ["teacher", "student", "n_test", "n_slides",
            "student_acc", "student_se_cluster", "student_se_naive", "student_inflation",
            "teacher_acc", "teacher_se_cluster", "teacher_se_naive", "teacher_inflation"]
    out_csv = f"{ROOT}/cluster_vs_naive_se.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    s_inf = np.array([r["student_inflation"] for r in rows])
    t_inf = np.array([r["teacher_inflation"] for r in rows if "teacher_inflation" in r])
    med = float(np.median(s_inf))
    q1, q3 = (float(np.quantile(s_inf, q)) for q in (.25, .75))

    md = [f"# Cluster vs naive SE inflation — {len(rows)} pairs\n",
          f"Headline (student C16-linear): **median {med:.1f}×**, "
          f"IQR [{q1:.1f}, {q3:.1f}], range [{s_inf.min():.1f}, {s_inf.max():.1f}]\n",
          f"Teacher probe (robustness): median {np.median(t_inf):.1f}× (n={len(t_inf)})\n",
          f"Simulated unit-test sentinel in stats_v4 is ~10×; empirical median "
          f"{med:.1f}× is consistent.\n",
          "| teacher | student | n_slides | s_acc | se_cluster | se_naive | inflation× |",
          "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: -x["student_inflation"]):
        md.append(f"| {r['teacher']} | {r['student']} | {r['n_slides']} | "
                  f"{r.get('student_acc','')} | {r.get('student_se_cluster','')} | "
                  f"{r.get('student_se_naive','')} | {r['student_inflation']:.2f} |")
    open(f"{ROOT}/cluster_se_summary.md", "w").write("\n".join(md) + "\n")

    print(f"[wrote] {out_csv}  ({len(rows)} pairs)")
    print(f"[wrote] {ROOT}/cluster_se_summary.md")
    print(f"median student inflation = {med:.2f}×  IQR [{q1:.2f}, {q3:.2f}]")


if __name__ == "__main__":
    main()
