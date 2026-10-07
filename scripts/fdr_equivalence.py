#!/usr/bin/env python3
"""Multiplicity control for the pair-level equivalence verdicts.

Families are (endpoint, student size): each holds the TOST p-values of the
panel's teachers distilled into that student size, for one equivalence
endpoint. Endpoints are the AUROC equivalence tests that were actually run:
CAMELYON16 linear and MLP probes, and Kather-MSI linear and MLP probes.

  * Benjamini-Bogomolov (2014): families selected by BH on Simes p-values
    at q; BH within each selected family at level R*q/m.
  * Benjamini-Yekutieli (2001) over the 30 headline (C16 linear) p-values,
    as a conservative check valid under arbitrary dependence.

The per-pair p-value is the bootstrap TOST p-value max(p_lower, p_upper)
written by eval_c16_equivalence.py / eval_kather_msi_equivalence.py.

Usage:
  python3 scripts/fdr_equivalence.py --c16 <c16 json> --kather <kather json> \
      --out outputs/v4_full/c16_270/fdr_equivalence.json
"""
import argparse
import json
from collections import Counter

from paper_cohort import filter_rows
from stats_v4 import benjamini_bogomolov, benjamini_yekutieli

Q = 0.05
STUDENTS = ("vit-tiny", "vit-small", "vit-base")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c16", required=True)
    ap.add_argument("--kather", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cohorts = {"c16": filter_rows(json.load(open(a.c16))["rows"]),
               "kather": filter_rows(json.load(open(a.kather))["rows"])}
    fams, keys = {}, {}
    for coh, rows in cohorts.items():
        for probe in ("linear", "mlp"):
            for stu in STUDENTS:
                rs = sorted((r for r in rows if r["student"] == stu),
                            key=lambda r: r["teacher"])
                name = f"{coh}/{probe}/{stu}"
                fams[name] = [r[f"{probe}_auc_p_tost"] for r in rs]
                keys[name] = [r["teacher"] for r in rs]

    bb = benjamini_bogomolov(fams, q=Q)

    head = [(stu, t, p) for stu in STUDENTS
            for t, p in zip(keys[f"c16/linear/{stu}"], fams[f"c16/linear/{stu}"])]
    by = benjamini_yekutieli([p for _, _, p in head], q=Q)
    unadj = sum(p < Q for _, _, p in head)

    headline_bb = sum(sum(bb["rejected"][f"c16/linear/{s}"]) for s in STUDENTS)
    out = {
        "q": Q,
        "n_families": len(fams),
        "n_selected": bb["n_selected"],
        "within_level": bb["within_level"],
        "families": {k: {"teachers": keys[k], "p_tost": fams[k],
                         "simes_p": bb["family_simes_p"][k],
                         "selected": bb["selected"][k],
                         "rejected": bb["rejected"][k]} for k in fams},
        "headline_c16_linear": {
            "n_pairs": len(head),
            "equivalent_unadjusted_p_lt_q": unadj,
            "equivalent_bb": headline_bb,
            "equivalent_by": int(by.sum()),
            "max_p_tost": max(p for _, _, p in head),
        },
        "equivalent_bb_by_family": {k: int(sum(v)) for k, v in bb["rejected"].items()},
    }
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out["headline_c16_linear"], indent=1))
    print(Counter({k: v for k, v in out["equivalent_bb_by_family"].items()}))


if __name__ == "__main__":
    main()
