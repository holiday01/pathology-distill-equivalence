#!/usr/bin/env python3
"""Side-by-side comparison of the 82-slide results the manuscript currently
reports against the 270-slide re-run.

Reads only; writes nothing. Every figure printed here is recomputed from the
result JSONs rather than read out of the manuscript, so a disagreement between
this table and the paper is a real disagreement.

Usage:
  python3 scripts/compare_82_vs_270.py
"""
import json
import statistics as st
from collections import Counter
from pathlib import Path

OLD = Path("outputs/v4_full")
NEW = Path("outputs/v4_full/c16_270")
MARGIN = 0.05


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def three_state(lo, hi, margin=MARGIN):
    import math
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return "inconclusive"
    if lo > -margin and hi < margin:
        return "equivalent"
    if lo >= margin or hi <= -margin:
        return "inequivalent"
    return "inconclusive"


def row(label, old, new, fmt="{}"):
    o = fmt.format(old) if old is not None else "--"
    n = fmt.format(new) if new is not None else "PENDING"
    flag = ""
    if isinstance(old, (int, float)) and isinstance(new, (int, float)) and old:
        d = (new - old) / abs(old) * 100
        flag = f"  ({d:+.0f}%)" if abs(d) >= 1 else "  (=)"
    print(f"  {label:44} {o:>16} {n:>16}{flag}")


def equiv_stats(rows):
    if not rows:
        return {}
    states = Counter(three_state(*r["linear_auc_ci"]) for r in rows)
    inf = sorted(r["linear_auc_inflation"] for r in rows
                 if r.get("linear_auc_inflation") is not None)
    q = st.quantiles(inf, n=4) if len(inf) >= 4 else [float("nan")] * 3
    t = {}
    for r in rows:
        if r.get("linear_t_auc") is not None:
            t.setdefault(r["teacher"], r["linear_t_auc"])
    return {
        "n_pairs": len(rows),
        "n_test_slides": rows[0].get("n_test_slides"),
        "n_test_tiles": rows[0].get("n_test_tiles"),
        "equivalent": states["equivalent"],
        "inequivalent": states["inequivalent"],
        "inconclusive": states["inconclusive"],
        "infl_median": st.median(inf) if inf else None,
        "infl_q1": q[0], "infl_q3": q[2],
        "infl_min": min(inf) if inf else None, "infl_max": max(inf) if inf else None,
        "teacher_mean_auc": st.mean(t.values()) if t else None,
        "verdicts": {(r["teacher"], r["student"]): three_state(*r["linear_auc_ci"])
                     for r in rows},
    }


def main():
    old_rows = (load(OLD / "c16_equiv_annotated.json") or {}).get("rows")
    new_rows = (load(NEW / "c16_equiv_annotated.json") or {}).get("rows")
    o, n = equiv_stats(old_rows), equiv_stats(new_rows)

    print("=" * 82)
    print(f"  {'':44} {'82 slides':>16} {'270 slides':>16}")
    print("=" * 82)
    print("\n-- evaluation cohort " + "-" * 60)
    row("test slides", o.get("n_test_slides"), n.get("n_test_slides"))
    row("test tiles", o.get("n_test_tiles"), n.get("n_test_tiles"))
    row("teacher-student pairs", o.get("n_pairs"), n.get("n_pairs"))

    print("\n-- three-state verdict at delta = 0.05 " + "-" * 42)
    row("equivalent", o.get("equivalent"), n.get("equivalent"))
    row("inequivalent", o.get("inequivalent"), n.get("inequivalent"))
    row("inconclusive  <- power-limited", o.get("inconclusive"), n.get("inconclusive"))

    print("\n-- the headline claim " + "-" * 59)
    row("cluster/naive SE inflation, median", o.get("infl_median"), n.get("infl_median"), "{:.2f}x")
    row("  IQR low", o.get("infl_q1"), n.get("infl_q1"), "{:.2f}x")
    row("  IQR high", o.get("infl_q3"), n.get("infl_q3"), "{:.2f}x")
    row("  range min", o.get("infl_min"), n.get("infl_min"), "{:.2f}x")
    row("  range max", o.get("infl_max"), n.get("infl_max"), "{:.2f}x")
    row("teacher mean tile AUROC", o.get("teacher_mean_auc"), n.get("teacher_mean_auc"), "{:.4f}")

    oi, ni = load(OLD / "intra_slide_icc.json"), load(NEW / "intra_slide_icc.json")
    print("\n-- intra-slide correlation " + "-" * 54)
    row("rho (ICC), CAMELYON16",
        oi and oi["c16"]["median_rho"], ni and ni["c16"]["median_rho"], "{:.4f}")
    row("m (tiles per slide)",
        oi and oi["c16"]["m"], ni and ni["c16"]["m"], "{:.0f}")

    oc, nc = load(OLD / "c16_calibration_annotated.json"), load(NEW / "c16_calibration_annotated.json")
    print("\n-- calibration " + "-" * 66)
    for lbl, key in (("median ECE, teacher", "linear_t_ece"), ("median ECE, student", "linear_s_ece")):
        def med(d):
            if not d:
                return None
            v = [r[key] for r in d["rows"] if r["teacher"] not in ("hibou-b", "hibou-l")]
            return st.median(v) if v else None
        row(lbl, med(oc), med(nc), "{:.4f}")

    print("\n-- external audit " + "-" * 63)
    og, ng = load(OLD / "gpfm_equiv.json"), load(NEW / "gpfm_equiv.json")
    for t in ("conch", "phikon", "uni"):
        def verdict(d):
            if not d:
                return None
            for r in d["rows"]:
                if r["teacher"] == t:
                    return r["both"]
            return None
        print(f"  {'GPFM vs ' + t:44} {str(verdict(og) or '--'):>16} "
              f"{str(verdict(ng) or 'PENDING'):>16}")

    if o.get("verdicts") and n.get("verdicts"):
        moved = [(k, o["verdicts"][k], n["verdicts"][k])
                 for k in o["verdicts"] if k in n["verdicts"]
                 and o["verdicts"][k] != n["verdicts"][k]]
        print(f"\n-- per-pair verdict changes: {len(moved)} of {len(o['verdicts'])} " + "-" * 30)
        for (te, stu), a, b in sorted(moved):
            print(f"  {te + ' x ' + stu:44} {a:>16} -> {b}")
        if not moved:
            print("  (no pair changed verdict)")
    print()


if __name__ == "__main__":
    main()
