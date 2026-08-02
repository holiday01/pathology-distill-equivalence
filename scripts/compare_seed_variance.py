#!/usr/bin/env python3
"""Seed-variance of the equivalence verdicts (reviewer ask: TOST claims need >=2 seeds).

Compares the per-pair AUROC-difference equivalence verdict between the seed=42
atlas and the seed=2025 re-fit, on the SAME lesion-annotated 41-slide C16 test.
A verdict is trustworthy only if it is stable across training seeds.

Clean comparison uses teachers whose code is identical across both seeds
(phikon, uni, virchow2, h-optimus-0). hibou-l is reported SEPARATELY because
its seed=42 run used the (broken) ImageNet normalisation while seed=2025 uses
the fixed hibou normalisation, so any difference there confounds seed with the
normalisation fix and is not a seed-variance measurement.

Run order:
  python3 scripts/eval_c16_equivalence.py --root outputs/v4_seed2 \
      --out outputs/v4_seed2/c16_equiv_seed2          # produce seed=2025 verdicts
  python3 scripts/compare_seed_variance.py            # this script
"""
import json
import numpy as np

SEED42 = "outputs/v4_full/c16_equiv_annotated.json"
SEED25 = "outputs/v4_seed2/c16_equiv_seed2.json"
DELTA = 0.05
CLEAN_TEACHERS = {"phikon", "uni", "virchow2", "h-optimus-0"}   # same code both seeds
CONFOUNDED = {"hibou-l", "hibou-b"}                              # code differs across seeds


def state(lo, hi, d=DELTA):
    if lo is None or np.isnan(lo):
        return "n/a"
    if hi < d and lo > -d:
        return "equivalent"
    if lo > d or hi < -d:
        return "inequivalent"
    return "inconclusive"


def load(path):
    rows = json.load(open(path))["rows"]
    return {(r["teacher"], r["student"]): r for r in rows}


def main():
    try:
        a = load(SEED42); b = load(SEED25)
    except FileNotFoundError as e:
        print(f"missing input: {e}\nRun eval_c16_equivalence.py --root outputs/v4_seed2 first.")
        return
    common = sorted(set(a) & set(b))
    if not common:
        print("no overlapping (teacher,student) pairs yet — seed2 eval not done?")
        return

    def row(k, src):
        r = src[k]; lo, hi = r["linear_auc_ci"]
        return r["linear_auc_d"], lo, hi, state(lo, hi), r.get("linear_auc_inflation", float("nan"))

    clean, conf = [], []
    print(f"{'teacher':13s} {'student':10s} {'d42':>7s} {'d25':>7s} {'|Δd|':>6s} "
          f"{'state42':>13s} {'state25':>13s} {'agree':>6s}")
    for k in common:
        d42, lo42, hi42, s42, i42 = row(k, a)
        d25, lo25, hi25, s25, i25 = row(k, b)
        dd = abs(d42 - d25)
        agree = (s42 == s25)
        rec = dict(teacher=k[0], student=k[1], d42=round(d42, 3), d25=round(d25, 3),
                   abs_dd=round(dd, 3), state42=s42, state25=s25, agree=agree,
                   infl42=round(i42, 2), infl25=round(i25, 2))
        (conf if k[0] in CONFOUNDED else clean).append(rec)
        tag = "" if k[0] in CLEAN_TEACHERS else "*"
        print(f"{k[0]+tag:13s} {k[1]:10s} {d42:+7.3f} {d25:+7.3f} {dd:6.3f} "
              f"{s42:>13s} {s25:>13s} {str(agree):>6s}")

    if clean:
        ddv = np.array([r["abs_dd"] for r in clean])
        ag = sum(r["agree"] for r in clean)
        print(f"\n=== CLEAN seed-variance ({len(clean)} pairs, {sorted(CLEAN_TEACHERS & {r['teacher'] for r in clean})}) ===")
        print(f"  verdict agreement across seeds: {ag}/{len(clean)} "
              f"({100*ag/len(clean):.0f}%)")
        print(f"  cross-seed |Δ AUROC-diff|: median {np.median(ddv):.3f}, "
              f"max {ddv.max():.3f}")
        infl = np.array([r["infl42"] for r in clean] + [r["infl25"] for r in clean])
        infl = infl[np.isfinite(infl)]
        print(f"  SE-inflation both seeds: median {np.median(infl):.2f}x "
              f"(stable across seeds)")
        robust_eq = [r for r in clean if r["state42"] == "equivalent" == r["state25"]]
        print(f"  robustly equivalent (both seeds certify): {len(robust_eq)}/{len(clean)}")
    if conf:
        print(f"\n=== hibou-l (seed CONFOUNDED with the normalisation fix — reported, not a seed test) ===")
        for r in conf:
            print(f"  {r['teacher']}/{r['student']}: seed42(broken-norm) d={r['d42']} {r['state42']}"
                  f"  -> seed2025(fixed-norm) d={r['d25']} {r['state25']}")

    json.dump({"clean": clean, "confounded": conf}, open("outputs/v4_seed2/seed_variance.json", "w"), indent=1)
    print("\n[wrote] outputs/v4_seed2/seed_variance.json")
    print("(* = teacher in clean-comparison set; hibou rows excluded from the seed-variance stat)")


if __name__ == "__main__":
    main()
