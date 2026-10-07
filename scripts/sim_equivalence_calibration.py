#!/usr/bin/env python3
"""Calibration simulation for the slide-cluster equivalence test.

Demonstrates the paper's central claim directly: when the true AUROC
difference sits exactly at the margin boundary (truly NOT inside the
equivalence zone), a TOST that uses the slide-cluster standard error is
calibrated -- it falsely declares equivalence at the nominal rate alpha --
whereas a TOST that uses the naive patch-level standard error (understated
by the empirical design-effect inflation) is anti-conservative, declaring
equivalence far more often than alpha. Teacher-level dependence is included
to show the per-comparison rate is unaffected by it.

At the boundary d_true = delta, a one-sided alpha TOST declares equivalence
iff d + z_alpha * se < delta, i.e. with prob alpha when se is correct. If se
is understated by a factor f (= cluster_se / naive_se), the effective z is
inflated and the false-equivalence probability rises well above alpha.
"""
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from paper_cohort import filter_rows

RES = Path("outputs/v4_full/c16_270")

DELTA = 0.05
_PANEL = filter_rows(json.load(open(RES / "c16_equiv_annotated.json"))["rows"])
# correct slide-cluster SE: median over the panel, from the 90% percentile CI
SE_CLUSTER = st.median((r["linear_auc_ci"][1] - r["linear_auc_ci"][0]) / (2 * 1.6448536)
                       for r in _PANEL)
# empirical median cluster/naive inflation on the reported panel (135 slides)
INFLATION = st.median(r["linear_auc_inflation"] for r in _PANEL)
TAU = 1.2 * SE_CLUSTER     # teacher-shared effect sd, NOT included in the SE
N_TEACH, N_STU = 10, 3
N_REP = 200000
Z90 = stats.norm.ppf(0.95)  # one-sided alpha=0.05 -> 90% CI


def false_equiv_rate(se_used, with_dependence, seed=1):
    """Pairs truly at the boundary (d_true = delta); rate of declaring equivalence."""
    rng = np.random.default_rng(seed)
    rep = N_REP
    if with_dependence:
        u = rng.normal(0, TAU, size=(rep, N_TEACH, 1))
        eps = rng.normal(0, SE_CLUSTER, size=(rep, N_TEACH, N_STU))
        d = DELTA + u + eps
    else:
        d = DELTA + rng.normal(0, SE_CLUSTER, size=(rep, N_TEACH, N_STU))
    # equivalence iff 90% CI (using se_used) within [-delta, delta]
    hi = d + Z90 * se_used
    lo = d - Z90 * se_used
    equ = (hi < DELTA) & (lo > -DELTA)
    return equ.mean()


def main():
    naive_se = SE_CLUSTER / INFLATION
    print(f"delta={DELTA}, cluster_se={SE_CLUSTER}, inflation={INFLATION} -> naive_se={naive_se:.5f}\n")
    print("false-equivalence rate at the margin boundary (nominal alpha=0.05):")
    out = {"inflation": INFLATION, "n_teachers": N_TEACH, "tau_over_se": TAU / SE_CLUSTER}
    for dep in (False, True):
        r_cluster = false_equiv_rate(SE_CLUSTER, dep)
        r_naive = false_equiv_rate(naive_se, dep)
        tag = "with teacher dependence" if dep else "independent"
        print(f"  [{tag:24s}] cluster-SE TOST: {r_cluster:.3f}   naive-SE TOST: {r_naive:.3f}")
        key = "dep" if dep else "indep"
        out[f"cluster_{key}"] = float(r_cluster)
        out[f"naive_{key}"] = float(r_naive)
    (RES / "sim_equivalence_calibration.json").write_text(json.dumps(out, indent=1))
    print(f"[wrote] {RES / 'sim_equivalence_calibration.json'}")


if __name__ == "__main__":
    main()
