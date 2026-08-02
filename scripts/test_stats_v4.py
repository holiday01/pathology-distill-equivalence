"""Literature-fidelity tests for scripts/stats_v4.py.

References:
  [Schuirmann1987] Schuirmann, D.J., "A comparison of the two one-sided tests
     procedure and the power approach for assessing the equivalence of average
     bioavailability", J Pharmacokinet Biopharm 1987.  Classical TOST.
  [Davison1997] Davison & Hinkley, "Bootstrap Methods and Their Application",
     Cambridge, 1997.  Cluster bootstrap §3.8 / §8.3.
  [BH1995] Benjamini & Hochberg, "Controlling the False Discovery Rate",
     JRSS-B 1995.  BH step-up procedure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from stats_v4 import (
    hierarchical_bh,
    slide_cluster_bootstrap,
    tost_equivalence,
)

_failures = []
_passes = 0


def check(label, cond, ref, detail=""):
    global _passes
    tag = "[PASS]" if cond else "[FAIL]"
    print(f"{tag}  {label}  (ref: {ref})")
    if detail:
        print(f"        {detail}")
    if cond:
        _passes += 1
    else:
        _failures.append(label)


# ---------------------------------------------------------------------------
# TOST
# ---------------------------------------------------------------------------

def test_tost_equivalent_case():
    """If true difference is zero (or small), TOST should reject both H0s
    with margin 0.03 given sufficient sample size."""
    rng = np.random.default_rng(0)
    n_slides, per_slide = 40, 60
    groups = np.repeat([f"s{i:03d}" for i in range(n_slides)], per_slide)
    # No bias: student and teacher both correct at rate 0.85, independent
    t = rng.binomial(1, 0.85, size=n_slides * per_slide).astype(float)
    s = rng.binomial(1, 0.85, size=n_slides * per_slide).astype(float)
    r = tost_equivalence(t, s, groups, margin=0.03, n_boot=500, seed=0)
    check(
        "TOST rejects both sides when true Δ ≈ 0 within ±0.03",
        r["equivalent"] and r["p_lower"] < 0.05 and r["p_upper"] < 0.05,
        "Schuirmann1987",
        f"Δ={r['d_mean']:+.4f}  p_L={r['p_lower']:.3f}  p_U={r['p_upper']:.3f}",
    )


def test_tost_rejects_when_outside_margin():
    """If student is 10% worse than teacher, margin=0.03 → not equivalent."""
    rng = np.random.default_rng(0)
    n_slides, per_slide = 40, 60
    groups = np.repeat([f"s{i:03d}" for i in range(n_slides)], per_slide)
    t = rng.binomial(1, 0.90, size=n_slides * per_slide).astype(float)
    s = rng.binomial(1, 0.80, size=n_slides * per_slide).astype(float)
    r = tost_equivalence(t, s, groups, margin=0.03, n_boot=500, seed=0)
    check(
        "TOST does NOT conclude equivalence when true Δ=-0.10 > margin",
        not r["equivalent"],
        "Schuirmann1987",
        f"Δ={r['d_mean']:+.4f}  equiv={r['equivalent']}",
    )


def test_tost_symmetry():
    """Margin symmetry: swapping teacher/student flips d_mean sign,
    p_lower and p_upper swap."""
    rng = np.random.default_rng(0)
    n_slides, per_slide = 30, 40
    groups = np.repeat([f"s{i:03d}" for i in range(n_slides)], per_slide)
    t = rng.binomial(1, 0.80, size=n_slides * per_slide).astype(float)
    s = rng.binomial(1, 0.85, size=n_slides * per_slide).astype(float)
    r_fwd = tost_equivalence(t, s, groups, margin=0.05, n_boot=400, seed=42)
    r_rev = tost_equivalence(s, t, groups, margin=0.05, n_boot=400, seed=42)
    ok = (abs(r_fwd["d_mean"] + r_rev["d_mean"]) < 1e-9
          and abs(r_fwd["p_lower"] - r_rev["p_upper"]) < 0.05
          and abs(r_fwd["p_upper"] - r_rev["p_lower"]) < 0.05)
    check(
        "TOST has correct sign/side symmetry under teacher↔student swap",
        ok,
        "Schuirmann1987",
        f"fwd Δ={r_fwd['d_mean']:+.3f}  rev Δ={r_rev['d_mean']:+.3f}",
    )


# ---------------------------------------------------------------------------
# Slide-cluster bootstrap (Davison & Hinkley 1997)
# ---------------------------------------------------------------------------

def test_cluster_bootstrap_se_greater_than_naive():
    """Intracluster correlation → cluster SE > naive patch SE.
    (Classic cluster-sampling result: ignoring clustering under-estimates SE.)"""
    rng = np.random.default_rng(0)
    n_slides, per_slide = 30, 100
    slide_effects = rng.normal(0, 1, size=n_slides)  # strong intracluster corr
    groups = np.repeat([f"s{i:03d}" for i in range(n_slides)], per_slide)
    x = np.repeat(slide_effects, per_slide) + 0.2 * rng.normal(size=n_slides * per_slide)
    # cluster bootstrap
    r_cluster = slide_cluster_bootstrap(x, groups, np.mean, n_boot=500, seed=0)
    # naive patch bootstrap
    naive_stats = []
    rng2 = np.random.default_rng(0)
    for _ in range(500):
        idx = rng2.integers(0, len(x), size=len(x))
        naive_stats.append(x[idx].mean())
    naive_se = float(np.std(naive_stats, ddof=1))
    ok = r_cluster["se"] > 3 * naive_se
    check(
        "cluster-bootstrap SE >> naive bootstrap SE under intracluster corr",
        ok,
        "Davison1997 §3.8",
        f"cluster_SE={r_cluster['se']:.4f}  naive_SE={naive_se:.4f}  ratio={r_cluster['se']/naive_se:.1f}×",
    )


def test_cluster_bootstrap_matches_without_clustering():
    """With no intracluster correlation (random pure noise), cluster SE
    should be close to naive SE."""
    rng = np.random.default_rng(0)
    n_slides, per_slide = 60, 50
    groups = np.repeat([f"s{i:03d}" for i in range(n_slides)], per_slide)
    x = rng.normal(size=n_slides * per_slide)
    r = slide_cluster_bootstrap(x, groups, np.mean, n_boot=500, seed=0)
    naive_se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    # ratio should be O(1) (not orders of magnitude off)
    ratio = r["se"] / naive_se if naive_se > 0 else 0
    check(
        "cluster-bootstrap SE ≈ analytic SE when intra-cluster var = 0",
        0.5 < ratio < 2.0,
        "Davison1997 §3.8",
        f"cluster_SE={r['se']:.4f}  analytic_SE={naive_se:.4f}  ratio={ratio:.2f}",
    )


def test_cluster_bootstrap_mean_unbiased():
    """Resample mean should match empirical mean."""
    rng = np.random.default_rng(0)
    n_slides = 50
    groups = np.repeat([f"s{i:03d}" for i in range(n_slides)], 30)
    x = rng.normal(2.0, 1.0, size=n_slides * 30)
    r = slide_cluster_bootstrap(x, groups, np.mean, n_boot=500, seed=0)
    emp = float(np.mean(x))
    check(
        "cluster-bootstrap central statistic equals sample statistic",
        abs(r["mean"] - emp) < 1e-12,
        "Davison1997",
        f"boot mean={r['mean']:.4f}  empirical={emp:.4f}",
    )


# ---------------------------------------------------------------------------
# Benjamini–Hochberg (BH 1995)
# ---------------------------------------------------------------------------

def test_bh_rejection_at_alpha():
    """For p-values [0.001, 0.01, 0.03, 0.05, 0.5], alpha=0.05 → BH rejects
    top 3 (p ≤ 0.05 × k/n for k=1,2,3)."""
    pvals = [0.001, 0.01, 0.03, 0.05, 0.50]
    r = hierarchical_bh({"t0": pvals}, alpha=0.05)
    reject = r["rejected"]["t0"]
    # Expect: indices with p ≤ i*alpha/n where n=5, alpha=0.05 → thresh [0.01,0.02,0.03,0.04,0.05]
    # p sorted = same order: 0.001 ≤ 0.01 ✓, 0.01 ≤ 0.02 ✓, 0.03 ≤ 0.03 ✓, 0.05 ≤ 0.04? ✗ but BH steps up
    # largest k where sorted_p[k] ≤ k*alpha/n: check each
    #   k=1: 0.001 ≤ 0.01 ✓
    #   k=2: 0.01 ≤ 0.02 ✓
    #   k=3: 0.03 ≤ 0.03 ✓
    #   k=4: 0.05 ≤ 0.04 ✗
    # → reject top 3
    expected = [True, True, True, False, False]
    check(
        "BH with step-up rejects top-3 for example pvals",
        reject == expected,
        "BH1995",
        f"got {reject}  expected {expected}",
    )


def test_bh_all_null():
    """Uniform p-values (all null) → expected # rejections ≤ alpha × m."""
    rng = np.random.default_rng(0)
    m = 200
    pvals = rng.uniform(size=m).tolist()
    r = hierarchical_bh({"t": pvals}, alpha=0.05)
    n_reject = sum(r["rejected"]["t"])
    check(
        "BH controls FDR on uniform null pvals (<= expected at alpha)",
        n_reject <= int(0.05 * m) + 3,  # small slack
        "BH1995",
        f"rejected {n_reject}/{m} at alpha=0.05 (expected ≤ {int(0.05*m)})",
    )


def test_bh_strong_signal():
    """All tiny p-values → all rejected."""
    pvals = [1e-5] * 10
    r = hierarchical_bh({"t": pvals}, alpha=0.05)
    check(
        "BH rejects all when every pval is tiny",
        all(r["rejected"]["t"]),
        "BH1995",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tests = [
        test_tost_equivalent_case,
        test_tost_rejects_when_outside_margin,
        test_tost_symmetry,
        test_cluster_bootstrap_se_greater_than_naive,
        test_cluster_bootstrap_matches_without_clustering,
        test_cluster_bootstrap_mean_unbiased,
        test_bh_rejection_at_alpha,
        test_bh_all_null,
        test_bh_strong_signal,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            _failures.append(f"{t.__name__}: EXCEPTION {e}")
            print(f"[FAIL]  {t.__name__}  (EXCEPTION: {e})")
    total = _passes + len(_failures)
    print(f"\n{_passes}/{total} individual checks passed")
    if _failures:
        print("FAILURES:")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("All stats_v4 literature-fidelity checks PASSED.")
    sys.exit(0)


if __name__ == "__main__":
    main()
