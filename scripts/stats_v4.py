"""Statistical utilities for v4 equivalence testing.

Slide-cluster bootstrap + TOST as specified in reviews/proposal_v4.md.
Rationale: patches from the same WSI are not independent, so naive
patch-level resampling under-estimates variance. Cluster by slide
(group id), resample slides with replacement, and take each slide's
patches whole.

Public API:
    slide_cluster_bootstrap(values, groups, stat_fn, n_boot, seed)
    tost_equivalence(t_correct, s_correct, groups, margin, n_boot, seed)
    hierarchical_bh(per_teacher_pvals, alpha)      (within-family BH only)
    benjamini_bogomolov(family_pvals, q)            (selective two-level FDR)
    benjamini_yekutieli(pvals, q)                   (BH under arbitrary dependence)
"""
from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np


def _by_slide(values: np.ndarray, groups: np.ndarray) -> Dict[str, np.ndarray]:
    out: Dict[str, list] = {}
    for v, g in zip(values, groups):
        out.setdefault(str(g), []).append(v)
    return {k: np.asarray(v) for k, v in out.items()}


def slide_cluster_bootstrap(
    values: np.ndarray,
    groups: np.ndarray,
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> Dict[str, float]:
    """Resample slides (clusters) with replacement.

    Each resample forms a pseudo-dataset by concatenating the patches of
    sampled slides; `stat_fn` is applied once to the pooled array.
    Returns mean + percentile CI + per-resample std for downstream use.
    """
    rng = np.random.default_rng(seed)
    buckets = _by_slide(np.asarray(values), np.asarray(groups))
    slide_ids = np.array(list(buckets.keys()))
    n_slides = len(slide_ids)
    if n_slides < 2:
        m = float(stat_fn(np.asarray(values)))
        return {"mean": m, "lo": m, "hi": m, "se": 0.0, "n_slides": n_slides}

    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_slides, size=n_slides)
        pooled = np.concatenate([buckets[slide_ids[i]] for i in idx])
        stats[b] = stat_fn(pooled)

    alpha = (1 - ci) / 2
    lo = float(np.quantile(stats, alpha))
    hi = float(np.quantile(stats, 1 - alpha))
    return {
        "mean": float(stat_fn(np.asarray(values))),
        "lo": lo,
        "hi": hi,
        "se": float(stats.std(ddof=1)),
        "n_slides": int(n_slides),
    }


def tost_equivalence(
    t_correct: np.ndarray,
    s_correct: np.ndarray,
    groups: np.ndarray,
    margin: float = 0.03,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Two-one-sided-test for equivalence of student vs teacher accuracy.

    Null hypotheses (rejected jointly iff equivalent):
        H0_lower: (s - t) <= -margin
        H0_upper: (s - t) >=  +margin

    Implementation: slide-cluster bootstrap on paired differences
    d_i = s_i - t_i; one-sided bootstrap p-values for each bound.
    Returns both p-values, equivalence decision at alpha=0.05, and the
    90% bootstrap CI (one-sided alpha on each side = standard TOST CI).
    """
    t = np.asarray(t_correct, dtype=float)
    s = np.asarray(s_correct, dtype=float)
    g = np.asarray(groups)
    assert t.shape == s.shape == g.shape, "t, s, groups must match shape"

    d = s - t
    boot = slide_cluster_bootstrap(d, g, np.mean, n_boot=n_boot, seed=seed, ci=0.90)
    d_mean = boot["mean"]
    # Regenerate raw bootstrap means for p-value (need distribution, not CI only)
    rng = np.random.default_rng(seed)
    buckets = _by_slide(d, g)
    slide_ids = np.array(list(buckets.keys()))
    n_slides = len(slide_ids)
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_slides, size=n_slides)
        means[b] = np.concatenate([buckets[slide_ids[i]] for i in idx]).mean()

    # One-sided p-values: shift so that the null-boundary is zero, then
    # compute how often the shifted bootstrap exceeds zero in the wrong
    # direction. This is the Davison-Hinkley pivot approach.
    # H0_lower: true d <= -margin.  Shift d by +margin so boundary=0; p = P(shifted <= 0)
    shifted_lower = means + margin
    p_lower = float(np.mean(shifted_lower <= 0))
    # H0_upper: true d >= +margin.  Shift d by -margin; p = P(shifted >= 0)
    shifted_upper = means - margin
    p_upper = float(np.mean(shifted_upper >= 0))

    alpha = 0.05
    reject_both = (p_lower < alpha) and (p_upper < alpha)

    return {
        "d_mean": d_mean,
        "d_ci90_lo": boot["lo"],  # 5th percentile
        "d_ci90_hi": boot["hi"],  # 95th percentile
        "p_lower": p_lower,
        "p_upper": p_upper,
        "equivalent": bool(reject_both),
        "margin": margin,
        "n_slides": int(n_slides),
    }


def hierarchical_bh(
    per_teacher_pvals: Dict[str, Sequence[float]],
    alpha: float = 0.05,
) -> Dict[str, list]:
    """Two-stage Benjamini-Hochberg:
    Stage 1: BH within each teacher (family) at level alpha.
    Stage 2: Collect teacher-level 'any-rejected' indicator; permutation
             test over teachers not implemented here (needs atlas-level
             null); returns stage-1 rejection mask + teacher-level min-p.

    per_teacher_pvals: {teacher_id: [p1, p2, ...]}
    Returns:
        {teacher_id: [bool, ...]}  rejected at family-wise FDR=alpha
        "teacher_min_p": {teacher_id: min_p_after_stage1}
    """
    def bh(pvals: np.ndarray, alpha: float) -> np.ndarray:
        n = len(pvals)
        order = np.argsort(pvals)
        sorted_p = pvals[order]
        thresh = alpha * (np.arange(1, n + 1) / n)
        below = sorted_p <= thresh
        if not below.any():
            return np.zeros(n, dtype=bool)
        k_star = np.max(np.where(below)[0])
        reject_sorted = np.zeros(n, dtype=bool)
        reject_sorted[: k_star + 1] = True
        out = np.zeros(n, dtype=bool)
        out[order] = reject_sorted
        return out

    rejected = {}
    min_p = {}
    for teacher, pvals in per_teacher_pvals.items():
        arr = np.asarray(pvals, dtype=float)
        rejected[teacher] = bh(arr, alpha).tolist()
        min_p[teacher] = float(arr.min()) if len(arr) else 1.0
    return {"rejected": rejected, "teacher_min_p": min_p}


def _bh_reject(pvals: np.ndarray, level: float) -> np.ndarray:
    """Benjamini-Hochberg step-up at `level`; returns a boolean mask."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    below = p[order] <= level * np.arange(1, n + 1) / n
    out = np.zeros(n, dtype=bool)
    if below.any():
        out[order[: np.max(np.where(below)[0]) + 1]] = True
    return out


def simes(pvals: Sequence[float]) -> float:
    """Simes combination p-value of a family: min_j m p_(j) / j."""
    p = np.sort(np.asarray(pvals, dtype=float))
    m = len(p)
    return float(np.min(m * p / np.arange(1, m + 1)))


def benjamini_bogomolov(
    family_pvals: Dict[str, Sequence[float]],
    q: float = 0.05,
) -> Dict[str, object]:
    """Selective inference on multiple families (Benjamini & Bogomolov 2014).

    1. Each family is summarised by its Simes p-value.
    2. Families are selected by BH at level q over the m family p-values;
       R families are selected.
    3. Within each selected family, BH is applied at level R*q/m.
    Unselected families have no rejections. This controls the expected
    average FDR over the selected families at q (for independent or
    PRDS families); it does not control the FDR pooled over all
    hypotheses.
    """
    names = list(family_pvals)
    m = len(names)
    fam_p = np.array([simes(family_pvals[k]) for k in names])
    sel = _bh_reject(fam_p, q)
    R = int(sel.sum())
    level = R * q / m if m else 0.0
    rejected = {}
    for k, s in zip(names, sel):
        p = np.asarray(family_pvals[k], dtype=float)
        rejected[k] = (_bh_reject(p, level) if s else np.zeros(len(p), bool)).tolist()
    return {"family_simes_p": dict(zip(names, fam_p.tolist())),
            "selected": dict(zip(names, sel.tolist())),
            "n_selected": R, "within_level": level, "rejected": rejected}


def benjamini_yekutieli(pvals: Sequence[float], q: float = 0.05) -> np.ndarray:
    """BH at level q / sum_{i<=n} 1/i: FDR control under arbitrary dependence."""
    p = np.asarray(pvals, dtype=float)
    c = np.sum(1.0 / np.arange(1, len(p) + 1)) if len(p) else 1.0
    return _bh_reject(p, q / c)


if __name__ == "__main__":
    # Smoke test
    rng = np.random.default_rng(0)
    n_slides = 40
    patches_per = 50
    groups = np.repeat([f"s{i:03d}" for i in range(n_slides)], patches_per)
    # Simulate paired correctness where student is 2% worse
    t = rng.binomial(1, 0.85, size=n_slides * patches_per).astype(float)
    s = t.copy()
    flip = rng.random(size=t.shape) < 0.02
    s[flip] = 1 - s[flip]
    r = tost_equivalence(t, s, groups, margin=0.03, n_boot=500)
    print("TOST smoke:", r)
