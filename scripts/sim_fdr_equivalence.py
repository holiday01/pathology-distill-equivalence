#!/usr/bin/env python3
"""Dependence simulation for the equivalence-FDR procedure.

Claim to validate: across the 12-teacher x 3-student grid, where the three
students of a teacher share a teacher-level effect (positive dependence),
the teacher->student gatekeeping procedure controls the family-wise
false-equivalence rate (declaring a pair equivalent when its true AUROC
difference exceeds the margin) at the nominal level, whereas testing each
cell independently does not.

Design (per replicate):
  - 12 teachers x 3 students = 36 pairs. A configurable fraction of pairs are
    truly INEQUIVALENT (|true dAUROC| = d_far > delta); the rest equivalent.
  - shared teacher effect u_i ~ N(0, tau^2) added to all three of teacher i's
    observed differences -> within-teacher dependence.
  - observed d_ij = dtrue_ij + u_i + eps_ij, eps ~ N(0, se^2) (se from the
    slide-cluster bootstrap scale).
  - TOST at delta: pair is declared equivalent iff the 90% CI [d-1.645 se,
    d+1.645 se] lies within [-delta, delta].
  - Gatekeeping: a teacher's three students are eligible for an equivalence
    claim only if the teacher's mean |d| passes a screen (mean over its 3
    students is itself inside the margin under a 1-sided check); cells of a
    teacher that fails the screen cannot be declared equivalent.
A false equivalence = a truly-inequivalent pair declared equivalent. We report
the family-wise false-equivalence rate (>=1 false equivalence per replicate)
for independent vs gatekeeping decisions.
"""
import numpy as np

DELTA = 0.05
D_FAR = 0.09          # true diff of inequivalent pairs (just beyond margin)
SE = 0.025            # per-pair slide-cluster SE (matches the empirical scale)
TAU = 0.03            # teacher-level dependence sd
N_TEACH, N_STU = 12, 3
FRAC_INEQ = 0.5       # half the pairs truly inequivalent
N_REP = 20000
Z90 = 1.6448536


def run(seed=42):
    rng = np.random.default_rng(seed)
    # fixed truth: which pairs are inequivalent
    n_pairs = N_TEACH * N_STU
    inequiv = np.zeros((N_TEACH, N_STU), dtype=bool)
    k = int(round(FRAC_INEQ * n_pairs))
    flat = rng.permutation(n_pairs)[:k]
    for f in flat:
        inequiv[f // N_STU, f % N_STU] = True
    dtrue = np.where(inequiv, D_FAR, 0.0)   # equivalent pairs centered at 0

    fe_indep = fe_gate = 0
    for _ in range(N_REP):
        u = rng.normal(0, TAU, size=N_TEACH)[:, None]        # shared teacher effect
        eps = rng.normal(0, SE, size=(N_TEACH, N_STU))
        d = dtrue + u + eps
        ci_lo, ci_hi = d - Z90 * SE, d + Z90 * SE
        equ = (ci_lo > -DELTA) & (ci_hi < DELTA)             # per-cell TOST

        # independent decision
        fe_i = np.any(equ & inequiv)
        # gatekeeping: teacher passes screen iff its mean |d| < delta
        teach_ok = np.abs(d.mean(axis=1)) < DELTA
        equ_gate = equ & teach_ok[:, None]
        fe_g = np.any(equ_gate & inequiv)
        fe_indep += fe_i
        fe_gate += fe_g

    return fe_indep / N_REP, fe_gate / N_REP


def main():
    fwer_indep, fwer_gate = run()
    print(f"settings: delta={DELTA}, d_far={D_FAR}, se={SE}, tau={TAU}, "
          f"grid={N_TEACH}x{N_STU}, {int(FRAC_INEQ*100)}% inequivalent, reps={N_REP}")
    print(f"family-wise false-equivalence rate:")
    print(f"  independent per-cell TOST : {fwer_indep:.3f}")
    print(f"  teacher->student gatekeeping: {fwer_gate:.3f}")
    print(f"nominal one-sided alpha = 0.05")
    ok = fwer_gate <= 0.05 + 0.005
    print(f"=> gatekeeping controls family-wise false-equivalence at nominal: {ok}")


if __name__ == "__main__":
    main()
