"""Literature-fidelity tests for scripts/eval_v4_addons.py.

Each test hand-computes the expected value per the originating paper's
equation, then checks the module's output matches.  Run with:

    python3 scripts/test_eval_v4_addons.py

Exit 0 on full pass.

References:
  [Guo2017] Guo et al., "On Calibration of Modern Neural Networks",
           ICML 2017.  https://arxiv.org/abs/1706.04599
  [Geifman2017] Geifman & El-Yaniv, "Selective Classification for Deep
           Neural Networks", NeurIPS 2017.  https://arxiv.org/abs/1705.08500
  [deJong2025] "Current Pathology Foundation Models are Unrobust to Medical
           Center Differences", arXiv:2501.18055, 2025.
  [Ozeki2024] PLISM dataset.  plismbench (Owkin/OME).

Each test prints [PASS] or [FAIL] with the literature reference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_v4_addons import (
    aurc,
    expected_calibration_error,
    max_calibration_error,
    medical_center_probe,
    plism_retrieval,
    risk_at_coverage,
    robustness_index,
    tcga_tss_code,
)

_failures = []
_passes = 0


def check(label, cond, ref, detail=""):
    global _passes
    tag = "[PASS]" if cond else "[FAIL]"
    msg = f"{tag}  {label}  (ref: {ref})"
    if detail:
        msg += f"\n        {detail}"
    print(msg)
    if cond:
        _passes += 1
    else:
        _failures.append(label)


# ---------------------------------------------------------------------------
# ECE — Guo 2017 Eq. 3
# ---------------------------------------------------------------------------

def test_ece_hand_computed():
    """Construct one bin only, where confidence=0.8 and accuracy=0.6.
    ECE = 1.0 × |0.8 - 0.6| = 0.2."""
    # Two classes, all predictions in [0.75, 0.85) confidence band
    probs = np.array([[0.8, 0.2]] * 10)
    # 6 correct (label 0), 4 wrong (label 1)
    labels = np.array([0] * 6 + [1] * 4)
    ece, bins = expected_calibration_error(probs, labels, n_bins=10)
    populated = [b for b in bins if b["n"] > 0]
    check(
        "ECE single-bin hand calculation",
        abs(ece - 0.2) < 1e-9 and len(populated) == 1,
        "Guo2017 Eq.3",
        f"got ECE={ece:.4f}, expected 0.2000; populated bins={len(populated)}",
    )


def test_ece_perfect_calibration():
    """If all predictions have confidence equal to group accuracy,
    ECE → 0 in the limit of many samples per bin."""
    rng = np.random.default_rng(0)
    # Generate labels with 70% class 0, confidence = 0.7.
    # Prediction is argmax([0.7, 0.3]) = 0; accuracy = P(label==0) = 0.7.
    # Confidence matches accuracy → ECE should be ~0.
    N = 10000
    probs = np.tile([0.7, 0.3], (N, 1))
    labels = (rng.random(N) > 0.7).astype(int)  # 70% zeros, 30% ones
    ece, _ = expected_calibration_error(probs, labels, n_bins=15)
    check(
        "ECE well-calibrated → near 0",
        ece < 0.01,
        "Guo2017",
        f"got ECE={ece:.4f} on 10k samples at calibrated 0.7",
    )


def test_ece_monotone_in_miscalibration():
    """Increasing confidence while keeping accuracy fixed must increase ECE."""
    N = 1000
    labels = np.zeros(N, dtype=int)
    labels[: N // 2] = 1  # 50% accuracy floor
    probs_low = np.tile([0.55, 0.45], (N, 1))
    probs_hi = np.tile([0.95, 0.05], (N, 1))
    e_low, _ = expected_calibration_error(probs_low, labels, n_bins=15)
    e_hi, _ = expected_calibration_error(probs_hi, labels, n_bins=15)
    check(
        "ECE monotone in miscalibration",
        e_hi > e_low,
        "Guo2017",
        f"low={e_low:.4f}  hi={e_hi:.4f}",
    )


# ---------------------------------------------------------------------------
# AURC — Geifman & El-Yaniv 2017
# ---------------------------------------------------------------------------

def test_aurc_perfect_classifier():
    """0% error at every coverage → AURC = 0."""
    N = 100
    probs = np.zeros((N, 2))
    probs[:, 0] = 0.9  # always predict class 0 with conf 0.9
    probs[:, 1] = 0.1
    labels = np.zeros(N, dtype=int)
    a = aurc(probs, labels)
    check(
        "AURC = 0 for perfect classifier",
        a < 1e-9,
        "Geifman2017",
        f"got AURC={a:.6f}",
    )


def test_aurc_random_classifier():
    """Random predictions uncorrelated with labels → AURC ≈ err_rate.
    (With selector-uninformative confidence, risk is flat at err_rate.)"""
    rng = np.random.default_rng(0)
    N = 5000
    probs = rng.random((N, 2))
    probs = probs / probs.sum(-1, keepdims=True)
    labels = rng.integers(0, 2, size=N)
    a = aurc(probs, labels)
    err_rate = (probs.argmax(-1) != labels).mean()
    check(
        "AURC ≈ err_rate when confidence is uninformative",
        abs(a - err_rate) < 0.05,
        "Geifman2017",
        f"AURC={a:.4f}  err_rate={err_rate:.4f}",
    )


def test_aurc_better_than_random():
    """Confidence-correlated-with-correctness classifier beats random."""
    rng = np.random.default_rng(0)
    N = 5000
    correct = rng.random(N) < 0.7
    # High conf when correct, low conf when wrong (well-ranked selector)
    conf = np.where(correct, rng.uniform(0.7, 1.0, N), rng.uniform(0.5, 0.7, N))
    probs = np.zeros((N, 2))
    probs[:, 0] = conf
    probs[:, 1] = 1 - conf
    labels = np.where(correct, 0, 1)
    a_ranked = aurc(probs, labels)
    # Shuffle confidence to destroy ranking
    rng.shuffle(conf)
    probs_rand = np.stack([conf, 1 - conf], axis=1)
    a_random = aurc(probs_rand, labels)
    check(
        "AURC(well-ranked selector) < AURC(random selector)",
        a_ranked < a_random,
        "Geifman2017",
        f"ranked={a_ranked:.4f}  random={a_random:.4f}",
    )


def test_risk_at_coverage():
    """At 100% coverage, risk should equal overall error rate."""
    rng = np.random.default_rng(1)
    N = 1000
    probs = rng.random((N, 2))
    probs = probs / probs.sum(-1, keepdims=True)
    labels = rng.integers(0, 2, size=N)
    err_rate = (probs.argmax(-1) != labels).mean()
    r = risk_at_coverage(probs, labels, coverage=1.0)
    check(
        "risk@cov=1.0 equals overall error rate",
        abs(r - err_rate) < 1e-9,
        "Geifman2017",
        f"risk={r:.4f}  err_rate={err_rate:.4f}",
    )


# ---------------------------------------------------------------------------
# Medical-center probe — de Jong 2025
# ---------------------------------------------------------------------------

def test_tcga_tss_extraction():
    cases = {
        "TCGA-AA-3518-01Z-00-DX1.5aad...": "AA",
        "TCGA-C8-A12P-01Z-00-DX1.670B...": "C8",
        "TCGA-G9-6348-01Z-00-DX1.4c00...": "G9",
        "tumor_001.tif": "UNK",
        "random_slide.svs": "UNK",
        # real TCGA filename form (what's in our /Nas):
        "TCGA-DD-AAVW-01Z-00-DX1.3EFC0BF5.svs": "DD",
    }
    ok = all(tcga_tss_code(k) == v for k, v in cases.items())
    check(
        "TCGA TSS extraction matches GDC barcode spec",
        ok,
        "GDC barcode spec + deJong2025",
        f"all {len(cases)} cases matched" if ok else
        {k: tcga_tss_code(k) for k in cases},
    )


def test_center_probe_separable():
    """Well-separated clusters → AUC → 1."""
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(+5, 1, size=(100, 32)),
        rng.normal(-5, 1, size=(100, 32)),
        rng.normal(0, 1, size=(100, 32)) + 10,
    ])
    y = np.array(["A"] * 100 + ["B"] * 100 + ["C"] * 100)
    r = medical_center_probe(X, y, seed=0)
    check(
        "center probe AUC → 1 on separable clusters",
        r["center_probe_auc"] > 0.98,
        "deJong2025",
        f"AUC={r['center_probe_auc']:.4f}  acc={r['center_probe_acc']:.4f}",
    )


def test_center_probe_random():
    """Random embeddings independent of center → AUC → 0.5."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 32))
    y = rng.choice(["A", "B", "C"], size=600)
    r = medical_center_probe(X, y, seed=0)
    check(
        "center probe AUC ≈ 0.5 on random embeddings",
        abs(r["center_probe_auc"] - 0.5) < 0.1,
        "deJong2025",
        f"AUC={r['center_probe_auc']:.4f}",
    )


def test_robustness_index_ratio():
    """Biology-dominant (task AUC 0.95, center 0.55): index should be > 1."""
    ri = robustness_index(task_auc=0.95, center_auc=0.55)
    check(
        "robustness_index > 1 when biology > center",
        ri > 1.0,
        "deJong2025",
        f"RI = {ri:.3f}",
    )
    ri2 = robustness_index(task_auc=0.60, center_auc=0.90)
    check(
        "robustness_index < 1 when center > biology",
        ri2 < 1.0,
        "deJong2025",
        f"RI = {ri2:.3f}",
    )


# ---------------------------------------------------------------------------
# PLISM retrieval — Ozeki 2024
# ---------------------------------------------------------------------------

def test_plism_identity_perfect_retrieval():
    """Same embeddings under two configs → top-1 = 1.0."""
    rng = np.random.default_rng(0)
    N, D = 50, 128
    embs = rng.normal(size=(N, D))
    res = plism_retrieval({"stain_A": embs, "stain_B": embs.copy()},
                          config_pairs=[("stain_A", "stain_B")])
    key = "stain_A->stain_B"
    check(
        "PLISM top-1 = 1.0 when query == gallery",
        abs(res[key]["top1"] - 1.0) < 1e-9,
        "Ozeki2024/plismbench",
        f"top1={res[key]['top1']}",
    )


def test_plism_random_retrieval():
    """Unrelated gallery → top-1 ≈ 1/N (chance)."""
    rng = np.random.default_rng(0)
    N = 100
    embs_a = rng.normal(size=(N, 128))
    embs_b = rng.normal(size=(N, 128))
    res = plism_retrieval({"A": embs_a, "B": embs_b},
                          config_pairs=[("A", "B")], top_k=(1, 10))
    key = "A->B"
    # top-1 chance = 1/N = 0.01; top-10 chance = 10/N = 0.10
    ok1 = res[key]["top1"] < 0.1
    ok10 = res[key]["top10"] < 0.25
    check(
        "PLISM top-K ≈ chance on unrelated embeddings",
        ok1 and ok10,
        "Ozeki2024",
        f"top1={res[key]['top1']}, top10={res[key]['top10']}",
    )


def test_plism_noisy_retrieval_degrades():
    """Add noise to gallery → top-1 drops monotonically."""
    rng = np.random.default_rng(0)
    embs = rng.normal(size=(100, 128))
    accs = []
    for sigma in [0.1, 0.5, 2.0, 10.0]:
        noisy = embs + rng.normal(scale=sigma, size=embs.shape)
        r = plism_retrieval({"A": embs, "B": noisy}, config_pairs=[("A", "B")])
        accs.append(r["A->B"]["top1"])
    monotone = all(accs[i] >= accs[i + 1] - 0.05 for i in range(len(accs) - 1))
    check(
        "PLISM retrieval degrades with injected noise",
        monotone,
        "Ozeki2024",
        f"accs={accs} (sigmas: 0.1,0.5,2,10)",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tests = [
        test_ece_hand_computed,
        test_ece_perfect_calibration,
        test_ece_monotone_in_miscalibration,
        test_aurc_perfect_classifier,
        test_aurc_random_classifier,
        test_aurc_better_than_random,
        test_risk_at_coverage,
        test_tcga_tss_extraction,
        test_center_probe_separable,
        test_center_probe_random,
        test_robustness_index_ratio,
        test_plism_identity_perfect_retrieval,
        test_plism_random_retrieval,
        test_plism_noisy_retrieval_degrades,
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
    print("All literature-fidelity checks PASSED.")
    sys.exit(0)


if __name__ == "__main__":
    main()
