"""Tests for the Methods 'Secondary Evaluations' assertions: linear CKA and the
slide-disjoint medical-center probe."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_distillation import linear_cka  # noqa: E402
import center_probe_slide_disjoint as cp  # noqa: E402


def _hsic_cka(X, Y):
    """Reference: Kornblith et al. 2019, Eq. for linear CKA via centred Gram matrices."""
    n = len(X)
    H = np.eye(n) - 1.0 / n
    K, L = X @ X.T, Y @ Y.T
    hsic = lambda A, B: np.trace(A @ H @ B @ H)
    return hsic(K, L) / np.sqrt(hsic(K, K) * hsic(L, L))


def test_cka_matches_gram_definition():
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(200, 16)), rng.normal(size=(200, 8))
    Y[:, :4] += X[:, :4]
    got = linear_cka(torch.from_numpy(X), torch.from_numpy(Y))
    assert abs(got - _hsic_cka(X, Y)) < 1e-6


def test_cka_invariant_to_rotation_and_scale_and_self_is_one():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 12))
    Q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    tX = torch.from_numpy(X)
    assert abs(linear_cka(tX, tX) - 1.0) < 1e-6
    assert abs(linear_cka(tX, torch.from_numpy(3.0 * X @ Q)) - 1.0) < 1e-6


def test_center_probe_folds_are_slide_disjoint_and_three_fold():
    from sklearn.model_selection import StratifiedGroupKFold
    assert cp.N_FOLDS == 3 and cp.MIN_SLIDES == 3
    rng = np.random.default_rng(2)
    slide = np.repeat(np.arange(30), 20)
    centre = np.repeat(np.arange(30) % 5, 20)
    cv = StratifiedGroupKFold(n_splits=cp.N_FOLDS, shuffle=True, random_state=cp.SEED)
    seen = np.zeros(len(slide), int)
    for tr, te in cv.split(rng.normal(size=(len(slide), 4)), centre, groups=slide):
        assert not set(slide[tr]) & set(slide[te])
        seen[te] += 1
    assert (seen == 1).all()  # every tile scored exactly once, out of fold


def test_center_probe_returns_macro_auc_on_out_of_fold_predictions():
    rng = np.random.default_rng(3)
    slide = np.repeat(np.arange(30), 20)
    centre = np.repeat(np.arange(30) % 5, 20)
    X = rng.normal(size=(len(slide), 8))
    auc_null, _ = cp.probe(X, centre, slide)
    X[:, 0] += 3 * centre  # separable signal
    auc_sig, _ = cp.probe(X, centre, slide)
    assert 0.3 < auc_null < 0.7 and auc_sig > 0.95
