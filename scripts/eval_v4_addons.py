"""Evaluation add-ons for v4 atlas per 2026-04-22 literature review.

Three modules, each referenced to the originating paper:

1. Calibration — ECE / MCE / AURC
   - ECE (Expected Calibration Error): Guo et al. "On Calibration of Modern
     Neural Networks" ICML 2017. ECE = Σ_m (|B_m|/n) × |acc(B_m) - conf(B_m)|
     where B_m is the m-th equal-width confidence bin.
   - MCE (Maximum Calibration Error): same paper; max over bins.
   - AURC (Area Under Risk-Coverage): Geifman & El-Yaniv "Selective
     Classification for Deep Neural Networks" NeurIPS 2017/18. For a
     confidence-based selector with threshold τ, coverage(τ) = P(conf ≥ τ)
     and risk(τ) = E[error | conf ≥ τ]. AURC = ∫ risk d coverage, sorted by
     confidence descending; lower is better.

2. Medical-center probe
   - Protocol: de Jong et al. "Current Pathology Foundation Models are
     Unrobust to Medical Center Differences" arXiv:2501.18055, 2025.
   - Train a logistic-regression probe embedding → medical center id.
   - Report macro one-vs-rest AUC (high = embedding encodes center identity).
   - Robustness index: task-probe-AUC / center-probe-AUC (ratio > 1 means
     biological signal dominates, ratio < 1 means center signal dominates).
   - TCGA center code = TSS (tissue source site), extracted from barcode
     `TCGA-{TSS}-{participant}-{sample}-{vial}-{portion}-{analyte}-{plate}-{center}`
     per GDC docs, e.g. 'TCGA-AA-3518-01Z-00-DX1' → TSS='AA'.

3. PLISM retrieval invariance
   - Protocol: Ozeki et al. PLISM dataset; plismbench (Owkin) public
     evaluation. 46 tissue types × 13 H&E stainings × 7 scanners.
   - For each pair of (stain, scanner) configs (c_a, c_b), extract embeddings
     on the same tissue tiles. Top-K retrieval accuracy = fraction of query
     tiles from c_a whose nearest-K neighbors in c_b include the matched
     tile.  Higher = more stain/scanner invariant.

Public API:
    expected_calibration_error(probs, labels, n_bins=15) -> (ece, bin_stats)
    max_calibration_error(probs, labels, n_bins=15) -> mce
    aurc(probs, labels) -> float
    risk_at_coverage(probs, labels, coverage=0.9) -> float
    medical_center_probe(embs, center_ids, seed=42) -> dict
    tcga_tss_code(slide_id) -> str
    plism_retrieval(model, plism_tiles, top_k=1) -> dict   # stub + docs
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. Calibration metrics
# ---------------------------------------------------------------------------

def _softmax_stats(probs: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if probs.ndim != 2:
        raise ValueError("probs must be (N, K) softmax outputs")
    conf = probs.max(axis=-1)
    preds = probs.argmax(axis=-1)
    correct = (preds == labels).astype(float)
    return conf, correct


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, List[dict]]:
    """Guo et al. 2017, ICML, Equation 3.

    ECE = Σ_m (n_m / N) × |acc(B_m) - conf(B_m)|

    Equal-width binning over [0, 1]. Last bin is inclusive on the right
    to include confidence = 1.0.
    """
    conf, correct = _softmax_stats(probs, labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    N = len(conf)
    ece = 0.0
    stats = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        n_m = int(mask.sum())
        if n_m == 0:
            stats.append({"lo": lo, "hi": hi, "n": 0, "conf": None, "acc": None})
            continue
        bin_conf = float(conf[mask].mean())
        bin_acc = float(correct[mask].mean())
        ece += (n_m / N) * abs(bin_acc - bin_conf)
        stats.append({"lo": lo, "hi": hi, "n": n_m, "conf": bin_conf, "acc": bin_acc})
    return float(ece), stats


def max_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    min_count: int = 1,
) -> float:
    """Guo et al. 2017 — max over populated bins of |acc - conf|."""
    conf, correct = _softmax_stats(probs, labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if mask.sum() < min_count:
            continue
        gap = abs(correct[mask].mean() - conf[mask].mean())
        mce = max(mce, float(gap))
    return mce


def aurc(probs: np.ndarray, labels: np.ndarray) -> float:
    """Area Under Risk-Coverage per Geifman & El-Yaniv 2017.

    Sort samples by confidence descending.  For coverage k/N (top-k by
    confidence), risk_k = (# errors in top-k) / k.  AURC = mean over k of
    risk_k (equivalent to trapezoid integration on equally-spaced coverage).
    """
    conf, correct = _softmax_stats(probs, labels)
    order = np.argsort(-conf, kind="stable")
    err_sorted = 1.0 - correct[order]
    N = len(err_sorted)
    cum_err = np.cumsum(err_sorted)
    k = np.arange(1, N + 1)
    risk = cum_err / k
    # integrate over coverage = k/N (grid step 1/N)
    return float(risk.mean())


def risk_at_coverage(
    probs: np.ndarray,
    labels: np.ndarray,
    coverage: float = 0.9,
) -> float:
    """Error rate on the top `coverage` fraction by confidence."""
    conf, correct = _softmax_stats(probs, labels)
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    order = np.argsort(-conf, kind="stable")
    k = max(1, int(np.ceil(coverage * len(conf))))
    return float(1.0 - correct[order][:k].mean())


# ---------------------------------------------------------------------------
# 2. Medical-center probe (de Jong et al. 2501.18055)
# ---------------------------------------------------------------------------

def tcga_tss_code(slide_id: str) -> str:
    """Extract TSS from a TCGA slide barcode.  Returns 'UNK' for non-TCGA."""
    s = str(slide_id)
    # Barcode form: TCGA-{TSS}-{participant}-...  TSS is 2 alphanumeric
    m = re.match(r"TCGA-([A-Z0-9]{2})-", s)
    if m:
        return m.group(1)
    # Sometimes slide_id is the full filename: try the first 'TCGA-' occurrence
    m2 = re.search(r"TCGA-([A-Z0-9]{2})-[A-Z0-9]{4}", s)
    return m2.group(1) if m2 else "UNK"


def medical_center_probe(
    embs: np.ndarray,
    center_ids: Sequence[str],
    seed: int = 42,
    min_per_class: int = 4,
    test_frac: float = 0.3,
) -> Dict[str, float]:
    """Linear probe: embedding → center id.

    Returns one-vs-rest macro AUC (de Jong protocol).  If only one center
    remains after filtering sparse classes, returns NaN.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        return {"error": f"sklearn required: {e}"}

    X = np.asarray(embs, dtype=float)
    y = np.asarray(center_ids)
    unique, counts = np.unique(y, return_counts=True)
    keep = unique[counts >= min_per_class]
    mask = np.isin(y, keep)
    X, y = X[mask], y[mask]
    n_classes = len(np.unique(y))
    if n_classes < 2 or len(X) < 10:
        return {
            "center_probe_auc": float("nan"),
            "center_probe_acc": float("nan"),
            "n_centers": int(n_classes),
            "n_samples": int(len(X)),
        }
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_frac, stratify=y, random_state=seed
    )
    clf = LogisticRegression(max_iter=2000, solver="liblinear" if n_classes == 2 else "lbfgs")
    clf.fit(Xtr, ytr)
    if n_classes == 2:
        probs = clf.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, probs)
    else:
        probs = clf.predict_proba(Xte)
        auc = roc_auc_score(yte, probs, multi_class="ovr", average="macro")
    return {
        "center_probe_auc": float(auc),
        "center_probe_acc": float(clf.score(Xte, yte)),
        "n_centers": int(n_classes),
        "n_samples": int(len(X)),
    }


def robustness_index(task_auc: float, center_auc: float) -> float:
    """de Jong's Robustness Index = task-info / center-info.

    Task AUC quantifies biological signal in embedding; Center AUC
    quantifies spurious confounder.  Ratio > 1 means biology dominates.
    Uses log-odds to make the ratio sensitive near chance level.
    """
    if not 0.5 < task_auc <= 1.0 or not 0.5 < center_auc <= 1.0:
        return float("nan")
    # Use "AUC above chance" as information proxy
    return float((task_auc - 0.5) / (center_auc - 0.5))


# ---------------------------------------------------------------------------
# 3. PLISM retrieval (stub — dataset download deferred)
# ---------------------------------------------------------------------------

def plism_retrieval(
    embs_by_config: Dict[str, np.ndarray],
    config_pairs: List[Tuple[str, str]] = None,
    top_k: Sequence[int] = (1, 5, 10),
) -> Dict[str, Dict]:
    """Top-K retrieval accuracy across (stain, scanner) configs.

    embs_by_config: {config_name: (N, D) np.ndarray}, tile indices aligned
                    across configs (row i in config A corresponds to the
                    same tissue tile as row i in config B).
    config_pairs:   list of (query_config, gallery_config) to evaluate.
                    default = all ordered pairs.

    Returns per-pair top-k accuracy; higher = more stain/scanner invariant.
    """
    names = list(embs_by_config.keys())
    if config_pairs is None:
        config_pairs = [(a, b) for a in names for b in names if a != b]
    # L2-normalize once
    normed = {k: v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
              for k, v in embs_by_config.items()}
    results = {}
    top_k = tuple(sorted(top_k))
    K = max(top_k)
    for a, b in config_pairs:
        X_q, X_g = normed[a], normed[b]
        if X_q.shape[0] != X_g.shape[0]:
            raise ValueError(f"configs {a}/{b} row mismatch")
        sim = X_q @ X_g.T
        topk_idx = np.argpartition(-sim, kth=K - 1, axis=1)[:, :K]
        # Refine sort within top-K
        for i in range(len(topk_idx)):
            order = np.argsort(-sim[i, topk_idx[i]])
            topk_idx[i] = topk_idx[i][order]
        true_idx = np.arange(len(X_q))[:, None]
        hits = (topk_idx == true_idx)
        acc = {f"top{k}": float(hits[:, :k].any(axis=1).mean()) for k in top_k}
        results[f"{a}->{b}"] = {"n_tiles": int(len(X_q)), **acc}
    return results


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Synthetic 3-class logits → softmax
    N, K = 500, 3
    logits = rng.normal(size=(N, K))
    probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    labels = probs.argmax(-1).copy()
    # Flip 20% labels so confidence ≠ accuracy
    flip = rng.random(N) < 0.20
    labels[flip] = rng.integers(0, K, size=flip.sum())

    ece, _ = expected_calibration_error(probs, labels, n_bins=10)
    mce = max_calibration_error(probs, labels, n_bins=10)
    a = aurc(probs, labels)
    r90 = risk_at_coverage(probs, labels, 0.9)
    print(f"ECE={ece:.4f}  MCE={mce:.4f}  AURC={a:.4f}  risk@cov0.9={r90:.4f}")

    # Medical-center probe: bimodal synthetic clusters
    centers = np.array(["A"] * 100 + ["B"] * 100 + ["C"] * 100)
    X1 = rng.normal(0, 1, size=(100, 16))
    X2 = rng.normal(3, 1, size=(100, 16))
    X3 = rng.normal(-3, 1, size=(100, 16))
    X = np.vstack([X1, X2, X3])
    r = medical_center_probe(X, centers)
    print("center probe:", r, "(expect AUC→1 since well-separated)")

    # PLISM smoke: same embeddings → top1 == 1.0
    embs = rng.normal(size=(30, 64))
    res = plism_retrieval({"c1": embs, "c2": embs + 1e-4 * rng.normal(size=embs.shape)})
    print("plism smoke:", res)
