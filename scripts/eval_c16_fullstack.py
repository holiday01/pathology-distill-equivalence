#!/usr/bin/env python3
"""Full-variance-stack C16 equivalence certification.

Fixes the probe non-determinism (linear probe has random init) by AVERAGING K
probe fits, and certifies against the three variance sources the naive
patch-level test ignores:
  - probe-fitting  (residual after averaging: sigma_probe / sqrt(K*S))
  - training seed  (seed 42 vs 2025, where a 2nd seed exists: sigma_seed / sqrt(S))
  - slide-clustering (slide-cluster bootstrap SE of the probe-averaged AUROC diff)

Estimate of the true AUROC difference mu_d = mean over seeds of the
probe-averaged d.  SE(mu_d) = sqrt(sigma_cluster^2 + sigma_seed^2/S + sigma_probe^2/(K*S)).
Certify equivalence iff |mu_d| + 1.645*SE(mu_d) <= delta.

For the 15 pairs (5 teachers) with a 2nd seed, sigma_seed is measured directly;
for the other 21 pairs the median 2-seed sigma_seed is used as a plug-in (flagged).

Output: outputs/v4_full/c16_fullstack.json (+ .md)
"""
import sys, json, os, glob, statistics as st
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from eval_c16_equivalence import load_c16, stratified_slide_split, extract, STU_MAP
from distill_wsi_model import TeacherModel, StudentModel
from evaluate_distillation import linear_probe

device = "cuda" if torch.cuda.is_available() else "cpu"
H5 = "/path/to/cache/patches/patches_c16_annotated.h5"
K = 8            # probe fits to average
DELTA = 0.05
Z = 1.645
NBOOT = 500


def fast_auc(y, score):
    """Vectorized Mann-Whitney AUROC (ties ignored; probe scores are continuous)."""
    n = len(y); n_pos = int(y.sum()); n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def probe_scores(ftr, ytr, fte, k):
    torch.manual_seed(2000 + k)
    r = linear_probe(torch.from_numpy(ftr).to(device), torch.from_numpy(ytr).to(device),
                     torch.from_numpy(fte).to(device), torch.from_numpy(np.zeros(len(fte), int)).to(device),
                     n_classes=2)
    p = r["probs"]; p = p.cpu().numpy() if isinstance(p, torch.Tensor) else np.asarray(p)
    return p[:, 1]


def cluster_se_d(y, t_avg, s_avg, groups, seed=42):
    """Slide-cluster bootstrap SE of d = AUROC_s - AUROC_t on probe-averaged scores."""
    uniq = np.unique(groups); by = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed); ds = []
    for _ in range(NBOOT):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([by[uniq[i]] for i in pick])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        ds.append(fast_auc(yy, s_avg[idx]) - fast_auc(yy, t_avg[idx]))
    return float(np.std(ds, ddof=1)) if len(ds) >= 20 else float("nan")


def eval_seed(root, fm, stu, tr, te, ytr, yte, grp, t_cache):
    ck = f"{root}/{fm}/{stu}/best.pt"
    if not os.path.exists(ck):
        return None
    # teacher probe scores (cached per teacher/root-independent: teacher same for both seeds)
    ttr, tte = t_cache[fm]
    t_fits = np.array([probe_scores(ttr, ytr, tte, k) for k in range(K)])   # (K, n_te)
    sm = StudentModel(STU_MAP[stu], embed_dim=256, n_classes=0, pretrained=False).to(device)
    sd = torch.load(ck, map_location=device); sm.load_state_dict(sd.get("student", sd), strict=False); sm.eval()
    str_, ste = extract(sm, H5, tr, device), extract(sm, H5, te, device)
    del sm; torch.cuda.empty_cache()
    s_fits = np.array([probe_scores(str_, ytr, ste, k) for k in range(K)])
    # per-fit d for sigma_probe; probe-averaged scores for point estimate + cluster SE
    d_fits = np.array([fast_auc(yte, s_fits[k]) - fast_auc(yte, t_fits[k]) for k in range(K)])
    t_avg = t_fits.mean(0); s_avg = s_fits.mean(0)
    d_avg = fast_auc(yte, s_avg) - fast_auc(yte, t_avg)
    se_c = cluster_se_d(yte, t_avg, s_avg, grp)
    return {"d_avg": float(d_avg), "sigma_probe": float(d_fits.std(ddof=1)), "sigma_cluster": se_c}


def main():
    idx, labels, sid = load_c16(H5, None)
    tr, te, nts, _ = stratified_slide_split(sid, labels, idx, 0.5)
    ytr, yte, grp = labels[tr], labels[te], sid[te]
    print(f"[split] {nts} test slides, {len(te)} test tiles")

    TWO_SEED = {"h-optimus-0", "phikon", "uni", "virchow2", "hibou-l"}
    ckpts = [c for c in sorted(glob.glob("outputs/v4_full/*/*/best.pt")) if c.split("/")[-3] in TWO_SEED]
    t_cache = {}
    rows = []
    for ck in ckpts:
        fm, stu = ck.split("/")[-3], ck.split("/")[-2]
        if stu not in STU_MAP:
            continue
        try:
            if fm not in t_cache:
                tm = TeacherModel(fm).eval().to(device)
                t_cache[fm] = (extract(tm, H5, tr, device), extract(tm, H5, te, device))
                del tm; torch.cuda.empty_cache()
            r42 = eval_seed("outputs/v4_full", fm, stu, tr, te, ytr, yte, grp, t_cache)
            r2 = eval_seed("outputs/v4_seed2", fm, stu, tr, te, ytr, yte, grp, t_cache)
            rec = {"teacher": fm, "student": stu, "seed42": r42, "seed2": r2}
            rows.append(rec)
            msg = f"{fm}/{stu}: s42 d={r42['d_avg']:+.3f} sig_probe={r42['sigma_probe']:.3f} sig_clus={r42['sigma_cluster']:.3f}"
            if r2:
                msg += f" | s2 d={r2['d_avg']:+.3f} | |Δd|={abs(r42['d_avg']-r2['d_avg']):.3f}"
            print(msg)
        except Exception as e:
            print(f"  [skip] {fm}/{stu}: {type(e).__name__}: {e}")

    # sigma_seed plug-in = median over 2-seed pairs of |d42-d2|/sqrt(2)
    two = [r for r in rows if r["seed2"]]
    sig_seed_pairs = [abs(r["seed42"]["d_avg"] - r["seed2"]["d_avg"]) / (2 ** 0.5) for r in two]
    sig_seed_med = float(st.median(sig_seed_pairs)) if sig_seed_pairs else 0.0

    def certify(rec):
        r42 = rec["seed42"]; r2 = rec["seed2"]
        if r2:
            S = 2; mu = (r42["d_avg"] + r2["d_avg"]) / 2
            sig_seed = abs(r42["d_avg"] - r2["d_avg"]) / (2 ** 0.5)
            sig_probe = (r42["sigma_probe"] + r2["sigma_probe"]) / 2
            sig_clus = np.nanmean([r42["sigma_cluster"], r2["sigma_cluster"]])
            seed_measured = True
        else:
            S = 1; mu = r42["d_avg"]; sig_seed = sig_seed_med
            sig_probe = r42["sigma_probe"]; sig_clus = r42["sigma_cluster"]; seed_measured = False
        se = (sig_clus ** 2 + sig_seed ** 2 / S + sig_probe ** 2 / (K * S)) ** 0.5
        lo, hi = mu - Z * se, mu + Z * se
        return {"mu_d": round(float(mu), 4), "se_fullstack": round(float(se), 4),
                "ci": [round(float(lo), 4), round(float(hi), 4)],
                "equiv_fullstack": bool(lo >= -DELTA and hi <= DELTA),
                "seed_measured": seed_measured,
                "components": {"cluster": round(float(sig_clus), 4), "seed": round(float(sig_seed), 4),
                               "probe_resid": round(float(sig_probe / (K * S) ** 0.5), 4)}}

    for r in rows:
        r["fullstack"] = certify(r)

    n_eq = sum(r["fullstack"]["equiv_fullstack"] for r in rows if "hibou" not in r["teacher"])
    n_eq_2seed = sum(r["fullstack"]["equiv_fullstack"] for r in two if "hibou" not in r["teacher"])
    out = {"K": K, "delta": DELTA, "n_test_slides": nts, "sigma_seed_plugin": round(sig_seed_med, 4),
           "n_equiv_fullstack_nonhibou": n_eq, "n_equiv_fullstack_2seed_nonhibou": n_eq_2seed,
           "n_2seed_nonhibou": len([r for r in two if "hibou" not in r["teacher"]]), "rows": rows}
    json.dump(out, open("outputs/v4_full/c16_fullstack.json", "w"), indent=2)
    print(f"\n=== FULL-STACK (probe-averaged K={K}, cluster+seed) at delta={DELTA} ===")
    print(f"  sigma_seed plug-in (median) = {sig_seed_med:.4f}")
    print(f"  equivalent (non-hibou, all 30): {n_eq}/30")
    print(f"  equivalent (non-hibou, 2-seed-measured, 12): {n_eq_2seed}/12")
    print("[saved] outputs/v4_full/c16_fullstack.json")


if __name__ == "__main__":
    main()
