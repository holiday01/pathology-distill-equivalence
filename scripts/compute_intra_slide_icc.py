#!/usr/bin/env python3
"""Independently measure the intra-slide correlation rho (ICC of the per-tile
teacher probe score) for CAMELYON16 and Kather-MSI, with a cluster-bootstrap CI.

This supplies the rho the design-effect argument needs, measured directly rather
than backed out of the observed inflation. Reports the predicted mean-type design
effect sqrt(1+(m-1)*rho) alongside the observed clustered-AUROC inflation, so the
correspondence (or lack of it) is explicit for the rank metric.

Output: outputs/v4_full/intra_slide_icc.json
"""
import sys, json, statistics as st
from pathlib import Path
import numpy as np, torch, h5py

sys.path.insert(0, str(Path(__file__).parent))
from eval_c16_equivalence import load_c16, stratified_slide_split, extract
from distill_wsi_model import TeacherModel
from evaluate_distillation import linear_probe

device = "cuda" if torch.cuda.is_available() else "cpu"
TEACHERS = ["uni2-h", "h-optimus-0", "virchow2", "uni"]
C16 = "/path/to/cache/patches/patches_c16_annotated.h5"
KAT = "/path/to/cache/patches/patches_kather_msi.h5"


def icc_oneway(x, groups):
    groups = np.asarray(groups); x = np.asarray(x, float)
    uniq = np.unique(groups); k = len(uniq); N = len(x)
    if k < 2 or N <= k:
        return float("nan")
    grand = x.mean(); SSB = 0.0; SSW = 0.0; ni = []
    for g in uniq:
        xi = x[groups == g]; ni.append(len(xi)); mi = xi.mean()
        SSB += len(xi) * (mi - grand) ** 2; SSW += ((xi - mi) ** 2).sum()
    MSB = SSB / (k - 1); MSW = SSW / (N - k)
    ni = np.array(ni); m0 = (N - (ni ** 2).sum() / N) / (k - 1)
    den = MSB + (m0 - 1) * MSW
    return float((MSB - MSW) / den) if den > 0 else float("nan")


def icc_ci(x, groups, nboot=1000, seed=42):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups); by = {g: np.where(groups == g)[0] for g in uniq}
    boot = []
    for _ in range(nboot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([by[uniq[i]] for i in pick])
        gg = np.concatenate([[j] * len(by[uniq[i]]) for j, i in enumerate(pick)])
        v = icc_oneway(x[idx], gg)
        if v == v:
            boot.append(v)
    return icc_oneway(x, groups), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def teacher_score(h5, tr, te, y_tr, y_te):
    r = linear_probe(torch.from_numpy(tr).to(device), torch.from_numpy(y_tr).to(device),
                     torch.from_numpy(te).to(device), torch.from_numpy(y_te).to(device), n_classes=2)
    p = r["probs"]; p = p.cpu().numpy() if isinstance(p, torch.Tensor) else np.asarray(p)
    return p[:, 1]


def measure(h5, tr_idx, te_idx, y_tr, y_te, grp_te, tag):
    n_cl = len(np.unique(grp_te)); m = len(te_idx) / n_cl
    iccs = []
    for t in TEACHERS:
        tm = TeacherModel(t).eval().to(device)
        ft = extract(tm, h5, tr_idx, device); fe = extract(tm, h5, te_idx, device)
        del tm; torch.cuda.empty_cache()
        sc = teacher_score(h5, ft, fe, y_tr, y_te)
        base, lo, hi = icc_ci(sc, grp_te)
        iccs.append(base)
        print(f"  [{tag}] {t:12s} ICC(score)={base:.3f} [{lo:.3f},{hi:.3f}]")
    med = st.median(iccs)
    de = (1 + (m - 1) * med) ** 0.5
    print(f"  [{tag}] median rho={med:.3f}  m~{m:.0f}  clusters={n_cl}  "
          f"predicted sqrt(DE)={de:.2f}x")
    return {"iccs": [round(x, 4) for x in iccs], "median_rho": round(med, 4),
            "m": round(m, 1), "n_clusters": int(n_cl), "predicted_de_inflation": round(de, 2)}


def main():
    out = {}
    # CAMELYON16 (41 test slides, deterministic split seed=42)
    idx, labels, sid = load_c16(C16, None)
    tr, te, nts, ntot = stratified_slide_split(sid, labels, idx, 0.5)
    print(f"C16: {nts} test slides, {len(te)} test tiles")
    out["c16"] = measure(C16, tr, te, labels[tr], labels[te], sid[te], "C16")
    # Kather-MSI (predefined test patients via is_test)
    with h5py.File(KAT, "r") as f:
        kl = f["labels"][:].astype(np.int64); kpid = f["slide_ids"][:].astype(np.int64)
        kte = f["is_test"][:].astype(bool)
    ktr = np.where(~kte)[0]; ktei = np.where(kte)[0]
    print(f"Kather: {len(np.unique(kpid[ktei]))} test patients, {len(ktei)} test tiles")
    out["kather"] = measure(KAT, ktr, ktei, kl[ktr], kl[ktei], kpid[ktei], "Kather")
    out["observed_auc_inflation"] = {"c16": 6.83, "kather": 4.32}
    json.dump(out, open("outputs/v4_full/intra_slide_icc.json", "w"), indent=2)
    print("\n=== SUMMARY (measured rho vs observed AUROC inflation) ===")
    for c in ("c16", "kather"):
        r = out[c]
        print(f"{c}: rho={r['median_rho']} (m~{r['m']:.0f}) -> predicted {r['predicted_de_inflation']}x "
              f"vs observed {out['observed_auc_inflation'][c]}x")
    print("[saved] outputs/v4_full/intra_slide_icc.json")


if __name__ == "__main__":
    main()
