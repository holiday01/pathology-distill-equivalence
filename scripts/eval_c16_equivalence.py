#!/usr/bin/env python3
"""Decoupled CAMELYON16 equivalence eval — fixes the n_slides=2 fatal flaw.

The main downstream eval shares a global slide-disjoint split with the
distillation pool, which happened to place only 2 C16 slides in the test
cluster, making the slide-cluster bootstrap and TOST statistically
undefined. This script is post-hoc on the FROZEN teacher/student encoders:
it re-extracts C16 features from a standalone C16 patch H5, makes its OWN
slide-stratified train/test split with many test slides (both classes),
and recomputes the linear/MLP probe, TOST equivalence, slide-cluster
bootstrap CI, and the cluster-vs-naive SE inflation — all at a valid cluster
count. It touches no distillation artifact and retrains nothing.

Iterates every outputs/v4_full/<teacher>/<student>/best.pt. Teacher features
are cached across a teacher's three students.

Usage:
  python3 scripts/eval_c16_equivalence.py \
      --h5 /path/to/cache/patches/patches_camelyon16_82.h5 \
      --test_frac 0.5 --out outputs/v4_full/c16_equiv_82
  # optional --max_slides 23 to reproduce the small-cohort comparison
"""
import argparse, glob, json, os, sys, csv
from pathlib import Path
import numpy as np
import h5py
import torch

sys.path.insert(0, str(Path(__file__).parent))
from distill_wsi_model import StudentModel, TeacherModel, _IMAGENET_MEAN, _IMAGENET_STD
from evaluate_distillation import linear_cka, cohen_kappa, linear_probe
from evaluate_v4_downstream import mlp_probe
from stats_v4 import slide_cluster_bootstrap, tost_equivalence

STU_MAP = {"vit-tiny": "vit_tiny_patch16_224",
           "vit-small": "vit_small_patch16_224",
           "vit-base": "vit_base_patch16_224"}
Z95 = 1.959964


def load_c16(h5_path, max_slides=None, seed=42):
    with h5py.File(h5_path, "r") as f:
        labels = f["labels"][:].astype(np.int64)
        slide_ids = f["slide_ids"][:].astype(np.int64)
        n = labels.shape[0]
    if max_slides is not None:
        keep_slides = np.unique(slide_ids)[:max_slides]
        mask = np.isin(slide_ids, keep_slides)
        idx = np.where(mask)[0]
    else:
        idx = np.arange(n)
    return idx, labels, slide_ids


def stratified_slide_split(slide_ids, labels, idx, test_frac, seed=42):
    """Split SLIDES (not tiles) into train/test, stratified by slide label,
    so both tumor and normal slides appear in test. A slide's label is the
    majority tile label (C16 slides are single-label by construction)."""
    rng = np.random.default_rng(seed)
    slides = np.unique(slide_ids[idx])
    slide_lab = {}
    for s in slides:
        m = (slide_ids == s) & np.isin(np.arange(len(slide_ids)), idx)
        slide_lab[s] = int(round(labels[m].mean()))
    test_slides = []
    for lab in (0, 1):
        grp = np.array([s for s in slides if slide_lab[s] == lab])
        rng.shuffle(grp)
        k = max(1, int(round(len(grp) * test_frac)))
        test_slides.extend(grp[:k].tolist())
    test_slides = set(test_slides)
    te = np.array([i for i in idx if slide_ids[i] in test_slides])
    tr = np.array([i for i in idx if slide_ids[i] not in test_slides])
    return tr, te, len(test_slides), len(slides)


@torch.no_grad()
def extract(model, h5_path, indices, device, bs=256):
    feats = []
    with h5py.File(h5_path, "r", swmr=True) as f:
        patches = f["patches"]
        for i in range(0, len(indices), bs):
            chunk = np.sort(indices[i:i + bs])
            arr = patches[chunk]  # (b,H,W,3) uint8
            t = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
            t = (t - _IMAGENET_MEAN) / _IMAGENET_STD
            out = model(t.to(device))
            feat = out["feat"] if isinstance(out, dict) else out
            feats.append(feat.float().cpu())
    return torch.cat(feats).numpy()


def inflation(d_correct_t, d_correct_s, groups, n):
    """cluster SE / naive SE of the paired accuracy difference d=s-t."""
    d = np.asarray(d_correct_s, float) - np.asarray(d_correct_t, float)
    se_cluster = slide_cluster_bootstrap(d, groups, np.mean, n_boot=1000, seed=42)["se"]
    se_naive = d.std(ddof=1) / np.sqrt(len(d))
    return se_cluster, se_naive, (se_cluster / se_naive if se_naive > 0 else float("nan"))


def _auc(y, score):
    """Binary AUC; nan if a resample is single-class."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(score)))


def auc_cluster_tost(y, t_score, s_score, groups, margin=0.05, n_boot=1000, seed=42):
    """Slide-cluster bootstrap TOST on the AUC difference d = AUC_s - AUC_t.

    AUC is imbalance-robust, unlike raw accuracy (which on C16's
    normal-dominated tiles sits below the majority-class baseline and makes
    the equivalence test degenerate). Resample SLIDES with replacement, pool
    their tiles, recompute both AUCs and d on each resample. Equivalent iff
    the 90% bootstrap CI of d lies within [-margin, +margin] (the standard
    TOST CI). Also returns the cluster-vs-naive SE inflation of d (slides vs
    tiles resampled)."""
    y = np.asarray(y); t = np.asarray(t_score); s = np.asarray(s_score)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in uniq}

    def d_on(idx):
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            return None
        return _auc(yy, s[idx]) - _auc(yy, t[idx])

    d_hat = _auc(y, s) - _auc(y, t)
    rng = np.random.default_rng(seed)
    dc = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([by[uniq[i]] for i in pick])
        v = d_on(idx)
        if v is not None and np.isfinite(v):
            dc.append(v)
    rng2 = np.random.default_rng(seed + 1)
    dn = []
    for _ in range(n_boot):
        idx = rng2.integers(0, len(y), size=len(y))
        v = d_on(idx)
        if v is not None and np.isfinite(v):
            dn.append(v)
    out = {"d": float(d_hat), "n_slides": int(len(uniq)),
           "t_auc": _auc(y, t), "s_auc": _auc(y, s)}
    if len(dc) < 20:
        out.update(lo=float("nan"), hi=float("nan"), equiv=False,
                   se_cluster=float("nan"), se_naive=float("nan"),
                   inflation=float("nan"))
        return out
    dc = np.array(dc)
    lo, hi = float(np.quantile(dc, 0.05)), float(np.quantile(dc, 0.95))
    se_c = float(dc.std(ddof=1))
    se_n = float(np.std(dn, ddof=1)) if len(dn) >= 20 else float("nan")
    # Bootstrap TOST p-values, one per one-sided null, with the +1
    # correction so a p-value is never exactly zero. The pair-level
    # equivalence p-value is max(p_lower, p_upper) (intersection-union);
    # these feed the Benjamini-Bogomolov multiplicity step in stats_v4.
    p_lower = float((np.sum(dc <= -margin) + 1) / (len(dc) + 1))
    p_upper = float((np.sum(dc >= margin) + 1) / (len(dc) + 1))
    out.update(p_lower=p_lower, p_upper=p_upper, p_tost=max(p_lower, p_upper))
    out.update(lo=lo, hi=hi, equiv=bool(lo > -margin and hi < margin),
               se_cluster=se_c, se_naive=se_n,
               inflation=(se_c / se_n if se_n and se_n > 0 else float("nan")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--root", default="outputs/v4_full")
    ap.add_argument("--test_frac", type=float, default=0.5)
    ap.add_argument("--max_slides", type=int, default=None)
    ap.add_argument("--tost_margin", type=float, default=0.03)
    ap.add_argument("--auc_margin", type=float, default=0.05,
                    help="equivalence margin on the AUC difference (headline metric)")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_scores", action="store_true",
                    help="write per-pair test-tile probe scores to <out dir>/scores/")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    idx, labels, slide_ids = load_c16(args.h5, args.max_slides)
    tr, te, n_test_slides, n_total_slides = stratified_slide_split(
        slide_ids, labels, idx, args.test_frac)
    y_tr = torch.from_numpy(labels[tr]).to(device)
    y_te_np = labels[te]
    grp_te = slide_ids[te]
    print(f"[split] {n_total_slides} slides -> train tiles {len(tr)} / "
          f"test tiles {len(te)} across {n_test_slides} test slides")

    ckpts = sorted(glob.glob(f"{args.root}/*/*/best.pt"))
    teacher_cache = {}
    rows = []
    for ck in ckpts:
        parts = ck.split("/")
        fm, stu_short = parts[-3], parts[-2]
        if stu_short not in STU_MAP:
            continue
        try:
            if fm not in teacher_cache:
                tm = TeacherModel(fm).eval().to(device)
                teacher_cache[fm] = (extract(tm, args.h5, tr, device, args.bs),
                                     extract(tm, args.h5, te, device, args.bs))
                del tm; torch.cuda.empty_cache()
            t_tr, t_te = teacher_cache[fm]

            sm = StudentModel(STU_MAP[stu_short], embed_dim=256, n_classes=0,
                              pretrained=False).to(device)
            sd = torch.load(ck, map_location=device)
            sm.load_state_dict(sd.get("student", sd), strict=False)
            sm.eval()
            s_tr = extract(sm, args.h5, tr, device, args.bs)
            s_te = extract(sm, args.h5, te, device, args.bs)
            del sm; torch.cuda.empty_cache()

            rec = {"teacher": fm, "student": stu_short,
                   "n_test_slides": n_test_slides, "n_test_tiles": len(te)}
            cka = float(linear_cka(torch.from_numpy(t_te), torch.from_numpy(s_te)))
            rec["cka"] = round(cka, 4)
            for kind, fn in (("linear", linear_probe), ("mlp", mlp_probe)):
                tr_r = fn(torch.from_numpy(t_tr).to(device), y_tr,
                          torch.from_numpy(t_te).to(device),
                          torch.from_numpy(y_te_np).to(device), n_classes=2)
                sr = fn(torch.from_numpy(s_tr).to(device), y_tr,
                        torch.from_numpy(s_te).to(device),
                        torch.from_numpy(y_te_np).to(device), n_classes=2)

                def pred(r):
                    p = r["preds"]
                    return (p.cpu().numpy() if isinstance(p, torch.Tensor)
                            else np.asarray(p))

                def prob1(r):
                    p = r.get("probs")
                    p = (p.cpu().numpy() if isinstance(p, torch.Tensor)
                         else np.asarray(p))
                    return p[:, 1] if p.ndim == 2 else p
                tp, sp = pred(tr_r), pred(sr)
                t_corr = (tp == y_te_np).astype(float)
                s_corr = (sp == y_te_np).astype(float)
                # --- accuracy-based (kept for reference; degenerate on imbalanced C16) ---
                tost = tost_equivalence(t_corr, s_corr, grp_te,
                                        margin=args.tost_margin, n_boot=1000, seed=42)
                sc, sn, infl = inflation(t_corr, s_corr, grp_te, len(te))
                rec[f"{kind}_t_acc"] = round(float(t_corr.mean()), 4)
                rec[f"{kind}_s_acc"] = round(float(s_corr.mean()), 4)
                rec[f"{kind}_d_mean"] = round(tost["d_mean"], 4)
                rec[f"{kind}_acc_equiv"] = bool(tost["equivalent"])
                rec[f"{kind}_kappa"] = round(cohen_kappa(tp, sp), 4)
                rec[f"{kind}_se_cluster"] = round(sc, 5)
                rec[f"{kind}_se_naive"] = round(sn, 5)
                rec[f"{kind}_acc_inflation"] = round(infl, 3)
                # --- AUC-based clustered TOST (HEADLINE; imbalance-robust) ---
                from sklearn.metrics import balanced_accuracy_score
                au = auc_cluster_tost(y_te_np, prob1(tr_r), prob1(sr), grp_te,
                                      margin=args.auc_margin, n_boot=1000, seed=42)
                rec[f"{kind}_t_auc"] = round(au["t_auc"], 4)
                rec[f"{kind}_s_auc"] = round(au["s_auc"], 4)
                rec[f"{kind}_auc_d"] = round(au["d"], 4)
                rec[f"{kind}_auc_ci"] = [round(au["lo"], 4), round(au["hi"], 4)]
                rec[f"{kind}_equiv"] = au["equiv"]          # headline equivalence
                rec[f"{kind}_auc_inflation"] = round(au["inflation"], 3)
                rec[f"{kind}_auc_p_tost"] = au.get("p_tost")
                if args.save_scores:
                    sdir = Path(args.out).parent / "scores"
                    sdir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(sdir / f"{fm}__{stu_short}__{kind}.npz",
                                        y=y_te_np, t=prob1(tr_r), s=prob1(sr),
                                        groups=grp_te.astype(str))
                rec[f"{kind}_t_bacc"] = round(float(balanced_accuracy_score(y_te_np, tp)), 4)
                rec[f"{kind}_s_bacc"] = round(float(balanced_accuracy_score(y_te_np, sp)), 4)
            rows.append(rec)
            print(f"  {fm:14s} {stu_short:10s} cka={rec['cka']:.3f} "
                  f"lin AUC T={rec['linear_t_auc']:.3f}/S={rec['linear_s_auc']:.3f} "
                  f"Δauc={rec['linear_auc_d']:+.3f} equiv={rec['linear_equiv']} "
                  f"infl={rec['linear_auc_inflation']:.2f}× nsl={n_test_slides}")
        except Exception as e:
            print(f"  [skip] {fm}/{stu_short}: {type(e).__name__}: {e}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"args": vars(args), "n_test_slides": n_test_slides,
               "n_total_slides": n_total_slides, "rows": rows},
              open(args.out + ".json", "w"), indent=2)
    if rows:
        cols = list(rows[0].keys())
        with open(args.out + ".csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
            for r in rows:
                w.writerow(r)
        lin = np.array([r["linear_auc_inflation"] for r in rows
                        if np.isfinite(r["linear_auc_inflation"])])
        eqv = sum(r["linear_equiv"] for r in rows)
        eqv_mlp = sum(r["mlp_equiv"] for r in rows)
        tauc = np.array([r["linear_t_auc"] for r in rows])
        sauc = np.array([r["linear_s_auc"] for r in rows])
        print(f"\n[{len(rows)} pairs | {n_test_slides} test slides | AUC margin {args.auc_margin}] "
              f"teacher AUC {tauc.mean():.3f} / student AUC {sauc.mean():.3f}; "
              f"AUC-TOST equiv linear {eqv}/{len(rows)} mlp {eqv_mlp}/{len(rows)}; "
              f"AUC-diff inflation median "
              f"{np.median(lin):.2f}× IQR [{np.quantile(lin,.25):.2f},{np.quantile(lin,.75):.2f}]")
        print(f"[wrote] {args.out}.json / .csv")


if __name__ == "__main__":
    main()
