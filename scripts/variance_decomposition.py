#!/usr/bin/env python3
"""Decompose the AUROC-difference variance of the C16 equivalence test into its
three sources, all ignored by a naive single-fit single-seed patch-level test:
  (1) test-set / slide-clustering  (slide-cluster bootstrap, already in the atlas)
  (2) probe-fitting  (linear probe has random init; refit K times)
  (3) training seed  (seed 42 vs seed 2025 students)

For the 12 non-hibou pairs with both training seeds: extract teacher (cached) +
student features on the 41-slide annotated C16 test set, fit the linear probe K
times (varying init) for teacher and student, compute d = AUROC_s - AUROC_t per
fit. Report the probe-fitting SD of d (within pair,seed), the between-seed shift
of the probe-mean d, and the slide-cluster SE, so the three components are
comparable.

Output: outputs/v4_full/variance_decomposition.json
"""
import sys, json, os, statistics as st
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from eval_c16_equivalence import load_c16, stratified_slide_split, extract, STU_MAP
from distill_wsi_model import TeacherModel, StudentModel
from evaluate_distillation import linear_probe

device = "cuda" if torch.cuda.is_available() else "cpu"
H5 = "/path/to/cache/patches/patches_c16_annotated.h5"
K = 8
TEACHERS = ["h-optimus-0", "phikon", "uni", "virchow2"]
STUDENTS = ["vit-tiny", "vit-small", "vit-base"]


def probe_scores(ftr, ytr, fte, k):
    torch.manual_seed(1000 + k)
    r = linear_probe(torch.from_numpy(ftr).to(device), torch.from_numpy(ytr).to(device),
                     torch.from_numpy(fte).to(device), torch.from_numpy(np.zeros(len(fte), int)).to(device),
                     n_classes=2)
    p = r["probs"]; p = p.cpu().numpy() if isinstance(p, torch.Tensor) else np.asarray(p)
    return p[:, 1]


def main():
    idx, labels, sid = load_c16(H5, None)
    tr, te, nts, _ = stratified_slide_split(sid, labels, idx, 0.5)
    ytr = labels[tr]; yte = labels[te]
    rows = []
    tcache = {}
    for fm in TEACHERS:
        if fm not in tcache:
            tm = TeacherModel(fm).eval().to(device)
            ttr = extract(tm, H5, tr, device); tte = extract(tm, H5, te, device)
            del tm; torch.cuda.empty_cache()
            t_auc = np.array([roc_auc_score(yte, probe_scores(ttr, ytr, tte, k)) for k in range(K)])
            tcache[fm] = (ttr, tte, t_auc)
        ttr, tte, t_auc = tcache[fm]
        for stu in STUDENTS:
            perseed = {}
            for root, tag in [("outputs/v4_full", "s42"), ("outputs/v4_seed2", "s2")]:
                ck = f"{root}/{fm}/{stu}/best.pt"
                if not os.path.exists(ck):
                    continue
                sm = StudentModel(STU_MAP[stu], embed_dim=256, n_classes=0, pretrained=False).to(device)
                sd = torch.load(ck, map_location=device)
                sm.load_state_dict(sd.get("student", sd), strict=False); sm.eval()
                sctr = extract(sm, H5, tr, device); scte = extract(sm, H5, te, device)
                del sm; torch.cuda.empty_cache()
                s_auc = np.array([roc_auc_score(yte, probe_scores(sctr, ytr, scte, k)) for k in range(K)])
                d = s_auc - t_auc  # paired by probe-init k
                perseed[tag] = {"mean_d": float(d.mean()), "sd_d_probe": float(d.std(ddof=1)),
                                "s_auc_mean": float(s_auc.mean()), "s_auc_sd": float(s_auc.std(ddof=1))}
            rec = {"teacher": fm, "student": stu, "t_auc_sd_probe": float(t_auc.std(ddof=1)), **perseed}
            if "s42" in perseed and "s2" in perseed:
                rec["between_seed_dshift"] = abs(perseed["s42"]["mean_d"] - perseed["s2"]["mean_d"])
            rows.append(rec)
            msg = f"{fm}/{stu}: t_auc_sd(probe)={rec['t_auc_sd_probe']:.4f}"
            for tag in ("s42", "s2"):
                if tag in perseed:
                    msg += f" | {tag} mean_d={perseed[tag]['mean_d']:+.3f} sd_d(probe)={perseed[tag]['sd_d_probe']:.4f}"
            if "between_seed_dshift" in rec:
                msg += f" | |Δd(seed)|={rec['between_seed_dshift']:.4f}"
            print(msg)

    # aggregate
    probe_sd = [r[t]["sd_d_probe"] for r in rows for t in ("s42", "s2") if t in r]
    seed_shift = [r["between_seed_dshift"] for r in rows if "between_seed_dshift" in r]
    # slide-cluster SE of d from the annotated bootstrap CI (half-width / 1.645)
    ann = {(x["teacher"], x["student"]): x for x in json.load(open("outputs/v4_full/c16_equiv_annotated.json"))["rows"]}
    clus_se = []
    for r in rows:
        a = ann.get((r["teacher"], r["student"]))
        if a and a.get("linear_auc_ci"):
            lo, hi = a["linear_auc_ci"]; clus_se.append((hi - lo) / (2 * 1.645))
    out = {"K": K, "n_pairs": len(rows), "rows": rows,
           "sigma_probe_median": round(float(st.median(probe_sd)), 4),
           "seed_shift_median": round(float(st.median(seed_shift)), 4) if seed_shift else None,
           "sigma_cluster_median": round(float(st.median(clus_se)), 4) if clus_se else None}
    json.dump(out, open("outputs/v4_full/variance_decomposition.json", "w"), indent=2)
    print("\n=== VARIANCE COMPONENTS (median across pairs) ===")
    print(f"  probe-fitting SD of d          sigma_probe   = {out['sigma_probe_median']}")
    print(f"  between-seed |shift| of mean d               = {out['seed_shift_median']}")
    print(f"  slide-cluster SE of d          sigma_cluster = {out['sigma_cluster_median']}")
    print(f"  (equivalence margin delta = 0.05)")
    print("[saved] outputs/v4_full/variance_decomposition.json")


if __name__ == "__main__":
    main()
