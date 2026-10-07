#!/usr/bin/env python3
"""SECOND-cohort equivalence eval: Kather colorectal MSI vs MSS, clustered by
patient. Uses the predefined Kather TRAIN/TEST patient split (is_test flag in
the H5). ~103 test patients => valid cluster-robust inference (>>25 clusters),
and MSI/MSS is a correct patient-level tile label (no focal-tumour noise).

Same machinery as the CAMELYON16 headline: frozen teacher/student encoders,
linear + MLP probe, AUROC difference under a slide(patient)-cluster TOST at
delta=0.05, plus the cluster-vs-naive SE inflation. Adds a high-cluster point
to the SE-inflation-vs-#clusters curve and a second equivalence task.
"""
import argparse, glob, json, csv, sys
from pathlib import Path
import numpy as np, h5py, torch
sys.path.insert(0, str(Path(__file__).parent))
from distill_wsi_model import StudentModel, TeacherModel
from evaluate_distillation import linear_cka, cohen_kappa, linear_probe
from evaluate_v4_downstream import mlp_probe
from eval_c16_equivalence import extract, auc_cluster_tost, STU_MAP

DELTA = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="/path/to/cache/patches/patches_kather_msi.h5")
    ap.add_argument("--root", default="outputs/v4_full")
    ap.add_argument("--out", default="outputs/v4_full/kather_msi_equiv")
    ap.add_argument("--bs", type=int, default=256)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with h5py.File(args.h5, "r") as f:
        labels = f["labels"][:].astype(np.int64)
        pid = f["slide_ids"][:].astype(np.int64)
        is_te = f["is_test"][:].astype(bool)
    tr = np.where(~is_te)[0]; te = np.where(is_te)[0]
    y_tr = torch.from_numpy(labels[tr]).to(device)
    y_te = labels[te]; grp_te = pid[te]
    n_pat_te = len(np.unique(grp_te))
    print(f"[split] train {len(tr)} tiles / test {len(te)} tiles, "
          f"{n_pat_te} test patients (clusters); "
          f"test MSI {int(y_te.sum())} / MSS {int((y_te==0).sum())}")

    ckpts = sorted(glob.glob(f"{args.root}/*/*/best.pt"))
    tcache = {}
    rows = []
    for ck in ckpts:
        parts = ck.split("/"); fm, stu = parts[-3], parts[-2]
        if stu not in STU_MAP: continue
        try:
            if fm not in tcache:
                tm = TeacherModel(fm).eval().to(device)
                tcache[fm] = (extract(tm, args.h5, tr, device, args.bs),
                              extract(tm, args.h5, te, device, args.bs))
                del tm; torch.cuda.empty_cache()
            t_tr, t_te = tcache[fm]
            sm = StudentModel(STU_MAP[stu], embed_dim=256, n_classes=0,
                              pretrained=False).to(device)
            sd = torch.load(ck, map_location=device)
            sm.load_state_dict(sd.get("student", sd), strict=False); sm.eval()
            s_tr = extract(sm, args.h5, tr, device, args.bs)
            s_te = extract(sm, args.h5, te, device, args.bs)
            del sm; torch.cuda.empty_cache()

            rec = {"teacher": fm, "student": stu, "n_test_patients": n_pat_te}
            for kind, fn in (("linear", linear_probe), ("mlp", mlp_probe)):
                tr_r = fn(torch.from_numpy(t_tr).to(device), y_tr,
                          torch.from_numpy(t_te).to(device),
                          torch.from_numpy(y_te).to(device), n_classes=2)
                sr = fn(torch.from_numpy(s_tr).to(device), y_tr,
                        torch.from_numpy(s_te).to(device),
                        torch.from_numpy(y_te).to(device), n_classes=2)
                p1 = lambda r: (r["probs"].cpu().numpy() if isinstance(r["probs"], torch.Tensor)
                                else np.asarray(r["probs"]))[:, 1]
                au = auc_cluster_tost(y_te, p1(tr_r), p1(sr), grp_te, margin=DELTA)
                lo, hi = au["lo"], au["hi"]
                st = ("equivalent" if (hi < DELTA and lo > -DELTA)
                      else "inequivalent" if (lo > DELTA or hi < -DELTA) else "inconclusive")
                rec[f"{kind}_t_auc"] = round(au["t_auc"], 4)
                rec[f"{kind}_s_auc"] = round(au["s_auc"], 4)
                rec[f"{kind}_auc_d"] = round(au["d"], 4)
                rec[f"{kind}_state"] = st
                rec[f"{kind}_inflation"] = round(au["inflation"], 3)
                rec[f"{kind}_auc_p_tost"] = au.get("p_tost")
            rows.append(rec)
            print(f"  {fm:14s} {stu:10s} lin T={rec['linear_t_auc']:.3f} "
                  f"S={rec['linear_s_auc']:.3f} {rec['linear_state']:12s} "
                  f"infl={rec['linear_inflation']:.2f}x")
        except Exception as e:
            print(f"  [skip] {fm}/{stu}: {type(e).__name__}: {e}")

    json.dump({"n_test_patients": n_pat_te, "rows": rows},
              open(args.out + ".json", "w"), indent=1)
    if rows:
        with open(args.out + ".csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
            for r in rows: w.writerow(r)
        from collections import Counter
        c = Counter(r["linear_state"] for r in rows)
        inf = np.array([r["linear_inflation"] for r in rows if np.isfinite(r["linear_inflation"])])
        ta = np.mean([r["linear_t_auc"] for r in rows])
        print(f"\n[{len(rows)} pairs | {n_pat_te} patient clusters] teacher AUC {ta:.3f}; "
              f"linear states {dict(c)}; inflation median {np.median(inf):.2f}x "
              f"IQR[{np.quantile(inf,.25):.2f},{np.quantile(inf,.75):.2f}]")
        print(f"[wrote] {args.out}.json/.csv")


if __name__ == "__main__":
    main()
