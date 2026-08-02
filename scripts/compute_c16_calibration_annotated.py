#!/usr/bin/env python3
"""ECE/MCE/AURC/risk@0.9 on the VALID 41-slide lesion-annotated C16 probe.

The atlas calibration numbers (atlas_summary.json c16lin_*_ece, etc.) were
computed on the degenerate 2-slide filename-label probe (teacher AUROC ~0.57),
which makes "calibration" of a near-chance classifier meaningless. This script
recomputes the same metrics on the 41-slide lesion-annotated probe used for the
headline TOST (teacher AUROC ~0.94), reusing eval_c16_equivalence's split /
feature extraction / probes and eval_v4_addons' calibration functions.

Leaves c16_equiv_annotated.json untouched.
Output: outputs/v4_full/c16_calibration_annotated.json (+ .md summary)
"""
import glob, json, sys, statistics as st
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
from eval_c16_equivalence import load_c16, stratified_slide_split, extract, STU_MAP
from distill_wsi_model import StudentModel, TeacherModel
from evaluate_distillation import linear_probe
from evaluate_v4_downstream import mlp_probe
from eval_v4_addons import (expected_calibration_error, max_calibration_error,
                            aurc, risk_at_coverage)

H5 = "/path/to/cache/patches/patches_c16_annotated.h5"
ROOT = "outputs/v4_full"
OUT = f"{ROOT}/c16_calibration_annotated.json"
device = "cuda" if torch.cuda.is_available() else "cpu"


def calib(probs2d, y):
    ece, _ = expected_calibration_error(probs2d, y)
    return {"ece": round(float(ece), 5),
            "mce": round(float(max_calibration_error(probs2d, y)), 5),
            "aurc": round(float(aurc(probs2d, y)), 5),
            "risk09": round(float(risk_at_coverage(probs2d, y, 0.9)), 5)}


def probs2d(r):
    p = r["probs"]
    return p.cpu().numpy() if isinstance(p, torch.Tensor) else np.asarray(p)


def main():
    idx, labels, slide_ids = load_c16(H5, None)
    tr, te, n_test_slides, n_total = stratified_slide_split(slide_ids, labels, idx, 0.5)
    y_tr = torch.from_numpy(labels[tr]).to(device)
    y_te = labels[te]
    y_te_t = torch.from_numpy(y_te).to(device)
    print(f"[split] {n_total} slides -> {n_test_slides} test slides / {len(te)} test tiles; "
          f"tumour-tile frac {y_te.mean():.3f}")

    ckpts = sorted(glob.glob(f"{ROOT}/*/*/best.pt"))
    tcache, rows = {}, []
    for ck in ckpts:
        parts = ck.split("/"); fm, stu = parts[-3], parts[-2]
        if stu not in STU_MAP:
            continue
        try:
            if fm not in tcache:
                tm = TeacherModel(fm).eval().to(device)
                tcache[fm] = (extract(tm, H5, tr, device), extract(tm, H5, te, device))
                del tm; torch.cuda.empty_cache()
            t_tr, t_te = tcache[fm]
            sm = StudentModel(STU_MAP[stu], embed_dim=256, n_classes=0, pretrained=False).to(device)
            sd = torch.load(ck, map_location=device)
            sm.load_state_dict(sd.get("student", sd), strict=False); sm.eval()
            s_tr = extract(sm, H5, tr, device); s_te = extract(sm, H5, te, device)
            del sm; torch.cuda.empty_cache()

            rec = {"teacher": fm, "student": stu, "n_test_slides": n_test_slides, "n_test_tiles": len(te)}
            for kind, fn in (("linear", linear_probe), ("mlp", mlp_probe)):
                tr_r = fn(torch.from_numpy(t_tr).to(device), y_tr,
                          torch.from_numpy(t_te).to(device), y_te_t, n_classes=2)
                sr = fn(torch.from_numpy(s_tr).to(device), y_tr,
                        torch.from_numpy(s_te).to(device), y_te_t, n_classes=2)
                for who, r in (("t", tr_r), ("s", sr)):
                    for k, v in calib(probs2d(r), y_te).items():
                        rec[f"{kind}_{who}_{k}"] = v
            rows.append(rec)
            print(f"  {fm:14s} {stu:10s} lin ECE T={rec['linear_t_ece']:.3f} S={rec['linear_s_ece']:.3f} "
                  f"AURC T={rec['linear_t_aurc']:.3f} S={rec['linear_s_aurc']:.3f}")
        except Exception as e:
            print(f"  [skip] {fm}/{stu}: {type(e).__name__}: {e}")

    json.dump({"h5": H5, "n_test_slides": n_test_slides, "probe": "41-slide lesion-annotated",
               "rows": rows}, open(OUT, "w"), indent=2)
    # summary over non-hibou (matching the paper's exclusion rule)
    nh = [r for r in rows if "hibou" not in r["teacher"]]
    def med(key):
        v = [r[key] for r in nh if r.get(key) is not None]
        return round(st.median(v), 4) if v else None
    thr = [(r["teacher"], r["student"]) for r in nh if r.get("linear_s_ece", 9) <= 0.05]
    thr_t = sum(1 for r in nh if r.get("linear_t_ece", 9) <= 0.05)
    md = [f"# C16 calibration on the 41-slide lesion-annotated probe\n",
          f"non-hibou pairs: {len(nh)} | test slides {n_test_slides}\n",
          f"- linear teacher median ECE {med('linear_t_ece')} -> student {med('linear_s_ece')}",
          f"- linear teacher median AURC {med('linear_t_aurc')} -> student {med('linear_s_aurc')}",
          f"- students meeting ECE<=0.05: {len(thr)}/{len(nh)} {thr}",
          f"- teachers meeting ECE<=0.05: {thr_t}/{len(nh)}\n"]
    open(f"{ROOT}/c16_calibration_annotated.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"[saved] {OUT}  ({len(rows)} pairs)")


if __name__ == "__main__":
    main()
