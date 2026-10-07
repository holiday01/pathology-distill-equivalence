#!/usr/bin/env python3
"""Slide-disjoint medical-center probe on TCGA-BRCA.

The earlier probe (eval_v4_addons.medical_center_probe) split tiles at random,
so tiles of one slide fell on both sides and the probe could identify the
centre by recognising the slide; every model saturated at AUC 1.000. Here the
split is by slide: StratifiedGroupKFold over slides (groups = slide, strata =
tissue-source site), restricted to sites with at least MIN_SLIDES slides, and
the one-vs-rest macro AUC is computed on pooled out-of-fold predictions.

Features: TILES_PER_SLIDE tiles per slide (fixed seed), frozen encoders, the
same preprocessing as the CAMELYON16 equivalence evaluation.

Usage:
  python3 scripts/center_probe_slide_disjoint.py \
      --out outputs/v4_full/center_probe_slide_disjoint.json
"""
import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from distill_wsi_model import StudentModel, TeacherModel
from eval_c16_equivalence import extract, STU_MAP
from eval_v4_addons import tcga_tss_code
from paper_cohort import keep_teacher

MIN_SLIDES = 3
TILES_PER_SLIDE = 100
N_FOLDS = 3
SEED = 42


def probe(X, centre, slide):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler
    classes = np.unique(centre)
    prob = np.zeros((len(X), len(classes)))
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in cv.split(X, centre, groups=slide):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(sc.transform(X[tr]), centre[tr])
        p = clf.predict_proba(sc.transform(X[te]))
        cols = [list(classes).index(c) for c in clf.classes_]
        prob[np.ix_(te, cols)] = p
    auc = roc_auc_score(centre, prob, multi_class="ovr", average="macro",
                        labels=classes)
    acc = float((classes[prob.argmax(1)] == centre).mean())
    return float(auc), acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/patches_brca.h5")
    ap.add_argument("--root", default="outputs/v4_full")
    ap.add_argument("--out", default="outputs/v4_full/center_probe_slide_disjoint.json")
    ap.add_argument("--bs", type=int, default=128)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with h5py.File(a.h5, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in f.attrs["slide_names"]]
        sid = f["slide_ids"][:]
    tss_of = {i: tcga_tss_code(names[i]) for i in np.unique(sid)}
    per = Counter(tss_of.values())
    keep_sites = {t for t, n in per.items() if n >= MIN_SLIDES and t != "UNK"}
    rng = np.random.default_rng(SEED)
    idx = []
    for s in sorted(np.unique(sid)):
        if tss_of[s] not in keep_sites:
            continue
        rows = np.where(sid == s)[0]
        idx.extend(rng.choice(rows, min(TILES_PER_SLIDE, len(rows)), replace=False))
    idx = np.sort(np.array(idx))
    slide = sid[idx]
    centre = np.array([tss_of[s] for s in slide])
    print(f"[data] {len(np.unique(slide))} slides, {len(keep_sites)} sites, {len(idx)} tiles", flush=True)

    rows = []
    teachers = sorted({Path(p).parts[-3] for p in glob.glob(f"{a.root}/*/*/best.pt")})
    for fm in teachers:
        if not keep_teacher(fm):
            continue
        tm = TeacherModel(fm).eval().to(device)
        with torch.no_grad():
            Xt = extract(tm, a.h5, idx, device, a.bs)
        del tm; torch.cuda.empty_cache()
        t_auc, t_acc = probe(Xt, centre, slide)
        print(f"  {fm:14s} teacher auc={t_auc:.3f}", flush=True)
        for stu in ("vit-tiny", "vit-small", "vit-base"):
            ck = Path(a.root) / fm / stu / "best.pt"
            if not ck.exists():
                continue
            sm = StudentModel(STU_MAP[stu], embed_dim=256, n_classes=0, pretrained=False)
            sd = torch.load(ck, map_location="cpu")
            sm.load_state_dict(sd.get("student", sd), strict=False)
            sm = sm.eval().to(device)
            with torch.no_grad():
                Xs = extract(sm, a.h5, idx, device, a.bs)
            del sm; torch.cuda.empty_cache()
            s_auc, s_acc = probe(Xs, centre, slide)
            rows.append({"teacher": fm, "student": stu,
                         "t_center_auc": t_auc, "s_center_auc": s_auc,
                         "t_center_acc": t_acc, "s_center_acc": s_acc})
            print(f"    {stu:10s} student auc={s_auc:.3f}", flush=True)
    json.dump({"n_slides": int(len(np.unique(slide))), "n_sites": len(keep_sites),
               "n_tiles": int(len(idx)), "min_slides_per_site": MIN_SLIDES,
               "tiles_per_slide": TILES_PER_SLIDE, "folds": N_FOLDS,
               "sites": {t: per[t] for t in sorted(keep_sites)}, "rows": rows},
              open(a.out, "w"), indent=1)
    print(f"[wrote] {a.out}")


if __name__ == "__main__":
    main()
