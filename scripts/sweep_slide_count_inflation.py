#!/usr/bin/env python3
"""Is the cluster/naive SE inflation driven by the number of test slides?

The manuscript claims it is not: that the inflation is the design effect of
within-slide tile correlation, set by tiles-per-slide m and intra-slide
correlation rho rather than by cluster count, and therefore does not shrink on
larger evaluations. Figure 3a is the evidence for that claim, and its three
points are hard-coded literals in figure_equivalence.py rather than values read
from any result file; two of the three (6.9x at 2 slides, 5.46x at 12) do not
appear anywhere under outputs/.

Meanwhile the 41-slide and 135-slide evaluations disagree with the mechanism:
rho rose 0.247 -> 0.349 and m rose 264 -> 295, so sqrt(1+(m-1)rho) predicts the
inflation should rise 8.12x -> 10.19x, but the measured inflation FELL
6.83x -> 5.07x. Prediction and observation moved in opposite directions.

Those two evaluations differ in three ways at once (slide count, m, and tumour
fraction 5.3% -> 15.3%), so neither can be blamed. This script varies slide
count alone:

  * one patch bundle, one labelling, one extraction  -> composition fixed
  * whole slides are sampled, never tiles            -> m fixed by construction
  * the training half is fixed across every setting  -> probe quality fixed
  * tumour-containing and normal slides are drawn in
    proportion                                       -> class balance fixed

so a trend in the inflation across cohort sizes can only come from cluster
count. If the inflation is flat, the manuscript's claim holds and Figure 3a can
be rebuilt from real numbers. If it rises with cluster count, then 6.83x at 41
slides was a small-sample artefact of estimating between-slide variance from
six tumour slides, and the mechanism section needs rewriting rather than
renumbering.

Features are extracted once per model and cached to disk, because that is the
only expensive step; every (size, replicate) setting then re-fits the probe and
re-runs the bootstrap on the cached features.

Usage:
  python3 scripts/sweep_slide_count_inflation.py \
      --h5 /path/to/cache/patches/patches_c16_annotated_270.h5 \
      --out outputs/v4_full/c16_270/slide_count_sweep.json
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from distill_wsi_model import StudentModel, TeacherModel
from evaluate_distillation import linear_probe
from eval_c16_equivalence import load_c16, extract, STU_MAP, _auc

# Four teachers spanning 86M to 1.13B, each with its three students. The
# reported statistic is the median over pairs, so a consistent subset is enough
# to compare cohort sizes against one another.
TEACHERS = ["phikon", "uni", "virchow2", "h-optimus-0"]
STUDENTS = ["vit-tiny", "vit-small", "vit-base"]
SIZES = [10, 20, 40, 80, 135]
REPS = 8
N_BOOT = 1000


def cluster_and_naive_se(y, t_score, s_score, groups, n_boot=N_BOOT, seed=0):
    """SE of d = AUROC_s - AUROC_t under slide resampling and under tile
    resampling. Returns (se_cluster, se_naive)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in uniq}

    def d_on(idx):
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            return None
        return _auc(yy, s_score[idx]) - _auc(yy, t_score[idx])

    dc = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        v = d_on(np.concatenate([by[uniq[i]] for i in pick]))
        if v is not None and np.isfinite(v):
            dc.append(v)
    dn = []
    n = len(y)
    for _ in range(n_boot):
        v = d_on(rng.integers(0, n, size=n))
        if v is not None and np.isfinite(v):
            dn.append(v)
    if len(dc) < 20 or len(dn) < 20:
        return float("nan"), float("nan")
    return float(np.std(dc, ddof=1)), float(np.std(dn, ddof=1))


def stratified_test_slides(slide_ids, labels, pool, n_test, rng):
    """Draw n_test slides from `pool`, keeping the tumour/normal slide ratio of
    the pool. A slide counts as tumour if it carries at least one tumour tile."""
    tum = {s for s in pool if labels[slide_ids == s].sum() > 0}
    t_pool = np.array(sorted(tum))
    n_pool = np.array(sorted(set(pool) - tum))
    frac = len(t_pool) / len(pool)
    k_t = max(2, int(round(n_test * frac)))          # >=2 so AUROC is defined
    k_t = min(k_t, len(t_pool))
    k_n = min(len(n_pool), n_test - k_t)
    return np.concatenate([rng.choice(t_pool, k_t, replace=False),
                           rng.choice(n_pool, k_n, replace=False)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out", default="outputs/v4_full/c16_270/slide_count_sweep.json")
    ap.add_argument("--cache", default="outputs/v4_full/c16_270/_sweep_feats")
    ap.add_argument("--atlas_root", default="outputs/v4_full")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    idx, labels, slide_ids = load_c16(a.h5)
    uniq = np.unique(slide_ids[idx])

    # Fixed 50/50 slide split. The train half never changes, so probe quality is
    # held constant and only the test cluster count varies.
    rng0 = np.random.default_rng(42)
    tum = {s for s in uniq if labels[slide_ids == s].sum() > 0}
    t_all, n_all = np.array(sorted(tum)), np.array(sorted(set(uniq) - tum))
    rng0.shuffle(t_all); rng0.shuffle(n_all)
    train_slides = set(t_all[:len(t_all) // 2]) | set(n_all[:len(n_all) // 2])
    test_pool = np.array(sorted(set(uniq) - train_slides))
    tr = np.array([i for i in idx if slide_ids[i] in train_slides])
    print(f"[split] train {len(train_slides)} slides / {len(tr):,} tiles | "
          f"test pool {len(test_pool)} slides "
          f"({len([s for s in test_pool if s in tum])} tumour-bearing)", flush=True)

    cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True)
    pool_idx = np.array([i for i in idx if slide_ids[i] in set(test_pool)])

    def feats(name, loader):
        f_tr, f_te = cache / f"{name}_tr.npy", cache / f"{name}_te.npy"
        if f_tr.exists() and f_te.exists():
            return np.load(f_tr), np.load(f_te)
        m = loader().eval().to(device)
        a_tr = extract(m, a.h5, tr, device)
        a_te = extract(m, a.h5, pool_idx, device)
        del m; torch.cuda.empty_cache()
        np.save(f_tr, a_tr); np.save(f_te, a_te)
        return a_tr, a_te

    y_tr = torch.from_numpy(labels[tr].astype(np.int64)).to(device)
    y_pool = labels[pool_idx].astype(np.int64)
    g_pool = slide_ids[pool_idx]

    def score(f_tr, f_te):
        r = linear_probe(torch.from_numpy(f_tr).to(device), y_tr,
                         torch.from_numpy(f_te).to(device),
                         torch.from_numpy(y_pool).to(device), n_classes=2)
        p = r["probs"]
        p = p.cpu().numpy() if isinstance(p, torch.Tensor) else np.asarray(p)
        return p[:, 1] if p.ndim == 2 else p

    rows = []
    for te_name in TEACHERS:
        t_tr, t_te = feats(f"T_{te_name}", lambda: TeacherModel(te_name))
        t_s = score(t_tr, t_te)
        for stu in STUDENTS:
            ck = Path(a.atlas_root) / te_name / stu / "best.pt"
            if not ck.exists():
                print(f"  [skip] {te_name} x {stu}: no checkpoint", flush=True)
                continue
            def load_student(ck=ck, stu=stu):
                sm = StudentModel(STU_MAP[stu], embed_dim=256, n_classes=0,
                                  pretrained=False)
                sd = torch.load(ck, map_location="cpu")
                sm.load_state_dict(sd.get("student", sd), strict=False)
                return sm
            s_tr, s_te = feats(f"S_{te_name}_{stu}", load_student)
            s_s = score(s_tr, s_te)
            for n_test in SIZES:
                if n_test > len(test_pool):
                    continue
                for rep in range(REPS):
                    rng = np.random.default_rng(1000 * n_test + rep)
                    sel = set(stratified_test_slides(
                        slide_ids, labels, test_pool, n_test, rng))
                    mask = np.array([g in sel for g in g_pool])
                    if len(np.unique(y_pool[mask])) < 2:
                        continue
                    sec, sen = cluster_and_naive_se(
                        y_pool[mask], t_s[mask], s_s[mask], g_pool[mask], seed=rep)
                    if not (np.isfinite(sec) and np.isfinite(sen)) or sen <= 0:
                        continue
                    rows.append({"teacher": te_name, "student": stu,
                                 "n_test_slides": int(n_test), "rep": rep,
                                 "n_tiles": int(mask.sum()),
                                 "m": float(mask.sum() / len(sel)),
                                 "se_cluster": sec, "se_naive": sen,
                                 "inflation": sec / sen})
            print(f"  [done] {te_name} x {stu}", flush=True)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"sizes": SIZES, "reps": REPS,
                               "teachers": TEACHERS, "rows": rows}, indent=2))
    print(f"\n[wrote] {out}  ({len(rows)} measurements)")

    print(f"\n{'test slides':>12} {'median m':>9} {'median inflation':>17} {'IQR':>16} {'n':>4}")
    import statistics as st
    for n_test in SIZES:
        v = sorted(r["inflation"] for r in rows if r["n_test_slides"] == n_test)
        mm = [r["m"] for r in rows if r["n_test_slides"] == n_test]
        if not v:
            continue
        q = st.quantiles(v, n=4) if len(v) >= 4 else [float("nan")] * 3
        print(f"{n_test:12} {st.median(mm):9.0f} {st.median(v):16.2f}x "
              f"{q[0]:7.2f}-{q[2]:.2f} {len(v):5}")


if __name__ == "__main__":
    main()
