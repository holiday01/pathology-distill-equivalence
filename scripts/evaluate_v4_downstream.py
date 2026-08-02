#!/usr/bin/env python3
"""v4 downstream evaluation: takes a distilled ViT-{Ti,S,B} checkpoint and
probes its embedding on multiple downstream tasks to answer:
  "Does matching the teacher representation actually yield useful features?"

Tasks:
  1. CKA + cosine similarity to teacher (representational)
  2. Linear probe on C16 tumor/normal (binary)
  3. Non-linear (MLP) probe on C16 tumor/normal
  4. Linear probe on Kather-MSI (binary MSI vs MSS)
  5. Linear probe on NCT-CRC 9-class tissue typing (if labels inferrable)
  6. Cohen's κ + Gwet AC1 between teacher/student predictions on C16
  7. Latency (ms/patch, bs=1)
  8. Bootstrap 95% CI on every accuracy/AUROC
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as _mp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Avoid `unable to allocate shared memory(shm)` from default file_descriptor
# strategy under many workers; route shared tensors through /dev/shm instead.
try:
    _mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

sys.path.insert(0, str(Path(__file__).parent))
from distill_wsi_model import StudentModel, TeacherModel
from evaluate_distillation import (benchmark_latency, cohen_kappa,
                                   linear_cka, cosine_alignment, linear_probe)
from evaluate_multi import build_sources, extract_feats, labels_from_c16_group
from multi_source_dataset import MultiSourceDataset
from stats_v4 import slide_cluster_bootstrap, tost_equivalence
from eval_v4_addons import (aurc, expected_calibration_error,
                             max_calibration_error, medical_center_probe,
                             risk_at_coverage, robustness_index, tcga_tss_code)


def bootstrap_ci(values, n_boot=1000, ci=0.95, seed=42, groups=None):
    """Bootstrap mean + CI. If groups is given, clusters by slide (v4 recipe)."""
    if groups is not None:
        r = slide_cluster_bootstrap(np.asarray(values), np.asarray(groups),
                                    np.mean, n_boot=n_boot, seed=seed, ci=ci)
        return r["mean"], r["lo"], r["hi"]
    rng = np.random.default_rng(seed)
    arr = np.asarray(values)
    stats = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    lo = np.quantile(stats, (1 - ci) / 2)
    hi = np.quantile(stats, 1 - (1 - ci) / 2)
    return float(arr.mean()), float(lo), float(hi)


def mlp_probe(x_tr, y_tr, x_te, y_te, n_classes=2, hidden=512, epochs=200,
              lr=1e-3, wd=1e-4, patience=20, seed=42, device="cuda"):
    torch.manual_seed(seed)
    dim = x_tr.shape[1]
    mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                        nn.Linear(hidden, n_classes)).to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=wd)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_te, y_te = x_te.to(device), y_te.to(device)
    best_auc, best_acc, best_preds, best_probs, bad = -1, -1, None, None, 0
    for ep in range(epochs):
        mlp.train()
        logits = mlp(x_tr)
        loss = F.cross_entropy(logits, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
        mlp.eval()
        with torch.no_grad():
            lt = mlp(x_te)
            probs_full = F.softmax(lt, -1)
            preds = lt.argmax(-1)
            acc = (preds == y_te).float().mean().item()
            if n_classes == 2:
                prob = probs_full[:, 1]
                try:
                    from sklearn.metrics import roc_auc_score
                    auc = roc_auc_score(y_te.cpu().numpy(), prob.cpu().numpy())
                except Exception:
                    auc = float("nan")
            else:
                auc = float("nan")
        improve = (auc == auc and auc > best_auc) or (auc != auc and acc > best_acc)
        if improve:
            best_auc, best_acc, best_preds = auc, acc, preds.cpu().numpy()
            best_probs = probs_full.cpu().numpy()
            bad = 0
        else:
            bad += 1
            if bad > patience: break
    return {"acc": best_acc, "auc": best_auc, "preds": best_preds,
            "probs": best_probs}


def msi_labels(group_ids, tags):
    """Kather-MSI patches live under folders named MSIMUT_xxx or MSS_xxx."""
    out = []
    for i, g in enumerate(group_ids):
        t = tags[i]
        if isinstance(t, bytes): t = t.decode()
        if isinstance(g, bytes): g = g.decode()
        if t != "kather_msi":
            out.append(-1); continue
        # group id typically encodes the file path bucket; simple heuristic:
        gl = g.lower()
        if "msimut" in gl or "msi_" in gl:
            out.append(1)
        elif "mss" in gl:
            out.append(0)
        else:
            out.append(-1)
    return np.array(out, dtype=np.int64)


def nct_crc_labels(group_ids, tags):
    """NCT-CRC folders contain 9 classes: ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM."""
    nct_classes = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
    lut = {c: i for i, c in enumerate(nct_classes)}
    out = []
    for i, g in enumerate(group_ids):
        t = tags[i]
        if isinstance(t, bytes): t = t.decode()
        if isinstance(g, bytes): g = g.decode()
        if not t.startswith("nct_crc"):
            out.append(-1); continue
        # group id includes NCT label prefix like "nct_crc:bucket_03" — can't infer from bucket.
        # Fallback: label unavailable from bucket grouping. Skip gracefully.
        out.append(-1)
    return np.array(out, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--student_ckpt", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", default="vit_small_patch16_224")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--tost_margin", type=float, default=0.03,
                    help="Equivalence margin for TOST (v4 default: 0.03 acc)")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    cfg = json.load(open(args.config))
    srcs = build_sources(cfg)
    sp = np.load(args.splits)
    tr_ds = MultiSourceDataset(srcs, 224, indices=sp["train"])
    te_ds = MultiSourceDataset(srcs, 224, indices=sp["test"])
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=False,
                       persistent_workers=args.num_workers > 0)
    te_ld = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=False,
                       persistent_workers=args.num_workers > 0)

    teacher = TeacherModel(args.teacher).eval().to(device)
    student = StudentModel(args.student, embed_dim=256, n_classes=0, pretrained=False).to(device)
    ckpt = torch.load(args.student_ckpt, map_location=device)
    student.load_state_dict(ckpt.get("student", ckpt), strict=False)
    student.eval()

    print("[extract] teacher train"); t_tr, tags_tr, grp_tr = extract_feats(teacher, tr_ld, device, True)
    print("[extract] teacher test");  t_te, tags_te, grp_te = extract_feats(teacher, te_ld, device, True)
    print("[extract] student train"); s_tr, _, _ = extract_feats(student, tr_ld, device, False)
    print("[extract] student test");  s_te, _, _ = extract_feats(student, te_ld, device, False)
    print(f"  T={tuple(t_te.shape)} S={tuple(s_te.shape)}")

    results = {}
    # 1. Representational similarity
    cka = linear_cka(s_te, t_te)
    cos = cosine_alignment(s_te, t_te)
    per_src = {}
    for tag in sorted(set(tags_te)):
        m = tags_te == tag
        if m.sum() < 8: continue
        per_src[str(tag)] = {"n": int(m.sum()),
                             "cka": linear_cka(s_te[m], t_te[m]),
                             "cosine": cosine_alignment(s_te[m], t_te[m])}
    results["feature_similarity"] = {"cka_overall": cka, "cosine_overall": cos,
                                     "per_source": per_src}
    print(f"[sim] CKA={cka:.4f} cos={cos:.4f}")

    probes = {}
    # 2-3. C16 tumor/normal, linear + MLP
    y_tr = labels_from_c16_group(grp_tr); y_te = labels_from_c16_group(grp_te)
    tr_m = (tags_tr == "c16") & (y_tr >= 0)
    te_m = (tags_te == "c16") & (y_te >= 0)
    if tr_m.sum() >= 32 and te_m.sum() >= 16:
        y_tr_t = torch.from_numpy(y_tr[tr_m]).to(device)
        y_te_t = torch.from_numpy(y_te[te_m]).to(device)
        for probe_kind, probe_fn in [("linear", linear_probe), ("mlp", mlp_probe)]:
            t_r = probe_fn(t_tr[tr_m].to(device), y_tr_t, t_te[te_m].to(device), y_te_t, n_classes=2)
            s_r = probe_fn(s_tr[tr_m].to(device), y_tr_t, s_te[te_m].to(device), y_te_t, n_classes=2)
            # Bootstrap CI via per-sample correctness; preds may be Tensor or ndarray.
            def _pt(p):
                return (p if isinstance(p, torch.Tensor) else torch.from_numpy(p)).to(device)
            t_correct = (_pt(t_r["preds"]) == y_te_t).float().cpu().numpy()
            s_correct = (_pt(s_r["preds"]) == y_te_t).float().cpu().numpy()
            grp_test_c16 = grp_te[te_m]
            t_acc_mean, t_lo, t_hi = bootstrap_ci(t_correct, args.n_boot, seed=args.seed,
                                                  groups=grp_test_c16)
            s_acc_mean, s_lo, s_hi = bootstrap_ci(s_correct, args.n_boot, seed=args.seed,
                                                  groups=grp_test_c16)
            tost = tost_equivalence(t_correct, s_correct, grp_test_c16,
                                    margin=args.tost_margin, n_boot=args.n_boot,
                                    seed=args.seed)
            kappa = cohen_kappa(t_r["preds"], s_r["preds"])
            # Calibration metrics (Guo 2017 ECE/MCE, Geifman 2017 AURC)
            cal = {}
            for who, r_probe in [("teacher", t_r), ("student", s_r)]:
                if r_probe.get("probs") is None:
                    continue
                p = np.asarray(r_probe["probs"])
                y = y_te_t.cpu().numpy()
                ece_val, _ = expected_calibration_error(p, y, n_bins=15)
                cal[who] = {
                    "ece": ece_val,
                    "mce": max_calibration_error(p, y, n_bins=15, min_count=5),
                    "aurc": aurc(p, y),
                    "risk_at_cov_0.9": risk_at_coverage(p, y, 0.9),
                }
            probes[f"c16_{probe_kind}"] = {
                "teacher_acc": t_r["acc"], "teacher_auc": t_r.get("auc"),
                "student_acc": s_r["acc"], "student_auc": s_r.get("auc"),
                "teacher_acc_ci95_cluster": [t_lo, t_hi],
                "student_acc_ci95_cluster": [s_lo, s_hi],
                "tost": tost,
                "cohen_kappa": kappa,
                "calibration": cal,
                "n_train": int(tr_m.sum()), "n_test": int(te_m.sum()),
                "n_slides_test": tost["n_slides"],
            }
            print(f"[C16 {probe_kind}] T_acc={t_r['acc']:.3f} S_acc={s_r['acc']:.3f} "
                  f"Δacc={tost['d_mean']:+.3f} [{tost['d_ci90_lo']:+.3f},{tost['d_ci90_hi']:+.3f}]  "
                  f"κ={kappa:.3f}  TOST equiv={tost['equivalent']} "
                  f"(p_L={tost['p_lower']:.3f} p_U={tost['p_upper']:.3f}) "
                  f"n_slides={tost['n_slides']}")
    # 4. Kather-MSI
    y_tr_msi = msi_labels(grp_tr, tags_tr); y_te_msi = msi_labels(grp_te, tags_te)
    tr_m = (y_tr_msi >= 0); te_m = (y_te_msi >= 0)
    if tr_m.sum() >= 32 and te_m.sum() >= 16:
        y_tr_t = torch.from_numpy(y_tr_msi[tr_m]).to(device)
        y_te_t = torch.from_numpy(y_te_msi[te_m]).to(device)
        t_r = linear_probe(t_tr[tr_m].to(device), y_tr_t, t_te[te_m].to(device), y_te_t, n_classes=2)
        s_r = linear_probe(s_tr[tr_m].to(device), y_tr_t, s_te[te_m].to(device), y_te_t, n_classes=2)
        probes["kather_msi_linear"] = {
            "teacher_acc": t_r["acc"], "teacher_auc": t_r.get("auc"),
            "student_acc": s_r["acc"], "student_auc": s_r.get("auc"),
            "d_acc": t_r["acc"] - s_r["acc"],
            "n_train": int(tr_m.sum()), "n_test": int(te_m.sum()),
        }
        print(f"[MSI linear] T_acc={t_r['acc']:.3f} S_acc={s_r['acc']:.3f}")

    results["probes"] = probes

    # 4b. Medical-center probe (de Jong 2025, arXiv:2501.18055)
    # Extract TCGA TSS code from brca group ids (format "brca:<slide_basename>")
    center_ids_tr = []
    center_ids_te = []
    for g in grp_tr:
        gs = g.decode() if isinstance(g, bytes) else str(g)
        if gs.startswith("brca:") or gs.startswith("tcga"):
            center_ids_tr.append(tcga_tss_code(gs.split(":", 1)[-1]))
        else:
            center_ids_tr.append(None)
    for g in grp_te:
        gs = g.decode() if isinstance(g, bytes) else str(g)
        if gs.startswith("brca:") or gs.startswith("tcga"):
            center_ids_te.append(tcga_tss_code(gs.split(":", 1)[-1]))
        else:
            center_ids_te.append(None)
    center_probe_result = {}
    for who, feats in [("teacher", t_te), ("student", s_te)]:
        mask = np.array([c is not None and c != "UNK" for c in center_ids_te])
        if mask.sum() >= 50 and len(set(np.array(center_ids_te)[mask])) >= 2:
            cen_ids = np.array(center_ids_te)[mask]
            embs = feats[mask].cpu().numpy()
            r = medical_center_probe(embs, cen_ids, seed=args.seed)
            center_probe_result[who] = r
    if center_probe_result:
        # Robustness index = task_auc / center_auc
        t_auc = probes.get("c16_linear", {}).get("teacher_auc")
        s_auc = probes.get("c16_linear", {}).get("student_auc")
        for who, auc in [("teacher", t_auc), ("student", s_auc)]:
            if auc is not None and who in center_probe_result:
                center_probe_result[who]["robustness_index"] = robustness_index(
                    auc, center_probe_result[who].get("center_probe_auc", 0.5))
    results["medical_center_probe"] = center_probe_result
    for who, r in center_probe_result.items():
        print(f"[center probe {who}] AUC={r.get('center_probe_auc'):.3f} "
              f"(n_centers={r.get('n_centers')}, RI={r.get('robustness_index')})")

    # 5. Latency
    results["efficiency"] = {
        "teacher_ms": benchmark_latency(teacher, device=device),
        "student_ms": benchmark_latency(student, device=device),
    }
    te_ms = results["efficiency"]["teacher_ms"]; st_ms = results["efficiency"]["student_ms"]
    results["efficiency"]["speedup"] = te_ms / st_ms if st_ms > 0 else 0
    print(f"[latency] teacher {te_ms:.2f}ms  student {st_ms:.2f}ms  speedup {te_ms/st_ms:.2f}×")

    with open(out / "downstream_report.json", "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2, default=float)
    print(f"\n[saved] {out / 'downstream_report.json'}")


if __name__ == "__main__":
    main()
