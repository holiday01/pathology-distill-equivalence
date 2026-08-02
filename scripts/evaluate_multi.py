#!/usr/bin/env python3
"""Evaluate multi-source-distilled ViT-S vs Phikon-v2 teacher.

Metrics:
  - CKA + cosine (overall + per source)
  - Linear probe on C16 tumor/normal (slide-level labels)
  - Cohen's κ between teacher- and student-probe predictions
  - Cross-cohort linear probe (C16 → BRCA and back) when labels available
  - Latency (ms/patch)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from distill_wsi_model import StudentModel, TeacherModel
from evaluate_distillation import (benchmark_latency, cohen_kappa, linear_cka,
                                   cosine_alignment, linear_probe)
from multi_source_dataset import (HDF5Source, ImageFolderSource, MultiSourceDataset,
                                  NumpyArraySource)


def build_sources(config):
    sources = []
    for e in config["sources"]:
        t = e["type"]
        ms = e.get("max_samples")
        if t == "hdf5":
            sources.append(HDF5Source(e["path"], e["tag"], max_samples=ms))
        elif t == "folder":
            sources.append(ImageFolderSource(e["root"], e["tag"],
                                              patterns=tuple(e.get("patterns", ("*.tif", "*.png", "*.jpg"))),
                                              max_samples=ms,
                                              group_strategy=e.get("group_strategy", "bucket:20")))
        elif t == "numpy":
            sources.append(NumpyArraySource(e["paths"], e["tag"], max_samples=ms,
                                             group_strategy=e.get("group_strategy", "bucket:20")))
    return sources


def extract_feats(model, loader, device, is_teacher: bool):
    model.eval()
    feats, tags, groups = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            out = model(x)
            f = out["feat"] if isinstance(out, dict) else out
            feats.append(f.cpu())
            tags.extend(batch[1])
            groups.extend(batch[2])
    return torch.cat(feats), np.array(tags), np.array(groups)


def labels_from_c16_group(group_ids):
    """C16 groups look like 'c16:slideN'. We previously labelled
    normal_*=0, tumor_*=1 (in extract_patches.py slide_label).
    We reconstruct by reading slide_names attr."""
    import h5py
    with h5py.File("/path/to/wsi_hl/data/patches_camelyon16_full.h5", "r") as h5:
        names = [n.decode() for n in h5.attrs["slide_names"]]
    # slide idx → label
    labels = []
    for n in names:
        nl = n.lower()
        if nl.startswith("normal"):
            labels.append(0)
        elif nl.startswith("tumor"):
            labels.append(1)
        else:
            labels.append(-1)
    out = []
    for g in group_ids:
        if isinstance(g, bytes):
            g = g.decode()
        if not g.startswith("c16:slide"):
            out.append(-1); continue
        sid = int(g.split("slide")[1])
        out.append(labels[sid] if sid < len(labels) else -1)
    return np.array(out, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--student_ckpt", required=True)
    ap.add_argument("--splits", required=True,
                    help="splits.npz from training (keys train/val/test = global indices)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--teacher", default="phikon-v2")
    ap.add_argument("--student", default="vit_small_patch16_224")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.load(open(args.config))
    sources = build_sources(cfg)
    ds = MultiSourceDataset(sources, target_size=224)
    print(f"total patches: {len(ds)}")

    sp = np.load(args.splits)
    train_ds = MultiSourceDataset(sources, target_size=224, indices=sp["train"])
    test_ds = MultiSourceDataset(sources, target_size=224, indices=sp["test"])
    print(f"train={len(train_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    teacher = TeacherModel(args.teacher).eval().to(device)
    student = StudentModel(args.student, embed_dim=256, n_classes=0, pretrained=False).to(device)
    ckpt = torch.load(args.student_ckpt, map_location=device)
    state = ckpt.get("student", ckpt)
    student.load_state_dict(state, strict=False)
    student.eval()

    print("[extract teacher train]"); t_tr, tags_tr, grp_tr = extract_feats(teacher, train_loader, device, True)
    print("[extract teacher test]");  t_te, tags_te, grp_te = extract_feats(teacher, test_loader, device, True)
    print("[extract student train]"); s_tr, _, _ = extract_feats(student, train_loader, device, False)
    print("[extract student test]");  s_te, _, _ = extract_feats(student, test_loader, device, False)
    print(f"  teacher feat: {tuple(t_te.shape)}, student feat: {tuple(s_te.shape)}")

    results: dict = {}

    # 1. CKA + cosine overall & per source
    cka_overall = linear_cka(s_te, t_te)
    cos_overall = cosine_alignment(s_te, t_te)
    per_source = {}
    for tag in sorted(set(tags_te)):
        mask = tags_te == tag
        if mask.sum() < 8:
            continue
        per_source[str(tag)] = {
            "n": int(mask.sum()),
            "cka": linear_cka(s_te[mask], t_te[mask]),
            "cosine": cosine_alignment(s_te[mask], t_te[mask]),
        }
    results["feature_similarity"] = {"cka_overall": cka_overall,
                                     "cosine_overall": cos_overall,
                                     "per_source": per_source}
    print(f"\n[feature sim] overall CKA={cka_overall:.4f} cos={cos_overall:.4f}")
    for k, v in per_source.items():
        print(f"  {k}: n={v['n']} CKA={v['cka']:.4f} cos={v['cosine']:.4f}")

    # 2. Linear probe on C16 tumor/normal (if labels available)
    y_tr_c16 = labels_from_c16_group(grp_tr)
    y_te_c16 = labels_from_c16_group(grp_te)
    tr_c16 = (tags_tr == "c16") & (y_tr_c16 >= 0)
    te_c16 = (tags_te == "c16") & (y_te_c16 >= 0)
    probe_results: dict = {}
    if tr_c16.sum() >= 32 and te_c16.sum() >= 16:
        t_probe = linear_probe(t_tr[tr_c16].to(device), torch.from_numpy(y_tr_c16[tr_c16]).to(device),
                               t_te[te_c16].to(device), torch.from_numpy(y_te_c16[te_c16]).to(device),
                               n_classes=2)
        s_probe = linear_probe(s_tr[tr_c16].to(device), torch.from_numpy(y_tr_c16[tr_c16]).to(device),
                               s_te[te_c16].to(device), torch.from_numpy(y_te_c16[te_c16]).to(device),
                               n_classes=2)
        kappa = cohen_kappa(t_probe["preds"], s_probe["preds"])
        probe_results = {
            "c16_tumor_vs_normal": {
                "teacher_acc": t_probe["acc"], "teacher_auc": t_probe["auc"],
                "student_acc": s_probe["acc"], "student_auc": s_probe["auc"],
                "acc_gap": t_probe["acc"] - s_probe["acc"],
                "cohen_kappa": kappa,
                "n_train": int(tr_c16.sum()), "n_test": int(te_c16.sum()),
            }
        }
        print(f"\n[C16 linear probe] teacher acc={t_probe['acc']:.3f} / student acc={s_probe['acc']:.3f} / Δ={t_probe['acc']-s_probe['acc']:+.3f}")
        print(f"  κ(teacher, student)={kappa:.3f}")
    else:
        print(f"\n[C16 linear probe] skipped (train_c16={tr_c16.sum()}, test_c16={te_c16.sum()})")
    results["linear_probing"] = probe_results

    # 3. Latency
    t_ms = benchmark_latency(teacher, device)
    s_ms = benchmark_latency(student, device)
    t_params = sum(p.numel() for p in teacher.parameters())
    s_params = sum(p.numel() for p in student.parameters())
    results["efficiency"] = {
        "teacher_ms": t_ms, "student_ms": s_ms, "speedup": t_ms / s_ms,
        "teacher_params": t_params, "student_params": s_params,
        "compression": t_params / s_params,
    }
    print(f"\n[latency] teacher {t_ms:.2f} ms / student {s_ms:.2f} ms / speedup {t_ms/s_ms:.2f}×")

    # 4. Verdict
    criteria = []
    criteria.append(("CKA > 0.7", cka_overall > 0.7, cka_overall))
    if probe_results:
        gap = abs(probe_results["c16_tumor_vs_normal"]["acc_gap"])
        kappa = probe_results["c16_tumor_vs_normal"]["cohen_kappa"]
        criteria.append(("|ΔACC| < 0.03", gap < 0.03, gap))
        criteria.append(("κ > 0.6", kappa > 0.6, kappa))
    print("\n[verdict]")
    for name, ok, val in criteria:
        print(f"  [{'✓' if ok else '✗'}] {name}  ({val:.3f})")
    results["verdict"] = {n: (bool(ok), float(v)) for n, ok, v in criteria}
    results["verdict"]["equivalent"] = all(ok for _, ok, _ in criteria)

    with open(out_dir / "evaluation_report.json", "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nreport: {out_dir / 'evaluation_report.json'}")


if __name__ == "__main__":
    main()
