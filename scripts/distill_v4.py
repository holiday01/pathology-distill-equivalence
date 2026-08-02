#!/usr/bin/env python3
"""v4 pilot distillation: L_CLS (1-cos, L2) + L_PAT (cos).

Pilot stage per v4 §4.2: measure teacher-student paired ρ on 3 teachers × 30 epochs
to decide n regime before full sweep with MGD + DINO.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from distill_wsi_model import StudentModel, TeacherModel
from distill_v4_loss import V4DistillLoss
from multi_source_dataset import (HDF5Source, ImageFolderSource, MultiSourceDataset,
                                  NumpyArraySource, group_split)


def build_sources(config):
    sources = []
    for entry in config["sources"]:
        t, tag = entry["type"], entry["tag"]
        ms = entry.get("max_samples")
        if t == "hdf5":
            sources.append(HDF5Source(entry["path"], tag, max_samples=ms))
        elif t == "folder":
            sources.append(ImageFolderSource(entry["root"], tag,
                                             patterns=tuple(entry.get("patterns",
                                                 ("*.tif", "*.png", "*.jpg"))),
                                             max_samples=ms,
                                             group_strategy=entry.get("group_strategy", "bucket:20")))
        elif t == "numpy":
            sources.append(NumpyArraySource(entry["paths"], tag, max_samples=ms,
                                             group_strategy=entry.get("group_strategy", "bucket:20")))
        print(f"  + {tag} [{t}]: {len(sources[-1])} samples")
    return sources


def run_epoch(student, teacher, loader, optimizer, criterion, device, train):
    student.train(train)
    tot, n = 0.0, 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    for batch in loader:
        patches = batch[0].to(device, non_blocking=True)
        with torch.no_grad():
            t_out = teacher(patches, return_patches=True)
        if train:
            optimizer.zero_grad()
        s_out = student(patches, return_patches=True)
        loss = criterion(s_out, t_out)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
        tot += loss.item(); n += 1
    peak_mb = (torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if torch.cuda.is_available() else 0
    return tot / max(n, 1), peak_mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", default="vit_small_patch16_224")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lam_cls", type=float, default=1.0)
    ap.add_argument("--lam_pat", type=float, default=0.5)
    ap.add_argument("--use_mgd", action="store_true")
    ap.add_argument("--use_dino", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.config) as f:
        cfg = json.load(f)

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    print("[sources]")
    srcs = build_sources(cfg)
    ds = MultiSourceDataset(srcs, target_size=224)
    print(f"  total: {len(ds)}")

    print("[group split]")
    tr, va, te = group_split(ds, val_frac=0.1, test_frac=0.1, seed=args.seed)
    tr_ds = MultiSourceDataset(srcs, 224, indices=ds.indices[tr])
    va_ds = MultiSourceDataset(srcs, 224, indices=ds.indices[va])
    print(f"  train={len(tr)} val={len(va)} test={len(te)}")

    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True, drop_last=True,
                       persistent_workers=(args.num_workers > 0))
    va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    print("[models]")
    teacher = TeacherModel(args.teacher).eval().to(device)
    student = StudentModel(args.student, embed_dim=256, n_classes=0, pretrained=True).to(device)
    criterion = V4DistillLoss(
        student_cls_dim=student.embed_dim,
        student_patch_dim=student.backbone.num_features,
        teacher_dim=teacher.embed_dim,
        lam_cls=args.lam_cls, lam_pat=args.lam_pat,
        use_mgd=args.use_mgd, use_dino=args.use_dino,
    ).to(device)
    t_params = sum(p.numel() for p in teacher.parameters())
    s_params = sum(p.numel() for p in student.parameters())
    print(f"  teacher={t_params:,}  student={s_params:,}  compression={t_params/s_params:.1f}x")
    print(f"  loss = L_CLS(λ={args.lam_cls}) + L_PAT(λ={args.lam_pat}) +"
          f" L_MGD({'on' if args.use_mgd else 'off'}) + L_DINO({'on' if args.use_dino else 'off'})")

    opt = torch.optim.AdamW(
        list(student.parameters()) + list(criterion.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history, best, best_ep, patience = [], float("inf"), -1, args.patience
    t0 = time.time()
    start_ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        tr_l, tr_peak_mb = run_epoch(student, teacher, tr_ld, opt, criterion, device, train=True)
        sch.step()
        va_l, va_peak_mb = run_epoch(student, teacher, va_ld, opt, criterion, device, train=False)
        ep_dt = time.time() - ep_t0
        dt = time.time() - t0
        history.append({"epoch": epoch, "train_loss": tr_l, "val_loss": va_l,
                        "elapsed_total_s": dt, "elapsed_epoch_s": ep_dt,
                        "peak_vram_train_mb": tr_peak_mb, "peak_vram_val_mb": va_peak_mb})
        mark = ""
        improved = va_l < best - 1e-5
        if improved:
            best, best_ep, patience = va_l, epoch, args.patience
            torch.save({"epoch": epoch, "student": student.state_dict(),
                        "criterion": criterion.state_dict(), "val_loss": va_l,
                        "config": vars(args)}, out / "best.pt")
            mark = " ★"
        else:
            patience -= 1
        lr = opt.param_groups[0]["lr"]
        print(f"  epoch {epoch:3d} | train {tr_l:.4f} | val {va_l:.4f}{mark} | lr {lr:.2e} | "
              f"ep {ep_dt:.0f}s tot {dt:.0f}s | vram {tr_peak_mb:.0f}MB")
        if patience <= 0:
            print(f"  [early stop] no improvement for {args.patience} epochs")
            break

    end_ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    np.savez(out / "splits.npz", train=ds.indices[tr], val=ds.indices[va], test=ds.indices[te])
    with open(out / "training_log.json", "w") as f:
        json.dump({"args": vars(args), "source_config": cfg, "history": history,
                   "best_val": best, "best_epoch": best_ep,
                   "teacher_params": t_params, "student_params": s_params,
                   "compression_ratio": t_params / s_params,
                   "start_ts": start_ts, "end_ts": end_ts}, f, indent=2)
    print(f"\n[done] best val {best:.4f} @ ep {best_ep}")
    print(f"       output: {out}")


if __name__ == "__main__":
    main()
