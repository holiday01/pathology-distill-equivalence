#!/usr/bin/env python3
"""Multi-source FM → ViT-S hybrid distillation with group-aware split."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from distill_wsi_model import (HybridDistillLoss, StudentModel, TeacherModel)
from multi_source_dataset import (HDF5Source, ImageFolderSource, MultiSourceDataset,
                                  NumpyArraySource, group_split)


def build_sources(config: dict) -> list:
    sources = []
    for entry in config["sources"]:
        t = entry["type"]
        tag = entry["tag"]
        ms = entry.get("max_samples")
        if t == "hdf5":
            sources.append(HDF5Source(entry["path"], tag, max_samples=ms))
        elif t == "folder":
            sources.append(ImageFolderSource(entry["root"], tag,
                                             patterns=tuple(entry.get("patterns", ("*.tif", "*.png", "*.jpg"))),
                                             max_samples=ms,
                                             group_strategy=entry.get("group_strategy", "bucket:20")))
        elif t == "numpy":
            sources.append(NumpyArraySource(entry["paths"], tag, max_samples=ms,
                                             group_strategy=entry.get("group_strategy", "bucket:20")))
        else:
            raise ValueError(f"unknown source type {t}")
        print(f"  + {tag} [{t}]: {len(sources[-1])} samples")
    return sources


def run_epoch(student, teacher, loader, optimizer, criterion, device, train: bool,
              grad_clip: float = 0.0):
    student.train(train)
    total, n = 0.0, 0
    for batch in loader:
        patches = batch[0].to(device, non_blocking=True)
        with torch.no_grad():
            teacher_out = teacher(patches, return_patches=True)
        if train:
            optimizer.zero_grad()
        student_out = student(patches, return_patches=True)
        loss = criterion(student_out, teacher_out)
        if train:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON with source list + hparams")
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=5,
                    help="early stop: epochs without val improvement")
    ap.add_argument("--teacher", default="phikon-v2")
    ap.add_argument("--student", default="vit_small_patch16_224")
    ap.add_argument("--pretrained_student", action="store_true", default=True)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--amp", action="store_true",
                    help="mixed precision forward")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with open(args.config) as f:
        config = json.load(f)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[sources]")
    sources = build_sources(config)
    dataset = MultiSourceDataset(sources, target_size=224)
    print(f"  total patches: {len(dataset)}")

    print("[group split]")
    train_idx, val_idx, test_idx = group_split(dataset, val_frac=args.val_frac,
                                               test_frac=args.test_frac, seed=args.seed)
    print(f"  train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    train_ds = MultiSourceDataset(sources, target_size=224, indices=dataset.indices[train_idx])
    val_ds = MultiSourceDataset(sources, target_size=224, indices=dataset.indices[val_idx])
    test_ds = MultiSourceDataset(sources, target_size=224, indices=dataset.indices[test_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    print("[models]")
    teacher = TeacherModel(args.teacher).eval().to(device)
    student = StudentModel(args.student, embed_dim=256, n_classes=0,
                           pretrained=args.pretrained_student).to(device)
    criterion = HybridDistillLoss(student.embed_dim, teacher.embed_dim,
                                  projector="mlp").to(device)

    t_params = sum(p.numel() for p in teacher.parameters())
    s_params = sum(p.numel() for p in student.parameters())
    print(f"  teacher={t_params:,}, student={s_params:,}, compression={t_params/s_params:.1f}x")

    all_params = list(student.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_val = float("inf")
    best_epoch = -1
    patience_left = args.patience
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss = run_epoch(student, teacher, train_loader, optimizer, criterion, device,
                            train=True, grad_clip=args.grad_clip)
        scheduler.step()
        val_loss = run_epoch(student, teacher, val_loader, optimizer, criterion, device,
                             train=False)
        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss,
                        "lr": lr, "elapsed": dt})
        improved = val_loss < best_val - 1e-5
        tag = " ★" if improved else ""
        print(f"  epoch {epoch:3d} | train {tr_loss:.4f} | val {val_loss:.4f}{tag} | lr {lr:.2e} | {dt:.0f}s")

        if improved:
            best_val = val_loss
            best_epoch = epoch
            patience_left = args.patience
            torch.save({
                "epoch": epoch,
                "student": student.state_dict(),
                "criterion": criterion.state_dict(),
                "val_loss": val_loss,
                "config": vars(args),
                "source_config": config,
            }, out_dir / "best.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  [early stop] no improvement for {args.patience} epochs")
                break

    torch.save({"epoch": epoch, "student": student.state_dict(),
                "criterion": criterion.state_dict()}, out_dir / "last.pt")

    # Save test indices for downstream equivalence evaluation
    np.savez(out_dir / "splits.npz",
             train=dataset.indices[train_idx],
             val=dataset.indices[val_idx],
             test=dataset.indices[test_idx])

    with open(out_dir / "training_log.json", "w") as f:
        json.dump({
            "args": vars(args),
            "source_config": config,
            "compression_ratio": t_params / s_params,
            "teacher_params": t_params,
            "student_params": s_params,
            "history": history,
            "best_epoch": best_epoch,
            "best_val": best_val,
        }, f, indent=2)

    print(f"\n[done] best val loss {best_val:.4f} @ epoch {best_epoch}")
    print(f"       output: {out_dir}")


if __name__ == "__main__":
    main()
