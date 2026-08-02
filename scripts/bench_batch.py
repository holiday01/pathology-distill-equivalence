#!/usr/bin/env python3
"""Quick throughput benchmark for distill_multi configs.

Measures samples/sec and peak VRAM for a (batch_size, num_workers) combo
by running warmup + timed batches against the real pipeline.
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
from distill_wsi_model import HybridDistillLoss, StudentModel, TeacherModel
from multi_source_dataset import MultiSourceDataset, group_split
from distill_multi import build_sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch_size", type=int, required=True)
    ap.add_argument("--num_workers", type=int, required=True)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--measure", type=int, default=30)
    ap.add_argument("--teacher", default="phikon-v2")
    ap.add_argument("--student", default="vit_small_patch16_224")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config) as f:
        config = json.load(f)

    print(f"[bench] bs={args.batch_size} nw={args.num_workers}")
    sources = build_sources(config)
    full = MultiSourceDataset(sources, target_size=224)
    train_idx, _, _ = group_split(full, val_frac=0.1, test_frac=0.1, seed=args.seed)
    train_ds = MultiSourceDataset(sources, target_size=224, indices=full.indices[train_idx])

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True,
                        persistent_workers=(args.num_workers > 0))

    device = "cuda"
    teacher = TeacherModel(args.teacher).eval().to(device)
    student = StudentModel(args.student, embed_dim=256, n_classes=0, pretrained=True).to(device)
    criterion = HybridDistillLoss(student.embed_dim, teacher.embed_dim, projector="mlp").to(device)
    optimizer = torch.optim.AdamW(list(student.parameters()) + list(criterion.parameters()),
                                  lr=args.lr, weight_decay=1e-4)

    torch.cuda.reset_peak_memory_stats()
    it = iter(loader)
    student.train(True)

    # warmup
    for _ in range(args.warmup):
        batch = next(it)
        patches = batch[0].to(device, non_blocking=True)
        with torch.no_grad():
            t_out = teacher(patches, return_patches=True)
        optimizer.zero_grad()
        s_out = student(patches, return_patches=True)
        loss = criterion(s_out, t_out)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # measure
    t0 = time.time()
    n_samples = 0
    for _ in range(args.measure):
        batch = next(it)
        patches = batch[0].to(device, non_blocking=True)
        with torch.no_grad():
            t_out = teacher(patches, return_patches=True)
        optimizer.zero_grad()
        s_out = student(patches, return_patches=True)
        loss = criterion(s_out, t_out)
        loss.backward()
        optimizer.step()
        n_samples += patches.shape[0]
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    sps = n_samples / dt
    print(f"[result] bs={args.batch_size} nw={args.num_workers} | "
          f"{sps:.1f} samples/sec | {dt/args.measure*1000:.0f} ms/batch | "
          f"peak {peak_gb:.2f} GB")
    # Machine-readable
    print(f"RESULT bs={args.batch_size} nw={args.num_workers} sps={sps:.2f} ms_per_batch={dt/args.measure*1000:.1f} peak_gb={peak_gb:.3f}")


if __name__ == "__main__":
    main()
