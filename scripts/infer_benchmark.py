#!/usr/bin/env python3
"""
Phikon-v2 (Teacher) vs Distilled ViT-S (Student) 推論速度對比
=============================================================
量測指標：
  - 吞吐量 (patches/sec)
  - 延遲 (ms/patch)
  - GPU 記憶體佔用 (MB)
  - 不同 batch size 下的表現

用法：
  python infer_benchmark.py
  python infer_benchmark.py --n_patches 5000 --repeats 5
"""

import argparse
import time
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np


# ──────────────────────────────────────────────────────────
# 模型載入
# ──────────────────────────────────────────────────────────

def load_teacher(device):
    from transformers import AutoModel
    print("  [Teacher] 載入 Phikon-v2 (owkin/phikon-v2)...")
    model = AutoModel.from_pretrained("owkin/phikon-v2")
    model = model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"           參數量: {n_params/1e6:.1f}M")
    return model, n_params


def load_student(ckpt_path, device):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from distill_wsi_model import StudentModel

    print(f"  [Student] 載入蒸餾後 ViT-S...")
    model = StudentModel("vit_small_patch16_224", embed_dim=256, n_classes=4)
    if Path(ckpt_path).exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"           已載入權重: {ckpt_path}")
    else:
        print(f"           [注意] 找不到 {ckpt_path}，使用隨機初始化")
    model = model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"           參數量: {n_params/1e6:.1f}M")
    return model, n_params


# ──────────────────────────────────────────────────────────
# 推論 benchmark
# ──────────────────────────────────────────────────────────

def warmup(model, dummy, device, is_teacher=False, n=10):
    """GPU warmup，避免冷啟動影響計時"""
    with torch.no_grad():
        for _ in range(n):
            if is_teacher:
                model(pixel_values=dummy)
            else:
                model(dummy)
    torch.cuda.synchronize()


def measure_throughput(model, batch_size, n_patches, device,
                       is_teacher=False, repeats=3):
    """
    量測吞吐量：用 CUDA events 精確計時，排除資料傳輸
    回傳：(patches/sec, ms/patch, peak_mem_MB)
    """
    dummy = torch.randn(batch_size, 3, 224, 224, device=device)

    # Warmup
    warmup(model, dummy, device, is_teacher=is_teacher, n=5)
    torch.cuda.reset_peak_memory_stats(device)

    latencies = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        n_batches = (n_patches + batch_size - 1) // batch_size

        start.record()
        with torch.no_grad():
            for _ in range(n_batches):
                if is_teacher:
                    model(pixel_values=dummy)
                else:
                    model(dummy)
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)
        latencies.append(elapsed_ms)

    total_patches = n_patches
    avg_ms = np.mean(latencies)
    ms_per_patch = avg_ms / total_patches
    patches_per_sec = total_patches / (avg_ms / 1000)
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2

    return patches_per_sec, ms_per_patch, peak_mem_mb


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_patches", type=int, default=3000,
                        help="模擬 WSI 的 patch 數量")
    parser.add_argument("--repeats", type=int, default=3,
                        help="每個設定重複幾次取平均")
    parser.add_argument("--ckpt", type=str,
                        default="outputs/distill/phikon_v2_hybrid/student_vit_small_patch16_224_hybrid.pt",
                        help="蒸餾後學生模型的 checkpoint 路徑")
    parser.add_argument("--output", type=str, default="outputs/benchmark/")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n推論速度測試")
    print(f"  裝置:    {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")
    print(f"  Patches: {args.n_patches:,}")
    print(f"  Repeats: {args.repeats}\n")

    # 載入模型
    print("[載入模型]")
    teacher, teacher_params = load_teacher(device)
    student, student_params = load_student(args.ckpt, device)
    compression = teacher_params / student_params
    print(f"\n  壓縮比: {compression:.1f}x  ({teacher_params/1e6:.1f}M → {student_params/1e6:.1f}M)\n")

    # 測試不同 batch size
    batch_sizes = [1, 8, 32, 64, 128]
    results = {"teacher": {}, "student": {}}

    print(f"[Batch size 掃描] n_patches={args.n_patches:,}\n")
    print(f"  {'Model':<12} {'BS':>4} {'Throughput':>14} {'ms/patch':>10} {'峰值 VRAM':>11}")
    print(f"  {'-'*55}")

    for bs in batch_sizes:
        for name, model, is_t in [("Teacher", teacher, True), ("Student", student, False)]:
            try:
                tps, mpp, mem = measure_throughput(
                    model, bs, args.n_patches, device,
                    is_teacher=is_t, repeats=args.repeats
                )
                key = name.lower()
                results[key][bs] = {"patches_per_sec": tps, "ms_per_patch": mpp, "peak_mem_mb": mem}
                print(f"  {name:<12} {bs:>4} {tps:>11,.0f}/s {mpp:>9.3f}ms {mem:>9.0f}MB")
            except torch.cuda.OutOfMemoryError:
                print(f"  {name:<12} {bs:>4} {'OOM':>14}")
                results[name.lower()][bs] = None
                torch.cuda.empty_cache()

    # 摘要（以 bs=32 為基準）
    print(f"\n[摘要] 以 batch_size=32 為基準")
    print(f"  {'':30} {'Teacher':>12} {'Student':>12} {'加速比':>8}")
    print(f"  {'-'*65}")

    ref_bs = 32
    t_res = results["teacher"].get(ref_bs)
    s_res = results["student"].get(ref_bs)

    if t_res and s_res:
        speedup = s_res["patches_per_sec"] / t_res["patches_per_sec"]
        mem_reduction = t_res["peak_mem_mb"] / s_res["peak_mem_mb"]
        print(f"  {'吞吐量 (patches/sec)':<30} {t_res['patches_per_sec']:>12,.0f} {s_res['patches_per_sec']:>12,.0f} {speedup:>7.1f}x")
        print(f"  {'延遲 (ms/patch)':<30} {t_res['ms_per_patch']:>12.3f} {s_res['ms_per_patch']:>12.3f} {t_res['ms_per_patch']/s_res['ms_per_patch']:>7.1f}x")
        print(f"  {'峰值 VRAM (MB)':<30} {t_res['peak_mem_mb']:>12.0f} {s_res['peak_mem_mb']:>12.0f} {mem_reduction:>7.1f}x")
        print(f"  {'模型參數量':<30} {teacher_params/1e6:>11.1f}M {student_params/1e6:>11.1f}M {compression:>7.1f}x")

        # 估算 1000 張 WSI 節省時間
        patches_per_slide = args.n_patches
        slides = 1000
        t_hours = (t_res["ms_per_patch"] * patches_per_slide * slides) / 1e3 / 3600
        s_hours = (s_res["ms_per_patch"] * patches_per_slide * slides) / 1e3 / 3600
        print(f"\n  [估算] 1000 張 WSI（各 {patches_per_slide:,} patches）")
        print(f"    Teacher: {t_hours:.1f} 小時")
        print(f"    Student: {s_hours:.1f} 小時")
        print(f"    節省時間: {t_hours - s_hours:.1f} 小時 ({(1 - s_hours/t_hours)*100:.0f}%↓)")

    # 儲存結果
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "infer_benchmark.json")
    with open(out_path, "w") as f:
        json.dump({
            "config": vars(args),
            "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "teacher_params": teacher_params,
            "student_params": student_params,
            "compression_ratio": compression,
            "results": {k: {str(bs): v for bs, v in vv.items()} for k, vv in results.items()}
        }, f, indent=2)
    print(f"\n  結果已儲存: {out_path}")


if __name__ == "__main__":
    main()
