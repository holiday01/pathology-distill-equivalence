#!/usr/bin/env python3
"""
WSI Pipeline Benchmark Script
==============================
量測 WSI 分析管線中各步驟的時間消耗：
  1. 組織偵測 (Tissue Detection)
  2. Patch 萃取 (Patch Extraction / Tiling)
  3. FM 特徵萃取 (Feature Extraction with Foundation Model)
  4. 聚合推論 (Aggregation)

用法：
  python benchmark_pipeline.py --wsi_dir /path/to/slides --output outputs/
  python benchmark_pipeline.py --wsi_dir /path/to/slides --model uni --batch_size 64
  python benchmark_pipeline.py --demo  # 使用合成資料模擬（不需要真實 WSI）
"""

import argparse
import time
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

# ──────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────

@dataclass
class StepTiming:
    name: str
    time_sec: float
    n_patches: int = 0
    extra: dict = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


@dataclass
class SlideTiming:
    slide_name: str
    steps: list
    total_time: float
    n_patches: int


# ──────────────────────────────────────────────────────────
# Step implementations
# ──────────────────────────────────────────────────────────

def step1_tissue_detection(slide_path: str) -> StepTiming:
    """組織偵測：使用 Otsu 閾值法或輕量模型"""
    t0 = time.perf_counter()
    try:
        import openslide
        slide = openslide.OpenSlide(slide_path)
        # 取最低倍率縮略圖做組織偵測
        thumb = slide.get_thumbnail((512, 512))
        thumb_np = np.array(thumb.convert("L"))
        # Otsu 閾值
        threshold = _otsu_threshold(thumb_np)
        tissue_mask = thumb_np < threshold
        tissue_ratio = tissue_mask.mean()
        slide.close()
    except ImportError:
        # Demo mode：模擬時間
        time.sleep(np.random.uniform(0.1, 0.5))
        tissue_ratio = np.random.uniform(0.3, 0.8)

    elapsed = time.perf_counter() - t0
    return StepTiming(
        name="tissue_detection",
        time_sec=elapsed,
        extra={"tissue_ratio": round(tissue_ratio, 3)}
    )


def step2_patch_extraction(slide_path: str,
                           patch_size: int = 224,
                           level: int = 0) -> StepTiming:
    """Patch 萃取：模擬或真實萃取"""
    t0 = time.perf_counter()
    n_patches = 0
    try:
        import openslide
        slide = openslide.OpenSlide(slide_path)
        w, h = slide.level_dimensions[0]
        stride = patch_size
        coords = []
        for y in range(0, h - patch_size, stride):
            for x in range(0, w - patch_size, stride):
                coords.append((x, y))
        n_patches = len(coords)
        # 實際萃取前 10 個 patch 作為 timing sample
        sample_n = min(10, n_patches)
        for x, y in coords[:sample_n]:
            _ = slide.read_region((x, y), level, (patch_size, patch_size))
        # 估算全部萃取時間（線性外推）
        if sample_n > 0:
            elapsed_sample = time.perf_counter() - t0
            estimated_full = elapsed_sample * (n_patches / sample_n)
        slide.close()
    except ImportError:
        # Demo mode：模擬
        n_patches = np.random.randint(1000, 9000)
        time.sleep(np.random.uniform(0.5, 3.0))

    elapsed = time.perf_counter() - t0
    return StepTiming(
        name="patch_extraction",
        time_sec=elapsed,
        n_patches=n_patches,
        extra={"patch_size": patch_size, "level": level}
    )


def step3_feature_extraction(n_patches: int,
                              model_name: str = "resnet50",
                              batch_size: int = 64,
                              demo: bool = False) -> StepTiming:
    """
    FM 特徵萃取：使用指定模型對所有 patches 做 forward pass
    model_name: 'resnet50', 'vit_small', 'vit_base', 'uni', 'conch', 'demo'
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    使用裝置: {device}, 模型: {model_name}, patches: {n_patches}")

    t0 = time.perf_counter()

    if demo or model_name == "demo":
        # 模擬各模型推論時間（每 patch，ms）
        model_speed = {
            "resnet50":  1.5,   # ms/patch
            "vit_small": 3.0,
            "vit_base":  8.0,
            "vit_large": 20.0,
            "uni":       20.0,
            "conch":     18.0,
            "gpfm":      22.0,
        }
        ms_per_patch = model_speed.get(model_name, 10.0)
        n_batches = (n_patches + batch_size - 1) // batch_size
        simulated_time = (n_patches * ms_per_patch / 1000)
        time.sleep(min(simulated_time * 0.01, 2.0))  # 加速 100x 模擬
        elapsed = simulated_time  # 回報估計時間
    else:
        # 真實推論
        model = _load_model(model_name, device)
        dummy_patches = torch.randn(min(batch_size, n_patches), 3, 224, 224).to(device)
        features = []
        n_batches = (n_patches + batch_size - 1) // batch_size
        with torch.no_grad():
            for i in range(n_batches):
                batch = dummy_patches[:min(batch_size, n_patches - i * batch_size)]
                feat = model(batch)
                features.append(feat.cpu())
        elapsed = time.perf_counter() - t0

    ms_per_patch = (elapsed / n_patches) * 1000 if n_patches > 0 else 0
    return StepTiming(
        name="feature_extraction",
        time_sec=elapsed,
        n_patches=n_patches,
        extra={
            "model": model_name,
            "batch_size": batch_size,
            "device": device,
            "ms_per_patch": round(ms_per_patch, 2)
        }
    )


def step4_aggregation(n_patches: int, method: str = "abmil") -> StepTiming:
    """特徵聚合（ABMIL / 線性）"""
    t0 = time.perf_counter()
    import torch
    # 模擬 features
    feat_dim = 768
    features = torch.randn(n_patches, feat_dim)
    if method == "linear":
        _ = features.mean(dim=0)
    elif method == "abmil":
        # 簡化注意力池化
        attn_weights = torch.softmax(torch.randn(n_patches), dim=0)
        _ = (features * attn_weights.unsqueeze(1)).sum(dim=0)
    elapsed = time.perf_counter() - t0
    return StepTiming(name="aggregation", time_sec=elapsed, extra={"method": method})


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def _otsu_threshold(gray_img: np.ndarray) -> float:
    """簡單 Otsu 閾值計算"""
    hist, bins = np.histogram(gray_img.flatten(), 256, [0, 256])
    hist = hist.astype(float) / hist.sum()
    best_thresh, best_var = 0, 0
    for t in range(1, 256):
        w0, w1 = hist[:t].sum(), hist[t:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0 = (hist[:t] * np.arange(t)).sum() / w0
        m1 = (hist[t:] * np.arange(t, 256)).sum() / w1
        var = w0 * w1 * (m0 - m1) ** 2
        if var > best_var:
            best_var, best_thresh = var, t
    return best_thresh


def _load_model(model_name: str, device: str):
    """載入模型（需安裝 timm）"""
    import torch.nn as nn
    try:
        import timm
        if model_name in ["resnet50"]:
            model = timm.create_model("resnet50", pretrained=False, num_classes=0)
        elif model_name in ["vit_small"]:
            model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
        elif model_name in ["vit_base"]:
            model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        elif model_name in ["vit_large"]:
            model = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0)
        else:
            print(f"    [警告] 未知模型 {model_name}，使用 ResNet-50 代替")
            model = timm.create_model("resnet50", pretrained=False, num_classes=0)
        return model.eval().to(device)
    except ImportError:
        # fallback: identity model
        import torch.nn as nn
        class IdentityModel(nn.Module):
            def forward(self, x):
                return x.mean(dim=[2, 3])
        return IdentityModel().to(device)


# ──────────────────────────────────────────────────────────
# Main benchmark logic
# ──────────────────────────────────────────────────────────

def benchmark_slide(slide_path: Optional[str],
                    model_name: str,
                    batch_size: int,
                    demo: bool = False) -> SlideTiming:
    """對單張 WSI 執行完整 benchmark"""
    slide_name = Path(slide_path).stem if slide_path else "DEMO_SLIDE"
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {slide_name}")
    print(f"{'='*60}")

    steps = []
    t_total = time.perf_counter()

    print("  [1/4] 組織偵測...")
    s1 = step1_tissue_detection(slide_path or "demo")
    steps.append(s1)
    print(f"        耗時: {s1.time_sec:.3f}s")

    print("  [2/4] Patch 萃取...")
    s2 = step2_patch_extraction(slide_path or "demo")
    steps.append(s2)
    print(f"        耗時: {s2.time_sec:.3f}s, patches: {s2.n_patches}")

    print("  [3/4] FM 特徵萃取...")
    s3 = step3_feature_extraction(
        n_patches=s2.n_patches if s2.n_patches > 0 else 3000,
        model_name=model_name,
        batch_size=batch_size,
        demo=demo
    )
    steps.append(s3)
    print(f"        耗時: {s3.time_sec:.3f}s ({s3.extra['ms_per_patch']}ms/patch)")

    print("  [4/4] 特徵聚合...")
    s4 = step4_aggregation(s3.n_patches)
    steps.append(s4)
    print(f"        耗時: {s4.time_sec:.3f}s")

    # 用各步驟時間加總（demo 模式中 feature_extraction 回傳估計時間）
    total = sum(s.time_sec for s in steps)
    return SlideTiming(slide_name=slide_name, steps=steps, total_time=total, n_patches=s2.n_patches)


def print_report(results: list):
    """印出 benchmark 報告"""
    print(f"\n{'='*60}")
    print("  BENCHMARK REPORT")
    print(f"{'='*60}")

    all_timings = {
        "tissue_detection": [],
        "patch_extraction": [],
        "feature_extraction": [],
        "aggregation": [],
    }

    for r in results:
        for s in r.steps:
            if s.name in all_timings:
                all_timings[s.name].append(s.time_sec)

    total_times = [r.total_time for r in results]

    print(f"\n  分析 {len(results)} 張 slides\n")
    print(f"  {'步驟':<25} {'平均時間':>10} {'佔比':>8}")
    print(f"  {'-'*45}")

    avg_total = np.mean(total_times)
    for step_name, times in all_timings.items():
        if times:
            avg = np.mean(times)
            pct = avg / avg_total * 100
            bar = "█" * int(pct / 5)
            print(f"  {step_name:<25} {avg:>9.2f}s {pct:>7.1f}%  {bar}")

    print(f"  {'-'*45}")
    print(f"  {'總計':<25} {avg_total:>9.2f}s {'100.0%':>8}")
    print(f"\n  ⚠️  主要瓶頸: {'feature_extraction' if all_timings['feature_extraction'] and np.mean(all_timings['feature_extraction']) > np.mean(all_timings['patch_extraction']) else 'patch_extraction'}")
    print(f"\n  推估 1000 張 slides 所需時間: {avg_total * 1000 / 3600:.1f} 小時")


def save_results(results: list, output_dir: str):
    """儲存結果到 JSON"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "benchmark_results.json")
    data = []
    for r in results:
        d = {
            "slide_name": r.slide_name,
            "total_time": r.total_time,
            "n_patches": r.n_patches,
            "steps": [asdict(s) for s in r.steps]
        }
        data.append(d)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n  結果已儲存至: {output_path}")


# ──────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WSI Pipeline Benchmark")
    parser.add_argument("--wsi_dir", type=str, default=None,
                        help="WSI 目錄路徑（.svs, .ndpi, .tiff）")
    parser.add_argument("--model", type=str, default="vit_base",
                        choices=["resnet50", "vit_small", "vit_base", "vit_large", "uni", "conch", "demo"],
                        help="使用的特徵萃取模型")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="推論 batch size")
    parser.add_argument("--output", type=str, default="outputs/",
                        help="結果輸出目錄")
    parser.add_argument("--demo", action="store_true",
                        help="使用合成資料模擬（不需要真實 WSI）")
    parser.add_argument("--n_demo_slides", type=int, default=5,
                        help="Demo 模式下模擬的 slide 數量")
    parser.add_argument("--compare_models", action="store_true",
                        help="比較不同模型的推論速度")
    args = parser.parse_args()

    results = []

    if args.demo or args.wsi_dir is None:
        print("\n[Demo 模式] 使用合成資料模擬 WSI pipeline...")
        print(f"模擬 {args.n_demo_slides} 張 slides，模型: {args.model}\n")
        for i in range(args.n_demo_slides):
            r = benchmark_slide(None, args.model, args.batch_size, demo=True)
            results.append(r)
    else:
        wsi_dir = Path(args.wsi_dir)
        slide_files = list(wsi_dir.glob("*.svs")) + \
                      list(wsi_dir.glob("*.ndpi")) + \
                      list(wsi_dir.glob("*.tiff")) + \
                      list(wsi_dir.glob("*.tif"))
        if not slide_files:
            print(f"[錯誤] 在 {wsi_dir} 未找到 WSI 檔案，改用 demo 模式")
            for i in range(args.n_demo_slides):
                r = benchmark_slide(None, args.model, args.batch_size, demo=True)
                results.append(r)
        else:
            for slide_path in slide_files:
                r = benchmark_slide(str(slide_path), args.model, args.batch_size)
                results.append(r)

    if args.compare_models:
        print("\n\n[模型速度比較] 使用 demo 模式...")
        compare_results = {}
        models = ["resnet50", "vit_small", "vit_base", "vit_large"]
        n_patches = 3000
        for model in models:
            t0 = time.perf_counter()
            s = step3_feature_extraction(n_patches, model, args.batch_size, demo=True)
            compare_results[model] = {
                "time_sec": s.time_sec,
                "ms_per_patch": s.extra["ms_per_patch"],
            }
        print(f"\n  {'模型':<15} {'總時間(s)':>12} {'每patch(ms)':>12}")
        print(f"  {'-'*40}")
        for model, v in compare_results.items():
            print(f"  {model:<15} {v['time_sec']:>12.2f} {v['ms_per_patch']:>12.2f}")

    print_report(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
