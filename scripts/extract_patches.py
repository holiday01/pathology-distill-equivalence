#!/usr/bin/env python3
"""
WSI → 前景 patches → HDF5

讀 pyramidal TIFF (CAMELYON16 格式) via tifffile，用 Otsu 在縮圖上算組織 mask，
然後在 20× 層級抽 224×224 前景 patches，存成 HDF5 供蒸餾訓練用。

用法：
  python extract_patches.py \
    --wsi_dir ~/wsi_hl/data/camelyon16 \
    --out ~/wsi_hl/data/patches_camelyon16.h5 \
    --patches_per_slide 500 \
    --patch_size 224 \
    --target_level 1
"""

import argparse
import random
import re
from pathlib import Path

import h5py
import numpy as np
import tifffile
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu


def build_tissue_mask(thumb_rgb: np.ndarray, min_luminance: float = 0.05):
    """縮圖 → 組織 mask（True = 組織）。"""
    gray = rgb2gray(thumb_rgb)
    try:
        thr = threshold_otsu(gray)
    except ValueError:
        thr = 0.8
    tissue = (gray < thr) & (gray > min_luminance)
    return tissue


def pick_thumb_level(levels_shapes, target_max_side: int = 4096):
    """挑一個縮圖層級用於組織偵測（長邊 <= target）。"""
    for i, shape in enumerate(levels_shapes):
        h, w = shape[:2]
        if max(h, w) <= target_max_side:
            return i
    return len(levels_shapes) - 1


def slide_label(path: Path) -> int:
    """normal_* → 0, tumor_* → 1, 其他 → 依字母順序分配"""
    name = path.stem.lower()
    if name.startswith("normal"):
        return 0
    if name.startswith("tumor"):
        return 1
    return 2


def extract_from_slide(
    wsi_path: Path,
    patches_per_slide: int,
    patch_size: int,
    target_level: int,
    tissue_ratio: float,
    rng: random.Random,
):
    """回傳 (patches ndarray [N, H, W, 3] uint8, coords [N, 2])。"""
    with tifffile.TiffFile(str(wsi_path)) as tf:
        series = tf.series[0]
        n_levels = len(series.levels)
        tgt = min(target_level, n_levels - 1)

        level_shapes = [lvl.shape for lvl in series.levels]
        thumb_idx = pick_thumb_level(level_shapes, target_max_side=4096)
        thumb_idx = max(thumb_idx, tgt + 1)  # 確保縮圖比 target 小
        thumb_idx = min(thumb_idx, n_levels - 1)

        thumb = series.levels[thumb_idx].asarray()  # (H, W, 3)
        tissue = build_tissue_mask(thumb)

        tgt_h, tgt_w = level_shapes[tgt][:2]
        thumb_h, thumb_w = thumb.shape[:2]
        scale_y = thumb_h / tgt_h
        scale_x = thumb_w / tgt_w

        # 用 zarr 接口做 lazy slicing，避免整張讀進 RAM
        z = series.levels[tgt].aszarr()
        import zarr
        arr = zarr.open(z, mode="r")

        nx = tgt_w // patch_size
        ny = tgt_h // patch_size
        coords_candidates = []
        tps = patch_size
        for iy in range(ny):
            y0 = iy * tps
            ty0 = int(y0 * scale_y)
            ty1 = int((y0 + tps) * scale_y)
            for ix in range(nx):
                x0 = ix * tps
                tx0 = int(x0 * scale_x)
                tx1 = int((x0 + tps) * scale_x)
                sub = tissue[ty0:ty1, tx0:tx1]
                if sub.size == 0:
                    continue
                if sub.mean() >= tissue_ratio:
                    coords_candidates.append((y0, x0))

        if not coords_candidates:
            return np.empty((0, patch_size, patch_size, 3), dtype=np.uint8), \
                   np.empty((0, 2), dtype=np.int64)

        rng.shuffle(coords_candidates)
        chosen = coords_candidates[:patches_per_slide]

        patches = np.empty((len(chosen), patch_size, patch_size, 3), dtype=np.uint8)
        coords_arr = np.asarray(chosen, dtype=np.int64)
        for i, (y0, x0) in enumerate(chosen):
            tile = arr[y0:y0 + tps, x0:x0 + tps]
            if tile.shape != (tps, tps, 3):
                # 邊界或 RGBA 情況處理
                if tile.ndim == 3 and tile.shape[-1] == 4:
                    tile = tile[..., :3]
                if tile.shape[0] != tps or tile.shape[1] != tps:
                    pad_h = tps - tile.shape[0]
                    pad_w = tps - tile.shape[1]
                    tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)),
                                  mode="constant", constant_values=255)
            patches[i] = tile

        return patches, coords_arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi_dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--patches_per_slide", type=int, default=500)
    ap.add_argument("--patch_size", type=int, default=224)
    ap.add_argument("--target_level", type=int, default=1,
                    help="pyramid level for patch extraction (0=40×, 1=20×, 2=10×)")
    ap.add_argument("--tissue_ratio", type=float, default=0.5,
                    help="a tile must contain at least this fraction of tissue")
    ap.add_argument("--pattern", type=str, default="*.tif",
                    help="glob pattern for WSI files")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    wsi_dir = Path(args.wsi_dir).expanduser()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wsis = sorted(wsi_dir.glob(args.pattern))
    if not wsis:
        raise SystemExit(f"找不到 WSI: {wsi_dir}/{args.pattern}")

    print(f"找到 {len(wsis)} 張 WSI")
    print(f"目標層級 level{args.target_level}, patch {args.patch_size}x{args.patch_size}, "
          f"每張最多 {args.patches_per_slide} patches")

    total_est = len(wsis) * args.patches_per_slide
    with h5py.File(out_path, "w") as h5:
        patches_ds = h5.create_dataset(
            "patches",
            shape=(0, args.patch_size, args.patch_size, 3),
            maxshape=(None, args.patch_size, args.patch_size, 3),
            dtype="uint8",
            chunks=(1, args.patch_size, args.patch_size, 3),
            compression="gzip",
            compression_opts=4,
        )
        labels_ds = h5.create_dataset(
            "labels", shape=(0,), maxshape=(None,), dtype="int64",
        )
        slide_ids_ds = h5.create_dataset(
            "slide_ids", shape=(0,), maxshape=(None,), dtype="int64",
        )
        coords_ds = h5.create_dataset(
            "coords", shape=(0, 2), maxshape=(None, 2), dtype="int64",
        )

        slide_names = []
        n_written = 0
        for slide_idx, wsi in enumerate(wsis):
            label = slide_label(wsi)
            print(f"[{slide_idx+1}/{len(wsis)}] {wsi.name} (label={label})")
            try:
                patches, coords = extract_from_slide(
                    wsi,
                    args.patches_per_slide,
                    args.patch_size,
                    args.target_level,
                    args.tissue_ratio,
                    rng,
                )
            except Exception as e:
                print(f"  [跳過] {e}")
                continue

            n = len(patches)
            print(f"  抽到 {n} patches")
            if n == 0:
                continue

            new_size = n_written + n
            for ds in (patches_ds, labels_ds, slide_ids_ds, coords_ds):
                if ds is patches_ds:
                    ds.resize((new_size, args.patch_size, args.patch_size, 3))
                elif ds is coords_ds:
                    ds.resize((new_size, 2))
                else:
                    ds.resize((new_size,))

            patches_ds[n_written:new_size] = patches
            labels_ds[n_written:new_size] = label
            slide_ids_ds[n_written:new_size] = slide_idx
            coords_ds[n_written:new_size] = coords
            slide_names.append(wsi.name)
            n_written = new_size

        h5.attrs["patch_size"] = args.patch_size
        h5.attrs["target_level"] = args.target_level
        h5.attrs["slide_names"] = np.array(slide_names, dtype="S")
        h5.attrs["source_dir"] = str(wsi_dir)

    print(f"\n寫入 {out_path}，共 {n_written} patches")


if __name__ == "__main__":
    main()
