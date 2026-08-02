#!/usr/bin/env python3
"""CAMELYON16 patch extraction with LESION-ANNOTATION tile labels.

The filename-based labeling (`tumor_*.tif` -> every tile labelled tumor) is
wrong at the tile level: a tumor slide is mostly normal tissue, so most of
its tiles are normal but get labelled tumor. That label noise made the
tile-level tumor/normal probe near-random (AUC ~0.57) and the equivalence
test degenerate. Here a tile is labelled tumor (1) iff its center falls
inside an annotated tumor polygon (ASAP XML, level-0 coords), else normal
(0). Normal slides have no annotation -> all tiles are 0 (correct).

Per tumor slide we balance-sample up to --tumor_cap tumor + --tumor_cap
normal tiles; normal slides contribute --normal_per_slide tiles.

Usage:
  python3 scripts/extract_c16_annotated.py \
      --wsi_dir /path/to/wsi_datasets/camelyon16 \
      --ann_dir data/c16_annotations \
      --out /path/to/cache/patches/patches_c16_annotated.h5
"""
import argparse, random, sys
from pathlib import Path
import xml.etree.ElementTree as ET
import h5py, numpy as np, tifffile
from matplotlib.path import Path as MplPath

sys.path.insert(0, str(Path(__file__).parent))
from extract_patches import build_tissue_mask, pick_thumb_level, slide_label


def parse_polygons(xml_path):
    """ASAP XML -> (tumor_polys, exclude_polys) as matplotlib Paths in
    level-0 pixel coords. CAMELYON16 groups: 'Tumor'/_0/_1 = metastasis,
    'Exclusion'/_2 = benign region carved out of a tumor polygon."""
    root = ET.parse(str(xml_path)).getroot()
    tumor, exclude = [], []
    for ann in root.iter("Annotation"):
        grp = (ann.get("PartOfGroup") or "").strip().lower()
        pts = [(float(c.get("X")), float(c.get("Y"))) for c in ann.iter("Coordinate")]
        if len(pts) < 3:
            continue
        poly = MplPath(np.asarray(pts))
        if grp in ("exclusion", "_2", "none"):
            exclude.append(poly)
        else:                       # tumor / _0 / _1 / metastases / default
            tumor.append(poly)
    return tumor, exclude


def tile_is_tumor(cx, cy, tumor, exclude):
    if not any(p.contains_point((cx, cy)) for p in tumor):
        return False
    if any(p.contains_point((cx, cy)) for p in exclude):
        return False
    return True


def candidate_tiles(wsi_path, patch_size, target_level, tissue_ratio):
    """Return (arr_zarr, chosen_coords[(y0,x0)..] tgt-level, scale0_y, scale0_x).
    Mirrors extract_patches.extract_from_slide tissue selection."""
    tf = tifffile.TiffFile(str(wsi_path))
    series = tf.series[0]
    n_levels = len(series.levels)
    tgt = min(target_level, n_levels - 1)
    level_shapes = [lvl.shape for lvl in series.levels]
    thumb_idx = pick_thumb_level(level_shapes, 4096)
    thumb_idx = min(max(thumb_idx, tgt + 1), n_levels - 1)
    thumb = series.levels[thumb_idx].asarray()
    tissue = build_tissue_mask(thumb)
    tgt_h, tgt_w = level_shapes[tgt][:2]
    l0_h, l0_w = level_shapes[0][:2]
    thumb_h, thumb_w = thumb.shape[:2]
    sy, sx = thumb_h / tgt_h, thumb_w / tgt_w          # tgt -> thumb
    s0y, s0x = l0_h / tgt_h, l0_w / tgt_w               # tgt -> level0
    import zarr
    arr = zarr.open(series.levels[tgt].aszarr(), mode="r")
    ps = patch_size
    coords = []
    for iy in range(tgt_h // ps):
        y0 = iy * ps
        ty0, ty1 = int(y0 * sy), int((y0 + ps) * sy)
        for ix in range(tgt_w // ps):
            x0 = ix * ps
            tx0, tx1 = int(x0 * sx), int((x0 + ps) * sx)
            sub = tissue[ty0:ty1, tx0:tx1]
            if sub.size and sub.mean() >= tissue_ratio:
                coords.append((y0, x0))
    return tf, arr, coords, s0y, s0x


def read_tile(arr, y0, x0, ps):
    tile = arr[y0:y0 + ps, x0:x0 + ps]
    if tile.ndim == 3 and tile.shape[-1] == 4:
        tile = tile[..., :3]
    if tile.shape != (ps, ps, 3):
        return None
    return np.asarray(tile, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi_dir", required=True)
    ap.add_argument("--ann_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--patch_size", type=int, default=224)
    ap.add_argument("--target_level", type=int, default=1)
    ap.add_argument("--tissue_ratio", type=float, default=0.5)
    ap.add_argument("--tumor_cap", type=int, default=250,
                    help="per tumor slide: up to this many tumor + this many normal tiles")
    ap.add_argument("--normal_per_slide", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    ps = args.patch_size
    wsis = sorted(Path(args.wsi_dir).glob("*.tif"))
    ann_dir = Path(args.ann_dir)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out, "w") as h5:
        P = h5.create_dataset("patches", (0, ps, ps, 3), maxshape=(None, ps, ps, 3),
                              dtype="uint8", chunks=(1, ps, ps, 3),
                              compression="gzip", compression_opts=4)
        L = h5.create_dataset("labels", (0,), maxshape=(None,), dtype="int64")
        S = h5.create_dataset("slide_ids", (0,), maxshape=(None,), dtype="int64")
        n_w = 0
        for sidx, wsi in enumerate(wsis):
            is_tumor_slide = wsi.stem.lower().startswith("tumor")
            try:
                tf, arr, coords, s0y, s0x = candidate_tiles(
                    wsi, ps, args.target_level, args.tissue_ratio)
            except Exception as e:
                print(f"[skip] {wsi.name}: {e}", flush=True); continue
            rng.shuffle(coords)
            tumor_polys = exclude_polys = []
            if is_tumor_slide:
                axml = ann_dir / f"{wsi.stem}.xml"
                if axml.exists():
                    tumor_polys, exclude_polys = parse_polygons(axml)
                else:
                    print(f"[warn] no XML for {wsi.name}, skipping its tiles", flush=True)
                    tf.close(); continue

            chosen = []  # (y0,x0,label)
            n_t = n_n = 0
            for (y0, x0) in coords:
                if is_tumor_slide:
                    cx = (x0 + ps / 2) * s0x
                    cy = (y0 + ps / 2) * s0y
                    lab = 1 if tile_is_tumor(cx, cy, tumor_polys, exclude_polys) else 0
                    if lab == 1 and n_t >= args.tumor_cap: continue
                    if lab == 0 and n_n >= args.tumor_cap: continue
                    if lab == 1: n_t += 1
                    else: n_n += 1
                else:
                    lab = 0
                    if n_n >= args.normal_per_slide: continue
                    n_n += 1
                chosen.append((y0, x0, lab))
                if is_tumor_slide and n_t >= args.tumor_cap and n_n >= args.tumor_cap:
                    break
                if (not is_tumor_slide) and n_n >= args.normal_per_slide:
                    break

            buf, labs = [], []
            for (y0, x0, lab) in chosen:
                t = read_tile(arr, y0, x0, ps)
                if t is not None:
                    buf.append(t); labs.append(lab)
            tf.close()
            if not buf:
                print(f"[{sidx+1}/{len(wsis)}] {wsi.name}: 0 tiles", flush=True); continue
            n = len(buf)
            P.resize(n_w + n, axis=0); P[n_w:n_w+n] = np.stack(buf)
            L.resize(n_w + n, axis=0); L[n_w:n_w+n] = np.asarray(labs)
            S.resize(n_w + n, axis=0); S[n_w:n_w+n] = sidx
            n_w += n
            print(f"[{sidx+1}/{len(wsis)}] {wsi.name}: {n} tiles "
                  f"(tumor={sum(labs)} normal={n-sum(labs)})", flush=True)
        print(f"\n[done] {n_w} tiles -> {out}", flush=True)
        # report overall balance
        print(f"total tumor tiles: {int(np.sum(L[:]))} / {n_w}", flush=True)


if __name__ == "__main__":
    main()
