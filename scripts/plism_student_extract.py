#!/usr/bin/env python3
"""Extract PLISM features for student models and run plismbench evaluate.

Uses the same output format as plismbench extract (features.npy per slide),
so plismbench evaluate can read them directly.

Output structure:
    outputs/plism/features/{teacher}/{student}/SLIDE_ID.tif/features.npy

where features.npy is (16278, 3+embed_dim) float32 with columns:
    [coord_x, coord_y, coord_z, feat_0, ..., feat_{embed_dim-1}]
sorted by (coord_x, coord_y).
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from distill_wsi_model import StudentModel

NUM_SLIDES = 91
NUM_TILES_PER_SLIDE = 16_278

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

STUDENT_ARCH = {
    "vit-tiny":  "vit_tiny_patch16_224",
    "vit-small": "vit_small_patch16_224",
    "vit-base":  "vit_base_patch16_224",
}


class PlismH5Dataset(Dataset):
    """Wraps a PLISM .tif.h5 slide for student feature extraction."""

    def __init__(self, h5_path: Path, transform):
        self.h5_path = h5_path
        self.transform = transform
        with h5py.File(h5_path, "r") as f:
            self.tile_ids = sorted(f.keys())
        assert len(self.tile_ids) == NUM_TILES_PER_SLIDE, \
            f"{h5_path.name}: expected {NUM_TILES_PER_SLIDE} tiles, got {len(self.tile_ids)}"

    def __len__(self):
        return len(self.tile_ids)

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r", libver="latest", swmr=True) as f:
            img = f[self.tile_ids[idx]][:]  # (H, W, 3) uint8
        return self.tile_ids[idx], self.transform(img)


def build_transform():
    return transforms.Compose([
        transforms.ToTensor(),  # (H, W, 3) → (3, H, W) float32 in [0,1]
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def sort_by_coords(arr: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(arr[:, 1:3], columns=["x", "y"])
    df.sort_values(["x", "y"], inplace=True)
    return arr[df.index.values]


def extract_slide(model, h5_path: Path, transform, device, batch_size=64, workers=4):
    """Extract features for all tiles in one PLISM slide H5."""
    ds = PlismH5Dataset(h5_path, transform)

    def collate(batch):
        tile_ids = [b[0] for b in batch]
        imgs = torch.stack([b[1] for b in batch])
        return tile_ids, imgs

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True, collate_fn=collate)
    all_rows = []
    with torch.no_grad():
        for tile_ids, imgs in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            feats = out["feat"] if isinstance(out, dict) else out
            feats = F.normalize(feats, dim=-1).cpu().numpy()
            coords = np.array(
                [tid.split("_")[1:] for tid in tile_ids], dtype=np.float32
            )
            all_rows.append(np.concatenate([coords, feats], axis=1))
    arr = np.concatenate(all_rows, axis=0).astype(np.float32)
    arr = sort_by_coords(arr)
    assert arr.shape[0] == NUM_TILES_PER_SLIDE, \
        f"Got {arr.shape[0]} tiles, expected {NUM_TILES_PER_SLIDE}"
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plism-dir", default="/path/to/wsi_datasets/plism")
    ap.add_argument("--out-root",  default="outputs/plism/features")
    ap.add_argument("--v4-root",   default="outputs/v4_full")
    ap.add_argument("--teacher",   default=None, help="restrict to one teacher")
    ap.add_argument("--student",   default=None, help="restrict to one student")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers",    type=int, default=4)
    ap.add_argument("--overwrite",  action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[plism-student] device={device}")

    transform = build_transform()
    plism_dir = Path(args.plism_dir)
    h5_files = sorted(plism_dir.glob("*.tif.h5"))
    assert len(h5_files) == NUM_SLIDES, \
        f"Expected {NUM_SLIDES} H5 files, found {len(h5_files)}"
    print(f"[plism-student] {len(h5_files)} PLISM slides")

    v4_root = Path(args.v4_root)
    out_root = Path(args.out_root)

    for fm_dir in sorted(v4_root.iterdir()):
        if not fm_dir.is_dir() or fm_dir.name in ("telemetry", "figures"):
            continue
        teacher_name = fm_dir.name
        if args.teacher and teacher_name != args.teacher:
            continue

        for stu_dir in sorted(fm_dir.iterdir()):
            if not stu_dir.is_dir():
                continue
            stu_short = stu_dir.name
            if args.student and stu_short != args.student:
                continue
            ckpt_path = stu_dir / "best.pt"
            if not ckpt_path.exists():
                continue

            out_dir = out_root / teacher_name / stu_short
            done_marker = out_dir / ".done"
            if done_marker.exists() and not args.overwrite:
                print(f"[skip] {teacher_name}/{stu_short} (already done)")
                continue

            print(f"\n[extract] {teacher_name}/{stu_short}")
            stu_arch = STUDENT_ARCH.get(stu_short, "vit_small_patch16_224")

            try:
                student = StudentModel(stu_arch, embed_dim=256, n_classes=0,
                                       pretrained=False).to(device)
                ckpt = torch.load(ckpt_path, map_location=device)
                student.load_state_dict(ckpt.get("student", ckpt), strict=False)
                student.eval()
            except Exception as e:
                print(f"  [skip] {teacher_name}/{stu_short}: checkpoint load failed: {e}")
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            errors = 0
            for h5_path in tqdm(h5_files, desc=f"{teacher_name}/{stu_short}"):
                slide_id = h5_path.stem  # e.g. "GIVH_AT2_to_GMH_S60.tif"
                slide_out = out_dir / slide_id
                feat_file = slide_out / "features.npy"
                if feat_file.exists() and not args.overwrite:
                    continue
                slide_out.mkdir(exist_ok=True)
                try:
                    arr = extract_slide(student, h5_path, transform, device,
                                        args.batch_size, args.workers)
                    np.save(str(feat_file), arr)
                except Exception as e:
                    print(f"  [warn] {slide_id}: {e}")
                    errors += 1

            done_marker.touch()
            print(f"  [done] {teacher_name}/{stu_short}: {errors} errors")

            del student
            torch.cuda.empty_cache()

    print("\n[plism-student] extraction complete")


if __name__ == "__main__":
    main()
