#!/usr/bin/env python3
"""PanNuke segmentation linear probe — paper-ready eval addon.

Protocol (standard PFM segmentation eval, c.f. HEST / UNI / GPFM papers):
  1. For each image (256x256), forward through frozen FM (teacher or student)
     to get patch tokens at 14x14 spatial resolution (ViT patch=16, input 224).
  2. Aggregate ground-truth mask to 14x14 by majority class per 16x16 pixel
     block; this gives N×196 supervision samples per image.
  3. Train a linear (LogReg) probe D→6 classes on the training fold's
     patch tokens; predict on test fold.
  4. Upsample 14x14 prediction to 256x256 via nearest-neighbor; compute
     pixel-wise Dice per class + mean-Dice (excluding background).
  5. Report teacher/student per-class Dice, ΔDice, and TOST equivalence
     using slide-cluster bootstrap on per-image Dice (image=cluster unit).

PanNuke channel order (mask.shape = N,256,256,6):
  0=Neoplastic, 1=Inflammatory, 2=Connective, 3=Dead,
  4=Epithelial(non-neoplastic), 5=Background.
Each channel holds an instance-id map (0=absent, >=1=instance). We binarize.

Usage:
  python scripts/evaluate_pannuke_seg.py \
    --teacher virchow2 --student vit_small_patch16_224 \
    --student_ckpt outputs/v4_full/virchow2/vit-small/best.pt \
    --pannuke_root data/external/pannuke \
    --output outputs/v4_full/virchow2/vit-small/pannuke_seg \
    --train_folds 1,2 --test_fold 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from distill_wsi_model import StudentModel, TeacherModel
from stats_v4 import slide_cluster_bootstrap, tost_equivalence


CLASS_NAMES = ["neoplastic", "inflammatory", "connective", "dead",
               "epithelial", "background"]


class PanNukeFold(Dataset):
    """Streams images + per-patch labels from a PanNuke fold."""

    def __init__(self, fold_path: Path, max_images: int | None = None,
                 patch_grid: int = 14):
        fp = Path(fold_path)
        # PanNuke nesting: fold_X_unz/Fold X/images/foldX/images.npy
        img_p = list(fp.glob("**/images.npy"))
        msk_p = list(fp.glob("**/masks.npy"))
        if not img_p or not msk_p:
            raise FileNotFoundError(f"images/masks.npy not found in {fp}")
        self.images = np.load(img_p[0], mmap_mode="r")  # (N,256,256,3) float64
        self.masks = np.load(msk_p[0], mmap_mode="r")   # (N,256,256,6) float64
        self.N = self.images.shape[0]
        if max_images is not None:
            self.N = min(self.N, max_images)
        self.patch_grid = patch_grid

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        img = self.images[i].astype(np.float32) / 255.0  # 0-1 by convention
        # Some PanNuke distros are 0-255 floats; check if max > 1 then divide
        if img.max() > 1.5:
            img = img / 255.0
        # mask: take argmax over channels to get class id per pixel
        msk = self.masks[i]
        # mask channel j has instance ids; foreground iff > 0
        binary = (msk > 0).astype(np.float32)  # (256,256,6)
        # For multi-class supervision: priority neoplastic > inflam > ... > epith
        # background = 1 if no foreground class, else 0
        cls_map = np.full((256, 256), 5, dtype=np.int64)  # default = background
        for c in range(5):  # 0..4 in priority order
            cls_map[(binary[..., c] > 0)] = c
        # Aggregate to patch grid via majority vote per 16×16 block
        g = self.patch_grid
        bs = 256 // g
        cls_grid = np.zeros((g, g), dtype=np.int64)
        for hh in range(g):
            for ww in range(g):
                block = cls_map[hh*bs:(hh+1)*bs, ww*bs:(ww+1)*bs]
                vals, counts = np.unique(block, return_counts=True)
                cls_grid[hh, ww] = int(vals[counts.argmax()])
        # Resize image to 224x224 to match ViT input
        img_t = torch.from_numpy(img).permute(2, 0, 1)  # 3,256,256
        img_t = F.interpolate(img_t.unsqueeze(0), size=224, mode="bilinear",
                               align_corners=False).squeeze(0)
        # ImageNet normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = (img_t - mean) / std
        return img_t, torch.from_numpy(cls_grid), torch.from_numpy(cls_map), i


@torch.no_grad()
def extract_patch_tokens(model, loader, device, is_teacher=False):
    """Returns tokens shape (N, P, D), patch-grid labels (N, P), pixel-mask (N,256,256), idx."""
    model.eval()
    tokens, labels, pixel_masks, indices = [], [], [], []
    for x, lab, pm, ii in loader:
        x = x.to(device, non_blocking=True)
        out = model(x, return_patches=True)
        if isinstance(out, dict):
            t = out["patch_feats"]  # (B, N_patches, D)
        else:
            t = out
        tokens.append(t.cpu())
        labels.append(lab)
        pixel_masks.append(pm)
        indices.append(ii)
    return (torch.cat(tokens), torch.cat(labels), torch.cat(pixel_masks),
            torch.cat(indices))


def linear_seg_probe(tr_tok, tr_lab, te_tok, te_lab, n_classes=6,
                     epochs=50, lr=1e-3, wd=1e-4, device="cuda"):
    """Linear D→K probe at patch-token level.  Returns predicted class per
    patch on test set, shape (N_te, P)."""
    D = tr_tok.shape[-1]
    clf = torch.nn.Linear(D, n_classes).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=wd)
    # Flatten (N, P, D) → (N*P, D)
    Xtr = tr_tok.reshape(-1, D).to(device)
    ytr = tr_lab.reshape(-1).to(device)
    for _ in range(epochs):
        logits = clf(Xtr)
        loss = F.cross_entropy(logits, ytr)
        opt.zero_grad(); loss.backward(); opt.step()
    Xte = te_tok.reshape(-1, D).to(device)
    with torch.no_grad():
        preds = clf(Xte).argmax(-1)
    return preds.view(te_tok.shape[0], te_tok.shape[1]).cpu().numpy()


def upsample_grid_to_pixel(grid_cls, target=256):
    """grid_cls: (g, g) int → (target, target) int via nearest neighbor."""
    g = grid_cls.shape[0]
    rep = target // g
    return np.repeat(np.repeat(grid_cls, rep, axis=0), rep, axis=1)


def per_image_dice(pred_pix, true_pix, n_classes=6, eps=1e-6):
    """Pixel-wise Dice per class for one image.  Returns array (n_classes,)."""
    out = np.zeros(n_classes, dtype=float)
    for c in range(n_classes):
        p = (pred_pix == c)
        t = (true_pix == c)
        inter = (p & t).sum()
        denom = p.sum() + t.sum()
        out[c] = (2 * inter) / (denom + eps) if denom > 0 else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", default="vit_small_patch16_224")
    ap.add_argument("--student_ckpt", required=True)
    ap.add_argument("--pannuke_root", default="data/external/pannuke")
    ap.add_argument("--train_folds", default="1,2")
    ap.add_argument("--test_fold", default="3")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_images_per_fold", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    print(f"[teacher] {args.teacher}")
    teacher = TeacherModel(args.teacher).eval().to(device)
    print(f"[student] {args.student} from {args.student_ckpt}")
    student = StudentModel(args.student, embed_dim=256, n_classes=0,
                            pretrained=False).to(device)
    ck = torch.load(args.student_ckpt, map_location=device, weights_only=False)
    student.load_state_dict(ck.get("student", ck), strict=False)
    student.eval()

    pn_root = Path(args.pannuke_root)
    train_folds = [int(x) for x in args.train_folds.split(",")]
    test_fold = int(args.test_fold)
    print(f"[folds] train={train_folds}  test={test_fold}")

    def make_loader(fold_n):
        ds = PanNukeFold(pn_root / f"fold_{fold_n}_unz",
                         max_images=args.max_images_per_fold)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True), ds

    tr_loaders = []
    for f in train_folds:
        ld, _ = make_loader(f)
        tr_loaders.append(ld)
    te_loader, te_ds = make_loader(test_fold)

    print("[extract] teacher train"); t_tr_tok, tr_lab, _, _ = (
        torch.cat([x[0] for x in [extract_patch_tokens(teacher, ld, device, True)
                                   for ld in tr_loaders]]),
        torch.cat([x[1] for x in [extract_patch_tokens(teacher, ld, device, True)
                                   for ld in tr_loaders]]),
        None, None,
    )
    # Simpler: re-extract (single pass) so labels align
    t_tr_tok_list, lab_tr_list = [], []
    for ld in tr_loaders:
        tok, lab, _, _ = extract_patch_tokens(teacher, ld, device, True)
        t_tr_tok_list.append(tok); lab_tr_list.append(lab)
    t_tr_tok = torch.cat(t_tr_tok_list); tr_lab = torch.cat(lab_tr_list)
    print(f"  teacher train tokens: {tuple(t_tr_tok.shape)}  labels: {tuple(tr_lab.shape)}")

    print("[extract] teacher test")
    t_te_tok, te_lab, te_pix, te_idx = extract_patch_tokens(teacher, te_loader, device, True)
    print("[extract] student train")
    s_tr_tok_list = []
    for ld in tr_loaders:
        tok, _, _, _ = extract_patch_tokens(student, ld, device, False)
        s_tr_tok_list.append(tok)
    s_tr_tok = torch.cat(s_tr_tok_list)
    print("[extract] student test")
    s_te_tok, _, _, _ = extract_patch_tokens(student, te_loader, device, False)

    print("[probe] teacher")
    t_pred_grid = linear_seg_probe(t_tr_tok, tr_lab, t_te_tok, te_lab,
                                   n_classes=6, device=device)
    print("[probe] student")
    s_pred_grid = linear_seg_probe(s_tr_tok, tr_lab, s_te_tok, te_lab,
                                   n_classes=6, device=device)

    # Per-image Dice
    print("[dice] per-image")
    t_dice = np.zeros((len(te_pix), 6))
    s_dice = np.zeros((len(te_pix), 6))
    image_ids = np.array([f"pannuke:fold{test_fold}:img{i}" for i in range(len(te_pix))])
    for i in range(len(te_pix)):
        true_pix = te_pix[i].numpy()
        t_pix = upsample_grid_to_pixel(t_pred_grid[i])
        s_pix = upsample_grid_to_pixel(s_pred_grid[i])
        t_dice[i] = per_image_dice(t_pix, true_pix)
        s_dice[i] = per_image_dice(s_pix, true_pix)

    # Per-class summary (excluding background for mDice)
    summary = {"per_class": {}}
    for c, name in enumerate(CLASS_NAMES):
        td = t_dice[:, c]; sd = s_dice[:, c]
        td_mean = float(np.nanmean(td)); sd_mean = float(np.nanmean(sd))
        # Slide-cluster bootstrap (image = cluster) for student-teacher Δ
        boot = slide_cluster_bootstrap(sd - td, image_ids, np.nanmean,
                                        n_boot=args.n_boot, seed=args.seed)
        tost = tost_equivalence(td, sd, image_ids,
                                margin=0.05, n_boot=args.n_boot, seed=args.seed)
        summary["per_class"][name] = {
            "teacher_dice": td_mean, "student_dice": sd_mean,
            "delta_mean": float(np.nanmean(sd - td)),
            "delta_ci90": [boot["lo"], boot["hi"]],
            "tost_equivalent": bool(tost["equivalent"]),
            "p_lower": tost["p_lower"], "p_upper": tost["p_upper"],
        }
        print(f"  {name:14s} T={td_mean:.4f}  S={sd_mean:.4f}  "
              f"Δ={summary['per_class'][name]['delta_mean']:+.4f}  "
              f"equiv={summary['per_class'][name]['tost_equivalent']}")
    # mDice excluding background (channels 0..4)
    fg_t = float(np.nanmean(t_dice[:, :5]))
    fg_s = float(np.nanmean(s_dice[:, :5]))
    summary["mDice_fg"] = {"teacher": fg_t, "student": fg_s, "delta": fg_s - fg_t}
    print(f"  mDice(fg) T={fg_t:.4f}  S={fg_s:.4f}  Δ={fg_s-fg_t:+.4f}")

    summary["args"] = vars(args)
    summary["n_train_images"] = int(t_tr_tok.shape[0])
    summary["n_test_images"] = int(t_te_tok.shape[0])

    np.save(out / "teacher_dice_per_image.npy", t_dice)
    np.save(out / "student_dice_per_image.npy", s_dice)
    with open(out / "pannuke_seg_report.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[saved] {out / 'pannuke_seg_report.json'}")


if __name__ == "__main__":
    main()
