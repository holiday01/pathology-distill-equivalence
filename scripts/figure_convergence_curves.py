#!/usr/bin/env python3
"""Supplementary figure D-1: per-teacher convergence curves for the
v4_full atlas. Pure CPU; reads only training_log.json files (no eval
required). Produces a 12-panel grid (one panel per teacher), each panel
overlays val loss curves for ViT-Ti / ViT-S / ViT-B.

Within-teacher convergence is the meaningful comparison (Ti vs S vs B
on the same teacher, same loss space). Cross-teacher loss is NOT
comparable (each teacher embeds in its own feature space) and the
script makes no attempt to normalise across teachers.

Output: paper/figures/figS_D1_convergence_curves.{pdf,png}
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("outputs/v4_full")
OUT = Path("paper/figures/figS_D1_convergence_curves")

TEACHER_ORDER = [
    "phikon", "hibou-b", "conch",
    "phikon-v2", "uni", "hibou-l",
    "virchow", "virchow2",
    "uni2-h",
    "h-optimus-0", "prov-gigapath", "midnight",
]
STUDENTS = ["vit-tiny", "vit-small", "vit-base"]
STUDENT_COLOR = {"vit-tiny": "#3a7ca5",
                 "vit-small": "#8b5cf6",
                 "vit-base": "#a83232"}
STUDENT_LABEL = {"vit-tiny": "ViT-Ti",
                 "vit-small": "ViT-S",
                 "vit-base": "ViT-B"}
TEACHER_PRETTY = {
    "phikon": "Phikon", "phikon-v2": "Phikon-v2",
    "hibou-b": "Hibou-B", "hibou-l": "Hibou-L",
    "conch": "CONCH", "uni": "UNI", "uni2-h": "UNI2-H",
    "virchow": "Virchow", "virchow2": "Virchow2",
    "h-optimus-0": "H-Optimus-0", "prov-gigapath": "Prov-GigaPath",
    "midnight": "Midnight-12k",
}


def load_curves(fm: str, stu: str):
    p = ROOT / fm / stu / "training_log.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    history = d.get("history", [])
    if not history:
        return None
    xs = [e["epoch"] for e in history]
    train = [e.get("train_loss") for e in history]
    val = [e.get("val_loss") for e in history]
    return xs, train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--ncols", type=int, default=4)
    args = ap.parse_args()

    nrows = math.ceil(len(TEACHER_ORDER) / args.ncols)
    fig, axes = plt.subplots(nrows, args.ncols,
                              figsize=(3.0 * args.ncols, 2.4 * nrows),
                              sharex=False)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, fm in zip(axes_flat, TEACHER_ORDER):
        for stu in STUDENTS:
            curves = load_curves(fm, stu)
            if curves is None:
                continue
            xs, _, val = curves
            if not val or all(v is None for v in val):
                continue
            ax.plot(xs, val,
                    color=STUDENT_COLOR[stu],
                    label=STUDENT_LABEL[stu],
                    linewidth=1.4)
        ax.set_title(TEACHER_PRETTY.get(fm, fm), fontsize=9)
        ax.set_xlabel("epoch", fontsize=8)
        ax.set_ylabel("val loss", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(alpha=0.25, linewidth=0.5)
        if ax is axes_flat[0]:
            ax.legend(fontsize=7, loc="upper right", frameon=False)

    # hide spare panels
    for ax in axes_flat[len(TEACHER_ORDER):]:
        ax.set_visible(False)

    fig.suptitle("Per-teacher convergence curves (val cosine distillation loss)",
                 fontsize=11, y=1.00)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        f = out.with_suffix(f".{ext}")
        fig.savefig(f, dpi=200, bbox_inches="tight")
        print(f"[ok] {f}")


if __name__ == "__main__":
    main()
