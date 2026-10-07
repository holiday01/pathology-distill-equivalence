#!/usr/bin/env python3
"""Supplementary figure D-1: per-teacher convergence curves for the
v4_full atlas. Pure CPU; reads only training_log.json files (no eval
required). Produces one panel per reported teacher, each panel
overlays val loss curves for ViT-Ti / ViT-S / ViT-B.

Within-teacher convergence is the meaningful comparison (Ti vs S vs B
on the same teacher, same loss space). Cross-teacher loss is NOT
comparable (each teacher embeds in its own feature space) and the
script makes no attempt to normalise across teachers.

Output: paper/figures/figS_D1_convergence_curves.{pdf,png}
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).parent))
from paper_cohort import filter_rows as _panel, keep_teacher as _keep

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("outputs/v4_full")
OUT = Path("paper/figures/figS_D1_convergence_curves")

TEACHER_ORDER = [  # same order as the other manuscript figures
    "phikon", "phikon-v2", "conch", "uni", "uni2-h", "prov-gigapath",
    "virchow", "virchow2", "h-optimus-0", "midnight",
    "hibou-b", "hibou-l",  # dropped by paper_cohort.keep_teacher
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

    # Rendered at the supplementary text width (A4, 1-in margins: 6.27 in) so
    # \includegraphics[width=\textwidth] does not rescale; all text >= 7 pt.
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    TEACHER_ORDER[:] = [t for t in TEACHER_ORDER if _keep(t)]
    nrows = math.ceil(len(TEACHER_ORDER) / args.ncols)
    fig, axes = plt.subplots(nrows, args.ncols, figsize=(6.27, 1.45 * nrows),
                             sharex=True, layout="constrained")
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    max_ep = 0
    early = []
    for ax, fm in zip(axes_flat, TEACHER_ORDER):
        for stu in STUDENTS:
            curves = load_curves(fm, stu)
            if curves is None:
                continue
            xs, _, val = curves
            if not val or all(v is None for v in val):
                continue
            max_ep = max(max_ep, max(xs))
            ax.plot(xs, val, color=STUDENT_COLOR[stu], label=STUDENT_LABEL[stu],
                    linewidth=1.0)
            early.append((fm, stu, xs[-1], xs[-1], val[-1], ax))
        ax.set_title(TEACHER_PRETTY.get(fm, fm), pad=2)
        ax.grid(alpha=0.25, linewidth=0.4)
    # Mark runs that ended before the full schedule (patience-5 early stop).
    stopped = []
    for fm, stu, last, _, v, ax in early:
        if last < max_ep:
            ax.plot([last], [v], "x", color=STUDENT_COLOR[stu], ms=4, mew=1.0)
            stopped.append(f"{TEACHER_PRETTY.get(fm, fm)} x {STUDENT_LABEL[stu]} (epoch {last})")
    for i, ax in enumerate(axes_flat[:len(TEACHER_ORDER)]):
        if i % args.ncols == 0:
            ax.set_ylabel("val loss")
        if i + args.ncols >= len(TEACHER_ORDER):
            ax.set_xlabel("epoch")
            ax.xaxis.set_tick_params(labelbottom=True)

    # The spare panel(s) carry the legend instead of staying empty.
    spare = axes_flat[len(TEACHER_ORDER):]
    for ax in spare:
        ax.set_axis_off()
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=STUDENT_COLOR[s], lw=1.2, label=STUDENT_LABEL[s])
               for s in STUDENTS]
    handles.append(Line2D([0], [0], marker="x", ls="", color="#444444", ms=4,
                          label="early stop"))
    if len(spare):
        spare[-1].legend(handles=handles, loc="center", frameon=False,
                         title="student", title_fontsize=7)
    else:
        axes_flat[0].legend(handles=handles, loc="upper right", frameon=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        f = out.with_suffix(f".{ext}")
        fig.savefig(f, dpi=600)
        print(f"[ok] {f}")
    print("[early stop]", "; ".join(stopped))


if __name__ == "__main__":
    main()
