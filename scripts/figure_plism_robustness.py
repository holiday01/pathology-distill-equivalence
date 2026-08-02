#!/usr/bin/env python3
"""F4 — PLISM cross-teacher robustness heatmap.

`plismbench evaluate` writes, per teacher, a `results.csv` at
    {metrics_dir}/{n_tiles}_tiles/{teacher}/results.csv
whose rows are the 4 robustness regimes (inter-scanner, inter-staining,
inter-scanner+inter-staining, all) and columns are cosine_similarity and
top_{1,3,5,10}_accuracy. Each cell is the string "mean (std) ; median (iqr)".

This renders a single (teacher x regime) heatmap per metric so the FMs can be
compared at a glance — the headline "which model is most stain/scanner
invariant" figure. Reads the *mean* from each cell.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Canonical teacher order (matches scripts/run_plism_eval.sh), small -> large.
TEACHER_ORDER = [
    "phikon", "phikonv2", "hibou_base", "hibou_large", "conch",
    "uni", "uni2h", "virchow", "virchow2", "hoptimus0",
    "provgigapath", "midnight_12k", "gpfm",
]

# results.csv row label -> display label, in the order we want columns to appear.
REGIMES = [
    ("inter-scanner", "Inter-scanner"),
    ("inter-staining", "Inter-staining"),
    ("inter-scanner, inter-staining", "Both"),
    ("all", "All"),
]


def parse_mean(cell: str) -> float:
    """'0.950 (0.020) ; 0.960 (0.010)' -> 0.950 (the mean)."""
    if not isinstance(cell, str):
        return float(cell)
    m = re.search(r"[-+]?\d*\.?\d+", cell.split(";")[0])
    return float(m.group()) if m else np.nan


def find_results(results_root: Path) -> dict[str, Path]:
    """Map teacher -> results.csv path under {n_tiles}_tiles/{teacher}/."""
    found: dict[str, Path] = {}
    for csv in results_root.glob("*_tiles/*/results.csv"):
        found[csv.parent.name] = csv  # later (larger n_tiles) dirs override
    return found


def build_matrix(results: dict[str, Path], metric: str):
    teachers = [t for t in TEACHER_ORDER if t in results]
    teachers += sorted(t for t in results if t not in TEACHER_ORDER)
    regime_keys = [k for k, _ in REGIMES]
    M = np.full((len(teachers), len(regime_keys)), np.nan)
    for i, t in enumerate(teachers):
        df = pd.read_csv(results[t], index_col=0)
        if metric not in df.columns:
            print(f"[F4] {t}: metric '{metric}' absent (cols={list(df.columns)})")
            continue
        idx = {str(k).strip(): k for k in df.index}
        for j, rk in enumerate(regime_keys):
            if rk in idx:
                M[i, j] = parse_mean(df.loc[idx[rk], metric])
    return M, teachers


def plot(M, teachers, metric: str, out: Path):
    labels = [lbl for _, lbl in REGIMES]
    fig, ax = plt.subplots(figsize=(0.9 * len(labels) + 2.5, 0.42 * len(teachers) + 1.6))
    vmin = np.nanmin(M) if np.isfinite(M).any() else 0.0
    im = ax.imshow(M, cmap="magma", vmin=vmin, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(teachers)), teachers, fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                v = M[i, j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if v < (vmin + 1.0) / 2 else "black")
    pretty = metric.replace("_", " ").replace("accuracy", "acc.")
    ax.set_title(f"PLISM robustness — {pretty} (mean)", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=pretty)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plism-results", default="outputs/plism/results")
    ap.add_argument("--out-dir", default="outputs/v4_full/figures")
    ap.add_argument("--metrics", nargs="+",
                    default=["top_1_accuracy", "cosine_similarity"])
    args = ap.parse_args()

    root = Path(args.plism_results)
    results = find_results(root)
    if not results:
        print(f"[F4] no results.csv under {root}/*_tiles/*/ — eval not done yet.")
        return
    print(f"[F4] found {len(results)} teachers: {', '.join(sorted(results))}")

    out_dir = Path(args.out_dir)
    for metric in args.metrics:
        M, teachers = build_matrix(results, metric)
        if not np.isfinite(M).any():
            print(f"[F4] {metric}: no values parsed, skipping.")
            continue
        out = out_dir / f"F4_plism_robustness_{metric}"
        plot(M, teachers, metric, out)
        print(f"[F4] {metric}: {len(teachers)} teachers -> {out}.{{png,pdf}}")

        # also drop a tidy mean table next to the figure for the paper
        tbl = pd.DataFrame(M, index=teachers, columns=[l for _, l in REGIMES])
        tbl.to_csv(out_dir / f"F4_plism_robustness_{metric}.csv")
    print("[F4] done.")


if __name__ == "__main__":
    main()
