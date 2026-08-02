#!/usr/bin/env python3
"""F2 — ECE bar chart (calibration after distillation).

For each (teacher, student) pair, bars show teacher and student ECE on
the C16 linear probe (Guo 2017 ECE, 15 bins). Lower = better calibrated.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="outputs/v4_full/atlas_summary.json")
    ap.add_argument("--probe", default="c16lin", help="prefix: c16lin or c16mlp or msilin")
    ap.add_argument("--out",   default="outputs/v4_full/figures/F2_ece_bars")
    args = ap.parse_args()

    rows = json.loads(Path(args.atlas).read_text())["rows"]
    pairs = []
    for r in rows:
        t_ece = r.get(f"{args.probe}_T_ece")
        s_ece = r.get(f"{args.probe}_S_ece")
        if t_ece is None or s_ece is None: continue
        if math.isnan(t_ece) or math.isnan(s_ece): continue
        pairs.append((f"{r['teacher']}/{r['student']}", t_ece, s_ece))

    if not pairs:
        print(f"[F2] no calibration cells filled for probe={args.probe}.")
        return

    pairs.sort(key=lambda p: p[1])
    labels = [p[0] for p in pairs]
    t = np.array([p[1] for p in pairs])
    s = np.array([p[2] for p in pairs])
    x = np.arange(len(pairs))

    fig, ax = plt.subplots(figsize=(max(6.0, 0.30 * len(pairs)), 4.0))
    ax.bar(x - 0.18, t, width=0.36, label="Teacher", color="#888")
    ax.bar(x + 0.18, s, width=0.36, label="Student", color="#3a7ca5")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=80, fontsize=7, ha="right")
    ax.set_ylabel("ECE (15 bins, equal-width)")
    ax.set_title(f"Calibration after distillation — probe={args.probe} (Guo 2017)")
    ax.axhline(0.05, color="darkred", linestyle="--", linewidth=0.8,
               label="ECE = 0.05 (clinical guideline)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[F2] {len(pairs)} bars  -> {out}.{{png,pdf}}")


if __name__ == "__main__":
    main()
