#!/usr/bin/env python3
"""F6 — TOST equivalence strip plot (Schuirmann 1987 + Davison cluster boot).

For each (teacher, student) pair, draw a horizontal segment for the
90% slide-cluster CI of (student − teacher) accuracy difference, dot at
the mean. Pairs sorted by mean d. Vertical dashed lines mark ±margin
(default 0.03). Colour: green if equivalent (CI ⊂ [-m, +m]), grey otherwise.
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
    ap.add_argument("--probe", default="c16lin")
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--out",   default="outputs/v4_full/figures/F6_tost_strip")
    args = ap.parse_args()

    rows = json.loads(Path(args.atlas).read_text())["rows"]
    pts = []
    for r in rows:
        d   = r.get(f"{args.probe}_d_mean")
        lo  = r.get(f"{args.probe}_d_ci90_lo")
        hi  = r.get(f"{args.probe}_d_ci90_hi")
        eq  = r.get(f"{args.probe}_equiv")
        if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in [d, lo, hi]):
            continue
        pts.append({"label": f"{r['teacher']}/{r['student']}",
                    "d": d, "lo": lo, "hi": hi, "eq": bool(eq)})

    if not pts:
        print(f"[F6] no TOST cells filled for probe={args.probe}.")
        return

    pts.sort(key=lambda p: p["d"])
    fig, ax = plt.subplots(figsize=(6.5, max(4.0, 0.20 * len(pts))))
    for i, p in enumerate(pts):
        col = "#2e7d32" if p["eq"] else "#777"
        ax.plot([p["lo"], p["hi"]], [i, i], "-", color=col, linewidth=2.0, alpha=0.85)
        ax.scatter([p["d"]], [i], color=col, s=18, zorder=3)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.axvspan(-args.margin, args.margin, color="#fde68a", alpha=0.3, zorder=0,
               label=f"equivalence band ±{args.margin}")
    ax.axvline(-args.margin, color="darkred", linestyle="--", linewidth=0.6)
    ax.axvline( args.margin, color="darkred", linestyle="--", linewidth=0.6)
    ax.set_yticks(range(len(pts)), [p["label"] for p in pts], fontsize=6)
    ax.set_xlabel("Δ accuracy = student − teacher  (90% slide-cluster CI)")
    ax.set_title(f"TOST equivalence — probe={args.probe}, margin=±{args.margin}")
    ax.legend(fontsize=8, loc="lower right")
    n_eq = sum(1 for p in pts if p["eq"])
    ax.text(0.02, 0.98, f"equivalent: {n_eq}/{len(pts)}", transform=ax.transAxes,
            fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[F6] {len(pts)} pairs ({n_eq} equivalent) -> {out}.{{png,pdf}}")


if __name__ == "__main__":
    main()
