#!/usr/bin/env python3
"""F1 — CKA heatmap (teachers × students).

One row per teacher, three columns (vit-tiny / vit-small / vit-base).
Cell value = student↔teacher CKA on test set. Higher = student matches
teacher representation. Reads atlas_summary.json.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STUDENTS = ["vit-tiny", "vit-small", "vit-base"]


def load_grid(atlas_json: Path):
    data = json.loads(atlas_json.read_text())
    rows = data["rows"]
    teachers = sorted({r["teacher"] for r in rows})
    grid = np.full((len(teachers), len(STUDENTS)), np.nan, dtype=float)
    for r in rows:
        try:
            i = teachers.index(r["teacher"])
            j = STUDENTS.index(r["student"])
        except ValueError:
            continue
        v = r.get("cka_overall")
        if isinstance(v, (int, float)) and not math.isnan(v):
            grid[i, j] = float(v)
    return grid, teachers


def plot(grid: np.ndarray, teachers: list, out: Path):
    fig, ax = plt.subplots(figsize=(4.0, max(3.5, 0.32 * len(teachers))))
    im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(STUDENTS)), STUDENTS)
    ax.set_yticks(range(len(teachers)), teachers)
    ax.set_xlabel("Student")
    ax.set_ylabel("Teacher (foundation model)")
    ax.set_title("Linear CKA: student vs teacher")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if math.isnan(v):
                ax.text(j, i, "·", ha="center", va="center", color="gray", fontsize=10)
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.6 else "black", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.04, label="CKA")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="outputs/v4_full/atlas_summary.json")
    ap.add_argument("--out",   default="outputs/v4_full/figures/F1_cka_heatmap")
    args = ap.parse_args()
    grid, teachers = load_grid(Path(args.atlas))
    n_have = int((~np.isnan(grid)).sum())
    if n_have == 0:
        print(f"[F1] no CKA cells populated yet ({grid.shape[0]}×{grid.shape[1]} grid all NaN). "
              f"Run downstream eval first.")
        return
    plot(grid, teachers, Path(args.out))
    print(f"[F1] cells filled: {n_have}/{grid.size}  -> {args.out}.{{png,pdf}}")


if __name__ == "__main__":
    main()
