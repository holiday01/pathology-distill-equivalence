#!/usr/bin/env python3
"""F3 — Medical-center probe scatter (de Jong 2025 critique).

x-axis: center AUC (how well embedding encodes the medical center).
y-axis: C16 task AUC (biological signal).
Each point = one model. Diagonal = chance ratio (RI=1).
Teachers and students plotted with different markers; pairs connected by
thin lines so the eye sees the per-distillation shift.
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
    ap.add_argument("--atlas",        default="outputs/v4_full/atlas_summary.json")
    ap.add_argument("--c16-annotated",
                    default="outputs/v4_full/c16_equiv_annotated.json",
                    help="annotated C16 TOST results for task AUROCs (overrides atlas NaN values)")
    ap.add_argument("--out",   default="outputs/v4_full/figures/F3_center_probe")
    args = ap.parse_args()

    rows = json.loads(Path(args.atlas).read_text())["rows"]

    # Build annotated C16 AUROC lookup: (teacher, student) → {t_auc, s_auc}
    ann_auc = {}
    c16ann_path = Path(args.c16_annotated)
    if c16ann_path.exists():
        ann_rows = json.loads(c16ann_path.read_text()).get("rows", [])
        for ar in ann_rows:
            ann_auc[(ar["teacher"], ar["student"])] = {
                "t": ar.get("linear_t_auc"),
                "s": ar.get("linear_s_auc"),
            }

    pts = []
    for r in rows:
        tc = r.get("center_T_auc"); sc = r.get("center_S_auc")
        key = (r["teacher"], r["student"])
        ta = ann_auc.get(key, {}).get("t") or r.get("c16lin_T_auc")
        sa = ann_auc.get(key, {}).get("s") or r.get("c16lin_S_auc")
        if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in [tc, sc, ta, sa]):
            continue
        pts.append({"label": f"{r['teacher']}/{r['student']}",
                    "tc": tc, "sc": sc, "ta": ta, "sa": sa})

    if not pts:
        print("[F3] no center-probe cells filled.")
        return

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    for p in pts:
        ax.plot([p["tc"], p["sc"]], [p["ta"], p["sa"]],
                color="#bbb", linewidth=0.5, zorder=1)
    tc = np.array([p["tc"] for p in pts]); ta = np.array([p["ta"] for p in pts])
    sc = np.array([p["sc"] for p in pts]); sa = np.array([p["sa"] for p in pts])
    ax.scatter(tc, ta, marker="^", s=42, color="#a83232", label="Teacher",  zorder=2)
    ax.scatter(sc, sa, marker="o", s=42, color="#3a7ca5", label="Student",  zorder=2)
    lo = min(0.5, tc.min(), sc.min(), ta.min(), sa.min()) - 0.02
    hi = max(1.0, tc.max(), sc.max(), ta.max(), sa.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=0.6,
            label="RI = 1 (task ≈ confounder)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Medical-center probe AUC (lower = less confounded)")
    ax.set_ylabel("C16 task AUC (higher = better signal)")
    ax.set_title("Center vs task signal across 36 distillation pairs")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    n_above = sum(1 for p in pts if (p["sa"] - p["sc"]) > (p["ta"] - p["tc"]))
    print(f"[F3] {len(pts)} pairs  ({n_above} with student > teacher RI gap)  -> {out}.{{png,pdf}}")


if __name__ == "__main__":
    main()
