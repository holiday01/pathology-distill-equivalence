#!/usr/bin/env python3
"""F5 — Energy Pareto: downstream accuracy / Dice vs training kWh.

Reads:
  paper_metrics.csv  (per-run training Wh, idle-subtracted)
  atlas_summary.json (per-run downstream + seg quality)

x-axis: train Wh / 1000  (kWh)
y-axis: chosen quality metric (default: c16_linear student accuracy;
        switchable to mDice_fg or kather-MSI accuracy via --metric).
Marker color encodes student size (Ti / S / B); marker shape encodes teacher.
Pareto frontier (non-dominated set) drawn as a dashed line.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STUDENT_COLORS = {"vit-tiny": "#3a7ca5", "vit-small": "#8b5cf6", "vit-base": "#a83232"}
STUDENT_SIZES  = {"vit-tiny": 30, "vit-small": 60, "vit-base": 110}


def metric_lookup(row: dict, metric: str):
    if metric == "c16_acc":   return row.get("c16lin_S_acc")
    if metric == "c16_auc":   return row.get("c16lin_S_auc")
    if metric == "msi_acc":   return row.get("msilin_S_acc")
    if metric == "mdice":     return row.get("seg_mDice_S")
    raise ValueError(f"unknown metric {metric}")


def pareto_front(pts, x_min=True, y_max=True):
    """Return indices of non-dominated points (low x, high y)."""
    pts = np.asarray(pts)
    n = len(pts); idx = []
    order = np.argsort(pts[:, 0])
    best_y = -np.inf
    for i in order:
        y = pts[i, 1]
        if y > best_y:
            idx.append(int(i))
            best_y = y
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-csv", default="outputs/v4_full/paper_metrics.csv")
    ap.add_argument("--atlas",       default="outputs/v4_full/atlas_summary.json")
    ap.add_argument("--c16-annotated",
                    default="outputs/v4_full/c16_equiv_annotated.json",
                    help="annotated C16 TOST results (overrides atlas AUROCs if present)")
    ap.add_argument("--metric",      default="c16_auc",
                    choices=["c16_acc", "c16_auc", "msi_acc", "mdice"])
    ap.add_argument("--out",         default="outputs/v4_full/figures/F5_energy_pareto")
    args = ap.parse_args()

    rows = json.loads(Path(args.atlas).read_text())["rows"]
    by_pair = {f"{r['teacher']}/{r['student']}": r for r in rows}

    # Overlay annotated C16 AUROCs (from the fixed-label evaluation) into by_pair rows.
    c16ann_path = Path(args.c16_annotated)
    if c16ann_path.exists():
        ann_rows = json.loads(c16ann_path.read_text()).get("rows", [])
        for ar in ann_rows:
            k = f"{ar['teacher']}/{ar['student']}"
            if k in by_pair:
                by_pair[k]["c16lin_S_auc"] = ar.get("linear_s_auc")
                by_pair[k]["c16lin_T_auc"] = ar.get("linear_t_auc")
                by_pair[k]["c16lin_S_acc"] = ar.get("linear_s_acc")
                by_pair[k]["c16lin_T_acc"] = ar.get("linear_t_acc")

    # paper_metrics.csv has "FM × Student" (e.g. "conch×vit-base") and "Wh (−idle)"
    energy = {}
    if Path(args.metrics_csv).exists():
        with open(args.metrics_csv) as f:
            for r in csv.DictReader(f):
                key = (r.get("Run (FM × Student)") or r.get("Run")
                       or r.get("run_id") or "").replace("×", "/")
                wh  = (r.get("Wh (−idle)") or r.get("Wh")
                       or r.get("energy_wh"))
                try:
                    energy[key] = float(wh) if wh else None
                except (ValueError, TypeError):
                    pass

    pts = []
    for k, r in by_pair.items():
        wh = energy.get(k)
        m  = metric_lookup(r, args.metric)
        if wh is None or m is None or (isinstance(m, float) and math.isnan(m)):
            continue
        pts.append({"label": k, "kwh": wh / 1000.0, "y": m, "student": r["student"]})

    if not pts:
        print("[F5] no rows have both Wh and downstream metric. Eval not yet run.")
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for p in pts:
        ax.scatter(p["kwh"], p["y"],
                   color=STUDENT_COLORS.get(p["student"], "#444"),
                   s=STUDENT_SIZES.get(p["student"], 50),
                   alpha=0.85, edgecolor="white", linewidth=0.4)

    if len(pts) >= 3:
        arr = np.array([[p["kwh"], p["y"]] for p in pts])
        idx = pareto_front(arr)
        front = arr[idx][np.argsort(arr[idx, 0])]
        ax.plot(front[:, 0], front[:, 1], "--", color="black", linewidth=0.8,
                label="Pareto front")

    for stu, c in STUDENT_COLORS.items():
        ax.scatter([], [], color=c, s=STUDENT_SIZES[stu], label=stu)
    ax.set_xlabel("Training energy (kWh, idle-subtracted)")
    ax.set_ylabel(f"Quality: {args.metric}")
    ax.set_title("Distillation energy vs downstream quality")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[F5] {len(pts)} points  -> {out}.{{png,pdf}}")


if __name__ == "__main__":
    main()
