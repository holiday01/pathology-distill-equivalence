#!/usr/bin/env python3
"""Supporting figures from available atlas data:
  fig_cka.pdf          — 12x3 teacher-student CKA heatmap (hibou rows flagged).
  fig_calibration.pdf  — teacher vs student ECE (a) and AURC (b) on C16 probe.
  fig_energy.pdf       — training energy (Wh) vs student C16 AUROC, Pareto front.
Also prints the CKA narrative numbers (unflagged range, ViT-B - ViT-Ti gap).
"""
import json, glob, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "savefig.dpi": 600, "figure.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
})
OUT = Path("paper/figures")
STUD = ["vit-tiny", "vit-small", "vit-base"]
HIBOU = {"hibou-b", "hibou-l"}


def load():
    d = {}
    for f in sorted(glob.glob("outputs/v4_full/*/*/downstream_report.json")):
        p = f.split("/"); t, s = p[-3], p[-2]
        r = json.load(open(f)).get("results", {})
        cal = r.get("probes", {}).get("c16_linear", {}).get("calibration", {})
        d[(t, s)] = {
            "cka": r.get("feature_similarity", {}).get("cka_overall"),
            "t_ece": cal.get("teacher", {}).get("ece"), "s_ece": cal.get("student", {}).get("ece"),
            "t_aurc": cal.get("teacher", {}).get("aurc"), "s_aurc": cal.get("student", {}).get("aurc"),
        }
    return d


def cka_heatmap(d):
    teachers = sorted({t for t, _ in d})
    M = np.full((len(teachers), 3), np.nan)
    for i, t in enumerate(teachers):
        for j, s in enumerate(STUD):
            v = d.get((t, s), {}).get("cka")
            if v is not None: M[i, j] = v
    fig, ax = plt.subplots(figsize=(4.4, 6.2))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0.1, vmax=1.0)
    ax.set_xticks(range(3)); ax.set_xticklabels([s.replace("vit-", "ViT-") for s in STUD])
    ax.set_yticks(range(len(teachers))); ax.set_yticklabels(teachers, fontsize=8)
    for i, t in enumerate(teachers):
        for j in range(3):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] < 0.6 else "black", fontsize=7)
        if t in HIBOU:
            ax.add_patch(plt.Rectangle((-0.5, i-0.5), 3, 1, fill=False,
                                       edgecolor="#D55E00", lw=2, ls="--"))
    fig.colorbar(im, ax=ax, fraction=0.046, label="linear CKA")
    ax.set_title("teacher$\\to$student CKA (C16)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cka.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_cka.png", dpi=600, bbox_inches="tight")
    # narrative numbers on unflagged set
    unfl = [d[(t, s)]["cka"] for (t, s) in d if t not in HIBOU and d[(t, s)]["cka"] is not None]
    gaps = []
    for t in teachers:
        if t in HIBOU: continue
        a, b = d.get((t, "vit-base"), {}).get("cka"), d.get((t, "vit-tiny"), {}).get("cka")
        if a is not None and b is not None: gaps.append(a - b)
    print(f"[CKA] unflagged n={len(unfl)} range [{min(unfl):.3f}, {max(unfl):.3f}]; "
          f"median ΔCKA(B-Ti)={np.median(gaps):.3f} IQR[{np.quantile(gaps,.25):.3f},{np.quantile(gaps,.75):.3f}] "
          f"over {len(gaps)} teachers")


def calibration(d=None):
    # Valid 41-slide lesion-annotated probe (not the degenerate 2-slide probe);
    # non-flagged set (hibou excluded) to match the paper's calibration narrative.
    rows = json.load(open("outputs/v4_full/c16_calibration_annotated.json"))["rows"]
    rows = [r for r in rows if "hibou" not in r["teacher"]]
    te = np.array([r["linear_t_ece"] for r in rows]); se = np.array([r["linear_s_ece"] for r in rows])
    ta = np.array([r["linear_t_aurc"] for r in rows]); sa = np.array([r["linear_s_aurc"] for r in rows])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.3))
    for ax, x, y, lbl in ((a1, te, se, "ECE"), (a2, ta, sa, "AURC")):
        lo = min(x.min(), y.min()); hi = max(x.max(), y.max())
        pad = 0.05 * (hi - lo + 1e-6)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k:", lw=0.8)
        ax.scatter(x, y, s=22, color="#0072B2", alpha=0.7, edgecolor="none")
        if lbl == "ECE":
            ax.axhline(0.05, color="#D55E00", lw=0.7, ls="--")
            ax.axvline(0.05, color="#D55E00", lw=0.7, ls="--")
        ax.set_xlabel(f"teacher {lbl}"); ax.set_ylabel(f"student {lbl}")
        ax.set_title(f"({'a' if lbl=='ECE' else 'b'}) {lbl}", fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle("Calibration before vs after distillation (41-slide lesion-annotated C16 probe)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_calibration.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_calibration.png", dpi=600, bbox_inches="tight")
    print(f"[calib] n={len(rows)} (annotated, non-hibou): median teacher ECE {np.median(te):.3f}, "
          f"student ECE {np.median(se):.3f}; students below 0.05: {int((se<=0.05).sum())}/{len(rows)}")


def energy():
    pm = {r["run_id"]: r for r in csv.DictReader(open("outputs/v4_full/paper_metrics.csv"))}
    eq = {(r["teacher"], r["student"]): r for r in
          json.load(open("outputs/v4_full/c16_equiv_annotated.json"))["rows"]}
    xs, ys, ms = [], [], []
    smark = {"vit-tiny": "o", "vit-small": "s", "vit-base": "^"}
    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    for rid, r in pm.items():
        try:
            wh = float(r["energy_wh"]); fm, s = r["fm"], r["student"]
        except (KeyError, ValueError):
            continue
        auc = eq.get((fm, s), {}).get("linear_s_auc")
        if auc is None or wh <= 0: continue
        xs.append(wh); ys.append(auc); ms.append(s)
        ax.scatter(wh, auc, marker=smark.get(s, "o"), s=34, color="#0072B2", alpha=0.7)
    # Pareto frontier (low energy, high AUROC)
    pts = sorted(zip(xs, ys))
    front, best = [], -1
    for x, y in pts:
        if y > best: front.append((x, y)); best = y
    if front:
        ax.plot([p[0] for p in front], [p[1] for p in front], "--",
                color="#D55E00", lw=1.5, label="Pareto frontier")
    from matplotlib.lines import Line2D
    leg = [Line2D([0],[0],marker=smark[s],color="w",markerfacecolor="#0072B2",
                  label=s.replace("vit-","ViT-"),ms=7) for s in STUD]
    leg.append(Line2D([0],[0],ls="--",color="#D55E00",label="Pareto frontier"))
    ax.legend(handles=leg, fontsize=8)
    ax.set_xlabel("training energy (Wh, idle-subtracted)")
    ax.set_ylabel("student C16 tile AUROC")
    ax.set_title("Training energy vs distilled-student quality", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_energy.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_energy.png", dpi=600, bbox_inches="tight")
    print(f"[energy] {len(xs)} runs plotted")


if __name__ == "__main__":
    d = load()
    cka_heatmap(d)
    calibration(d)
    energy()
