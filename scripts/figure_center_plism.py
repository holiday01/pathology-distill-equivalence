#!/usr/bin/env python3
"""Generate the two remaining main figures:
  fig_center.pdf  — medical-centre AUC vs CAMELYON16 task AUC (de Jong protocol),
                    teacher->student trajectories, RI=1 diagonal.
  fig_plism.pdf   — (a) 13 per-teacher 13x13 cross-staining top-1 matrices (diag masked);
                    (b) inter-scanner vs inter-staining scatter.
Writes to paper/figures/.
"""
import json, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 8, "savefig.dpi": 600, "figure.dpi": 600,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
OUT = Path("paper/figures")
HIBOU = {"hibou-b", "hibou-l", "hibou_base", "hibou_large"}
BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#999999"


def fig_center():
    atlas = {(r["teacher"], r["student"]): r for r in json.load(open("outputs/v4_full/atlas_summary.json"))["rows"]}
    ann = {(r["teacher"], r["student"]): r for r in json.load(open("outputs/v4_full/c16_equiv_annotated.json"))["rows"]}
    # Collect first, so the axis limits are derived from the data instead of
    # hard-coded. The previous version clipped at y=0.75 and silently cut the
    # five sub-0.75 hibou students off the bottom of the frame, leaving their
    # trajectory connectors running off the axis with no visible endpoint.
    pts = []
    for (t, s), a in atlas.items():
        an = ann.get((t, s))
        if an is None or an.get("linear_t_auc") is None:
            continue
        cx_t, cx_s = a.get("center_T_auc"), a.get("center_S_auc")
        if cx_t is None or cx_s is None:
            continue
        pts.append((t in HIBOU, cx_t, cx_s, an["linear_t_auc"], an["linear_s_auc"]))

    ylo = min(min(p[3], p[4]) for p in pts) - 0.03
    yhi = max(max(p[3], p[4]) for p in pts) + 0.02
    xlo = min(min(p[1], p[2]) for p in pts)
    xhi = max(max(p[1], p[2]) for p in pts)
    xpad = 0.12 * (xhi - xlo)

    fig, (ax, axz) = plt.subplots(1, 2, figsize=(7.6, 3.9),
                                  gridspec_kw={"width_ratios": [1, 1]})

    def draw(a_, hib_lw):
        for hib, cx_t, cx_s, ty_t, ty_s in pts:
            col = ORANGE if hib else BLUE
            a_.plot([cx_t, cx_s], [ty_t, ty_s], color=col, lw=hib_lw, alpha=0.55, zorder=2)
            a_.scatter([cx_t], [ty_t], s=26, marker="^", color=col,
                       edgecolor="k", linewidth=0.3, zorder=4)
            a_.scatter([cx_s], [ty_s], s=16, marker="o", color=col,
                       alpha=0.85, edgecolor="none", zorder=3)

    # --- (a) equal-aspect view: the RI=1 diagonal is only meaningful when the
    #         two AUC axes share a scale, so panel (a) uses common limits.
    lo, hi = min(ylo, xlo) - 0.01, 1.01
    ax.plot([lo, hi], [lo, hi], color="k", lw=0.8, ls="--", zorder=1)
    ax.fill_between([lo, hi], [lo, lo], [lo, hi], color=GREY, alpha=0.12, zorder=0)
    ax.text(hi - 0.03, lo + 0.04, "RI $<$ 1\n(centre $>$ task)",
            fontsize=7, ha="right", va="bottom", color=GREY)
    draw(ax, 0.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("medical-centre probe AUC\n(TCGA-BRCA tissue-source site)")
    ax.set_ylabel("CAMELYON16 task AUROC")
    ax.set_title("(a) common scale, RI $=$ 1 diagonal", fontsize=9)
    leg = [Line2D([0], [0], marker="^", color="w", markerfacecolor="k", markersize=7, label="teacher"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor="k", markersize=6, label="student"),
           Line2D([0], [0], color=ORANGE, lw=2, label="hibou (flagged)")]
    ax.legend(handles=leg, fontsize=7, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25)

    # --- (b) x-zoom onto the range the centre AUCs actually occupy. Panel (a)
    #         necessarily hides the spread because the centre probe saturates.
    draw(axz, 0.5)
    axz.set_xlim(xlo - xpad, xhi + xpad)
    axz.set_ylim(ylo, yhi)
    axz.set_xlabel("medical-centre probe AUC (zoom)")
    axz.set_ylabel("CAMELYON16 task AUROC")
    axz.set_title("(b) centre-AUC range expanded", fontsize=9)
    axz.grid(alpha=0.25)
    axz.tick_params(labelsize=7)

    fig.suptitle("Centre signal saturates (AUC $\\approx$ 1.000); task signal varies",
                 fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_center.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_center.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[wrote] fig_center.pdf")


def fig_plism():
    rows = list(csv.DictReader(open("outputs/v4_full/figures/F4_plism_matrix_top_1_by_staining_per_teacher.csv")))
    teachers = []
    for r in rows:
        if r["teacher"] not in teachers:
            teachers.append(r["teacher"])
    stains = sorted({r["staining_a"] for r in rows})
    si = {s: i for i, s in enumerate(stains)}
    mats = {t: np.full((len(stains), len(stains)), np.nan) for t in teachers}
    for r in rows:
        mats[r["teacher"]][si[r["staining_a"]], si[r["staining_b"]]] = float(r["top_1_mean"])
    for t in teachers:
        np.fill_diagonal(mats[t], np.nan)  # mask diagonal

    # robustness scatter data (panel b)
    rob = {}
    for r in csv.reader(open("outputs/v4_full/figures/F4_plism_robustness_top_1_accuracy.csv")):
        if r and not r[0].startswith("Inter") and r[0]:
            try:
                rob[r[0]] = (float(r[1]), float(r[2]))  # inter-scanner, inter-staining
            except Exception:
                pass

    ncol = 5
    nrow_a = int(np.ceil(len(teachers) / ncol))
    # Panel (b) previously spanned only the first 2/5 columns at height_ratio
    # 1.35 vs. the heatmaps' 1 each -- a small, cramped scatter with 13
    # overlapping 5.5pt labels. Give it the full row width and a taller
    # share so points and labels have room to separate.
    fig = plt.figure(figsize=(11, 2.1 * nrow_a + 4.6))
    gs = fig.add_gridspec(nrow_a + 1, ncol, height_ratios=[1] * nrow_a + [2.6], hspace=0.55, wspace=0.25)
    vmax = np.nanpercentile([mats[t][~np.isnan(mats[t])].max() for t in teachers], 90)
    im = None
    for i, t in enumerate(teachers):
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        im = ax.imshow(mats[t], cmap="viridis", vmin=0, vmax=max(vmax, 0.05), aspect="equal")
        flag = " §" if t in HIBOU else ""
        ax.set_title(t.replace("_", "-") + flag, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=fig.axes, fraction=0.012, pad=0.01, label="cross-staining top-1")
    # panel (b): inter-scanner vs inter-staining
    axb = fig.add_subplot(gs[nrow_a, :])
    for name, (sc, stn) in rob.items():
        hib = name in HIBOU
        axb.scatter(sc, stn, s=50, color=ORANGE if hib else BLUE, edgecolor="k", linewidth=0.4, zorder=3)
        axb.annotate(name.replace("_", "-"), (sc, stn), fontsize=8, xytext=(4, 4),
                     textcoords="offset points")
    axb.set_xlabel("inter-scanner top-1", fontsize=9); axb.set_ylabel("inter-staining top-1", fontsize=9)
    axb.set_title("(b) scanner vs stain robustness (dissociable)", fontsize=10)
    axb.tick_params(labelsize=8)
    axb.margins(0.12)
    axb.grid(alpha=0.25)
    fig.suptitle("(a) PLISM cross-staining top-1 retrieval, per teacher (13 stainings, diagonal masked)",
                 fontsize=9, y=0.995)
    fig.savefig(OUT / "fig_plism.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_plism.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] fig_plism.pdf ({len(teachers)} teachers, {len(stains)} stainings)")


if __name__ == "__main__":
    fig_center()
    fig_plism()
