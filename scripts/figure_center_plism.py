#!/usr/bin/env python3
"""Generate the two remaining main figures:
  fig_center.pdf  — medical-centre AUC vs CAMELYON16 task AUC (de Jong protocol),
                    teacher->student trajectories, RI=1 diagonal.
  fig_plism.pdf   — (a) per-teacher 13x13 cross-staining top-1 matrices (diagonal
                    masked), one per teacher retained by paper_cohort.keep_teacher
                    (11 incl. GPFM); (b) inter-scanner vs inter-staining top-1 per
                    teacher. Drawn at IEEE figure* width (7.16 in), >=7 pt text.
Writes to paper/figures/.
"""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).parent))
from paper_cohort import filter_rows as _panel, keep_teacher as _keep
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
    ann = {(r["teacher"], r["student"]): r for r in _panel(json.load(open("outputs/v4_full/c16_270/c16_equiv_annotated.json"))["rows"])}
    # Axis limits are derived from the data rather than hard-coded, so no
    # point or connector can fall outside the frame.
    pts = []
    for (t, s), a in atlas.items():
        an = ann.get((t, s))
        if an is None or an.get("linear_t_auc") is None:
            continue
        cx_t, cx_s = a.get("center_T_auc"), a.get("center_S_auc")
        if cx_t is None or cx_s is None:
            continue
        pts.append((False, cx_t, cx_s, an["linear_t_auc"], an["linear_s_auc"]))

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
           Line2D([0], [0], marker="o", color="w", markerfacecolor="k", markersize=6, label="student")]
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


PLISM_DISPLAY = {
    "phikon": "Phikon", "phikonv2": "Phikon-v2", "conch": "CONCH", "uni": "UNI",
    "uni2h": "UNI2-H", "provgigapath": "Prov-GigaPath", "virchow": "Virchow",
    "virchow2": "Virchow2", "hoptimus0": "H-Optimus-0", "midnight_12k": "Midnight-12k",
    "gpfm": "GPFM", "hibou_base": "Hibou-B", "hibou_large": "Hibou-L",
}
# Label offsets (points) for panel (b), chosen so the clustered labels
# (Phikon/Phikon-v2; UNI/UNI2-H/Virchow/Prov-GigaPath) do not collide.
PLISM_OFFSET = {
    "phikon": (5, -3, "left"), "phikonv2": (-5, 3, "right"),
    "uni": (5, -5, "left"), "uni2h": (-5, 2, "right"), "virchow": (-5, 6, "right"),
    "virchow2": (5, 0, "left"), "provgigapath": (5, -4, "left"),
    "conch": (5, -4, "left"), "hoptimus0": (-5, 0, "right"),
    "midnight_12k": (-5, 0, "right"), "gpfm": (5, -4, "left"),
}


def fig_plism():
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.linewidth": 0.6})
    rows = list(csv.DictReader(open("outputs/v4_full/figures/F4_plism_matrix_top_1_by_staining_per_teacher.csv")))
    rows = [r for r in rows if _keep(r["teacher"])]
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
    rob = {k: v for k, v in rob.items() if _keep(k)}

    # Layout: (a) heatmap grid on the left (4 columns), (b) scatter on the right.
    ncol = 4
    nrow = int(np.ceil(len(teachers) / ncol))
    fig = plt.figure(figsize=(7.16, 3.55), layout="constrained")
    sfa, sfb = fig.subfigures(1, 2, width_ratios=[1.12, 1], wspace=0.02)
    gs = sfa.add_gridspec(nrow, ncol)
    vmax = np.nanpercentile([mats[t][~np.isnan(mats[t])].max() for t in teachers], 90)
    im = None
    axs = []
    for i, t in enumerate(teachers):
        ax = sfa.add_subplot(gs[i // ncol, i % ncol])
        im = ax.imshow(mats[t], cmap="viridis", vmin=0, vmax=max(vmax, 0.05), aspect="equal")
        ax.set_title(PLISM_DISPLAY.get(t, t), fontsize=7, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        axs.append(ax)
    # The spare grid cell(s) hold the colour bar instead of staying empty.
    spare = [sfa.add_subplot(gs[j // ncol, j % ncol]) for j in range(len(teachers), nrow * ncol)]
    for sp in spare:
        sp.set_axis_off()
    if spare:
        cax = spare[0].inset_axes([0.1, 0.1, 0.12, 0.8])
        cb = sfa.colorbar(im, cax=cax)
    else:
        cb = sfa.colorbar(im, ax=axs, fraction=0.03)
    cb.set_label("cross-staining\ntop-1", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    sfa.suptitle("(a)", x=0.01, ha="left", fontsize=8.5, fontweight="bold")

    axb = sfb.subplots()
    for name, (sc, stn) in rob.items():
        axb.scatter(sc, stn, s=18, color=BLUE, edgecolor="k", linewidth=0.4, zorder=3)
        dx, dy, ha = PLISM_OFFSET.get(name, (4, 3, "left"))
        axb.annotate(PLISM_DISPLAY.get(name, name), (sc, stn), fontsize=7,
                     xytext=(dx, dy), textcoords="offset points", ha=ha, va="center")
    axb.set_xlabel("inter-scanner top-1")
    axb.set_ylabel("inter-staining top-1")
    xs = [v[0] for v in rob.values()]; ys = [v[1] for v in rob.values()]
    axb.set_xlim(0, max(xs) * 1.12); axb.set_ylim(0, max(ys) * 1.12)
    axb.grid(alpha=0.25, lw=0.4)
    axb.set_title("(b)", loc="left", fontsize=8.5, fontweight="bold")
    fig.savefig(OUT / "fig_plism.pdf")
    fig.savefig(OUT / "fig_plism.png", dpi=600)
    plt.close(fig)
    print(f"[wrote] fig_plism.pdf ({len(teachers)} teachers, {len(stains)} stainings): "
          f"{[PLISM_DISPLAY.get(t, t) for t in teachers]}")


if __name__ == "__main__":
    # fig_center() is no longer called: the centre probe is reported in
    # Supplementary Note D as text only, and no manuscript file includes it.
    fig_plism()
