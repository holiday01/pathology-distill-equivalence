#!/usr/bin/env python3
"""Headline figures for the equivalence-first paper:
  fig_tost_forest.pdf  — (a) three-state forest of slide-cluster 90% CIs on the
                          AUROC difference for the 36 pairs; (b) equivalence
                          count vs margin delta.
  fig_se_inflation.pdf — slide-cluster / naive SE inflation vs #test slides,
                          showing it is tile-density-driven (flat ~7x), not
                          slide-count-driven.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Publication settings (Medical Image Analysis / Elsevier): vector PDF with
# editable TrueType text (fonttype 42), sans-serif, >=7pt everywhere.
matplotlib.rcParams.update({
    "savefig.dpi": 600, "figure.dpi": 600,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
})

OUT = Path("paper/figures"); OUT.mkdir(parents=True, exist_ok=True)
DELTA = 0.05
# Okabe-Ito colourblind-safe: blue=equivalent, vermillion=inequivalent, grey=inconclusive
GREEN, RED, GREY = "#0072B2", "#D55E00", "#999999"


def state(lo, hi, d=DELTA):
    if hi < d and lo > -d: return "equivalent", GREEN
    if lo > d or hi < -d: return "inequivalent", RED
    return "inconclusive", GREY


def forest_and_sensitivity():
    rows = json.load(open("outputs/v4_full/c16_equiv_annotated.json"))["rows"]
    rows = sorted(rows, key=lambda r: r["linear_auc_d"])
    sens = json.load(open("outputs/v4_full/delta_sensitivity.json"))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 8.0),
                                  gridspec_kw={"width_ratios": [2.3, 1]})
    counts = {"equivalent": 0, "inequivalent": 0, "inconclusive": 0}
    for i, r in enumerate(rows):
        lo, hi = r["linear_auc_ci"]; d = r["linear_auc_d"]
        st, c = state(lo, hi); counts[st] += 1
        ax.plot([lo, hi], [i, i], color=c, lw=2.2, solid_capstyle="round")
        ax.plot(d, i, "o", color=c, ms=4.5)
    ax.axvspan(-DELTA, DELTA, color="#000000", alpha=0.05)
    ax.axvline(-DELTA, ls="--", c="k", lw=0.9); ax.axvline(DELTA, ls="--", c="k", lw=0.9)
    ax.axvline(0, ls=":", c="k", lw=0.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['teacher']}/{r['student'].replace('vit-','')}" for r in rows],
                       fontsize=7)
    ax.set_xlabel(r"AUROC difference  $\mathrm{AUROC}_S-\mathrm{AUROC}_T$  (90% slide-cluster CI)")
    ax.set_title(f"(a) three-state equivalence forest ({len(rows)} pairs, 41 test slides)",
                 fontsize=10)
    from matplotlib.lines import Line2D
    leg = [Line2D([0],[0],color=GREEN,lw=3,label=f"equivalent ({counts['equivalent']})"),
           Line2D([0],[0],color=RED,lw=3,label=f"inequivalent ({counts['inequivalent']})"),
           Line2D([0],[0],color=GREY,lw=3,label=f"inconclusive ({counts['inconclusive']})")]
    ax.legend(handles=leg, loc="lower right", fontsize=8, frameon=True)
    ax.margins(y=0.01)

    xs = [p[0] for p in sens]; ys = [p[1] for p in sens]
    ax2.plot(xs, ys, "-o", color="#1f77b4", ms=4)
    ax2.axvline(DELTA, ls="--", c="k", lw=0.9)
    ax2.annotate(f"pre-registered\n$\\delta=0.05$: 10/36", xy=(0.05, 10),
                 xytext=(0.07, 4), fontsize=8,
                 arrowprops=dict(arrowstyle="->", lw=0.8))
    ax2.set_xlabel(r"equivalence margin $\delta$ (AUROC)")
    ax2.set_ylabel("pairs certified equivalent / 36")
    ax2.set_title("(b) margin sensitivity", fontsize=10)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_tost_forest.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_tost_forest.png", dpi=600, bbox_inches="tight")
    print("[wrote] fig_tost_forest.pdf  states:", counts)


def inflation_curve():
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10, 4.2))
    # (a) within CAMELYON16: inflation vs #test slides -> flat (slide-count invariant)
    pts = [(2, 6.9, "2"), (12, 5.46, "12"), (41, 6.83, "41")]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    axa.plot(xs, ys, "-o", color="#d62728", ms=7, lw=2)
    for x, y, t in pts:
        axa.annotate(f"{y:.1f}$\\times$", (x, y), textcoords="offset points",
                     xytext=(4, 8), fontsize=8)
    axa.axhline(1.0, ls=":", c="k", lw=0.8)
    axa.text(15, 1.3, "naive SE (no inflation)", fontsize=8)
    axa.set_xlabel("number of independent test slides (clusters)")
    axa.set_ylabel(r"slide-cluster / naive SE inflation  ($\times$)")
    axa.set_ylim(0, 8); axa.set_xlim(0, 45)
    axa.set_title("(a) CAMELYON16: invariant to slide count", fontsize=10)
    axa.grid(alpha=0.3)

    # (b) across cohorts: inflation vs tiles-per-cluster -> design-effect-driven
    coh = [(58, 4.32, "Kather MSI\n103 patients"), (258, 6.83, "CAMELYON16\n41 slides")]
    cx = [c[0] for c in coh]; cy = [c[1] for c in coh]
    axb.plot(cx, cy, "o", color="#1f77b4", ms=9)
    for x, y, t in coh:
        axb.annotate(t + f"\n{y:.1f}$\\times$", (x, y), textcoords="offset points",
                     xytext=(8, -4), fontsize=8, va="center")
    # design-effect direction sqrt(1+(m-1)rho)
    mm = np.linspace(20, 320, 100)
    rho = (6.83**2 - 1) / (258 - 1)
    axb.plot(mm, np.sqrt(1 + (mm - 1) * rho), "--", color="#1f77b4", alpha=0.5,
             label=r"$\sqrt{1+(m-1)\rho}$ design effect")
    axb.axhline(1.0, ls=":", c="k", lw=0.8)
    axb.set_xlabel("tiles per cluster  $m$")
    axb.set_ylabel(r"SE inflation  ($\times$)")
    axb.set_ylim(0, 8); axb.set_xlim(0, 320)
    axb.legend(fontsize=8, loc="lower right")
    axb.set_title("(b) two cohorts: tiles-per-cluster drives it", fontsize=10)
    axb.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_se_inflation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_se_inflation.png", dpi=600, bbox_inches="tight")
    print("[wrote] fig_se_inflation.pdf (2-panel: slide-count invariance + cohort scaling)")


if __name__ == "__main__":
    forest_and_sensitivity()
    inflation_curve()
