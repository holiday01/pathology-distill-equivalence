#!/usr/bin/env python3
"""Headline figures for the equivalence-first paper:
  fig_tost_forest.pdf  — (a) three-state forest of slide-cluster 90% CIs on the
                          AUROC difference for the panel pairs; (b) clustered and
                          naive equivalence counts vs margin delta.
  fig_se_inflation.pdf — slide-cluster / naive SE inflation vs #test slides,
                          (a) slide-count sweep on the headline test half, (b) measured
                          inflation against the mean-type design effect.
"""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).parent))
from paper_cohort import filter_rows as _panel, keep_teacher as _keep
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Publication settings: vector PDF with editable TrueType text (fonttype 42),
# sans-serif, drawn at final print size (IEEE figure* = 7.16 in) so that the
# LaTeX \includegraphics[width=\textwidth] does not rescale; >=7 pt text.
matplotlib.rcParams.update({
    "savefig.dpi": 600, "figure.dpi": 600,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})
TEXTW = 7.16  # IEEE \textwidth, inches
DISPLAY = {
    "phikon": "Phikon", "phikon-v2": "Phikon-v2", "conch": "CONCH", "uni": "UNI",
    "uni2-h": "UNI2-H", "prov-gigapath": "Prov-GigaPath", "virchow": "Virchow",
    "virchow2": "Virchow2", "h-optimus-0": "H-Optimus-0", "midnight": "Midnight-12k",
    "gpfm": "GPFM",
}
STUD_LABEL = {"vit-tiny": "ViT-Ti", "vit-small": "ViT-S", "vit-base": "ViT-B"}


def _save(fig, name):
    # No bbox_inches="tight": keep the exact print size.
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=600)
    plt.close(fig)


OUT = Path("paper/figures"); OUT.mkdir(parents=True, exist_ok=True)
DELTA = 0.05
# Okabe-Ito colourblind-safe: blue=equivalent, vermillion=inequivalent, grey=inconclusive
GREEN, RED, GREY = "#0072B2", "#D55E00", "#999999"


def state(lo, hi, d=DELTA):
    if hi < d and lo > -d: return "equivalent", GREEN
    if lo > d or hi < -d: return "inequivalent", RED
    return "inconclusive", GREY


def forest_and_sensitivity():
    import math
    res = json.load(open("outputs/v4_full/c16_270/c16_equiv_annotated.json"))
    rows = sorted(_panel(res["rows"]), key=lambda r: r["linear_auc_d"])
    # Margin sensitivity is recomputed from the panel rows rather than read
    # from delta_sensitivity.json, which was written before the panel filter.
    # Both counts follow make_macros.py exactly: the naive (patch-level)
    # interval is d +/- (cluster half-width / inflation); rows without a finite
    # inflation are skipped for both counts.
    grid = [round(0.005 * k, 3) for k in range(1, 31)]

    def counts(margin):
        ne = ce = 0
        for r in rows:
            lo, hi = r["linear_auc_ci"]
            d, f = r["linear_auc_d"], r.get("linear_auc_inflation")
            if not f or not math.isfinite(f):
                continue
            nh = ((hi - lo) / 2) / f
            ne += state(d - nh, d + nh, margin)[0] == "equivalent"
            ce += state(lo, hi, margin)[0] == "equivalent"
        return ne, ce
    sens = [(d, *counts(d)) for d in grid]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(TEXTW, 4.3), layout="constrained",
                                  gridspec_kw={"width_ratios": [1.9, 1]})
    counts_ = {"equivalent": 0, "inequivalent": 0, "inconclusive": 0}
    for i, r in enumerate(rows):
        lo, hi = r["linear_auc_ci"]; d = r["linear_auc_d"]
        st_, c = state(lo, hi); counts_[st_] += 1
        ax.plot([lo, hi], [i, i], color=c, lw=1.6, solid_capstyle="round")
        ax.plot(d, i, "o", color=c, ms=3)
    ax.axvspan(-DELTA, DELTA, color="#000000", alpha=0.05)
    ax.axvline(-DELTA, ls="--", c="k", lw=0.7); ax.axvline(DELTA, ls="--", c="k", lw=0.7)
    ax.axvline(0, ls=":", c="k", lw=0.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{DISPLAY.get(r['teacher'], r['teacher'])} / "
                        f"{STUD_LABEL.get(r['student'], r['student'])}" for r in rows],
                       fontsize=7)
    ax.tick_params(axis="y", length=2, pad=1.5)
    ax.set_xlabel(r"$\mathrm{AUROC}_S-\mathrm{AUROC}_T$ (90% slide-cluster CI)")
    ax.set_title("(a)", loc="left", fontweight="bold")
    from matplotlib.lines import Line2D
    leg = [Line2D([0],[0],color=GREEN,lw=2.5,label=f"equivalent ({counts_['equivalent']})"),
           Line2D([0],[0],color=RED,lw=2.5,label=f"inequivalent ({counts_['inequivalent']})"),
           Line2D([0],[0],color=GREY,lw=2.5,label=f"inconclusive ({counts_['inconclusive']})")]
    ax.legend(handles=leg, loc="lower right", frameon=True, framealpha=0.95)
    ax.set_ylim(-0.7, len(rows) - 0.3)

    xs = [p[0] for p in sens]; ne = [p[1] for p in sens]; ce = [p[2] for p in sens]
    ax2.plot(xs, ne, "-s", color="#E69F00", ms=2.2, lw=1.0,
             label="naive (patch-level) SE")
    ax2.plot(xs, ce, "-o", color=GREEN, ms=2.2, lw=1.0,
             label="slide-cluster SE")
    ax2.axvline(DELTA, ls="--", c="k", lw=0.7)
    n_ne, n_ce = counts(DELTA)
    ax2.annotate(f"$\\delta=0.05$:\nnaive {n_ne}/{len(rows)}\ncluster {n_ce}/{len(rows)}",
                 xy=(DELTA, n_ce), xytext=(0.075, n_ce * 0.7), fontsize=7,
                 arrowprops=dict(arrowstyle="->", lw=0.6))
    ax2.set_xlabel(r"equivalence margin $\delta$ (AUROC)")
    ax2.set_ylabel(f"pairs certified equivalent (of {len(rows)})")
    ax2.set_ylim(0, len(rows) + 1); ax2.set_xlim(0, max(xs) + 0.005)
    ax2.set_title("(b)", loc="left", fontweight="bold")
    ax2.legend(loc="lower right", frameon=True, framealpha=0.95)
    ax2.grid(alpha=0.3, lw=0.4)
    _save(fig, "fig_tost_forest")
    print("[wrote] fig_tost_forest.pdf  states:", counts_,
          " naive/cluster at", {d: (a_, b_) for d, a_, b_ in sens if d in (0.01, 0.02, 0.03, 0.05)})


def inflation_curve():
    """Two panels, both read from result files rather than typed in.

    (a) the slide-count sweep on the HEADLINE test half
        (slide_count_sweep_headline.json, scripts/sweep_slide_count_headline.py):
        whole slides are subsampled from the headline test half with the
        tumour-bearing ratio kept and m fixed, so the full-size point equals the
        headline inflation. There is deliberately no fallback to the older
        slide_count_sweep.json, which drew its own split.
    (b) the two cohorts against the mean-type design effect evaluated at the
        MEASURED intra-slide correlation.
    """
    import statistics as st
    sweep_path = Path("outputs/v4_full/c16_270/slide_count_sweep_headline.json")
    if not sweep_path.exists():
        raise FileNotFoundError(
            f"{sweep_path} not found: run scripts/sweep_slide_count_headline.py first "
            "(no fallback to slide_count_sweep.json)")
    sweep = json.load(open(sweep_path))
    sw_rows = _panel(sweep["rows"])
    icc_new = json.load(open("outputs/v4_full/c16_270/intra_slide_icc.json"))
    rows = _panel(json.load(open("outputs/v4_full/c16_270/c16_equiv_annotated.json"))["rows"])
    kather = _panel(json.load(open("outputs/v4_full/kather_msi_equiv.json"))["rows"])
    n_kat = icc_new["kather"]["n_clusters"]
    n_c16 = icc_new["c16"]["n_clusters"]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(TEXTW, 2.7), layout="constrained")

    sizes = sorted({r["n_test_slides"] for r in sw_rows})
    med, lo, hi = [], [], []
    for n in sizes:
        v = sorted(r["inflation"] for r in sw_rows if r["n_test_slides"] == n)
        q = st.quantiles(v, n=4)
        med.append(st.median(v)); lo.append(q[0]); hi.append(q[2])
    axa.fill_between(sizes, lo, hi, color="#d62728", alpha=0.15,
                     label=r"IQR over pairs $\times$ replicates")
    axa.plot(sizes, med, "-o", color="#d62728", ms=4, lw=1.3, label="median")
    for x, y in zip(sizes, med):
        axa.annotate(f"{y:.1f}$\\times$", (x, y), textcoords="offset points",
                     xytext=(0, 6), ha="center", fontsize=7)
    axa.axhline(1.0, ls=":", c="k", lw=0.7)
    axa.text(max(sizes) * 1.05, 1.0, "naive SE (no inflation)", fontsize=7,
             va="bottom", ha="right")
    axa.set_xlabel("number of test slides (clusters)")
    axa.set_ylabel(r"slide-cluster / naive SE ($\times$)")
    axa.set_ylim(0, max(hi) * 1.4)
    axa.set_xlim(0, max(sizes) * 1.08)
    axa.set_xticks(sizes)
    axa.set_title("(a)", loc="left", fontweight="bold")
    axa.legend(loc="upper center", ncol=2, frameon=True, framealpha=0.95); axa.grid(alpha=0.3, lw=0.4)

    c16_inf = st.median([r["linear_auc_inflation"] for r in rows
                         if r.get("linear_auc_inflation")])
    kat_inf = st.median([r["linear_inflation"] for r in kather
                         if r.get("linear_inflation")])
    coh = [(icc_new["kather"]["m"], kat_inf, icc_new["kather"]["median_rho"],
            f"Kather-MSI ({n_kat} patients)", (8, -12), "left"),
           (icc_new["c16"]["m"], c16_inf, icc_new["c16"]["median_rho"],
            f"CAMELYON16 ({n_c16} slides)", (-6, 16), "right")]
    for x, y, rho, lab, off, ha in coh:
        axb.plot(x, y, "o", color="#1f77b4", ms=5, zorder=3)
        axb.annotate(f"{lab}\n{y:.1f}$\\times$ (measured $\\rho$={rho:.2f})",
                     (x, y), textcoords="offset points", xytext=off,
                     fontsize=7, va="center", ha=ha)
    mm = np.linspace(20, 340, 120)
    for rho, ls in ((icc_new["c16"]["median_rho"], "--"),
                    (icc_new["kather"]["median_rho"], ":")):
        axb.plot(mm, np.sqrt(1 + (mm - 1) * rho), ls, color="#777777", lw=1.0,
                 label=rf"$\sqrt{{1+(m-1)\rho}}$, $\rho$={rho:.2f}")
    axb.set_xlabel("tiles per cluster $m$")
    axb.set_ylabel(r"SE inflation ($\times$)")
    axb.set_ylim(0, 12); axb.set_xlim(0, 350)
    axb.legend(loc="upper left", frameon=True, framealpha=0.95)
    axb.set_title("(b)", loc="left", fontweight="bold")
    axb.grid(alpha=0.3, lw=0.4)

    _save(fig, "fig_se_inflation")
    print(f"[wrote] fig_se_inflation.pdf  (headline sweep {sizes}, medians "
          f"{[round(v,2) for v in med]}; headline C16 {c16_inf:.2f}, Kather {kat_inf:.2f})")


if __name__ == "__main__":
    forest_and_sensitivity()
    inflation_curve()
