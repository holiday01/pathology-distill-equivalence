#!/usr/bin/env python3
"""Supporting figures from available atlas data, drawn at final print size
(single IEEE column, 3.5 in wide; no scaling in LaTeX):
  fig_cka.pdf          — teacher x student linear CKA heatmap (10 teachers x 3
                         students), pooled over each run's held-out test split
                         of the six-source distillation pool (cka_overall).
  fig_calibration.pdf  — teacher vs student ECE (a) and AURC (b) on the
                         lesion-annotated CAMELYON16 linear probe.
  fig_energy.pdf       — training energy (kWh) vs student C16 AUROC, colour =
                         teacher, marker = student size, Pareto frontier as a
                         step line with the Pareto points highlighted.
Also prints the CKA narrative numbers (range, ViT-B - ViT-Ti gap).
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).parent))
from paper_cohort import filter_rows as _panel, keep_teacher as _keep
import json, glob, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter

matplotlib.rcParams.update({
    "savefig.dpi": 600, "figure.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})
OUT = Path("paper/figures")
COLW = 3.5  # IEEE \columnwidth, inches
STUD = ["vit-tiny", "vit-small", "vit-base"]
STUD_LABEL = {"vit-tiny": "ViT-Ti", "vit-small": "ViT-S", "vit-base": "ViT-B"}
HIBOU = {"hibou-b", "hibou-l"}
DISPLAY = {
    "phikon": "Phikon", "phikon-v2": "Phikon-v2", "conch": "CONCH", "uni": "UNI",
    "uni2-h": "UNI2-H", "prov-gigapath": "Prov-GigaPath", "virchow": "Virchow",
    "virchow2": "Virchow2", "h-optimus-0": "H-Optimus-0", "midnight": "Midnight-12k",
    "gpfm": "GPFM",
}
# Fixed teacher order (and colour) shared by every panel of this script.
ORDER = ["phikon", "phikon-v2", "conch", "uni", "uni2-h", "prov-gigapath",
         "virchow", "virchow2", "h-optimus-0", "midnight"]
# 10-colour qualitative palette (matplotlib tab10)
TCOL = dict(zip(ORDER, plt.get_cmap("tab10").colors))


def disp(t):
    return DISPLAY.get(t, t)


def save(fig, name):
    # No bbox_inches="tight": the PDF must keep the exact figsize so that
    # \includegraphics[width=\columnwidth] does not rescale it.
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=600)
    plt.close(fig)


def load():
    d = {}
    for f in sorted(glob.glob("outputs/v4_full/*/*/downstream_report.json")):
        p = f.split("/"); t, s = p[-3], p[-2]
        r = json.load(open(f)).get("results", {})
        if not _keep(t):
            continue
        d[(t, s)] = {"cka": r.get("feature_similarity", {}).get("cka_overall")}
    return d


def cka_heatmap(d):
    present = {t for t, _ in d}
    teachers = [t for t in ORDER if t in present] + sorted(present - set(ORDER))
    M = np.full((len(teachers), 3), np.nan)
    for i, t in enumerate(teachers):
        for j, s in enumerate(STUD):
            v = d.get((t, s), {}).get("cka")
            if v is not None: M[i, j] = v
    fig, ax = plt.subplots(figsize=(COLW, 3.3), layout="constrained")
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0.7, vmax=1.0)
    ax.set_xticks(range(3)); ax.set_xticklabels([STUD_LABEL[s] for s in STUD])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(teachers))); ax.set_yticklabels([disp(t) for t in teachers])
    ax.tick_params(length=0)
    for i in range(len(teachers)):
        for j in range(3):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] < 0.865 else "black", fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03)
    cb.set_label("linear CKA (teacher vs student)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    save(fig, "fig_cka")
    vals = [d[k]["cka"] for k in d if d[k]["cka"] is not None]
    gaps = []
    for t in teachers:
        a, b = d.get((t, "vit-base"), {}).get("cka"), d.get((t, "vit-tiny"), {}).get("cka")
        if a is not None and b is not None: gaps.append(a - b)
    print(f"[CKA] n={len(vals)} range [{min(vals):.3f}, {max(vals):.3f}]; "
          f"median ΔCKA(B-Ti)={np.median(gaps):.3f} IQR[{np.quantile(gaps,.25):.3f},{np.quantile(gaps,.75):.3f}] "
          f"over {len(gaps)} teachers")


def calibration(d=None):
    # Lesion-annotated CAMELYON16 probe on the full-cohort test half.
    res = json.load(open("outputs/v4_full/c16_270/c16_calibration_annotated.json"))
    rows = _panel(res["rows"])
    te = np.array([r["linear_t_ece"] for r in rows]); se = np.array([r["linear_s_ece"] for r in rows])
    ta = np.array([r["linear_t_aurc"] for r in rows]); sa = np.array([r["linear_s_aurc"] for r in rows])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COLW, 1.95), layout="constrained")
    for ax, x, y, lbl, letter in ((a1, te, se, "ECE", "a"), (a2, ta, sa, "AURC", "b")):
        lo = 0.0; hi = max(x.max(), y.max()) * 1.08
        ax.plot([lo, hi], [lo, hi], "k:", lw=0.7)
        ax.scatter(x, y, s=10, color="#0072B2", alpha=0.75, edgecolor="none")
        if lbl == "ECE":
            ax.axhline(0.05, color="#D55E00", lw=0.6, ls="--")
            ax.axvline(0.05, color="#D55E00", lw=0.6, ls="--")
            fmt = FuncFormatter(lambda v, _: f"{v:g}")
            unit = ""
        else:
            # AURC values are ~1e-3: plot on a x10^3 scale so ticks stay short.
            fmt = FuncFormatter(lambda v, _: f"{v*1e3:.0f}")
            unit = r" ($\times10^{-3}$)"
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(MaxNLocator(4))
            axis.set_major_formatter(fmt)
        ax.set_xlabel(f"teacher {lbl}{unit}"); ax.set_ylabel(f"student {lbl}{unit}")
        ax.set_title(f"({letter})", loc="left", fontsize=8.5, fontweight="bold")
        ax.grid(alpha=0.3, lw=0.4)
    save(fig, "fig_calibration")
    print(f"[calib] n={len(rows)} (annotated panel): median teacher ECE {np.median(te):.3f}, "
          f"student ECE {np.median(se):.3f}; students below 0.05: {int((se<=0.05).sum())}/{len(rows)}; "
          f"AURC student worse {int((sa>ta).sum())}/{len(rows)}")


def energy():
    pm = {r["run_id"]: r for r in csv.DictReader(open("outputs/v4_full/paper_metrics.csv"))
          if _keep(r["fm"])}
    eq = {(r["teacher"], r["student"]): r for r in
          _panel(json.load(open("outputs/v4_full/c16_270/c16_equiv_annotated.json"))["rows"])}
    pts = []
    for rid, r in pm.items():
        try:
            wh = float(r["energy_wh"]); fm, s = r["fm"], r["student"]
        except (KeyError, ValueError):
            continue
        auc = eq.get((fm, s), {}).get("linear_s_auc")
        if auc is None or wh <= 0: continue
        pts.append((wh / 1000.0, auc, fm, s))
    smark = {"vit-tiny": "o", "vit-small": "s", "vit-base": "^"}
    fig, ax = plt.subplots(figsize=(COLW, 3.6), layout="constrained")
    # Pareto frontier (low energy, high AUROC) -- same rule as make_macros.pareto
    pts.sort(key=lambda p: p[0])
    front, best = [], -1.0
    for p in pts:
        if p[1] > best:
            front.append(p); best = p[1]
    fx = [p[0] for p in front] + [max(p[0] for p in pts) * 1.03]
    fy = [p[1] for p in front] + [front[-1][1]]
    ax.step(fx, fy, where="post", color="#444444", lw=0.9, ls="--", zorder=1)
    for kwh, auc, fm, s in pts:
        ax.scatter(kwh, auc, marker=smark[s], s=20, color=TCOL.get(fm, "k"),
                   edgecolor="none", alpha=0.9, zorder=3)
    ax.scatter([p[0] for p in front], [p[1] for p in front], s=70, facecolor="none",
               edgecolor="k", lw=0.9, zorder=4)
    ax.set_xlabel("training energy (kWh, idle-subtracted)")
    ax.set_ylabel("student CAMELYON16 tile AUROC")
    ax.grid(alpha=0.3, lw=0.4)
    present = [t for t in ORDER if any(p[2] == t for p in pts)]
    th = [Line2D([0], [0], marker="o", ls="", color=TCOL[t], ms=4.5, label=disp(t))
          for t in present]
    sh = [Line2D([0], [0], marker=smark[s], ls="", color="#555555", ms=4.5,
                 label=STUD_LABEL[s]) for s in STUD]
    sh.append(Line2D([0], [0], marker="o", ls="--", color="#444444", mfc="none",
                     mec="k", ms=7, lw=0.9, label="Pareto"))
    # One legend below the axes: two rows of teachers (colour), then a row of
    # student sizes (marker) and the Pareto key. Legend fills column-major, so
    # the handles are interleaved per column.
    blank = Line2D([0], [0], ls="", marker="", label=" ")
    sh = sh + [blank] * (4 - len(sh))
    th = th + [blank] * (12 - len(th))
    hs = []
    for c in range(4):
        hs += [th[c], th[c + 4], th[c + 8], sh[c]]
    fig.legend(handles=hs, loc="outside lower center", ncol=4, frameon=False,
               handletextpad=0.2, columnspacing=0.8, borderaxespad=0.2,
               fontsize=7, handlelength=1.5)
    save(fig, "fig_energy")
    print(f"[energy] {len(pts)} runs plotted; Pareto points: "
          + ", ".join(f"{disp(p[2])}x{STUD_LABEL[p[3]]} ({p[0]:.2f} kWh, {p[1]:.3f})" for p in front))


if __name__ == "__main__":
    d = load()
    cka_heatmap(d)
    calibration(d)
    energy()
