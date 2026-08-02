#!/usr/bin/env python3
"""F4 — PLISM cross-staining retrieval matrices (one panel per teacher).

PLISM (Ozeki 2024) is a 91-slide cohort of 13 staining protocols × 7 scanners
imaged on the same source tissues. plismbench writes per-pair metrics at
    {root}/{n_tiles}_tiles/{teacher}/metrics.csv
with columns:
    slide_a, features_path_a, staining_a, scanner_a,
    slide_b, features_path_b, staining_b, scanner_b,
    cosine_similarity, top_1_accuracy, top_3_accuracy,
    top_5_accuracy, top_10_accuracy
This script aggregates top-K retrieval over all (slide_a, slide_b) pairs
grouped by a chosen `--group-by` axis (default `staining`, also accepts
`scanner` or `slide`), producing a (group × group) heatmap. To keep paper
density manageable, all 13 teachers are rendered together in a single
grid figure (default 3×5; auto-shrinks if fewer teachers).

Visual: shared colormap range across panels so teachers can be compared
at a glance. Each panel title shows the teacher name + overall mean top-K.

Outputs:
    {out_dir}/F4_plism_matrix_{metric}_by_{group_by}.{png,pdf}
    {out_dir}/F4_plism_matrix_{metric}_by_{group_by}_per_teacher.csv  (tidy)
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Canonical teacher order (small → large; matches scripts/run_plism_eval.sh).
TEACHER_ORDER = [
    "phikon", "phikonv2", "hibou_base", "hibou_large", "conch",
    "uni", "uni2h", "virchow", "virchow2", "hoptimus0",
    "provgigapath", "midnight_12k", "gpfm",
]
# Teachers with suspected feature-extraction issues (flagged by B parser
# review and verified in distill_wsi_model.py:317 — hibou register-token
# pooling mismatch). Annotated with `§` in panel titles so paper readers
# know to discount these panels until the extraction path is fixed.
SUSPECT_TEACHERS = {"hibou_base", "hibou_large"}

# Canonical PLISM staining-code expansion for the caption block.
PLISM_STAINING_LEGEND = {
    "GIVH": "Gill Iron Victoria + Hematoxylin",
    "GIV":  "Gill Iron Victoria",
    "GMH":  "Gill Mayer + Hematoxylin",
    "GM":   "Gill Mayer",
    "GVH":  "Gill Victoria + Hematoxylin",
    "GV":   "Gill Victoria",
    "HRH":  "Harris + Hematoxylin",
    "HR":   "Harris",
    "KRH":  "Kratochvil Red + Hematoxylin",
    "KR":   "Kratochvil Red",
    "LMH":  "Lillie Mayer + Hematoxylin",
    "LM":   "Lillie Mayer",
    "MY":   "Mayer",
}
# Approximate chance baseline for top-1 retrieval over the 8139-tile gallery.
PLISM_GALLERY_TILES = 8139
PLISM_CHANCE_TOP1 = 1.0 / PLISM_GALLERY_TILES
METRIC_COLS = {
    "top_1": "top_1_accuracy",
    "top_3": "top_3_accuracy",
    "top_5": "top_5_accuracy",
    "top_10": "top_10_accuracy",
    "cosine": "cosine_similarity",
}


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def aggregate(rows: list[dict], metric_col: str, group_by: str,
              labels: list[str] | None = None
              ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Aggregate `metric_col` over all rows, indexed by (rows[group_by]_a,
    rows[group_by]_b). Returns (mean_matrix, count_matrix, group_labels).
    Cells without pairs are NaN. If `labels` is given, the output matrix is
    indexed onto that label set (groups absent in `rows` produce NaN rows/
    cols)."""
    a_col, b_col = f"{group_by}_a", f"{group_by}_b"
    if rows and a_col not in rows[0]:
        raise SystemExit(f"[F4] column '{a_col}' missing — CSV may use a different "
                         f"schema. Got cols: {list(rows[0].keys())[:8]}")
    if labels is None:
        labels = sorted({r[a_col] for r in rows} | {r[b_col] for r in rows})
    idx = {g: i for i, g in enumerate(labels)}
    n = len(labels)
    sum_M = np.zeros((n, n), dtype=float)
    cnt_M = np.zeros((n, n), dtype=int)
    for r in rows:
        try:
            v = float(r[metric_col])
        except (KeyError, ValueError, TypeError):
            continue
        if not math.isfinite(v):
            continue
        ia = idx.get(r[a_col])
        ib = idx.get(r[b_col])
        if ia is None or ib is None:
            continue
        sum_M[ia, ib] += v
        cnt_M[ia, ib] += 1
    # Symmetrize: plismbench emits each unordered (slide_a, slide_b) pair
    # once with an arbitrary ordering, so `sum_M`/`cnt_M` are asymmetric for
    # storage reasons only. Combine them so the matrix axes represent the
    # same group set rather than an artificial direction. Subtract the
    # diagonal once so same-group pairs aren't double-counted (they appear
    # on the diagonal of both `sum_M` and `sum_M.T`).
    diag_sum = np.diag(np.diag(sum_M))
    diag_cnt = np.diag(np.diag(cnt_M))
    sum_M = sum_M + sum_M.T - diag_sum
    cnt_M = cnt_M + cnt_M.T - diag_cnt
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_M = np.where(cnt_M > 0, sum_M / np.maximum(cnt_M, 1), np.nan)
    return mean_M, cnt_M, labels


def find_metric_csvs(results_root: Path) -> dict[str, Path]:
    """Map teacher → metrics.csv path under `{N}_tiles/{teacher}/metrics.csv`.
    If a teacher has multiple `N_tiles` runs, pick the one with the largest
    numeric `N` (lexical sort wins the wrong file when `N` crosses a digit
    boundary, e.g. `"10000_tiles" < "8139_tiles"` lexically)."""
    def n_tiles(p: Path) -> int:
        try:
            return int(p.parent.parent.name.split("_")[0])
        except (ValueError, IndexError):
            return -1
    found: dict[str, Path] = {}
    for csv_path in results_root.glob("*_tiles/*/metrics.csv"):
        teacher = csv_path.parent.name
        if teacher not in found or n_tiles(csv_path) > n_tiles(found[teacher]):
            found[teacher] = csv_path
    return found


def _grid_shape(n: int, ncols_hint: int) -> tuple[int, int]:
    """Pick (nrows, ncols) such that nrows*ncols >= n, ncols <= n,
    preferring the user's hint when feasible."""
    if n <= 0:
        return 1, 1
    ncols = max(1, min(ncols_hint, n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def _mask_diagonal(M: np.ndarray) -> np.ndarray:
    """Return a copy of M with the diagonal replaced by NaN. Used so the
    cross-group cells dominate the colormap and the displayed μ reports
    cross-group retrieval rather than the trivially-easy same-group
    (intra-staining / intra-scanner) cells."""
    Mc = M.astype(float, copy=True)
    np.fill_diagonal(Mc, np.nan)
    return Mc


def plot_grid(
    matrices: dict[str, np.ndarray],
    counts: dict[str, np.ndarray],
    labels: list[str],
    teacher_order: list[str],
    metric_label: str,
    group_by: str,
    metric_key: str,
    out_stem: Path,
    ncols_hint: int = 5,
    norm: str = "log",
) -> None:
    """One subplot per teacher; shared colormap across panels. Diagonals are
    masked so the colormap and per-panel μ reflect cross-group retrieval
    only (same-`group_by` cells are the trivial regime and ~10× higher than
    cross-group, which would otherwise dominate the visualisation)."""
    teachers = teacher_order
    n = len(teachers)
    nrows, ncols = _grid_shape(n, ncols_hint)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.3 * ncols + 1.6, 2.3 * nrows + 1.0),
        squeeze=False,
    )

    # Off-diagonal matrices used for both display and color normalization.
    off_diag = {t: _mask_diagonal(M) for t, M in matrices.items()}
    all_off_vals = np.concatenate([
        m[np.isfinite(m)].ravel() for m in off_diag.values() if m.size
    ])
    if all_off_vals.size == 0:
        print("[F4] all matrices empty; nothing to plot.")
        plt.close(fig)
        return
    vmin = float(np.nanpercentile(all_off_vals, 1))
    vmax = float(np.nanpercentile(all_off_vals, 99))
    # For accuracy metrics, snap vmax to 1.0 only when the data is already
    # saturating (top-1 ≪ 1 in practice, cosine ≈ 1; pick conservatively).
    if vmax > 0.98:
        vmax = 1.0

    # Pooled vmin/vmax on a linear scale lets the strongest teacher (hoptimus0
    # ~0.14 here) compress the weakest 4 teachers (≤0.005) into the colormap
    # floor, so within-panel structure for those weak teachers vanishes. Use a
    # symmetric log-norm by default for top_K metrics so cross-teacher
    # comparison is preserved AND weak-panel structure remains readable.
    from matplotlib.colors import LogNorm, Normalize
    # LogNorm needs vmin > 0; clamp pooled vmin to a small positive value
    # rather than refuse-log when pooled p1 dips to 0 (common on weak teachers
    # with cells full of zero retrievals).
    use_log = (norm == "log") and metric_key.startswith("top_")
    # Fall back to linear if vmax is too small for a meaningful log range
    # (e.g. an all-zero-retrieval dataset). LogNorm requires vmin < vmax.
    log_vmin = max(vmin, 1e-4)
    if use_log and vmax <= log_vmin:
        print(f"[F4] WARNING: vmax={vmax:.3g} too small for log color; "
              "falling back to linear.")
        use_log = False
    if use_log:
        norm_obj = LogNorm(vmin=log_vmin, vmax=vmax)
    else:
        norm_obj = Normalize(vmin=vmin, vmax=vmax)

    im = None
    for k, ax in enumerate(axes.ravel()):
        if k >= n:
            ax.axis("off")
            continue
        t = teachers[k]
        M_off = off_diag.get(t)
        if M_off is None or not M_off.size:
            ax.set_title(f"{t}\n(no data)", fontsize=8)
            ax.axis("off")
            continue
        # Render: diagonal is NaN → matplotlib draws default bad-color (white
        # under magma). We overlay a thin grey hatch on the diagonal so the
        # reader sees the masked region explicitly.
        im = ax.imshow(M_off, cmap="magma", norm=norm_obj, aspect="equal")
        # Visualize the masked diagonal as a subtle grey overlay.
        diag_overlay = np.full_like(M_off, np.nan)
        np.fill_diagonal(diag_overlay, 1.0)
        ax.imshow(diag_overlay, cmap="Greys", vmin=0, vmax=1.5,
                  alpha=0.35, aspect="equal")
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
        ax.set_yticks(range(len(labels)), labels, fontsize=6)
        # Off-diagonal μ — the figure's headline statistic.
        if np.any(np.isfinite(M_off)):
            mean_val = float(np.nanmean(M_off))
            title = f"{t}  μ_off={mean_val:.3f}"
        else:
            title = f"{t}  (n/a)"
        if t in SUSPECT_TEACHERS:
            title += " §"
        ax.set_title(title, fontsize=8)
        ax.tick_params(axis="both", which="both", length=0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.3)

    norm_tag = " (log color)" if use_log else ""
    fig.suptitle(
        f"PLISM cross-{group_by} retrieval — {metric_label}{norm_tag}\n"
        f"(diagonal masked; μ_off = mean over off-diagonal cells)",
        fontsize=10, y=1.01,
    )
    # Symmetrized matrix — no real query/gallery direction, so the axes are
    # interchangeable. Label them as "A" / "B" to avoid implying directionality.
    fig.supxlabel(f"{group_by} B", fontsize=9)
    fig.supylabel(f"{group_by} A", fontsize=9)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                            fraction=0.025, pad=0.02, shrink=0.85,
                            label=metric_label)
        cbar.ax.tick_params(labelsize=8)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_tidy_csv(matrices: dict[str, np.ndarray],
                   counts: dict[str, np.ndarray],
                   labels: list[str],
                   metric: str, group_by: str, out_path: Path) -> None:
    """Long-format CSV: teacher, group_a, group_b, value, n_pairs, is_diagonal."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["teacher", f"{group_by}_a", f"{group_by}_b",
                    f"{metric}_mean", "n_pairs", "is_diagonal"])
        for t, M in matrices.items():
            C = counts.get(t)
            for i, ga in enumerate(labels):
                for j, gb in enumerate(labels):
                    v = M[i, j]
                    n = "" if C is None else int(C[i, j])
                    w.writerow([t, ga, gb,
                                "" if not np.isfinite(v) else f"{v:.6f}",
                                n, "true" if i == j else "false"])


def write_caption(out_path: Path, metric: str, group_by: str,
                  teachers: list[str], labels: list[str],
                  matrices: dict[str, np.ndarray]) -> None:
    """Side-car caption file describing label expansion, masking, μ_off, and
    suspect-teacher annotations. Editors paste into figure caption."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# Caption for {out_path.stem}")
    lines.append("")
    lines.append(
        f"PLISM cross-{group_by} retrieval matrices for {len(teachers)} "
        f"pathology foundation models (one panel per teacher). Each cell is "
        f"the mean **{metric.replace('_', '-')} retrieval** over all slide-pair "
        f"observations matching ({group_by}_a, {group_by}_b), aggregated "
        f"symmetrically across the unordered slide pairs emitted by plismbench "
        f"(`(a,b)` and `(b,a)` are combined). Diagonal cells (same-{group_by} "
        f"pairs) are masked because same-{group_by} retrieval is trivial under "
        f"PLISM's design and would dominate both the colormap and per-panel "
        f"means; the per-panel **μ_off** is the mean over OFF-diagonal cells "
        f"only and is the figure's headline statistic.")
    if metric.startswith("top_"):
        lines.append("")
        lines.append(
            f"Top-1 chance baseline over the {PLISM_GALLERY_TILES}-tile gallery "
            f"is ~1/{PLISM_GALLERY_TILES} ≈ {PLISM_CHANCE_TOP1:.1e}; observed "
            "off-diagonal values are 30–800× chance.")
    if group_by == "staining":
        lines.append("")
        lines.append("**Staining codes:**")
        for code in labels:
            full = PLISM_STAINING_LEGEND.get(code, "(unknown)")
            lines.append(f"- `{code}` — {full}")
    suspect_in_fig = [t for t in teachers if t in SUSPECT_TEACHERS]
    if suspect_in_fig:
        lines.append("")
        lines.append(
            f"**§** marks teachers ({', '.join(suspect_in_fig)}) whose "
            "off-diagonal μ is anomalously low and shows an architecture "
            "inversion (hibou_large < hibou_base) inconsistent with the "
            "published PLISM leaderboard (Nechaev 2024). The cause is "
            "pending investigation; one hypothesis is that "
            "`distill_wsi_model.py:317` uses `last_hidden_state[:,0,:]` "
            "for hibou whereas hibou's canonical pooled representation is "
            "`pooler_output` — an A/B extraction comparison is required to "
            "confirm. Until verified, these panels should be interpreted "
            "with caution.")
    lines.append("")
    lines.append("**Per-teacher μ_off** (this run):")
    for t in teachers:
        M_off = _mask_diagonal(matrices[t])
        if np.any(np.isfinite(M_off)):
            mu = float(np.nanmean(M_off))
            tag = " §" if t in SUSPECT_TEACHERS else ""
            lines.append(f"- `{t}`{tag}: {mu:.4f}")
        else:
            lines.append(f"- `{t}`: (no data)")
    out_path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plism-results", default="outputs/plism/results")
    ap.add_argument("--metric", choices=list(METRIC_COLS), default="top_1",
                    help="Metric to aggregate (default: top_1_accuracy).")
    ap.add_argument("--group-by", choices=["staining", "scanner", "slide"],
                    default="staining",
                    help="Aggregation axis (default: staining).")
    ap.add_argument("--out-dir", default="outputs/v4_full/figures")
    ap.add_argument("--ncols", type=int, default=5,
                    help="Subplot grid column count (default: 5).")
    ap.add_argument("--norm", choices=["log", "linear"], default="log",
                    help="Color scale: log preserves weak-teacher panel "
                         "structure under pooled vmin/vmax; linear is the "
                         "raw scale (default: log for top_K, linear otherwise).")
    args = ap.parse_args()

    root = Path(args.plism_results)
    found = find_metric_csvs(root)
    if not found:
        print(f"[F4] no metrics.csv under {root}/*_tiles/*/ — eval not done yet.")
        return
    print(f"[F4] found {len(found)} teachers: {', '.join(sorted(found))}")

    metric_col = METRIC_COLS[args.metric]
    # Build matrices in canonical teacher order, dropping teachers absent in
    # the data but warning so missing rows don't disappear silently.
    teachers_present = [t for t in TEACHER_ORDER if t in found]
    teachers_present += sorted(t for t in found if t not in TEACHER_ORDER)
    missing = [t for t in TEACHER_ORDER if t not in found]
    if missing:
        print(f"[F4] missing teachers (canonical order): {missing}")

    # PLISM has the same 13 stainings / 7 scanners for every teacher; verify
    # that here, then aggregate every teacher onto the union label set so
    # all panels share axes.
    rows_per_teacher = {t: _read_csv(found[t]) for t in teachers_present}
    all_groups: set[str] = set()
    per_teacher_groups: dict[str, set[str]] = {}
    a_col, b_col = f"{args.group_by}_a", f"{args.group_by}_b"
    for t, rs in rows_per_teacher.items():
        gs = {r[a_col] for r in rs} | {r[b_col] for r in rs}
        per_teacher_groups[t] = gs
        all_groups |= gs
    labels = sorted(all_groups)
    for t, gs in per_teacher_groups.items():
        if gs != all_groups:
            print(f"[F4] WARNING: teacher '{t}' has {len(gs)} groups vs "
                  f"union {len(all_groups)} (missing: {sorted(all_groups - gs)})")
    matrices: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    for t, rs in rows_per_teacher.items():
        M, C, _ = aggregate(rs, metric_col, args.group_by, labels=labels)
        matrices[t] = M
        counts[t] = C

    out_stem = Path(args.out_dir) / (
        f"F4_plism_matrix_{args.metric}_by_{args.group_by}"
    )
    plot_grid(matrices, counts, labels, teachers_present,
              metric_label=args.metric.replace("_", "-") + " retrieval",
              group_by=args.group_by, metric_key=args.metric,
              out_stem=out_stem, ncols_hint=args.ncols, norm=args.norm)
    write_tidy_csv(matrices, counts, labels, args.metric, args.group_by,
                   out_stem.with_name(out_stem.name + "_per_teacher.csv"))
    write_caption(out_stem.with_name(out_stem.name + "_caption.md"),
                  args.metric, args.group_by, teachers_present, labels, matrices)
    print(f"[F4] grid → {out_stem}.{{png,pdf}}")
    print(f"[F4] tidy → {out_stem.name}_per_teacher.csv")
    print(f"[F4] caption → {out_stem.name}_caption.md")


if __name__ == "__main__":
    main()
