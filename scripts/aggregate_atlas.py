#!/usr/bin/env python3
"""Aggregate atlas-level results across all (teacher × student) runs.

Inputs (per run dir <root>/<fm>/<student>/):
  training_log.json            — from distill_v4.py (always present once trained)
  downstream_report.json       — from evaluate_v4_downstream.py
  pannuke_seg/pannuke_seg_report.json — from evaluate_pannuke_seg.py
  + outputs/plism/results/<teacher>.json — per-teacher PLISM result (teacher-level only)

Outputs (atlas-level, written to <root>/):
  atlas_summary.csv  — flat row per (teacher, student); paper-friendly
  atlas_summary.json — nested form preserving structure (figure scripts read this)
  atlas_summary.md   — short markdown digest

Designed to be defensive: any missing report -> row still emitted with NaN cells.
Re-runnable; idempotent; no side effects beyond writing the three summary files.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe(d: Optional[dict], *keys, default=None):
    """Walk nested dict, returning default if any step is missing."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"  [warn] cannot parse {path}: {e}")
        return None


def _flatten_probe(probe: Optional[dict], prefix: str) -> Dict[str, Any]:
    """Flatten one probe (c16_linear / c16_mlp / kather_msi_linear) into csv columns."""
    if probe is None:
        return {}
    out = {
        f"{prefix}_T_acc": probe.get("teacher_acc"),
        f"{prefix}_S_acc": probe.get("student_acc"),
        f"{prefix}_T_auc": probe.get("teacher_auc"),
        f"{prefix}_S_auc": probe.get("student_auc"),
        f"{prefix}_T_acc_ci_lo": _safe(probe, "teacher_acc_ci95_cluster", default=[None, None])[0],
        f"{prefix}_T_acc_ci_hi": _safe(probe, "teacher_acc_ci95_cluster", default=[None, None])[1],
        f"{prefix}_S_acc_ci_lo": _safe(probe, "student_acc_ci95_cluster", default=[None, None])[0],
        f"{prefix}_S_acc_ci_hi": _safe(probe, "student_acc_ci95_cluster", default=[None, None])[1],
        f"{prefix}_d_mean":   _safe(probe, "tost", "d_mean"),
        f"{prefix}_d_ci90_lo":_safe(probe, "tost", "d_ci90_lo"),
        f"{prefix}_d_ci90_hi":_safe(probe, "tost", "d_ci90_hi"),
        f"{prefix}_p_lower":  _safe(probe, "tost", "p_lower"),
        f"{prefix}_p_upper":  _safe(probe, "tost", "p_upper"),
        f"{prefix}_equiv":    _safe(probe, "tost", "equivalent"),
        f"{prefix}_n_slides": _safe(probe, "tost", "n_slides"),
        f"{prefix}_kappa":    probe.get("cohen_kappa"),
        f"{prefix}_T_ece":    _safe(probe, "calibration", "teacher", "ece"),
        f"{prefix}_S_ece":    _safe(probe, "calibration", "student", "ece"),
        f"{prefix}_T_mce":    _safe(probe, "calibration", "teacher", "mce"),
        f"{prefix}_S_mce":    _safe(probe, "calibration", "student", "mce"),
        f"{prefix}_T_aurc":   _safe(probe, "calibration", "teacher", "aurc"),
        f"{prefix}_S_aurc":   _safe(probe, "calibration", "student", "aurc"),
        f"{prefix}_T_risk09": _safe(probe, "calibration", "teacher", "risk_at_cov_0.9"),
        f"{prefix}_S_risk09": _safe(probe, "calibration", "student", "risk_at_cov_0.9"),
    }
    return out


def _flatten_pannuke(report: Optional[dict]) -> Dict[str, Any]:
    if report is None:
        return {}
    out = {
        "seg_mDice_T": _safe(report, "mDice_fg", "teacher"),
        "seg_mDice_S": _safe(report, "mDice_fg", "student"),
        "seg_mDice_d": _safe(report, "mDice_fg", "delta"),
    }
    per_class = report.get("per_class") or {}
    for name, c in per_class.items():
        out[f"seg_{name}_T"]     = c.get("teacher_dice")
        out[f"seg_{name}_S"]     = c.get("student_dice")
        out[f"seg_{name}_d"]     = c.get("delta_mean")
        out[f"seg_{name}_equiv"] = c.get("tost_equivalent")
    return out


def _flatten_plism(plism_per_teacher: Optional[dict]) -> Dict[str, Any]:
    """plismbench output structure depends on its CLI. Common keys we look for:
       cross_scanner_top1, cross_staining_top1, both_top1 (may be absent)."""
    if plism_per_teacher is None:
        return {}
    candidates = ["cross_scanner", "cross_staining", "both", "overall"]
    out: Dict[str, Any] = {}
    for c in candidates:
        cell = plism_per_teacher.get(c)
        if isinstance(cell, dict):
            for k in ["top1", "top3", "top5", "top10"]:
                if k in cell:
                    out[f"plism_{c}_{k}"] = cell[k]
        elif isinstance(cell, (int, float)):
            out[f"plism_{c}"] = cell
    return out


def _load_plism(plism_root: Path) -> Dict[str, dict]:
    """Map teacher_name -> parsed result dict (best-effort)."""
    out: Dict[str, dict] = {}
    if not plism_root.exists():
        return out
    for p in plism_root.glob("*.json"):
        teacher = p.stem
        d = _load_json(p)
        if d:
            out[teacher] = d
    return out


def collect_row(fm_dir: Path, stu: str, plism_for_fm: Optional[dict]) -> Optional[dict]:
    stu_dir = fm_dir / stu
    if not stu_dir.is_dir():
        return None
    row: Dict[str, Any] = {"teacher": fm_dir.name, "student": stu}

    tlog = _load_json(stu_dir / "training_log.json")
    if tlog:
        row["best_val"]          = tlog.get("best_val")
        row["best_epoch"]        = tlog.get("best_epoch")
        row["teacher_params"]    = tlog.get("teacher_params")
        row["student_params"]    = tlog.get("student_params")
        row["compression"]       = tlog.get("compression_ratio")
        history = tlog.get("history") or []
        row["epochs_run"]        = len(history)
        if history:
            ep_seconds = [h.get("elapsed_epoch_s") for h in history if h.get("elapsed_epoch_s")]
            if ep_seconds:
                row["per_epoch_s"]      = sum(ep_seconds) / len(ep_seconds)
            total_s = (history[-1].get("elapsed_total_s")
                       or history[-1].get("elapsed"))
            if total_s:
                row["train_total_h"] = float(total_s) / 3600.0
            vrams = [h.get("peak_vram_train_mb") for h in history if h.get("peak_vram_train_mb")]
            if vrams:
                row["peak_vram_mb"] = max(vrams)
    else:
        row["epochs_run"] = 0

    row["has_best_pt"] = (stu_dir / "best.pt").exists()

    ds = _load_json(stu_dir / "downstream_report.json")
    if ds:
        results = ds.get("results", {})
        row["cka_overall"]    = _safe(results, "feature_similarity", "cka_overall")
        row["cosine_overall"] = _safe(results, "feature_similarity", "cosine_overall")
        probes = results.get("probes", {})
        row.update(_flatten_probe(probes.get("c16_linear"),       "c16lin"))
        row.update(_flatten_probe(probes.get("c16_mlp"),          "c16mlp"))
        row.update(_flatten_probe(probes.get("kather_msi_linear"),"msilin"))
        center = results.get("medical_center_probe", {}) or {}
        row["center_T_auc"] = _safe(center, "teacher", "center_probe_auc")
        row["center_S_auc"] = _safe(center, "student", "center_probe_auc")
        row["center_T_RI"]  = _safe(center, "teacher", "robustness_index")
        row["center_S_RI"]  = _safe(center, "student", "robustness_index")
        eff = results.get("efficiency", {}) or {}
        row["latency_T_ms"] = eff.get("teacher_ms")
        row["latency_S_ms"] = eff.get("student_ms")
        row["speedup"]      = eff.get("speedup")

    seg = _load_json(stu_dir / "pannuke_seg" / "pannuke_seg_report.json")
    row.update(_flatten_pannuke(seg))

    # PLISM is a teacher-level result; copy into both student rows for convenience
    row.update(_flatten_plism(plism_for_fm))

    # Stash structured forms (used by atlas_summary.json + figure scripts)
    row["_struct"] = {
        "training_log": tlog,
        "downstream":   ds,
        "pannuke":      seg,
        "plism":        plism_for_fm,
    }
    return row


def write_csv(rows: List[dict], path: Path) -> None:
    """Flat CSV (excludes the _struct field)."""
    if not rows:
        path.write_text("")
        return
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k == "_struct" or k in seen:
                continue
            keys.append(k); seen.add(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def write_json(rows: List[dict], path: Path) -> None:
    payload = {
        "rows": [{k: v for k, v in r.items() if k != "_struct"} for r in rows],
        "structured": {f"{r['teacher']}__{r['student']}": r["_struct"] for r in rows},
    }
    path.write_text(json.dumps(payload, indent=2, default=lambda x: None
                               if (isinstance(x, float) and math.isnan(x)) else str(x)))


def write_md(rows: List[dict], path: Path) -> None:
    lines = ["# Atlas Summary", "",
             f"_{len(rows)} rows. NaN = report not yet produced._", ""]
    cols = [("teacher", 16), ("student", 10), ("epochs_run", 6),
            ("best_val", 8), ("cka_overall", 8),
            ("c16lin_d_mean", 9), ("c16lin_equiv", 6),
            ("seg_mDice_d", 10), ("speedup", 8)]
    header = "| " + " | ".join(c[0] for c in cols) + " |"
    sep    = "|" + "|".join("-" * (c[1] + 2) for c in cols) + "|"
    lines += [header, sep]
    for r in rows:
        cells = []
        for name, _w in cols:
            v = r.get(name)
            if v is None:
                cells.append("·")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}" if abs(v) < 1000 else f"{v:.2e}")
            elif isinstance(v, bool):
                cells.append("Y" if v else "N")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/v4_full",
                    help="atlas root (contains <fm>/<student>/ subdirs)")
    ap.add_argument("--plism-root", default="outputs/plism/results")
    ap.add_argument("--students", nargs="+",
                    default=["vit-tiny", "vit-small", "vit-base"])
    args = ap.parse_args()

    root = Path(args.root)
    plism = _load_plism(Path(args.plism_root))

    rows: List[dict] = []
    for fm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if fm_dir.name in {"telemetry"}:
            continue
        for stu in args.students:
            r = collect_row(fm_dir, stu, plism.get(fm_dir.name))
            if r:
                rows.append(r)

    out_dir = root
    write_csv(rows, out_dir / "atlas_summary.csv")
    write_json(rows, out_dir / "atlas_summary.json")
    write_md(rows, out_dir / "atlas_summary.md")
    n_eval = sum(1 for r in rows if r.get("c16lin_d_mean") is not None)
    n_seg  = sum(1 for r in rows if r.get("seg_mDice_d") is not None)
    print(f"[aggregate_atlas] rows={len(rows)} downstream={n_eval} pannuke={n_seg} "
          f"plism_teachers={len(plism)}")
    print(f"  -> {out_dir / 'atlas_summary.csv'}")
    print(f"  -> {out_dir / 'atlas_summary.json'}")
    print(f"  -> {out_dir / 'atlas_summary.md'}")


if __name__ == "__main__":
    main()
