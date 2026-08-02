#!/usr/bin/env python3
"""Aggregate per-FM distillation runs into a single comparison table."""
import argparse
import json
import math
from pathlib import Path


def _num(x):
    if x is None: return None
    try:
        f = float(x)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def collect(root: Path):
    rows = []
    for fm_dir in sorted(root.iterdir()):
        if not fm_dir.is_dir():
            continue
        fm = fm_dir.name
        eval_path = fm_dir / "eval" / "evaluation_report.json"
        distill_log = fm_dir / "distill" / "training_log.json"
        row = {"fm": fm}
        if eval_path.exists():
            with open(eval_path) as f:
                ev = json.load(f)
            r = ev["results"]
            row["cka"]       = _num(r["feature_similarity"]["cka_overall"])
            row["cos"]       = _num(r["feature_similarity"]["cosine_overall"])
            lp = r.get("linear_probing", {}).get("c16_tumor_vs_normal", {})
            row["teacher_acc"] = _num(lp.get("teacher_acc"))
            row["student_acc"] = _num(lp.get("student_acc"))
            row["d_acc"]     = _num(lp.get("acc_gap"))
            row["kappa"]     = _num(lp.get("cohen_kappa"))
            ef = r["efficiency"]
            row["teacher_ms"] = _num(ef["teacher_ms"])
            row["student_ms"] = _num(ef["student_ms"])
            row["speedup"]    = _num(ef["speedup"])
            row["teacher_M"]  = ef["teacher_params"] / 1e6
            row["compression"] = _num(ef["compression"])
            row["equivalent"] = r["verdict"]["equivalent"]
        if distill_log.exists():
            with open(distill_log) as f:
                tr = json.load(f)
            row["best_val"]    = _num(tr.get("best_val"))
            row["best_epoch"]  = tr.get("best_epoch")
            row["epochs_run"]  = len(tr.get("history", []))
        rows.append(row)
    return rows


def fmt(x, p=3, pct=False):
    if x is None: return "  -  "
    if pct: return f"{x*100:5.1f}%"
    return f"{x:.{p}f}"


def print_table(rows):
    # Rank by CKA descending (primary), d_acc ascending (tiebreak)
    scored = [r for r in rows if r.get("cka") is not None]
    scored.sort(key=lambda r: (-(r["cka"] or 0), r.get("d_acc") or 1))
    print()
    h = f"{'rank':>4} {'fm':<15} {'teacher':>8} {'comp':>5} {'CKA':>6} {'ΔACC':>7} {'κ':>6} {'spdup':>6} {'best_val':>9} {'ep':>3} {'eq':>3}"
    print(h)
    print("-" * len(h))
    for i, r in enumerate(scored, 1):
        eq = "✓" if r.get("equivalent") else "✗"
        print(f"{i:>4} {r['fm']:<15} "
              f"{fmt(r.get('teacher_M'),1):>7}M "
              f"{fmt(r.get('compression'),1):>5} "
              f"{fmt(r.get('cka')):>6} "
              f"{fmt(r.get('d_acc'), pct=True):>7} "
              f"{fmt(r.get('kappa')):>6} "
              f"{fmt(r.get('speedup'),2):>5}x "
              f"{fmt(r.get('best_val'),4):>9} "
              f"{(r.get('epochs_run') or 0):>3} "
              f"{eq:>3}")
    # Missing FMs
    missing = [r for r in rows if r.get("cka") is None]
    if missing:
        print()
        print("[pending/failed]")
        for r in missing:
            print(f"  {r['fm']:<15} epochs_run={r.get('epochs_run','-')}  best_val={r.get('best_val','-')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/fm_comparison")
    ap.add_argument("--json", action="store_true", help="also write summary.json")
    args = ap.parse_args()
    root = Path(args.root)
    rows = collect(root)
    print_table(rows)
    with open(root / "summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n[saved] {root / 'summary.json'}")


if __name__ == "__main__":
    main()
