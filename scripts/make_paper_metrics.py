#!/usr/bin/env python3
"""Collect per-FM memory/time metrics for paper tables.

Outputs:
  paper_metrics.csv — per-FM row: teacher_params, compression, peak_vram, total_time_min,
                      per_epoch_median_s, total_epochs, gpu_util_p50, power_p50, temp_p95
  paper_metrics.md  — markdown table for manuscript
Sources:
  training_log.json per FM (has timing + vram per epoch after distill_v4.py upgrade)
  telemetry/gpu_timeseries.csv — system-wide 10-s nvidia-smi samples
"""
import argparse
import csv
import json
import statistics
from datetime import datetime
from pathlib import Path


def parse_log_line_legacy(line):
    """Extract epoch info from a log text line of the legacy format:
      'epoch   1 | train 0.54 | val 0.48 ★ | lr 9.97e-05 | 2017s'
    Returns (epoch_elapsed_s, total_elapsed_s) or None.
    """
    import re
    m = re.search(r"epoch\s+(\d+)\s+\|.*?\|\s+(?:ep\s+(\d+)s\s+tot\s+(\d+)s|(\d+)s)", line)
    if not m:
        return None
    ep_idx = int(m.group(1))
    if m.group(2) is not None:
        ep_s = int(m.group(2)); tot_s = int(m.group(3))
    else:
        tot_s = int(m.group(4)); ep_s = None
    return ep_idx, ep_s, tot_s


def collect_fm(fm_dir: Path) -> dict:
    """Collect metrics for one FM directory."""
    row = {"fm": fm_dir.name}
    tl = fm_dir / "training_log.json"
    log_txt = fm_dir / "distill.log"

    if tl.exists():
        with open(tl) as f:
            d = json.load(f)
        row["teacher_params"] = d.get("teacher_params")
        row["student_params"] = d.get("student_params")
        row["compression"] = d.get("compression_ratio")
        row["start_ts"] = d.get("start_ts")
        row["end_ts"] = d.get("end_ts")
        history = d.get("history", [])
        if history:
            # New format: elapsed_epoch_s; Legacy format: only cumulative 'elapsed'
            per_ep_key = "elapsed_epoch_s" if "elapsed_epoch_s" in history[0] else None
            total_key = "elapsed_total_s" if "elapsed_total_s" in history[0] else "elapsed"
            if per_ep_key:
                per_epoch = [h[per_ep_key] for h in history if h.get(per_ep_key)]
            else:
                # Derive per-epoch from cumulative
                cum = [h[total_key] for h in history]
                per_epoch = [cum[0]] + [cum[i] - cum[i - 1] for i in range(1, len(cum))]
            if per_epoch:
                row["per_epoch_median_s"] = statistics.median(per_epoch)
                row["per_epoch_min_s"] = min(per_epoch)
                row["per_epoch_max_s"] = max(per_epoch)
            row["total_time_s"] = history[-1].get(total_key, 0)
            vram_peaks = [h.get("peak_vram_train_mb") for h in history if h.get("peak_vram_train_mb")]
            if vram_peaks:
                row["peak_vram_mb"] = max(vram_peaks)
                row["median_vram_mb"] = statistics.median(vram_peaks)
            row["epochs_run"] = len(history)
            row["best_val"] = d.get("best_val")
            row["best_epoch"] = d.get("best_epoch")

    # Legacy fallback: parse distill.log for per-epoch timing (no VRAM)
    if "total_time_s" not in row and log_txt.exists():
        import re as _re
        eps, totals = [], []
        last_total = 0
        for line in log_txt.read_text().splitlines():
            r = parse_log_line_legacy(line)
            if r:
                idx, ep_s, tot_s = r
                totals.append(tot_s)
                if ep_s is not None:
                    eps.append(ep_s)
                else:
                    eps.append(tot_s - last_total)
                last_total = tot_s
            # Also sniff params line: "teacher=85,798,656  student=21,764,224  compression=3.9x"
            m = _re.search(r"teacher=([\d,]+)\s+student=([\d,]+)\s+compression=([\d.]+)", line)
            if m:
                row["teacher_params"] = int(m.group(1).replace(",", ""))
                row["student_params"] = int(m.group(2).replace(",", ""))
                row["compression"] = float(m.group(3))
        if totals:
            row["total_time_s"] = totals[-1]
            row["epochs_run"] = len(totals)
            row["per_epoch_median_s"] = statistics.median(eps) if eps else None
            row["per_epoch_min_s"] = min(eps) if eps else None
            row["per_epoch_max_s"] = max(eps) if eps else None

    # Sweep-log timestamp sniff (retroactive for pre-upgrade runs):
    # look up "[pilot] <fm>" header in the parent v4_pilot.log for a start_ts,
    # then derive end_ts = start + total_time_s.
    sweep_log = fm_dir.parent.parent / "v4_pilot.log"
    if sweep_log.exists() and not row.get("start_ts") and row.get("total_time_s"):
        import re as _re
        lines = sweep_log.read_text().splitlines()
        for i, ln in enumerate(lines):
            if _re.search(rf"\[pilot\]\s+{_re.escape(fm_dir.name)}\b", ln):
                for nxt in lines[i:i + 4]:
                    m = _re.search(r"started\s+(\S+)", nxt)
                    if m:
                        row["start_ts"] = m.group(1)
                        break
                break
        if row.get("start_ts"):
            try:
                from datetime import timedelta
                st = datetime.fromisoformat(row["start_ts"])
                row["end_ts"] = (st + timedelta(seconds=row["total_time_s"])).isoformat()
            except Exception:
                pass
    return row


def collect_telemetry(csv_path: Path, window_start=None, window_end=None,
                      idle_power_w: float = 50.0) -> dict:
    """Summarize nvidia-smi telemetry within an optional time window.
    Integrates energy (Wh) over the window, subtracting idle baseline.
    """
    if not csv_path.exists():
        return {}
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                ts = datetime.fromisoformat(r["ts"])
            except Exception:
                continue
            if window_start and ts < window_start: continue
            if window_end and ts > window_end: continue
            try:
                rows.append({
                    "ts": ts,
                    "util": float(r["gpu_util_pct"]),
                    "mem": float(r["mem_used_mib"]),
                    "power": float(r["power_w"]),
                    "temp": float(r["temp_c"]),
                    "fan": float(r["fan_pct"]),
                })
            except Exception:
                continue
    if not rows:
        return {}
    def p(vs, q):
        s = sorted(vs); n = len(s); i = max(0, min(n - 1, int(n * q)))
        return s[i]
    utils = [r["util"] for r in rows]
    mems = [r["mem"] for r in rows]
    powers = [r["power"] for r in rows]
    temps = [r["temp"] for r in rows]
    fans = [r["fan"] for r in rows]
    # Trapezoidal integrate power (W) × dt (s) → Joules → Wh
    energy_j_raw = 0.0
    energy_j_idle_sub = 0.0
    for i in range(1, len(rows)):
        dt = (rows[i]["ts"] - rows[i - 1]["ts"]).total_seconds()
        p_avg = 0.5 * (rows[i]["power"] + rows[i - 1]["power"])
        energy_j_raw += p_avg * dt
        energy_j_idle_sub += max(0.0, p_avg - idle_power_w) * dt
    return {
        "gpu_util_p50": statistics.median(utils),
        "gpu_util_p95": p(utils, 0.95),
        "power_w_p50": statistics.median(powers),
        "power_w_p95": p(powers, 0.95),
        "temp_c_p50": statistics.median(temps),
        "temp_c_p95": p(temps, 0.95),
        "fan_pct_p50": statistics.median(fans),
        "smi_mem_mb_p95": p(mems, 0.95),
        "n_samples": len(rows),
        "energy_wh_raw": energy_j_raw / 3600.0,
        "energy_wh_idle_sub": energy_j_idle_sub / 3600.0,
        "duration_s": (rows[-1]["ts"] - rows[0]["ts"]).total_seconds(),
        "idle_power_w_used": idle_power_w,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/v4_pilot")
    ap.add_argument("--telemetry", default="outputs/v4_pilot/telemetry/gpu_timeseries.csv")
    ap.add_argument("--flops", default="outputs/fm_flops.json")
    ap.add_argument("--idle_w", type=float, default=50.0,
                    help="assumed GPU idle power (W) subtracted for distillation-specific energy")
    args = ap.parse_args()
    root = Path(args.root)
    tele = Path(args.telemetry)
    flops_path = Path(args.flops)
    flops_data = {}
    if flops_path.exists():
        with open(flops_path) as f:
            flops_data = json.load(f)
    rows = []
    # Two-level directory support: <fm>/ or <fm>/<student>/
    candidates = []
    for fm_dir in sorted(root.iterdir()):
        if not fm_dir.is_dir() or fm_dir.name == "telemetry":
            continue
        sub_dirs = [d for d in fm_dir.iterdir() if d.is_dir()]
        has_subruns = any((d / "training_log.json").exists() or (d / "distill.log").exists()
                          for d in sub_dirs)
        if has_subruns:
            for sd in sorted(sub_dirs):
                candidates.append((fm_dir.name, sd.name, sd))
        else:
            candidates.append((fm_dir.name, None, fm_dir))
    for fm_name, stu_name, src_dir in candidates:
        r = collect_fm(src_dir)
        if "total_time_s" not in r:
            continue
        r["fm"] = fm_name
        if stu_name:
            r["student"] = stu_name
            r["run_id"] = f"{fm_name}×{stu_name}"
        else:
            r["student"] = "vit-small"
            r["run_id"] = fm_name
        # FM-level Wh integration via training window
        if r.get("start_ts") and r.get("end_ts"):
            try:
                w_start = datetime.fromisoformat(r["start_ts"])
                w_end = datetime.fromisoformat(r["end_ts"])
                w_tele = collect_telemetry(tele, w_start, w_end, idle_power_w=args.idle_w)
                if w_tele:
                    r["energy_wh"] = w_tele["energy_wh_idle_sub"]
                    r["energy_wh_raw"] = w_tele["energy_wh_raw"]
                    r["power_p50_w"] = w_tele["power_w_p50"]
            except Exception:
                pass
        # FLOPs
        f_t = flops_data.get(r["fm"])
        f_s = flops_data.get("student_vit_small")
        if f_t and "gflops" in f_t:
            r["teacher_gflops"] = f_t["gflops"]
        if f_s and "gflops" in f_s:
            r["student_gflops"] = f_s["gflops"]
        if r.get("teacher_gflops") and r.get("student_gflops"):
            r["gflops_ratio"] = r["teacher_gflops"] / r["student_gflops"]
        rows.append(r)
    # Overall telemetry summary
    tele_sum = collect_telemetry(tele, idle_power_w=args.idle_w)

    # CSV
    if rows:
        out_csv = root / "paper_metrics.csv"
        cols = ["run_id", "fm", "student", "teacher_params", "student_params", "compression",
                "epochs_run", "per_epoch_min_s", "per_epoch_median_s", "per_epoch_max_s",
                "total_time_s", "peak_vram_mb", "median_vram_mb",
                "energy_wh", "energy_wh_raw", "power_p50_w",
                "teacher_gflops", "student_gflops", "gflops_ratio",
                "best_val", "best_epoch"]
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows: w.writerow(r)
        print(f"[csv] {out_csv}")

    # Markdown table
    out_md = root / "paper_metrics.md"
    with open(out_md, "w") as f:
        f.write("# Training Resource Metrics (paper-ready)\n\n")
        f.write("## Per-run (training)\n\n")
        f.write("| Run (FM × Student) | Teacher (M) | Student (M) | Comp. | T-GFLOPs | S-GFLOPs | GFLOPs× | "
                "Epochs | Per-ep (s) | Total (h) | Peak VRAM (MB) | Wh (−idle) | Best val |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            tp = r.get("teacher_params", 0) or 0
            sp = r.get("student_params", 0) or 0
            comp = r.get("compression", 0) or 0
            tg = r.get("teacher_gflops")
            sg = r.get("student_gflops")
            gr = r.get("gflops_ratio")
            ep = r.get("epochs_run", "-")
            ep_med = r.get("per_epoch_median_s")
            ep_med_s = f"{ep_med:.0f}" if ep_med else "-"
            total_h = r.get("total_time_s", 0) / 3600 if r.get("total_time_s") else 0
            vram = r.get("peak_vram_mb")
            vram_s = f"{vram:.0f}" if vram is not None else "—"
            wh = r.get("energy_wh")
            wh_s = f"{wh:.1f}" if wh else "—"
            val = r.get("best_val")
            val_s = f"{val:.4f}" if val else "-"
            flops_cells = (f" {tg:.1f} | {sg:.1f} | {gr:.1f}× |"
                           if (tg and sg and gr) else " — | — | — |")
            f.write(f"| {r.get('run_id', r['fm'])} | {tp/1e6:.1f} | {sp/1e6:.1f} | {comp:.1f}× |"
                    + flops_cells
                    + f" {ep} | {ep_med_s} | {total_h:.2f} | {vram_s} | {wh_s} | {val_s} |\n")
        f.write(f"\n_Wh reported is idle-subtracted at {args.idle_w:.0f}W baseline; "
                f"raw Wh (no subtraction) is in the CSV._\n")

        f.write("\n## System-wide GPU telemetry (10-s samples, full sweep window)\n\n")
        if tele_sum:
            f.write(f"- Samples: {tele_sum['n_samples']} × 10s ≈ {tele_sum['n_samples']*10/3600:.1f} h\n")
            f.write(f"- GPU utilization: median {tele_sum['gpu_util_p50']:.0f}% / p95 {tele_sum['gpu_util_p95']:.0f}%\n")
            f.write(f"- Power: median {tele_sum['power_w_p50']:.0f} W / p95 {tele_sum['power_w_p95']:.0f} W\n")
            f.write(f"- Temperature: median {tele_sum['temp_c_p50']:.0f}°C / p95 {tele_sum['temp_c_p95']:.0f}°C\n")
            f.write(f"- Fan: median {tele_sum['fan_pct_p50']:.0f}%\n")
            f.write(f"- nvidia-smi mem p95: {tele_sum['smi_mem_mb_p95']:.0f} MB\n")
            f.write(f"- Total energy (raw): {tele_sum['energy_wh_raw']:.1f} Wh\n")
            f.write(f"- Total energy (idle-sub {args.idle_w:.0f}W): {tele_sum['energy_wh_idle_sub']:.1f} Wh\n")
        else:
            f.write("_Telemetry CSV not yet populated._\n")

        f.write("\n## Hardware / protocol\n")
        f.write("- GPU: NVIDIA GeForce RTX 5090 (Blackwell, 32 GB GDDR7)\n")
        f.write("- Precision: FP32 training, mixed inference latency reported elsewhere\n")
        f.write("- Student architecture: ViT-S/16 (timm vit_small_patch16_224, ImageNet-init, 22M)\n")
        f.write(f"- Idle-power baseline for Wh subtraction: {args.idle_w:.0f} W (conservative)\n")
    print(f"[md]  {out_md}")


if __name__ == "__main__":
    main()
