#!/usr/bin/env python3
"""Sweep 結果彙整：掃描 outputs/sweep_xxx/distill + evaluation，
產生一張跨組別對比表（summary.md + summary.json）。"""

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"  [警告] 無法解析 {path}: {e}")
        return None


def collect(sweep_dir: Path):
    rows = []
    distill_root = sweep_dir / "distill"
    eval_root = sweep_dir / "evaluation"
    if not distill_root.exists():
        print(f"[錯誤] 找不到 {distill_root}")
        return rows

    for run_dir in sorted(distill_root.iterdir()):
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        tlog = load_json(run_dir / "training_log.json")
        ereport = load_json(eval_root / name / "evaluation_report.json")

        row = {"run": name}
        if tlog:
            cfg = tlog.get("config", {})
            hist = tlog.get("history", [])
            row["method"] = cfg.get("method")
            row["batch_size"] = cfg.get("batch_size")
            row["lr"] = cfg.get("lr")
            row["pretrained"] = cfg.get("pretrained_student", False)
            row["warmup"] = cfg.get("warmup_ratio", 0.0)
            row["proj"] = cfg.get("projector", "linear")
            row["wd"] = cfg.get("weight_decay")
            row["grad_clip"] = cfg.get("grad_clip", 0.0)
            row["final_loss"] = hist[-1]["loss"] if hist else None
            row["final_acc"] = hist[-1]["acc"] if hist else None
            row["train_time"] = hist[-1]["time"] if hist else None

        if ereport:
            res = ereport.get("results", {})
            row["cka"] = res.get("feature_similarity", {}).get("linear_cka")
            row["cosine"] = res.get("feature_similarity", {}).get("cosine")
            lp = res.get("linear_probing", {})
            row["teacher_acc"] = lp.get("teacher_acc")
            row["student_acc"] = lp.get("student_acc")
            row["acc_gap"] = lp.get("acc_gap")
            row["kappa"] = res.get("agreement", {}).get("cohen_kappa")
            eff = res.get("efficiency", {})
            row["speedup"] = eff.get("speedup")
            row["student_latency_ms"] = eff.get("student_latency_ms")
            row["equivalent"] = res.get("verdict", {}).get("equivalent", False)
        rows.append(row)
    return rows


def fmt(v, spec=".4f", na="—"):
    if v is None:
        return na
    if isinstance(v, bool):
        return "Y" if v else "N"
    try:
        return f"{v:{spec}}"
    except (ValueError, TypeError):
        return str(v)


def render_md(rows, out_path: Path):
    lines = []
    lines.append("# Sweep Summary\n")
    lines.append(f"共 {len(rows)} 組實驗。依 CKA 由高至低排序。\n")
    lines.append("")
    lines.append("## 主要指標對照表")
    lines.append("")
    header = (
        "| Run | Method | BS | LR | PT | Warm | Proj | WD | GC "
        "| Loss | CKA | ΔACC | κ | Speed× | Eqv |"
    )
    sep = (
        "|-----|--------|----|----|----|------|------|----|----"
        "|------|-----|------|---|--------|-----|"
    )
    lines.append(header)
    lines.append(sep)

    # 依 CKA 排序
    sorted_rows = sorted(rows, key=lambda r: (r.get("cka") or -1), reverse=True)
    for r in sorted_rows:
        lines.append(
            f"| {r['run']} | {r.get('method','')} "
            f"| {r.get('batch_size','')} | {fmt(r.get('lr'), '.0e')} "
            f"| {fmt(r.get('pretrained'), '')} | {fmt(r.get('warmup'), '.2f')} "
            f"| {r.get('proj','')} | {fmt(r.get('wd'), '.0e')} "
            f"| {fmt(r.get('grad_clip'), '.1f')} "
            f"| {fmt(r.get('final_loss'), '.4f')} "
            f"| {fmt(r.get('cka'), '.4f')} "
            f"| {fmt(r.get('acc_gap'), '+.3f')} "
            f"| {fmt(r.get('kappa'), '.3f')} "
            f"| {fmt(r.get('speedup'), '.2f')} "
            f"| {fmt(r.get('equivalent'), '')} |"
        )

    lines.append("")
    lines.append("## 說明")
    lines.append("- **CKA**：特徵表徵相似度，越接近 1 越像 teacher")
    lines.append("- **ΔACC**：linear probe 下 teacher - student 的差（越小越好）")
    lines.append("- **κ (Cohen's kappa)**：預測一致性，>0.6 視為高度一致")
    lines.append("- **Eqv**：CKA>0.7 且 |ΔACC|<0.03 且 κ>0.6 才算等效")
    lines.append("- Demo 資料為隨機噪聲，CKA 絕對值偏低屬正常；看的是**相對差異**")
    lines.append("")

    lines.append("## 關鍵觀察（由程式自動計算）")
    lines.append("")

    # 計算一些關鍵 delta
    def find_run(name_prefix):
        for r in rows:
            if r["run"].startswith(name_prefix):
                return r
        return None

    baseline = find_run("01_baseline")
    pretrained = find_run("02_pretrained")
    warmup = find_run("03_warmup")
    mlp = find_run("04_mlp_proj")
    best = find_run("05_best")
    bs128 = find_run("06_best_bs128")
    bs256 = find_run("07_best_bs256")
    bs512 = find_run("08_best_bs512")
    feat = find_run("09_best_feature")
    rel = find_run("10_best_relation")

    def delta(a, b, key):
        if not a or not b: return "—"
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None: return "—"
        return f"{vb - va:+.4f}"

    lines.append(f"- Pretrained student 效果 (CKA): {delta(baseline, pretrained, 'cka')}")
    lines.append(f"- 加 Warmup 的增量 (CKA): {delta(pretrained, warmup, 'cka')}")
    lines.append(f"- 改 MLP projector 的增量 (CKA): {delta(warmup, mlp, 'cka')}")
    lines.append(f"- WD=0.05 + grad_clip 的增量 (CKA): {delta(mlp, best, 'cka')}")
    lines.append("")
    lines.append("### Batch size 影響（固定其他設定為 best）")
    for tag, r in [("bs=32 ", best), ("bs=128", bs128), ("bs=256", bs256), ("bs=512", bs512)]:
        if r:
            lines.append(
                f"- {tag} → CKA={fmt(r.get('cka'), '.4f')}, "
                f"Loss={fmt(r.get('final_loss'), '.4f')}, "
                f"time={fmt(r.get('train_time'), '.1f')}s"
            )
    lines.append("")
    lines.append("### Method 比較（同用 best 超參數，bs=128）")
    for tag, r in [("hybrid  ", bs128), ("feature ", feat), ("relation", rel)]:
        if r:
            lines.append(
                f"- {tag} → CKA={fmt(r.get('cka'), '.4f')}, "
                f"Loss={fmt(r.get('final_loss'), '.4f')}, "
                f"κ={fmt(r.get('kappa'), '.3f')}"
            )
    lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"Summary 寫入: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", required=True, type=Path)
    args = ap.parse_args()

    rows = collect(args.sweep_dir)
    if not rows:
        print("[錯誤] 找不到任何 run 結果")
        return

    (args.sweep_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    render_md(rows, args.sweep_dir / "summary.md")

    # Console 快速預覽
    print("\n── 摘要預覽（依 CKA 排序）──")
    print(f"{'Run':<22} {'Method':<10} {'BS':>4} {'CKA':>8} {'Loss':>10} {'ΔACC':>8}")
    for r in sorted(rows, key=lambda x: (x.get("cka") or -1), reverse=True):
        print(
            f"{r['run']:<22} {str(r.get('method','')):<10} "
            f"{str(r.get('batch_size','')):>4} "
            f"{fmt(r.get('cka'), '.4f'):>8} "
            f"{fmt(r.get('final_loss'), '.4f'):>10} "
            f"{fmt(r.get('acc_gap'), '+.3f'):>8}"
        )


if __name__ == "__main__":
    main()
