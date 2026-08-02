#!/usr/bin/env python3
"""Patch downstream_report.json with medical-center probe results.

The original evaluate_v4_downstream.py built group IDs as "brca:slide{N}"
(integer slide index), so tcga_tss_code() always returned "UNK" and the
center probe was silently skipped.  The BRCA H5 file stores the actual
TCGA barcodes in attrs["slide_names"].  This script:

  1. Builds slide_id → TCGA barcode mapping from that attribute.
  2. For each teacher: loads once, extracts test-split BRCA features.
  3. For each student under that teacher: loads checkpoint, extracts features.
  4. Runs medical_center_probe() for teacher and student independently.
  5. Patches medical_center_probe into downstream_report.json in-place.
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from distill_wsi_model import StudentModel, TeacherModel
from evaluate_multi import build_sources
from multi_source_dataset import MultiSourceDataset
from eval_v4_addons import medical_center_probe, robustness_index, tcga_tss_code


def build_slide_name_map(brca_h5_path: str) -> dict:
    with h5py.File(brca_h5_path, "r") as f:
        names = f.attrs.get("slide_names", [])
        return {i: (n.decode() if isinstance(n, bytes) else str(n))
                for i, n in enumerate(names)}


def group_to_tss(grp, slide_name_map: dict) -> str | None:
    if isinstance(grp, bytes):
        grp = grp.decode()
    if grp.startswith("brca:slide"):
        try:
            sid = int(grp.split("slide", 1)[1])
            barcode = slide_name_map.get(sid, "")
            return tcga_tss_code(barcode)
        except (ValueError, IndexError):
            pass
    return None


def extract_brca_feats(model, loader, device, slide_name_map) -> tuple:
    """Return (features_np, tss_codes_array) for BRCA samples only."""
    model.eval()
    feats, tss = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            out = model(x)
            f = out["feat"] if isinstance(out, dict) else out
            feats.append(f.cpu())
            for grp in batch[2]:
                tss.append(group_to_tss(grp, slide_name_map))
    all_feats = torch.cat(feats).numpy()
    all_tss = np.array(tss, dtype=object)
    mask = np.array([t is not None and t != "UNK" for t in all_tss])
    return all_feats[mask], all_tss[mask]


def run_probe(feats, tss, task_auc, seed) -> dict:
    if len(feats) < 50 or len(set(tss)) < 2:
        return {}
    result = medical_center_probe(feats, tss, seed=seed)
    if task_auc is not None and "center_probe_auc" in result:
        result["robustness_index"] = robustness_index(task_auc, result["center_probe_auc"])
    return result


def get_task_auc(c16_rows, teacher_name, student_short) -> float | None:
    for r in c16_rows:
        if r["teacher"] == teacher_name and r["student"] == student_short:
            return r.get("linear_s_auc")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/v4_full")
    ap.add_argument("--config", default="configs/multi_source_v1.json")
    ap.add_argument("--brca-h5", default="data/patches_brca.h5")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[center probe] device={device}")

    slide_name_map = build_slide_name_map(args.brca_h5)
    print(f"[center probe] {len(slide_name_map)} BRCA slides, "
          f"{len({tcga_tss_code(v) for v in slide_name_map.values()} - {'UNK'})} TSS codes")

    cfg = json.load(open(args.config))
    srcs = build_sources(cfg)

    c16_rows = []
    c16_path = Path(args.root) / "c16_equiv_annotated.json"
    if c16_path.exists():
        c16_rows = json.load(open(c16_path)).get("rows", [])

    STUDENT_MAP = {
        "vit-tiny":  "vit_tiny_patch16_224",
        "vit-small": "vit_small_patch16_224",
        "vit-base":  "vit_base_patch16_224",
    }

    root = Path(args.root)
    results_all = []

    for fm_dir in sorted(root.iterdir()):
        if not fm_dir.is_dir() or fm_dir.name in ("telemetry", "figures"):
            continue
        teacher_name = fm_dir.name

        # Gather student runs for this teacher that need patching
        student_runs = []
        for stu_dir in sorted(fm_dir.iterdir()):
            if not stu_dir.is_dir():
                continue
            rp = stu_dir / "downstream_report.json"
            sp = stu_dir / "splits.npz"
            ck = stu_dir / "best.pt"
            if not (rp.exists() and sp.exists() and ck.exists()):
                continue
            report = json.load(open(rp))
            if report["results"].get("medical_center_probe"):
                print(f"[skip] {teacher_name}/{stu_dir.name}")
                continue
            student_runs.append((stu_dir, report))

        if not student_runs:
            continue

        print(f"\n{'='*60}")
        print(f"[teacher] {teacher_name}  ({len(student_runs)} students to probe)")
        if args.dry_run:
            continue

        # Load teacher once and extract BRCA features using the first student's test split
        # (all students under the same teacher share the same underlying dataset split seed)
        first_dir, _ = student_runs[0]
        sp = np.load(first_dir / "splits.npz")
        te_ds = MultiSourceDataset(srcs, 224, indices=sp["test"])
        te_ld = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=False,
                           persistent_workers=args.num_workers > 0)

        t_feats, t_tss = None, None
        try:
            teacher = TeacherModel(teacher_name).eval().to(device)
            print(f"  [extract] teacher BRCA features")
            t_feats, t_tss = extract_brca_feats(teacher, te_ld, device, slide_name_map)
            del teacher
            torch.cuda.empty_cache()
            print(f"  teacher: {len(t_feats)} BRCA patches, "
                  f"{len(set(t_tss))} centers")
        except Exception as e:
            print(f"  [warn] teacher extract failed: {e}")

        for stu_dir, report in student_runs:
            stu_short = stu_dir.name
            stu_arch = STUDENT_MAP.get(stu_short, "vit_small_patch16_224")
            print(f"\n  [student] {stu_short}")

            sp_s = np.load(stu_dir / "splits.npz")
            te_ds_s = MultiSourceDataset(srcs, 224, indices=sp_s["test"])
            te_ld_s = DataLoader(te_ds_s, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=False,
                                 persistent_workers=args.num_workers > 0)

            s_feats, s_tss = None, None
            try:
                student = StudentModel(stu_arch, embed_dim=256, n_classes=0,
                                       pretrained=False).to(device)
                ckpt = torch.load(stu_dir / "best.pt", map_location=device)
                student.load_state_dict(ckpt.get("student", ckpt), strict=False)
                student.eval()
                print(f"    [extract] student BRCA features")
                s_feats, s_tss = extract_brca_feats(student, te_ld_s, device, slide_name_map)
                del student
                torch.cuda.empty_cache()
                print(f"    student: {len(s_feats)} BRCA patches, "
                      f"{len(set(s_tss))} centers")
            except Exception as e:
                print(f"    [warn] student extract failed: {e}")

            task_auc_t = None
            task_auc_s = get_task_auc(c16_rows, teacher_name, stu_short)

            center_result = {}
            if t_feats is not None and len(t_feats) > 0:
                r = run_probe(t_feats, t_tss, task_auc_t, args.seed)
                if r:
                    center_result["teacher"] = r
                    print(f"    teacher → AUC={r.get('center_probe_auc'):.3f} "
                          f"n_centers={r.get('n_centers')} RI={r.get('robustness_index')}")
            if s_feats is not None and len(s_feats) > 0:
                r = run_probe(s_feats, s_tss, task_auc_s, args.seed)
                if r:
                    center_result["student"] = r
                    print(f"    student → AUC={r.get('center_probe_auc'):.3f} "
                          f"n_centers={r.get('n_centers')} RI={r.get('robustness_index')}")

            report["results"]["medical_center_probe"] = center_result
            with open(stu_dir / "downstream_report.json", "w") as fh:
                json.dump(report, fh, indent=2, default=float)
            print(f"    [saved] {stu_dir}/downstream_report.json")
            results_all.append({
                "teacher": teacher_name, "student": stu_short,
                "t_auc": center_result.get("teacher", {}).get("center_probe_auc"),
                "s_auc": center_result.get("student", {}).get("center_probe_auc"),
                "t_ri":  center_result.get("teacher", {}).get("robustness_index"),
                "s_ri":  center_result.get("student", {}).get("robustness_index"),
            })

    print(f"\n[done] patched {len(results_all)} runs")
    fmt = lambda v: f"{v:.3f}" if (v is not None and v == v) else "nan"
    for r in results_all:
        print(f"  {r['teacher']}/{r['student']}: "
              f"T_auc={fmt(r['t_auc'])} S_auc={fmt(r['s_auc'])} "
              f"T_RI={fmt(r['t_ri'])} S_RI={fmt(r['s_ri'])}")


if __name__ == "__main__":
    main()
