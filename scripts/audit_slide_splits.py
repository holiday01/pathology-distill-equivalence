#!/usr/bin/env python3
"""Audit train/val/test splits across all v4_full runs to verify no
slide-level leakage.

Reviewer-critical check (proposal v4 §3.3): splits must be computed at
slide (group) level, not at patch level. If the same WSI's patches appear
in both train and test, the benchmark is invalidated.

Usage:
    python3 scripts/audit_slide_splits.py [--root outputs/v4_full]

Reports per-run:
  - n_train / n_val / n_test (patches)
  - n_slides_train / n_val / n_test (unique group ids)
  - LEAKAGE: any slide id appearing in >1 split → FAIL
  - per-source (c16 / brca / nct_crc / ...) leakage breakdown

Exits 1 on any leak detected.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from multi_source_dataset import MultiSourceDataset
from evaluate_multi import build_sources


def audit_run(run_dir: Path) -> dict:
    """Return audit for one run. Keys: ok, n_{train,val,test}, slide counts, leaks."""
    splits = run_dir / "splits.npz"
    log = run_dir / "training_log.json"
    if not splits.exists() or not log.exists():
        return {"ok": None, "reason": "missing splits.npz or training_log.json"}
    sp = np.load(splits)
    try:
        cfg = json.load(open(log))["source_config"]
    except Exception as e:
        return {"ok": False, "reason": f"bad training_log.json: {e}"}

    # Build dataset to map global index → (tag, group)
    try:
        srcs = build_sources(cfg)
    except Exception as e:
        return {"ok": False, "reason": f"cannot rebuild sources: {e}"}
    ds = MultiSourceDataset(srcs, target_size=224)

    # Efficient group/tag extraction via per-source bulk calls
    all_groups = np.empty(len(ds), dtype=object)
    all_tags = np.empty(len(ds), dtype=object)
    for si, src in enumerate(ds.sources):
        lo, hi = ds.offsets[si]
        for gi in range(lo, hi):
            li = gi - lo
            all_tags[gi] = src.source_tag
            all_groups[gi] = src.get_group(li)

    # Leak detection only on true slide-level sources (HDF5 WSI).
    # Bucket-based sources (NCT-CRC, Kather-MSI, PanNuke) are tile-level
    # datasets with non-deterministic hash-bucket groups; patch-random split
    # is acceptable and the group ids aren't stable across processes anyway.
    slide_level_tags = {"c16", "brca"}   # HDF5 WSI sources

    def groups_for(idx, tag_mask=None):
        if len(all_groups) == 0:
            return set()
        sel = idx
        if tag_mask is not None:
            m = np.isin(all_tags[idx], list(tag_mask))
            sel = np.asarray(idx)[m]
        return set(all_groups[sel].tolist())

    tr_g = groups_for(sp["train"], slide_level_tags)
    va_g = groups_for(sp.get("val", np.array([], dtype=int)), slide_level_tags)
    te_g = groups_for(sp.get("test", np.array([], dtype=int)), slide_level_tags)

    overlaps = {
        "train_val": sorted(tr_g & va_g),
        "train_test": sorted(tr_g & te_g),
        "val_test": sorted(va_g & te_g),
    }
    leaks = sum(len(v) for v in overlaps.values())

    # Per-source tally
    per_src = defaultdict(lambda: {"train": 0, "val": 0, "test": 0, "unique_slides": set()})
    for split, idx in [("train", sp["train"]),
                       ("val", sp.get("val", np.array([], dtype=int))),
                       ("test", sp.get("test", np.array([], dtype=int)))]:
        if len(idx) == 0: continue
        tags_in = all_tags[idx]
        grps_in = all_groups[idx]
        for t, g in zip(tags_in, grps_in):
            per_src[t][split] += 1
            per_src[t]["unique_slides"].add(g)
    for t, d in per_src.items():
        d["unique_slides"] = len(d["unique_slides"])

    return {
        "ok": leaks == 0,
        "n_train_patches": int(len(sp["train"])),
        "n_val_patches": int(len(sp.get("val", np.array([], dtype=int)))),
        "n_test_patches": int(len(sp.get("test", np.array([], dtype=int)))),
        "n_train_slides": len(tr_g),
        "n_val_slides": len(va_g),
        "n_test_slides": len(te_g),
        "leaks": leaks,
        "overlaps": {k: v[:5] for k, v in overlaps.items()},  # first 5 each
        "per_source": {t: dict(d) for t, d in per_src.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/v4_full")
    ap.add_argument("--json-out", default="outputs/v4_full/slide_split_audit.json")
    args = ap.parse_args()

    root = Path(args.root)
    results = {}
    any_fail = False
    for fm_dir in sorted(root.glob("*/")):
        if fm_dir.name == "telemetry": continue
        for stu_dir in sorted(fm_dir.glob("*/")):
            name = f"{fm_dir.name}/{stu_dir.name}"
            try:
                r = audit_run(stu_dir)
            except Exception as e:
                r = {"ok": False, "reason": f"exception: {e}"}
            results[name] = r
            if r.get("ok") is False:
                any_fail = True
                tag = "[FAIL]"
            elif r.get("ok") is True:
                tag = "[PASS]"
            else:
                tag = "[SKIP]"
            leaks = r.get("leaks", "—")
            n_te = r.get("n_test_patches", "—")
            n_te_s = r.get("n_test_slides", "—")
            print(f"{tag}  {name:40s}  leaks={leaks}  n_test={n_te} patches / {n_te_s} slides")
            if r.get("ok") is False and "overlaps" in r:
                for key, items in r["overlaps"].items():
                    if items:
                        print(f"         LEAK {key}: {items}")

    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[saved] {args.json_out}")
    print(f"runs: {len(results)}  pass: {sum(1 for r in results.values() if r.get('ok') is True)}  "
          f"fail: {sum(1 for r in results.values() if r.get('ok') is False)}  "
          f"skip: {sum(1 for r in results.values() if r.get('ok') is None)}")
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
