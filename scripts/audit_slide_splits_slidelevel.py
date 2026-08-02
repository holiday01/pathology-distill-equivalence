#!/usr/bin/env python3
"""Slide-level leakage audit, robust to relocated tile-source folders.

Identical leak semantics to audit_slide_splits.py, but rebuilds ONLY the
slide-level HDF5 sources (c16, brca) — the only sources leak-detection uses
(slide_level_tags). Tile-level folder sources (NCT-CRC, Kather-MSI, PanNuke)
use a deterministic disjoint 20-bucket split (train/val/test buckets are
disjoint by construction) and several of their raw folders have since been
relocated; they are not needed to verify slide-level leakage.

c16 and brca are always sources[0], sources[1] in every run's config, so
their global split indices occupy [0, L) with L = len(c16)+len(brca); indices
>= L (tile-source patches) are filtered out before the slide-ID overlap check.

Usage: python3 scripts/audit_slide_splits_slidelevel.py
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from multi_source_dataset import MultiSourceDataset
from evaluate_multi import build_sources

SLIDE_TAGS = {"c16", "brca"}


def audit_run(run_dir: Path) -> dict:
    splits = run_dir / "splits.npz"
    log = run_dir / "training_log.json"
    if not splits.exists() or not log.exists():
        return {"ok": None, "reason": "missing splits.npz or training_log.json"}
    sp = np.load(splits, allow_pickle=True)
    try:
        cfg = json.load(open(log))["source_config"]
    except Exception as e:
        return {"ok": False, "reason": f"bad training_log.json: {e}"}

    slide_cfg = {"sources": [e for e in cfg["sources"]
                             if e.get("type") == "hdf5" and e.get("tag") in SLIDE_TAGS]}
    if not slide_cfg["sources"]:
        return {"ok": None, "reason": "no slide-level (c16/brca) sources in config"}
    # order sanity: c16/brca must be the leading sources so [0,L) == slide-level patches
    lead = [e.get("tag") for e in cfg["sources"][:len(slide_cfg["sources"])]]
    if set(lead) != set(e["tag"] for e in slide_cfg["sources"]):
        return {"ok": False, "reason": f"slide sources not leading (lead={lead})"}
    try:
        ds = MultiSourceDataset(build_sources(slide_cfg), target_size=224)
    except Exception as e:
        return {"ok": False, "reason": f"cannot rebuild slide sources: {e}"}

    L = len(ds)
    groups = np.empty(L, dtype=object)
    tags = np.empty(L, dtype=object)
    for si, src in enumerate(ds.sources):
        lo, hi = ds.offsets[si]
        for gi in range(lo, hi):
            tags[gi] = src.source_tag
            groups[gi] = src.get_group(gi - lo)

    def gset(idx):
        idx = np.asarray(idx)
        idx = idx[idx < L]
        return set(groups[idx].tolist()) if len(idx) else set()

    tr = gset(sp["train"]); va = gset(sp.get("val", np.array([], int))); te = gset(sp.get("test", np.array([], int)))
    overlaps = {"train_val": sorted(tr & va), "train_test": sorted(tr & te), "val_test": sorted(va & te)}
    leaks = sum(len(v) for v in overlaps.values())

    per_src = defaultdict(lambda: {"train": 0, "val": 0, "test": 0, "unique_slides": set()})
    for split, idx in [("train", sp["train"]), ("val", sp.get("val", np.array([], int))), ("test", sp.get("test", np.array([], int)))]:
        idx = np.asarray(idx); idx = idx[idx < L]
        for t, g in zip(tags[idx], groups[idx]):
            per_src[t][split] += 1
            per_src[t]["unique_slides"].add(g)
    for t, d in per_src.items():
        d["unique_slides"] = len(d["unique_slides"])

    return {"ok": leaks == 0, "leaks": leaks,
            "n_train_slides": len(tr), "n_val_slides": len(va), "n_test_slides": len(te),
            "n_slide_level_patches": int(sum((np.asarray(sp[k]) < L).sum() for k in ("train", "val", "test") if k in sp.files)),
            "overlaps": {k: v[:5] for k, v in overlaps.items()},
            "per_source": {t: dict(d) for t, d in per_src.items()},
            "note": "slide-level (c16/brca) leak check; tile sources use disjoint deterministic buckets"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/v4_full")
    ap.add_argument("--json-out", default="outputs/v4_full/slide_split_audit.json")
    args = ap.parse_args()
    root = Path(args.root)
    results, any_fail = {}, False
    for fm_dir in sorted(root.glob("*/")):
        if fm_dir.name == "telemetry":
            continue
        for stu_dir in sorted(fm_dir.glob("*/")):
            if "partial" in stu_dir.name:
                continue
            name = f"{fm_dir.name}/{stu_dir.name}"
            try:
                r = audit_run(stu_dir)
            except Exception as e:
                r = {"ok": False, "reason": f"exception: {e}"}
            results[name] = r
            tag = {True: "[PASS]", False: "[FAIL]"}.get(r.get("ok"), "[SKIP]")
            print(f"{tag}  {name:34s}  leaks={r.get('leaks','—')}  slides tr/va/te="
                  f"{r.get('n_train_slides','—')}/{r.get('n_val_slides','—')}/{r.get('n_test_slides','—')}")
            if r.get("ok") is False:
                any_fail = True
                print(f"        reason/overlaps: {r.get('reason', r.get('overlaps'))}")
    with open(args.json_out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    npass = sum(1 for r in results.values() if r.get("ok") is True)
    print(f"\n[saved] {args.json_out}\nruns: {len(results)}  pass: {npass}  "
          f"fail: {sum(1 for r in results.values() if r.get('ok') is False)}  "
          f"skip: {sum(1 for r in results.values() if r.get('ok') is None)}")
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
