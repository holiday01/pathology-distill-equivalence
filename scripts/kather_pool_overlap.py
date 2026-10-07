#!/usr/bin/env python3
"""How much of the Kather-MSI evaluation was in the distillation pool.

configs/multi_source_v1.json samples 30,000 PNGs (random.Random(42), as in
multi_source_dataset.FolderSource) from data/external/kather_msi, which holds
both the CRC_DX_TRAIN_* and CRC_DX_TEST_* folders. The Kather equivalence
evaluation uses the CRC_DX_TEST patients. This reproduces the sample and
counts test tiles and test patients in it.
"""
import json
import random
import re
from pathlib import Path

ROOT = Path("data/external/kather_msi")
OUT = Path("outputs/v4_full/kather_pool_overlap.json")


def patient(p):
    m = re.search(r"TCGA-\w\w-\w\w\w\w", p.name)
    return m.group(0) if m else None


def main():
    paths = sorted(ROOT.rglob("*.png"))
    pool = sorted(random.Random(42).sample(paths, 30000))
    test_all = {patient(p) for p in paths if "CRC_DX_TEST" in str(p)}
    test_pool = [p for p in pool if "CRC_DX_TEST" in str(p)]
    out = {"n_png": len(paths), "n_pool": len(pool),
           "n_pool_test_tiles": len(test_pool),
           "n_test_patients": len(test_all - {None}),
           "n_test_patients_in_pool": len({patient(p) for p in test_pool} - {None})}
    OUT.write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main()
