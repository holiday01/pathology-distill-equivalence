#!/usr/bin/env python3
"""Build a patient-clustered Kather-MSI H5 for a SECOND equivalence cohort.

MSI vs MSS is a patient-level molecular property, so a patient's filename
label is a CORRECT tile label (unlike CAMELYON16's focal tumour) -- no label
noise. Tiles are named blk-XXXX-TCGA-AA-BBBB-... ; the TCGA barcode is the
patient = the cluster. We subsample --per_patient tiles per patient and store
patches/labels(MSI=1,MSS=0)/patient_ids/split using the predefined Kather
TRAIN/TEST partition. TEST has ~103 patients -> >25 clusters for valid
cluster-robust inference.
"""
import argparse, os, re, random
from pathlib import Path
import numpy as np, h5py
from PIL import Image

PAT = re.compile(r"(TCGA-[A-Z0-9]+-[A-Z0-9]+)")
DIRS = {  # (split, class) -> dir, label
    ("train", "MSIMUT"): ("CRC_DX_TRAIN_MSIMUT_unz/MSIMUT", 1),
    ("train", "MSS"):    ("CRC_DX_TRAIN_MSS_unz/MSS", 0),
    ("test", "MSIMUT"):  ("CRC_DX_TEST_MSIMUT_unz/MSIMUT", 1),
    ("test", "MSS"):     ("CRC_DX_TEST_MSS_unz/MSS", 0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/external/kather_msi")
    ap.add_argument("--out", default="/path/to/cache/patches/patches_kather_msi.h5")
    ap.add_argument("--per_patient", type=int, default=60)
    ap.add_argument("--patch_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    ps = args.patch_size

    # gather files per (split,patient)
    by = {}  # (split, patient) -> [(path,label)]
    pid_map = {}
    for (split, cls), (sub, lab) in DIRS.items():
        d = Path(args.base) / sub
        if not d.is_dir():
            print(f"[warn] missing {d}"); continue
        for fn in os.listdir(d):
            m = PAT.search(fn)
            if not m: continue
            pat = m.group(1)
            by.setdefault((split, pat), []).append((str(d / fn), lab))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as h5:
        P = h5.create_dataset("patches", (0, ps, ps, 3), maxshape=(None, ps, ps, 3),
                              dtype="uint8", chunks=(1, ps, ps, 3),
                              compression="gzip", compression_opts=4)
        L = h5.create_dataset("labels", (0,), maxshape=(None,), dtype="int64")
        S = h5.create_dataset("slide_ids", (0,), maxshape=(None,), dtype="int64")  # patient cluster
        SP = h5.create_dataset("is_test", (0,), maxshape=(None,), dtype="int64")
        n_w = 0
        for (split, pat), items in sorted(by.items()):
            if pat not in pid_map:
                pid_map[pat] = len(pid_map)
            cid = pid_map[pat]
            rng.shuffle(items)
            items = items[:args.per_patient]
            buf, labs = [], []
            for path, lab in items:
                try:
                    im = Image.open(path).convert("RGB")
                    if im.size != (ps, ps):
                        im = im.resize((ps, ps))
                    buf.append(np.asarray(im, dtype=np.uint8)); labs.append(lab)
                except Exception:
                    continue
            if not buf: continue
            n = len(buf)
            P.resize(n_w + n, axis=0); P[n_w:n_w+n] = np.stack(buf)
            L.resize(n_w + n, axis=0); L[n_w:n_w+n] = np.asarray(labs)
            S.resize(n_w + n, axis=0); S[n_w:n_w+n] = cid
            SP.resize(n_w + n, axis=0); SP[n_w:n_w+n] = (1 if split == "test" else 0)
            n_w += n
        # report
        lab = h5["labels"][:]; te = h5["is_test"][:]; sid = h5["slide_ids"][:]
        print(f"[done] {n_w} tiles -> {args.out}")
        for nm, mask in (("train", te == 0), ("test", te == 1)):
            pats = np.unique(sid[mask])
            print(f"  {nm}: {mask.sum()} tiles, {len(pats)} patients, "
                  f"MSI {int(lab[mask].sum())} / MSS {int((lab[mask]==0).sum())}")


if __name__ == "__main__":
    main()
