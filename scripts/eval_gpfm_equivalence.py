#!/usr/bin/env python3
"""External audit no. 2: does GPFM certify equivalent to the teachers it was
distilled from?

GPFM (Ma et al. 2025) is a publicly released ViT-L distilled from CONCH,
Phikon and UNI under a unified knowledge-distillation objective. Unlike
H0-mini it is a MULTI-teacher distillation, so its authors never claimed
equivalence to any single teacher; the question asked here is our own. Each
of the three (teacher, GPFM) pairs is run through the identical
slide-cluster TOST used for the atlas: lesion-annotated CAMELYON16 tiles,
the same slide-stratified split, delta = 0.05, linear and MLP probes.

The GPFM checkpoint stores DINOv2 BlockChunk-style keys
("backbone.blocks.<chunk>.<global_idx>.*") with the block index already
global, so un-chunking is a prefix rewrite. Because a silent loading error
would look exactly like a real negative result, main() refuses to report any
verdict until the loaded model clears a sanity gate on teacher-grade probe
performance.

Usage:
  python3 scripts/eval_gpfm_equivalence.py \
      --h5 /path/to/cache/patches/patches_c16_annotated.h5 \
      --out outputs/v4_full/gpfm_equiv.json
"""
import argparse, json, re, sys
from pathlib import Path

import numpy as np
import h5py
import torch

sys.path.insert(0, str(Path(__file__).parent))
from distill_wsi_model import TeacherModel
from evaluate_distillation import linear_probe
from evaluate_v4_downstream import mlp_probe
from eval_c16_equivalence import load_c16, stratified_slide_split, auc_cluster_tost

GPFM_CKPT = "/path/to/GPFM.pth"
TEACHERS = ["conch", "phikon", "uni"]
DELTA = 0.05
BS = 32
PROBE_REPS = 8    # matches K in eval_c16_fullstack.py

# DINOv2 / GPFM inference preprocessing is ImageNet mean-std at 224.
IN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IN_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# A correctly loaded ViT-L pathology encoder sits far above this on the
# lesion-annotated tumour/normal probe; the hibou extraction bug that this
# project hit earlier showed up precisely as a probe collapsing toward
# chance, so the gate is deliberately blunt.
SANITY_MIN_AUROC = 0.80


def load_gpfm(device):
    """Rebuild GPFM as a timm DINOv2 ViT-L/14 and load the released weights."""
    import timm
    sd = torch.load(GPFM_CKPT, map_location="cpu")["teacher"]
    out = {}
    for k, v in sd.items():
        if not k.startswith("backbone."):
            continue                                    # drop the DINO head
        k = k[len("backbone."):]
        m = re.match(r"blocks\.\d+\.(\d+)\.(.*)", k)     # un-chunk BlockChunk
        if m:
            k = f"blocks.{m.group(1)}.{m.group(2)}"
        out[k] = v

    model = timm.create_model("vit_large_patch14_dinov2.lvd142m",
                              pretrained=False, num_classes=0,
                              img_size=224, dynamic_img_size=False)
    missing, unexpected = model.load_state_dict(out, strict=False)
    missing = [k for k in missing if "mask_token" not in k]
    unexpected = [k for k in unexpected if "mask_token" not in k]
    print(f"[gpfm] loaded {len(out)} tensors; missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"[gpfm]   missing e.g. {missing[:4]}")
    if unexpected:
        print(f"[gpfm]   unexpected e.g. {unexpected[:4]}")
    if missing or unexpected:
        raise SystemExit("[gpfm] ABORT: state dict did not map cleanly; "
                         "a partial load would produce a fake negative result")
    return model.eval().to(device)


@torch.no_grad()
def extract(model, h5_path, indices, device, imagenet_norm=True, bs=BS):
    mean, std = IN_MEAN.to(device), IN_STD.to(device)
    feats = []
    with h5py.File(h5_path, "r", swmr=True) as f:
        patches = f["patches"]
        for i in range(0, len(indices), bs):
            chunk = np.sort(indices[i:i + bs])
            x = torch.from_numpy(patches[chunk]).permute(0, 3, 1, 2).float().to(device) / 255.0
            if imagenet_norm:
                x = (x - mean) / std
            if hasattr(model, "forward_features"):
                tok = model.forward_features(x)
                v = tok[:, 0, :] if tok.ndim == 3 else tok
            else:
                v = model(x)["feat"]
            feats.append(v.float().cpu())
    return torch.cat(feats).numpy()


def three_state(lo, hi, margin=DELTA):
    """Equivalent / inequivalent / inconclusive from the 90% cluster CI, using
    the same rule as the atlas. Verified to reproduce the published 10/8/18
    split exactly on the 36 atlas pairs before being used here."""
    import math
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return "inconclusive"
    if lo > -margin and hi < margin:
        return "equivalent"
    if lo >= margin or hi <= -margin:
        return "inequivalent"
    return "inconclusive"


def _one_fit(f_tr, y_tr, f_te, y_te, device, kind, k):
    torch.manual_seed(2000 + k)
    fn = linear_probe if kind == "linear" else mlp_probe
    r = fn(torch.from_numpy(f_tr).to(device), torch.from_numpy(y_tr).to(device),
           torch.from_numpy(f_te).to(device), torch.from_numpy(y_te).to(device),
           n_classes=2)
    p = r.get("probs")
    p = p.cpu().numpy() if isinstance(p, torch.Tensor) else np.asarray(p)
    return p[:, 1] if p.ndim == 2 else p


def probe_scores(f_tr, y_tr, f_te, y_te, device, kind, reps=PROBE_REPS):
    """Positive-class probability, averaged over `reps` probe fits.

    The probe has random initialisation, and a single fit moves the AUROC by
    enough to flip a marginal verdict: two runs of this script gave GPFM
    sanity AUROCs of 0.9295 and 0.9444. Averaging K fits with seeds
    2000+k is the same convention eval_c16_fullstack.py uses, so a verdict
    here is comparable to the atlas full-stack verdicts rather than to a
    single-fit estimate."""
    return np.mean([_one_fit(f_tr, y_tr, f_te, y_te, device, kind, k)
                    for k in range(reps)], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out", default="outputs/v4_full/gpfm_equiv.json")
    ap.add_argument("--test_frac", type=float, default=0.5)
    ap.add_argument("--probe_reps", type=int, default=PROBE_REPS)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    idx, labels, slide_ids = load_c16(a.h5)
    tr, te, n_test_slides, n_slides = stratified_slide_split(
        slide_ids, labels, idx, a.test_frac)
    y_tr, y_te, grp_te = labels[tr].astype(np.int64), labels[te].astype(np.int64), slide_ids[te]
    print(f"[split] {n_slides} slides -> {len(tr)}/{len(te)} tiles, "
          f"{n_test_slides} test slides, {int(y_te.sum())} tumour tiles", flush=True)

    gpfm = load_gpfm(device)
    s_tr = extract(gpfm, a.h5, tr, device)
    s_te = extract(gpfm, a.h5, te, device)
    del gpfm; torch.cuda.empty_cache()

    # Sanity gate. A loader bug looks identical to a genuine inequivalence
    # verdict, so nothing is reported until the student clears it.
    from sklearn.metrics import roc_auc_score
    s_scores = probe_scores(s_tr, y_tr, s_te, y_te, device, "linear", a.probe_reps)
    s_auc = roc_auc_score(y_te, s_scores)
    print(f"[sanity] GPFM linear-probe AUROC = {s_auc:.4f} "
          f"(gate {SANITY_MIN_AUROC})", flush=True)
    if not np.isfinite(s_auc) or s_auc < SANITY_MIN_AUROC:
        raise SystemExit("[gpfm] ABORT: probe below the sanity gate. Treat this as a "
                         "loading or preprocessing fault, not as evidence about GPFM.")

    rows = []
    for tname in TEACHERS:
        print(f"\n[teacher] {tname}", flush=True)
        tm = TeacherModel(tname).eval().to(device)
        t_tr = extract(tm, a.h5, tr, device)
        t_te = extract(tm, a.h5, te, device)
        del tm; torch.cuda.empty_cache()

        rec = {"teacher": tname, "student": "GPFM",
               "n_test_slides": int(n_test_slides), "n_test_tiles": int(len(te)),
               "n_tumour_tiles": int(y_te.sum())}
        for kind in ("linear", "mlp"):
            t_s = probe_scores(t_tr, y_tr, t_te, y_te, device, kind, a.probe_reps)
            s_s = probe_scores(s_tr, y_tr, s_te, y_te, device, kind, a.probe_reps)
            r = auc_cluster_tost(y_te, t_s, s_s, grp_te, margin=DELTA)
            rec[f"{kind}_t_auc"] = round(float(r["t_auc"]), 4)
            rec[f"{kind}_s_auc"] = round(float(r["s_auc"]), 4)
            rec[f"{kind}_auc_d"] = round(float(r["d"]), 4)
            rec[f"{kind}_ci"] = [round(float(r["lo"]), 4), round(float(r["hi"]), 4)]
            st = three_state(r["lo"], r["hi"], DELTA)
            rec[f"{kind}_state"] = st
            rec[f"{kind}_inflation"] = round(float(r["inflation"]), 3)
            print(f"  {kind}: teacher {r['t_auc']:.4f} vs GPFM {r['s_auc']:.4f}  "
                  f"d={r['d']:+.4f} CI[{r['lo']:+.4f},{r['hi']:+.4f}]  {st}", flush=True)
        rec["both"] = (rec["linear_state"] if rec["linear_state"] == rec["mlp_state"]
                       else "inconclusive")
        rows.append(rec)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"sanity_gpfm_linear_auroc": round(float(s_auc), 4),
                               "delta": DELTA, "probe_reps": a.probe_reps, "rows": rows}, indent=2))
    print(f"\n[wrote] {out}")
    for r in rows:
        print(f"  {r['teacher']:8} vs GPFM -> {r['both']}")


if __name__ == "__main__":
    main()
