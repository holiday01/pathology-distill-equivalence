#!/usr/bin/env python3
"""H6 named field-audit flip: does H0-mini's published 'comparable to its
teacher' claim survive slide-clustering?

H0-mini (bioptimus/H0-mini, Filiot et al. 2025) is a ViT-B distilled from
H-Optimus-0 and reported as comparable to its teacher under metric proximity
without confidence intervals or slide-clustering. We run the SAME annotated
CAMELYON16 AUROC equivalence test (lesion-annotated tiles, 41 test slides,
slide-cluster TOST, delta=0.05) on the (H-Optimus-0, H0-mini) pair and report
whether it certifies equivalence or flips to inconclusive / inequivalent.

Both models are loaded and normalised the way the bioptimus authors specify
(native mean/std, SwiGLU MLP for H0-mini) so the comparison reproduces the
conditions under which the 'comparable' claim was made.
"""
import sys
from pathlib import Path
import numpy as np, h5py, torch
sys.path.insert(0, str(Path(__file__).parent))
from evaluate_distillation import linear_cka, cohen_kappa, linear_probe
from evaluate_v4_downstream import mlp_probe
from eval_c16_equivalence import load_c16, stratified_slide_split, auc_cluster_tost

H5 = "/path/to/cache/patches/patches_c16_annotated.h5"
DELTA = 0.05
# Native bioptimus normalisation (H0-mini / H-Optimus-0 pretrained_cfg)
BIO_MEAN = torch.tensor([0.707223, 0.578729, 0.703617]).view(3, 1, 1)
BIO_STD = torch.tensor([0.211883, 0.230117, 0.177517]).view(3, 1, 1)
BS = 48  # small: H-Optimus-0 (1.13B) must fit alongside the seed2 sweep


def load_teacher(device):
    import timm
    m = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True,
                          num_classes=0, init_values=1e-5,
                          dynamic_img_size=False).eval().to(device)
    return m


def load_h0mini(device):
    import timm
    m = timm.create_model("hf-hub:bioptimus/H0-mini", pretrained=True,
                          mlp_layer=timm.layers.SwiGLUPacked,
                          act_layer=torch.nn.SiLU).eval().to(device)
    return m


@torch.no_grad()
def extract_cls(model, indices, device, bs=BS):
    """CLS-token features under native bioptimus normalisation."""
    mean = BIO_MEAN.to(device); std = BIO_STD.to(device)
    feats = []
    with h5py.File(H5, "r", swmr=True) as f:
        patches = f["patches"]
        for i in range(0, len(indices), bs):
            chunk = np.sort(indices[i:i + bs])
            arr = patches[chunk]
            x = torch.from_numpy(arr).permute(0, 3, 1, 2).float().to(device) / 255.0
            x = (x - mean) / std
            tok = model.forward_features(x)        # (B, 1+reg+N, D)
            cls = tok[:, 0, :] if tok.ndim == 3 else tok
            feats.append(cls.float().cpu())
    return torch.cat(feats).numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    idx, labels, slide_ids = load_c16(H5)
    tr, te, n_test, n_tot = stratified_slide_split(slide_ids, labels, idx, 0.5)
    y_tr = torch.from_numpy(labels[tr]).to(device)
    y_te = labels[te]; grp_te = slide_ids[te]
    print(f"[split] {n_tot} slides -> {len(tr)} train / {len(te)} test tiles, "
          f"{n_test} test slides", flush=True)

    teacher = load_teacher(device)
    t_tr = extract_cls(teacher, tr, device); t_te = extract_cls(teacher, te, device)
    del teacher; torch.cuda.empty_cache()
    print(f"[teacher] H-Optimus-0 feat dim {t_te.shape[1]}", flush=True)

    student = load_h0mini(device)
    s_tr = extract_cls(student, tr, device); s_te = extract_cls(student, te, device)
    del student; torch.cuda.empty_cache()
    print(f"[student] H0-mini feat dim {s_te.shape[1]}  "
          f"CKA={float(linear_cka(torch.from_numpy(t_te), torch.from_numpy(s_te))):.3f}",
          flush=True)

    res = {}
    for kind, fn in (("linear", linear_probe), ("mlp", mlp_probe)):
        tr_r = fn(torch.from_numpy(t_tr).to(device), y_tr,
                  torch.from_numpy(t_te).to(device),
                  torch.from_numpy(y_te).to(device), n_classes=2)
        sr = fn(torch.from_numpy(s_tr).to(device), y_tr,
                torch.from_numpy(s_te).to(device),
                torch.from_numpy(y_te).to(device), n_classes=2)
        prob = lambda r: (r["probs"].cpu().numpy() if isinstance(r["probs"], torch.Tensor)
                          else np.asarray(r["probs"]))[:, 1]
        au = auc_cluster_tost(y_te, prob(tr_r), prob(sr), grp_te, margin=DELTA)
        lo, hi = au["lo"], au["hi"]
        state = ("equivalent" if (hi < DELTA and lo > -DELTA)
                 else "inequivalent" if (lo > DELTA or hi < -DELTA) else "inconclusive")
        res[kind] = (au, state)
        print(f"[{kind}] H-Optimus-0 AUC={au['t_auc']:.3f}  H0-mini AUC={au['s_auc']:.3f}  "
              f"dAUC={au['d']:+.3f}  CI[{lo:+.3f},{hi:+.3f}] -> {state.upper()}  "
              f"infl={au['inflation']:.2f}x", flush=True)

    lin = res["linear"][1]
    print("\n=== VERDICT (H-Optimus-0 -> H0-mini, lesion-annotated 41-slide C16) ===")
    print(f"linear AUROC equivalence at delta={DELTA}: {lin.upper()}")
    if lin != "equivalent":
        print("=> FLIP: the published 'comparable' claim does NOT certify "
              "equivalence under slide-clustering.")
    else:
        print("=> H0-mini's comparability claim survives slide-clustering.")
    import json
    json.dump({k: {**v[0], "state": v[1]} for k, v in res.items()},
              open("outputs/v4_full/h0mini_flip.json", "w"), indent=1)
    print("[wrote] outputs/v4_full/h0mini_flip.json")


if __name__ == "__main__":
    main()
