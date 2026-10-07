#!/usr/bin/env python3
"""Count forward FLOPs for every teacher and every student with one counter.

The earlier outputs/fm_flops.json mixed conventions: teachers were ptflops
MACs x 2, which misses attention implemented through
F.scaled_dot_product_attention (so ViT-L UNI undercounted), while the student
values in paper_metrics.csv were timm MAC counts (1.26/4.61/17.58). Phikon
(ViT-B/16) -> ViT-B/16 then reported a GFLOPs ratio of 1.92 instead of 1.

Here every model is counted with torch.utils.flop_counter.FlopCounterMode,
which traces aten matmul/conv/SDPA ops, on one 224x224 image. Following the
convention of the ViT literature (ViT-B/16 = 17.6 GFLOPs), we report
GFLOPs = MACs = FlopCounterMode count / 2.

Output: outputs/fm_flops.json (the ptflops version is kept as
outputs/fm_flops_ptflops.json).
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).parent))
from distill_wsi_model import FM_REGISTRY, StudentModel, TeacherModel

STUDENTS = {"vit-tiny": "vit_tiny_patch16_224",
            "vit-small": "vit_small_patch16_224",
            "vit-base": "vit_base_patch16_224"}


def gmacs(fn, x):
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        fn(x)
    return fc.get_total_flops() / 2 / 1e9


def n_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


def main():
    out = Path("outputs/fm_flops.json")
    old = Path("outputs/fm_flops_ptflops.json")
    if out.exists() and not old.exists():
        shutil.copy(out, old)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1, 3, 224, 224, device=dev)
    res = {"_convention": "GFLOPs = GMACs (FlopCounterMode/2), 224x224, batch 1"}
    for key, arch in STUDENTS.items():
        s = StudentModel(arch, 256, 0, pretrained=False).to(dev).eval()
        res[f"student_{key}"] = {"gflops": gmacs(lambda t: s(t, return_patches=False), x),
                                 "params_M": n_params(s), "input": 224}
        print(key, res[f"student_{key}"], flush=True)
        del s
    for name in FM_REGISTRY:
        try:
            t = TeacherModel(name).to(dev).eval()
            res[name] = {"gflops": gmacs(lambda z: t(z, return_patches=False), x),
                         "params_M": n_params(t), "input": 224}
            print(name, res[name], flush=True)
            del t
        except Exception as e:  # gated or unavailable weights
            res[name] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
            print(name, "FAIL", res[name]["error"], flush=True)
        if dev == "cuda":
            torch.cuda.empty_cache()
    out.write_text(json.dumps(res, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
