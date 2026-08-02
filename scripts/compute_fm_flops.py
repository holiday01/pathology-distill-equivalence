#!/usr/bin/env python3
"""Compute GFLOPs for all teachers + the ViT-S student at their native resolutions.
Output: outputs/fm_flops.json — reused by make_paper_metrics.py.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from distill_wsi_model import FM_REGISTRY, StudentModel, TeacherModel


class TeacherFeatureOnly(torch.nn.Module):
    """Wrap TeacherModel so ptflops sees a tensor-out forward()."""
    def __init__(self, tm):
        super().__init__()
        self.tm = tm

    def forward(self, x):
        o = self.tm(x, return_patches=False)
        return o["feat"]


class StudentFeatureOnly(torch.nn.Module):
    def __init__(self, sm):
        super().__init__()
        self.sm = sm

    def forward(self, x):
        o = self.sm(x, return_patches=False)
        return o["feat"]


def count_flops(model: torch.nn.Module, input_size: tuple):
    """Return MACs, Params using ptflops. We report GFLOPs = 2 × GMACs."""
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(
            model, input_size, as_strings=False, print_per_layer_stat=False, verbose=False,
        )
    return macs, params


def main():
    out_path = Path("outputs/fm_flops.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {}

    # Student ViT-S at 224
    print("[student] vit_small_patch16_224 @ 224")
    s = StudentFeatureOnly(StudentModel("vit_small_patch16_224", 256, 0, pretrained=False).cuda().eval())
    macs, params = count_flops(s, (3, 224, 224))
    results["student_vit_small"] = {"gflops": 2 * macs / 1e9, "params_M": params / 1e6, "input": 224}
    print(f"  GFLOPs={results['student_vit_small']['gflops']:.2f}  params={results['student_vit_small']['params_M']:.2f}M")
    del s
    torch.cuda.empty_cache()

    # Teachers
    for name in FM_REGISTRY:
        input_res = 224  # all teachers process 224x224 distillation patches (CONCH trunk -> 196 tokens)
        print(f"[teacher] {name} @ {input_res}")
        try:
            t = TeacherFeatureOnly(TeacherModel(name).cuda().eval())
            macs, params = count_flops(t, (3, input_res, input_res))
            results[name] = {"gflops": 2 * macs / 1e9, "params_M": params / 1e6, "input": input_res}
            print(f"  GFLOPs={results[name]['gflops']:.2f}  params={results[name]['params_M']:.2f}M")
            del t
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FAIL {type(e).__name__}: {str(e)[:120]}")
            results[name] = {"error": str(e)[:200]}

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
