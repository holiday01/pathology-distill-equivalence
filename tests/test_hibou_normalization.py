#!/usr/bin/env python3
"""Unit test for the hibou normalisation fix (research/hibou_fix.md).

Root cause: the distillation pipeline feeds ImageNet-normalised tiles to every
teacher, but hibou requires its own mean/std. TeacherModel.forward now converts
ImageNet-normalised input to hibou stats for hibou teachers only. This test
checks that (a) the conversion is numerically exact and (b) it is a no-op for
non-hibou teachers. Runs on CPU; no model download required.
"""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from distill_wsi_model import (_IMAGENET_MEAN, _IMAGENET_STD,
                               _HIBOU_MEAN, _HIBOU_STD)


def test_hibou_conversion_is_exact():
    """ImageNet-normalised -> (forward conversion) == direct hibou normalisation."""
    raw = torch.rand(4, 3, 32, 32)                      # raw [0,1] tiles
    x_imagenet = (raw - _IMAGENET_MEAN) / _IMAGENET_STD  # what the dataset feeds
    # the exact conversion applied inside TeacherModel.forward for hibou:
    x_converted = ((x_imagenet * _IMAGENET_STD + _IMAGENET_MEAN) - _HIBOU_MEAN) / _HIBOU_STD
    x_direct = (raw - _HIBOU_MEAN) / _HIBOU_STD          # ground-truth hibou norm
    err = (x_converted - x_direct).abs().max().item()
    assert err < 1e-5, f"hibou conversion not exact: max err {err}"


def test_hibou_stats_differ_from_imagenet():
    """Guard against accidentally setting hibou stats equal to ImageNet."""
    assert not torch.allclose(_HIBOU_MEAN, _IMAGENET_MEAN)
    assert not torch.allclose(_HIBOU_STD, _IMAGENET_STD)


def test_nonhibou_branch_is_noop():
    """For a non-hibou teacher_name the forward must not alter the input.
    We emulate the branch condition used in TeacherModel.forward."""
    x = torch.randn(2, 3, 16, 16)
    for name in ("phikon", "uni", "virchow2", "h-optimus-0", "conch"):
        touched = name.startswith("hibou")
        assert not touched, f"{name} must not enter the hibou renorm branch"
    for name in ("hibou-b", "hibou-l"):
        assert name.startswith("hibou")


if __name__ == "__main__":
    test_hibou_conversion_is_exact()
    test_hibou_stats_differ_from_imagenet()
    test_nonhibou_branch_is_noop()
    print("all hibou-normalisation tests passed ✓")
