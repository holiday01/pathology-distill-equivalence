#!/usr/bin/env python3
"""
WSI Foundation Model 蒸餾訓練框架
====================================
支援多種蒸餾策略：
  - response:  logit-based（軟標籤蒸餾）
  - feature:   中間層特徵對齊（ViTKD 風格）
  - relation:  patch 間關係蒸餾（HVisKD 風格）
  - hybrid:    feature + relation 組合（GPFM 風格）

用法：
  python distill_wsi_model.py --teacher conch --student vit_small --method hybrid
  python distill_wsi_model.py --teacher uni --student resnet50 --method feature --epochs 50
  python distill_wsi_model.py --demo  # 快速測試（無需真實資料）
"""

import argparse
import time
import os
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ──────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────

class ResponseDistillLoss(nn.Module):
    """Response-based KD（Hinton 原始方法）"""
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.T = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, labels=None):
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.T, dim=-1),
            F.softmax(teacher_logits / self.T, dim=-1),
            reduction="batchmean"
        ) * (self.T ** 2)
        if labels is not None:
            hard_loss = F.cross_entropy(student_logits, labels)
            return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        return soft_loss


class FeatureDistillLoss(nn.Module):
    """Feature-based KD（ViTKD 風格）：對齊中間層特徵"""
    def __init__(self, student_dim: int, teacher_dim: int, projector: str = "linear"):
        super().__init__()
        # 投影層：讓學生特徵對齊教師特徵空間
        if projector == "mlp":
            hidden = max(student_dim, teacher_dim)
            self.projector = nn.Sequential(
                nn.Linear(student_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, teacher_dim),
            )
        else:
            self.projector = nn.Linear(student_dim, teacher_dim)

    def forward(self, student_feat, teacher_feat):
        student_proj = self.projector(student_feat)
        return F.mse_loss(student_proj, teacher_feat.detach())


class RelationDistillLoss(nn.Module):
    """Relation-based KD（HVisKD 風格）：patch 間關係蒸餾"""
    def __init__(self):
        super().__init__()

    def forward(self, student_feats, teacher_feats):
        """
        student_feats: (B, N, D_s)
        teacher_feats: (B, N, D_t)
        """
        # 計算 patch 間餘弦相似度矩陣
        s_norm = F.normalize(student_feats, dim=-1)
        t_norm = F.normalize(teacher_feats, dim=-1)

        s_relation = torch.bmm(s_norm, s_norm.transpose(1, 2))  # (B, N, N)
        t_relation = torch.bmm(t_norm, t_norm.transpose(1, 2))  # (B, N, N)

        return F.mse_loss(s_relation, t_relation.detach())


class HybridDistillLoss(nn.Module):
    """Hybrid KD（GPFM 風格）：feature + relation + response"""
    def __init__(self, student_dim: int, teacher_dim: int,
                 w_feat: float = 1.0, w_rel: float = 0.5, w_resp: float = 0.3,
                 projector: str = "linear"):
        super().__init__()
        self.feature_loss = FeatureDistillLoss(student_dim, teacher_dim, projector=projector)
        self.relation_loss = RelationDistillLoss()
        self.w_feat = w_feat
        self.w_rel = w_rel
        self.w_resp = w_resp

    def forward(self, student_out, teacher_out):
        """
        student_out / teacher_out: dict 包含 'logits', 'feat', 'patch_feats'
        """
        total_loss = 0
        if "feat" in student_out and "feat" in teacher_out:
            total_loss += self.w_feat * self.feature_loss(
                student_out["feat"], teacher_out["feat"]
            )
        if "patch_feats" in student_out and "patch_feats" in teacher_out:
            total_loss += self.w_rel * self.relation_loss(
                student_out["patch_feats"], teacher_out["patch_feats"]
            )
        return total_loss


# ──────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────

class StudentModel(nn.Module):
    """
    輕量學生模型：包裝 timm backbone + 投影頭
    可配置：resnet50, vit_small, vit_base, mobilevit
    """
    def __init__(self, backbone_name: str = "vit_small_patch16_224",
                 embed_dim: int = 256, n_classes: int = 0,
                 pretrained: bool = False):
        super().__init__()
        try:
            import timm
            self.backbone = timm.create_model(
                backbone_name, pretrained=pretrained, num_classes=0
            )
            feat_dim = self.backbone.num_features
        except (ImportError, RuntimeError):
            # Fallback lightweight model（timm 未安裝時使用）
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 128, 4, 4), nn.ReLU(),
                nn.Conv2d(128, 256, 4, 2), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten()
            )
            feat_dim = 256

        self.embed_dim = embed_dim  # proj_head 輸出維度，作為 feature loss 的輸入維度
        self.proj_head = nn.Linear(feat_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes) if n_classes > 0 else nn.Identity()

    def forward(self, x, return_patches: bool = False):
        # 用 forward_features 取得 patch tokens（ViT 回傳 [B, N, D]，CNN 回傳 [B, D]）
        if hasattr(self.backbone, "forward_features"):
            all_feats = self.backbone.forward_features(x)  # (B, N, D) for ViT
        else:
            all_feats = self.backbone(x)

        if len(all_feats.shape) == 3:  # ViT: (B, N, D)
            patch_feats = all_feats[:, 1:, :]  # 去掉 CLS token，保留 patch tokens
            feat = all_feats[:, 0, :]          # CLS token 作為全局特徵
        else:
            patch_feats = all_feats.unsqueeze(1)
            feat = all_feats

        feat_proj = self.proj_head(feat)
        logits = self.classifier(feat_proj)
        out = {"logits": logits, "feat": feat_proj}
        if return_patches:
            out["patch_feats"] = patch_feats
        return out



# Foundation Model registry
# fields per entry: (model_id, embed_dim, loader_type, extra_kwargs)
# loader_type:
#   hf_vit       - transformers.ViTModel (phikon)
#   hf_auto      - transformers.AutoModel (phikon-v2)
#   hf_auto_trc  - AutoModel(trust_remote_code=True) (hibou; needs onnx stub)
#   timm         - timm.create_model (UNI, H-Optimus, GigaPath; extra_kwargs passed through)
#   timm_virchow - timm with SwiGLU (Virchow, Virchow2)
#   timm_uni2h   - timm with SwiGLU + 8 reg tokens (UNI2-h)
#   conch        - MahmoodLab CONCH via conch package
FM_REGISTRY = {
    "phikon":        ("owkin/phikon",                      768,  "hf_vit",        {}),
    "phikon-v2":     ("owkin/phikon-v2",                   1024, "hf_auto",       {}),
    "uni":           ("hf-hub:MahmoodLab/UNI",             1024, "timm",          {"init_values": 1e-5, "dynamic_img_size": True}),
    "uni2-h":        ("hf-hub:MahmoodLab/UNI2-h",          1536, "timm_uni2h",    {}),
    "virchow":       ("hf-hub:paige-ai/Virchow",           1280, "timm_virchow",  {}),
    "virchow2":      ("hf-hub:paige-ai/Virchow2",          1280, "timm_virchow",  {}),
    "h-optimus-0":   ("hf-hub:bioptimus/H-optimus-0",      1536, "timm",          {"init_values": 1e-5, "dynamic_img_size": False}),
    "prov-gigapath": ("hf-hub:prov-gigapath/prov-gigapath", 1536, "timm",         {}),
    "hibou-b":       ("histai/hibou-b",                    768,  "hf_auto_trc",   {}),
    "hibou-l":       ("histai/hibou-L",                    1024, "hf_auto_trc",   {}),
    "conch":         ("MahmoodLab/CONCH",                  768,  "conch",         {}),
    "midnight":      ("kaiko-ai/midnight",                 1536, "hf_auto_trc",   {}),
    "h-optimus-1":   ("hf-hub:bioptimus/H-optimus-1",      1536, "timm",          {"init_values": 1e-5, "dynamic_img_size": False}),
}

# Backwards-compat alias (older code may still reference HF_MODELS)
HF_MODELS = {k: v[:3] for k, v in FM_REGISTRY.items() if v[2] in ("hf_vit", "hf_auto")}


def _install_onnx_stub():
    """Hibou's configuration_dinov2.py imports transformers.onnx (removed in 5.x).
    Inject a minimal stub so import succeeds; OnnxConfig is unused at inference."""
    import sys, types
    if "transformers.onnx" not in sys.modules:
        stub = types.ModuleType("transformers.onnx")
        class OnnxConfig: pass
        stub.OnnxConfig = OnnxConfig
        sys.modules["transformers.onnx"] = stub
    # Hibou's modeling_dinov2.py also imports get_aligned_output_features_output_indices
    # from transformers.utils.backbone_utils (removed in 5.x). Inject a stub — it's only
    # used by DinoV2Backbone which we don't instantiate.
    from transformers.utils import backbone_utils
    if not hasattr(backbone_utils, "get_aligned_output_features_output_indices"):
        def _stub(out_features, out_indices, stage_names):
            return out_features, out_indices
        backbone_utils.get_aligned_output_features_output_indices = _stub
    # Head-pruning helpers removed from transformers 5.x; hibou imports them but
    # never calls them during encoding. Stub returns match old signatures.
    from transformers import pytorch_utils as _pu
    import torch.nn as _nn
    if not hasattr(_pu, "find_pruneable_heads_and_indices"):
        def _fphi(heads, n_heads, head_size, already_pruned_heads):
            return set(), _nn.Parameter(torch.empty(0), requires_grad=False)
        _pu.find_pruneable_heads_and_indices = _fphi
    if not hasattr(_pu, "prune_linear_layer"):
        def _pll(layer, index, dim=0):
            return layer
        _pu.prune_linear_layer = _pll


class TeacherModel(nn.Module):
    """
    Foundation Model teacher wrapper. Supports 11 pathology FMs via FM_REGISTRY.
    Outputs {"feat": CLS, "patch_feats": last-196 patch tokens} at 224x224 input.
    """
    def __init__(self, teacher_name: str = "phikon"):
        super().__init__()
        key = teacher_name.lower()
        if key not in FM_REGISTRY:
            raise ValueError(f"Unknown teacher '{teacher_name}'. Available: {list(FM_REGISTRY)}")
        model_id, embed_dim, loader_type, extra = FM_REGISTRY[key]
        self.teacher_name = key
        self.loader_type = loader_type
        self.embed_dim = embed_dim
        print(f"  載入 FM [{loader_type}]: {model_id}")

        if loader_type == "hf_vit":
            from transformers import ViTModel
            self.backbone = ViTModel.from_pretrained(model_id, add_pooling_layer=False)
        elif loader_type == "hf_auto":
            from transformers import AutoModel
            self.backbone = AutoModel.from_pretrained(model_id)
        elif loader_type == "hf_auto_trc":
            _install_onnx_stub()
            from transformers import AutoModel
            self.backbone = AutoModel.from_pretrained(model_id, trust_remote_code=True)
            # transformers 5.x no longer exposes get_head_mask on all model classes; hibou
            # relies on it internally. Patch a no-op version.
            if not hasattr(self.backbone, "get_head_mask"):
                def _ghm(self_, head_mask, num_hidden_layers, is_attention_chunked=False):
                    return [None] * num_hidden_layers
                import types as _types
                self.backbone.get_head_mask = _types.MethodType(_ghm, self.backbone)
        elif loader_type == "timm":
            import timm
            self.backbone = timm.create_model(model_id, pretrained=True, num_classes=0, **extra)
        elif loader_type == "timm_virchow":
            import timm
            from timm.layers import SwiGLUPacked
            self.backbone = timm.create_model(
                model_id, pretrained=True, num_classes=0,
                mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU, **extra,
            )
        elif loader_type == "timm_uni2h":
            import timm
            from timm.layers import SwiGLUPacked
            self.backbone = timm.create_model(
                "vit_giant_patch14_224",
                pretrained=True,
                pretrained_cfg_overlay={"hf_hub_id": "MahmoodLab/UNI2-h"},
                img_size=224, patch_size=14, depth=24, num_heads=24,
                init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2,
                num_classes=0, no_embed_class=True,
                mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU,
                reg_tokens=8, dynamic_img_size=True,
            )
        elif loader_type == "conch":
            from huggingface_hub import hf_hub_download
            from conch.open_clip_custom import create_model_from_pretrained
            ckpt = hf_hub_download(model_id, "pytorch_model.bin")
            coca, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=ckpt)
            # Only keep visual trunk (ViT-B/16 → 768-d); drop text tower
            self.backbone = coca.visual.trunk
        else:
            raise ValueError(f"unknown loader_type {loader_type}")

    @torch.no_grad()
    def forward(self, x, return_patches: bool = False):
        lt = self.loader_type
        if self.teacher_name.startswith("hibou"):
            # Pipeline feeds ImageNet-normalised tensors; hibou needs its own
            # stats. Convert exactly (undo ImageNet, apply hibou). Only the
            # hibou branch is touched, so non-hibou teachers are byte-identical.
            im = _IMAGENET_MEAN.to(x.device, x.dtype); ist = _IMAGENET_STD.to(x.device, x.dtype)
            hm = _HIBOU_MEAN.to(x.device, x.dtype); hst = _HIBOU_STD.to(x.device, x.dtype)
            x = ((x * ist + im) - hm) / hst
        if lt in ("hf_vit", "hf_auto", "hf_auto_trc"):
            out = self.backbone(pixel_values=x)
            all_feats = out.last_hidden_state  # (B, N+S, D), S = special tokens
        else:
            # timm / timm_virchow / conch (VisionTransformer)
            all_feats = self.backbone.forward_features(x)

        if all_feats.ndim == 3:
            # CLS token: index 0 is convention across all supported FMs
            feat = all_feats[:, 0, :]
            # patch tokens: take LAST 196 (skips any register tokens in Virchow/UNI2-h/Hibou)
            patch_feats = all_feats[:, -196:, :]
        else:
            feat = all_feats
            patch_feats = all_feats.unsqueeze(1)

        out = {"feat": feat}
        if return_patches:
            out["patch_feats"] = patch_feats
        return out


# ──────────────────────────────────────────────────────────
# Toy dataset（demo 用）
# ──────────────────────────────────────────────────────────

class DemoPatchDataset(Dataset):
    """合成 patch 資料集（無需真實 WSI）"""
    def __init__(self, n_slides: int = 50, patches_per_slide: int = 64,
                 patch_size: int = 224, n_classes: int = 4):
        self.patches = torch.randn(n_slides * patches_per_slide, 3, patch_size, patch_size)
        self.labels = torch.randint(0, n_classes, (n_slides * patches_per_slide,))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx], self.labels[idx]


# ImageNet 統計量（Phikon / Phikon-v2 / timm ViT 皆使用同樣正規化）
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Hibou requires its own normalisation (histai AutoImageProcessor), NOT ImageNet.
# Feeding ImageNet-normalised tiles to hibou degraded its features and held the
# distilled students near zero (best_val ~0.02 vs ~0.3 for other teachers).
# TeacherModel.forward converts ImageNet-normalised input to these stats for
# hibou teachers only; all other teachers are untouched.
_HIBOU_MEAN = torch.tensor([0.7068, 0.5755, 0.7220]).view(3, 1, 1)
_HIBOU_STD = torch.tensor([0.1950, 0.2316, 0.1816]).view(3, 1, 1)


class WSIPatchDataset(Dataset):
    """從 extract_patches.py 產出的 HDF5 讀 real WSI patches，
    做 ImageNet 正規化後回傳 (tensor[C,H,W], label)。"""

    def __init__(self, h5_path: str, normalize: bool = True,
                 indices=None):
        self.h5_path = h5_path
        self.normalize = normalize
        self._h5 = None  # 延遲開啟（讓 num_workers > 0 時 fork 後再開）
        with h5py.File(h5_path, "r") as h5:
            self.length = h5["patches"].shape[0]
            self.labels_cache = h5["labels"][:].astype(np.int64)
        if indices is None:
            self.indices = np.arange(self.length)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._ensure_open()
        real_idx = int(self.indices[idx])
        patch = self._h5["patches"][real_idx]  # (H, W, 3) uint8
        label = int(self.labels_cache[real_idx])

        t = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0  # (3, H, W)
        if self.normalize:
            t = (t - _IMAGENET_MEAN) / _IMAGENET_STD
        return t, label


# ──────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────

def train_one_epoch(student, teacher, loader, optimizer, criterion, device, epoch,
                    grad_clip: float = 0.0, warmup_scheduler=None, params_for_clip=None):
    student.train()
    total_loss = 0
    n_batches = 0
    for i, (patches, labels) in enumerate(loader):
        patches, labels = patches.to(device), labels.to(device)

        # Teacher forward（無梯度）
        teacher_out = teacher(patches, return_patches=True)

        # Student forward
        student_out = student(patches, return_patches=True)

        # Loss（根據 criterion 類型傳入不同格式）
        if isinstance(criterion, ResponseDistillLoss):
            loss = criterion(student_out["logits"], teacher_out.get("logits", student_out["logits"]), labels)
        elif isinstance(criterion, FeatureDistillLoss):
            loss = criterion(student_out["feat"], teacher_out["feat"])
        elif isinstance(criterion, RelationDistillLoss):
            loss = criterion(
                student_out["patch_feats"].unsqueeze(0) if student_out["patch_feats"].dim() == 2 else student_out["patch_feats"],
                teacher_out["patch_feats"].unsqueeze(0) if teacher_out["patch_feats"].dim() == 2 else teacher_out["patch_feats"]
            )
        else:
            loss = criterion(student_out, teacher_out)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0 and params_for_clip is not None:
            torch.nn.utils.clip_grad_norm_(params_for_clip, grad_clip)
        optimizer.step()
        if warmup_scheduler is not None:
            warmup_scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if i % 20 == 0:
            print(f"    Epoch {epoch} | Batch {i}/{len(loader)} | Loss: {loss.item():.4f}")

    return total_loss / n_batches


def evaluate(student, loader, device):
    """評估學生模型（簡單準確率）"""
    student.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for patches, labels in loader:
            patches, labels = patches.to(device), labels.to(device)
            out = student(patches)
            if isinstance(out["logits"], torch.Tensor) and out["logits"].shape[-1] > 1:
                preds = out["logits"].argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
    return correct / total if total > 0 else 0.0


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WSI FM 蒸餾訓練")
    parser.add_argument("--teacher", type=str, default="phikon",
                        help="教師模型 (phikon/phikon-v2/vit_large_patch16_224)")
    parser.add_argument("--student", type=str, default="vit_small_patch16_224",
                        help="學生模型 (vit_small_patch16_224/resnet50/...)")
    parser.add_argument("--method", type=str, default="hybrid",
                        choices=["response", "feature", "relation", "hybrid"],
                        help="蒸餾方法")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="outputs/distill/")
    parser.add_argument("--wsi_dir", type=str, default=None,
                        help="（保留舊 API）WSI patch 資料目錄")
    parser.add_argument("--patches_h5", type=str, default=None,
                        help="extract_patches.py 產出的 HDF5，啟用真實 WSI 訓練")
    parser.add_argument("--demo", action="store_true",
                        help="使用合成資料快速測試")
    # 新增的超參數（sweep 用）
    parser.add_argument("--run_name", type=str, default=None,
                        help="實驗名稱，輸出寫到 output/run_name/；無則不建子目錄")
    parser.add_argument("--pretrained_student", action="store_true",
                        help="student backbone 使用 ImageNet pretrained init")
    parser.add_argument("--warmup_ratio", type=float, default=0.0,
                        help="前多少比例的 steps 做 linear warmup (0~1)")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=0.0,
                        help="梯度裁剪上限 (L2 norm)；0 = 不裁")
    parser.add_argument("--projector", type=str, default="linear",
                        choices=["linear", "mlp"],
                        help="Feature distill 的 projector 類型")
    parser.add_argument("--n_slides", type=int, default=20,
                        help="Demo dataset slide 數")
    parser.add_argument("--patches_per_slide", type=int, default=None,
                        help="每個 slide patch 數；預設等於 batch_size（保留舊行為）")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 測試 CUDA 是否可正常使用
    device = "cpu"
    if torch.cuda.is_available():
        try:
            _test = torch.zeros(10, 10).cuda()
            _test2 = torch.mm(_test, _test)  # 觸發 cublas 初始化
            device = "cuda"
        except (RuntimeError, Exception):
            print("  [注意] CUDA 不可用，改用 CPU")
            device = "cpu"
    print(f"\n蒸餾設定")
    print(f"  Teacher: {args.teacher}")
    print(f"  Student: {args.student}")
    print(f"  Method:  {args.method}")
    print(f"  Device:  {device}")
    print(f"  Epochs:  {args.epochs}")

    # 初始化模型
    print("\n[初始化模型...]")
    teacher = TeacherModel(args.teacher).eval().to(device)

    # n_classes：若給 HDF5，依實際 label 決定；否則沿用預設 4
    if args.patches_h5 is not None:
        with h5py.File(args.patches_h5, "r") as _h5:
            n_classes = int(_h5["labels"][:].max()) + 1
        print(f"  自動偵測 n_classes = {n_classes}（來自 HDF5 labels）")
    else:
        n_classes = 4
    student = StudentModel(args.student, embed_dim=256, n_classes=n_classes,
                           pretrained=args.pretrained_student).to(device)

    teacher_dim = teacher.embed_dim
    student_dim = student.embed_dim

    print(f"  Teacher 特徵維度: {teacher_dim}")
    print(f"  Student 特徵維度: {student_dim}")
    print(f"  Teacher 參數量: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"  Student 參數量: {sum(p.numel() for p in student.parameters()):,}")
    compression_ratio = sum(p.numel() for p in teacher.parameters()) / \
                        sum(p.numel() for p in student.parameters())
    print(f"  壓縮比: {compression_ratio:.1f}x")

    # 選擇 loss
    if args.method == "response":
        criterion = ResponseDistillLoss(temperature=4.0, alpha=0.7)
    elif args.method == "feature":
        criterion = FeatureDistillLoss(student_dim, teacher_dim,
                                       projector=args.projector).to(device)
    elif args.method == "relation":
        criterion = RelationDistillLoss()
    else:  # hybrid
        criterion = HybridDistillLoss(student_dim, teacher_dim,
                                      projector=args.projector).to(device)

    # 資料集
    print("\n[準備資料...]")
    patches_per_slide = args.patches_per_slide or args.batch_size
    if args.demo or (args.wsi_dir is None and args.patches_h5 is None):
        print(f"  使用合成資料（demo 模式）n_slides={args.n_slides}, "
              f"patches_per_slide={patches_per_slide}")
        dataset = DemoPatchDataset(n_slides=args.n_slides,
                                   patches_per_slide=patches_per_slide)
    elif args.patches_h5 is not None:
        print(f"  載入真實 WSI patches HDF5: {args.patches_h5}")
        dataset = WSIPatchDataset(args.patches_h5, normalize=True)
        print(f"  樣本數: {len(dataset)}；label 分布: "
              f"{np.bincount(dataset.labels_cache).tolist()}")
    else:
        raise SystemExit(
            f"--wsi_dir ({args.wsi_dir}) 指定但沒有給 --patches_h5；"
            f"請先跑 extract_patches.py 產出 HDF5"
        )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=(device == "cuda"))

    # Optimizer
    all_params = list(student.parameters())
    if hasattr(criterion, "parameters"):
        all_params += list(criterion.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler: warmup + cosine annealing
    total_steps = args.epochs * max(1, len(loader))
    warmup_steps = int(args.warmup_ratio * total_steps)
    warmup_scheduler = None
    if warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps),
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    print(f"\n[開始蒸餾訓練...]\n")
    history = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(student, teacher, loader, optimizer, criterion, device, epoch,
                               grad_clip=args.grad_clip,
                               warmup_scheduler=warmup_scheduler,
                               params_for_clip=all_params)
        scheduler.step()
        acc = evaluate(student, loader, device)
        elapsed = time.time() - t_start
        current_lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "loss": loss, "acc": acc, "time": elapsed, "lr": current_lr})
        print(f"  → Epoch {epoch}/{args.epochs} | Loss: {loss:.4f} | Acc: {acc:.3f} | LR: {current_lr:.2e} | 時間: {elapsed:.1f}s")

    # 儲存模型與記錄（支援 run_name 子目錄）
    out_dir = args.output
    if args.run_name:
        out_dir = os.path.join(args.output, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"student_{args.student.replace('/', '_')}_{args.method}.pt")
    torch.save(student.state_dict(), model_path)
    print(f"\n模型已儲存: {model_path}")

    log_path = os.path.join(out_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "config": vars(args),
            "compression_ratio": compression_ratio,
            "teacher_params": sum(p.numel() for p in teacher.parameters()),
            "student_params": sum(p.numel() for p in student.parameters()),
            "history": history
        }, f, indent=2)
    print(f"訓練記錄已儲存: {log_path}")

    print(f"\n蒸餾完成！壓縮比: {compression_ratio:.1f}x")
    print(f"最終 Loss: {history[-1]['loss']:.4f}")


if __name__ == "__main__":
    main()
