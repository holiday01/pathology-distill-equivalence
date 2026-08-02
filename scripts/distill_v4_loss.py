"""v4 distillation loss: L_CLS (1-cos, L2-norm) + L_PAT (cos) + L_MGD + L_DINO.
Target identity:
  L_CLS/L_PAT/L_MGD  -> frozen cached teacher features (external knowledge)
  L_DINO             -> EMA-of-STUDENT (self-distillation, online augmented views)
Disjoint pathways per v4 §2.1 / R1-NC1.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClsCosineLoss(nn.Module):
    """L_CLS = 1 - cosine(L2(proj(s_CLS)), L2(t_CLS))."""

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        hidden = max(student_dim, teacher_dim)
        self.projector = nn.Sequential(
            nn.Linear(student_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, teacher_dim),
        )

    def forward(self, s_cls, t_cls):
        s_proj = self.projector(s_cls)
        s = F.normalize(s_proj, dim=-1)
        t = F.normalize(t_cls.detach(), dim=-1)
        return 1.0 - (s * t).sum(dim=-1).mean()


class PatchCosineLoss(nn.Module):
    """L_PAT = mean over tokens of (1 - cosine(s_patch_proj, t_patch))."""

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.projector = nn.Linear(student_dim, teacher_dim)

    def forward(self, s_patches, t_patches):
        s_proj = self.projector(s_patches)
        s = F.normalize(s_proj, dim=-1)
        t = F.normalize(t_patches.detach(), dim=-1)
        return 1.0 - (s * t).sum(dim=-1).mean()


class MgdLoss(nn.Module):
    """Masked Generative Distillation (Yang ECCV 2022) on intermediate layer.
    Randomly masks a fraction of student patch tokens, a lightweight conv generator
    reconstructs the teacher's intermediate-layer patch tokens.
    """

    def __init__(self, student_dim: int, teacher_dim: int, mask_ratio: float = 0.5,
                 grid_size: int = 14):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.grid = grid_size
        self.align = nn.Linear(student_dim, teacher_dim)
        self.generator = nn.Sequential(
            nn.Conv2d(teacher_dim, teacher_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(teacher_dim, teacher_dim, kernel_size=3, padding=1),
        )

    def forward(self, s_patches, t_patches_mid):
        # s_patches: (B, N, Ds); t_patches_mid: (B, N, Dt) ; N = grid**2 = 196
        B, N, _ = s_patches.shape
        s_aligned = self.align(s_patches)
        mask = torch.rand(B, N, device=s_patches.device) < self.mask_ratio
        s_masked = s_aligned.clone()
        s_masked[mask] = 0
        g = self.grid
        x = s_masked.transpose(1, 2).reshape(B, -1, g, g)
        gen = self.generator(x).reshape(B, -1, N).transpose(1, 2)
        t = t_patches_mid.detach()
        return F.mse_loss(gen, t)


class DinoSelfDistillLoss(nn.Module):
    """DINO-style self-distillation on CLS.
    EMA-of-student serves as target; operates on multi-crop online augmentations.
    Loss = cross-entropy between sharpened teacher distribution (over K prototypes)
    and student distribution.  Centering buffer updated on teacher outputs.
    """

    def __init__(self, student_dim: int, out_dim: int = 8192,
                 teacher_temp: float = 0.04, student_temp: float = 0.1,
                 center_momentum: float = 0.9, ema_momentum: float = 0.996):
        super().__init__()
        self.student_head = self._mk_head(student_dim, out_dim)
        self.teacher_head = self._mk_head(student_dim, out_dim)
        for p in self.teacher_head.parameters():
            p.requires_grad = False
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ema_momentum = ema_momentum

    @staticmethod
    def _mk_head(in_dim, out_dim, hidden=2048, bottleneck=256):
        return nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, bottleneck),
            nn.utils.weight_norm(nn.Linear(bottleneck, out_dim, bias=False)),
        )

    @torch.no_grad()
    def ema_update(self, student_backbone_params, teacher_backbone_params):
        m = self.ema_momentum
        for tp, sp in zip(self.teacher_head.parameters(), self.student_head.parameters()):
            tp.data.mul_(m).add_(sp.data, alpha=1 - m)
        # External caller is responsible for backbone EMA.

    def forward(self, student_views_cls, teacher_views_cls):
        """
        student_views_cls: (V_s, B, D) — V_s global+local augmented views from student
        teacher_views_cls: (V_t, B, D) — V_t global views from EMA-student
        """
        Vs, B, _ = student_views_cls.shape
        Vt = teacher_views_cls.shape[0]
        s_out = self.student_head(student_views_cls.reshape(Vs * B, -1))
        with torch.no_grad():
            t_out = self.teacher_head(teacher_views_cls.reshape(Vt * B, -1)).detach()
            t_soft = F.softmax((t_out - self.center) / self.teacher_temp, dim=-1)
            t_soft = t_soft.view(Vt, B, -1)
            # update center
            batch_center = t_out.mean(dim=0, keepdim=True)
            self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)
        s_log = F.log_softmax(s_out / self.student_temp, dim=-1).view(Vs, B, -1)
        # cross-view loss: each teacher view × each different student view
        loss, n = 0.0, 0
        for i in range(Vt):
            for j in range(Vs):
                if i == j:
                    continue
                loss = loss - (t_soft[i] * s_log[j]).sum(dim=-1).mean()
                n += 1
        return loss / max(n, 1)


class V4DistillLoss(nn.Module):
    """Top-level v4 loss: L_CLS + L_PAT + L_MGD + L_DINO.

    student_cls_dim = proj_head output dim (CLS path)
    student_patch_dim = backbone native dim (patch path, before proj_head)
    """

    def __init__(self, student_cls_dim: int, student_patch_dim: int, teacher_dim: int,
                 lam_cls: float = 1.0, lam_pat: float = 0.5,
                 lam_mgd: float = 0.5, lam_dino: float = 0.5,
                 use_dino: bool = True, use_mgd: bool = True):
        super().__init__()
        self.cls_loss = ClsCosineLoss(student_cls_dim, teacher_dim)
        self.pat_loss = PatchCosineLoss(student_patch_dim, teacher_dim)
        self.mgd_loss = MgdLoss(student_patch_dim, teacher_dim) if use_mgd else None
        self.dino_loss = DinoSelfDistillLoss(student_cls_dim) if use_dino else None
        self.lam_cls = lam_cls
        self.lam_pat = lam_pat
        self.lam_mgd = lam_mgd
        self.lam_dino = lam_dino

    def forward(self, s, t, s_views_cls=None, t_views_cls=None):
        """s, t are dicts with {feat, patch_feats, patch_feats_mid(optional)}."""
        total = self.lam_cls * self.cls_loss(s["feat"], t["feat"])
        total = total + self.lam_pat * self.pat_loss(s["patch_feats"], t["patch_feats"])
        if self.mgd_loss is not None and "patch_feats_mid" in t:
            total = total + self.lam_mgd * self.mgd_loss(s["patch_feats"], t["patch_feats_mid"])
        if self.dino_loss is not None and s_views_cls is not None and t_views_cls is not None:
            total = total + self.lam_dino * self.dino_loss(s_views_cls, t_views_cls)
        return total
