#!/usr/bin/env python3
"""
WSI 蒸餾模型驗證框架
====================================
驗證 student 模型是否與 teacher FM 具有等效功效。

三個驗證層次：
  1. 特徵層相似度：CKA、Cosine similarity
  2. 下游任務等效性：Linear probing（分類 AUC/ACC）
  3. 預測一致性：Cohen's kappa、Retrieval recall@k
  4. 效率比較：推論延遲、吞吐量

用法：
  python evaluate_distillation.py --teacher phikon --student_ckpt outputs/distill/student_xxx.pt
  python evaluate_distillation.py --demo  # 合成資料快速驗證
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from distill_wsi_model import StudentModel, TeacherModel, DemoPatchDataset, WSIPatchDataset


# ──────────────────────────────────────────────────────────
# 1. 特徵相似度指標
# ──────────────────────────────────────────────────────────

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA（Kornblith et al., 2019）：量測兩組特徵的表徵相似度。
    X: (N, d1), Y: (N, d2) — 回傳 [0, 1] 之間的相似度分數
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    xty = (X.T @ Y).norm("fro") ** 2
    xtx = (X.T @ X).norm("fro")
    yty = (Y.T @ Y).norm("fro")
    return (xty / (xtx * yty + 1e-8)).item()


def cosine_alignment(X: torch.Tensor, Y: torch.Tensor) -> float:
    """樣本層級的平均 cosine similarity（需投影到同維度）。"""
    if X.shape[1] != Y.shape[1]:
        d = min(X.shape[1], Y.shape[1])
        X = X[:, :d]
        Y = Y[:, :d]
    X = F.normalize(X, dim=-1)
    Y = F.normalize(Y, dim=-1)
    return (X * Y).sum(dim=-1).mean().item()


# ──────────────────────────────────────────────────────────
# 2. Linear probing
# ──────────────────────────────────────────────────────────

def extract_features(model, loader, device, is_teacher=False):
    """萃取 encoder 特徵（CLS token 或 pooled feature）。"""
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for patches, y in loader:
            patches = patches.to(device)
            out = model(patches)
            f = out["feat"] if isinstance(out, dict) else out
            feats.append(f.cpu())
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def linear_probe(train_feat, train_y, test_feat, test_y, n_classes, epochs=30, lr=1e-3):
    """Frozen encoder + linear head：標準 FM 評估 protocol。"""
    device = train_feat.device
    clf = nn.Linear(train_feat.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-4)

    for _ in range(epochs):
        logits = clf(train_feat)
        loss = F.cross_entropy(logits, train_y)
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        test_logits = clf(test_feat)
        probs = F.softmax(test_logits, dim=-1)
        preds = test_logits.argmax(dim=-1)
        acc = (preds == test_y).float().mean().item()
        # Multi-class AUC（macro one-vs-rest，僅在 sklearn 可用時）
        try:
            from sklearn.metrics import roc_auc_score
            probs_np = probs.cpu().numpy()
            y_true = test_y.cpu().numpy()
            # binary: pass positive-class score (1-D); multiclass: full prob matrix
            if len(np.unique(y_true)) <= 1:
                auc = float("nan")
            elif n_classes == 2:
                auc = roc_auc_score(y_true, probs_np[:, 1])
            else:
                auc = roc_auc_score(y_true, probs_np, multi_class="ovr", average="macro")
        except Exception:
            auc = float("nan")
    return {"acc": acc, "auc": auc, "preds": preds.cpu(), "probs": probs.cpu()}


# ──────────────────────────────────────────────────────────
# 3. 預測一致性
# ──────────────────────────────────────────────────────────

def cohen_kappa(y1, y2) -> float:
    """Cohen's κ：兩個分類器的預測一致性（扣除隨機同意）。"""
    y1 = y1.detach().cpu().numpy() if isinstance(y1, torch.Tensor) else np.asarray(y1)
    y2 = y2.detach().cpu().numpy() if isinstance(y2, torch.Tensor) else np.asarray(y2)
    n = len(y1)
    classes = np.unique(np.concatenate([y1, y2]))
    po = (y1 == y2).mean()
    pe = sum(((y1 == c).sum() / n) * ((y2 == c).sum() / n) for c in classes)
    return float((po - pe) / (1 - pe + 1e-8))


def retrieval_recall(query_feat, gallery_feat, labels, k=5):
    """Retrieval recall@k：用 embedding 檢索同類樣本的比率。"""
    q = F.normalize(query_feat, dim=-1)
    g = F.normalize(gallery_feat, dim=-1)
    sim = q @ g.T
    # 排除自身
    sim.fill_diagonal_(-1)
    topk = sim.topk(k, dim=-1).indices
    retrieved_labels = labels[topk]
    hits = (retrieved_labels == labels.unsqueeze(1)).any(dim=-1).float().mean()
    return hits.item()


# ──────────────────────────────────────────────────────────
# 4. 推論效率
# ──────────────────────────────────────────────────────────

def benchmark_latency(model, device, input_shape=(1, 3, 224, 224), n_iter=50, warmup=10):
    """量測單張 patch 的平均推論時間（ms）。"""
    model.eval()
    x = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.time() - t0) / n_iter * 1000
    return elapsed


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WSI 蒸餾模型驗證")
    parser.add_argument("--teacher", type=str, default="phikon")
    parser.add_argument("--student", type=str, default="vit_small_patch16_224")
    parser.add_argument("--student_ckpt", type=str, default=None,
                        help="訓練好的 student 權重；無則用隨機初始化（demo）")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--output", type=str, default="outputs/evaluation/")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--patches_h5", type=str, default=None,
                        help="真實 WSI patches HDF5；給了就取代 demo")
    parser.add_argument("--test_ratio", type=float, default=0.3,
                        help="train/test 切割比例（僅 --patches_h5 模式生效）")
    parser.add_argument("--seed", type=int, default=42,
                        help="固定 seed，讓不同 student 在同一份 patches 上比較")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if device == "cuda":
            torch.mm(torch.zeros(8, 8).cuda(), torch.zeros(8, 8).cuda())
    except Exception:
        device = "cpu"

    print(f"\n驗證設定")
    print(f"  Teacher: {args.teacher}")
    print(f"  Student: {args.student}")
    print(f"  Device:  {device}")

    # 初始化模型
    print("\n[載入模型...]")
    teacher = TeacherModel(args.teacher).eval().to(device)
    student = StudentModel(args.student, embed_dim=256, n_classes=args.n_classes).to(device)
    if args.student_ckpt and os.path.exists(args.student_ckpt):
        sd = torch.load(args.student_ckpt, map_location=device)
        model_sd = student.state_dict()
        filtered = {
            k: v for k, v in sd.items()
            if k in model_sd and v.shape == model_sd[k].shape
        }
        skipped = [k for k in sd.keys() if k not in filtered]
        student.load_state_dict(filtered, strict=False)
        if skipped:
            print(f"  [note] 跳過形狀不合的 keys: {skipped}")
        print(f"  已載入 student 權重: {args.student_ckpt}")
    else:
        print(f"  [警告] 無 student 權重，使用隨機初始化（僅 demo 用）")
    student.eval()

    # 資料集（train / test 分割）
    print("\n[準備資料...]")
    if args.patches_h5 is not None:
        full = WSIPatchDataset(args.patches_h5, normalize=True)
        n_total = len(full)
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(n_total)
        n_test = max(1, int(n_total * args.test_ratio))
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        train_set = WSIPatchDataset(args.patches_h5, normalize=True, indices=train_idx)
        test_set = WSIPatchDataset(args.patches_h5, normalize=True, indices=test_idx)
        n_classes_effective = int(full.labels_cache.max()) + 1
        if n_classes_effective != args.n_classes:
            print(f"  [自動] n_classes {args.n_classes} → {n_classes_effective} "
                  f"（依 HDF5 label 範圍）")
            args.n_classes = n_classes_effective
        print(f"  train={len(train_set)}, test={len(test_set)}, "
              f"n_classes={args.n_classes}")
    else:
        train_set = DemoPatchDataset(n_slides=15, patches_per_slide=args.batch_size,
                                      n_classes=args.n_classes)
        test_set = DemoPatchDataset(n_slides=8, patches_per_slide=args.batch_size,
                                     n_classes=args.n_classes)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # ── 萃取特徵 ──
    print("\n[萃取特徵...]")
    t_train_f, train_y = extract_features(teacher, train_loader, device, is_teacher=True)
    t_test_f, test_y = extract_features(teacher, test_loader, device, is_teacher=True)
    s_train_f, _ = extract_features(student, train_loader, device)
    s_test_f, _ = extract_features(student, test_loader, device)

    print(f"  Teacher 特徵: {tuple(t_test_f.shape)}")
    print(f"  Student 特徵: {tuple(s_test_f.shape)}")

    results = {}

    # ── 1. 特徵相似度 ──
    print("\n[1/4] 特徵相似度")
    cka = linear_cka(s_test_f, t_test_f)
    cos = cosine_alignment(s_test_f, t_test_f)
    print(f"  Linear CKA: {cka:.4f}  (越接近 1 越好)")
    print(f"  Cosine:     {cos:.4f}")
    results["feature_similarity"] = {"linear_cka": cka, "cosine": cos}

    # ── 2. Linear probing（下游任務等效性）──
    print("\n[2/4] Linear probing（frozen encoder + linear head）")
    t_probe = linear_probe(t_train_f.to(device), train_y.to(device),
                           t_test_f.to(device), test_y.to(device), args.n_classes)
    s_probe = linear_probe(s_train_f.to(device), train_y.to(device),
                           s_test_f.to(device), test_y.to(device), args.n_classes)
    print(f"  Teacher: ACC={t_probe['acc']:.3f}  AUC={t_probe['auc']:.3f}")
    print(f"  Student: ACC={s_probe['acc']:.3f}  AUC={s_probe['auc']:.3f}")
    print(f"  Gap    : ΔACC={t_probe['acc']-s_probe['acc']:+.3f}  (判準：|ΔACC| < 0.03 視為等效)")
    results["linear_probing"] = {
        "teacher_acc": t_probe["acc"], "teacher_auc": t_probe["auc"],
        "student_acc": s_probe["acc"], "student_auc": s_probe["auc"],
        "acc_gap": t_probe["acc"] - s_probe["acc"],
    }

    # ── 3. 預測一致性 ──
    print("\n[3/4] 預測一致性")
    kappa = cohen_kappa(t_probe["preds"], s_probe["preds"])
    t_recall = retrieval_recall(t_test_f, t_test_f, test_y, k=5)
    s_recall = retrieval_recall(s_test_f, s_test_f, test_y, k=5)
    print(f"  Cohen's κ (teacher vs student 預測): {kappa:.3f}")
    print(f"  Retrieval Recall@5 — Teacher: {t_recall:.3f}  Student: {s_recall:.3f}")
    results["agreement"] = {
        "cohen_kappa": kappa,
        "teacher_recall_at_5": t_recall,
        "student_recall_at_5": s_recall,
    }

    # ── 4. 效率比較 ──
    print("\n[4/4] 推論效率")
    t_ms = benchmark_latency(teacher, device)
    s_ms = benchmark_latency(student, device)
    t_params = sum(p.numel() for p in teacher.parameters())
    s_params = sum(p.numel() for p in student.parameters())
    print(f"  Teacher: {t_ms:.2f} ms/patch  |  {t_params/1e6:.1f}M params")
    print(f"  Student: {s_ms:.2f} ms/patch  |  {s_params/1e6:.1f}M params")
    print(f"  加速比:  {t_ms/s_ms:.2f}×    |  壓縮比: {t_params/s_params:.2f}×")
    results["efficiency"] = {
        "teacher_latency_ms": t_ms, "student_latency_ms": s_ms,
        "speedup": t_ms / s_ms,
        "teacher_params": t_params, "student_params": s_params,
        "compression_ratio": t_params / s_params,
    }

    # ── 總結判準 ──
    print("\n" + "=" * 50)
    print("等效性判準")
    print("=" * 50)
    pass_cka = bool(cka > 0.7)
    pass_gap = bool(abs(results["linear_probing"]["acc_gap"]) < 0.03)
    pass_kappa = bool(kappa > 0.6)
    print(f"  [{'✓' if pass_cka else '✗'}] CKA > 0.7          ({cka:.3f})")
    print(f"  [{'✓' if pass_gap else '✗'}] |ΔACC| < 0.03       ({results['linear_probing']['acc_gap']:+.3f})")
    print(f"  [{'✓' if pass_kappa else '✗'}] Cohen's κ > 0.6    ({kappa:.3f})")
    verdict = "等效" if (pass_cka and pass_gap and pass_kappa) else "未達標"
    print(f"\n  結論：Student {verdict} Teacher FM")
    results["verdict"] = {
        "pass_cka": pass_cka, "pass_acc_gap": pass_gap, "pass_kappa": pass_kappa,
        "equivalent": pass_cka and pass_gap and pass_kappa,
    }

    # 儲存報告
    os.makedirs(args.output, exist_ok=True)
    report_path = os.path.join(args.output, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"\n報告已儲存: {report_path}")


if __name__ == "__main__":
    main()
