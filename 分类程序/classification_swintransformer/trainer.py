# trainer.py
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    loss_sum = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        loss_sum += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))

    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def _collect_test_predictions(model, loader, device):
    """返回 (y_true, y_pred) 两个 numpy 数组，用于后续混淆矩阵 & 分类报告。"""
    import numpy as np

    model.eval()
    all_y, all_pred = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(x)
        pred = logits.argmax(dim=1)
        all_y.append(y.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())

    return np.concatenate(all_y), np.concatenate(all_pred)


def _init_history_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_acc",
            "val_loss", "val_acc", "lr",
        ])


def _append_history_row(
    path: Path, epoch: int,
    train_loss: float, train_acc: float,
    val_loss: float, val_acc: float,
    lr: float,
) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{train_loss:.6f}", f"{train_acc:.6f}",
            f"{val_loss:.6f}", f"{val_acc:.6f}",
            f"{lr:.6e}",
        ])


def _plot_curves(out_dir: Path, model_name: str) -> None:
    """读 train_history.csv 画训练曲线图（loss + accuracy 两子图）。"""
    history_csv = out_dir / "train_history.csv"
    if not history_csv.exists():
        print(f"[WARN] history csv not found: {history_csv}")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as e:
        print(f"[WARN] matplotlib/pandas missing, skip plotting: {e}")
        return

    df = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    # loss
    axes[0].plot(df["epoch"], df["train_loss"], color="#2563eb",
                 linewidth=1.8, label="train_loss")
    axes[0].plot(df["epoch"], df["val_loss"], color="#93c5fd",
                 linewidth=1.8, linestyle="--", label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} - Training / Validation Loss")
    axes[0].legend(frameon=False)
    axes[0].grid(True, linestyle=":", alpha=0.5)

    # accuracy
    axes[1].plot(df["epoch"], df["train_acc"], color="#059669",
                 linewidth=1.8, label="train_acc")
    axes[1].plot(df["epoch"], df["val_acc"], color="#6ee7b7",
                 linewidth=1.8, linestyle="--", label="val_acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title(f"{model_name} - Training / Validation Accuracy")
    axes[1].legend(frameon=False, loc="lower right")
    axes[1].grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    out_path = out_dir / "train_curves.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] training curves saved -> {out_path}")


def _final_test_eval(
    model, test_loader, out_dir: Path,
    classes: list[str], model_name: str, device,
) -> None:
    """在测试集上做一次最终评估，落盘混淆矩阵 + 分类报告 + 摘要 json。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from sklearn.metrics import (
            classification_report, confusion_matrix, f1_score,
        )
    except ImportError as e:
        print(f"[WARN] sklearn/seaborn missing, skip final test eval: {e}")
        return

    y_true, y_pred = _collect_test_predictions(model, test_loader, device)
    n = int(len(y_true))
    correct = int((y_true == y_pred).sum())
    acc = correct / max(n, 1)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted",
                                 zero_division=0))

    # ------------ confusion matrix ------------
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=True,
        xticklabels=classes, yticklabels=classes, ax=ax,
        linewidths=0.4, linecolor="#cbd5e1",
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(f"Confusion Matrix on Test Set - {model_name}\n"
                 f"Top-1 Acc = {acc * 100:.2f}%  "
                 f"(n = {n}, macro-F1 = {macro_f1:.4f})")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    plt.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] confusion matrix saved -> {cm_path}")

    # ------------ classification_report.txt ------------
    report_text = classification_report(
        y_true, y_pred, target_names=classes, digits=4, zero_division=0,
    )
    report_path = out_dir / "classification_report.txt"
    report_path.write_text(
        f"Model: {model_name}\n"
        f"Test samples: {n}\n"
        f"Top-1 Accuracy: {acc:.4f}\n"
        f"Macro-F1: {macro_f1:.4f}\n"
        f"Weighted-F1: {weighted_f1:.4f}\n\n"
        f"{report_text}\n",
        encoding="utf-8",
    )
    print(f"[OK] classification report saved -> {report_path}")

    # ------------ per_class_metrics.csv ------------
    report_dict = classification_report(
        y_true, y_pred, target_names=classes, digits=6,
        zero_division=0, output_dict=True,
    )
    per_class_csv = out_dir / "per_class_metrics.csv"
    with per_class_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1_score", "support"])
        for cls in classes:
            row = report_dict.get(cls, {})
            w.writerow([
                cls,
                f"{row.get('precision', 0):.6f}",
                f"{row.get('recall', 0):.6f}",
                f"{row.get('f1-score', 0):.6f}",
                int(row.get("support", 0)),
            ])
    print(f"[OK] per-class metrics saved -> {per_class_csv}")

    # ------------ test_summary.json ------------
    summary = {
        "model_name": model_name,
        "test_samples": n,
        "correct": correct,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classes": classes,
        "confusion_matrix": cm.tolist(),
    }
    summary_path = out_dir / "test_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] test summary saved -> {summary_path}")
    print(f"[TEST-FINAL] acc={acc:.4f}  macro-F1={macro_f1:.4f}  "
          f"weighted-F1={weighted_f1:.4f}  n={n}")


def _get(cfg, name: str, default):
    """兼容 config.py 里可能没有某些字段的情况"""
    return getattr(cfg, name, default)


def _resolve_out_dir(base: Path, auto_increment: bool) -> Path:
    """解析训练输出目录。

    - auto_increment=False: 直接返回 base（可能覆盖已有产物）。
    - auto_increment=True : 若 base 已存在且非空，则依次尝试
      base_2, base_3, ... 直到找到空目录，参照 Ultralytics 的做法。
    """
    if not auto_increment:
        return base
    if not base.exists() or not any(base.iterdir()):
        return base
    parent, name = base.parent, base.name
    idx = 2
    while True:
        cand = parent / f"{name}_{idx}"
        if not cand.exists() or not any(cand.iterdir()):
            return cand
        idx += 1


def train(cfg, train_ds, val_ds, test_ds=None):
    # -------------------------
    # 设备选择
    # -------------------------
    use_cpu = bool(_get(cfg, "USE_CPU", False))
    device = torch.device("cuda" if torch.cuda.is_available() and not use_cpu else "cpu")

    print(f"[INFO] device = {device}")

    if device.type == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            print(f"[INFO] GPU = {gpu_name}, compute_capability={cc}")
        except Exception:
            pass

    # -------------------------
    # 读取配置
    # -------------------------
    base_out = Path(_get(cfg, "OUT_DIR", "runs/swin"))
    auto_increment = bool(_get(cfg, "AUTO_INCREMENT_OUT", True))
    out_dir = _resolve_out_dir(base_out, auto_increment)
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir != base_out:
        print(f"[INFO] base dir {base_out} 已占用，改用新目录 {out_dir}")

    batch_size = int(_get(cfg, "BATCH_SIZE", 32))
    workers = int(_get(cfg, "WORKERS", 0))  # Windows 默认 0 更稳
    lr = float(_get(cfg, "LR", 3e-4))
    wd = float(_get(cfg, "WEIGHT_DECAY", 0.05))
    epochs = int(_get(cfg, "EPOCHS", 30))
    model_name = str(_get(cfg, "MODEL_NAME", "swin_tiny_patch4_window7_224"))
    pretrained = bool(_get(cfg, "PRETRAINED", True))
    test_every = int(_get(cfg, "TEST_EVERY", 5))

    # 性能/稳定性参数（可在 config.py 里覆写）
    # 梯度累积：等效 batch = batch_size * grad_accum
    grad_accum = int(_get(cfg, "GRAD_ACCUM", 1))
    max_grad_norm = float(_get(cfg, "MAX_GRAD_NORM", 1.0))  # 梯度裁剪
    use_amp = bool(_get(cfg, "USE_AMP", True)) and (device.type == "cuda")

    # cudnn 优化（对固定分辨率通常更快）
    torch.backends.cudnn.benchmark = True

    # -------------------------
    # 保存类别顺序（推理/部署很重要）
    # -------------------------
    (out_dir / "classes.txt").write_text("\n".join(train_ds.classes), encoding="utf-8")

    # -------------------------
    # 训练曲线日志（CSV）
    # -------------------------
    history_csv = out_dir / "train_history.csv"
    _init_history_csv(history_csv)
    print(f"[INFO] training history -> {history_csv}")

    # -------------------------
    # DataLoader
    # -------------------------
    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(workers > 0),
        prefetch_factor=1 if workers > 0 else None,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(workers > 0),
        prefetch_factor=1 if workers > 0 else None,
    )

    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=(workers > 0),
            prefetch_factor=1 if workers > 0 else None,
        )

    # -------------------------
    # 模型/优化器/调度器
    # -------------------------
    num_classes = len(train_ds.classes)
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Cosine LR（简单好用）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # AMP
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # -------------------------
    # 训练循环
    # -------------------------
    best_acc = 0.0

    print(
        f"[INFO] model={model_name} | classes={num_classes} | "
        f"batch={batch_size} | grad_accum={grad_accum} | amp={use_amp} | workers={workers}"
    )

    for epoch in range(1, epochs + 1):
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", ncols=110)

        for step, (x, y) in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
                # 梯度累积：把 loss 缩小
                loss = loss / max(grad_accum, 1)

            # backward
            scaler.scale(loss).backward()

            # 统计（注意这里用“未缩放前”的等效 loss 统计：乘回去）
            batch_size_now = x.size(0)
            running_loss += float(loss.item()) * batch_size_now * max(grad_accum, 1)
            pred = logits.argmax(dim=1)
            running_correct += int((pred == y).sum().item())
            running_total += int(batch_size_now)

            # 每 grad_accum step 更新一次参数
            if step % grad_accum == 0:
                # 反缩放后裁剪梯度
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss = running_loss / max(running_total, 1)
            train_acc = running_correct / max(running_total, 1)
            pbar.set_postfix(loss=f"{train_loss:.4f}", acc=f"{train_acc:.4f}")

        # epoch 结束时，如果还有没 step 的累计梯度，补一次更新
        # （当 len(loader) 不是 grad_accum 的整数倍时）
        if (len(train_loader) % max(grad_accum, 1)) != 0:
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # 验证
        val_loss, val_acc = evaluate(model, val_loader, device)
        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        print(
            f"[EPOCH {epoch}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f} | "
            f"lr={current_lr:.2e}"
        )

        # 落盘训练曲线 CSV
        _append_history_row(
            history_csv, epoch,
            train_loss, train_acc,
            val_loss, val_acc, current_lr,
        )

        # 保存 last / best
        torch.save(model.state_dict(), out_dir / "last.pt")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best.pt")

        # 可选 test
        if test_loader is not None and (epoch == epochs or epoch % test_every == 0):
            t_loss, t_acc = evaluate(model, test_loader, device)
            print(f"[TEST] loss={t_loss:.4f}, acc={t_acc:.4f}")

    print(f"[DONE] Best Val Acc = {best_acc:.4f}")
    print(f"[INFO] weights saved in: {out_dir.resolve()}")

    # -------------------------
    # 训练结束：画曲线 + 加载 best 做最终测试评估
    # -------------------------
    _plot_curves(out_dir, model_name=model_name)

    if test_loader is not None:
        best_ckpt = out_dir / "best.pt"
        if best_ckpt.exists():
            try:
                state = torch.load(str(best_ckpt), map_location=device)
                model.load_state_dict(state, strict=True)
                print(f"[INFO] loaded best.pt for final test evaluation")
            except Exception as e:
                print(f"[WARN] failed to reload best.pt, using last weights: {e}")

        _final_test_eval(
            model, test_loader, out_dir,
            classes=train_ds.classes, model_name=model_name, device=device,
        )
