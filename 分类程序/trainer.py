from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

from models import DualHeadClassifier, freeze_backbone
from utils import (
    default_out_dir_for_backbone,
    normalize_backbone_name,
    resolve_experiment_dir,
    save_label_maps,
)


def _get(cfg, name: str, default):
    return getattr(cfg, name, default)


# ============================================================
# 评估
# ============================================================

@torch.no_grad()
def evaluate(
    model: DualHeadClassifier,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> Dict[str, float]:
    """
    在 loader 上评估双头模型。

    返回 dict:
        brand_loss, brand_acc, model_loss, model_acc, total_loss
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    brand_loss_sum = 0.0
    brand_correct = 0
    model_loss_sum = 0.0
    model_correct = 0
    total = 0

    for x, brand_y, model_y in loader:
        x = x.to(device, non_blocking=True)
        brand_y = brand_y.to(device, non_blocking=True)
        model_y = model_y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            brand_logits, model_logits = model(x)
            b_loss = criterion(brand_logits, brand_y)
            m_loss = criterion(model_logits, model_y)

        bs = x.size(0)
        brand_loss_sum += b_loss.item() * bs
        model_loss_sum += m_loss.item() * bs
        brand_correct += (brand_logits.argmax(1) == brand_y).sum().item()
        model_correct += (model_logits.argmax(1) == model_y).sum().item()
        total += bs

    n = max(total, 1)
    return {
        "brand_loss": brand_loss_sum / n,
        "brand_acc": brand_correct / n,
        "model_loss": model_loss_sum / n,
        "model_acc": model_correct / n,
        "total_loss": (brand_loss_sum + model_loss_sum) / n,
    }


@torch.no_grad()
def _collect_test_predictions(model, loader, device, use_amp: bool = True):
    all_brand_y, all_brand_pred = [], []
    all_model_y, all_model_pred = [], []
    model.eval()
    for x, brand_y, model_y in loader:
        x = x.to(device, non_blocking=True)
        brand_y = brand_y.to(device, non_blocking=True)
        model_y = model_y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            brand_logits, model_logits = model(x)
        all_brand_y.append(brand_y.detach().cpu())
        all_brand_pred.append(brand_logits.argmax(1).detach().cpu())
        all_model_y.append(model_y.detach().cpu())
        all_model_pred.append(model_logits.argmax(1).detach().cpu())
    return (
        torch.cat(all_brand_y).numpy(),
        torch.cat(all_brand_pred).numpy(),
        torch.cat(all_model_y).numpy(),
        torch.cat(all_model_pred).numpy(),
    )


def _init_history_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_brand_acc",
            "train_model_acc",
            "val_brand_acc",
            "val_model_acc",
            "brand_loss",
            "model_loss",
            "lr",
        ])


def _append_history_row(
    path: Path,
    epoch: int,
    train_brand_acc: float,
    train_model_acc: float,
    val_brand_acc: float,
    val_model_acc: float,
    brand_loss: float,
    model_loss: float,
    lr: float,
) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{train_brand_acc:.6f}",
            f"{train_model_acc:.6f}",
            f"{val_brand_acc:.6f}",
            f"{val_model_acc:.6f}",
            f"{brand_loss:.6f}",
            f"{model_loss:.6f}",
            f"{lr:.6e}",
        ])


def _plot_curves(out_dir: Path, model_name: str) -> None:
    history_csv = out_dir / "train_history.csv"
    if not history_csv.exists():
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        return

    df = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    axes[0].plot(df["epoch"], df["train_model_acc"], color="#2563eb", linewidth=1.8, label="train_model_acc")
    axes[0].plot(df["epoch"], df["val_model_acc"], color="#93c5fd", linewidth=1.8, linestyle="--", label="val_model_acc")
    axes[0].plot(df["epoch"], df["train_brand_acc"], color="#059669", linewidth=1.4, label="train_brand_acc")
    axes[0].plot(df["epoch"], df["val_brand_acc"], color="#6ee7b7", linewidth=1.4, linestyle="--", label="val_brand_acc")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title(f"{model_name} - Brand / Model Accuracy")
    axes[0].legend(frameon=False)
    axes[0].grid(True, linestyle=":", alpha=0.5)

    axes[1].plot(df["epoch"], df["brand_loss"], color="#f59e0b", linewidth=1.8, label="brand_loss")
    axes[1].plot(df["epoch"], df["model_loss"], color="#ef4444", linewidth=1.8, label="model_loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title(f"{model_name} - Brand / Model Loss")
    axes[1].legend(frameon=False)
    axes[1].grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_dir / "train_curves.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    axes[0].plot(df["epoch"], df["train_brand_acc"], color="#0ea5e9", linewidth=1.8, label="train_brand_acc")
    axes[0].plot(df["epoch"], df["val_brand_acc"], color="#93c5fd", linewidth=1.8, linestyle="--", label="val_brand_acc")
    axes[0].plot(df["epoch"], df["train_model_acc"], color="#10b981", linewidth=1.8, label="train_model_acc")
    axes[0].plot(df["epoch"], df["val_model_acc"], color="#6ee7b7", linewidth=1.8, linestyle="--", label="val_model_acc")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title(f"{model_name} - Brand / Model Accuracy Split")
    axes[0].legend(frameon=False)
    axes[0].grid(True, linestyle=":", alpha=0.5)

    axes[1].plot(df["epoch"], df["brand_loss"], color="#f59e0b", linewidth=1.8, label="brand_loss")
    axes[1].plot(df["epoch"], df["model_loss"], color="#ef4444", linewidth=1.8, label="model_loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title(f"{model_name} - Brand / Model Loss Split")
    axes[1].legend(frameon=False)
    axes[1].grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "brand_model_split.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(df["epoch"], df["lr"], color="#8b5cf6", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title(f"{model_name} - Learning Rate")
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "lr_curve.png", bbox_inches="tight")
    plt.close(fig)


def _save_dataset_distribution(out_dir: Path, train_ds, val_ds, test_ds=None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    def count_by_model(dataset):
        counts = {name: 0 for name in dataset.model_classes}
        for _path, _brand_idx, model_idx in dataset.samples:
            counts[dataset.model_classes[model_idx]] += 1
        return counts

    train_counts = count_by_model(train_ds)
    val_counts = count_by_model(val_ds)
    test_counts = count_by_model(test_ds) if test_ds is not None else {name: 0 for name in train_ds.model_classes}
    labels = list(train_ds.model_classes)
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.6), 4.5), dpi=150)
    ax.bar(x - width, [train_counts[l] for l in labels], width, label="train", color="#2563eb")
    ax.bar(x, [val_counts[l] for l in labels], width, label="val", color="#f59e0b")
    ax.bar(x + width, [test_counts[l] for l in labels], width, label="test", color="#10b981")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Samples")
    ax.set_title("Dataset Distribution by Model")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "dataset_distribution.png", bbox_inches="tight")
    plt.close(fig)


def _save_confusion_and_report(
    y_true,
    y_pred,
    classes,
    prefix: str,
    out_dir: Path,
    model_name: str,
) -> dict:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import classification_report, confusion_matrix, f1_score
    except ImportError:
        return {}

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        linewidths=0.4,
        linecolor="#cbd5e1",
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(f"{prefix} Confusion Matrix - {model_name}\nAcc={acc * 100:.2f}%")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix.lower()}_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    report_text = classification_report(
        y_true, y_pred, target_names=classes, digits=4, zero_division=0,
    )
    (out_dir / f"{prefix.lower()}_classification_report.txt").write_text(
        f"Model: {model_name}\nAccuracy: {acc:.4f}\nMacro-F1: {macro_f1:.4f}\n\n{report_text}\n",
        encoding="utf-8",
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
    }


def _save_dualhead_test_summary(
    out_dir: Path,
    model_name: str,
    brand_classes: list[str],
    model_classes: list[str],
    brand_metrics: dict,
    model_metrics: dict,
) -> None:
    summary = {
        "model_name": model_name,
        "brand_classes": brand_classes,
        "model_classes": model_classes,
        "brand_metrics": brand_metrics,
        "model_metrics": model_metrics,
    }
    (out_dir / "test_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _save_gradcam_artifacts(
    model: DualHeadClassifier,
    test_ds,
    out_dir: Path,
    model_name: str,
    img_size: int,
    device: torch.device,
) -> None:
    try:
        import cv2
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image
    except ImportError:
        return

    class GradCAM:
        def __init__(self, network, target_module, reshape_fn=None):
            self.model = network.eval()
            self.target_module = target_module
            self.activations = None
            self.gradients = None
            self.reshape_fn = reshape_fn
            target_module.register_forward_hook(self._forward_hook)
            target_module.register_full_backward_hook(self._backward_hook)

        def _forward_hook(self, _module, _inp, out):
            self.activations = out

        def _backward_hook(self, _module, _grad_in, grad_out):
            self.gradients = grad_out[0]

        def __call__(self, x: torch.Tensor, task: str, target_idx: int) -> np.ndarray:
            brand_logits, model_logits = self.model(x)
            logits = brand_logits if task == "brand" else model_logits
            self.model.zero_grad(set_to_none=True)
            logits[0, target_idx].backward(retain_graph=False)
            activations = self.activations
            gradients = self.gradients
            if self.reshape_fn is not None:
                activations = self.reshape_fn(activations)
                gradients = self.reshape_fn(gradients)
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=True)
            cam = torch.relu(cam)
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()
            return cam.squeeze().detach().cpu().numpy()

    def pick_target_layer(network, name: str):
        lower = name.lower()
        if "swin" in lower:
            return network.backbone.layers[-1].blocks[-1].norm2
        if "mobilenet" in lower:
            return network.backbone.blocks[-1]
        return network.backbone

    def swin_reshape(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 4:
            return tensor.permute(0, 3, 1, 2).contiguous()
        if tensor.dim() == 3:
            b, n, c = tensor.shape
            h = int(round(n ** 0.5))
            w = n // h
            return tensor.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        raise RuntimeError(f"Unexpected Swin tensor shape: {tuple(tensor.shape)}")

    def tensor_to_rgb(x: torch.Tensor) -> np.ndarray:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = x.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        img = img * std + mean
        img = np.clip(img, 0, 1) * 255
        return img.astype(np.uint8)

    def overlay_cam(img_rgb: np.ndarray, cam: np.ndarray) -> np.ndarray:
        h, w = img_rgb.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return (heatmap * 0.45 + img_rgb * 0.55).astype(np.uint8)

    target_layer = pick_target_layer(model, model_name)
    reshape_fn = swin_reshape if "swin" in model_name.lower() else None
    cam_extractor = GradCAM(model, target_layer, reshape_fn=reshape_fn)
    transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    samples_by_class: dict[int, list[Path]] = {idx: [] for idx in range(len(test_ds.model_classes))}
    for path, _brand_idx, model_idx in test_ds.samples:
        bucket = samples_by_class[model_idx]
        if len(bucket) < 3:
            bucket.append(Path(path))
        if all(len(items) >= 3 for items in samples_by_class.values()):
            break

    samples = [
        (path, model_idx)
        for model_idx, paths in samples_by_class.items()
        for path in paths
    ]

    if not samples:
        return

    save_root = out_dir / "gradcam_samples"
    save_root.mkdir(parents=True, exist_ok=True)
    grid_items = []
    for path, model_idx in samples:
        pil = Image.open(path).convert("RGB")
        x = transform(pil).unsqueeze(0).to(device).requires_grad_(True)
        cam = cam_extractor(x, task="model", target_idx=model_idx)
        img_rgb = tensor_to_rgb(x)
        overlay = overlay_cam(img_rgb, cam)
        cls_name = test_ds.model_classes[model_idx]
        cls_dir = save_root / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        out_path = cls_dir / f"{path.stem}_cam.jpg"
        Image.fromarray(overlay).save(out_path, quality=90)
        grid_items.append((cls_name, img_rgb, overlay))

    classes = test_ds.model_classes
    rows = len(classes)
    cols = 6
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.6), dpi=150)
    if rows == 1:
        axes = [axes]

    buckets: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {cls: [] for cls in classes}
    for cls_name, rgb, overlay in grid_items:
        buckets.setdefault(cls_name, []).append((rgb, overlay))

    for row_idx, cls_name in enumerate(classes):
        row_axes = axes[row_idx]
        items = buckets.get(cls_name, [])
        for col_idx in range(3):
            left_ax = row_axes[col_idx * 2]
            right_ax = row_axes[col_idx * 2 + 1]
            if col_idx < len(items):
                rgb, overlay = items[col_idx]
                left_ax.imshow(rgb)
                left_ax.set_title(f"{cls_name}", fontsize=8)
                right_ax.imshow(overlay)
                right_ax.set_title("CAM", fontsize=8)
            left_ax.set_xticks([])
            left_ax.set_yticks([])
            right_ax.set_xticks([])
            right_ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_dir / "gradcam_grid.png", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 训练主函数
# ============================================================

def train(cfg, train_ds, val_ds, test_ds=None):
    progress_cb = _get(cfg, "PROGRESS_CALLBACK", None)
    use_tqdm = bool(_get(cfg, "USE_TQDM", True))
    use_cpu = bool(_get(cfg, "USE_CPU", False))
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not use_cpu else "cpu"
    )
    print(f"[INFO] device = {device}")

    if device.type == "cuda":
        try:
            name = torch.cuda.get_device_name(0)
            print(f"[INFO] GPU = {name}")
        except Exception:
            pass

    # ---- 读取配置 ----
    model_name = normalize_backbone_name(
        str(_get(cfg, "MODEL_NAME", "mobilenetv3_large_100"))
    )
    out_dir_cfg = str(_get(cfg, "OUT_DIR", "")).strip()
    out_dir = resolve_experiment_dir(out_dir_cfg or default_out_dir_for_backbone(model_name))

    batch_size = int(_get(cfg, "BATCH_SIZE", 32))
    workers = int(_get(cfg, "WORKERS", 0))
    lr = float(_get(cfg, "LR", 3e-4))
    wd = float(_get(cfg, "WEIGHT_DECAY", 0.05))
    epochs = int(_get(cfg, "EPOCHS", 30))
    pretrained = bool(_get(cfg, "PRETRAINED", True))
    pretrained_source = str(_get(cfg, "PRETRAINED_SOURCE", "auto"))
    test_every = int(_get(cfg, "TEST_EVERY", 5))

    alpha = float(_get(cfg, "BRAND_LOSS_WEIGHT", 0.3))
    beta = float(_get(cfg, "MODEL_LOSS_WEIGHT", 0.7))
    freeze_ratio = float(_get(cfg, "FREEZE_BACKBONE_RATIO", 0.0))

    grad_accum = int(_get(cfg, "GRAD_ACCUM", 1))
    max_grad_norm = float(_get(cfg, "MAX_GRAD_NORM", 1.0))
    use_amp = bool(_get(cfg, "USE_AMP", True)) and (device.type == "cuda")

    torch.backends.cudnn.benchmark = True

    # ---- 保存标签映射 ----
    save_label_maps(
        str(out_dir),
        train_ds.brand_classes,
        train_ds.model_classes,
        train_ds.brand_to_models,
    )
    history_csv = out_dir / "train_history.csv"
    _init_history_csv(history_csv)
    _save_dataset_distribution(out_dir, train_ds, val_ds, test_ds)

    # ---- DataLoader ----
    pin = device.type == "cuda"
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=pin,
        drop_last=False,
        persistent_workers=(workers > 0),
        prefetch_factor=1 if workers > 0 else None,
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    # ---- 模型 ----
    num_brands = len(train_ds.brand_classes)
    num_models = len(train_ds.model_classes)

    model = DualHeadClassifier(
        backbone_name=model_name,
        num_brands=num_brands,
        num_models=num_models,
        pretrained=pretrained,
        pretrained_source=pretrained_source,
    ).to(device)

    if freeze_ratio > 0:
        freeze_backbone(model, freeze_ratio)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=wd,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- 训练信息 ----
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[INFO] model={model_name} | brands={num_brands} | models={num_models}\n"
        f"[INFO] out_dir={out_dir}\n"
        f"[INFO] params: total={total_params:,} trainable={trainable_params:,}\n"
        f"[INFO] batch={batch_size} | grad_accum={grad_accum} | amp={use_amp}\n"
        f"[INFO] loss weights: brand={alpha} model={beta}"
    )
    if progress_cb:
        progress_cb({
            "stage": "init",
            "message": "训练初始化完成",
            "model_name": model_name,
            "out_dir": str(out_dir),
            "device": str(device),
            "epochs": epochs,
            "batch_size": batch_size,
        })

    # ---- 训练循环 ----
    best_model_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()

        run_brand_loss = 0.0
        run_model_loss = 0.0
        run_brand_correct = 0
        run_model_correct = 0
        run_total = 0

        optimizer.zero_grad(set_to_none=True)
        loader_iter = train_loader
        if use_tqdm:
            loader_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", ncols=120)

        for step, (x, brand_y, model_y) in enumerate(loader_iter, start=1):
            x = x.to(device, non_blocking=True)
            brand_y = brand_y.to(device, non_blocking=True)
            model_y = model_y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                brand_logits, model_logits = model(x)
                b_loss = criterion(brand_logits, brand_y)
                m_loss = criterion(model_logits, model_y)
                loss = (alpha * b_loss + beta * m_loss) / max(grad_accum, 1)

            scaler.scale(loss).backward()

            bs = x.size(0)
            run_brand_loss += b_loss.item() * bs
            run_model_loss += m_loss.item() * bs
            run_brand_correct += (brand_logits.argmax(1) == brand_y).sum().item()
            run_model_correct += (model_logits.argmax(1) == model_y).sum().item()
            run_total += bs

            if step % grad_accum == 0:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            n = max(run_total, 1)
            if use_tqdm:
                loader_iter.set_postfix(
                    b_acc=f"{run_brand_correct / n:.3f}",
                    m_acc=f"{run_model_correct / n:.3f}",
                )

        # 处理末尾未凑满 grad_accum 的梯度
        if len(train_loader) % max(grad_accum, 1) != 0:
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        # ---- 验证 ----
        val_m = evaluate(model, val_loader, device, use_amp)
        n = max(run_total, 1)
        print(
            f"[EPOCH {epoch}] "
            f"train: brand_acc={run_brand_correct / n:.4f} "
            f"model_acc={run_model_correct / n:.4f} | "
            f"val: brand_acc={val_m['brand_acc']:.4f} "
            f"model_acc={val_m['model_acc']:.4f}"
        )
        _append_history_row(
            history_csv,
            epoch,
            run_brand_correct / n,
            run_model_correct / n,
            val_m["brand_acc"],
            val_m["model_acc"],
            val_m["brand_loss"],
            val_m["model_loss"],
            optimizer.param_groups[0]["lr"],
        )
        if progress_cb:
            progress_cb({
                "stage": "epoch_end",
                "epoch": epoch,
                "epochs": epochs,
                "train_brand_acc": run_brand_correct / n,
                "train_model_acc": run_model_correct / n,
                "val_brand_acc": val_m["brand_acc"],
                "val_model_acc": val_m["model_acc"],
                "best_model_acc": best_model_acc,
                "out_dir": str(out_dir),
            })

        # ---- 保存权重 ----
        torch.save(model.state_dict(), out_dir / "last.pt")
        if val_m["model_acc"] > best_model_acc:
            best_model_acc = val_m["model_acc"]
            torch.save(model.state_dict(), out_dir / "best.pt")

        # ---- 可选 test ----
        if test_loader and (epoch == epochs or epoch % test_every == 0):
            test_m = evaluate(model, test_loader, device, use_amp)
            print(
                f"[TEST]  brand_acc={test_m['brand_acc']:.4f} "
                f"model_acc={test_m['model_acc']:.4f}"
            )

    print(f"[DONE] Best Val Model Acc = {best_model_acc:.4f}")
    print(f"[INFO] weights saved in: {out_dir.resolve()}")
    _plot_curves(out_dir, model_name)
    if test_loader is not None:
        best_ckpt = out_dir / "best.pt"
        if best_ckpt.exists():
            state = torch.load(str(best_ckpt), map_location=device)
            model.load_state_dict(state, strict=True)
        (
            brand_true,
            brand_pred,
            model_true,
            model_pred,
        ) = _collect_test_predictions(model, test_loader, device, use_amp)
        brand_metrics = _save_confusion_and_report(
            brand_true,
            brand_pred,
            train_ds.brand_classes,
            "brand",
            out_dir,
            model_name,
        )
        model_metrics = _save_confusion_and_report(
            model_true,
            model_pred,
            train_ds.model_classes,
            "model",
            out_dir,
            model_name,
        )
        _save_dualhead_test_summary(
            out_dir,
            model_name,
            train_ds.brand_classes,
            train_ds.model_classes,
            brand_metrics,
            model_metrics,
        )
        _save_gradcam_artifacts(
            model,
            test_ds,
            out_dir,
            model_name,
            int(_get(cfg, "IMG_SIZE", 224)),
            device,
        )
    if progress_cb:
        progress_cb({
            "stage": "done",
            "message": "训练完成",
            "best_model_acc": best_model_acc,
            "out_dir": str(out_dir.resolve()),
        })
