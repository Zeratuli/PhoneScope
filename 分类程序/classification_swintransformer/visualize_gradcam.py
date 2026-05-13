"""GradCAM 可视化脚本（Swin Transformer / MobileNetV3）。

在测试集中对每个类别抽取若干张样本，产出 GradCAM 热力图叠加到原图上，
保存为一张汇总大图。依赖纯 PyTorch + OpenCV + matplotlib，不需要
pytorch-grad-cam 第三方包。

用法：
  conda activate classification
  cd 分类程序/classification_swintransformer
  python visualize_gradcam.py --out_dir runs/swin     --model_name swin_tiny_patch4_window7_224
  python visualize_gradcam.py --out_dir runs/mobilenet --model_name mobilenetv3_large_100

输出：
  <out_dir>/gradcam_samples/<class>/<basename>_cam.jpg
  <out_dir>/gradcam_grid.png    (汇总大图，K 行 × N 列)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


# ----------------------------- 通用 GradCAM -----------------------------

class GradCAM:
    """适用于任意 nn.Module 的 GradCAM。

    - reshape_fn(activation) -> (B, C, H, W)
      Swin 等 Transformer 的 feature 是 (B, H*W, C) 或 (B, H, W, C)，需要重排。
    """

    def __init__(self, model, target_layer, reshape_fn: Callable | None = None):
        self.model = model.eval()
        self.target_layer = target_layer
        self.reshape_fn = reshape_fn
        self._activations = None
        self._gradients = None

        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self._activations = out

    def _bwd_hook(self, module, grad_in, grad_out):
        self._gradients = grad_out[0]

    def __call__(self, x: torch.Tensor, target_idx: int | None = None) -> np.ndarray:
        """x: (1, 3, H, W) on model device。返回归一化后的 (H, W) heatmap。"""
        logits = self.model(x)
        if target_idx is None:
            target_idx = int(logits.argmax(dim=1).item())
        self.model.zero_grad(set_to_none=True)
        logits[0, target_idx].backward(retain_graph=False)

        activations = self._activations
        gradients = self._gradients
        if self.reshape_fn is not None:
            activations = self.reshape_fn(activations)
            gradients = self.reshape_fn(gradients)

        # activations / gradients 形状均为 (1, C, H, W)
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.squeeze().detach().cpu().numpy()


# ----------------------------- 模型相关工具 -----------------------------

def _swin_reshape(t: torch.Tensor) -> torch.Tensor:
    """Swin Transformer (B, H, W, C) 或 (B, H*W, C) -> (B, C, H, W)。"""
    if t.dim() == 4:
        # (B, H, W, C)
        return t.permute(0, 3, 1, 2).contiguous()
    if t.dim() == 3:
        # (B, N, C) — N 必须是完全平方
        b, n, c = t.shape
        h = int(round(n ** 0.5))
        w = n // h
        return t.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    return t


def pick_target_layer(model, model_name: str):
    """对不同骨干网络挑选合适的 GradCAM 目标层。"""
    name_lower = model_name.lower()
    if "swin" in name_lower:
        # timm Swin：最后一个 stage 的最后一个 block 的 norm2
        try:
            layer = model.layers[-1].blocks[-1].norm2
        except (AttributeError, IndexError):
            # Swin V2 结构略有不同
            layer = model.layers[-1].blocks[-1].norm1
        return layer, _swin_reshape
    if "mobilenet" in name_lower:
        # timm MobileNetV3：最后一个 block（Conv+BN）
        try:
            layer = model.blocks[-1]
        except (AttributeError, IndexError):
            layer = model.conv_head
        return layer, None
    # 兜底：最后一个 Conv2d
    last_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError(f"未知骨干 {model_name}，找不到合适的 GradCAM 层")
    return last_conv, None


# ----------------------------- 图像处理 -----------------------------

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def build_preprocess(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN.tolist(), _IMAGENET_STD.tolist()),
    ])


def tensor_to_rgb(x: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) normalized -> (H,W,3) uint8 RGB"""
    img = x.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    img = img * _IMAGENET_STD + _IMAGENET_MEAN
    img = np.clip(img, 0, 1) * 255
    return img.astype(np.uint8)


def overlay_cam(img_rgb: np.ndarray, cam: np.ndarray,
                alpha: float = 0.45) -> np.ndarray:
    """把 CAM (H',W') 叠加到 RGB 图 (H, W, 3)。"""
    h, w = img_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (heatmap * alpha + img_rgb * (1 - alpha)).astype(np.uint8)
    return overlay


# ----------------------------- 主流程 -----------------------------

def iter_class_samples(data_dir: Path, classes: list[str],
                       per_class: int) -> dict:
    """每类取前 per_class 张测试图。"""
    out = {}
    for cls in classes:
        d = data_dir / "test" / cls
        if not d.exists():
            print(f"[WARN] test dir 不存在: {d}")
            out[cls] = []
            continue
        imgs = sorted(p for p in d.iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        out[cls] = imgs[:per_class]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True,
                    help="模型目录，例如 runs/swin 或 runs/mobilenet")
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--weights", default="",
                    help="权重路径，默认 <out_dir>/best.pt")
    ap.add_argument("--classes_file", default="",
                    help="类别文件，默认 <out_dir>/classes.txt")
    ap.add_argument("--data_dir", default=str(HERE / "data"))
    ap.add_argument("--per_class", type=int, default=3,
                    help="每类可视化几张（默认 3）")
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    weights = Path(args.weights) if args.weights else out_dir / "best.pt"
    classes_file = Path(args.classes_file) if args.classes_file \
        else out_dir / "classes.txt"
    if not weights.exists():
        raise FileNotFoundError(f"权重不存在: {weights}")
    if not classes_file.exists():
        raise FileNotFoundError(f"classes.txt 不存在: {classes_file}")

    classes = [ln.strip() for ln in
               classes_file.read_text(encoding="utf-8").splitlines()
               if ln.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import timm
    model = timm.create_model(args.model_name, pretrained=False,
                              num_classes=len(classes)).to(device)
    state = torch.load(str(weights), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    target_layer, reshape_fn = pick_target_layer(model, args.model_name)
    cam_extractor = GradCAM(model, target_layer, reshape_fn)

    preprocess = build_preprocess(args.img_size)
    samples = iter_class_samples(Path(args.data_dir), classes, args.per_class)

    cam_root = out_dir / "gradcam_samples"
    cam_root.mkdir(parents=True, exist_ok=True)

    # 收集用于 grid 的小图
    grid_items: list[tuple[str, Path, np.ndarray, np.ndarray, int, float]] = []
    for cls_idx, cls in enumerate(classes):
        for p in samples.get(cls, []):
            pil = Image.open(p).convert("RGB")
            x = preprocess(pil).unsqueeze(0).to(device).requires_grad_(True)
            cam = cam_extractor(x, target_idx=cls_idx)
            img_rgb = tensor_to_rgb(x)
            overlay = overlay_cam(img_rgb, cam)

            # 预测置信度（用来显示）
            with torch.no_grad():
                prob = F.softmax(model(x), dim=1)[0]
                pred_idx = int(prob.argmax().item())
                conf = float(prob[pred_idx].item())

            save_dir = cam_root / cls
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / f"{p.stem}_cam.jpg"
            Image.fromarray(overlay).save(out_path, quality=90)
            print(f"[OK] {p.name}  gt={cls}  pred={classes[pred_idx]}  "
                  f"conf={conf:.3f}  -> {out_path.name}")
            grid_items.append((cls, p, img_rgb, overlay, pred_idx, conf))

    # ---------- 汇总网格图 ----------
    if grid_items:
        rows = len(classes)
        cols = max(args.per_class, 1) * 2  # 原图 + 叠加
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(cols * 2.4, rows * 2.6), dpi=150)
        if rows == 1:
            axes = np.array([axes])
        if cols == 1:
            axes = axes[:, None]

        # 把 grid_items 按类别分桶
        buckets = {cls: [] for cls in classes}
        for cls, p, rgb, overlay, pred_idx, conf in grid_items:
            buckets[cls].append((p, rgb, overlay, pred_idx, conf))

        for r, cls in enumerate(classes):
            items = buckets.get(cls, [])
            for c in range(args.per_class):
                left = axes[r, c * 2]
                right = axes[r, c * 2 + 1]
                if c < len(items):
                    p, rgb, overlay, pred_idx, conf = items[c]
                    left.imshow(rgb)
                    left.set_title(f"{cls}", fontsize=8)
                    right.imshow(overlay)
                    right.set_title(
                        f"→ {classes[pred_idx]} ({conf*100:.1f}%)",
                        fontsize=8)
                for ax in (left, right):
                    ax.set_xticks([]); ax.set_yticks([])

        plt.suptitle(f"Grad-CAM on test set ({args.model_name})", fontsize=12)
        plt.tight_layout()
        grid_path = out_dir / "gradcam_grid.png"
        plt.savefig(grid_path, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[OK] grid saved -> {grid_path}")


if __name__ == "__main__":
    main()
