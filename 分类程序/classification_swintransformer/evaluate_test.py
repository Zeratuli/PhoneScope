"""独立事后评估脚本。

用途：在已有 best.pt 的情况下，单独重新生成 confusion_matrix.png /
classification_report.txt / per_class_metrics.csv / test_summary.json，
而无需重训整个模型。

用法示例（在 classification_swintransformer 目录下）：

    # Swin
    python evaluate_test.py --out_dir runs/swin --model_name swin_tiny_patch4_window7_224

    # MobileNetV3
    python evaluate_test.py --out_dir runs/mobilenet --model_name mobilenetv3_large_100

可选参数：
    --data_dir    默认 "data"
    --img_size    默认 224
    --batch_size  默认 32
    --weights     默认 <out_dir>/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# 确保可以 import 同目录的 trainer.py / datasets.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import build_transforms
from trainer import _final_test_eval


def _build_test_loader(data_dir: str, img_size: int, batch_size: int):
    from torchvision import datasets as tv_datasets

    _, val_tf = build_transforms(img_size=img_size, aug_level="light")

    test_dir = Path(data_dir) / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"测试集目录不存在: {test_dir.resolve()}")

    test_ds = tv_datasets.ImageFolder(str(test_dir), transform=val_tf)
    loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=False,
    )
    return loader, test_ds.classes


def main():
    parser = argparse.ArgumentParser(description="事后评估已训练模型")
    parser.add_argument("--out_dir", required=True,
                        help="模型权重所在目录（含 best.pt / classes.txt）")
    parser.add_argument("--model_name", required=True,
                        help="timm 模型名，如 swin_tiny_patch4_window7_224")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weights", default="",
                        help="权重路径，默认 <out_dir>/best.pt")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    weights = Path(args.weights) if args.weights else out_dir / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"找不到权重文件: {weights.resolve()}")

    classes_file = out_dir / "classes.txt"
    if not classes_file.exists():
        raise FileNotFoundError(
            f"找不到类别文件: {classes_file.resolve()}\n"
            f"（请先训练，或手工放置 classes.txt 与类别目录保持一致）")
    classes = [ln.strip() for ln in
               classes_file.read_text(encoding="utf-8").splitlines() if ln.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")
    print(f"[INFO] weights = {weights}")
    print(f"[INFO] classes = {classes}")

    import timm
    model = timm.create_model(args.model_name, pretrained=False,
                              num_classes=len(classes)).to(device)
    state = torch.load(str(weights), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    test_loader, test_classes = _build_test_loader(
        args.data_dir, args.img_size, args.batch_size)

    if test_classes != classes:
        print("[WARN] test 目录类别顺序与 classes.txt 不一致:")
        print(f"  test : {test_classes}")
        print(f"  file : {classes}")
        print("  将以 classes.txt 为准进行评估。")

    _final_test_eval(
        model, test_loader, out_dir,
        classes=classes, model_name=args.model_name, device=device,
    )


if __name__ == "__main__":
    main()
