"""
在 test 集上评估双头模型，输出品牌/型号 Accuracy 与分类报告。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from torchvision import transforms

from models import DualHeadClassifier
from utils import (
    IMG_EXTS,
    build_canonical_model_key,
    check_consistency,
    default_out_dir_for_backbone,
    load_label_maps,
    normalize_backbone_name,
    supported_backbone_choices,
)


def parse_args():
    parser = argparse.ArgumentParser(description="双头手机分类测试集评估")
    parser.add_argument(
        "--backbone",
        choices=supported_backbone_choices(),
        default="mobilenetv3",
        help="选择 backbone：mobilenetv3 或 swin",
    )
    parser.add_argument("--run-dir", default="", help="运行目录，默认按 backbone 自动推导")
    parser.add_argument("--weights", default="", help="权重路径，默认 <run_dir>/best.pt")
    parser.add_argument("--test-dir", default="data/test", help="测试集目录")
    parser.add_argument("--img-size", type=int, default=224)
    return parser.parse_args()


def build_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15),
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def resolve_paths(args):
    model_name = normalize_backbone_name(args.backbone)
    run_dir = Path(args.run_dir or default_out_dir_for_backbone(model_name))
    weights_path = Path(args.weights) if args.weights else run_dir / "best.pt"
    return model_name, run_dir, weights_path


def main():
    args = parse_args()
    model_name, run_dir, weights_path = resolve_paths(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")
    print(f"[INFO] backbone = {model_name}")
    print(f"[INFO] run_dir = {run_dir}")

    brand_classes, model_classes, brand_to_models = load_label_maps(str(run_dir))

    model = DualHeadClassifier(
        backbone_name=model_name,
        num_brands=len(brand_classes),
        num_models=len(model_classes),
        pretrained=False,
    )
    if not weights_path.exists():
        raise FileNotFoundError(f"找不到权重文件: {weights_path.resolve()}")
    state = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    tf = build_transform(args.img_size)

    test_root = Path(args.test_dir)
    if not test_root.exists():
        print(f"[ERROR] test 目录不存在: {test_root.resolve()}")
        return

    brand_true, brand_pred = [], []
    model_true, model_pred = [], []
    consistent_count = 0
    total = 0
    errors = []

    for brand_dir in sorted(test_root.iterdir()):
        if not brand_dir.is_dir():
            continue
        for model_dir in sorted(brand_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            gt_brand = brand_dir.name
            gt_model = build_canonical_model_key(gt_brand, model_dir.name)

            for img_path in sorted(model_dir.iterdir()):
                if not img_path.is_file() or img_path.suffix.lower() not in IMG_EXTS:
                    continue

                try:
                    img = Image.open(img_path).convert("RGB")
                    x = tf(img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        b_logits, m_logits = model(x)

                    b_idx = b_logits.argmax(1).item()
                    m_idx = m_logits.argmax(1).item()
                    pred_brand = brand_classes[b_idx]
                    pred_model = model_classes[m_idx]

                    brand_true.append(gt_brand)
                    brand_pred.append(pred_brand)
                    model_true.append(gt_model)
                    model_pred.append(pred_model)

                    consistent = check_consistency(pred_brand, pred_model, brand_to_models)
                    if consistent:
                        consistent_count += 1
                    total += 1

                    if pred_model != gt_model:
                        b_conf = F.softmax(b_logits, dim=1)[0, b_idx].item()
                        m_conf = F.softmax(m_logits, dim=1)[0, m_idx].item()
                        errors.append(
                            f"  {img_path.name}: "
                            f"真实={gt_brand}/{gt_model} -> "
                            f"预测={pred_brand}({b_conf:.3f})/{pred_model}({m_conf:.3f})"
                        )
                except Exception as e:
                    print(f"[SKIP] {img_path}: {e}")

    if total == 0:
        print("[WARN] test 集中没有找到图片")
        return

    print(f"\n{'='*60}")
    print(f"  测试集评估结果  ({total} 张图片)")
    print(f"{'='*60}")

    print(f"\n--- 厂商分类 ---")
    print(f"Accuracy: {accuracy_score(brand_true, brand_pred):.4f}")
    print(classification_report(brand_true, brand_pred, zero_division=0))

    print(f"--- 型号分类 ---")
    print(f"Accuracy: {accuracy_score(model_true, model_pred):.4f}")
    print(classification_report(model_true, model_pred, zero_division=0))

    print(f"--- 一致性 ---")
    print(f"厂商-型号一致: {consistent_count}/{total} ({consistent_count/total:.4f})")

    if errors:
        print(f"\n--- 错误样本 (共 {len(errors)} 个) ---")
        for e in errors[:20]:
            print(e)
        if len(errors) > 20:
            print(f"  ... 还有 {len(errors) - 20} 个错误")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
