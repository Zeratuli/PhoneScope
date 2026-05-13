"""
手机分类推理脚本 - 双头预测 + 一致性校验 + 3D匹配接口 + CSV + 可视化
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models import DualHeadClassifier, match_3d
from utils import (
    IMG_EXTS,
    check_consistency,
    default_out_dir_for_backbone,
    load_label_maps,
    normalize_backbone_name,
    supported_backbone_choices,
)


def parse_args():
    parser = argparse.ArgumentParser(description="双头手机分类推理")
    parser.add_argument(
        "--backbone",
        choices=supported_backbone_choices(),
        default="mobilenetv3",
        help="选择 backbone：mobilenetv3 或 swin",
    )
    parser.add_argument("--run-dir", default="", help="运行目录，默认按 backbone 自动推导")
    parser.add_argument("--weights", default="", help="权重路径，默认 <run_dir>/best.pt")
    parser.add_argument("--input", default="data/test", help="输入图片或目录")
    parser.add_argument("--save-csv", default="", help="结果 CSV 路径，默认 <run_dir>/predict_results.csv")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--show-seconds", type=int, default=3)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--match-3d-db", default="", help="3D 特征数据库 .pt 路径")
    return parser.parse_args()


def build_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(
            int(img_size * 1.15),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def iter_images(input_dir: str):
    root = Path(input_dir)
    if root.is_file():
        if root.suffix.lower() in IMG_EXTS:
            yield root
        return

    if not root.exists():
        raise FileNotFoundError(f"输入路径不存在: {root.resolve()}")

    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def load_3d_database(db_path: Optional[str]) -> Optional[Dict]:
    if not db_path:
        return None
    p = Path(db_path)
    if not p.exists():
        print(f"[WARN] 3D 数据库不存在: {p.resolve()}, 跳过 3D 匹配")
        return None
    return torch.load(str(p), map_location="cpu")


@torch.no_grad()
def predict_one(
    model: DualHeadClassifier,
    tf: transforms.Compose,
    img_path: Path,
    device: torch.device,
    brand_classes: List[str],
    model_classes: List[str],
    brand_to_models: Dict[str, List[str]],
    topk: int = 3,
    db_3d: Optional[Dict] = None,
) -> Dict:
    pil_img = Image.open(img_path).convert("RGB")
    x = tf(pil_img).unsqueeze(0).to(device)

    brand_logits, model_logits = model(x)
    features = model.extract_features(x)

    brand_prob = F.softmax(brand_logits, dim=1)[0]
    model_prob = F.softmax(model_logits, dim=1)[0]

    bk = min(topk, len(brand_classes))
    b_scores, b_idxs = torch.topk(brand_prob, bk)
    brand_topk = [
        (brand_classes[i], round(s, 4))
        for s, i in zip(b_scores.tolist(), b_idxs.tolist())
    ]

    mk = min(topk, len(model_classes))
    m_scores, m_idxs = torch.topk(model_prob, mk)
    model_topk = [
        (model_classes[i], round(s, 4))
        for s, i in zip(m_scores.tolist(), m_idxs.tolist())
    ]

    brand_name = brand_topk[0][0]
    model_name = model_topk[0][0]
    consistent = check_consistency(brand_name, model_name, brand_to_models)

    return {
        "brand": {
            "name": brand_name,
            "confidence": brand_topk[0][1],
            "topk": brand_topk,
        },
        "model": {
            "name": model_name,
            "confidence": model_topk[0][1],
            "topk": model_topk,
        },
        "consistent": consistent,
        "features": features.cpu(),
        "match_3d": match_3d(features, db_3d),
        "pil_image": pil_img,
    }


def pil_to_cv2(pil_img: Image.Image, max_h: int = 720) -> np.ndarray:
    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    if h > max_h:
        scale = max_h / h
        bgr = cv2.resize(
            bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    return bgr


def make_text_panel(lines: List[str], h: int, w: int = 540) -> np.ndarray:
    panel = np.full((h, w, 3), 245, dtype=np.uint8)
    y = 30
    for i, line in enumerate(lines):
        font_scale = 0.65 if i == 0 else 0.50
        thickness = 2 if i == 0 else 1
        cv2.putText(
            panel, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (10, 10, 10), thickness, cv2.LINE_AA,
        )
        y += 28
    return panel


def resolve_paths(args):
    model_name = normalize_backbone_name(args.backbone)
    run_dir = Path(args.run_dir or default_out_dir_for_backbone(model_name))
    weights_path = Path(args.weights) if args.weights else run_dir / "best.pt"
    save_csv = Path(args.save_csv) if args.save_csv else run_dir / "predict_results.csv"
    return model_name, run_dir, weights_path, save_csv


def main():
    args = parse_args()
    model_name, run_dir, weights_path, save_csv = resolve_paths(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")
    print(f"[INFO] backbone = {model_name}")
    print(f"[INFO] run_dir = {run_dir}")

    brand_classes, model_classes, brand_to_models = load_label_maps(str(run_dir))
    print(f"[INFO] brands={brand_classes}, models={model_classes}")

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
    db_3d = load_3d_database(args.match_3d_db)

    images = list(iter_images(args.input))
    if not images:
        raise RuntimeError(f"在 {Path(args.input).resolve()} 中没有找到图片")
    print(f"[INFO] found {len(images)} images")

    save_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_header = [
        "image", "brand_pred", "brand_conf",
        "model_pred", "model_conf", "consistent",
    ]
    csv_rows = []

    window_name = "Phone Classifier (ESC=quit, SPACE=pause, n=next)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    paused = False

    for n, img_path in enumerate(images, start=1):
        while paused:
            key = cv2.waitKey(50) & 0xFF
            if key == 27:
                _write_csv(save_csv, csv_header, csv_rows)
                cv2.destroyAllWindows()
                return
            if key == ord(" "):
                paused = False
            elif key == ord("n"):
                break

        try:
            result = predict_one(
                model, tf, img_path, device,
                brand_classes, model_classes, brand_to_models,
                topk=args.topk, db_3d=db_3d,
            )
        except Exception as e:
            print(f"[SKIP] {img_path} | {e}")
            continue

        brand = result["brand"]
        mdl = result["model"]
        consistent = result["consistent"]
        match_info = result["match_3d"]

        status = "OK" if consistent else "MISMATCH"
        print(
            f"[{n}/{len(images)}] {img_path.name} -> "
            f"Brand: {brand['name']}({brand['confidence']:.3f}) "
            f"Model: {mdl['name']}({mdl['confidence']:.3f}) [{status}]"
        )

        csv_rows.append([
            str(img_path), brand["name"], f"{brand['confidence']:.4f}",
            mdl["name"], f"{mdl['confidence']:.4f}", str(consistent),
        ])

        img_bgr = pil_to_cv2(result["pil_image"], max_h=720)
        h = img_bgr.shape[0]

        lines = [
            f"[{n}/{len(images)}] {img_path.name}",
            "",
            f"Brand: {brand['name']}  ({brand['confidence']:.4f})",
        ]
        for rank, (bname, bconf) in enumerate(brand["topk"], 1):
            lines.append(f"  Top-{rank}: {bname}  {bconf:.4f}")
        lines.append("")
        lines.append(f"Model: {mdl['name']}  ({mdl['confidence']:.4f})")
        for rank, (mname, mconf) in enumerate(mdl["topk"], 1):
            lines.append(f"  Top-{rank}: {mname}  {mconf:.4f}")
        lines.append("")
        lines.append(f"Consistent: {status}")

        if match_info:
            lines.append(
                f"3D Match: {match_info.get('matched_model', 'N/A')} "
                f"(sim={match_info.get('similarity', 0):.4f})"
            )

        lines.append("")
        lines.append("Keys: ESC quit | SPACE pause | n next")

        panel = make_text_panel(lines, h=h)
        canvas = np.hstack([img_bgr, panel])
        cv2.imshow(window_name, canvas)

        t0 = time.time()
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                _write_csv(save_csv, csv_header, csv_rows)
                cv2.destroyAllWindows()
                return
            if key == ord(" "):
                paused = True
                break
            if key == ord("n"):
                break
            if (time.time() - t0) >= args.show_seconds:
                break

    _write_csv(save_csv, csv_header, csv_rows)
    print(f"[DONE] results saved to: {save_csv.resolve()}")
    cv2.destroyAllWindows()


def _write_csv(path: Path, header: list, rows: list):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":
    main()
