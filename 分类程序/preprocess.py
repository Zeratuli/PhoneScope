"""
批量背景去除 + 自动划分 train/val/test

将 raw_data/{brand}/{model}/*.jpg 中的原始手机照片:
  1. 去除背景
  2. 按比例自动划分到 data/train, data/val, data/test

使用方式:
    pip install rembg onnxruntime
    python preprocess.py

可选参数:
    python preprocess.py --input raw_data --output data --ratio 0.7 0.15 0.15
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from rembg import remove
from tqdm import tqdm

from utils import build_canonical_model_key


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def process_image(
    src: Path,
    dst: Path,
    bg_color: str = "white",
    max_size: int = 0,
) -> bool:
    """去除背景并保存。"""
    try:
        img = Image.open(src).convert("RGB")
        result = remove(img)

        if bg_color == "white":
            white_bg = Image.new("RGB", result.size, (255, 255, 255))
            white_bg.paste(result, mask=result.split()[3])
            result = white_bg

        if max_size > 0:
            w, h = result.size
            if max(w, h) > max_size:
                scale = max_size / max(w, h)
                result = result.resize(
                    (int(w * scale), int(h * scale)),
                    Image.LANCZOS,
                )

        dst.parent.mkdir(parents=True, exist_ok=True)

        if bg_color == "white":
            result.save(str(dst), format="JPEG", quality=95)
        else:
            result.save(str(dst), format="PNG")

        return True
    except Exception as e:
        print(f"[FAIL] {src} -> {e}")
        return False


def split_files(
    files: List[Path],
    ratio: Tuple[float, float, float],
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """将文件列表按比例随机划分为 train/val/test。"""
    rng = random.Random(seed)
    shuffled = list(files)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = max(1, int(n * ratio[0]))
    n_val = max(1, int(n * ratio[1]))

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]

    # 如果 test 为空但还有剩余比例，至少保证每个集合有样本
    if not test and n >= 3:
        test = [val.pop()]

    return train, val, test


def collect_by_category(input_root: Path) -> dict:
    """
    扫描 input_root/{brand}/{model}/ 结构，按 (brand, model) 分组收集图片。

    返回: {(brand, model): [Path, ...], ...}
    """
    groups = {}

    for brand_dir in sorted(input_root.iterdir()):
        if not brand_dir.is_dir():
            continue
        for model_dir in sorted(brand_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            imgs = sorted(
                p for p in model_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMG_EXTS
            )
            if imgs:
                groups[(brand_dir.name, model_dir.name)] = imgs

    return groups


def main():
    parser = argparse.ArgumentParser(description="批量去背景 + 自动划分数据集")
    parser.add_argument("--input", default="raw_data", help="原始图片根目录 (brand/model/)")
    parser.add_argument("--output", default="data", help="输出根目录 (自动生成 train/val/test)")
    parser.add_argument(
        "--ratio", nargs=3, type=float, default=[0.7, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
        help="train/val/test 划分比例 (默认 0.7 0.15 0.15)",
    )
    parser.add_argument(
        "--bg", default="white", choices=["white", "transparent"],
        help="背景类型: white=白底(JPG), transparent=透明底(PNG)",
    )
    parser.add_argument(
        "--size", type=int, default=0,
        help="长边最大尺寸, 0=不缩放",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="跳过已存在的输出文件",
    )
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)
    ratio = tuple(args.ratio)

    if abs(sum(ratio) - 1.0) > 0.01:
        print(f"[ERROR] 比例之和应为 1.0, 当前为 {sum(ratio):.2f}")
        return

    if not input_root.exists():
        print(f"[ERROR] 输入目录不存在: {input_root.resolve()}")
        return

    groups = collect_by_category(input_root)
    if not groups:
        print(f"[WARN] 在 {input_root.resolve()} 中没有找到图片")
        print(f"[HINT] 请按 raw_data/厂商/型号/ 结构放置图片，例如:")
        print(f"       raw_data/Apple/iPhone_13/photo1.jpg")
        return

    # 统计并展示
    total_images = sum(len(v) for v in groups.values())
    print(f"[INFO] 发现 {len(groups)} 个型号, 共 {total_images} 张图片")
    print(f"[INFO] 划分比例: train={ratio[0]}, val={ratio[1]}, test={ratio[2]}")
    print(f"[INFO] 背景: {args.bg}, 缩放: {'不缩放' if args.size == 0 else args.size}")
    print()

    # 按类别划分并处理
    tasks = []  # (src_path, dst_path)

    for (brand, model_name), files in sorted(groups.items()):
        train_files, val_files, test_files = split_files(files, ratio, args.seed)

        print(
            f"  {brand}/{model_name}: "
            f"{len(files)} 张 -> "
            f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)}"
        )

        ext = ".jpg" if args.bg == "white" else ".png"

        for split_name, split_files_list in [
            ("train", train_files),
            ("val", val_files),
            ("test", test_files),
        ]:
            for i, src in enumerate(split_files_list):
                canonical_model = build_canonical_model_key(brand, model_name)
                dst_name = f"{canonical_model}_{i + 1:05d}{ext}"
                dst = output_root / split_name / brand / model_name / dst_name
                tasks.append((src, dst))

    # 过滤已存在
    if args.skip_existing:
        before = len(tasks)
        tasks = [(s, d) for s, d in tasks if not d.exists()]
        skipped = before - len(tasks)
        if skipped > 0:
            print(f"\n[INFO] 跳过 {skipped} 个已存在文件")

    print(f"\n[INFO] 开始处理 {len(tasks)} 张图片...")

    success = 0
    fail = 0

    for src, dst in tqdm(tasks, desc="去背景+划分", ncols=100):
        if process_image(src, dst, bg_color=args.bg, max_size=args.size):
            success += 1
        else:
            fail += 1

    print(f"\n[DONE] 成功: {success}, 失败: {fail}")
    print(f"[INFO] 输出目录: {output_root.resolve()}")
    print(f"[HINT] 下一步: python run_train.py")


if __name__ == "__main__":
    main()
