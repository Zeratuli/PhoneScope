"""数据集统计与可视化。

产出两张图与一份 CSV：
  - class_distribution.png  —— 分组柱状图：横轴类别，3 根柱子代表 train/val/test
  - class_pie.png           —— 三类整体分布饼图
  - class_distribution.csv  —— 表格化的数字

用法：
  python visualize_dataset.py
  python visualize_dataset.py --data_dir D:\\path\\to\\data --out_dir runs\\_dataset
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(root: Path) -> dict[str, dict[str, int]]:
    """返回 {class_name: {split: count}}"""
    out: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            cnt = sum(1 for p in cls_dir.iterdir()
                      if p.is_file() and p.suffix.lower() in IMG_EXTS)
            out.setdefault(cls_dir.name, {"train": 0, "val": 0, "test": 0})
            out[cls_dir.name][split] = cnt
    return out


def save_csv(stats: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "train", "val", "test", "total"])
        for cls, counts in stats.items():
            total = sum(counts.values())
            w.writerow([cls, counts["train"], counts["val"],
                        counts["test"], total])


def plot_grouped_bar(stats: dict, path: Path) -> None:
    classes = list(stats.keys())
    splits = ("train", "val", "test")
    colors = {"train": "#2563eb", "val": "#f59e0b", "test": "#10b981"}

    x = np.arange(len(classes))
    width = 0.26

    fig, ax = plt.subplots(figsize=(max(7, len(classes) * 1.6), 4.5), dpi=150)
    for i, split in enumerate(splits):
        counts = [stats[c][split] for c in classes]
        bars = ax.bar(x + (i - 1) * width, counts, width,
                      label=split, color=colors[split])
        for b, v in zip(bars, counts):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                    str(v), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ") for c in classes],
                       rotation=15, ha="right")
    ax.set_ylabel("Number of samples")
    ax.set_title("Dataset Class Distribution (per split)")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_pie(stats: dict, path: Path) -> None:
    classes = list(stats.keys())
    totals = [sum(stats[c].values()) for c in classes]
    colors = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#06b6d4"]

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.pie(totals,
           labels=[c.replace("_", " ") for c in classes],
           colors=colors[: len(classes)],
           autopct=lambda p: f"{p:.1f}%\n({int(round(p * sum(totals) / 100))})",
           startangle=90, wedgeprops={"linewidth": 1.2, "edgecolor": "white"},
           textprops={"fontsize": 9})
    ax.set_title(f"Overall Class Distribution (total = {sum(totals)})")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "runs" / "_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = count_images(Path(args.data_dir))
    if not stats:
        raise SystemExit(f"在 {args.data_dir} 没找到任何类别子目录")

    csv_path = out_dir / "class_distribution.csv"
    bar_path = out_dir / "class_distribution.png"
    pie_path = out_dir / "class_pie.png"

    save_csv(stats, csv_path)
    plot_grouped_bar(stats, bar_path)
    plot_pie(stats, pie_path)

    # 控制台汇总
    print("=" * 55)
    print(f"Dataset root : {args.data_dir}")
    print(f"Output dir   : {out_dir}")
    print("-" * 55)
    print(f"{'class':<20} {'train':>8} {'val':>6} {'test':>6} {'total':>7}")
    print("-" * 55)
    grand_total = 0
    for cls, counts in stats.items():
        tot = sum(counts.values())
        grand_total += tot
        print(f"{cls:<20} {counts['train']:>8} {counts['val']:>6} "
              f"{counts['test']:>6} {tot:>7}")
    print("-" * 55)
    print(f"{'SUM':<20} {'':>8} {'':>6} {'':>6} {grand_total:>7}")
    print(f"\n[OK] csv  -> {csv_path}")
    print(f"[OK] bar  -> {bar_path}")
    print(f"[OK] pie  -> {pie_path}")


if __name__ == "__main__":
    main()
