from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Dict, Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader

from utils import build_canonical_model_key, build_label_maps, IMG_EXTS


# ============================================================
# 数据增强
# ============================================================

def build_transforms(
    img_size: int = 224,
    aug_level: str = "light",
) -> Tuple[transforms.Compose, transforms.Compose]:
    """返回 (train_transform, val_transform)。"""

    val_tf = transforms.Compose([
        transforms.Resize(
            int(img_size * 1.15),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if aug_level == "none":
        return val_tf, val_tf

    if aug_level == "medium":
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.8, 1.0),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.15, 0.15, 0.15, 0.03),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        return train_tf, val_tf

    # light (默认)
    train_tf = transforms.Compose([
        transforms.Resize(
            int(img_size * 1.15),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.08, 0.08, 0.08, 0.02),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return train_tf, val_tf


# ============================================================
# 层级手机数据集
# ============================================================

class HierarchicalPhoneDataset(Dataset):
    """
    两层目录数据集: root/{brand}/{model}/*.jpg

    每条样本返回 (image_tensor, brand_idx, model_idx)。
    """

    def __init__(
        self,
        root: str,
        transform: Optional[transforms.Compose] = None,
        brand_classes: Optional[List[str]] = None,
        model_classes: Optional[List[str]] = None,
        brand_to_models: Optional[Dict[str, List[str]]] = None,
    ):
        self.root = Path(root)
        self.transform = transform

        if brand_classes is None or model_classes is None or brand_to_models is None:
            brand_classes, model_classes, brand_to_models = build_label_maps(root)

        self.brand_classes = brand_classes
        self.model_classes = model_classes
        self.brand_to_models = brand_to_models

        self.brand_to_idx = {b: i for i, b in enumerate(self.brand_classes)}
        self.model_to_idx = {m: i for i, m in enumerate(self.model_classes)}

        # 扫描所有图片: (path, brand_idx, model_idx)
        self.samples: List[Tuple[str, int, int]] = []
        for brand_dir in sorted(self.root.iterdir()):
            if not brand_dir.is_dir():
                continue
            brand_name = brand_dir.name
            if brand_name not in self.brand_to_idx:
                continue
            brand_idx = self.brand_to_idx[brand_name]

            for model_dir in sorted(brand_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                model_name = build_canonical_model_key(brand_name, model_dir.name)
                if model_name not in self.model_to_idx:
                    continue
                model_idx = self.model_to_idx[model_name]

                for img_path in sorted(model_dir.iterdir()):
                    if img_path.is_file() and img_path.suffix.lower() in IMG_EXTS:
                        self.samples.append((str(img_path), brand_idx, model_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        path, brand_idx, model_idx = self.samples[idx]
        img = default_loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, brand_idx, model_idx


# ============================================================
# 构建 train / val / test 数据集
# ============================================================

def build_datasets(
    data_dir: str,
    img_size: int = 224,
    aug_level: str = "light",
) -> Tuple[HierarchicalPhoneDataset, HierarchicalPhoneDataset, Optional[HierarchicalPhoneDataset]]:
    """
    根据 data_dir 下的 train/val/test 子目录构建数据集。
    所有子集共享相同的 brand_classes 和 model_classes (以 train 为准)。
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"请确认存在:\n  {train_dir}\n  {val_dir}")

    train_tf, val_tf = build_transforms(img_size, aug_level)

    # 以 train 的标签映射为基准
    brand_classes, model_classes, brand_to_models = build_label_maps(str(train_dir))

    train_ds = HierarchicalPhoneDataset(
        str(train_dir), train_tf, brand_classes, model_classes, brand_to_models
    )
    val_ds = HierarchicalPhoneDataset(
        str(val_dir), val_tf, brand_classes, model_classes, brand_to_models
    )

    test_ds = None
    if test_dir.exists():
        try:
            test_ds = HierarchicalPhoneDataset(
                str(test_dir), val_tf, brand_classes, model_classes, brand_to_models
            )
        except Exception:
            test_ds = None

    num_train = len(train_ds)
    num_val = len(val_ds)
    num_test = len(test_ds) if test_ds else 0
    print(f"[DATA] brands={len(brand_classes)}, models={len(model_classes)}")
    print(f"[DATA] train={num_train}, val={num_val}, test={num_test}")

    return train_ds, val_ds, test_ds
