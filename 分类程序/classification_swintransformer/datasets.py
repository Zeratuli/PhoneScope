# datasets.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, List

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.io import read_image, ImageReadMode


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_transforms(img_size: int = 224, aug_level: str = "light") -> Tuple[transforms.Compose, transforms.Compose]:
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    if aug_level == "none":
        train_tf = transforms.Compose([
            transforms.Resize(int(img_size * 1.15), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])
        return train_tf, val_tf

    if aug_level == "medium":
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0),
                                         interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.12, 0.12, 0.12, 0.02),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])
        return train_tf, val_tf

    # light（推荐）：比 RandomResizedCrop 更省 CPU
    train_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.08, 0.08, 0.08, 0.02),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])
    return train_tf, val_tf


class FastImageFolder(Dataset):
    """
    ImageFolder 的加速版：
    - 读取后端可选：PIL（兼容） / torchvision.io.read_image（通常更快）
    - 可选 RAM 缓存（训练集推荐可开，val/test一般不用）
    """
    def __init__(self, root: str, transform=None, read_backend: str = "tv", cache_mode: str = "none"):
        self.root = Path(root)
        self.transform = transform
        self.read_backend = read_backend  # "pil" or "tv"
        self.cache_mode = cache_mode      # "none" or "ram"

        # 复用 ImageFolder 的类别映射
        tmp = datasets.ImageFolder(str(self.root))
        self.classes = tmp.classes
        self.class_to_idx = tmp.class_to_idx
        self.samples = tmp.samples  # List[(path, class_idx)]

        self._ram_cache = None
        if self.cache_mode == "ram":
            # 注意：缓存的是“解码后的 Tensor(C,H,W)”；占内存
            self._ram_cache = [None] * len(self.samples)

    def __len__(self):
        return len(self.samples)

    def _read_tv(self, path: str) -> torch.Tensor:
        # read_image: uint8 [C,H,W], RGB
        img = read_image(path, mode=ImageReadMode.RGB)
        return img  # uint8 tensor

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]

        if self._ram_cache is not None and self._ram_cache[idx] is not None:
            img = self._ram_cache[idx]
        else:
            if self.read_backend == "tv":
                img = self._read_tv(path)  # uint8 tensor
            else:
                # PIL 分支：交给 ImageFolder 的默认 loader
                # 为了不引入额外依赖，我们用 ImageFolder 的 loader
                from torchvision.datasets.folder import default_loader
                img = default_loader(path)

            if self._ram_cache is not None:
                self._ram_cache[idx] = img

        if self.read_backend == "tv":
            # torchvision transforms 期望 PIL 或 Tensor？
            # 我们用 ToPILImage 统一成 PIL 再走你已有 transforms（最稳）
            if self.transform is not None:
                img = transforms.ToPILImage()(img)
                img = self.transform(img)
            else:
                img = img.float() / 255.0
        else:
            if self.transform is not None:
                img = self.transform(img)

        return img, label


def build_datasets(data_dir: str, img_size: int, aug_level: str = "light",
                   read_backend: str = "tv", cache_mode: str = "none"):
    data_dir = Path(data_dir)
    train_tf, val_tf = build_transforms(img_size, aug_level)

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"请确认存在：\n{train_dir}\n{val_dir}")

    # train 用 FastImageFolder（可缓存、可加速）
    train_ds = FastImageFolder(str(train_dir), transform=train_tf, read_backend=read_backend, cache_mode=cache_mode)
    # val/test 不建议缓存（省内存），读取后端照样可以 tv
    val_ds = FastImageFolder(str(val_dir), transform=val_tf, read_backend=read_backend, cache_mode="none")

    test_ds = None
    if test_dir.exists():
        try:
            test_ds = FastImageFolder(str(test_dir), transform=val_tf, read_backend=read_backend, cache_mode="none")
        except Exception:
            test_ds = None

    return train_ds, val_ds, test_ds
