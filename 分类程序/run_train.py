"""
训练入口 - 读取 config.py 配置, 构建数据集并启动训练。

使用方式:
    conda activate classification
    python run_train.py

推荐方式:
    python run_train.py --backbone mobilenetv3
    python run_train.py --backbone swin
"""

import argparse

import config as cfg
from datasets import build_datasets
from trainer import train
from utils import default_out_dir_for_backbone, normalize_backbone_name, supported_backbone_choices


def parse_args():
    parser = argparse.ArgumentParser(description="双头手机分类训练入口")
    parser.add_argument(
        "--backbone",
        choices=supported_backbone_choices(),
        default=None,
        help="选择训练 backbone：mobilenetv3 或 swin",
    )
    parser.add_argument("--data-dir", default=None, help="训练数据根目录，默认读取 config.DATA_DIR")
    parser.add_argument("--out-dir", default=None, help="训练输出目录，默认按 backbone 自动推导")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=None, help="批大小")
    parser.add_argument("--img-size", type=int, default=None, help="输入图像尺寸")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=None, help="权重衰减")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--grad-accum", type=int, default=None, help="梯度累积步数")
    parser.add_argument("--freeze-ratio", type=float, default=None, help="冻结 backbone 比例")
    parser.add_argument("--test-every", type=int, default=None, help="每 N 轮测试一次")
    parser.add_argument(
        "--aug-level",
        choices=["none", "light", "medium"],
        default=None,
        help="数据增强等级",
    )
    parser.add_argument(
        "--read-backend",
        choices=["pil", "tv"],
        default=None,
        help="图像读取后端",
    )
    parser.add_argument("--amp", dest="use_amp", action="store_true", help="启用 AMP")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false", help="禁用 AMP")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", help="启用预训练权重")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="禁用预训练权重")
    parser.add_argument("--cpu", dest="use_cpu", action="store_true", help="强制使用 CPU")
    parser.add_argument("--gpu", dest="use_cpu", action="store_false", help="优先使用 GPU")
    parser.add_argument("--tqdm", dest="use_tqdm", action="store_true", help="启用 tqdm 进度条")
    parser.add_argument("--no-tqdm", dest="use_tqdm", action="store_false", help="禁用 tqdm 进度条")
    parser.set_defaults(use_amp=None, pretrained=None, use_cpu=None, use_tqdm=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.backbone:
        cfg.MODEL_NAME = normalize_backbone_name(args.backbone)
    else:
        cfg.MODEL_NAME = normalize_backbone_name(cfg.MODEL_NAME)

    if args.data_dir:
        cfg.DATA_DIR = args.data_dir

    if args.out_dir:
        cfg.OUT_DIR = args.out_dir
    elif not str(getattr(cfg, "OUT_DIR", "")).strip():
        cfg.OUT_DIR = default_out_dir_for_backbone(cfg.MODEL_NAME)

    optional_overrides = {
        "EPOCHS": args.epochs,
        "BATCH_SIZE": args.batch_size,
        "IMG_SIZE": args.img_size,
        "LR": args.lr,
        "WEIGHT_DECAY": args.weight_decay,
        "WORKERS": args.workers,
        "GRAD_ACCUM": args.grad_accum,
        "FREEZE_BACKBONE_RATIO": args.freeze_ratio,
        "TEST_EVERY": args.test_every,
        "AUG_LEVEL": args.aug_level,
        "READ_BACKEND": args.read_backend,
        "USE_AMP": args.use_amp,
        "PRETRAINED": args.pretrained,
        "USE_CPU": args.use_cpu,
        "USE_TQDM": args.use_tqdm,
    }
    for key, value in optional_overrides.items():
        if value is not None:
            setattr(cfg, key, value)

    train_ds, val_ds, test_ds = build_datasets(
        data_dir=cfg.DATA_DIR,
        img_size=cfg.IMG_SIZE,
        aug_level=cfg.AUG_LEVEL,
    )
    train(cfg, train_ds, val_ds, test_ds)


if __name__ == "__main__":
    main()
