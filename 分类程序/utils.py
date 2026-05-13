from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

BACKBONE_ALIASES = {
    "mobilenet": "mobilenetv3_large_100",
    "mobilenetv3": "mobilenetv3_large_100",
    "mobilenetv3_large_100": "mobilenetv3_large_100",
    "swin": "swin_tiny_patch4_window7_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "swin_tiny_patch4_window7_224": "swin_tiny_patch4_window7_224",
}

BACKBONE_RUN_DIRS = {
    "mobilenetv3_large_100": "runs/mobilenetv3",
    "swin_tiny_patch4_window7_224": "runs/swin",
}

BACKBONE_PRESETS = {
    "mobilenetv3_large_100": {
        "IMG_SIZE": 224,
        "AUG_LEVEL": "light",
        "BATCH_SIZE": 16,
        "EPOCHS": 40,
        "LR": 1e-4,
        "WEIGHT_DECAY": 0.03,
        "GRAD_ACCUM": 1,
        "FREEZE_BACKBONE_RATIO": 0.2,
        "TEST_EVERY": 5,
        "READ_BACKEND": "pil",
        "PRETRAINED": True,
        "USE_AMP": True,
    },
    "swin_tiny_patch4_window7_224": {
        "IMG_SIZE": 224,
        "AUG_LEVEL": "light",
        "BATCH_SIZE": 8,
        "EPOCHS": 30,
        "LR": 5e-5,
        "WEIGHT_DECAY": 0.05,
        "GRAD_ACCUM": 2,
        "FREEZE_BACKBONE_RATIO": 0.5,
        "TEST_EVERY": 5,
        "READ_BACKEND": "pil",
        "PRETRAINED": True,
        "USE_AMP": True,
    },
}


def supported_backbone_choices() -> list[str]:
    """User-facing short backbone names supported by the dual-head trainer."""
    return ["mobilenetv3", "swin"]


def normalize_backbone_name(name: str) -> str:
    """Normalize user/backbone aliases to a timm model name."""
    key = name.strip()
    normalized = BACKBONE_ALIASES.get(key)
    if normalized is None:
        raise ValueError(
            f"不支持的 backbone: {name}. "
            f"可选: {', '.join(supported_backbone_choices())}"
        )
    return normalized


def default_out_dir_for_backbone(model_name: str) -> str:
    """Return the default run directory for a supported backbone."""
    normalized = normalize_backbone_name(model_name)
    return BACKBONE_RUN_DIRS[normalized]


def get_backbone_preset(model_name: str) -> dict:
    """Return a copy of the recommended small-dataset preset for a backbone."""
    normalized = normalize_backbone_name(model_name)
    return dict(BACKBONE_PRESETS[normalized])


def resolve_experiment_dir(base_dir: str | Path) -> Path:
    """Create a YOLO-style numbered run directory under the model folder."""
    base = Path(base_dir)
    if re.fullmatch(r"run_\d{3}", base.name):
        base.mkdir(parents=True, exist_ok=True)
        return base

    base.mkdir(parents=True, exist_ok=True)
    indices = []
    for child in base.iterdir():
        if not child.is_dir():
            continue
        m = re.fullmatch(r"run_(\d{3})", child.name)
        if m:
            indices.append(int(m.group(1)))

    next_idx = max(indices, default=0) + 1
    exp_dir = base / f"run_{next_idx:03d}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def resolve_run_dir_for_inference(path: str | Path) -> Path:
    """Resolve an inference/eval run dir, preferring the latest numbered run."""
    base = Path(path)
    if (base / "brand_classes.txt").exists() and (base / "model_classes.txt").exists():
        return base
    candidates = []
    if base.exists():
        for child in base.iterdir():
            if not child.is_dir():
                continue
            m = re.fullmatch(r"run_(\d{3})", child.name)
            if m and (child / "brand_classes.txt").exists() and (child / "model_classes.txt").exists():
                candidates.append((int(m.group(1)), child))
    if not candidates:
        return base
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def build_canonical_model_key(brand_name: str, model_name: str) -> str:
    """Convert a brand/model directory pair to a canonical model key."""
    brand = brand_name.strip().replace(" ", "_")
    model = model_name.strip().replace(" ", "_")

    if brand.lower() == "huawei":
        return f"HUAWEI_{model.upper()}"
    if brand.lower() == "apple":
        return model
    if brand.lower() == "xiaomi":
        if model.upper().startswith(("REDMI_", "POCO_", "MI_")):
            return model
        return f"Xiaomi_{model}"
    return f"{brand}_{model}"


def build_label_maps(data_dir: str) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """
    扫描两层目录 data_dir/{brand}/{model}/ 构建标签映射。

    返回:
        brand_classes  : 排序后的厂商列表, e.g. ["Apple", "Huawei", "Xiaomi"]
        model_classes  : 排序后的型号列表, e.g. ["iPhone_13", "Nova_10", "Redmi_K80_Pro"]
        brand_to_models: 厂商 -> 型号列表, e.g. {"Apple": ["iPhone_13"], ...}
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在: {root.resolve()}")

    brand_to_models: Dict[str, List[str]] = {}

    for brand_dir in sorted(root.iterdir()):
        if not brand_dir.is_dir():
            continue
        brand_name = brand_dir.name
        models = []
        for model_dir in sorted(brand_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            models.append(build_canonical_model_key(brand_name, model_dir.name))
        if models:
            brand_to_models[brand_name] = models

    brand_classes = sorted(brand_to_models.keys())
    model_classes = sorted(
        m for models in brand_to_models.values() for m in models
    )

    return brand_classes, model_classes, brand_to_models


def check_consistency(
    brand_pred: str,
    model_pred: str,
    brand_to_models: Dict[str, List[str]],
) -> bool:
    """检查预测的型号是否属于预测的厂商。"""
    expected_models = brand_to_models.get(brand_pred, [])
    return model_pred in expected_models


def save_label_maps(
    out_dir: str,
    brand_classes: List[str],
    model_classes: List[str],
    brand_to_models: Dict[str, List[str]],
) -> None:
    """将类别映射保存到训练输出目录。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "brand_classes.txt").write_text(
        "\n".join(brand_classes), encoding="utf-8"
    )
    (out / "model_classes.txt").write_text(
        "\n".join(model_classes), encoding="utf-8"
    )
    (out / "brand_to_models.json").write_text(
        json.dumps(brand_to_models, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_label_maps(
    run_dir: str,
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """从训练输出目录加载类别映射。"""
    d = Path(run_dir)

    brand_path = d / "brand_classes.txt"
    model_path = d / "model_classes.txt"
    mapping_path = d / "brand_to_models.json"

    for p in (brand_path, model_path, mapping_path):
        if not p.exists():
            raise FileNotFoundError(f"找不到标签文件: {p.resolve()}")

    brand_classes = [
        line.strip()
        for line in brand_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    model_classes = [
        line.strip()
        for line in model_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    brand_to_models = json.loads(mapping_path.read_text(encoding="utf-8"))

    return brand_classes, model_classes, brand_to_models
