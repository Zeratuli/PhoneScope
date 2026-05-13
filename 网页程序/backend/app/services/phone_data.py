import json
from typing import Optional

from app.database import get_db
from app.models.db_models import PhoneModel

PARAM_LABELS = {
    "manufacturer": "制造商",
    "brand": "品牌",
    "model": "型号",
    "released": "发布日期",
    "screen": "屏幕",
    "processor": "处理器",
    "ram": "运行内存",
    "storage": "存储",
    "rear_camera": "后置相机",
    "front_camera": "前置相机",
    "battery": "电池",
    "os": "操作系统",
    "dimensions": "尺寸",
    "weight": "重量",
    "colors": "配色",
}

_MODEL_REGISTRY = {
    "iPhone_13": {
        "brand_name": "Apple",
        "series_name": "iPhone 13",
        "display_name": "iPhone 13",
        "aliases": ["Apple_iPhone_13", "IPHONE_13"],
        "specs": {
            "manufacturer": "Apple",
            "brand": "iPhone",
            "model": "iPhone 13",
            "released": "2021年9月",
            "screen": '6.1" Super Retina XDR OLED, 2532×1170, 60Hz',
            "processor": "Apple A15 Bionic (5nm)",
            "ram": "4 GB",
            "storage": "128 / 256 / 512 GB",
            "rear_camera": "12MP 广角 + 12MP 超广角，传感器位移防抖",
            "front_camera": "12MP TrueDepth (Face ID)",
            "battery": "3227 mAh, 20W 有线 / 15W MagSafe",
            "os": "iOS 15 (可升级至 iOS 18)",
            "dimensions": "146.7 × 71.5 × 7.65 mm",
            "weight": "173 g",
            "colors": "星光色 / 午夜色 / 蓝色 / 粉色 / 红色 / 绿色",
        },
    },
    "REDMI_K80_Pro": {
        "brand_name": "Xiaomi",
        "series_name": "Redmi K80 Pro",
        "display_name": "Redmi K80 Pro",
        "aliases": ["Xiaomi_Redmi_K80_Pro", "Redmi_K80_Pro"],
        "specs": {
            "manufacturer": "Xiaomi (小米)",
            "brand": "Redmi",
            "model": "Redmi K80 Pro",
            "released": "2024年12月",
            "screen": '6.67" AMOLED 2K (3200×1440), LTPO 120Hz, 4500nit',
            "processor": "Qualcomm Snapdragon 8 Elite (3nm)",
            "ram": "12 / 16 GB LPDDR5X",
            "storage": "256 / 512 GB / 1TB UFS 4.0",
            "rear_camera": "50MP OIS 主摄 + 32MP 长焦 + 8MP 超广角",
            "front_camera": "20MP",
            "battery": "6000 mAh, 120W 有线快充 / 50W 无线",
            "os": "HyperOS 2.0 (Android 15)",
            "dimensions": "160.3 × 74.6 × 8.4 mm",
            "weight": "213 g",
            "colors": "暗夜 / 晴雪 / 幻紫 / 浅蓝",
        },
    },
    "HUAWEI_NOVA_10": {
        "brand_name": "Huawei",
        "series_name": "Nova 10",
        "display_name": "HUAWEI nova 10",
        "aliases": ["Nova_10", "Huawei_Nova_10", "HUAWEI_Nova_10"],
        "specs": {
            "manufacturer": "Huawei (华为)",
            "brand": "Nova",
            "model": "HUAWEI nova 10",
            "released": "2022年7月",
            "screen": '6.7" OLED, 2400×1080, 120Hz, P3广色域',
            "processor": "Qualcomm Snapdragon 778G 4G",
            "ram": "8 GB LPDDR5",
            "storage": "128 / 256 GB",
            "rear_camera": "50MP RYYB 主摄 + 8MP 超广角 + 2MP 微距",
            "front_camera": "60MP 超广角 + 8MP 人像",
            "battery": "4000 mAh, 66W 华为超级快充",
            "os": "HarmonyOS 2",
            "dimensions": "162.2 × 73.9 × 6.88 mm",
            "weight": "168 g",
            "colors": "曜金黑 / 10号色 / 普罗旺斯 / 绮境森林",
        },
    },
}

_SEED_DATA = {
    model_key: item["specs"]
    for model_key, item in _MODEL_REGISTRY.items()
}

_ALIAS_TO_KEY = {}
for model_key, item in _MODEL_REGISTRY.items():
    _ALIAS_TO_KEY[model_key.lower()] = model_key
    for alias in item.get("aliases", []):
        _ALIAS_TO_KEY[alias.lower()] = model_key


def normalize_model_key(name: str | None) -> Optional[str]:
    """Map legacy labels to a canonical model key."""
    if not name:
        return None
    return _ALIAS_TO_KEY.get(name.strip().lower(), name.strip())


def get_phone_identity(model_name: str | None) -> Optional[dict]:
    canonical_key = normalize_model_key(model_name)
    if not canonical_key:
        return None
    item = _MODEL_REGISTRY.get(canonical_key)
    if not item:
        return {
            "model_key": canonical_key,
            "brand_name": None,
            "series_name": None,
            "display_name": canonical_key.replace("_", " "),
        }
    return {
        "model_key": canonical_key,
        "brand_name": item.get("brand_name"),
        "series_name": item.get("series_name"),
        "display_name": item.get("display_name"),
    }


def seed_phone_models() -> None:
    """Insert seed data into phone_models if the table is empty."""
    try:
        with get_db() as db:
            if db.query(PhoneModel).count() > 0:
                return
            for key, specs in _SEED_DATA.items():
                row = PhoneModel(
                    model_key=key,
                    manufacturer=specs.get("manufacturer"),
                    brand=specs.get("brand"),
                    model_name=specs.get("model"),
                    specs_json=json.dumps(specs, ensure_ascii=False),
                )
                db.add(row)
    except Exception:
        pass


def get_phone_info(model_name: str) -> Optional[dict]:
    """Query phone specs from database; fall back to seed dict."""
    canonical_key = normalize_model_key(model_name)
    if not canonical_key:
        return None
    try:
        with get_db() as db:
            row = db.query(PhoneModel).filter(
                PhoneModel.model_key == canonical_key,
                PhoneModel.is_active == True,  # noqa: E712
            ).first()
            if row and row.specs_json:
                return json.loads(row.specs_json)
    except Exception:
        pass
    return _SEED_DATA.get(canonical_key)
