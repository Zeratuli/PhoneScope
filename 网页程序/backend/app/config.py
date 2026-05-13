from pathlib import Path
from dotenv import load_dotenv
import os
import re

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

APP_NAME = os.getenv("APP_NAME", "PhoneScope")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

UPLOAD_DIR = BASE_DIR / os.getenv("UPLOAD_DIR", "uploads")
RESULTS_DIR = BASE_DIR / os.getenv("RESULTS_DIR", "results")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_BATCH_SIZE_MB = int(os.getenv("MAX_BATCH_SIZE_MB", "200"))

CORS_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
]
RATE_LIMIT = os.getenv("RATE_LIMIT", "30/minute")
UPLOAD_RATE_LIMIT = os.getenv("UPLOAD_RATE_LIMIT", "10/minute")

ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY", "changeme")
ADMIN_PAGE_PASSWORD = os.getenv("ADMIN_PAGE_PASSWORD", "996")

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "")
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.7"))

SWIN_MODEL_PATH = os.getenv("SWIN_MODEL_PATH", "")
SWIN_CLASSES_PATH = os.getenv("SWIN_CLASSES_PATH", "")
SWIN_MODEL_NAME = os.getenv("SWIN_MODEL_NAME", "swin_tiny_patch4_window7_224")

MOBILENET_MODEL_PATH = os.getenv("MOBILENET_MODEL_PATH", "")
MOBILENET_CLASSES_PATH = os.getenv("MOBILENET_CLASSES_PATH", "")
MOBILENET_MODEL_NAME = os.getenv("MOBILENET_MODEL_NAME", "mobilenetv3_large_100")

DUALHEAD_ENABLED = os.getenv("DUALHEAD_ENABLED", "false").lower() == "true"
DUALHEAD_MOBILENET_RUN_DIR = os.getenv("DUALHEAD_MOBILENET_RUN_DIR", "")
DUALHEAD_MOBILENET_MODEL_PATH = os.getenv("DUALHEAD_MOBILENET_MODEL_PATH", "")
DUALHEAD_MOBILENET_BRAND_CLASSES_PATH = os.getenv("DUALHEAD_MOBILENET_BRAND_CLASSES_PATH", "")
DUALHEAD_MOBILENET_MODEL_CLASSES_PATH = os.getenv("DUALHEAD_MOBILENET_MODEL_CLASSES_PATH", "")
DUALHEAD_MOBILENET_BRAND_TO_MODELS_PATH = os.getenv("DUALHEAD_MOBILENET_BRAND_TO_MODELS_PATH", "")
DUALHEAD_MOBILENET_BACKBONE_NAME = os.getenv("DUALHEAD_MOBILENET_BACKBONE_NAME", "mobilenetv3_large_100")

DUALHEAD_SWIN_RUN_DIR = os.getenv("DUALHEAD_SWIN_RUN_DIR", "")
DUALHEAD_SWIN_MODEL_PATH = os.getenv("DUALHEAD_SWIN_MODEL_PATH", "")
DUALHEAD_SWIN_BRAND_CLASSES_PATH = os.getenv("DUALHEAD_SWIN_BRAND_CLASSES_PATH", "")
DUALHEAD_SWIN_MODEL_CLASSES_PATH = os.getenv("DUALHEAD_SWIN_MODEL_CLASSES_PATH", "")
DUALHEAD_SWIN_BRAND_TO_MODELS_PATH = os.getenv("DUALHEAD_SWIN_BRAND_TO_MODELS_PATH", "")
DUALHEAD_SWIN_BACKBONE_NAME = os.getenv("DUALHEAD_SWIN_BACKBONE_NAME", "swin_tiny_patch4_window7_224")

USE_MOCK_SERVICES = os.getenv("USE_MOCK_SERVICES", "true").lower() == "true"

MYSQL_URL = os.getenv("MYSQL_URL", "mysql+pymysql://phonescope:phonescope@localhost:3306/phonescope")
FUSION_EDGE_SHRINK_RATIO = float(os.getenv("FUSION_EDGE_SHRINK_RATIO", "0.08"))
FUSION_MIN_CONFIDENCE = float(os.getenv("FUSION_MIN_CONFIDENCE", "0.5"))
FUSION_MIN_AREA_RATIO = float(os.getenv("FUSION_MIN_AREA_RATIO", "0.01"))
DEFAULT_FRAME_INTERVAL = int(os.getenv("DEFAULT_FRAME_INTERVAL", "30"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/webm"}
ALLOWED_TYPES = ALLOWED_IMAGE_TYPES | ALLOWED_VIDEO_TYPES


def _resolve_latest_run_dir(run_root: str) -> str:
    if not run_root:
        return ""
    root = Path(run_root)
    if not root.exists() or not root.is_dir():
        return ""
    candidates: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        m = re.fullmatch(r"run_(\d{3})", child.name)
        if not m:
            continue
        required = [
            child / "best.pt",
            child / "brand_classes.txt",
            child / "model_classes.txt",
            child / "brand_to_models.json",
        ]
        if all(p.exists() for p in required):
            candidates.append((int(m.group(1)), child))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: item[0])
    return str(candidates[-1][1])


def _within_run(run_dir: str, filename: str) -> str:
    if not run_dir:
        return ""
    path = Path(run_dir) / filename
    return str(path) if path.exists() else ""


DUALHEAD_MOBILENET_LATEST_RUN_DIR = _resolve_latest_run_dir(DUALHEAD_MOBILENET_RUN_DIR)
DUALHEAD_SWIN_LATEST_RUN_DIR = _resolve_latest_run_dir(DUALHEAD_SWIN_RUN_DIR)

if DUALHEAD_MOBILENET_LATEST_RUN_DIR:
    DUALHEAD_MOBILENET_MODEL_PATH = _within_run(DUALHEAD_MOBILENET_LATEST_RUN_DIR, "best.pt")
    DUALHEAD_MOBILENET_BRAND_CLASSES_PATH = _within_run(DUALHEAD_MOBILENET_LATEST_RUN_DIR, "brand_classes.txt")
    DUALHEAD_MOBILENET_MODEL_CLASSES_PATH = _within_run(DUALHEAD_MOBILENET_LATEST_RUN_DIR, "model_classes.txt")
    DUALHEAD_MOBILENET_BRAND_TO_MODELS_PATH = _within_run(DUALHEAD_MOBILENET_LATEST_RUN_DIR, "brand_to_models.json")

if DUALHEAD_SWIN_LATEST_RUN_DIR:
    DUALHEAD_SWIN_MODEL_PATH = _within_run(DUALHEAD_SWIN_LATEST_RUN_DIR, "best.pt")
    DUALHEAD_SWIN_BRAND_CLASSES_PATH = _within_run(DUALHEAD_SWIN_LATEST_RUN_DIR, "brand_classes.txt")
    DUALHEAD_SWIN_MODEL_CLASSES_PATH = _within_run(DUALHEAD_SWIN_LATEST_RUN_DIR, "model_classes.txt")
    DUALHEAD_SWIN_BRAND_TO_MODELS_PATH = _within_run(DUALHEAD_SWIN_LATEST_RUN_DIR, "brand_to_models.json")
