import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from contextlib import asynccontextmanager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from app.config import APP_NAME, APP_VERSION, CORS_ORIGINS, USE_MOCK_SERVICES, DEBUG
from app.config import (
    YOLO_MODEL_PATH,
    SWIN_MODEL_PATH, SWIN_CLASSES_PATH, SWIN_MODEL_NAME,
    MOBILENET_MODEL_PATH, MOBILENET_CLASSES_PATH, MOBILENET_MODEL_NAME,
    DUALHEAD_ENABLED,
    DUALHEAD_MOBILENET_MODEL_PATH,
    DUALHEAD_MOBILENET_BRAND_CLASSES_PATH,
    DUALHEAD_MOBILENET_MODEL_CLASSES_PATH,
    DUALHEAD_MOBILENET_BRAND_TO_MODELS_PATH,
    DUALHEAD_MOBILENET_BACKBONE_NAME,
    DUALHEAD_SWIN_MODEL_PATH,
    DUALHEAD_SWIN_BRAND_CLASSES_PATH,
    DUALHEAD_SWIN_MODEL_CLASSES_PATH,
    DUALHEAD_SWIN_BRAND_TO_MODELS_PATH,
    DUALHEAD_SWIN_BACKBONE_NAME,
)
from app.middleware.security import (
    limiter,
    ServiceToggleMiddleware,
    rate_limit_exceeded_handler,
)
from app.routers import health, detect, export, admin
from app.routers import fusion as fusion_router
from app.routers import logs as logs_router
from app.services.detector import DetectorService
from app.services.classifier import ClassifierService
from app.services.pipeline import PipelineService
from app.services.exporter import ExportService
from app.services.fusion import FusionService
from app.database import init_db


@asynccontextmanager
async def lifespan(application: FastAPI):
    try:
        init_db()
    except Exception:
        pass
    yield


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    docs_url="/docs" if DEBUG else None,
    redoc_url="/redoc" if DEBUG else None,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

app.add_middleware(ServiceToggleMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if USE_MOCK_SERVICES:
    detector = DetectorService()
    classifier_swin = ClassifierService()
    classifier_mobilenet = ClassifierService()
else:
    detector = DetectorService(model_path=YOLO_MODEL_PATH)
    if DUALHEAD_ENABLED:
        classifier_swin = ClassifierService(
            weights_path=DUALHEAD_SWIN_MODEL_PATH,
            dualhead=True,
            brand_classes_path=DUALHEAD_SWIN_BRAND_CLASSES_PATH,
            model_classes_path=DUALHEAD_SWIN_MODEL_CLASSES_PATH,
            brand_to_models_path=DUALHEAD_SWIN_BRAND_TO_MODELS_PATH,
            backbone_name=DUALHEAD_SWIN_BACKBONE_NAME,
        )
        classifier_mobilenet = ClassifierService(
            weights_path=DUALHEAD_MOBILENET_MODEL_PATH,
            dualhead=True,
            brand_classes_path=DUALHEAD_MOBILENET_BRAND_CLASSES_PATH,
            model_classes_path=DUALHEAD_MOBILENET_MODEL_CLASSES_PATH,
            brand_to_models_path=DUALHEAD_MOBILENET_BRAND_TO_MODELS_PATH,
            backbone_name=DUALHEAD_MOBILENET_BACKBONE_NAME,
        )
    else:
        classifier_swin = ClassifierService(
            weights_path=SWIN_MODEL_PATH,
            classes_path=SWIN_CLASSES_PATH,
            model_name=SWIN_MODEL_NAME,
        )
        classifier_mobilenet = ClassifierService(
            weights_path=MOBILENET_MODEL_PATH,
            classes_path=MOBILENET_CLASSES_PATH,
            model_name=MOBILENET_MODEL_NAME,
        )

pipeline = PipelineService(detector, classifier_swin)
fusion_service = FusionService(detector, classifier_swin)
exporter = ExportService()

app.include_router(health.router, prefix="/api/v1")
app.include_router(detect.router, prefix="/api/v1")
app.include_router(fusion_router.router, prefix="/api/v1")
app.include_router(export.router, prefix="/api/v1")
app.include_router(admin.router, prefix="/api/v1")
app.include_router(logs_router.router, prefix="/api/v1")
