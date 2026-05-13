from datetime import datetime, timezone
from fastapi import APIRouter

from app.config import APP_VERSION
from app.schemas.models import HealthResponse
from app.middleware.security import get_service_status

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    from app.main import classifier_mobilenet, classifier_swin, detector
    models_loaded = (
        detector.is_loaded()
        and classifier_mobilenet.is_loaded()
        and classifier_swin.is_loaded()
    )
    return HealthResponse(
        status="ok" if get_service_status() else "disabled",
        models_loaded=models_loaded,
        version=APP_VERSION,
        timestamp=datetime.now(timezone.utc),
    )
