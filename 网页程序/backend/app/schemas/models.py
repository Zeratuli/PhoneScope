from pydantic import BaseModel
from datetime import datetime


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str
    timestamp: datetime


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionItem(BaseModel):
    bbox: BoundingBox
    confidence: float
    label: str
    crop_base64: str | None = None


class TopKItem(BaseModel):
    name: str
    confidence: float


class PhoneSpec(BaseModel):
    manufacturer: str = ""
    brand: str = ""
    model: str = ""
    released: str = ""
    screen: str = ""
    processor: str = ""
    ram: str = ""
    storage: str = ""
    rear_camera: str = ""
    front_camera: str = ""
    battery: str = ""
    os: str = ""
    dimensions: str = ""
    weight: str = ""
    colors: str = ""


class ClassificationItem(BaseModel):
    model_name: str
    brand_name: str | None = None
    series_name: str | None = None
    display_name: str | None = None
    confidence: float
    top_k: list[TopKItem]
    phone_spec: PhoneSpec | None = None


class ImageResult(BaseModel):
    image_id: str
    filename: str
    width: int
    height: int
    detections: list[DetectionItem]
    classifications: list[ClassificationItem]
    annotated_image_base64: str
    processing_time_ms: float


class AnalysisResponse(BaseModel):
    success: bool
    result: ImageResult


class BatchTaskResponse(BaseModel):
    success: bool
    task_id: str
    total_files: int
    message: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: float
    current_file: str | None = None
    results: list[ImageResult] | None = None
    created_at: datetime
    error: str | None = None


class ExportRequest(BaseModel):
    task_id: str
    include_images: bool = True


class AdminRequest(BaseModel):
    secret_key: str
    action: str


class AdminResponse(BaseModel):
    success: bool
    service_enabled: bool
    message: str


class FrameEvidence(BaseModel):
    frame_index: int
    filename: str
    width: int
    height: int
    detections: list[DetectionItem]
    best_detection_index: int | None = None
    quality_score: float = 0.0
    is_valid: bool = False
    is_best: bool = False
    annotated_image_base64: str = ""


class FusionResult(BaseModel):
    session_id: str
    mode: str
    total_frames: int
    valid_frames: int
    best_frame_index: int | None = None
    frames: list[FrameEvidence]
    final_model_name: str | None = None
    final_brand_name: str | None = None
    final_series_name: str | None = None
    final_display_name: str | None = None
    final_confidence: float | None = None
    final_top_k: list[TopKItem] | None = None
    final_phone_spec: PhoneSpec | None = None
    best_crop_base64: str | None = None
    processing_time_ms: float = 0.0


class FusionResponse(BaseModel):
    success: bool
    result: FusionResult
